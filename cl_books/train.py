"""LoRA training for one book, with optional protection of the previous book.

Methods:
  naive    - train on the new book only
  replay   - mix old-book QA (reference answers) into the new-book data
  distill  - add a top-k KL term to the previous model's own answers on old-book questions

Usage:
  python -m cl_books.train --base Qwen/Qwen2.5-7B-Instruct --book A --out models/m1 --merge
  python -m cl_books.train --base models/m1/merged --book B --method distill \
      --rehearsal data/A/rehearsal.jsonl --out models/m2_distill
"""
import argparse
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from . import config
from .common import answer_ids, prompt_ids, read_jsonl
from .losses import topk_distill_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--book", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--method", choices=["naive", "replay", "distill"], default="naive")
    p.add_argument("--old-book", default="A")
    p.add_argument("--replay-ratio", type=float, default=0.25)
    p.add_argument("--rehearsal", type=Path)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--kd-batch", type=int, default=4)
    p.add_argument("--merge", action="store_true", help="also save a merged full model (needed as next step's base)")
    args = p.parse_args()
    if args.method == "distill" and not args.rehearsal:
        p.error("--method distill needs --rehearsal")
    return args


def ce_example(tokenizer, question: str, answer: str) -> dict:
    prompt = prompt_ids(tokenizer, question)
    target = answer_ids(tokenizer, answer)
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def build_ce_data(tokenizer, args) -> list[dict]:
    rows = read_jsonl(config.DATA_DIR / args.book / "train.jsonl")
    data = [ce_example(tokenizer, r["question"], r["answer"]) for r in rows]
    if args.method == "replay":
        old = read_jsonl(config.DATA_DIR / args.old_book / "train.jsonl")
        n_old = min(len(old), int(len(data) * args.replay_ratio))
        old = random.Random(config.SEED).sample(old, n_old)
        data += [ce_example(tokenizer, r["question"], r["answer"]) for r in old]
    return data


def pad_ce(batch: list[dict], pad_id: int, device) -> dict:
    width = max(len(ex["input_ids"]) for ex in batch)
    ids = [ex["input_ids"] + [pad_id] * (width - len(ex["input_ids"])) for ex in batch]
    labels = [ex["labels"] + [-100] * (width - len(ex["labels"])) for ex in batch]
    mask = [[1] * len(ex["input_ids"]) + [0] * (width - len(ex["input_ids"])) for ex in batch]
    as_t = lambda x: torch.tensor(x, device=device)  # noqa: E731
    return {"input_ids": as_t(ids), "labels": as_t(labels), "attention_mask": as_t(mask)}


def kd_step_loss(model, batch: list[dict], pad_id: int, device) -> torch.Tensor:
    seqs = [ex["prompt_ids"] + ex["answer_ids"] for ex in batch]
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs], device=device)
    attn = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
    logits = model(input_ids=ids, attention_mask=attn).logits
    rows, tk_ids, tk_lps = [], [], []
    for b, ex in enumerate(batch):
        start = len(ex["prompt_ids"]) - 1  # logits at position t predict token t + 1
        positions = torch.arange(start, start + len(ex["answer_ids"]), device=device)
        rows.append(logits[b, positions])
        tk_ids.append(torch.tensor(ex["topk_ids"], device=device))
        tk_lps.append(torch.tensor(ex["topk_logprobs"], device=device))
    student = torch.cat(rows)
    mask = torch.ones(student.shape[0], device=device)
    return topk_distill_loss(student, torch.cat(tk_ids), torch.cat(tk_lps), mask)


def load_model(base: str, device):
    revision = config.BASE_REVISION if base == config.BASE_MODEL else None
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, lora_dropout=0.05,
        target_modules=config.LORA_TARGETS, task_type="CAUSAL_LM",
    )
    return tokenizer, get_peft_model(model, lora)


def main() -> None:
    args = parse_args()
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda")
    tokenizer, model = load_model(args.base, device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    ce_data = build_ce_data(tokenizer, args)
    kd_data = [r for r in read_jsonl(args.rehearsal) if r["answer_ids"]] if args.method == "distill" else []
    kd_cycle = itertools.cycle(kd_data) if kd_data else None
    steps_per_epoch = math.ceil(len(ce_data) / args.batch)
    total = steps_per_epoch * args.epochs

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total), total)

    wandb_run = None
    if os.environ.get("WANDB_PROJECT"):
        import wandb
        wandb_run = wandb.init(name=args.out.name, config=vars(args) | {"n_ce": len(ce_data), "n_kd": len(kd_data)})

    model.train()
    started = time.time()
    rng = random.Random(config.SEED)
    for epoch in range(args.epochs):
        order = list(range(len(ce_data)))
        rng.shuffle(order)
        for i in range(steps_per_epoch):
            batch = [ce_data[j] for j in order[i * args.batch:(i + 1) * args.batch]]
            loss_ce = model(**pad_ce(batch, pad_id, device)).loss
            loss = loss_ce
            loss_kd = torch.zeros((), device=device)
            if kd_cycle:
                loss_kd = kd_step_loss(model, [next(kd_cycle) for _ in range(args.kd_batch)], pad_id, device)
                loss = loss + args.kd_weight * loss_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step = epoch * steps_per_epoch + i
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{total} ce {loss_ce.item():.4f} kd {loss_kd.item():.4f}")
                if wandb_run:
                    wandb_run.log({"loss/ce": loss_ce.item(), "loss/kd": loss_kd.item(), "lr": sched.get_last_lr()[0]}, step=step)

    hours = (time.time() - started) / 3600
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out / "adapter")
    tokenizer.save_pretrained(args.out / "adapter")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out / "merged", safe_serialization=True)
        tokenizer.save_pretrained(args.out / "merged")
    meta = {"method": args.method, "book": args.book, "base": args.base, "train_gpu_hours": round(hours, 3),
            "n_ce": len(ce_data), "n_kd": len(kd_data), "args": {k: str(v) for k, v in vars(args).items()}}
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if wandb_run:
        wandb_run.summary["train_gpu_hours"] = hours
        wandb_run.finish()
    print(f"done in {hours:.2f} GPU hours -> {args.out}")


if __name__ == "__main__":
    main()
