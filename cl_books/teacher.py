"""Self-distillation targets: the step-1 model (M1) answers book-A questions and we keep its top-k logprobs.

Usage: python -m cl_books.teacher --model models/m1/merged --out data/A/rehearsal.jsonl
"""
import argparse
import random
from pathlib import Path

from . import config
from .common import make_llm, prompt_ids, read_jsonl, write_jsonl


def topk_rows(token_ids: list[int], logprobs: list[dict], k: int) -> tuple[list[list[int]], list[list[float]]]:
    """vLLM returns, per position, a dict token_id -> Logprob holding the top-k plus the sampled token."""
    ids, lps = [], []
    for position in logprobs[: len(token_ids)]:
        best = sorted(position.items(), key=lambda kv: kv[1].logprob, reverse=True)[:k]
        ids.append([tid for tid, _ in best])
        lps.append([lp.logprob for _, lp in best])
    return ids, lps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--book", default="A")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from vllm import SamplingParams

    questions = [row["question"] for row in read_jsonl(config.DATA_DIR / args.book / "train.jsonl")]
    questions = random.Random(config.SEED).sample(questions, min(config.REHEARSAL_SIZE, len(questions)))

    llm = make_llm(args.model)
    tokenizer = llm.get_tokenizer()
    prompts = [{"prompt_token_ids": prompt_ids(tokenizer, q)} for q in questions]
    params = SamplingParams(temperature=0.0, max_tokens=config.MAX_ANSWER_TOKENS, logprobs=config.TOPK)
    rows = []
    for prompt, out in zip(prompts, llm.generate(prompts, params)):
        gen = out.outputs[0]
        ids, lps = topk_rows(list(gen.token_ids), gen.logprobs, config.TOPK)
        rows.append({
            "prompt_ids": prompt["prompt_token_ids"],
            "answer_ids": list(gen.token_ids),
            "topk_ids": ids,
            "topk_logprobs": lps,
        })
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rehearsal rows to {args.out}")


if __name__ == "__main__":
    main()
