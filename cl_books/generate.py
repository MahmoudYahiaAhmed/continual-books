"""Greedy answers of one model on the eval questions of every book.

Usage: python -m cl_books.generate --name m2_distill --model models/m1/merged --lora models/m2_distill/adapter
"""
import argparse

from . import config
from .common import make_llm, prompt_ids, read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--lora")
    args = p.parse_args()

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    revision = config.BASE_REVISION if args.model == config.BASE_MODEL else None
    llm = make_llm(args.model, revision, enable_lora=bool(args.lora))
    tokenizer = llm.get_tokenizer()
    lora = LoRARequest(args.name, 1, args.lora) if args.lora else None
    params = SamplingParams(temperature=0.0, max_tokens=config.MAX_ANSWER_TOKENS)

    rows = []
    for book in config.BOOKS:
        evals = read_jsonl(config.DATA_DIR / book / "eval.jsonl")
        prompts = [{"prompt_token_ids": prompt_ids(tokenizer, r["question"])} for r in evals]
        for i, (row, out) in enumerate(zip(evals, llm.generate(prompts, params, lora_request=lora))):
            rows.append({"model": args.name, "book": book, "idx": i, "question": row["question"],
                         "reference": row["answer"], "prediction": out.outputs[0].text.strip()})
    write_jsonl(config.RESULTS_DIR / "answers" / f"{args.name}.jsonl", rows)
    print(f"{args.name}: {len(rows)} answers")


if __name__ == "__main__":
    main()
