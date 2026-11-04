"""Grade every answer file with base Qwen2.5-7B-Instruct as judge, plus a string-match check.

Usage: python -m cl_books.judge
"""
import re

from . import config
from .common import make_llm, read_jsonl, write_jsonl

JUDGE_PROMPT = """Question: {question}
Reference answer: {reference}
Model answer: {prediction}

Does the model answer state the same fact as the reference? Different wording or extra detail is fine; a different, missing or hedged fact is not. Reply with only "yes" or "no"."""


def normalize(text: str) -> str:
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def contains_match(reference: str, prediction: str) -> bool:
    ref = normalize(reference)
    return bool(ref) and ref in normalize(prediction)


def parse_verdict(text: str) -> bool:
    return text.strip().lower().startswith("yes")


def main() -> None:
    from vllm import SamplingParams

    files = sorted((config.RESULTS_DIR / "answers").glob("*.jsonl"))
    rows = [row for f in files for row in read_jsonl(f)]
    llm = make_llm(config.BASE_MODEL, config.BASE_REVISION)
    params = SamplingParams(temperature=0.0, max_tokens=3)
    conversations = [[{"role": "user", "content": JUDGE_PROMPT.format(**r)}] for r in rows]
    verdicts = [parse_verdict(o.outputs[0].text) for o in llm.chat(conversations, params)]
    judged = [r | {"judge": v, "contains": contains_match(r["reference"], r["prediction"])} for r, v in zip(rows, verdicts)]
    for f in files:
        name = f.stem
        write_jsonl(config.RESULTS_DIR / "judged" / f"{name}.jsonl", [r for r in judged if r["model"] == name])
    print(f"judged {len(judged)} answers from {len(files)} models")


if __name__ == "__main__":
    main()
