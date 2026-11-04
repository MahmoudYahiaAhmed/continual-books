"""Turn book passages into QA pairs with Qwen2.5-7B-Instruct itself as the teacher.

Usage: python -m cl_books.qa
Writes data/<book>/train.jsonl (original questions) and data/<book>/eval.jsonl
(reworded questions for a held-out sample of facts).
"""
import json
import random

from . import config
from .books import chunk_paragraphs, fetch_book, strip_boilerplate
from .common import make_llm, write_jsonl

QA_PROMPT = """Passage from a novel:
\"\"\"{passage}\"\"\"

Write {n} question-answer pairs about specific facts stated in this passage (names, places, objects, numbers, events).
Rules:
- Each question must make sense on its own without the passage: name the people involved. Never mention the book title or "the passage".
- The answer must be short (1 to 8 words) and stated in the passage.
- Also write "paraphrase": the same question reworded with different words and the same answer.
Return only a JSON list: [{{"question": "...", "answer": "...", "paraphrase": "..."}}]"""

_KEYS = ("question", "answer", "paraphrase")


def parse_qa_json(raw: str) -> list[dict]:
    """Extract valid QA dicts from a model reply; malformed replies yield an empty list."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    return [
        {k: item[k].strip() for k in _KEYS}
        for item in items
        if isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k].strip() for k in _KEYS)
    ]


def dedupe(facts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for fact in facts:
        key = fact["question"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(fact)
    return unique


def split_facts(facts: list[dict], n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Every fact is trained with its original question; a sample is tested with the reworded one."""
    order = list(range(len(facts)))
    random.Random(seed).shuffle(order)
    train = [{"question": f["question"], "answer": f["answer"]} for f in facts]
    evals = [{"question": facts[i]["paraphrase"], "answer": facts[i]["answer"]} for i in order[:n_eval]]
    return train, evals


def main() -> None:
    from vllm import SamplingParams

    llm = make_llm(config.BASE_MODEL, config.BASE_REVISION)
    params = SamplingParams(temperature=0.3, max_tokens=900, seed=config.SEED)
    for name, book in config.BOOKS.items():
        raw = fetch_book(book["gutenberg_id"], config.DATA_DIR / "raw")
        passages = chunk_paragraphs(strip_boilerplate(raw), config.CHUNK_CHARS)
        conversations = [
            [{"role": "user", "content": QA_PROMPT.format(passage=p, n=config.QA_PER_CHUNK)}] for p in passages
        ]
        outputs = llm.chat(conversations, params)
        facts = dedupe([qa for out in outputs for qa in parse_qa_json(out.outputs[0].text)])
        train, evals = split_facts(facts, config.EVAL_PER_BOOK, config.SEED)
        write_jsonl(config.DATA_DIR / name / "train.jsonl", train)
        write_jsonl(config.DATA_DIR / name / "eval.jsonl", evals)
        print(f"book {name}: {len(passages)} passages, {len(train)} train facts, {len(evals)} eval questions")


if __name__ == "__main__":
    main()
