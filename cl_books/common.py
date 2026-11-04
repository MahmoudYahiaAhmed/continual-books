"""Small helpers shared by every stage."""
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_ids(tokenizer, question: str) -> list[int]:
    """Token ids of a single-turn chat prompt, ending right where the assistant starts."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)


def answer_ids(tokenizer, answer: str) -> list[int]:
    """Assistant answer tokens followed by the chat end token, as the model must emit them."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return tokenizer.encode(answer, add_special_tokens=False) + [end_id]


def make_llm(model_path: str, revision: str | None = None, enable_lora: bool = False):
    from vllm import LLM

    from . import config

    return LLM(
        model=model_path,
        revision=revision,
        dtype="bfloat16",
        seed=config.SEED,
        gpu_memory_utilization=config.GPU_MEM_UTIL,
        max_model_len=4096,
        enable_lora=enable_lora,
        max_lora_rank=64,
    )
