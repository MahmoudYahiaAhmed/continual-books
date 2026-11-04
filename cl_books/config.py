"""Central configuration. Every value can be overridden with an environment variable of the same name."""
import os
from pathlib import Path


def _env(name: str, default):
    raw = os.environ.get(name)
    return default if raw is None else type(default)(raw)


BASE_MODEL = _env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BASE_REVISION = _env("BASE_REVISION", "main")

ROOT = Path(_env("CB_ROOT", str(Path(__file__).resolve().parents[1])))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Two Dr. Thorndyke novels by R. Austin Freeman (public domain): same detective,
# same narrator, same London legal setting, so the topics overlap on purpose.
BOOKS = {
    "A": {"gutenberg_id": 11128, "title": "The Red Thumb Mark"},
    "B": {"gutenberg_id": 12187, "title": "The Mystery of 31 New Inn"},
}

SEED = _env("SEED", 1234)
CHUNK_CHARS = _env("CHUNK_CHARS", 1800)
QA_PER_CHUNK = _env("QA_PER_CHUNK", 4)
EVAL_PER_BOOK = _env("EVAL_PER_BOOK", 150)
REHEARSAL_SIZE = _env("REHEARSAL_SIZE", 800)
TOPK = _env("TOPK", 20)
MAX_ANSWER_TOKENS = _env("MAX_ANSWER_TOKENS", 48)
GPU_MEM_UTIL = _env("GPU_MEM_UTIL", 0.85)

LORA_RANK = _env("LORA_RANK", 32)
LORA_ALPHA = _env("LORA_ALPHA", 64)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
