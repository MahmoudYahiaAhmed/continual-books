"""Build the results table from judged answers and write it into results/results.md and README.md.

Usage: python -m cl_books.report
"""
import json
import os
import random

from . import config
from .common import read_jsonl

ORDER = ["base", "m1", "m2_naive", "m2_replay", "m2_distill"]
LABELS = {
    "base": "Base Qwen2.5-7B-Instruct",
    "m1": "M1: + book A (LoRA)",
    "m2_naive": "M1 + book B, naive",
    "m2_replay": "M1 + book B, replay 25%",
    "m2_distill": "M1 + book B, self-distill to M1",
}
MARK_START, MARK_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def correct_vector(rows: list[dict], book: str) -> list[bool]:
    return [r["judge"] for r in sorted((r for r in rows if r["book"] == book), key=lambda r: r["idx"])]


def retention(before: list[bool], after: list[bool]) -> tuple[int, int]:
    """How many answers the earlier model got right are still right afterwards."""
    kept = sum(1 for b, a in zip(before, after) if b and a)
    return kept, sum(before)


def paired_gain_ci(before: list[bool], after: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean gain in correct answers (after - before) with a paired bootstrap 95% interval."""
    diffs = [int(a) - int(b) for b, a in zip(before, after)]
    rng = random.Random(seed)
    n = len(diffs)
    sums = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return float(sum(diffs)), float(sums[int(0.025 * n_boot)]), float(sums[int(0.975 * n_boot)])


def train_hours(name: str) -> str:
    meta = config.MODELS_DIR / name / "train_meta.json"
    return f"{json.loads(meta.read_text())['train_gpu_hours']:.2f}" if meta.exists() else "-"


COLUMNS = ["Model", "Book A correct", "Book B correct", "Book A kept (of M1-correct)",
           "Book B gain vs M1 [95% CI]", "Train GPU h"]


def summary_rows(results: dict[str, list[dict]]) -> list[list[str]]:
    m1_a = correct_vector(results["m1"], "A") if "m1" in results else None
    m1_b = correct_vector(results["m1"], "B") if "m1" in results else None
    rows = []
    for name in [n for n in ORDER if n in results]:
        a, b = correct_vector(results[name], "A"), correct_vector(results[name], "B")
        kept, gain = "-", "-"
        if name.startswith("m2") and m1_a is not None:
            k, tot = retention(m1_a, a)
            kept = f"{k}/{tot} ({100 * k / max(tot, 1):.0f}%)"
            g, lo, hi = paired_gain_ci(m1_b, b)
            gain = f"{g:+.0f} [{lo:+.0f}, {hi:+.0f}]"
        rows.append([LABELS.get(name, name), f"{sum(a)}/{len(a)}", f"{sum(b)}/{len(b)}", kept, gain, train_hours(name)])
    return rows


def to_markdown(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def write_readme(table: str) -> None:
    readme = config.ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(MARK_START), text.find(MARK_END)
    if start < 0 or end < start:
        return
    readme.write_text(text[: start + len(MARK_START)] + "\n" + table + "\n" + text[end:], encoding="utf-8")


def main() -> None:
    judged_dir = config.RESULTS_DIR / "judged"
    results = {f.stem: read_jsonl(f) for f in judged_dir.glob("*.jsonl")}
    rows = summary_rows(results)
    table = to_markdown(rows)
    (config.RESULTS_DIR / "results.md").write_text(table + "\n", encoding="utf-8")
    write_readme(table)
    print(table)
    if os.environ.get("WANDB_PROJECT"):
        import wandb

        run = wandb.init(name="results", job_type="report")
        run.log({"results": wandb.Table(columns=COLUMNS, data=rows)})
        run.finish()


if __name__ == "__main__":
    main()
