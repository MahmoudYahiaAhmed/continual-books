"""Download a Gutenberg book, strip the licence boilerplate and split it into passages."""
import re
from pathlib import Path

import requests

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG E-?BOOK[^*]*\*\*\*", re.I)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK", re.I)


def fetch_book(gid: int, cache_dir: Path) -> str:
    cache = cache_dir / f"pg{gid}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    resp = requests.get(GUTENBERG_URL.format(gid=gid), timeout=60)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig", errors="replace")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def strip_boilerplate(text: str) -> str:
    start = _START.search(text)
    end = _END.search(text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Gutenberg START/END markers not found")
    return text[start.end():end.start()].strip()


def chunk_paragraphs(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing; a paragraph longer than max_chars becomes its own chunk."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        if current and size + len(para) > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c) > 200]
