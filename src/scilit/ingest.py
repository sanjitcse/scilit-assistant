"""Corpus loading and section-aware chunking.

Input format is JSONL, one paper per line:
  {"paper_id": "...", "title": "...", "doi": "...", "sections": {"abstract": "...", ...}}

Producing that JSONL from real PDFs (e.g. GROBID TEI -> sections) is intentionally outside this
module; see docs/limitations.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from .textutil import sentences
from .types import Chunk


def load_papers(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_paper(paper: dict, max_words: int = 180) -> list[Chunk]:
    chunks: list[Chunk] = []
    pid, title = paper["paper_id"], paper["title"]
    for section, text in paper["sections"].items():
        if len(text.split()) <= max_words:
            chunks.append(Chunk(f"{pid}:{section}", pid, title, section, text))
            continue
        buf: list[str] = []
        n = 0
        for sent in sentences(text):
            w = len(sent.split())
            if buf and n + w > max_words:
                chunks.append(Chunk(f"{pid}:{section}:{len(chunks)}", pid, title, section, " ".join(buf)))
                buf, n = [], 0
            buf.append(sent)
            n += w
        if buf:
            chunks.append(Chunk(f"{pid}:{section}:{len(chunks)}", pid, title, section, " ".join(buf)))
    return chunks


def load_chunks(path: str | Path, max_words: int = 180) -> list[Chunk]:
    return [c for p in load_papers(path) for c in chunk_paper(p, max_words)]
