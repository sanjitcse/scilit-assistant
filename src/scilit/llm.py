"""LLM backends.

OfflineLLM    deterministic, heuristic, no network. It exists so the pipeline, tests and eval
              harness run anywhere. Its numbers validate the *harness*, not the method.
AnthropicLLM  real model calls (temperature 0, on-disk cache, JSONL call trace).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .textutil import entity_terms, sentences, terms, tokens, STOPWORDS
from .types import Chunk

FIELDS = ["task", "dataset", "metric", "value", "baseline", "baseline_value", "limitation"]


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _field(value: str, chunk: Chunk, quote: str) -> dict:
    return {"value": value.strip(), "chunk_id": chunk.chunk_id, "quote": quote.strip()}


class OfflineLLM:
    name = "offline-heuristic"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def rewrite(self, question: str) -> str:
        self.usage["calls"] += 1
        return " ".join(t for t in tokens(question) if t not in STOPWORDS)

    def closed_book(self, question: str) -> str:
        self.usage["calls"] += 1
        return ""  # an offline stub has no parametric knowledge

    def answer(self, question: str, chunks: list[Chunk]) -> dict:
        self.usage["calls"] += 1
        q = set(terms(question))
        if not chunks or not q:
            return {"abstain": True, "claims": []}
        have: set[str] = set()
        for c in chunks:
            have |= set(terms(c.title + " " + c.text))
        if any(e not in have for e in entity_terms(question)):
            return {"abstain": True, "claims": []}  # question names something not in the evidence
        best = None
        for c in chunks:  # chunks arrive best-first; strict '>' keeps the earliest on ties
            title_terms = set(terms(c.title))  # the paper title is context for every sentence in it
            for s in sentences(c.text):
                own = set(terms(s))
                cov = (len(q & (own | title_terms)) / len(q), len(q & own) / len(q))  # title-blended, then sentence-only
                if best is None or cov > best[0]:
                    best = (cov, s, c)
        if best is None or best[0][0] < self.threshold:
            return {"abstain": True, "claims": []}
        return {"abstain": False, "claims": [{"text": best[1], "chunk_ids": [best[2].chunk_id]}]}

    def entails(self, claim: str, evidence: str) -> bool:
        self.usage["calls"] += 1
        ct = set(terms(claim))
        return bool(ct) and len(ct & set(terms(evidence))) / len(ct) >= 0.9

    def extract(self, paper_id: str, chunks: list[Chunk]) -> dict:
        self.usage["calls"] += 1
        out: dict = {}
        for c in chunks:
            if c.section == "abstract" and (m := re.search(r"We study (.+?)\.", c.text)):
                out["task"] = _field(m.group(1), c, m.group(0))
            elif c.section == "methods" and (m := re.search(r"on the (\S+) dataset", c.text)):
                out["dataset"] = _field(m.group(1), c, m.group(0))
            elif c.section == "results" and (
                m := re.search(r"achieves (.+?) of (.+?), compared with (.+?) for (.+?)\.", c.text)
            ):
                out["metric"] = _field(m.group(1), c, m.group(1))
                out["value"] = _field(m.group(2), c, m.group(2))
                out["baseline_value"] = _field(m.group(3), c, m.group(3))
                out["baseline"] = _field(m.group(4), c, m.group(4))
            elif c.section == "limitations" and (m := re.search(r"limitation of \S+ is that (.+?)\.", c.text)):
                out["limitation"] = _field(m.group(1), c, m.group(1))
        return out


ANSWER_SYSTEM = (
    "You answer questions about scientific papers using ONLY the provided chunks. "
    'Return a single JSON object: {"abstain": bool, "claims": [{"text": str, "chunk_ids": [str]}]}. '
    "Each claim must be directly supported by the chunks it cites; cite by chunk id exactly as given. "
    "If the chunks do not contain the answer, return {\"abstain\": true, \"claims\": []}. No prose outside JSON."
)
EXTRACT_SYSTEM = (
    "Extract structured facts from one paper's chunks. Return a single JSON object with any of these keys "
    f"{FIELDS}. Each value is {{\"value\": str, \"chunk_id\": str, \"quote\": str}} where quote is copied "
    "verbatim from the cited chunk and contains the value. Omit fields the text does not state. No prose."
)
ENTAIL_SYSTEM = (
    'Decide whether the EVIDENCE fully supports the CLAIM. Return {"entailed": true|false} only. '
    "Numbers, names and qualifiers in the claim must all appear in or follow from the evidence."
)
REWRITE_SYSTEM = (
    "Rewrite the question as a keyword-style search query for scientific literature. "
    'Return {"query": str} only.'
)
CLOSED_SYSTEM = "Answer the question about the scientific literature concisely from your own knowledge."


def _fmt_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] ({c.title}, {c.section})\n{c.text}" for c in chunks)


class AnthropicLLM:
    def __init__(self, model: str | None = None, cache: bool = True, cache_dir: str = ".cache/llm",
                 trace_path: str | None = None, max_tokens: int = 1024):
        import anthropic  # pip install 'scilit[anthropic]'

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("SCILIT_MODEL", "claude-sonnet-5-5")
        self.name = f"anthropic:{self.model}"
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = Path(trace_path) if trace_path else None
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def _call(self, system: str, user: str) -> str:
        key = hashlib.sha256(json.dumps([self.model, system, user]).encode()).hexdigest()
        if self.cache_dir and (p := self.cache_dir / f"{key}.txt").exists():
            return p.read_text(encoding="utf-8")
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        try:
            resp = self.client.messages.create(temperature=0, **kwargs)
        except self._anthropic.BadRequestError:  # some models reject sampling parameters
            resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        if self.cache_dir:
            (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        if self.trace_path:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": self.model, "system": system, "user": user, "output": text}) + "\n")
        return text

    def rewrite(self, question: str) -> str:
        try:
            return parse_json(self._call(REWRITE_SYSTEM, question)).get("query", question)
        except ValueError:
            return question

    def closed_book(self, question: str) -> str:
        return self._call(CLOSED_SYSTEM, question).strip()

    def answer(self, question: str, chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"abstain": True, "claims": []}
        try:
            return parse_json(self._call(ANSWER_SYSTEM, f"QUESTION: {question}\n\nCHUNKS:\n{_fmt_chunks(chunks)}"))
        except ValueError:
            return {"abstain": True, "claims": []}  # unparseable output is treated as no answer

    def entails(self, claim: str, evidence: str) -> bool:
        try:
            return bool(parse_json(self._call(ENTAIL_SYSTEM, f"CLAIM: {claim}\n\nEVIDENCE: {evidence}")).get("entailed"))
        except ValueError:
            return False

    def extract(self, paper_id: str, chunks: list[Chunk]) -> dict:
        try:
            return parse_json(self._call(EXTRACT_SYSTEM, f"PAPER {paper_id}\n\n{_fmt_chunks(chunks)}"))
        except ValueError:
            return {}


def make_llm(kind: str = "offline", **kw):
    if kind == "offline":
        return OfflineLLM()
    if kind == "anthropic":
        return AnthropicLLM(**kw)
    raise ValueError(f"unknown llm backend: {kind}")
