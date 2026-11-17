"""Three narrow agents. Each is a plain class with typed inputs/outputs; no framework required."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .llm import FIELDS
from .retrieval import HybridIndex, LexicalReranker, rrf_merge
from .textutil import norm_ws, numbers
from .types import Chunk, Claim


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    queries: list[str]
    timings: dict[str, float] = field(default_factory=dict)


class RetrievalAgent:
    """Rewrites the query, retrieves with the configured mode, merges pools (RRF), reranks."""

    def __init__(self, index: HybridIndex, llm, k: int = 5, candidate_k: int = 20, mode: str = "hybrid",
                 rewrite: bool = True, rerank: bool = True, reranker=None):
        self.index, self.llm, self.k, self.candidate_k = index, llm, k, candidate_k
        self.mode, self.rewrite, self.rerank = mode, rewrite, rerank
        self.reranker = reranker or LexicalReranker(index)

    def run(self, question: str, k: int | None = None) -> RetrievalResult:
        k = k or self.k
        t0 = perf_counter()
        queries = [question]
        if self.rewrite:
            r = self.llm.rewrite(question)
            if r and r != question:
                queries.append(r)
        pools = [self.index.search(q, self.candidate_k, self.mode) for q in queries]
        hits = rrf_merge(pools) if len(pools) > 1 else pools[0]
        hits = self.reranker.rerank(question, hits, k) if self.rerank else hits[:k]
        return RetrievalResult([c for c, _ in hits], queries, {"retrieve": (perf_counter() - t0) * 1000})


class ExtractionAgent:
    """Structured extraction with a deterministic grounding check on every field."""

    def __init__(self, index: HybridIndex, llm):
        self.index, self.llm = index, llm

    def run(self, paper_id: str) -> dict[str, dict]:
        chunks = self.index.chunks_for(paper_id)
        by_id = {c.chunk_id: c for c in chunks}
        raw = self.llm.extract(paper_id, chunks)
        grounded = {}
        for name in FIELDS:
            f = raw.get(name)
            if not isinstance(f, dict) or f.get("chunk_id") not in by_id:
                continue
            quote, value = str(f.get("quote", "")), str(f.get("value", ""))
            if value and norm_ws(quote) in norm_ws(by_id[f["chunk_id"]].text) and norm_ws(value) in norm_ws(quote):
                grounded[name] = {"value": value, "chunk_id": f["chunk_id"], "quote": quote}
        return grounded


@dataclass
class Verdict:
    ok: bool
    reason: str


class VerificationAgent:
    """Deterministic checks first (citation exists, numbers appear in evidence), then one entailment call."""

    def __init__(self, llm):
        self.llm = llm

    def check(self, claims: list[Claim], chunks: list[Chunk]) -> list[Verdict]:
        by_id = {c.chunk_id: c for c in chunks}
        out = []
        for cl in claims:
            if not cl.chunk_ids:
                out.append(Verdict(False, "no citation"))
                continue
            missing = [i for i in cl.chunk_ids if i not in by_id]
            if missing:
                out.append(Verdict(False, f"cited chunks not retrieved: {missing}"))
                continue
            evidence = " ".join(by_id[i].text for i in cl.chunk_ids)
            ev_nums = set(numbers(evidence))
            bad = [n for n in numbers(cl.text) if n not in ev_nums]
            if bad:
                out.append(Verdict(False, f"numbers not in evidence: {bad}"))
            elif not self.llm.entails(cl.text, evidence):
                out.append(Verdict(False, "not entailed by cited evidence"))
            else:
                out.append(Verdict(True, "ok"))
        return out
