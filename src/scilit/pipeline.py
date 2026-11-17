"""End-to-end QA pipeline, baselines and the named system variants used in evaluation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from .agents import RetrievalAgent, VerificationAgent
from .ingest import load_chunks
from .retrieval import HybridIndex
from .types import Answer, Claim


@dataclass(frozen=True)
class PipelineConfig:
    k: int = 5
    candidate_k: int = 20
    mode: str = "hybrid"  # bm25 | dense | hybrid
    rewrite: bool = True
    rerank: bool = True
    verify: bool = True
    max_retries: int = 1  # extra attempt (k doubled) when verification rejects every claim


def build_index(corpus_path: str | Path, embedder=None) -> HybridIndex:
    return HybridIndex(load_chunks(corpus_path), embedder)


class Pipeline:
    def __init__(self, index: HybridIndex, llm, cfg: PipelineConfig = PipelineConfig(), reranker=None):
        self.index, self.llm, self.cfg = index, llm, cfg
        self.retriever = RetrievalAgent(index, llm, cfg.k, cfg.candidate_k, cfg.mode, cfg.rewrite, cfg.rerank, reranker)
        self.verifier = VerificationAgent(llm)

    def ask(self, question: str) -> Answer:
        t0 = perf_counter()
        timings = {"retrieve": 0.0, "answer": 0.0, "verify": 0.0}
        k, retrieved, supported, verdict_log, queries = self.cfg.k, [], [], [], []
        attempts = self.cfg.max_retries + 1 if self.cfg.verify else 1
        used = 0
        for _ in range(attempts):
            used += 1
            ret = self.retriever.run(question, k=k)
            timings["retrieve"] += ret.timings["retrieve"]
            retrieved, queries = [c.chunk_id for c in ret.chunks], ret.queries
            t = perf_counter()
            raw = self.llm.answer(question, ret.chunks)
            timings["answer"] += (perf_counter() - t) * 1000
            claims = [Claim(c["text"], list(c.get("chunk_ids", []))) for c in raw.get("claims", [])]
            if raw.get("abstain") or not claims:
                supported = []
                break  # model abstained: do not retry
            if not self.cfg.verify:
                supported = claims
                break
            t = perf_counter()
            verdicts = self.verifier.check(claims, ret.chunks)
            timings["verify"] += (perf_counter() - t) * 1000
            verdict_log.append([{"claim": c.text, "ok": v.ok, "reason": v.reason} for c, v in zip(claims, verdicts)])
            supported = [c for c, v in zip(claims, verdicts) if v.ok]
            if supported:
                break
            k *= 2
        timings["total"] = (perf_counter() - t0) * 1000
        return Answer(question, supported, not supported, retrieved, timings,
                      {"queries": queries, "attempts": used, "verdicts": verdict_log})


class ClosedBook:
    """Baseline: the LLM answers from parametric knowledge, no retrieval, no citations."""

    def __init__(self, llm):
        self.llm = llm

    def ask(self, question: str) -> Answer:
        t0 = perf_counter()
        text = self.llm.closed_book(question)
        ms = (perf_counter() - t0) * 1000
        claims = [Claim(text, [])] if text else []
        return Answer(question, claims, not claims, [], {"retrieve": 0.0, "answer": ms, "verify": 0.0, "total": ms})


_FULL = PipelineConfig()
VARIANTS: dict[str, PipelineConfig] = {
    "dense_rag": replace(_FULL, mode="dense", rewrite=False, rerank=False, verify=False),
    "full": _FULL,
    "full-no_rewrite": replace(_FULL, rewrite=False),
    "full-no_rerank": replace(_FULL, rerank=False),
    "full-no_hybrid": replace(_FULL, mode="dense"),
    "full-no_verify": replace(_FULL, verify=False, max_retries=0),
}
SYSTEMS = ["closed_book", *VARIANTS]


def make_system(name: str, index: HybridIndex, llm, k: int = 5):
    if name == "closed_book":
        return ClosedBook(llm)
    return Pipeline(index, llm, replace(VARIANTS[name], k=k))
