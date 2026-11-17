"""Hybrid retrieval: BM25 + dense (RRF fusion) + reranker. Pure stdlib by default."""
from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass

from .textutil import terms, tokens, STOPWORDS
from .types import Chunk

Hit = tuple[Chunk, float]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(1, len(docs))
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_terms: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in query_terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


class HashingEmbedder:
    """Deterministic char-n-gram hashing embedder (offline stand-in for a neural encoder)."""

    def __init__(self, dim: int = 2048, ngrams: tuple[int, ...] = (3, 4, 5)):
        self.dim, self.ngrams = dim, ngrams

    def embed(self, text: str) -> dict[int, float]:
        counts: Counter = Counter()
        for tok in tokens(text):
            if tok in STOPWORDS:
                continue
            p = f"<{tok}>"
            for n in self.ngrams:
                for i in range(len(p) - n + 1):
                    counts[zlib.crc32(p[i : i + n].encode()) % self.dim] += 1
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}


class SentenceTransformerEmbedder:
    """Optional neural embedder: pip install 'scilit[dense]'."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model)

    def embed(self, text: str) -> dict[int, float]:
        v = self.m.encode(text, normalize_embeddings=True)
        return {i: float(x) for i, x in enumerate(v)}


def _dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def rrf_merge(pools: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    score: dict[str, float] = defaultdict(float)
    by_id: dict[str, Chunk] = {}
    for pool in pools:
        for rank, (chunk, _) in enumerate(pool, start=1):
            score[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            by_id[chunk.chunk_id] = chunk
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(by_id[i], s) for i, s in ranked]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder or HashingEmbedder()
        self.bm25 = BM25([terms(c.index_text) for c in chunks])
        self.vecs = [self.embedder.embed(c.index_text) for c in chunks]

    def chunks_for(self, paper_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.paper_id == paper_id]

    def _rank(self, scores: list[float], k: int) -> list[Hit]:
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [(self.chunks[i], scores[i]) for i in order[:k] if scores[i] > 0]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        if mode == "bm25":
            return self._rank(self.bm25.scores(terms(query)), k)
        if mode == "dense":
            q = self.embedder.embed(query)
            return self._rank([_dot(q, v) for v in self.vecs], k)
        if mode == "hybrid":
            pool = max(k, 20)
            return rrf_merge([self.search(query, pool, "bm25"), self.search(query, pool, "dense")])[:k]
        raise ValueError(f"unknown retrieval mode: {mode}")


class LexicalReranker:
    """IDF-weighted query-term coverage plus a bigram bonus. Offline stand-in for a cross-encoder."""

    def __init__(self, index: HybridIndex):
        self.idf = index.bm25.idf
        self.max_idf = max(self.idf.values(), default=1.0)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        q = terms(query)
        qset, qbi = set(q), set(zip(q, q[1:]))
        denom = sum(self.idf.get(t, self.max_idf) for t in qset) or 1.0
        out = []
        for chunk, _ in hits:
            ct = terms(chunk.index_text)
            cs, cbi = set(ct), set(zip(ct, ct[1:]))
            cov = sum(self.idf.get(t, self.max_idf) for t in qset if t in cs) / denom
            bi = len(qbi & cbi) / max(1, len(qbi))
            out.append((chunk, cov + 0.5 * bi))
        out.sort(key=lambda h: (-h[1], h[0].chunk_id))
        return out[:k]


class CrossEncoderReranker:
    """Optional neural reranker: pip install 'scilit[dense]'."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.m = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scores = self.m.predict([(query, c.text) for c, _ in hits])
        out = sorted(zip((c for c, _ in hits), map(float, scores)), key=lambda h: (-h[1], h[0].chunk_id))
        return out[:k]
