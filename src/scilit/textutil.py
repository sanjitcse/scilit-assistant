"""Small, dependency-free text helpers shared by retrieval, agents and metrics."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9\-\.]*[A-Za-z0-9])?")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

STOPWORDS = frozenset(
    """a an the of on in to for with and or by at from that this these those as than is are was
    were be been what which who whom how does do did can could would should will it its into over
    under per via not no we our their there about author authors report reports reported paper""".split()
)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def stem(t: str) -> str:
    """Deliberately tiny stemmer: plural 's' and trailing 'e'."""
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def terms(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def numbers(text: str) -> list[str]:
    return NUM_RE.findall(text)


def entity_terms(question: str) -> list[str]:
    """Name-like tokens in a question (inner capitals, digits, or mid-sentence capitals)."""
    out = []
    for i, tok in enumerate(TOKEN_RE.findall(question)):
        if i == 0:
            continue
        low = tok.lower()
        if low in STOPWORDS:
            continue
        if any(c.isupper() for c in tok[1:]) or any(c.isdigit() for c in tok) or tok[0].isupper():
            out.append(stem(low))
    return out


def norm_ws(s: str) -> str:
    return " ".join(s.split()).lower()
