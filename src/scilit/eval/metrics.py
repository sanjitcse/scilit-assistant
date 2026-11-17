from __future__ import annotations

import math

from ..textutil import norm_ws
from ..types import Answer


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = (len(s) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def answer_correct(ans: Answer, item: dict) -> bool:
    """Unanswerable: correct iff abstained. Answerable: not abstained and every gold keyword present."""
    if not item["answerable"]:
        return ans.abstained
    if ans.abstained:
        return False
    text = norm_ws(ans.text)
    return all(norm_ws(k) in text for k in item["gold_keywords"])


def citation_scores(ans: Answer, item: dict) -> tuple[float | None, float | None]:
    """(precision, recall) of cited chunk ids vs gold evidence. None when undefined."""
    if not item["answerable"]:
        return None, None
    gold, cited = set(item["evidence"]), ans.cited
    precision = len(cited & gold) / len(cited) if cited else None
    recall = len(cited & gold) / len(gold) if gold else None
    return precision, recall


def retrieval_recall(ans: Answer, item: dict) -> float | None:
    if not item["answerable"] or not item["evidence"]:
        return None
    return len(set(ans.retrieved) & set(item["evidence"])) / len(item["evidence"])


def extraction_prf(pred: dict[str, dict], gold: dict[str, str]) -> tuple[int, int, int]:
    """Returns (true_positives, n_predicted, n_gold) for field-level exact match (whitespace/case-normalised)."""
    tp = sum(1 for k, v in pred.items() if k in gold and norm_ws(v["value"]) == norm_ws(gold[k]))
    return tp, len(pred), len(gold)


def prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")
