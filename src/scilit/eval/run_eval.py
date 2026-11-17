"""Evaluate systems on the QA + extraction sets.

  python -m scilit.eval.run_eval                          # offline stub: validates the harness only
  python -m scilit.eval.run_eval --llm anthropic --no-cache --repeats 5   # real experiment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..agents import ExtractionAgent
from ..llm import make_llm
from ..pipeline import SYSTEMS, build_index, make_system
from .metrics import (answer_correct, citation_scores, extraction_prf, mean, percentile, prf,
                      retrieval_recall)

ROOT = Path(__file__).resolve().parents[3]


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def evaluate_system(system, llm, items: list[dict], repeats: int) -> tuple[dict, list[dict]]:
    before = dict(llm.usage)
    stage: dict[str, list[float]] = defaultdict(list)
    correct, per_type, prec, rec, rret, stable, traces = [], defaultdict(list), [], [], [], 0, []
    for it in items:
        answers = [system.ask(it["question"]) for _ in range(repeats)]
        a = answers[0]
        for x in answers:
            for k, v in x.timings.items():
                stage[k].append(v)
        stable += len({(x.abstained, x.text, tuple(x.retrieved)) for x in answers}) == 1
        ok = answer_correct(a, it)
        correct.append(ok)
        per_type[it["type"]].append(ok)
        p, r = citation_scores(a, it)
        prec.append(p)
        rec.append(0.0 if (r is None and it["answerable"]) else r)
        rret.append(retrieval_recall(a, it))
        traces.append({"id": it["id"], "question": it["question"], "answer": a.text, "abstained": a.abstained,
                       "cited": sorted(a.cited), "retrieved": a.retrieved, "correct": ok, "trace": a.trace})
    ans_items = [c for c, it in zip(correct, items) if it["answerable"]]
    una_items = [c for c, it in zip(correct, items) if not it["answerable"]]
    n_tok = {k: llm.usage[k] - before[k] for k in llm.usage}
    n = max(1, len(items) * repeats)
    summary = {
        "n_questions": len(items), "repeats": repeats,
        "accuracy": mean(correct), "accuracy_answerable": mean(ans_items), "abstention_accuracy": mean(una_items),
        "accuracy_by_type": {k: mean(v) for k, v in per_type.items()},
        "citation_precision": mean(prec), "citation_recall": mean(rec), "retrieval_recall_at_k": mean(rret),
        "latency_ms": {k: {"p50": percentile(v, 50), "p95": percentile(v, 95)} for k, v in stage.items()},
        "reproducibility": stable / len(items),
        "llm_usage_per_question": {k: v / n for k, v in n_tok.items()},
    }
    return summary, traces


def evaluate_extraction(index, llm, gold_rows: list[dict]) -> dict:
    agent, tp, npred, ngold = ExtractionAgent(index, llm), 0, 0, 0
    for row in gold_rows:
        a, b, c = extraction_prf(agent.run(row["paper_id"]), row["fields"])
        tp, npred, ngold = tp + a, npred + b, ngold + c
    p, r, f1 = prf(tp, npred, ngold)
    return {"precision": p, "recall": r, "f1": f1, "n_gold_fields": ngold}


def markdown_table(results: dict) -> str:
    head = ("| System | Acc | Acc (answerable) | Abstain acc | Cite P | Cite R | Retr R@k | p50 ms | p95 ms | Repro |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    f = lambda x: "n/a" if x != x else f"{x:.2f}"  # noqa: E731
    rows = []
    for name, s in results["systems"].items():
        t = s["latency_ms"].get("total", {"p50": float("nan"), "p95": float("nan")})
        rows.append(f"| {name} | {f(s['accuracy'])} | {f(s['accuracy_answerable'])} | {f(s['abstention_accuracy'])} | "
                    f"{f(s['citation_precision'])} | {f(s['citation_recall'])} | {f(s['retrieval_recall_at_k'])} | "
                    f"{t['p50']:.2f} | {t['p95']:.2f} | {f(s['reproducibility'])} |")
    e = results["extraction"]
    return head + "\n".join(rows) + (f"\n\nExtraction (field-level): P={e['precision']:.2f} R={e['recall']:.2f} "
                                     f"F1={e['f1']:.2f} over {e['n_gold_fields']} gold fields\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="offline", choices=["offline", "anthropic"])
    ap.add_argument("--corpus", default=str(ROOT / "data/sample_corpus.jsonl"))
    ap.add_argument("--qa", default=str(ROOT / "eval/datasets/qa.jsonl"))
    ap.add_argument("--extraction", default=str(ROOT / "eval/datasets/extraction.jsonl"))
    ap.add_argument("--systems", nargs="*", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true", help="disable the LLM response cache (use for latency/variance)")
    ap.add_argument("--out", default=str(ROOT / "experiments"))
    args = ap.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {} if args.llm == "offline" else {"cache": not args.no_cache, "trace_path": str(out_dir / "llm_calls.jsonl")}
    llm = make_llm(args.llm, **kw)
    index = build_index(args.corpus)
    items = _jsonl(Path(args.qa))

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip() or None
    except OSError:
        commit = None
    results = {
        "manifest": {"run_id": run_id, "llm": llm.name, "k": args.k, "repeats": args.repeats,
                     "cache": args.llm != "offline" and not args.no_cache, "python": sys.version.split()[0],
                     "platform": platform.platform(), "git_commit": commit,
                     "data_sha256": {p: _sha(Path(p)) for p in (args.corpus, args.qa, args.extraction)}},
        "systems": {}, "extraction": {},
    }
    for name in args.systems:
        summary, traces = evaluate_system(make_system(name, index, llm, args.k), llm, items, args.repeats)
        results["systems"][name] = summary
        with open(out_dir / f"traces_{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in traces)
    results["extraction"] = evaluate_extraction(index, llm, _jsonl(Path(args.extraction)))
    table = markdown_table(results)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / "results.md").write_text(table)
    print(f"[{llm.name}] run {run_id} -> {out_dir}\n")
    print(table)


if __name__ == "__main__":
    main()
