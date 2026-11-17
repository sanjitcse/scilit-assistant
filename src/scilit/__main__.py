"""CLI:  python -m scilit ask "What IoU does GlacierNet achieve on AlpineSat-12k?"
        python -m scilit extract P001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import ExtractionAgent
from .llm import make_llm
from .pipeline import Pipeline, PipelineConfig, build_index

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(prog="scilit")
    ap.add_argument("--llm", default="offline", choices=["offline", "anthropic"])
    ap.add_argument("--corpus", default=str(ROOT / "data/sample_corpus.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("extract").add_argument("paper_id")
    args = ap.parse_args()

    index, llm = build_index(args.corpus), make_llm(args.llm)
    if args.cmd == "ask":
        a = Pipeline(index, llm, PipelineConfig()).ask(args.question)
        print(a.text)
        for c in a.claims:
            for cid in c.chunk_ids:
                print(f"  [{cid}] {index.by_id[cid].text[:160]}")
        print(f"\n({a.timings['total']:.1f} ms; retrieved {a.retrieved})")
    else:
        print(json.dumps(ExtractionAgent(index, llm).run(args.paper_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
