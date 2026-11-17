import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/sample_corpus.jsonl"


def load_qa():
    with open(ROOT / "eval/datasets/qa.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
