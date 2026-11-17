# scilit: an LLM research assistant for scientific literature

Question answering over scientific papers with **cited, verified answers**, structured information extraction, and a
systematic evaluation harness (factual accuracy, citation correctness, latency, reproducibility) against baselines.

```
question ─► RetrievalAgent ─► answer (LLM, cited) ─► VerificationAgent ─► cited answer | INSUFFICIENT EVIDENCE
            rewrite → BM25 + dense → RRF → rerank        ▲ retry once with larger k if every claim is rejected
paper ───► ExtractionAgent ─► {task, dataset, metric, value, baseline, baseline_value, limitation}
            each field carries chunk_id + verbatim quote, and is dropped if the quote is not in that chunk
```

The core library is **pure standard library** (no dependencies). Real LLM calls and neural retrieval are optional extras.

> **Status: tested scaffold + harness.** All bundled results use a synthetic corpus and a heuristic offline stub in place of
> an LLM. They validate the plumbing, not the method. See [docs/limitations.md](docs/limitations.md) before quoting
> anything, and run the real experiment (below) to get meaningful numbers.

## Quickstart

```bash
git clone <this repo> && cd scilit-assistant
python -m unittest discover -s tests -t .          # 30 tests, no network (or: pip install -e ".[dev]" && pytest)
PYTHONPATH=src python -m scilit ask "What IoU does GlacierNet achieve on AlpineSat-12k?"
PYTHONPATH=src python -m scilit extract P002
PYTHONPATH=src python -m scilit.eval.run_eval --repeats 3            # offline stub: harness check
PYTHONPATH=src python -m scilit.eval.run_eval --repeats 3 --k 1     # stress setting: variants separate
```

### Real experiment (Anthropic API)

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...            # optional: export SCILIT_MODEL=<model string>  (default: claude-sonnet-5-5)
python -m scilit --llm anthropic ask "..."
python -m scilit.eval.run_eval --llm anthropic --no-cache --repeats 5
```

Calls use temperature 0 (retried without it if a model rejects the parameter), and every prompt/response is logged to
`experiments/<run>/llm_calls.jsonl`. Without `--no-cache`, responses are cached on disk under `.cache/llm/`, which makes
re-runs free and bit-identical but makes latency numbers meaningless.

### Using your own papers

1. Produce a JSONL, one paper per line: `{"paper_id", "title", "doi", "sections": {"abstract": "...", "methods": "..."}}`
   (e.g. GROBID TEI → sections; PDF parsing is not included).
2. Write 50–150 questions in the format of `eval/datasets/qa.jsonl` (`question`, `answerable`, `gold_keywords`,
   `evidence` chunk ids like `PAPERID:results`, `type`) **before looking at system output**, plus a few unanswerable ones.
3. `python -m scilit.eval.run_eval --corpus my.jsonl --qa my_qa.jsonl --extraction my_extraction.jsonl --llm anthropic`.

## What is evaluated

| Dimension | Implementation (`src/scilit/eval/`) |
|---|---|
| Factual accuracy | Gold-keyword containment on answerable questions; abstention required on unanswerable ones. Overall, per-type and abstention accuracy. |
| Citation correctness | Chunk-level precision (cited ∈ gold evidence) and recall (gold evidence cited). |
| Retrieval | Evidence recall@k. |
| Extraction | Field-level exact-match P/R/F1 vs `extraction.jsonl`. |
| Latency | p50/p95 end-to-end and per stage (retrieve / answer / verify), plus LLM calls and tokens per question. |
| Reproducibility | Fraction of questions whose answer + retrieved set are identical across N repeats; manifest with model, python, git commit, data hashes; per-question traces. |

**Systems compared:** `closed_book` (no retrieval) · `dense_rag` (single-pass dense retrieval) · `full` · ablations
`full-no_rewrite`, `full-no_rerank`, `full-no_hybrid`, `full-no_verify`.

## Reference output (offline stub, synthetic corpus: harness validation only)

Stress setting, `--k 1`, 43 questions (40 answerable, 3 unanswerable), 3 repeats:

| System | Acc | Acc (answerable) | Abstain acc | Cite P | Cite R | Retr R@k | Repro |
|---|---|---|---|---|---|---|---|
| closed_book | 0.07 | 0.00 | 1.00 | n/a | 0.00 | 0.00 | 1.00 |
| dense_rag | 0.70 | 0.68 | 1.00 | 0.68 | 0.68 | 0.68 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| full-no_rewrite | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| full-no_rerank | 0.63 | 0.60 | 1.00 | 0.60 | 0.60 | 0.60 | 1.00 |
| full-no_hybrid | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| full-no_verify | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The only thing this shows is that the harness can tell variants apart: removing the reranker hurts at k=1, while the
rewrite, hybrid and verify ablations have no effect on this tiny synthetic set. At k=5 everything saturates at 1.00
(`docs/results_offline_k5.md`). Full tables with latency are in `docs/`.

## Layout

```
src/scilit/   textutil · ingest · retrieval (BM25, hashing/ST embedder, RRF, rerankers) · llm (offline + Anthropic)
              agents (Retrieval, Extraction, Verification) · pipeline (config, baselines, variants) · eval/ · __main__
eval/datasets qa.jsonl, extraction.jsonl          data/  sample_corpus.jsonl (synthetic)
scripts/      make_sample_data.py (regenerates corpus + gold)    tests/  30 tests     docs/  limitations + reference results
```

## Design notes

- Agents are plain classes with typed inputs/outputs; there is no agent framework. The "verification agent" is mostly
  deterministic (citation exists, every number in the claim appears in the cited evidence) plus one entailment call.
- Every claim must carry chunk-level citations; "INSUFFICIENT EVIDENCE" is a first-class output, and unparseable model
  output is treated as abstention rather than a guess.
- The offline stub and the Anthropic client implement one interface (`rewrite`, `answer`, `entails`, `extract`,
  `closed_book`), so swapping in another provider means implementing five methods in `llm.py`.

## License

MIT. Replace the placeholder name in `LICENSE`.
