# Limitations

Read this before quoting any number from this repository.

## What the bundled results are (and are not)

- The sample corpus (`data/sample_corpus.jsonl`) is **synthetic**: 8 fictional papers, 32 chunks, built from templates by
  `scripts/make_sample_data.py`. Every system, dataset and number in it is invented.
- The gold labels (`eval/datasets/*.jsonl`) are generated from the same fixture. They cannot disagree with the corpus,
  which is good for harness tests and bad as evidence of real-world accuracy.
- The `offline` LLM backend is a **heuristic stub** (sentence/term overlap, regex extraction). It exists so that the
  pipeline, tests and eval harness run with no network or API key. Numbers produced with it validate the *plumbing*
  (metrics, logging, ablations wiring, determinism). They say nothing about how an LLM would perform.
- Extraction F1 of 1.00 offline is expected: the stub's regexes match the generator's templates.
- At k=5 the offline benchmark saturates (every retrieval variant reaches 1.00) because the corpus has 32 chunks.
  `--k 1` is the stress setting in which the variants separate (`docs/results_offline_k1.md`).
- The `closed_book` row scores 1.00 on abstention only because the stub never answers. With a real LLM this baseline
  will hallucinate on unanswerable questions; that is what the baseline is there to measure.

## What has not been exercised

- `AnthropicLLM` is unit-tested against a fake client (parsing, caching, temperature fallback) but has **not** been run
  against the live API in the environment this repo was built in. Run a real experiment before drawing conclusions:
  `python -m scilit.eval.run_eval --llm anthropic --no-cache --repeats 5`.
- `SentenceTransformerEmbedder` and `CrossEncoderReranker` are provided as optional drop-ins and are untested here.
- Ingestion starts from a JSONL of already-sectioned papers. **PDF parsing (e.g. GROBID) is not implemented.** Real PDFs
  bring table/figure/equation extraction errors that this repo does not model.

## Method limitations to expect on real data

- **Metric validity.** Answer correctness is gold-keyword containment: cheap and deterministic, but it penalises correct
  paraphrases and rewards keyword-stuffed answers. For real runs add a calibrated LLM judge (different model family
  from the generator) and report its agreement with human labels.
- **Citation correctness** is measured at chunk granularity against gold evidence chunks. A claim can cite the right
  chunk and still misstate it; the verification agent's number check and entailment call reduce but do not remove this.
- **Verification** relies on the same model family unless you configure otherwise, so correlated errors are possible.
- **Latency** in offline mode is microseconds of Python and is not representative. Real latency is dominated by API calls;
  use `--no-cache` or cached responses will make repeated runs look instant and perfectly reproducible.
- **Reproducibility** is controlled by temperature 0, pinned model string, disk cache and logged traces, but hosted models
  can still change behaviour between dates. Record the model string and run date from `results.json`.
- **Scale.** Retrieval is an in-memory linear scan. Fine for thousands of chunks; use an ANN index beyond ~100k.
- **Scope.** English text only; single-hop questions only; no multi-paper synthesis questions in the sample QA set.
