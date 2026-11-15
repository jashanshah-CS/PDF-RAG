# Retrieval quality gates

Version 3 protects retrieval quality with a repeatable benchmark, explicit
acceptance thresholds, automated tests, and GitHub Actions. The quality check
runs before answer generation, so Ollama speed and wording do not affect the
result.

## What is checked

The versioned policy is
`evaluation/retrieval_v3_quality_gates.json`. A report is accepted only when:

| Gate | Required value | Why it matters |
| --- | ---: | --- |
| Retriever | `reranked` | Prevents validating the wrong search pipeline |
| Top K | `5` | Keeps measurements comparable |
| Total cases | `20` | Detects accidentally removed benchmark cases |
| Supported cases | `18` | Preserves the answerable test set |
| Unsupported probes | `2` | Preserves refusal-calibration examples |
| Recall@5 | `>= 1.0` | Every expected evidence passage remains in the top five |
| Mean reciprocal rank | `>= 1.0` | Expected evidence remains in the first position |
| Source accuracy@5 | `>= 1.0` | Every expected document is retrieved |
| Location accuracy@5 | `>= 1.0` | Every expected page or URL is retrieved |
| Multi-source coverage@5 | `>= 1.0` | Comparison questions retrieve every required source |
| Context reduction | `>= 20%` | Sentence windows continue to reduce prompt size |

These values are intentionally strict because the committed synthetic corpus
currently achieves perfect deterministic relevance metrics. A threshold should
only change when the benchmark itself changes or an explicitly reviewed
trade-off is supported by evidence. Do not lower a threshold merely to make a
failing change pass.

Latency is recorded in every report but is not a CI gate. Shared CI runners
have variable CPU load, so a fixed millisecond limit would create flaky builds.
The unsupported-query top score is also recorded but remains observational
until a refusal threshold has been calibrated with a larger dataset.

## Run locally

Run all unit and integration tests:

```powershell
uv run pytest -q
```

Check the committed reranker report without rebuilding embeddings:

```powershell
uv run python check_retrieval_quality.py
```

For a real regression check, generate a fresh report and validate it:

```powershell
uv run python evaluate_retrieval.py `
  --retriever reranked `
  --output tmp/retrieval-v3-current.json

uv run python check_retrieval_quality.py `
  --report tmp/retrieval-v3-current.json
```

The evaluator deliberately loads the embedding model from the local
`.model-cache`. Start the application once or otherwise download the configured
embedding model before running the benchmark on a new computer.

## What happens in GitHub Actions

`.github/workflows/retrieval-quality.yml` runs for pull requests and pushes to
`version-3`. It:

1. installs the dependencies exactly from `uv.lock`;
2. restores or downloads the local embedding model;
3. runs the complete Pytest suite;
4. creates a fresh reranked retrieval report from the tracked PDF corpus;
5. applies the committed quality policy; and
6. uploads the generated JSON report as a workflow artifact, even when a gate
   fails.

A regression produces a non-zero exit code and lists every failed metric. For
example:

```text
Retrieval quality gates failed:
  - summary.recall_at_5: 0.95 < minimum 1.0
```

## Files involved

- `evaluation/retrieval_v3_cases.json`: human-authored questions and evidence
  judgements.
- `evaluation/retrieval_v3_quality_gates.json`: required metadata and minimum
  metrics.
- `evaluate_retrieval.py`: generates a fresh benchmark report.
- `check_retrieval_quality.py`: command-line gate checker.
- `src/rag_project/quality_gates.py`: reusable validation logic.
- `tests/test_quality_gates.py`: passing, failing, malformed, and missing-metric
  test cases.
- `.github/workflows/retrieval-quality.yml`: CI automation.
