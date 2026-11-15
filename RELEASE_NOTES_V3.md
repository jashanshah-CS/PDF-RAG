# Version 3 Final

Version 3 upgrades the search layer of the local RAG application from pure
semantic similarity to a measured, multi-stage retrieval pipeline. Generation,
storage, citations, OCR, source selection, and the Version 2 interface remain
local and continue to work with the improved evidence ordering.

## Retrieval pipeline

```text
Selected ChromaDB chunks
    ↓
Semantic candidate ranking + BM25 candidate ranking
    ↓
Weighted reciprocal rank fusion
    ↓
Confidence-aware sentence windows
    ↓
Deterministic second-stage reranking
    ↓
Top five evidence passages for local Qwen3 8B
```

## Highlights

- Added a versioned JSON benchmark containing 20 human-authored retrieval
  cases: 18 supported questions and two unsupported probes.
- Added a standalone BM25 inverted index for exact terms, identifiers, numbers,
  and policy language.
- Combined semantic and BM25 rankings using deterministic weighted reciprocal
  rank fusion rather than mixing incompatible raw scores.
- Added confidence-aware sentence windows that retain relevant neighbouring
  sentences and use semantic fallback for paraphrased questions.
- Added a deterministic local reranker over the top five focused candidates.
  It combines the original hybrid rank, vector rank, and focused-text BM25 rank.
- Preserved document, page, URL, chunk, OCR, and citation metadata throughout
  every retrieval stage.
- Added explicit quality gates for benchmark identity, case counts, Recall@5,
  MRR, source accuracy, location accuracy, multi-source coverage, and context
  reduction.
- Added GitHub Actions automation that installs locked dependencies, runs every
  test, generates a fresh report, enforces the gates, and uploads the report.

## Benchmark results

All committed retrievers use the same 20-case benchmark and top-five result
limit.

| Retrieval stage | Recall@5 | MRR | Context reduction |
| --- | ---: | ---: | ---: |
| Version 3 semantic baseline | 100% | 0.958 | 0% |
| BM25 | 100% | 1.000 | 0% |
| Hybrid weighted RRF | 100% | 1.000 | 0% |
| Sentence windows | 100% | 1.000 | 21.9% locally |
| Final reranked pipeline | 100% | 1.000 | 21.9% locally |

On the broader private 24-case PDF benchmark, reranking preserved 100%
Recall@5 and improved MRR from 0.871 to 0.879. Mean retrieval time increased
from approximately 0.96 seconds to 1.28 seconds. That private corpus is not
committed because it contains user-provided documents.

GitHub's Linux runner measured 13.3% context reduction for the same tracked
benchmark, while Windows measured 21.9%. The quality policy therefore uses a
10% cross-platform floor. Relevance gates remain strict at 100% Recall@5 and
1.000 MRR.

## Verification

- All 134 automated tests pass.
- A fresh local reranker benchmark passes every committed quality gate.
- The GitHub Actions test-and-benchmark workflow passes on `version-3`.
- Python compilation and Git whitespace checks pass.
- The final tracked benchmark achieves 100% Recall@5, 1.000 MRR, 100% source
  accuracy, 100% location accuracy, and 100% multi-source coverage.
- The unrelated locally regenerated semantic baseline report is deliberately
  excluded from the release commits.

## Reproduce the checks

```powershell
uv sync --frozen
uv run pytest -q

uv run python evaluate_retrieval.py `
  --retriever reranked `
  --output tmp/retrieval-v3-release.json

uv run python check_retrieval_quality.py `
  --report tmp/retrieval-v3-release.json
```

The embedding model must already be available in `.model-cache` for the local
benchmark. GitHub Actions restores or downloads it before running the evaluator.

## Known limitations

- The committed benchmark is intentionally small and synthetic. It prevents
  regressions but does not prove performance on every writing style or domain.
- The reranker is deterministic rank fusion, not a larger learned cross-encoder.
  This keeps the application local and predictable but may miss subtle
  query-passage relationships.
- Unsupported-query similarity scores are recorded but not yet gated. A larger
  labelled dataset is needed before choosing a reliable refusal threshold.
- Retrieval latency is reported but not gated because CPU speed varies across
  local machines and shared CI runners.
- PDF extraction can produce slightly different sentence boundaries across
  operating systems, which changes the exact context-reduction percentage.
- OCR is English-only and can struggle with handwriting, damaged scans, and
  complex layouts.
- Website ingestion remains limited to static same-host HTML and does not run
  JavaScript.
- Local Qwen3 8B can still make generation mistakes even when the correct
  evidence is retrieved, so users should inspect the visible source cards.

## Version 3 commit sequence

- `dead5c1` — retrieval benchmark baseline
- `089f24c` — BM25 keyword retrieval
- `cc0cd8e` — hybrid reciprocal rank fusion
- `b55f74d` — confidence-aware sentence windows
- `a6eab26` — deterministic local reranker
- `1ce4e9d` — automated retrieval quality gates
- `cd377ea` — cross-platform context quality floor
