# Optional Cross-encoder Reranker Experiment

## Decision

**Retain the deterministic Version 3 reranker in production.**

The optional cross-encoder matched the existing retriever's already-perfect
quality on this benchmark but was 2.94 times slower and required an additional
91,815,798 bytes (about 87.6 MiB) of local model cache. It therefore failed the
predeclared requirement for a meaningful MRR improvement.

## Implementation

The experiment uses `cross-encoder/ms-marco-MiniLM-L6-v2` through
`sentence-transformers 6.1.0`. The official API describes CrossEncoder as a
model that scores sentence pairs rather than producing embeddings and supports
both `predict()` and `local_files_only`:
[Sentence Transformers CrossEncoder documentation](https://sbert.net/docs/package_reference/cross_encoder/model.html).

The pipeline is:

1. Run the existing hybrid retriever.
2. Focus each candidate with the existing sentence-window logic.
3. Expand the candidate pool to 10 passages.
4. Score each `(question, passage)` pair with the CPU cross-encoder.
5. Sort scores descending with stable input-order tie breaking.
6. Return the top five passages.

The model loader is offline by default. A first download requires the explicit
`--allow-model-download` flag. If the optional model is absent or prediction
fails in fallback mode, `CrossEncoderRetriever` uses the current deterministic
reranker. Strict benchmark mode disables this fallback so a misleading report
cannot silently measure the wrong implementation.

## Results

| Metric | Deterministic control | Cross-encoder | Difference |
|---|---:|---:|---:|
| Recall@5 | 100.0% | 100.0% | 0.0 pp |
| MRR | 1.000 | 1.000 | 0.000 |
| Source accuracy@5 | 100.0% | 100.0% | 0.0 pp |
| Location accuracy@5 | 100.0% | 100.0% | 0.0 pp |
| Multi-source coverage@5 | 100.0% | 100.0% | 0.0 pp |
| Mean latency | 262.9 ms | 773.8 ms | +510.9 ms |
| Latency ratio | 1.00× | 2.94× | +1.94× |
| Extra model cache | 0 | 91,815,798 bytes | +87.6 MiB |

The reports are committed as:

- `evaluation/retrieval_v4_deterministic_control_results.json`
- `evaluation/retrieval_v4_cross_encoder_results.json`
- `evaluation/retrieval_v4_reranker_decision.json`

## Adoption rule

`compare_rerankers.py` adopts the learned reranker only when:

- Recall@5 does not regress;
- MRR improves by at least 0.01;
- mean latency remains within 3× of the deterministic control.

The observed MRR gain was 0.000, so the generated decision is
`retain-deterministic`. This prevents adding model size and latency merely
because a learned method sounds more sophisticated.

## Reproduction

```powershell
uv run python evaluate_retrieval.py `
  --retriever reranked `
  --output evaluation/retrieval_v4_deterministic_control_results.json

uv run python evaluate_retrieval.py `
  --retriever cross-encoder `
  --allow-model-download `
  --output evaluation/retrieval_v4_cross_encoder_results.json

uv run python compare_rerankers.py `
  --deterministic evaluation/retrieval_v4_deterministic_control_results.json `
  --cross-encoder evaluation/retrieval_v4_cross_encoder_results.json
```

After the first download, omit `--allow-model-download` to prove the experiment
runs entirely from the local cache.
