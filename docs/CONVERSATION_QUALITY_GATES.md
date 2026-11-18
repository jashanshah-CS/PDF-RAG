# Version 4 conversational quality gates

The Version 4 benchmark protects the behaviour around a document conversation,
not the wording quality of a particular language model. It uses deterministic
model-response fixtures and runs entirely offline so results remain repeatable
on a developer computer and in GitHub Actions.

## What is exercised

The 14 cases in `evaluation/conversation_v4_cases.json` pass through production
helpers and cover six categories:

| Category | Cases | Production behaviour |
| --- | ---: | --- |
| Follow-up rewrite | 3 | Contextual questions become standalone retrieval queries |
| Ambiguous reference | 2 | Ambiguous wording remains unchanged instead of inventing context |
| Unsupported answer | 2 | Unsupported generated claims become the safe refusal |
| Citation correctness | 3 | Source labels resolve to the exact evidence and confidence state |
| Conversation isolation | 2 | One conversation's rewrite prompt cannot contain another's history |
| Incomplete generation | 2 | Failed or cancelled drafts cannot become completed cited answers |

The evaluator calls the real query rewriter, grounded-answer finalizer,
citation mapper, confidence calibrator, and SQLite conversation store. Ollama is
replaced only at the model-request boundary with the response stored in each
case. This keeps the test deterministic while preserving the application logic
around the model.

## Acceptance policy

`evaluation/conversation_v4_quality_gates.json` requires:

- exactly 14 cases and 14 passes;
- an overall pass rate of `1.0`; and
- category accuracy of `1.0` for all six categories.

The thresholds are deliberately strict because every check is deterministic.
If behaviour intentionally changes, update the human-readable case first,
review the new result, and then update the policy. Do not reduce a threshold to
hide a regression.

## Run locally

Generate a fresh report and enforce the policy:

```powershell
uv run python evaluate_conversations.py `
  --output tmp/conversation-v4-current.json

uv run python check_conversation_quality.py `
  --report tmp/conversation-v4-current.json
```

The first command exits unsuccessfully for invalid benchmark data. The second
lists every missing or regressed metric and exits unsuccessfully when any gate
fails.

Run the full regression suite as well:

```powershell
uv run pytest -q
```

## Continuous integration

`.github/workflows/retrieval-quality.yml` runs on pull requests and pushes to
`main`, `version-3`, and `version-4`. It:

1. installs locked dependencies;
2. runs the complete Pytest suite;
3. rebuilds and gates the Version 3 retrieval report;
4. rebuilds and gates the Version 4 conversational report; and
5. uploads both reports as one workflow artifact, including on failure.

The conversational benchmark does not load an embedding model, contact Ollama,
or access the network.

## Files involved

- `evaluation/conversation_v4_cases.json`: versioned human-readable cases.
- `evaluation/conversation_v4_quality_gates.json`: required values and minimums.
- `evaluation/conversation_v4_results.json`: committed passing baseline report.
- `src/rag_project/conversation_evaluation.py`: loader and evaluator.
- `evaluate_conversations.py`: report generator.
- `check_conversation_quality.py`: command-line gate checker.
- `tests/test_conversation_evaluation.py`: evaluator and regression tests.
- `.github/workflows/retrieval-quality.yml`: fresh CI evaluation and artifacts.
