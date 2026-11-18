# Version 4 Final

Version 4 turns the local document Q&A interface into a persistent,
evidence-grounded conversation workspace. It preserves Version 3's measured
retrieval pipeline while adding multi-turn context, streaming generation,
claim-level evidence, and deterministic conversational quality gates.

## Conversation pipeline

```text
Active conversation and its selected sources
    ↓
Bounded completed-message history
    ↓
Contextual follow-up rewritten as a standalone retrieval question
    ↓
Version 3 hybrid search, sentence windows, and deterministic reranking
    ↓
Top five evidence passages
    ↓
Cancellable local Ollama streaming job
    ↓
Validated atomic claims with source relationships
    ↓
Supported, uncertain, or unsupported evidence state
    ↓
Completed message and citations stored atomically in SQLite
```

## Highlights

- Added persistent SQLite conversations with stable identifiers, titles,
  selected source scopes, messages, statuses, retrieval queries, confidence,
  and claim-specific citations.
- Added create, rename, select, search, and delete controls plus an accessible
  Light and Dark mode conversation timeline.
- Restricted every follow-up rewrite to the active conversation's bounded,
  completed history so another conversation cannot leak into retrieval.
- Treated history and document text as untrusted data that cannot override
  grounding instructions.
- Added background streaming jobs with visible running, completed, cancelled,
  and failed states.
- Persisted only complete validated answers; cancellation, failure, and process
  recovery discard partial answer text.
- Associated every factual claim with the exact retrieved passages that support
  it and made those passages inspectable from the answer.
- Added discrete supported, uncertain, and unsupported evidence states without
  presenting similarity as a probability.
- Benchmarked an optional local cross-encoder and retained the deterministic
  reranker because the learned model added no quality on the committed corpus.
- Added a deterministic conversational benchmark and enforced it alongside the
  Version 3 retrieval gates in GitHub Actions.

## Storage and isolation

Version 4 uses `data/conversations.db` for conversations and messages. ChromaDB
continues to store searchable document chunks and embeddings, while the legacy
Version 2 question history remains readable and unchanged.

Each conversation stores its own ordered source keys. Follow-up rewriting reads
only completed user and assistant messages from the active conversation. Failed,
cancelled, or pending messages are excluded from rewrite context. Deleting a
conversation cascades to its messages and citations without deleting indexed
documents or another conversation's data.

## Quality measurements

The committed conversational benchmark contains 14 cases across six categories:

| Category | Cases | Result |
| --- | ---: | ---: |
| Follow-up rewriting | 3 | 100% |
| Ambiguous references | 2 | 100% |
| Unsupported answers | 2 | 100% |
| Citation correctness | 3 | 100% |
| Conversation isolation | 2 | 100% |
| Failed or cancelled generation | 2 | 100% |

The benchmark substitutes deterministic fixtures only at the Ollama request
boundary. It still executes the production query rewriter, answer finalizer,
citation mapper, confidence calibration, and SQLite conversation store.

The optional `cross-encoder/ms-marco-MiniLM-L6-v2` experiment matched the
deterministic control at 100% Recall@5 and MRR 1.000. Its mean latency was
773.8 ms versus 262.9 ms for the control, about 2.94 times slower, and it added
91,815,798 bytes of model cache. The production app therefore retains the
deterministic reranker.

## Verification

- All 198 automated tests pass locally on the completed Version 4 branch.
- All 14 conversational cases pass every committed quality gate.
- The deterministic retrieval control retains 100% Recall@5 and MRR 1.000 and
  passes the Version 3 policy.
- The optional cross-encoder decision is recorded as `retain-deterministic`.
- Python compilation and Git whitespace checks pass.
- GitHub Actions rebuilds and gates both retrieval and conversation reports.
- The unrelated locally regenerated Version 3 semantic baseline report remains
  deliberately excluded from all release commits.

## Reproduce the checks

```powershell
uv sync --frozen
uv run pytest -q

uv run python evaluate_retrieval.py `
  --retriever reranked `
  --output tmp/retrieval-v4-release.json

uv run python check_retrieval_quality.py `
  --report tmp/retrieval-v4-release.json

uv run python evaluate_conversations.py `
  --output tmp/conversation-v4-release.json

uv run python check_conversation_quality.py `
  --report tmp/conversation-v4-release.json
```

The retrieval evaluator needs the configured embedding model in
`.model-cache`. GitHub Actions restores or downloads it. The conversational
benchmark is offline and does not require an embedding model or Ollama.

## Known limitations

- Conversations are local and single-user; authentication and shared
  organization workspaces are out of scope.
- Follow-up rewriting depends on the configured local model in the live app. If
  rewriting fails, the app safely searches the original question.
- A small local generation model can still produce incomplete or poorly worded
  answers, so users should inspect the attached evidence.
- The conversational benchmark validates deterministic application behaviour,
  not every possible natural-language response from Ollama.
- The retrieval corpus remains intentionally small and synthetic. Passing its
  gates prevents known regressions but does not prove quality in every domain.
- OCR is English-only and may struggle with handwriting, damaged scans, and
  complex layouts.
- Website ingestion supports static same-host HTML and does not execute
  JavaScript.

## Version 4 commit sequence

- `5bbe6a0` — Version 4 conversational roadmap
- `b307892` — persistent conversation storage
- `e330083` — conversation management and chat interface
- `d19190d` — context-aware follow-up rewriting
- `4a5fe8e` — cancellable streaming generation
- `3e44011` — claim-level citations and calibrated confidence
- `83265bf` — optional cross-encoder benchmark
- `b066722` — conversational benchmark and CI gates
