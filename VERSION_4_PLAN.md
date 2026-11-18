# Version 4 Plan: Conversational and Trustworthy RAG

Version 4 will turn the single-question interface into a multi-turn document
conversation while preserving the local-first privacy, evidence grounding, and
retrieval quality established in Versions 1–3.

## Goals

- Let users create and continue separate document conversations.
- Resolve follow-up questions without allowing chat history to replace source
  evidence.
- Stream local Ollama answers and allow generation to be stopped.
- Attach citations to individual claims and make every cited passage inspectable.
- Calibrate supported-answer confidence and safe refusal behaviour.
- Measure conversational quality with a versioned JSON benchmark and CI gates.

## Planned implementation order

1. **Conversation storage** — completed
   - Add SQLite conversation and message records with stable identifiers.
   - Store roles, timestamps, selected sources, rewritten retrieval queries,
     answers, and citations.
   - Keep existing Version 2 question history readable and unchanged.

2. **Conversation interface** — completed
   - Add create, rename, select, search, and delete conversation controls.
   - Render messages as an accessible chat timeline in Light and Dark modes.
   - Keep document selection explicit for every conversation.

3. **Grounded follow-up questions** — completed
   - Convert context-dependent follow-ups into standalone retrieval queries.
   - Retrieve evidence using the final Version 3 reranked pipeline.
   - Prevent instructions in chat history or documents from overriding the
     application grounding rules.

4. **Streaming generation** — completed
   - Stream Ollama output into the current assistant message.
   - Expose retrieval, generation, completion, cancellation, and failure states.
   - Persist only complete, validated answers.

5. **Claim-level citations and confidence** — completed
   - Associate answer claims with retrieved evidence identifiers.
   - Open the exact source card when a citation is selected.
   - Show calibrated supported, uncertain, or unsupported states without
     presenting a raw similarity score as probability.

6. **Optional learned reranker experiment** — completed; deterministic retained
   - Benchmark a small local cross-encoder against the Version 3 deterministic
     reranker.
   - Adopt it only if quality improvement justifies model size and latency.

7. **Evaluation and CI** — completed
   - Add versioned JSON cases for follow-ups, ambiguous references,
     unsupported questions, citation correctness, and conversation isolation.
   - Add automated regression thresholds and GitHub Actions checks.

## Completion criteria

- Conversations remain isolated from one another.
- Follow-up retrieval uses the correct standalone meaning and selected sources.
- Every displayed citation resolves to retrieved evidence.
- Unsupported questions produce a safe, tested response.
- Interrupted or failed generation does not create a valid completed answer.
- All existing Version 1–3 tests continue to pass.
- The Version 4 conversational benchmark passes its committed quality gates.

All completion criteria are enforced by the committed conversational benchmark,
the full Pytest suite, and GitHub Actions.

## Out of scope

- CSV ingestion.
- Mandatory cloud APIs or hosted model providers.
- Multi-user authentication and organization accounts.
- Replacing ChromaDB or SQLite without measured evidence that it is necessary.
