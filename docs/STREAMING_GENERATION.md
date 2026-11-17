# Streaming and Cancellable Generation

Version 4 streams local Ollama output while keeping conversation storage
transactional and trustworthy.

## Runtime flow

1. The app saves the user question and retrieves reranked evidence.
2. SQLite receives an assistant placeholder with `pending` status.
3. A daemon worker sends an Ollama chat request with `stream: true`.
4. Ollama's newline-delimited JSON events are decoded incrementally.
5. The answer field is extracted from the partial structured JSON and displayed
   by a Streamlit fragment that polls the thread-safe job manager.
6. The completed JSON object is parsed and validated with the same grounding,
   completeness, and citation checks as non-streaming generation.
7. SQLite atomically replaces the placeholder and its citations with one
   terminal state: `completed`, `cancelled`, or `failed`.

The user can switch conversations while a worker continues. A refreshed browser
session adopts the most recent in-memory job. If the whole app process exits,
pending placeholders are recovered as failed on the next startup instead of
being mistaken for completed answers.

## Cancellation

The **Stop generating** button immediately marks the visible job as cancelled
and sets a thread-safe cancellation event. The streaming HTTP reader checks
that event between Ollama events and closes the response by unwinding the
worker. The cancelled state is therefore persisted without waiting for another
token; the underlying local connection may take until the next model event to
finish unwinding.

Partial drafts live only in memory. On cancellation, they are discarded and the
database stores a short cancellation notice without citations. This prevents an
unfinished JSON fragment or unvalidated claim from appearing in conversation
history as a valid answer.

## Failure handling

- Network, timeout, malformed stream, and Ollama errors become `failed` messages.
- Empty or invalid final structured output is rejected.
- Citations are written only for `completed` messages.
- Updating the terminal message and replacing its citations happens in one
  SQLite transaction.
- A standalone question or follow-up rewrite completes before the streaming job
  begins; the job receives the final retrieval query and fixed evidence set.

## Tests

- `tests/test_generator.py` covers incremental structured JSON extraction,
  escaped text, final validation, and cancellation.
- `tests/test_generation_jobs.py` covers completed, cancelled, and failed worker
  states plus partial-draft removal.
- `tests/test_conversation_store.py` covers atomic placeholder finalization,
  citation restrictions, and interrupted-process recovery.
- The complete suite and Version 3 retrieval benchmark remain CI gates.
