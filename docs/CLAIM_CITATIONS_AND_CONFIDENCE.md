# Claim-level Citations and Confidence

Version 4 associates evidence with individual factual claims instead of only
listing sources for an entire answer.

## Structured answer contract

Ollama returns three fields:

- `answer`: the readable draft used during streaming;
- `source_numbers`: every evidence block used anywhere in the answer;
- `claims`: atomic claim objects containing plain `text` and only the
  `source_numbers` that directly support that claim.

Python validates every source number, removes any citation labels inserted by
the model, and produces one line per claim with application-controlled labels.
If a claim lacks a usable source number, the existing lexical grounding fallback
must find overlapping evidence or generation fails. If cited evidence has no
content overlap with a claim, the application replaces the answer with its safe
unsupported refusal.

## Evidence states

The confidence display is deliberately categorical:

- **Supported** — every claim has valid evidence, at least the required number
  of overlapping content terms, and at least 50% content-term coverage.
- **Uncertain** — every claim has some evidence overlap, but at least one claim
  has limited overlap or coverage.
- **Unsupported** — the answer is a refusal, lacks claim citations, or contains
  a claim with no evidence overlap.

These labels describe evidence coverage, not whether the language model is
mathematically likely to be correct. ChromaDB similarity and reranker scores are
not probabilities and are never displayed as confidence percentages.

## Interface

The completed chat message displays its evidence state. Each claim is followed
by one or more controls labelled with the claim number, source number, and
source name. Selecting a control opens the exact persisted passage, location,
and original webpage link when applicable.

Unsupported questions show the safe refusal and an Unsupported state without a
misleading source card.

## Persistence and migration

Conversation schema version 2 adds:

- `confidence` on assistant messages;
- `claim_number` and `claim_text` on citations;
- uniqueness by `(message_id, claim_number, source_number)`, allowing the same
  source passage to support multiple different claims.

Existing schema-version-1 citations migrate as legacy whole-answer citations.
They remain readable with their previous source-card presentation.

## Tests

Automated coverage includes claim parsing, ordered sources, repeated sources
across claims, supported/uncertain/unsupported classification, safe refusal for
unrelated evidence, atomic SQLite persistence, and schema-version-1 migration.
The complete project suite and retrieval quality benchmark remain CI gates.
