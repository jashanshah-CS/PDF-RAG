# Follow-up Query Rewriting

Version 4 can resolve context-dependent questions before retrieval. For
example, after asking about a remote-work policy, “What about contractors?” can
become a self-contained question that names the policy and contractors.

## Processing flow

1. Load completed user and assistant messages from the active conversation.
2. Keep at most the latest six messages and 4,000 characters.
3. Skip rewriting when the new question is already self-contained.
4. For a contextual follow-up, ask the configured local Ollama model for one
   structured `standalone_query` at temperature zero.
5. Search with the standalone query through the final Version 3 hybrid,
   reranked retrieval pipeline.
6. Generate the answer from that query and the retrieved evidence.
7. Save both the user's original wording and the standalone retrieval query.

When the two forms differ, the chat timeline displays `Searched as:` beneath
the original message. This makes the retrieval interpretation inspectable
without replacing what the user typed.

## Safety and failure handling

- Only history belonging to the active conversation is used.
- Failed and cancelled messages are excluded.
- History and the current question are explicitly treated as untrusted text.
- The rewriting prompt says not to answer, follow embedded instructions, or
  introduce entities and facts absent from the conversation.
- Output must match a one-field JSON schema and stay below 500 characters.
- Invalid JSON, empty or oversized output, and Ollama errors all fall back to
  the original question.
- Standalone questions do not make an additional Ollama request.

These controls reduce prompt-injection and cross-conversation leakage risk,
but the rewritten query is still model-produced text. Keeping it visible helps
users notice and correct a mistaken interpretation.

## Tests

`tests/test_query_rewriter.py` covers standalone questions, contextual
detection, structured request settings, history bounds, conversation
isolation, invalid output, and Ollama failure fallback. Existing retrieval and
conversation tests ensure the new step does not replace source selection,
reranking, citations, or durable message storage.
