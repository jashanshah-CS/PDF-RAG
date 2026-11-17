# Version 4 conversation interface

The Version 4 interface connects Streamlit to the persistent
`ConversationStore`. It replaces the one-off answer panel with a durable chat
timeline while keeping the Version 2 question-history view available as a
separate legacy record.

## Sidebar controls

The sidebar now provides:

- creation of a new conversation;
- search across conversation titles and message text;
- selection of the active conversation;
- renaming with stable conversation identity; and
- confirmed deletion of the active conversation.

Conversation IDs, rather than titles, are used as identity. Two conversations
may therefore share the title `New conversation` without selection or rename
operations affecting the wrong record. Deleting the final conversation creates
a fresh empty replacement so the application always has an active workspace.

## Conversation-specific sources

Each conversation owns an ordered set of ChromaDB source keys. Switching
conversations restores that conversation's source selection, removes sources
that no longer exist, and automatically includes newly uploaded or refreshed
sources in the active conversation. `Use all`, `Clear`, and multiselect changes
are persisted immediately.

Source scope and citations remain distinct:

- source scope defines which documents retrieval may search; and
- citations identify which retrieved passages support an assistant message.

Changing a conversation's scope does not delete documents from ChromaDB.

## Chat timeline

Messages render chronologically with Streamlit's user and assistant chat
components. Completed assistant messages show their saved evidence in expandable
cards, including source name, page or URL, chunk excerpt, and original webpage
link when present. Failed and cancelled messages have visible status labels.

When a question is submitted, the application:

1. saves the user message;
2. records the original question as the temporary standalone query;
3. runs the Version 3 reranked retrieval pipeline against the conversation's
   selected sources;
4. generates a grounded local Ollama answer;
5. saves the assistant message and only the citations used by that answer; and
6. keeps writing the successful exchange to the existing question history for
   backward compatibility.

The first question automatically gives an untouched `New conversation` a
compact title. True context-aware query rewriting is deliberately deferred to
the next Version 4 step; until then, the standalone query equals the original
question.

## Failure behaviour

- A question that cannot be persisted is not sent to retrieval.
- Retrieval and Ollama failures create a visible assistant message with status
  `failed`.
- If the new conversation write fails after generation, the answer remains
  visible for the current run and the error is shown.
- The existing question-history write is best-effort and cannot erase a
  successfully persisted conversation message.

## Testing

Deterministic helper tests cover active-conversation fallback, duplicate-title
safe identity, automatic title generation, source-scope switching, new-upload
selection, and conversion of answer labels into durable citations. Storage tests
continue to cover persistence, isolation, transactions, cascade deletion, and
validation.

The browser smoke test uses isolated temporary ChromaDB and SQLite paths. It
verifies the rendered Version 4 header, conversation selector, create button,
management panel, empty timeline, rename persistence, and matching dropdown and
timeline titles without modifying the user's real document library.
