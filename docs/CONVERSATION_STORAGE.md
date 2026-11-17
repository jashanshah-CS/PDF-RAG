# Version 4 conversation storage

Version 4 stores multi-turn conversations in local SQLite at
`data/conversations.db`. This is separate from the existing
`data/history.db`, so introducing chat sessions does not migrate, reinterpret,
or delete Version 2 question history.

## Schema

The schema is normalized into four tables:

| Table | Purpose |
| --- | --- |
| `conversations` | Stable ID, title, creation time, and latest activity time |
| `conversation_sources` | Ordered ChromaDB source keys allowed for retrieval |
| `conversation_messages` | User and assistant content, status, original question, standalone query, and timestamp |
| `message_citations` | Ordered source, location, chunk, URL, and evidence attached to an assistant message |

Foreign keys use `ON DELETE CASCADE`. Deleting a conversation therefore removes
only that conversation's selected-source links, messages, and citations; it
does not remove documents from ChromaDB or entries from the old question
history.

SQLite `PRAGMA user_version` records the schema version. The store creates
version 1 transactionally and rejects databases created by a newer application
version rather than guessing how to read them.

## Stored message states

Messages support four explicit states:

- `pending`: generation has started but is not complete;
- `completed`: content is ready to display as a valid message;
- `failed`: generation or validation failed;
- `cancelled`: the user stopped generation.

Only assistant messages may contain citations. Every citation has a unique
source number within its message, and page, chunk, and source numbers are
validated before insertion.

## Atomicity and isolation

`ConversationStore.add_message()` inserts the message and every citation in one
SQLite transaction. If any citation fails a database constraint, the message
and all earlier citation inserts roll back together. Tests deliberately force
this failure path.

Every message and selected source is linked to one conversation ID. Listing
messages requires that ID, and deleting one conversation cannot affect another.
Conversation activity timestamps never move backwards when older imported
messages are added.

## Configuration

The default path follows `RAG_DATA_DIR`:

```text
RAG_DATA_DIR=data
CONVERSATION_PATH=data/conversations.db
```

`CONVERSATION_PATH` is optional. Docker Compose already persists `/app/data`,
so the default conversation database is included in the existing `rag-data`
volume without another service or volume.

## Current boundary

This step provides the tested persistence API only. The Streamlit conversation
sidebar and chat timeline will consume it in the next Version 4 step. Existing
single-question history remains the active user interface until that integration
is implemented and tested.

## Verification

Focused tests cover:

- database reopening and persistence;
- ordered, deduplicated source selection;
- rename and source-scope updates;
- chronological messages and citation reconstruction;
- title and message-content search;
- conversation isolation and cascading deletion;
- failed and cancelled states;
- invalid roles, statuses, timestamps, limits, and citations;
- atomic rollback after a forced SQLite constraint failure;
- monotonic activity timestamps; and
- rejection of unsupported future schema versions.

Run them with:

```powershell
uv run pytest tests/test_conversation_store.py tests/test_config.py -q
```
