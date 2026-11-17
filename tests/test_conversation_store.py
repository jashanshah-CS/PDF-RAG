from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from uuid import UUID

import pytest

import src.rag_project.conversation_store as conversation_store_module
from src.rag_project.conversation_store import ConversationStore, MessageCitation


START = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def citation(number: int = 1, source: str = "policy.pdf") -> MessageCitation:
    return MessageCitation(
        source_name=source,
        source_type="pdf",
        location="Page 4",
        page_number=4,
        url=None,
        chunk_number=2,
        source_number=number,
        excerpt="Annual leave is 29 days.",
    )


def test_conversation_survives_reopening_with_ordered_unique_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversations.db"
    created = ConversationStore(path).create(
        "Leave policy",
        selected_sources=("pdf:benefits", "pdf:policy", "pdf:benefits", " "),
        created_at=START,
    )

    reopened = ConversationStore(path).get(created.conversation_id)

    assert reopened == created
    assert reopened is not None
    assert reopened.selected_sources == ("pdf:benefits", "pdf:policy")


def test_rename_and_source_update_are_persisted(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    created = store.create("Original", created_at=START)

    renamed = store.rename(
        created.conversation_id,
        "Updated title",
        updated_at=START + timedelta(minutes=1),
    )
    updated = store.set_selected_sources(
        created.conversation_id,
        ("pdf:a", "web:b"),
        updated_at=START + timedelta(minutes=2),
    )

    assert renamed.title == "Updated title"
    assert updated.selected_sources == ("pdf:a", "web:b")
    assert updated.updated_at == START + timedelta(minutes=2)


def test_messages_and_citations_survive_restart_in_chronological_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversations.db"
    store = ConversationStore(path)
    conversation = store.create("Benefits", created_at=START)
    store.add_message(
        conversation.conversation_id,
        "assistant",
        "Employees receive 29 days. [Source 1]",
        citations=(citation(),),
        created_at=START + timedelta(seconds=2),
    )
    store.add_message(
        conversation.conversation_id,
        "user",
        "What is the leave allowance?",
        standalone_query="What is the annual leave allowance?",
        created_at=START + timedelta(seconds=1),
    )

    messages = ConversationStore(path).list_messages(conversation.conversation_id)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].original_question == "What is the leave allowance?"
    assert messages[0].standalone_query == "What is the annual leave allowance?"
    assert messages[1].citations == (citation(),)


def test_search_matches_titles_and_message_content(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    leave = store.create("Benefits", created_at=START)
    office = store.create("Office rules", created_at=START + timedelta(minutes=1))
    store.add_message(leave.conversation_id, "user", "How much annual leave?")

    assert [item.conversation_id for item in store.list("annual")] == [
        leave.conversation_id
    ]
    assert [item.conversation_id for item in store.list("office")] == [
        office.conversation_id
    ]


def test_conversations_are_isolated_and_delete_cascades(tmp_path: Path) -> None:
    path = tmp_path / "conversations.db"
    store = ConversationStore(path)
    first = store.create("First")
    second = store.create("Second")
    store.add_message(first.conversation_id, "assistant", "First answer", citations=(citation(),))
    store.add_message(second.conversation_id, "user", "Second question")

    assert [message.content for message in store.list_messages(second.conversation_id)] == [
        "Second question"
    ]
    assert store.delete(first.conversation_id) is True
    assert store.delete(first.conversation_id) is False
    assert store.get(first.conversation_id) is None

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM message_citations").fetchone()[0] == 0
    assert len(store.list_messages(second.conversation_id)) == 1


def test_failed_and_cancelled_message_statuses_are_persisted(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Statuses")

    store.add_message(conversation.conversation_id, "assistant", "Generation failed", status="failed")
    store.add_message(conversation.conversation_id, "assistant", "Stopped", status="cancelled")

    assert [message.status for message in store.list_messages(conversation.conversation_id)] == [
        "failed",
        "cancelled",
    ]


def test_pending_assistant_message_can_be_atomically_completed(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Streaming")
    pending = store.add_message(
        conversation.conversation_id,
        "assistant",
        "Preparing",
        status="pending",
    )

    completed = store.update_assistant_message(
        pending.message_id,
        "Employees receive 29 days. [Source 1]",
        "completed",
        citations=(citation(),),
        confidence="supported",
    )

    assert completed.status == "completed"
    assert completed.citations == (citation(),)
    assert completed.confidence == "supported"
    assert store.list_messages(conversation.conversation_id)[0] == completed


def test_recover_pending_messages_does_not_keep_partial_answer(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Recovery")
    store.add_message(conversation.conversation_id, "assistant", "Partial secret", status="pending")
    store.add_message(conversation.conversation_id, "assistant", "Complete", status="completed")

    assert store.recover_pending_messages() == 1
    messages = store.list_messages(conversation.conversation_id)
    assert messages[0].status == "failed"
    assert messages[0].content == "Generation was interrupted before completion."
    assert messages[1].content == "Complete"


def test_finalization_rejects_user_pending_and_failed_citations(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Validation")
    user = store.add_message(conversation.conversation_id, "user", "Question")
    pending = store.add_message(conversation.conversation_id, "assistant", "Preparing", status="pending")

    with pytest.raises(ValueError, match="assistant"):
        store.update_assistant_message(user.message_id, "No", "failed")
    with pytest.raises(ValueError, match="completed"):
        store.update_assistant_message(pending.message_id, "No", "failed", citations=(citation(),))
    with pytest.raises(KeyError, match="Unknown message"):
        store.update_assistant_message("missing", "No", "failed")


@pytest.mark.parametrize("role", ["system", "tool", ""])
def test_rejects_invalid_message_roles(tmp_path: Path, role: str) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Validation")

    with pytest.raises(ValueError, match="Message role"):
        store.add_message(conversation.conversation_id, role, "Content")


def test_rejects_invalid_content_status_citations_and_limits(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Validation")

    with pytest.raises(ValueError, match="Message content"):
        store.add_message(conversation.conversation_id, "user", " ")
    with pytest.raises(ValueError, match="Message status"):
        store.add_message(conversation.conversation_id, "assistant", "Answer", status="unknown")
    with pytest.raises(ValueError, match="Only assistant"):
        store.add_message(conversation.conversation_id, "user", "Question", citations=(citation(),))
    with pytest.raises(ValueError, match="completed"):
        store.add_message(
            conversation.conversation_id,
            "assistant",
            "Preparing",
            status="pending",
            confidence="supported",
        )
    with pytest.raises(ValueError, match="unique"):
        store.add_message(
            conversation.conversation_id,
            "assistant",
            "Answer",
            citations=(citation(), citation(source="other.pdf")),
        )
    with pytest.raises(ValueError, match="limit"):
        store.list(limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.list_messages(conversation.conversation_id, limit=0)


def test_rejects_naive_timestamps_and_unknown_conversations(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")

    with pytest.raises(ValueError, match="timezone"):
        store.create("Naive", created_at=datetime(2026, 11, 17))
    with pytest.raises(KeyError, match="Unknown conversation"):
        store.add_message("missing", "user", "Question")
    with pytest.raises(KeyError, match="Unknown conversation"):
        store.list_messages("missing")
    with pytest.raises(KeyError, match="Unknown conversation"):
        store.rename("missing", "Title")


def test_message_and_citations_roll_back_together_on_database_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Atomic")
    fixed_uuid = UUID("00000000-0000-0000-0000-000000000001")
    monkeypatch.setattr(conversation_store_module, "uuid4", lambda: fixed_uuid)

    with pytest.raises(ValueError, match="could not be saved"):
        store.add_message(
            conversation.conversation_id,
            "assistant",
            "Answer",
            citations=(citation(1), citation(2, "other.pdf")),
        )

    assert store.list_messages(conversation.conversation_id) == []


def test_updated_at_never_moves_backwards(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "conversations.db")
    conversation = store.create("Monotonic", created_at=START)
    later = START + timedelta(hours=1)
    store.add_message(conversation.conversation_id, "user", "Later", created_at=later)
    store.add_message(conversation.conversation_id, "assistant", "Earlier", created_at=START)

    assert store.get(conversation.conversation_id).updated_at == later


def test_rejects_database_from_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="newer schema version"):
        ConversationStore(path)


def test_version_one_database_migrates_claims_and_confidence(tmp_path: Path) -> None:
    path = tmp_path / "version-one.db"
    with sqlite3.connect(path) as connection:
        ConversationStore._migrate_to_version_1(connection)
        connection.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?)",
            ("conversation", "Legacy", START.isoformat(), START.isoformat()),
        )
        connection.execute(
            "INSERT INTO conversation_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "message", "conversation", "assistant", "Legacy [Source 1]",
                "completed", None, None, START.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO message_citations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "citation", "message", "legacy.pdf", "pdf", "page 1", 1,
                None, 1, 1, "Legacy evidence",
            ),
        )

    migrated = ConversationStore(path).list_messages("conversation")[0]

    assert migrated.confidence is None
    assert migrated.citations[0].claim_number == 1
    assert migrated.citations[0].claim_text == ""
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
