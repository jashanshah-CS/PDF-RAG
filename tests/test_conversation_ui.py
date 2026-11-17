from datetime import UTC, datetime

from src.rag_project.conversation_store import Conversation
from src.rag_project.conversation_ui import (
    choose_active_conversation,
    conversation_title,
    message_citations,
    reconcile_conversation_sources,
)
from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.search import SearchResult


NOW = datetime(2026, 11, 17, tzinfo=UTC)


def conversation(identifier: str) -> Conversation:
    return Conversation(identifier, identifier, (), NOW, NOW)


def result(source: str, page: int) -> SearchResult:
    chunk = DocumentChunk(
        document_id=f"pdf-{source}",
        source_type=SourceType.PDF,
        source_name=source,
        chunk_number=3,
        text=f"Evidence from {source}.",
        location=SourceLocation(page_number=page),
        added_at=NOW,
    )
    return SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), 0.9)


def test_active_conversation_keeps_valid_request_or_uses_most_recent() -> None:
    conversations = [conversation("newest"), conversation("older")]

    assert choose_active_conversation(conversations, "older") == "older"
    assert choose_active_conversation(conversations, "missing") == "newest"
    assert choose_active_conversation([], "missing") is None


def test_title_is_clean_and_truncated_at_word_safe_boundary() -> None:
    assert conversation_title("  What   is the leave policy? ") == "What is the leave policy?"
    assert conversation_title(" ") == "New conversation"
    assert conversation_title("abcdefghij", maximum_length=8) == "abcdefg…"


def test_title_rejects_unusable_limit() -> None:
    try:
        conversation_title("Question", maximum_length=3)
    except ValueError as error:
        assert "at least four" in str(error)
    else:
        raise AssertionError("A title limit below four should be rejected")


def test_switched_conversation_loads_saved_source_scope() -> None:
    selected = reconcile_conversation_sources(
        conversation_id="conversation-b",
        loaded_conversation_id="conversation-a",
        saved_keys=("pdf:b", "pdf:removed"),
        session_keys=["pdf:a"],
        available_keys=["pdf:a", "pdf:b"],
    )

    assert selected == ["pdf:b"]


def test_current_conversation_keeps_session_scope_and_adds_new_uploads() -> None:
    selected = reconcile_conversation_sources(
        conversation_id="conversation-a",
        loaded_conversation_id="conversation-a",
        saved_keys=("pdf:old",),
        session_keys=["pdf:a"],
        available_keys=["pdf:a", "pdf:new"],
        newly_added_keys=["pdf:new"],
    )

    assert selected == ["pdf:a", "pdf:new"]


def test_answer_source_labels_become_ordered_durable_citations() -> None:
    citations = message_citations(
        "The answer uses the second source first. [Source 2] [Source 1]",
        [result("first.pdf", 1), result("second.pdf", 4)],
    )

    assert [item.source_number for item in citations] == [2, 1]
    assert citations[0].source_name == "second.pdf"
    assert citations[0].page_number == 4
    assert citations[0].excerpt == "Evidence from second.pdf."


def test_same_source_can_support_two_distinct_claims() -> None:
    citations = message_citations(
        "First fact from policy. [Source 1]\nSecond fact from policy. [Source 1]",
        [result("policy.pdf", 3)],
    )

    assert [(item.claim_number, item.source_number) for item in citations] == [
        (1, 1),
        (2, 1),
    ]
    assert citations[0].claim_text == "First fact from policy."
    assert citations[1].claim_text == "Second fact from policy."
