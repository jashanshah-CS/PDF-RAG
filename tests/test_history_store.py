from datetime import UTC, datetime

from src.rag_project.documents import DocumentChunk, SourceLocation, SourceType
from src.rag_project.embeddings import EmbeddedChunk
from src.rag_project.history_store import QuestionHistoryStore
from src.rag_project.search import SearchResult


def result(name: str, page: int) -> SearchResult:
    chunk = DocumentChunk(
        document_id=f"document-{name}",
        source_type=SourceType.PDF,
        source_name=name,
        chunk_number=1,
        text="Annual leave is 29 days.",
        location=SourceLocation(page_number=page),
        added_at=datetime(2026, 10, 7, tzinfo=UTC),
    )
    return SearchResult(EmbeddedChunk(chunk, (1.0, 0.0)), 0.9)


def test_history_survives_reopening_and_saves_used_citations(tmp_path) -> None:
    path = tmp_path / "history.db"
    store = QuestionHistoryStore(path)
    store.add(
        "How much leave?",
        "Employees receive 29 days. [Source 2]",
        [result("other.pdf", 1), result("benefits.pdf", 4)],
        created_at=datetime(2026, 10, 7, 12, 0, tzinfo=UTC),
    )

    entries = QuestionHistoryStore(path).list()

    assert len(entries) == 1
    assert entries[0].question == "How much leave?"
    assert entries[0].citations[0].source_name == "benefits.pdf"
    assert entries[0].citations[0].page_number == 4


def test_history_search_export_and_clear(tmp_path) -> None:
    store = QuestionHistoryStore(tmp_path / "history.db")
    store.add("Leave question", "29 days [Source 1]", [result("benefits.pdf", 4)])
    store.add("Office question", "London [Source 1]", [result("office.pdf", 1)])

    matches = store.list("leave")
    exported = store.export_json()

    assert [entry.question for entry in matches] == ["Leave question"]
    assert '"source_name": "benefits.pdf"' in exported
    store.clear()
    assert store.list() == []


def test_history_rejects_naive_timestamp(tmp_path) -> None:
    store = QuestionHistoryStore(tmp_path / "history.db")

    try:
        store.add(
            "Question",
            "Answer",
            [],
            created_at=datetime(2026, 10, 7),
        )
    except ValueError as error:
        assert "timezone" in str(error)
    else:
        raise AssertionError("A naive timestamp should be rejected")
