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
        selected_sources=("PDF · benefits.pdf",),
    )

    entries = QuestionHistoryStore(path).list()

    assert len(entries) == 1
    assert entries[0].question == "How much leave?"
    assert entries[0].citations[0].source_name == "benefits.pdf"
    assert entries[0].citations[0].page_number == 4
    assert entries[0].selected_sources == ("PDF · benefits.pdf",)


def test_history_search_export_and_clear(tmp_path) -> None:
    store = QuestionHistoryStore(tmp_path / "history.db")
    store.add("Leave question", "29 days [Source 1]", [result("benefits.pdf", 4)])
    store.add("Office question", "London [Source 1]", [result("office.pdf", 1)])

    matches = store.list("leave")
    exported = store.export_json()

    assert [entry.question for entry in matches] == ["Leave question"]
    assert '"source_name": "benefits.pdf"' in exported
    assert '"selected_sources"' in exported
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


def test_existing_history_database_is_migrated_without_losing_rows(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "old-history.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE question_history (
                entry_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO question_history VALUES (?, ?, ?, ?, ?)",
            ("old", "Old question", "Old answer", "[]", "2026-10-07T12:00:00+00:00"),
        )

    entries = QuestionHistoryStore(path).list()

    assert len(entries) == 1
    assert entries[0].question == "Old question"
    assert entries[0].selected_sources == ()
