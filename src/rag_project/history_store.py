"""Readable local SQLite history for questions and grounded answers."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from src.rag_project.search import SearchResult


DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "history.db"


@dataclass(frozen=True)
class HistoryCitation:
    """One source citation saved with a generated answer."""

    source_name: str
    source_type: str
    location: str
    page_number: int | None
    url: str | None
    chunk_number: int


@dataclass(frozen=True)
class HistoryEntry:
    """One persisted question-and-answer exchange."""

    entry_id: str
    question: str
    answer: str
    citations: tuple[HistoryCitation, ...]
    created_at: datetime
    selected_sources: tuple[str, ...] = ()


def _answer_citations(
    answer: str, results: list[SearchResult]
) -> tuple[HistoryCitation, ...]:
    numbers = sorted(
        {int(value) for value in re.findall(r"\[Source\s+(\d+)\]", answer, re.I)}
    )
    citations: list[HistoryCitation] = []
    for number in numbers:
        if not 1 <= number <= len(results):
            continue
        chunk = results[number - 1].embedded_chunk.chunk
        citations.append(
            HistoryCitation(
                source_name=chunk.source_name,
                source_type=chunk.source_type.value,
                location=chunk.location.label(),
                page_number=chunk.location.page_number,
                url=chunk.location.url,
                chunk_number=chunk.chunk_number,
            )
        )
    return tuple(citations)


class QuestionHistoryStore:
    """Persist and query answer history using Python's built-in SQLite."""

    def __init__(self, path: Path = DEFAULT_HISTORY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS question_history (
                    entry_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    selected_sources_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(question_history)"
                ).fetchall()
            }
            if "selected_sources_json" not in columns:
                connection.execute(
                    "ALTER TABLE question_history ADD COLUMN "
                    "selected_sources_json TEXT NOT NULL DEFAULT '[]'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_created_at "
                "ON question_history(created_at DESC)"
            )

    def add(
        self,
        question: str,
        answer: str,
        results: list[SearchResult],
        *,
        created_at: datetime | None = None,
        selected_sources: tuple[str, ...] = (),
    ) -> HistoryEntry:
        """Save one successful answer and the sources it actually cited."""
        if not question.strip() or not answer.strip():
            raise ValueError("A question and answer are required.")
        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("The history timestamp must include a timezone.")
        entry = HistoryEntry(
            entry_id=str(uuid4()),
            question=question.strip(),
            answer=answer.strip(),
            citations=_answer_citations(answer, results),
            created_at=timestamp,
            selected_sources=tuple(
                dict.fromkeys(source.strip() for source in selected_sources if source.strip())
            ),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO question_history (
                    entry_id, question, answer, citations_json, created_at,
                    selected_sources_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.question,
                    entry.answer,
                    json.dumps([asdict(citation) for citation in entry.citations]),
                    entry.created_at.isoformat(),
                    json.dumps(entry.selected_sources),
                ),
            )
        return entry

    def list(self, search: str = "", limit: int = 100) -> list[HistoryEntry]:
        """Return newest entries first, optionally filtered by question or answer."""
        if limit <= 0:
            raise ValueError("History limit must be greater than zero.")
        query = "SELECT * FROM question_history"
        parameters: list[object] = []
        if search.strip():
            query += " WHERE question LIKE ? OR answer LIKE ?"
            pattern = f"%{search.strip()}%"
            parameters.extend([pattern, pattern])
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def clear(self) -> None:
        """Delete all saved question history without touching source indexes."""
        with self._connect() as connection:
            connection.execute("DELETE FROM question_history")

    def export_json(self) -> str:
        """Return all history as portable, readable JSON."""
        payload = []
        for entry in self.list(limit=100_000):
            payload.append(
                {
                    "entry_id": entry.entry_id,
                    "question": entry.question,
                    "answer": entry.answer,
                    "citations": [asdict(citation) for citation in entry.citations],
                    "selected_sources": list(entry.selected_sources),
                    "created_at": entry.created_at.isoformat(),
                }
            )
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> HistoryEntry:
        citations = tuple(
            HistoryCitation(**citation)
            for citation in json.loads(str(row["citations_json"]))
        )
        return HistoryEntry(
            entry_id=str(row["entry_id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            citations=citations,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            selected_sources=tuple(
                json.loads(str(row["selected_sources_json"] or "[]"))
            ),
        )
