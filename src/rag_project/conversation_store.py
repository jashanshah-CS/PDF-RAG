"""Persistent multi-turn conversations backed by local SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from uuid import uuid4

from src.rag_project.config import SETTINGS


CURRENT_SCHEMA_VERSION = 2
DEFAULT_CONVERSATION_PATH = SETTINGS.conversation_path
VALID_ROLES = frozenset({"user", "assistant"})
VALID_STATUSES = frozenset({"pending", "completed", "failed", "cancelled"})
VALID_CONFIDENCE = frozenset({"supported", "uncertain", "unsupported"})


@dataclass(frozen=True)
class Conversation:
    """One independently scoped document conversation."""

    conversation_id: str
    title: str
    selected_sources: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MessageCitation:
    """Evidence attached to one persisted assistant message."""

    source_name: str
    source_type: str
    location: str
    page_number: int | None
    url: str | None
    chunk_number: int
    source_number: int
    excerpt: str
    claim_number: int = 1
    claim_text: str = ""


@dataclass(frozen=True)
class ConversationMessage:
    """One user or assistant message and its retrieval metadata."""

    message_id: str
    conversation_id: str
    role: str
    content: str
    status: str
    original_question: str | None
    standalone_query: str | None
    citations: tuple[MessageCitation, ...]
    created_at: datetime
    confidence: str | None = None


def _timestamp(value: datetime | None, *, field: str) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"The {field} timestamp must include a timezone.")
    return timestamp.astimezone(UTC)


def _required_text(value: str, *, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} is required.")
    return cleaned


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _source_keys(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


class ConversationStore:
    """Create, query, and delete isolated local document conversations."""

    def __init__(self, path: Path = DEFAULT_CONVERSATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > CURRENT_SCHEMA_VERSION:
                raise RuntimeError(
                    "The conversation database was created by a newer "
                    f"schema version ({version})."
                )
            if version < 1:
                self._migrate_to_version_1(connection)
                version = 1
            if version < 2:
                self._migrate_to_version_2(connection)

    @staticmethod
    def _migrate_to_version_1(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE conversation_sources (
                conversation_id TEXT NOT NULL,
                position INTEGER NOT NULL CHECK (position >= 0),
                source_key TEXT NOT NULL,
                PRIMARY KEY (conversation_id, source_key),
                UNIQUE (conversation_id, position),
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE conversation_messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('pending', 'completed', 'failed', 'cancelled')
                ),
                original_question TEXT,
                standalone_query TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE message_citations (
                citation_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                location TEXT NOT NULL,
                page_number INTEGER,
                url TEXT,
                chunk_number INTEGER NOT NULL CHECK (chunk_number > 0),
                source_number INTEGER NOT NULL CHECK (source_number > 0),
                excerpt TEXT NOT NULL,
                UNIQUE (message_id, source_number),
                FOREIGN KEY (message_id) REFERENCES conversation_messages(message_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX idx_conversations_updated "
            "ON conversations(updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX idx_messages_conversation_created "
            "ON conversation_messages(conversation_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX idx_citations_message "
            "ON message_citations(message_id, source_number)"
        )
        connection.execute("PRAGMA user_version = 1")

    @staticmethod
    def _migrate_to_version_2(connection: sqlite3.Connection) -> None:
        """Add answer confidence and claim-specific citation relationships."""
        connection.execute(
            "ALTER TABLE conversation_messages ADD COLUMN confidence TEXT "
            "CHECK (confidence IS NULL OR confidence IN "
            "('supported', 'uncertain', 'unsupported'))"
        )
        connection.execute("DROP INDEX IF EXISTS idx_citations_message")
        connection.execute(
            "ALTER TABLE message_citations RENAME TO message_citations_v1"
        )
        connection.execute(
            """
            CREATE TABLE message_citations (
                citation_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                location TEXT NOT NULL,
                page_number INTEGER,
                url TEXT,
                chunk_number INTEGER NOT NULL CHECK (chunk_number > 0),
                source_number INTEGER NOT NULL CHECK (source_number > 0),
                excerpt TEXT NOT NULL,
                claim_number INTEGER NOT NULL CHECK (claim_number > 0),
                claim_text TEXT NOT NULL,
                UNIQUE (message_id, claim_number, source_number),
                FOREIGN KEY (message_id) REFERENCES conversation_messages(message_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO message_citations (
                citation_id, message_id, source_name, source_type, location,
                page_number, url, chunk_number, source_number, excerpt,
                claim_number, claim_text
            )
            SELECT citation_id, message_id, source_name, source_type, location,
                page_number, url, chunk_number, source_number, excerpt, 1, ''
            FROM message_citations_v1
            """
        )
        connection.execute("DROP TABLE message_citations_v1")
        connection.execute(
            "CREATE INDEX idx_citations_message "
            "ON message_citations(message_id, claim_number, source_number)"
        )
        connection.execute("PRAGMA user_version = 2")

    def create(
        self,
        title: str = "New conversation",
        *,
        selected_sources: tuple[str, ...] = (),
        created_at: datetime | None = None,
    ) -> Conversation:
        """Create a conversation and its ordered source scope atomically."""
        timestamp = _timestamp(created_at, field="conversation")
        conversation = Conversation(
            conversation_id=str(uuid4()),
            title=_required_text(title, field="Conversation title"),
            selected_sources=_source_keys(selected_sources),
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (
                    conversation.conversation_id,
                    conversation.title,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                ),
            )
            self._replace_sources(
                connection,
                conversation.conversation_id,
                conversation.selected_sources,
            )
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        """Return one conversation, or None when it does not exist."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return self._row_to_conversation(connection, row) if row else None

    def list(self, search: str = "", limit: int = 100) -> list[Conversation]:
        """Return recently active conversations, optionally searching messages."""
        if limit <= 0:
            raise ValueError("Conversation limit must be greater than zero.")
        query = "SELECT * FROM conversations"
        parameters: list[object] = []
        if search.strip():
            query += (
                " WHERE title LIKE ? OR EXISTS ("
                "SELECT 1 FROM conversation_messages "
                "WHERE conversation_messages.conversation_id = "
                "conversations.conversation_id AND content LIKE ?)"
            )
            pattern = f"%{search.strip()}%"
            parameters.extend((pattern, pattern))
        query += " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._row_to_conversation(connection, row) for row in rows]

    def rename(
        self,
        conversation_id: str,
        title: str,
        *,
        updated_at: datetime | None = None,
    ) -> Conversation:
        """Rename a conversation and mark it as recently updated."""
        timestamp = _timestamp(updated_at, field="conversation update")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title = ?, "
                "updated_at = MAX(updated_at, ?) "
                "WHERE conversation_id = ?",
                (
                    _required_text(title, field="Conversation title"),
                    timestamp.isoformat(),
                    conversation_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return self._row_to_conversation(connection, row)

    def set_selected_sources(
        self,
        conversation_id: str,
        selected_sources: tuple[str, ...],
        *,
        updated_at: datetime | None = None,
    ) -> Conversation:
        """Replace a conversation's ordered retrieval scope atomically."""
        timestamp = _timestamp(updated_at, field="conversation update")
        sources = _source_keys(selected_sources)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET updated_at = MAX(updated_at, ?) "
                "WHERE conversation_id = ?",
                (timestamp.isoformat(), conversation_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            self._replace_sources(connection, conversation_id, sources)
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return self._row_to_conversation(connection, row)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        status: str = "completed",
        original_question: str | None = None,
        standalone_query: str | None = None,
        citations: tuple[MessageCitation, ...] = (),
        confidence: str | None = None,
        created_at: datetime | None = None,
    ) -> ConversationMessage:
        """Save one message and all its citations in a single transaction."""
        if role not in VALID_ROLES:
            raise ValueError(f"Message role must be one of: {', '.join(sorted(VALID_ROLES))}.")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Message status must be one of: {', '.join(sorted(VALID_STATUSES))}."
            )
        if citations and role != "assistant":
            raise ValueError("Only assistant messages can contain citations.")
        citation_keys = {
            (citation.claim_number, citation.source_number) for citation in citations
        }
        if len(citation_keys) != len(citations):
            raise ValueError(
                "Citation source numbers must be unique within each claim."
            )
        if confidence is not None and confidence not in VALID_CONFIDENCE:
            raise ValueError(
                "Message confidence must be supported, uncertain, or unsupported."
            )
        if confidence is not None and role != "assistant":
            raise ValueError("Only assistant messages can have confidence.")
        if confidence is not None and status != "completed":
            raise ValueError("Only completed messages can have confidence.")
        for citation in citations:
            self._validate_citation(citation)

        timestamp = _timestamp(created_at, field="message")
        cleaned_content = _required_text(content, field="Message content")
        message = ConversationMessage(
            message_id=str(uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=cleaned_content,
            status=status,
            original_question=(
                _optional_text(original_question)
                if original_question is not None
                else (cleaned_content if role == "user" else None)
            ),
            standalone_query=_optional_text(standalone_query),
            citations=citations,
            created_at=timestamp,
            confidence=confidence,
        )
        try:
            with self._connect() as connection:
                if not self._exists(connection, conversation_id):
                    raise KeyError(f"Unknown conversation: {conversation_id}")
                connection.execute(
                    """
                    INSERT INTO conversation_messages (
                        message_id, conversation_id, role, content, status,
                        original_question, standalone_query, created_at, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.conversation_id,
                        message.role,
                        message.content,
                        message.status,
                        message.original_question,
                        message.standalone_query,
                        message.created_at.isoformat(),
                        message.confidence,
                    ),
                )
                for citation in citations:
                    connection.execute(
                        """
                        INSERT INTO message_citations (
                            citation_id, message_id, source_name, source_type,
                            location, page_number, url, chunk_number,
                            source_number, excerpt, claim_number, claim_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid4()),
                            message.message_id,
                            citation.source_name,
                            citation.source_type,
                            citation.location,
                            citation.page_number,
                            citation.url,
                            citation.chunk_number,
                            citation.source_number,
                            citation.excerpt,
                            citation.claim_number,
                            citation.claim_text,
                        ),
                    )
                connection.execute(
                    "UPDATE conversations SET updated_at = MAX(updated_at, ?) "
                    "WHERE conversation_id = ?",
                    (message.created_at.isoformat(), conversation_id),
                )
        except sqlite3.IntegrityError as exception:
            raise ValueError("The message and its citations could not be saved.") from exception
        return message

    def list_messages(
        self, conversation_id: str, *, limit: int = 1_000
    ) -> list[ConversationMessage]:
        """Return a conversation's messages in chronological order."""
        if limit <= 0:
            raise ValueError("Message limit must be greater than zero.")
        with self._connect() as connection:
            if not self._exists(connection, conversation_id):
                raise KeyError(f"Unknown conversation: {conversation_id}")
            rows = connection.execute(
                "SELECT * FROM conversation_messages "
                "WHERE conversation_id = ? "
                "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [self._row_to_message(connection, row) for row in rows]

    def update_assistant_message(
        self,
        message_id: str,
        content: str,
        status: str,
        *,
        citations: tuple[MessageCitation, ...] = (),
        confidence: str | None = None,
        updated_at: datetime | None = None,
    ) -> ConversationMessage:
        """Atomically finalize one assistant placeholder and its citations."""
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Message status must be one of: {', '.join(sorted(VALID_STATUSES))}."
            )
        if status == "pending":
            raise ValueError("A pending message cannot be finalized as pending.")
        if citations and status != "completed":
            raise ValueError("Only completed messages can contain citations.")
        if confidence is not None and confidence not in VALID_CONFIDENCE:
            raise ValueError(
                "Message confidence must be supported, uncertain, or unsupported."
            )
        if status == "completed" and confidence is None:
            confidence = "uncertain"
        if status != "completed" and confidence is not None:
            raise ValueError("Only completed messages can have confidence.")
        citation_keys = {
            (citation.claim_number, citation.source_number) for citation in citations
        }
        if len(citation_keys) != len(citations):
            raise ValueError(
                "Citation source numbers must be unique within each claim."
            )
        for citation in citations:
            self._validate_citation(citation)
        timestamp = _timestamp(updated_at, field="message update")
        cleaned_content = _required_text(content, field="Message content")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT role, status, conversation_id FROM conversation_messages "
                    "WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown message: {message_id}")
                if row["role"] != "assistant":
                    raise ValueError("Only assistant messages can be finalized.")
                if row["status"] != "pending":
                    raise ValueError("Only pending assistant messages can be finalized.")
                connection.execute(
                    "UPDATE conversation_messages SET content = ?, status = ?, "
                    "confidence = ? "
                    "WHERE message_id = ?",
                    (cleaned_content, status, confidence, message_id),
                )
                connection.execute(
                    "DELETE FROM message_citations WHERE message_id = ?",
                    (message_id,),
                )
                for citation in citations:
                    connection.execute(
                        """
                        INSERT INTO message_citations (
                            citation_id, message_id, source_name, source_type,
                            location, page_number, url, chunk_number,
                            source_number, excerpt, claim_number, claim_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid4()), message_id, citation.source_name,
                            citation.source_type, citation.location,
                            citation.page_number, citation.url,
                            citation.chunk_number, citation.source_number,
                            citation.excerpt,
                            citation.claim_number,
                            citation.claim_text,
                        ),
                    )
                connection.execute(
                    "UPDATE conversations SET updated_at = MAX(updated_at, ?) "
                    "WHERE conversation_id = ?",
                    (timestamp.isoformat(), row["conversation_id"]),
                )
                updated = connection.execute(
                    "SELECT * FROM conversation_messages WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                return self._row_to_message(connection, updated)
        except sqlite3.IntegrityError as exception:
            raise ValueError("The message and its citations could not be updated.") from exception

    def recover_pending_messages(self) -> int:
        """Mark placeholders left by an interrupted app process as failed."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_messages SET content = ?, status = 'failed' "
                "WHERE role = 'assistant' AND status = 'pending'",
                ("Generation was interrupted before completion.",),
            )
            return cursor.rowcount

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation, its messages, sources, and citations."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _exists(connection: sqlite3.Connection, conversation_id: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _replace_sources(
        connection: sqlite3.Connection,
        conversation_id: str,
        sources: tuple[str, ...],
    ) -> None:
        connection.execute(
            "DELETE FROM conversation_sources WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.executemany(
            "INSERT INTO conversation_sources VALUES (?, ?, ?)",
            (
                (conversation_id, position, source)
                for position, source in enumerate(sources)
            ),
        )

    @staticmethod
    def _validate_citation(citation: MessageCitation) -> None:
        _required_text(citation.source_name, field="Citation source name")
        _required_text(citation.source_type, field="Citation source type")
        _required_text(citation.location, field="Citation location")
        _required_text(citation.excerpt, field="Citation excerpt")
        if citation.page_number is not None and citation.page_number <= 0:
            raise ValueError("Citation page number must be greater than zero.")
        if citation.chunk_number <= 0 or citation.source_number <= 0:
            raise ValueError("Citation chunk and source numbers must be greater than zero.")
        if citation.claim_number <= 0:
            raise ValueError("Citation claim number must be greater than zero.")

    @staticmethod
    def _row_to_conversation(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> Conversation:
        source_rows = connection.execute(
            "SELECT source_key FROM conversation_sources "
            "WHERE conversation_id = ? ORDER BY position",
            (row["conversation_id"],),
        ).fetchall()
        return Conversation(
            conversation_id=str(row["conversation_id"]),
            title=str(row["title"]),
            selected_sources=tuple(str(item["source_key"]) for item in source_rows),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    @staticmethod
    def _row_to_message(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ConversationMessage:
        citation_rows = connection.execute(
            "SELECT * FROM message_citations WHERE message_id = ? "
            "ORDER BY source_number",
            (row["message_id"],),
        ).fetchall()
        citations = tuple(
            MessageCitation(
                source_name=str(item["source_name"]),
                source_type=str(item["source_type"]),
                location=str(item["location"]),
                page_number=item["page_number"],
                url=item["url"],
                chunk_number=int(item["chunk_number"]),
                source_number=int(item["source_number"]),
                excerpt=str(item["excerpt"]),
                claim_number=int(item["claim_number"]),
                claim_text=str(item["claim_text"]),
            )
            for item in citation_rows
        )
        return ConversationMessage(
            message_id=str(row["message_id"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            status=str(row["status"]),
            original_question=row["original_question"],
            standalone_query=row["standalone_query"],
            citations=citations,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            confidence=row["confidence"],
        )
