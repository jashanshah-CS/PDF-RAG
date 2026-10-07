"""Source-neutral document models shared by every ingestion pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any


class SourceType(StrEnum):
    """The source formats supported by the RAG application."""

    PDF = "pdf"
    WEBSITE = "website"


@dataclass(frozen=True)
class SourceLocation:
    """A position within a source that can be shown in a citation."""

    page_number: int | None = None
    row_number: int | None = None
    url: str | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number <= 0:
            raise ValueError("Page numbers must be greater than zero.")
        if self.row_number is not None and self.row_number <= 0:
            raise ValueError("Row numbers must be greater than zero.")

    def label(self) -> str:
        """Return a concise, human-readable source position."""
        parts: list[str] = []
        if self.page_number is not None:
            parts.append(f"page {self.page_number}")
        if self.row_number is not None:
            parts.append(f"row {self.row_number}")
        if self.section:
            parts.append(self.section)
        if self.url and not parts:
            parts.append(self.url)
        return " · ".join(parts) if parts else "source"


@dataclass(frozen=True)
class Document:
    """Extracted text and citation metadata from one source location."""

    document_id: str
    source_type: SourceType
    source_name: str
    text: str
    location: SourceLocation
    added_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("A document ID is required.")
        if not isinstance(self.source_type, SourceType):
            raise ValueError("A supported source type is required.")
        if not self.source_name.strip():
            raise ValueError("A source name is required.")
        if not self.text.strip():
            raise ValueError("Document text cannot be empty.")
        if not isinstance(self.location, SourceLocation):
            raise ValueError("A source location is required.")
        if self.added_at.tzinfo is None or self.added_at.utcoffset() is None:
            raise ValueError("The date added must include a timezone.")


@dataclass(frozen=True)
class DocumentChunk:
    """One searchable passage that retains its parent source metadata."""

    document_id: str
    source_type: SourceType
    source_name: str
    chunk_number: int
    text: str
    location: SourceLocation
    added_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("A document ID is required.")
        if not isinstance(self.source_type, SourceType):
            raise ValueError("A supported source type is required.")
        if not self.source_name.strip():
            raise ValueError("A source name is required.")
        if self.chunk_number <= 0:
            raise ValueError("Chunk numbers must be greater than zero.")
        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty.")
        if not isinstance(self.location, SourceLocation):
            raise ValueError("A source location is required.")
        if self.added_at.tzinfo is None or self.added_at.utcoffset() is None:
            raise ValueError("The date added must include a timezone.")

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        chunk_number: int,
        text: str,
    ) -> "DocumentChunk":
        """Create a chunk while preserving all parent source metadata."""
        return cls(
            document_id=document.document_id,
            source_type=document.source_type,
            source_name=document.source_name,
            chunk_number=chunk_number,
            text=text,
            location=document.location,
            added_at=document.added_at,
            metadata=dict(document.metadata),
        )

    def citation_label(self) -> str:
        """Return the source name and exact location for display."""
        return f"{self.source_name} — {self.location.label()}"


def create_document_id(
    source_type: SourceType,
    source_name: str,
    content: bytes,
) -> str:
    """Create a stable identifier from a source's type, name, and content."""
    if not source_name.strip():
        raise ValueError("A source name is required.")
    if not content:
        raise ValueError("Source content cannot be empty.")

    digest = sha256()
    digest.update(source_type.value.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_name.strip().encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    return f"{source_type.value}-{digest.hexdigest()}"
