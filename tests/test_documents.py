from datetime import UTC, datetime

import pytest

from src.rag_project.documents import (
    Document,
    DocumentChunk,
    SourceLocation,
    SourceType,
    create_document_id,
)


ADDED_AT = datetime(2026, 10, 5, tzinfo=UTC)


def test_document_id_is_stable_for_the_same_source() -> None:
    first = create_document_id(SourceType.PDF, "handbook.pdf", b"same content")
    second = create_document_id(SourceType.PDF, "handbook.pdf", b"same content")

    assert first == second
    assert first.startswith("pdf-")


def test_document_id_changes_with_source_content() -> None:
    first = create_document_id(SourceType.PDF, "handbook.pdf", b"first version")
    second = create_document_id(SourceType.PDF, "handbook.pdf", b"second version")

    assert first != second


def test_chunk_inherits_parent_document_metadata() -> None:
    document = Document(
        document_id="pdf-handbook",
        source_type=SourceType.PDF,
        source_name="handbook.pdf",
        text="Annual leave is 25 days.",
        location=SourceLocation(page_number=4),
        added_at=ADDED_AT,
        metadata={"media_type": "application/pdf"},
    )

    chunk = DocumentChunk.from_document(
        document,
        chunk_number=1,
        text=document.text,
    )

    assert chunk.document_id == document.document_id
    assert chunk.source_type == SourceType.PDF
    assert chunk.source_name == "handbook.pdf"
    assert chunk.location == SourceLocation(page_number=4)
    assert chunk.added_at == ADDED_AT
    assert chunk.metadata == {"media_type": "application/pdf"}
    assert chunk.citation_label() == "handbook.pdf — page 4"


@pytest.mark.parametrize(
    "location",
    [SourceLocation(page_number=4), SourceLocation(row_number=12)],
)
def test_location_labels_are_source_appropriate(location: SourceLocation) -> None:
    assert location.label() in {"page 4", "row 12"}


def test_document_requires_timezone_aware_date() -> None:
    with pytest.raises(ValueError, match="timezone"):
        Document(
            document_id="pdf-handbook",
            source_type=SourceType.PDF,
            source_name="handbook.pdf",
            text="Some text",
            location=SourceLocation(page_number=1),
            added_at=datetime(2026, 10, 5),
        )


def test_source_type_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        SourceType("audio")
