import pytest
from datetime import UTC, datetime

from src.rag_project.chunker import chunk_documents
from src.rag_project.documents import Document, SourceLocation, SourceType


def make_page(text: str, page_number: int = 1) -> Document:
    return Document(
        document_id="pdf-handbook",
        source_type=SourceType.PDF,
        source_name="handbook.pdf",
        text=text,
        location=SourceLocation(page_number=page_number),
        added_at=datetime(2026, 10, 5, tzinfo=UTC),
    )


def test_short_page_becomes_one_chunk_with_citation_metadata() -> None:
    chunks = chunk_documents([make_page("Annual leave is 25 days.", page_number=4)])

    assert len(chunks) == 1
    assert chunks[0].document_id == "pdf-handbook"
    assert chunks[0].source_name == "handbook.pdf"
    assert chunks[0].source_type == SourceType.PDF
    assert chunks[0].location.page_number == 4
    assert chunks[0].chunk_number == 1
    assert chunks[0].text == "Annual leave is 25 days."


def test_chunks_overlap_without_losing_words() -> None:
    page = make_page("zero one two three four five six seven eight nine")

    chunks = chunk_documents([page], chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "zero one two three four five",
        "four five six seven eight nine",
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_settings_are_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_documents([make_page("some text")], chunk_size, overlap)
