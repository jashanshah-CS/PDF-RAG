import pytest

from src.rag_project.chunker import chunk_pdf_pages
from src.rag_project.pdf_loader import PdfPage


def make_page(text: str, page_number: int = 1) -> PdfPage:
    return PdfPage(
        document_name="handbook.pdf",
        page_number=page_number,
        text=text,
    )


def test_short_page_becomes_one_chunk_with_citation_metadata() -> None:
    chunks = chunk_pdf_pages([make_page("Annual leave is 25 days.", page_number=4)])

    assert len(chunks) == 1
    assert chunks[0].document_name == "handbook.pdf"
    assert chunks[0].page_number == 4
    assert chunks[0].chunk_number == 1
    assert chunks[0].text == "Annual leave is 25 days."


def test_chunks_overlap_without_losing_words() -> None:
    page = make_page("zero one two three four five six seven eight nine")

    chunks = chunk_pdf_pages([page], chunk_size=6, overlap=2)

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
        chunk_pdf_pages([make_page("some text")], chunk_size, overlap)
