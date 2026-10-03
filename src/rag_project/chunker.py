"""Split extracted PDF pages into searchable passages."""

from dataclasses import dataclass

from src.rag_project.pdf_loader import PdfPage


@dataclass(frozen=True)
class PdfChunk:
    """One searchable passage with the metadata needed for a citation."""

    document_name: str
    page_number: int
    chunk_number: int
    text: str


def chunk_pdf_pages(
    pages: list[PdfPage],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[PdfChunk]:
    """Split every page into word-based chunks with a shared overlap."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks: list[PdfChunk] = []
    step = chunk_size - overlap

    for page in pages:
        words = page.text.split()

        for chunk_number, start in enumerate(range(0, len(words), step), start=1):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue

            chunks.append(
                PdfChunk(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_number=chunk_number,
                    text=" ".join(chunk_words),
                )
            )

            if start + chunk_size >= len(words):
                break

    return chunks
