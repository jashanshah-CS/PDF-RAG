"""Split unified documents into searchable passages."""

from src.rag_project.documents import Document, DocumentChunk


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[DocumentChunk]:
    """Split source records into word-based chunks with a shared overlap."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap

    for document in documents:
        words = document.text.split()

        for chunk_number, start in enumerate(range(0, len(words), step), start=1):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue

            chunks.append(
                DocumentChunk.from_document(
                    document,
                    chunk_number=chunk_number,
                    text=" ".join(chunk_words),
                )
            )

            if start + chunk_size >= len(words):
                break

    return chunks


def chunk_pdf_pages(
    pages: list[Document],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[DocumentChunk]:
    """Backward-compatible name for chunking extracted PDF pages."""
    return chunk_documents(pages, chunk_size, overlap)
