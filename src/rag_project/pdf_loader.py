"""Load PDF pages into the unified document model."""

from collections.abc import Iterable
from datetime import UTC, datetime
from io import BytesIO

from pypdf import PdfReader

from src.rag_project.documents import (
    Document,
    SourceLocation,
    SourceType,
    create_document_id,
)


def extract_pdf_pages(
    pdf_bytes: bytes,
    document_name: str,
    *,
    added_at: datetime | None = None,
) -> list[Document]:
    """Extract non-empty text from a PDF while preserving page numbers."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[Document] = []
    document_id = create_document_id(SourceType.PDF, document_name, pdf_bytes)
    ingestion_time = added_at or datetime.now(UTC)

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(
                Document(
                    document_id=document_id,
                    source_type=SourceType.PDF,
                    source_name=document_name,
                    text=text,
                    location=SourceLocation(page_number=page_number),
                    added_at=ingestion_time,
                    metadata={"media_type": "application/pdf"},
                )
            )

    return pages


def extract_pdf_files(
    pdf_files: Iterable[tuple[str, bytes]],
    *,
    added_at: datetime | None = None,
) -> list[Document]:
    """Extract multiple PDFs into one deduplicated document collection."""
    ingestion_time = added_at or datetime.now(UTC)
    documents: list[Document] = []
    seen_document_ids: set[str] = set()

    for document_name, pdf_bytes in pdf_files:
        pages = extract_pdf_pages(
            pdf_bytes,
            document_name,
            added_at=ingestion_time,
        )
        if not pages:
            continue

        document_id = pages[0].document_id
        if document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)
        documents.extend(pages)

    return documents
