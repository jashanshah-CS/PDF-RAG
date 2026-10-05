"""Load PDF pages into the unified document model."""

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
