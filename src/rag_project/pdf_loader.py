"""PDF loading utilities for Version 1."""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPage:
    """Text extracted from one PDF page, with citation metadata."""

    document_name: str
    page_number: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes, document_name: str) -> list[PdfPage]:
    """Extract non-empty text from a PDF while preserving page numbers."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[PdfPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(
                PdfPage(
                    document_name=document_name,
                    page_number=page_number,
                    text=text,
                )
            )

    return pages

