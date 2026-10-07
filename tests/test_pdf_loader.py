from io import BytesIO
from datetime import UTC, datetime

from pypdf import PdfWriter

from src.rag_project.documents import SourceLocation, SourceType
from src.rag_project.pdf_loader import extract_pdf_files, extract_pdf_pages


def test_empty_upload_is_rejected() -> None:
    try:
        extract_pdf_pages(b"", "empty.pdf")
    except ValueError as error:
        assert str(error) == "The uploaded PDF is empty."
    else:
        raise AssertionError("Expected an empty PDF to be rejected")


def test_blank_pdf_returns_no_text_pages(monkeypatch) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)

    monkeypatch.setattr(
        "src.rag_project.pdf_loader.ocr_pdf_page",
        lambda pdf_bytes, page_index: "",
    )

    assert extract_pdf_pages(output.getvalue(), "blank.pdf") == []


def test_extracted_pages_use_unified_document_metadata(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, stream):
            self.pages = [
                FakePage("This is selectable text on page one."),
                FakePage("This is selectable text on page two."),
            ]

    monkeypatch.setattr("src.rag_project.pdf_loader.PdfReader", FakeReader)
    added_at = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)

    pages = extract_pdf_pages(b"pdf bytes", "handbook.pdf", added_at=added_at)

    assert len(pages) == 2
    assert pages[0].document_id == pages[1].document_id
    assert pages[0].document_id.startswith("pdf-")
    assert pages[0].source_type == SourceType.PDF
    assert pages[0].source_name == "handbook.pdf"
    assert pages[0].location == SourceLocation(page_number=1)
    assert pages[1].location == SourceLocation(page_number=2)
    assert pages[0].added_at == added_at
    assert pages[0].metadata == {
        "media_type": "application/pdf",
        "extraction_method": "text",
    }


def test_image_only_page_uses_ocr_and_preserves_page_number(monkeypatch) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return ""

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage()]

    monkeypatch.setattr("src.rag_project.pdf_loader.PdfReader", FakeReader)
    monkeypatch.setattr(
        "src.rag_project.pdf_loader.ocr_pdf_page",
        lambda pdf_bytes, page_index: "Scanned invoice total GBP 125",
    )

    pages = extract_pdf_pages(b"image pdf", "scan.pdf")

    assert len(pages) == 1
    assert pages[0].text == "Scanned invoice total GBP 125"
    assert pages[0].location.page_number == 1
    assert pages[0].metadata["extraction_method"] == "ocr"


def test_extracts_multiple_pdfs_into_one_collection(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage(stream.read().decode("utf-8"))]

    monkeypatch.setattr("src.rag_project.pdf_loader.PdfReader", FakeReader)
    added_at = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)

    pages = extract_pdf_files(
        [
            ("handbook.pdf", b"Annual leave policy details"),
            ("benefits.pdf", b"Health benefits policy details"),
        ],
        added_at=added_at,
    )

    assert [page.source_name for page in pages] == [
        "handbook.pdf",
        "benefits.pdf",
    ]
    assert len({page.document_id for page in pages}) == 2
    assert all(page.added_at == added_at for page in pages)


def test_duplicate_pdf_is_indexed_only_once(monkeypatch) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "Annual leave policy details"

    class FakeReader:
        def __init__(self, stream):
            self.pages = [FakePage()]

    monkeypatch.setattr("src.rag_project.pdf_loader.PdfReader", FakeReader)

    pages = extract_pdf_files(
        [
            ("handbook.pdf", b"same pdf"),
            ("handbook.pdf", b"same pdf"),
        ]
    )

    assert len(pages) == 1
