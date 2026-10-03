from io import BytesIO

from pypdf import PdfWriter

from src.rag_project.pdf_loader import extract_pdf_pages


def test_empty_upload_is_rejected() -> None:
    try:
        extract_pdf_pages(b"", "empty.pdf")
    except ValueError as error:
        assert str(error) == "The uploaded PDF is empty."
    else:
        raise AssertionError("Expected an empty PDF to be rejected")


def test_blank_pdf_returns_no_text_pages() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)

    assert extract_pdf_pages(output.getvalue(), "blank.pdf") == []
