"""Local OCR helpers for image-only PDF pages."""

from io import BytesIO
import os
from pathlib import Path
import shutil

import pymupdf
from PIL import Image
import pytesseract


class OCRUnavailableError(RuntimeError):
    """Raised when an image-only page needs OCR but Tesseract is unavailable."""


def find_tesseract_command() -> str:
    """Find Tesseract from configuration, PATH, or common Windows locations."""
    configured = os.getenv("TESSERACT_CMD", "").strip()
    candidates = [
        configured,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise OCRUnavailableError(
        "This PDF contains image-only pages, but Tesseract OCR is not installed. "
        "Install Tesseract or set TESSERACT_CMD to its executable path."
    )


def ocr_pdf_page(pdf_bytes: bytes, page_index: int, *, dpi: int = 200) -> str:
    """Render one zero-based PDF page and recognize its text locally."""
    if page_index < 0:
        raise ValueError("Page index cannot be negative.")
    if dpi < 72:
        raise ValueError("OCR resolution must be at least 72 DPI.")

    pytesseract.pytesseract.tesseract_cmd = find_tesseract_command()
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if page_index >= pdf.page_count:
            raise ValueError("Page index is outside this PDF.")
        pixmap = pdf[page_index].get_pixmap(dpi=dpi, alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image, lang="eng").strip()
