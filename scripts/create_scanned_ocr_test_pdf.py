"""Create an image-only PDF fixture for the Version 2 OCR test."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas


OUTPUT = Path("output/pdf/version2-evaluation/scanned_ocr_test.pdf")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def create_scanned_pdf() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1240, 1754
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy = "#19344d"
    blue = "#2e6da4"
    draw.rectangle((0, 0, width, 150), fill=navy)
    draw.text((90, 52), "SCANNED OCR TEST", fill="white", font=font(38))
    draw.text((90, 245), "Northwind Equipment Receipt", fill=navy, font=font(56))
    draw.text((90, 345), "Receipt number: NW-2048", fill=blue, font=font(34))
    lines = [
        "Purchase date: 6 October 2026",
        "Item: Portable document scanner",
        "Department: Research Operations",
        "Total paid: GBP 347.50",
        "Approval code: RAVEN-82",
    ]
    y = 500
    for line in lines:
        draw.text((110, y), line, fill="#202a33", font=font(38))
        y += 105
    draw.rectangle((90, 1100, 1150, 1320), outline=blue, width=4)
    draw.text((125, 1155), "This page contains image pixels only.", fill=navy, font=font(32))
    draw.text((125, 1215), "Tesseract must recognize the text.", fill=navy, font=font(32))

    temporary_directory = Path("tmp/pdfs/scanned-ocr-test")
    temporary_directory.mkdir(parents=True, exist_ok=True)
    image_path = temporary_directory / "scanned-page.png"
    image.save(image_path, format="PNG", dpi=(150, 150))
    canvas = Canvas(str(OUTPUT), pagesize=A4)
    canvas.drawImage(str(image_path), 0, 0, width=A4[0], height=A4[1])
    canvas.showPage()
    canvas.save()
    return OUTPUT.resolve()


if __name__ == "__main__":
    print(create_scanned_pdf())
