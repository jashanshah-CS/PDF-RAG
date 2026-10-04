"""Create a fictional employee handbook for local RAG testing."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PATH = Path("output/pdf/northstar_employee_handbook.pdf")
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#276FBF")
PALE_BLUE = colors.HexColor("#EAF3FC")
GREEN = colors.HexColor("#218C74")
GREY = colors.HexColor("#5F6B76")


def add_page_chrome(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 17 * mm, width, 17 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(18 * mm, height - 10.5 * mm, "NORTHSTAR ANALYTICS")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 18 * mm, height - 10.5 * mm, "Synthetic RAG test document")
    canvas.setStrokeColor(colors.HexColor("#D8DEE5"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9.5 * mm, "Fictional content - not an actual company policy")
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=27, leading=32, textColor=NAVY, alignment=TA_CENTER,
        spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], fontSize=13, leading=18,
        textColor=BLUE, alignment=TA_CENTER, spaceAfter=10 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=NAVY, spaceBefore=2 * mm,
        spaceAfter=5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=BLUE, spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"], fontSize=10.5,
        leading=15, textColor=colors.HexColor("#263238"), spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"], fontSize=10.5, leading=15,
        textColor=NAVY, backColor=PALE_BLUE, borderColor=BLUE,
        borderWidth=0.7, borderPadding=9, spaceBefore=4 * mm,
        spaceAfter=5 * mm,
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=25 * mm, bottomMargin=21 * mm,
        title="Northstar Analytics Employee Handbook",
        author="Synthetic RAG Test Generator",
        subject="Fictional employee handbook for testing retrieval and answer generation",
    )
    story = []
    body = styles["BodyCustom"]

    # Page 1 - introduction and ownership.
    story.extend([
        Spacer(1, 23 * mm),
        Paragraph("Employee Handbook", styles["TitleCustom"]),
        Paragraph("Northstar Analytics | Effective 1 March 2026", styles["Subtitle"]),
        Paragraph(
            "This document is entirely fictional and was created only to test a local "
            "retrieval-augmented generation (RAG) application.", styles["Callout"]),
        Paragraph("1. Purpose and scope", styles["Section"]),
        Paragraph(
            "Northstar Analytics is a fictional data consultancy. This handbook applies to "
            "all permanent and fixed-term employees in the United Kingdom. Contractors and "
            "agency workers should follow the terms of their own agreements unless a policy "
            "explicitly says otherwise.", body),
        Paragraph("Policy ownership", styles["Subsection"]),
        Paragraph(
            "The People Operations team owns this handbook and reviews it every September. "
            "Questions should be sent to people@northstar.example. Urgent payroll questions "
            "should be sent to payroll@northstar.example.", body),
        Paragraph("Acknowledgement", styles["Subsection"]),
        Paragraph(
            "Employees must acknowledge the handbook in the staff portal within 14 calendar "
            "days of their start date. The handbook provides guidance and does not form part "
            "of an employment contract.", body),
    ])

    # Page 2 - work arrangements and leave.
    story.extend([
        PageBreak(),
        Paragraph("2. Work arrangements and leave", styles["Section"]),
        Paragraph("Hybrid working", styles["Subsection"]),
        Paragraph(
            "Employees may work remotely for up to three days per week with manager approval. "
            "Team members must remain available during the core collaboration hours of "
            "10:00 to 15:00, Monday to Friday. A manager may require office attendance for "
            "client meetings, workshops, or security-sensitive work.", body),
        Paragraph("Annual leave", styles["Subsection"]),
        Paragraph(
            "Full-time employees receive 27 days of paid annual leave per holiday year, in "
            "addition to public holidays. Part-time allowances are calculated pro rata. "
            "Employees may carry over no more than five unused days, and carried-over leave "
            "must be used by 31 March of the following holiday year.", body),
        Paragraph("Probation and sickness", styles["Subsection"]),
        Paragraph(
            "The standard probation period is six months. Employees who are unwell must tell "
            "their manager before 09:30 on the first day of absence. A fit note is required "
            "for an absence lasting more than seven consecutive calendar days.", body),
        Paragraph(
            "Leave requests should normally be submitted at least ten working days in advance. "
            "Managers consider operational coverage and respond in the staff portal.",
            styles["Callout"]),
    ])

    # Page 3 - benefits and expenses, including structured information.
    story.extend([
        PageBreak(),
        Paragraph("3. Benefits and expenses", styles["Section"]),
        Paragraph(
            "The following benefits are available after successful completion of probation. "
            "Any unused allowance expires at the end of the stated period and cannot be paid "
            "as salary.", body),
    ])
    benefits = [
        ["Benefit", "Company contribution or limit", "Period"],
        ["Learning allowance", "GBP 1,200", "Per calendar year"],
        ["Wellness allowance", "GBP 45", "Per month"],
        ["Pension", "6% employer when employee pays 4%", "Each payroll period"],
    ]
    table = Table(benefits, colWidths=[43 * mm, 79 * mm, 43 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F8FA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D0D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        table,
        Spacer(1, 7 * mm),
        Paragraph("Expense claims", styles["Subsection"]),
        Paragraph(
            "Receipts must be submitted through the expense portal within 30 calendar days "
            "of the purchase. The daily meal limit is GBP 25 for domestic travel and GBP 40 "
            "for international travel. Alcohol is not reimbursable unless a director approved "
            "a client event in writing before the event.", body),
        Paragraph(
            "Rail travel should be booked in standard class. Flights longer than eight hours "
            "may be booked in premium economy with prior approval from the Finance Director.", body),
    ])

    # Page 4 - security and conduct.
    story.extend([
        PageBreak(),
        Paragraph("4. Security and conduct", styles["Section"]),
        Paragraph("Information security", styles["Subsection"]),
        Paragraph(
            "Multi-factor authentication is required for every company account. Confidential "
            "client information may be accessed only on encrypted company-managed devices. "
            "It must not be copied to personal email, consumer file-sharing services, or "
            "unapproved removable media.", body),
        Paragraph("Incident reporting", styles["Subsection"]),
        Paragraph(
            "A suspected security incident must be reported within 30 minutes of discovery "
            "to security@northstar.example. If email is unavailable, employees should call "
            "the number printed on the back of their staff badge. Employees should preserve "
            "evidence and must not attempt their own forensic investigation.", body),
        Paragraph("Records and gifts", styles["Subsection"]),
        Paragraph(
            "Client project records are retained for seven years after project closure. Gifts "
            "or hospitality worth more than GBP 75 must be declared to Compliance within five "
            "working days, whether the gift was accepted or declined.", body),
        Paragraph("Emergency communication", styles["Subsection"]),
        Paragraph(
            "Office closures and urgent safety instructions are communicated through the "
            "Northstar Alert service. Employees are responsible for keeping their mobile "
            "number current in the staff portal.", body),
        Paragraph(
            "End of synthetic handbook. All names, addresses, policies, limits, and procedures "
            "in this file were invented for software testing.", styles["Callout"]),
    ])

    doc.build(story, onFirstPage=add_page_chrome, onLaterPages=add_page_chrome)
    print(f"Created {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    build_pdf()
