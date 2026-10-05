"""Create five fictional PDFs for Version 2 multi-source evaluation."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


OUTPUT_DIR = Path("output/pdf/version2-evaluation")
NAVY = colors.HexColor("#18324A")
BLUE = colors.HexColor("#356FA3")
PALE_BLUE = colors.HexColor("#EAF3FA")
GREY = colors.HexColor("#5B6770")


DOCUMENTS = [
    {
        "filename": "atlas_travel_policy.pdf",
        "title": "Atlas Travel and Expense Policy",
        "subtitle": "Effective 1 January 2026",
        "pages": [
            (
                "Booking and approval",
                [
                    ("Approved booking channel", "All rail, hotel, and air travel must be booked through Atlas Travel. Employees may book directly only when Atlas Travel confirms in writing that it cannot provide a suitable option."),
                    ("Approval threshold", "A line manager must approve every overnight trip. Any single booking expected to cost more than GBP 750 also requires approval from the Finance Operations Lead before purchase."),
                    ("Travel class", "Rail journeys must use standard class. Flights under eight hours must use economy. Premium economy is permitted for flights lasting eight hours or longer when the department director approves it in advance."),
                ],
            ),
            (
                "Expenses and deadlines",
                [
                    ("Meal limits", "The daily meal limit is GBP 32 for domestic travel and GBP 48 for international travel. Alcohol is not reimbursable unless it was part of a pre-approved client event."),
                    ("Claim deadline", "Expense claims and itemised receipts must be submitted within 21 calendar days of the purchase. Late claims require a written explanation and Finance Operations approval."),
                    ("Ground transport", "Public transport should be used when practical. A taxi may be claimed after 21:30, when carrying heavy company equipment, or when a documented accessibility need makes public transport unsuitable."),
                ],
            ),
        ],
    },
    {
        "filename": "beacon_security_standard.pdf",
        "title": "Beacon Information Security Standard",
        "subtitle": "Mandatory controls for all staff",
        "pages": [
            (
                "Account and data protection",
                [
                    ("Passwords", "Company passwords must contain at least 16 characters. Password reuse across company and personal services is prohibited, and multi-factor authentication is required for every cloud account."),
                    ("Data classification", "Information is classified as Public, Internal, Confidential, or Restricted. Restricted data may be stored only in the encrypted Beacon Vault and accessed from a company-managed device."),
                    ("Removable media", "USB storage devices and other removable media are prohibited unless the Security Director issues a time-limited written exception."),
                ],
            ),
            (
                "Incidents and records",
                [
                    ("Incident reporting", "A suspected security incident must be reported to the Security Desk within 20 minutes of discovery. Staff must preserve evidence and must not conduct their own forensic investigation."),
                    ("Record retention", "Security access logs are retained for 18 months. Closed investigation records are retained for six years from the closure date."),
                    ("Phishing exercises", "Beacon runs a simulated phishing exercise every quarter. Employees who fail two exercises in a rolling year must complete refresher training within ten working days."),
                ],
            ),
        ],
    },
    {
        "filename": "cedar_benefits_guide.pdf",
        "title": "Cedar Employee Benefits Guide",
        "subtitle": "Benefits year 2026",
        "pages": [
            (
                "Leave and family support",
                [
                    ("Annual leave", "Full-time employees receive 29 days of paid annual leave in addition to public holidays. Up to four unused days may be carried over and must be used by 30 April."),
                    ("Volunteer leave", "Every employee may take two paid volunteer days per calendar year for work with a registered charity."),
                    ("Family support", "Primary carers receive 20 weeks of fully paid parental leave. Secondary carers receive six weeks of fully paid parental leave."),
                ],
            ),
            (
                "Allowances and pension",
                [
                    ("Learning allowance", "Employees receive a GBP 1,500 learning allowance per calendar year after completing probation. The allowance cannot be exchanged for salary."),
                    ("Wellness allowance", "The company reimburses up to GBP 55 per month for eligible fitness, wellbeing, or mindfulness services."),
                    ("Pension contribution", "Cedar contributes 7 percent of pensionable salary when the employee contributes at least 5 percent. Contributions begin in the first full payroll period after enrolment."),
                ],
            ),
        ],
    },
    {
        "filename": "project_orion_brief.pdf",
        "title": "Project Orion Delivery Brief",
        "subtitle": "Inventory forecasting programme",
        "pages": [
            (
                "Purpose and delivery",
                [
                    ("Objective", "Project Orion will deploy an inventory forecasting service to the Glasgow and Leeds distribution centres by 14 November 2026."),
                    ("Ownership", "Maya Chen is the executive sponsor. Ruben Patel is the delivery manager, and the Data Platform team owns the production service."),
                    ("Budget", "The approved programme budget is GBP 680,000. Any forecast overrun greater than 8 percent must be escalated to the investment committee."),
                ],
            ),
            (
                "Measures and delivery risks",
                [
                    ("Success measures", "The programme aims to reduce stockout events by 12 percent and achieve at least 92 percent weekly demand-forecast accuracy during the first eight weeks after launch."),
                    ("Pilot schedule", "The Glasgow pilot begins on 7 September 2026. The Leeds pilot begins on 5 October 2026, subject to completion of data-quality checks."),
                    ("Primary risk", "The primary delivery risk is inconsistent supplier lead-time data. The mitigation is a weekly exception report reviewed by Procurement and the Data Platform team."),
                ],
            ),
        ],
    },
    {
        "filename": "summit_office_manual.pdf",
        "title": "Summit Office Operations Manual",
        "subtitle": "Workplace procedures",
        "pages": [
            (
                "Access and workplace use",
                [
                    ("Opening hours", "Standard office access is available from 07:00 to 20:00 on working days. Weekend access requires Facilities approval at least two working days in advance."),
                    ("Desk booking", "Employees must reserve a desk by 16:00 on the previous working day. Unoccupied bookings are released automatically at 10:30."),
                    ("Visitors", "Every visitor must sign in, wear a red visitor badge, and remain with a host while inside secure work areas."),
                ],
            ),
            (
                "Safety and facilities",
                [
                    ("Evacuation", "The primary assembly point is Riverside Square. If Riverside Square is unavailable, staff must use the secondary assembly point at Market Lane car park."),
                    ("Emergency drills", "Each office conducts an evacuation drill every quarter. Fire wardens record attendance and submit the drill report within two working days."),
                    ("Facilities requests", "Urgent heating, water, power, or access-control faults have a target response time of 30 minutes. Routine requests have a target response time of two working days."),
                ],
            ),
        ],
    },
]


def styles():
    sheet = getSampleStyleSheet()
    sheet.add(ParagraphStyle(
        name="TestTitle", parent=sheet["Title"], fontName="Helvetica-Bold",
        fontSize=25, leading=30, textColor=NAVY, alignment=TA_CENTER,
        spaceAfter=5 * mm,
    ))
    sheet.add(ParagraphStyle(
        name="TestSubtitle", parent=sheet["Normal"], fontSize=12, leading=16,
        textColor=BLUE, alignment=TA_CENTER, spaceAfter=12 * mm,
    ))
    sheet.add(ParagraphStyle(
        name="TestSection", parent=sheet["Heading1"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=NAVY, spaceAfter=6 * mm,
    ))
    sheet.add(ParagraphStyle(
        name="TestHeading", parent=sheet["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=BLUE, spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    ))
    sheet.add(ParagraphStyle(
        name="TestBody", parent=sheet["BodyText"], fontSize=10.5, leading=15,
        textColor=colors.HexColor("#263238"), spaceAfter=4 * mm,
    ))
    sheet.add(ParagraphStyle(
        name="TestNote", parent=sheet["BodyText"], fontSize=9.5, leading=14,
        textColor=NAVY, backColor=PALE_BLUE, borderColor=BLUE,
        borderWidth=0.7, borderPadding=8, spaceAfter=6 * mm,
    ))
    return sheet


def page_chrome(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 17 * mm, width, 17 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(18 * mm, height - 10.5 * mm, "VERSION 2 EVALUATION SET")
    canvas.setStrokeColor(colors.HexColor("#D6DEE5"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9.5 * mm, "Fictional content created for local RAG testing")
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def create_document(specification, style_sheet):
    output_path = OUTPUT_DIR / specification["filename"]
    document = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=26 * mm, bottomMargin=21 * mm,
        title=specification["title"], author="Synthetic RAG Test Generator",
        subject="Fictional Version 2 evaluation document",
    )
    story = [
        Spacer(1, 18 * mm),
        Paragraph(specification["title"], style_sheet["TestTitle"]),
        Paragraph(specification["subtitle"], style_sheet["TestSubtitle"]),
        Paragraph(
            "This document is fictional and exists only to test retrieval, grounded "
            "answers, source separation, citations, and refusals.",
            style_sheet["TestNote"],
        ),
    ]
    for page_index, (section, topics) in enumerate(specification["pages"]):
        if page_index:
            story.append(PageBreak())
        story.append(Paragraph(section, style_sheet["TestSection"]))
        for heading, body in topics:
            story.append(Paragraph(heading, style_sheet["TestHeading"]))
            story.append(Paragraph(body, style_sheet["TestBody"]))
    document.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style_sheet = styles()
    for specification in DOCUMENTS:
        path = create_document(specification, style_sheet)
        print(path.resolve())


if __name__ == "__main__":
    main()
