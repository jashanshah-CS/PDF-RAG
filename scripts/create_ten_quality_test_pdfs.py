"""Create ten fictional PDFs for the Version 2 answer-quality benchmark."""

from pathlib import Path

import create_version2_test_pdfs as pdf_builder


OUTPUT_DIR = Path("output/pdf/quality-test-10")

DOCUMENTS = [
    {
        "filename": "aurora_remote_work_policy.pdf",
        "title": "Aurora Remote Work Policy",
        "subtitle": "Effective March 2026",
        "pages": [("Remote work rules", [
            ("Office attendance", "Employees must work from their assigned office on Tuesdays and Thursdays. A department head may approve a temporary exception for up to six weeks."),
            ("Equipment", "Aurora provides one laptop, one monitor, and a GBP 240 home-office allowance every two years."),
            ("Working abroad", "Working outside the United Kingdom is limited to 15 calendar days in a rolling 12-month period and requires People Operations approval before travel."),
        ])],
    },
    {
        "filename": "bluebird_customer_support_handbook.pdf",
        "title": "Bluebird Customer Support Handbook",
        "subtitle": "Service standards for 2026",
        "pages": [("Response and escalation", [
            ("Priority one", "Priority-one incidents require an initial response within 15 minutes and an update every 30 minutes until service is restored."),
            ("Priority two", "Priority-two incidents require an initial response within two hours and an update every four hours."),
            ("Escalation", "An unresolved priority-one incident must be escalated to the duty manager after 45 minutes."),
        ])],
    },
    {
        "filename": "coral_procurement_rules.pdf",
        "title": "Coral Procurement Rules",
        "subtitle": "Purchasing controls",
        "pages": [("Approvals and suppliers", [
            ("Quotation requirement", "Purchases above GBP 5,000 require three written supplier quotations."),
            ("Executive approval", "A purchase above GBP 25,000 requires approval from both the budget owner and the Chief Financial Officer."),
            ("Conflicts", "Employees must declare a supplier conflict of interest before participating in vendor selection."),
        ])],
    },
    {
        "filename": "delta_data_retention_schedule.pdf",
        "title": "Delta Data Retention Schedule",
        "subtitle": "Approved corporate retention periods",
        "pages": [("Retention periods", [
            ("Customer contracts", "Signed customer contracts must be retained for seven years after the contract ends."),
            ("Recruitment records", "Unsuccessful applicant records must be deleted 12 months after the recruitment decision."),
            ("Invoices", "Supplier invoices must be retained for six years after the end of the relevant financial year."),
        ])],
    },
    {
        "filename": "ember_training_catalogue.pdf",
        "title": "Ember Training Catalogue",
        "subtitle": "Professional development programme",
        "pages": [("Courses and eligibility", [
            ("Leadership Essentials", "Leadership Essentials runs for eight weeks and is available to employees who have managed people for at least six months."),
            ("Cloud Foundations", "Cloud Foundations is a three-day course open to all permanent employees."),
            ("Exam support", "Ember reimburses one certification exam fee per calendar year when the employee passes the exam."),
        ])],
    },
    {
        "filename": "falcon_business_continuity_plan.pdf",
        "title": "Falcon Business Continuity Plan",
        "subtitle": "Critical service recovery",
        "pages": [("Recovery priorities", [
            ("Payments service", "The payments service has a recovery time objective of two hours and a recovery point objective of 15 minutes."),
            ("Customer portal", "The customer portal has a recovery time objective of six hours."),
            ("Coordination", "The Crisis Management Team meets within 30 minutes of a severity-one continuity event."),
        ])],
    },
    {
        "filename": "greenwood_sustainability_plan.pdf",
        "title": "Greenwood Sustainability Plan",
        "subtitle": "Environmental targets to 2030",
        "pages": [("Targets and reporting", [
            ("Carbon target", "Greenwood will reduce scope-one and scope-two emissions by 46 percent from the 2022 baseline by December 2030."),
            ("Electricity", "All company-operated offices must use 100 percent renewable electricity by the end of 2027."),
            ("Reporting", "Progress is reviewed quarterly and published in an annual sustainability report each May."),
        ])],
    },
    {
        "filename": "harbour_expenses_guide.pdf",
        "title": "Harbour Expenses Guide",
        "subtitle": "Employee reimbursement rules",
        "pages": [("Allowances and claims", [
            ("Mileage", "Use of a personal car for approved business travel is reimbursed at 45 pence per mile for the first 10,000 miles in a tax year."),
            ("Home internet", "Remote employees may claim up to GBP 30 per month toward home internet costs."),
            ("Deadline", "Expense claims must be submitted within 30 calendar days of the transaction date."),
        ])],
    },
    {
        "filename": "indigo_product_release_process.pdf",
        "title": "Indigo Product Release Process",
        "subtitle": "Production change governance",
        "pages": [("Release controls", [
            ("Code freeze", "The standard production code freeze begins at 17:00 every Thursday."),
            ("Approval", "A production release requires approval from Quality Assurance and the service owner."),
            ("Rollback", "Every release must include a tested rollback plan that can restore service within 20 minutes."),
        ])],
    },
    {
        "filename": "juniper_visitor_safety_guide.pdf",
        "title": "Juniper Visitor Safety Guide",
        "subtitle": "Rules for guests and contractors",
        "pages": [("Visitor requirements", [
            ("Identification", "Visitors must present photo identification and wear a yellow visitor badge at all times."),
            ("Supervision", "Visitors must remain with their host in laboratories and server rooms."),
            ("Emergency point", "During an evacuation, visitors must report to the north car park assembly point with their host."),
        ])],
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_builder.OUTPUT_DIR = OUTPUT_DIR
    style_sheet = pdf_builder.styles()
    for specification in DOCUMENTS:
        print(pdf_builder.create_document(specification, style_sheet).resolve())


if __name__ == "__main__":
    main()
