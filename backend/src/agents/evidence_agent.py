import sys
import io
from typing import Dict

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class EvidenceChecklistAgent:
    """Generate domain-specific evidentiary checklists compliant with BSA 2023."""

    def __init__(self):
        labour_items = [
            "Appointment letter / Employment contract",
            "All salary slips (last 6–12 months)",
            "Bank statements showing salary credits and non-payment period",
            "Resignation email / Relieving or termination letter",
            "Full & Final (FnF) computation sheet or HR email correspondence",
            "WhatsApp messages / internal communications (with Section 63 BSA 2023 certificate)",
            "Employee ID card / official badge",
            "Attendance records / timesheets / proof of work delivered",
        ]
        self.checklists = {
            "consumer": [
                "Tax invoice / purchase receipt / bill",
                "Product packaging, serial number, and photographs of defect",
                "Warranty card / guarantee certificate / terms of service",
                "Written communication with seller or customer support (emails, tickets)",
                "Screenshots of product listing / advertisement claims",
                "Proof of payment (UPI/bank statement, transaction receipt)",
                "Legal notice copy and postal/email delivery acknowledgment",
            ],
            "labour": labour_items,
            "labor": labour_items,
            "rent": [
                "Registered or notarized Lease / Leave and License Agreement",
                "Security deposit payment receipt / bank transfer proof",
                "Monthly rent payment receipts / bank statements",
                "Dated photographs/videos of premises condition at entry and exit",
                "Maintenance and utility bills (electricity, water, society charges)",
                "Eviction or termination notice under Section 106 Transfer of Property Act 1882",
                "Written correspondence / WhatsApp messages with landlord/tenant",
            ],
            "rti": [
                "Identity and address proof of applicant (Citizenship under Section 3 RTI Act 2005)",
                "Form A application under Section 6(1) RTI Act 2005",
                "Specific particulars and period of public records sought",
                "Proof of Rs. 10/- fee payment (IPO / Court Fee Stamp / DD / e-RTI receipt)",
                "Speed Post / Registered Post receipt and tracking record",
            ],
            "criminal": [
                "Written complaint copy / FIR copy under Section 173 BNSS 2023",
                "Medico-Legal Certificate (MLC) / medical discharge summary (if physical injury)",
                "High-resolution photographs of crime scene, injury, or property damage",
                "Names, addresses, and phone numbers of eyewitnesses",
                "CCTV footage / audio-video recordings with Section 63 BSA 2023 certificate",
                "Purchase invoice / ownership proof of stolen or snatched articles",
            ],
            "cyber": [
                "Bank account statement highlighting unauthorized debit and UTR number",
                "Screenshots of fraudulent UPI transaction, VPA ID, phishing SMS/link, or fake QR",
                "National Cyber Crime Reporting Portal (1930 / cybercrime.gov.in) acknowledgment number",
                "Written bank branch representation for account freeze / lien under RBI Circular",
                "Call logs, email headers, and exported WhatsApp/Telegram chats",
                "Section 63 Bharatiya Sakshya Adhiniyam (BSA) 2023 electronic evidence certificate",
            ],
            "defamation": [
                "Unaltered screenshots / URLs / recordings of the defamatory publication",
                "Section 63 BSA 2023 certificate for electronic/social media posts",
                "Witness statements of third parties who viewed the defamatory content",
                "Proof of reputational or professional harm suffered",
                "Copy of cease-and-desist / retraction legal notice served",
            ],
            "family": [
                "Marriage certificate and wedding photographs",
                "Income tax returns (ITRs), salary slips, and bank statements of both spouses",
                "Birth certificates and school/medical expense receipts of children",
                "Documentary or electronic evidence of cruelty/desertion (if applicable)",
                "Details of shared household and streedhan / matrimonial assets",
            ],
            "commercial": [
                "Signed contract, work order, or Master Service Agreement (MSA)",
                "Tax invoices, delivery challans, and GST filings",
                "Ledger account statement and bank payment trail",
                "Dishonoured cheque and bank return memo (if Section 138 NI Act applies)",
                "Email trail of payment reminders and formal demand notice",
            ],
            "civil": [
                "Primary contract, title deed, or agreement establishing legal right",
                "Bank statements, payment receipts, and financial loss computation",
                "Chronological written correspondence between parties",
                "Copy of formal legal demand notice and postal acknowledgment (RPAD)",
                "Electronic records accompanied by a Section 63 BSA 2023 certificate",
            ],
        }

    def get_checklist(self, domain: str) -> Dict:
        """Get evidence checklist for domain (normalizing labor/labour aliases)."""
        key = (domain or "civil").strip().lower()
        if key not in self.checklists:
            key = "civil"
        items = self.checklists[key]
        return {
            "domain": key,
            "items": items,
            "total_items": len(items),
            "instruction": (
                f"Gather all {len(items)} evidentiary records before filing your {key} proceeding. "
                "Electronic records (WhatsApp, emails, CCTV, screenshots) require a Section 63 BSA 2023 certificate."
            ),
        }

    def print_checklist(self, domain: str) -> str:
        """Generate formatted checklist."""
        checklist = self.get_checklist(domain)
        output = f"\n📋 Evidence Checklist for {checklist['domain'].upper()} Case\n"
        output += "=" * 50 + "\n"
        output += f"Total items to gather: {checklist['total_items']}\n\n"
        for i, item in enumerate(checklist["items"], 1):
            output += f"☐ {i}. {item}\n"
        output += "\n" + checklist["instruction"] + "\n"
        return output
