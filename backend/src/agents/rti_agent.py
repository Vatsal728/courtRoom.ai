"""
rti_agent.py - RTI Section 6 Application Generator.
Generates official Form A printable PDF, editable DOCX, and formatted text for submission.
"""
import sys
import io
from datetime import datetime
from typing import Dict, Any

from src.document_factory import build_rti_application

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class RTIApplicationAgent:
    """Generate RTI (Right to Information) Form A applications."""

    def generate_rti_application(self, data: Dict[str, Any]) -> str:
        """Returns standard Section 6 application text."""
        return f"""
RTI APPLICATION (Section 6, RTI Act 2005)

Date: {datetime.now().strftime('%d-%m-%Y')}

TO,
The Public Information Officer (PIO),
{data.get('pio_office', 'Government Office Name')},
{data.get('pio_address', 'Office Address')},

Dear Sir/Madam,

I hereby apply under Section 6 of the Right to Information Act, 2005, to obtain the following information:

INFORMATION REQUESTED:
{data.get('information_sought', 'Describe the information you want')}

APPLICATION FEE:
Enclosed Rs. 10/- via Indian Postal Order (IPO) / Demand Draft / Court Fee Stamp.

DETAILS OF APPLICANT:
Name: {data.get('applicant_name', 'Applicant Name')}
Address: {data.get('applicant_address', 'Applicant Address')}
Phone: {data.get('applicant_phone', 'Phone Number')}
Email: {data.get('applicant_email', 'Email Address')}

Yours faithfully,

{data.get('applicant_name', 'Applicant Name')}
Signature: ___________________
"""

    def generate_both(self, data: Dict[str, Any], output_base: str = None) -> Dict[str, Any]:
        """Generates Form A PDF and editable DOCX."""
        applicant = data.get("applicant_name") or "Applicant"
        clean_name = str(applicant).replace(" ", "_")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = output_base or f"RTI_Application_{clean_name}_{ts}"

        doc_paths = build_rti_application(data, base_filename)
        text_app = self.generate_rti_application(data)
        return {
            "status": "ready",
            "filename": f"{base_filename}.pdf",
            "docx_filename": f"{base_filename}.docx",
            "pdf_path": doc_paths["pdf_path"],
            "docx_path": doc_paths["docx_path"],
            "base_filename": base_filename,
            "application_text": text_app
        }
