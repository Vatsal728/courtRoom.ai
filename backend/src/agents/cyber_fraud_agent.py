"""
cyber_fraud_agent.py - Emergency Cyber Fraud & Bank Freeze Agent with Qwen 27B Drafting.
Generates immediate Bank Branch Representation (Lien/Account Freeze under RBI Circulars)
and National Cyber Crime Reporting Portal (1930) complaint filing text in PDF and DOCX.
"""
from typing import Dict, Any
from datetime import datetime
from src.document_factory import build_cyber_fraud_letter
from src.groq_drafter import get_groq_drafter, normalize_tool_params


class CyberFraudAgent:
    """Agent for handling UPI scams, unauthorized bank debits, and cyber frauds."""

    def generate_filing(self, data: Dict[str, Any], output_base: str = None) -> Dict[str, Any]:
        data = normalize_tool_params("cyber_fraud", data)
        victim = data.get("victim_name") or "Victim"
        clean_name = str(victim).replace(" ", "_")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = output_base or f"Cyber_Fraud_Freeze_Notice_{clean_name}_{ts}"

        if not data.get("custom_body_paras"):
            try:
                drafter = get_groq_drafter()
                paras = drafter.draft_cyber_fraud_letter(data)
                if paras:
                    data["custom_body_paras"] = paras
            except Exception:
                pass

        doc_paths = build_cyber_fraud_letter(data, base_filename)
        amount = data.get("amount_lost") or "0"
        bank = data.get("victim_bank_name") or "Bank"
        utr = data.get("transaction_id_utr") or "N/A"
        vpa = data.get("beneficiary_upi_vpa_or_account") or "N/A"
        portal_instructions = (
            "1. Immediately dial 1930 (National Cyber Crime Helpline) and report UTR "
            f"{utr} to trigger an automated NCRP lien on {vpa}.\n"
            "2. Log on to https://cybercrime.gov.in and file a formal complaint under 'Report Cyber Financial Fraud'.\n"
            f"3. Submit the signed PDF/DOCX Bank Freeze Requisition to {bank} within 3 working days under "
            "RBI Circular DBR.No.Leg.BC.78/09.07.005/2017-18 and obtain a stamped acknowledgment on the counterfoil."
        )
        return {
            "status": "ready",
            "filename": f"{base_filename}.pdf",
            "docx_filename": f"{base_filename}.docx",
            "pdf_path": doc_paths["pdf_path"],
            "docx_path": doc_paths["docx_path"],
            "base_filename": base_filename,
            "custom_body_paras": data.get("custom_body_paras", []),
            "letter_text": f"Emergency Freeze Requisition to {bank} for Rs. {amount}/- (UTR: {utr}, Beneficiary: {vpa})",
            "portal_instructions": portal_instructions,
        }
