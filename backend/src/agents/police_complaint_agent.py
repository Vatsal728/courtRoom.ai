"""
police_complaint_agent.py - Written Police Complaint Agent with Qwen 27B Drafting.
Drafts formal written complaints to SHO under Section 173 BNSS 2023 with receiving counterfoil.
"""
from typing import Dict, Any
from datetime import datetime
from src.document_factory import build_police_complaint, infer_bns_offence_sections
from src.groq_drafter import get_groq_drafter, normalize_tool_params


class PoliceComplaintAgent:
    """Agent for drafting police complaints under Bharatiya Nagarik Suraksha Sanhita (BNSS 2023)."""

    def generate_complaint(self, data: Dict[str, Any], output_base: str = None) -> Dict[str, Any]:
        data = normalize_tool_params("police_complaint", data)
        if not str(data.get("applicable_section") or "").strip():
            data["applicable_section"] = infer_bns_offence_sections(data.get("incident_facts", ""))

        complainant = data.get("complainant_name") or "Complainant"
        clean_name = str(complainant).replace(" ", "_")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = output_base or f"Police_Complaint_SHO_{clean_name}_{ts}"

        if not data.get("custom_body_paras"):
            try:
                drafter = get_groq_drafter()
                paras = drafter.draft_police_complaint(data)
                if paras:
                    data["custom_body_paras"] = paras
            except Exception:
                pass

        doc_paths = build_police_complaint(data, base_filename)
        station = data.get("police_station_name") or "Local Police Station"
        bns_sec = data.get("applicable_section")
        return {
            "status": "ready",
            "filename": f"{base_filename}.pdf",
            "docx_filename": f"{base_filename}.docx",
            "pdf_path": doc_paths["pdf_path"],
            "docx_path": doc_paths["docx_path"],
            "base_filename": base_filename,
            "custom_body_paras": data.get("custom_body_paras", []),
            "complaint_text": f"Written Police Complaint to SHO, {station} under Section 173 BNSS 2023 & {bns_sec}.",
        }
