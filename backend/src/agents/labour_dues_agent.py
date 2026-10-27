"""
labour_dues_agent.py - Unpaid Salary & Labour Dues Claim Agent with Qwen 27B Drafting.
Drafts legal demand notice to employers and petitions under Code on Wages 2019.
"""
from typing import Dict, Any
from datetime import datetime
from src.document_factory import build_labour_dues
from src.groq_drafter import get_groq_drafter, normalize_tool_params


class LabourDuesAgent:
    """Agent for recovering unpaid salary, FnF settlement, and severance dues."""

    def generate_demand(self, data: Dict[str, Any], output_base: str = None) -> Dict[str, Any]:
        data = normalize_tool_params("labour_dues", data)
        employee = data.get("employee_name") or "Employee"
        clean_name = str(employee).replace(" ", "_")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = output_base or f"Salary_Demand_Notice_{clean_name}_{ts}"

        if not data.get("custom_body_paras"):
            try:
                drafter = get_groq_drafter()
                paras = drafter.draft_labour_dues(data)
                if paras:
                    data["custom_body_paras"] = paras
            except Exception:
                pass

        doc_paths = build_labour_dues(data, base_filename)
        company = data.get("company_name") or "Employer"
        amt = data.get("total_dues_amount") or "0"
        return {
            "status": "ready",
            "filename": f"{base_filename}.pdf",
            "docx_filename": f"{base_filename}.docx",
            "pdf_path": doc_paths["pdf_path"],
            "docx_path": doc_paths["docx_path"],
            "base_filename": base_filename,
            "custom_body_paras": data.get("custom_body_paras", []),
            "notice_text": f"Formal Demand Notice to {company} for Rs. {amt}/- under Sections 17 & 18 Code on Wages 2019 & Section 15 Payment of Wages Act 1936.",
        }
