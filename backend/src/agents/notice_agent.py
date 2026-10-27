"""
notice_agent.py - Formal Legal Notice Generator with Qwen 27B LLM Drafting.
Uses Groq Legal Drafter for advocate-grade legal pleading clauses,
and document_factory for dual PDF and Word (.docx) export.
"""
import sys
import io
import os
from datetime import datetime
from typing import Dict, Any

from src.paths import resolve
from src.document_factory import build_legal_notice
from src.groq_drafter import get_groq_drafter

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class LegalNoticeAgent:
    """Generate formal legal demand notice PDF and DOCX with Qwen 27B drafting."""

    def generate_notice(self, case_data: Dict[str, Any], output_path: str = None) -> str:
        res = self.generate_both(case_data, output_base=None)
        if output_path:
            dest = str(resolve(output_path))
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            if dest != res["pdf_path"]:
                import shutil
                shutil.copy2(res["pdf_path"], dest)
            return dest
        return res["pdf_path"]

    def generate_both(self, case_data: Dict[str, Any], output_base: str = None) -> Dict[str, str]:
        sender = case_data.get("sender_name") or "Sender"
        clean_name = str(sender).replace(" ", "_")[:30]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = output_base or f"Legal_Notice_{clean_name}_{ts}"

        # Enhance with Qwen 27B advocate drafting if not already provided
        if not case_data.get("custom_body_paras"):
            try:
                drafter = get_groq_drafter()
                paras = drafter.draft_legal_notice(case_data)
                if paras:
                    case_data["custom_body_paras"] = paras
            except Exception:
                pass

        doc_paths = build_legal_notice(case_data, base_filename)
        return {
            "status": "ready",
            "filename": f"{base_filename}.pdf",
            "docx_filename": f"{base_filename}.docx",
            "pdf_path": doc_paths["pdf_path"],
            "docx_path": doc_paths["docx_path"],
            "base_filename": base_filename,
            "custom_body_paras": case_data.get("custom_body_paras", [])
        }
