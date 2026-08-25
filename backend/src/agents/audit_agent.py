"""
audit_agent.py - Rule-based document audit.

Checks pasted/uploaded document text against per-domain checklists in
config/document_audit_rules.json. Fully deterministic: every "issue" is a
missing keyword from a required clause. No LLM is used, so no invented
legal clauses or numbers.
"""
import json
import re
import sys
import io
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv(override=True)

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class DocumentAuditAgent:
    def __init__(self, rules_path: str = None):
        path = rules_path or str(
            Path(__file__).resolve().parent.parent.parent / "config" / "document_audit_rules.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            self._rules = json.load(f)
        self.documents = self._rules.get("documents", {})
        self.disclaimer = self._rules.get("disclaimer", "")

    def supported_domains(self) -> List[str]:
        return sorted(self.documents.keys())

    def _domain(self, domain: str) -> str:
        d = (domain or "civil").lower()
        return d if d in self.documents else "civil"

    def audit(self, text: str, domain: str = "civil") -> Dict:
        """Audit `text` against the domain checklist. Deterministic."""
        d = self._domain(domain)
        cfg = self.documents[d]
        text = re.sub(r"\s+", " ", (text or "").lower())

        issues, present, missing = [], [], []
        for field in cfg.get("fields", []):
            found = any(kw in text for kw in field.get("keywords", []))
            entry = {
                "id": field["id"],
                "label": field["label"],
                "severity": field.get("severity", "medium"),
                "hint": field.get("hint", ""),
            }
            if found:
                present.append(entry)
            else:
                missing.append(entry)
                issues.append({**entry, "issue": f"'{field['label']}' not found in the document."})

        total = len(cfg.get("fields", []))
        score = round((len(present) / total) * 100) if total else 0
        risk = "LOW" if score >= 80 else ("MEDIUM" if score >= 50 else "HIGH")

        return {
            "domain": d,
            "document_type": cfg.get("label", d),
            "audit_intro": cfg.get("intro", ""),
            "score": score,
            "risk": risk,
            "present_count": len(present),
            "missing_count": len(missing),
            "total_checks": total,
            "present": present,
            "issues": issues,
            "missing_fields": [m["id"] for m in missing],
            "suggestions": [
                f"Add a clear clause for: {m['label']} ({m['hint']})" for m in missing[:6]
            ],
            "disclaimer": self.disclaimer,
        }

    def summary(self, result: Dict) -> str:
        return (
            f"{result['document_type']}: {result['present_count']}/{result['total_checks']} "
            f"clauses found ({result['score']}% - {result['risk']} risk). "
            + ("Missing: " + "; ".join(m["label"] for m in result["issues"][:5]) if result["issues"] else "Looks complete.")
        )


if __name__ == "__main__":
    agent = DocumentAuditAgent()
    sample = """
    RENTAL AGREEMENT between Mr. Rajesh Sharma (landlord) and Mr. Rahul Mehta (tenant).
    Premises: Flat 401, Sunshine Apartments, Ahmedabad.
    Monthly rent is Rs 15,000 payable by the 5th. Security deposit Rs 50,000 refundable on vacating.
    Term: 11 months. Electricity and water bills are paid by the tenant.
    """
    result = agent.audit(sample, "rent")
    print("supported:", agent.supported_domains())
    print(agent.summary(result))
    for i in result["issues"]:
        print(f"  - [{i['severity']}] {i['label']}")
