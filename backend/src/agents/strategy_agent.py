"""
strategy_agent.py - Case strategy builder (deterministic).

Assembles a structured case strategy from the case description + facts:
legal route, forums, evidence checklist, limitation deadlines, compensation
range (from compensation_engine.py) and a step-by-step action plan.

Every number (compensation, deadlines) comes from the deterministic engine /
config. The LLM never generates figures; it only supplies the description.
"""
import re
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.compensation_engine import get_compensation_engine
from src.agents.evidence_agent import EvidenceChecklistAgent
from src.agents.deadline_agent import DeadlineTrackerAgent

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_DISCLAIMER = (
    "This strategy is auto-generated from the facts you provided for planning. "
    "It is not legal advice; a lawyer should confirm the law, forum and "
    "limitation period for your specific facts before you act."
)

_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|rupees?|₹)\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore|cr|thousand|k)?\b"
    r"|\b([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore|cr)\b"
    r"|\b([\d]{5,})\b"
    r"|\b([\d]{1,2},\d{2},\d{3}(?:,\d{2,3})*|\d{1,3},\d{3}(?:,\d{3})*)\b",
    re.I,
)

_MULT = {
    "lakh": 100_000,
    "lac": 100_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "thousand": 1_000,
    "k": 1_000,
}

_DOMAIN_HINTS = [
    ("rent", ["landlord", "tenant", "rent", "deposit", "eviction", "lease", "premises", "flat", "apartment", "vacat"]),
    ("consumer", ["consumer", "bought", "purchased", "product", "refund", "warranty", "defective", "seller", "service", "replacement", "builder", "possession", "occupancy"]),
    ("labor", ["salary", "wage", "employer", "employee", "fired", "dismissed", "terminat", "layoff", "gratuity", "appointment letter", "bonus", "fnf"]),
    ("criminal", ["stole", "stolen", "robbery", "snatch", "snatched", "cheated", "fraud", "scam", "assault", "attack", "hit", "murder", "threat", "theft"]),
    ("cyber", ["hacked", "hack", "cyber", "phishing", "data breach", "otp", "upi", "kyc", "instagram", "whatsapp", "ransomware"]),
    ("defamation", ["defamat", "defamed", "reputation", "slander", "libel", "false allegation"]),
    ("family", ["maintenance", "custody", "husband", "wife", "divorce", "alimony", "dowry", "children"]),
    ("commercial", ["contract", "breach", "invoice", "payment due", "supplier", "goods", "partnership", "debt", "interest", "cheque"]),
]

_DOMAIN_MAP = {
    "rent": "rent",
    "civil": "civil",
    "civil_only": "civil",
    "general": "civil",
    "labor": "labor",
    "labour": "labor",
    "consumer": "consumer",
    "criminal": "criminal",
    "cyber": "cyber",
    "defamation": "defamation",
    "family": "family",
    "commercial": "commercial",
    "commercial_contract": "commercial",
}

_DEADLINE_MAP = {
    "consumer": "consumer",
    "labor": "labour_salary",
    "rent": "rent_eviction",
    "criminal": "criminal_fir",
    "commercial": "civil_recovery",
    "civil": "civil_recovery",
}

_FORUMS = {
    "rent": ["Civil Court / Rent Controller (district where the property is situated)", "Consumer Commission if paid brokerage/property management services are deficient"],
    "consumer": ["District / State Consumer Disputes Redressal Commission (within 2 years under Section 69, Consumer Protection Act 2019)", "RERA Authority (if real estate / builder possession delay)"],
    "labor": ["Authority under Code on Wages 2019 / District Labour Commissioner (Sections 17, 18 & 45)", "Labour Court / Industrial Tribunal (industrial disputes)"],
    "criminal": ["Local Police Station (FIR under Section 173 BNSS 2023) - then Judicial Magistrate Court", "Application before Magistrate under Section 175(3) BNSS 2023 if police refuse FIR"],
    "cyber": ["National Cyber Crime Reporting Portal (cybercrime.gov.in / Helpline 1930) & Cyber Crime Police Station", "Bank Branch & Banking Ombudsman (RBI CMS Portal) under RBI Zero Liability Circular"],
    "family": ["Family Court of competent local jurisdiction (maintenance / custody / matrimonial relief)"],
    "defamation": ["Judicial Magistrate Court (criminal complaint under Section 356 BNS 2023) + Civil Court suit for damages"],
    "commercial": ["Commercial Court / Civil Court (recovery & damages under Section 73 Contract Act 1872)", "Arbitration Tribunal if agreement contains an arbitration clause"],
    "civil": ["Civil Court of competent jurisdiction (within 3-year limitation under Limitation Act 1963)"],
}

_ACTIONS = {
    "rent": [
        "Serve a formal statutory notice under Section 106 of the Transfer of Property Act, 1882 demanding arrears/deposit refund within 15 days.",
        "Compile the lease agreement, security deposit receipt, rent transfer trail, and handover photographs.",
        "If the opposite party defaults after 15 days, institute a petition before the Rent Controller / Civil Court.",
    ],
    "consumer": [
        "Issue a written grievance to the seller/builder's nodal officer and preserve the ticket/email trail.",
        "Serve a formal 15-day Legal Demand Notice quantifying principal refund, interest, and mental agony damages.",
        "File a verified complaint on e-Daakhil before the Consumer Commission within the 2-year limitation under Section 69 CPA 2019.",
    ],
    "labor": [
        "Serve a formal 7-day Legal Demand Notice on the Company Directors & HR under Sections 17(2) and 18 of the Code on Wages, 2019 and Section 15 of the Payment of Wages Act, 1936.",
        "Compile your appointment letter, salary slips, bank statements, resignation/relieving emails, and FnF calculation.",
        "File a recovery claim before the Authority under the Code on Wages / District Labour Commissioner seeking principal dues plus up to 10x penalty.",
    ],
    "criminal": [
        "Submit a formal written complaint to the SHO under Section 173 BNSS 2023 and obtain a stamped receiving counterfoil with GD/DD number.",
        "Preserve medical reports (MLC), photographs, witness contacts, and CCTV/electronic records with a Section 63 BSA 2023 certificate.",
        "If the SHO fails to register the FIR, send the complaint to the SP/DCP under Section 173(4) BNSS 2023 and move the Magistrate under Section 175(3) BNSS 2023.",
    ],
    "cyber": [
        "Immediately call 1930 and file a complaint on cybercrime.gov.in to trigger an automated lien on the recipient trail.",
        "Submit the written Bank Account Freeze Requisition to your Bank Branch Manager under RBI Circular DBR.No.Leg.BC.78/09.07.005/2017-18 within 3 working days.",
        "Preserve transaction UTRs, SMS alerts, UPI screenshots, and bank statements with a Section 63 BSA 2023 certificate.",
    ],
    "family": [
        "File a maintenance/relief petition before the Family Court (including application for interim maintenance under Section 144 BNSS 2023 / personal law).",
        "Compile marriage proof, income/asset disclosures, and children's educational/medical expense receipts.",
    ],
    "defamation": [
        "Preserve unaltered URLs, screenshots, and metadata accompanied by a Section 63 BSA 2023 electronic evidence certificate.",
        "Serve a peremptory Cease-and-Desist & Retraction Legal Notice demanding unconditional takedown and damages.",
        "File a criminal complaint under Section 356 BNS 2023 and/or a civil suit for defamation damages.",
    ],
    "commercial": [
        "Serve a formal Legal Demand Notice quantifying principal dues and 18% p.a. commercial interest under Section 73 Contract Act 1872.",
        "Invoke the contract's dispute-resolution / arbitration clause or initiate pre-institution commercial mediation.",
        "File a summary recovery suit (Order XXXVII CPC) or Section 138 NI Act complaint within statutory limitation.",
    ],
    "civil": [
        "Assemble all documentary and electronic proof of the claim (with Section 63 BSA 2023 certification).",
        "Serve a 15-day statutory Legal Notice demanding compliance/compensation.",
        "Institute a civil suit before the court of competent pecuniary and territorial jurisdiction within limitation.",
    ],
}


class CaseStrategyAgent:
    def __init__(self):
        self.engine = get_compensation_engine()
        self.evidence = EvidenceChecklistAgent()
        self.deadlines = DeadlineTrackerAgent()

    @staticmethod
    def _detect_domain(text: str) -> str:
        t = (text or "").lower()
        scores: Dict[str, int] = {}
        for domain, hints in _DOMAIN_HINTS:
            score = 0
            for h in hints:
                if len(h) <= 4 or h.isalpha():
                    if re.search(rf"\b{re.escape(h)}\b", t):
                        score += 1
                elif h in t:
                    score += 1
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "civil"

    @staticmethod
    def _extract_amounts(text: str) -> Dict[str, float]:
        """Find rupee amounts including Indian comma groupings and lakh/crore multipliers."""
        amounts: List[float] = []
        for m in _AMOUNT_RE.finditer(text or ""):
            num_str = m.group(1) or m.group(3) or m.group(5) or m.group(6) or ""
            unit = (m.group(2) or m.group(4) or "").lower().rstrip("s")
            try:
                val = float(num_str.replace(",", ""))
            except ValueError:
                continue
            if unit in _MULT:
                val *= _MULT[unit]
            if not (100 <= val <= 10_000_000_000):
                continue
            if 1900 <= val <= 2100 and len(num_str.replace(",", "")) == 4 and not unit:
                continue  # year, not rupee amount
            if val not in amounts:
                amounts.append(val)
        t = (text or "").lower()
        result: Dict[str, float] = {}
        if not amounts:
            return result
        if any(w in t for w in ("deposit", "advance")):
            result["deposit_amount"] = amounts[0]
        elif any(w in t for w in ("salary", "wage", "unpaid", "fnf")):
            result["unpaid_amount"] = sum(amounts) if len(amounts) > 1 else amounts[0]
        elif any(w in t for w in ("stole", "stolen", "robbed", "snatched")):
            result["stolen_value"] = amounts[0]
        elif any(w in t for w in ("cheated", "fraud", "scam", "debited")):
            result["cheated_amount"] = amounts[0]
        elif any(w in t for w in ("medical", "hospital", "treatment", "injury")):
            result["medical_bills"] = amounts[0]
        else:
            result["amount"] = max(amounts)
            result["loss_amount"] = max(amounts)
        return result

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", text or "")
        if not m:
            return None
        return m.group(1).replace("/", "-")

    def build(
        self,
        description: str,
        domain: str = None,
        incident_date: str = None,
        facts: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        facts = dict(facts or {})
        facts.update(self._extract_amounts(description))
        if not incident_date:
            incident_date = self._extract_date(description) or facts.get("incident_date")
        domain = domain or self._detect_domain(description)
        domain_key = _DOMAIN_MAP.get((domain or "civil").lower(), "civil")

        compensation = self.engine.estimate(domain_key, facts)

        deadline = None
        deadline_type = _DEADLINE_MAP.get(domain_key)
        if deadline_type and incident_date:
            dl_res = self.deadlines.calculate_deadline(deadline_type, incident_date)
            if "error" not in dl_res:
                deadline = dl_res

        checklist_obj = self.evidence.get_checklist(domain_key)
        ev_list = checklist_obj.get("items", []) if isinstance(checklist_obj, dict) else list(checklist_obj or [])
        strengths, weaknesses = self._assess(description, facts, compensation, incident_date)

        strategy_res = {
            "domain": domain_key,
            "summary": self._summary(domain_key, description, compensation),
            "assessment": {"strengths": strengths, "weaknesses": weaknesses},
            "legal_route": {
                "criminal": domain_key in ("criminal", "cyber", "defamation"),
                "civil": True,
                "forums": _FORUMS.get(domain_key, _FORUMS["civil"]),
            },
            "compensation_estimate": compensation,
            "evidence_checklist": ev_list,
            "evidence_checklist_details": checklist_obj,
            "deadline": deadline,
            "action_plan": _ACTIONS.get(domain_key, _ACTIONS["civil"]),
            "disclaimer": _DISCLAIMER,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"Case_Strategy_{domain_key}_{ts}"
        try:
            from src.document_factory import build_case_strategy

            doc_data = {
                "summary": strategy_res["summary"],
                "domain": domain_key,
                "compensation_estimate": compensation,
                "forums": strategy_res["legal_route"]["forums"],
                "evidence_checklist": ev_list,
                "stages": [
                    {"name": f"Stage {i+1}", "description": a, "timeline": "Within statutory limitation"}
                    for i, a in enumerate(strategy_res["action_plan"])
                ],
            }
            doc_paths = build_case_strategy(doc_data, base_name)
            strategy_res["filename"] = f"{base_name}.pdf"
            strategy_res["docx_filename"] = f"{base_name}.docx"
            strategy_res["pdf_path"] = doc_paths["pdf_path"]
            strategy_res["docx_path"] = doc_paths["docx_path"]
            strategy_res["base_filename"] = base_name
        except Exception:
            pass

        return strategy_res

    def _summary(self, domain_key: str, description: str, compensation) -> str:
        cap = (
            "your recoverable range is estimated at "
            f"Rs {compensation['min_amount']:,.0f} - Rs {compensation['max_amount']:,.0f}"
        )
        return (
            f"This is classified as a {domain_key.replace('_', ' ')} matter. "
            f"Based on the stated facts, {cap} under applicable statutory remedies, "
            "supported by the forum roadmap and BSA 2023 evidentiary checklist below."
        )

    @staticmethod
    def _assess(description: str, facts: Dict[str, float], compensation, incident_date: Optional[str] = None) -> tuple:
        strengths, weaknesses = [], []
        if any(
            k in facts
            for k in ("deposit_amount", "unpaid_amount", "amount", "loss_amount", "stolen_value", "cheated_amount", "medical_bills")
        ):
            strengths.append("There is a clear, quantifiable monetary claim that can be proved through banking/documentary records.")
        if incident_date or re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", description or ""):
            strengths.append("A specific incident date is identified, anchoring the statutory limitation period.")
        if compensation.get("max_amount", 0) > 0:
            strengths.append("The claimed relief is deterministically quantifiable for pleading before the forum.")
        if any(w in (description or "").lower() for w in ("witness", "receipt", "screenshot", "whatsapp", "cctv", "police", "fir", "agreement", "utr")):
            strengths.append("Corroborative documentary or electronic evidence is already referenced.")
        if not any(
            k in facts
            for k in ("deposit_amount", "unpaid_amount", "amount", "loss_amount", "stolen_value", "cheated_amount", "medical_bills")
        ):
            weaknesses.append("No liquidated monetary amount is stated — quantify your pecuniary loss before filing.")
        if not (incident_date or re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", description or "")):
            weaknesses.append("No exact incident date is stated — specify the date to verify statutory limitation compliance.")
        if len((description or "").strip()) < 60:
            weaknesses.append("Brief factual narrative — expand with chronology, party names, and prior notices sent.")
        return strengths, weaknesses
