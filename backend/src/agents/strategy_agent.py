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

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_DISCLAIMER = (
    "This strategy is auto-generated from the facts you provided for planning. "
    "It is not legal advice; a lawyer should confirm the law, forum and "
    "limitation period for your specific facts before you act."
)

_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|rupees?)\s*([\d,]+)"
    r"|\b([\d]{5,})\b"
    r"|\b([\d]{1,3},\d{3}(?:,\d{3})*)\b",
    re.I,
)

_DOMAIN_HINTS = [
    ("rent", ["landlord", "tenant", "rent", "deposit", "eviction", "lease", "premises", "flat", "apartment", "vacat"]),
    ("consumer", ["consumer", "bought", "purchased", "product", "refund", "warranty", "defective", "seller", "service", "replacement"]),
    ("labor", ["salary", "wage", "employer", "employee", "fired", "dismissed", "terminat", "layoff", "gratuity", "appointment letter", "bonus"]),
    ("criminal", ["stole", "stolen", "robbery", "cheated", "fraud", "scam", "assault", "attack", "hit", "murder", "threat", "theft"]),
    ("cyber", ["hacked", "hack", "cyber", "phishing", "data breach", "otp", "upi fraud", "instagram", "whatsapp", "ransomware"]),
    ("defamation", ["defamat", "defamed", "reputation", "slander", "libel", "false allegation"]),
    ("family", ["maintenance", "custody", "husband", "wife", "divorce", "alimony", "dowry", "children"]),
    ("commercial", ["contract", "breach", "invoice", "payment due", "supplier", "goods", "partnership", "debt", "interest"]),
]

# response_type / domain -> engine rule key
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

# engine rule key -> deadline case_type
_DEADLINE_MAP = {
    "consumer": "consumer",
    "labor": "labour_salary",
    "rent": "rent_eviction",
    "criminal": "criminal_fir",
}

_FORUMS = {
    "rent": ["Civil Court / Rent Controller (district where the property is)", "Consumer Commission if services of the landlord/agent are paid for"],
    "consumer": ["District Consumer Commission (2 years from deficiency, Consumer Protection Act 2019)"],
    "labor": ["Labour Commissioner / Claims Authority (wage claims under Code on Wages 2019)", "Labour Court / Industrial Tribunal (industrial disputes)"],
    "criminal": ["Police Station (FIR) - then Criminal Court", "You can also file a private criminal complaint with the Magistrate"],
    "cyber": ["Cyber Crime Portal (cybercrime.gov.in) and the police", "Complaint to your bank for reversal/chargeback"],
    "family": ["Family Court of your jurisdiction (maintenance / custody)"],
    "defamation": ["Criminal complaint (BNS 2023 Section 356) + Civil suit for damages"],
    "commercial": ["Civil Court (recovery/breach)", "Arbitration if the contract has an arbitration clause"],
    "civil": ["Civil Court of competent jurisdiction"],
}

_ACTIONS = {
    "rent": [
        "Send a formal legal notice demanding the deposit/rent relief and keep proof of service.",
        "Preserve the lease, deposit receipt, payment records and photos.",
        "If the landlord does not comply within 15 days, file in the Rent Controller / Civil Court.",
    ],
    "consumer": [
        "File a written complaint with the seller's customer care and keep the ticket/email.",
        "Send a formal legal notice demanding refund/replacement.",
        "Escalate to the District Consumer Commission within 2 years of the deficiency.",
    ],
    "labor": [
        "Raise the wage/compensation claim with the Labour Commissioner in writing.",
        "Collect appointment letter, salary slips and bank statements.",
        "For dismissal/lay-off, approach the Labour Court / Industrial Tribunal before limitation runs out.",
    ],
    "criminal": [
        "File an FIR at the police station (or e-FIR) immediately; get a signed copy.",
        "Preserve medical reports, photos, witnesses and CCTV.",
        "If police refuse, file a private complaint before the Magistrate under BNSS 2023.",
    ],
    "cyber": [
        "Report on the Cyber Crime portal (cybercrime.gov.in) within 48-72 hours.",
        "Freeze the transaction via your bank / UPI app immediately.",
        "Collect screenshots, transaction IDs and any communication as evidence.",
    ],
    "family": [
        "File a maintenance petition in the Family Court; interim maintenance can be sought.",
        "Keep proof of marriage, income, and any prior agreements.",
    ],
    "defamation": [
        "Document the exact statement, date, medium and reach (screenshots).",
        "Send a legal notice demanding retraction/apology.",
        "File a criminal complaint (BNS 2023 Section 356) and/or a civil suit for damages.",
    ],
    "commercial": [
        "Send a legal notice quantifying the loss and demanding payment.",
        "Invoke the contract's dispute-resolution clause (arbitration if present).",
        "Sue for recovery of the loss plus interest (Contract Act Section 73).",
    ],
    "civil": [
        "Assemble all documentary proof of the claim and its value.",
        "Send a legal notice demanding compliance/compensation.",
        "File the suit in the court with jurisdiction, within the limitation period.",
    ],
}


class CaseStrategyAgent:
    def __init__(self):
        self.engine = get_compensation_engine()
        self.evidence = EvidenceChecklistAgent()
        self.deadlines = DeadlineTrackerAgent()

    # ── Fact extraction (deterministic) ────────────────────────────────
    @staticmethod
    def _detect_domain(text: str) -> str:
        t = (text or "").lower()
        for domain, hints in _DOMAIN_HINTS:
            if any(h in t for h in hints):
                return domain
        return "civil"

    @staticmethod
    def _extract_amounts(text: str) -> Dict[str, float]:
        """Find rupee amounts with explicit markers; never year/date digits."""
        amounts: List[float] = []
        for group in _AMOUNT_RE.findall(text or ""):
            raw = next((g for g in group if g), "")
            try:
                val = float(raw.replace(",", ""))
            except ValueError:
                continue
            if not (100 <= val <= 1_000_000_000):
                continue
            if 1900 <= val <= 2100 and len(raw.replace(",", "")) == 4:
                continue  # a year, not an amount
            if val not in amounts:
                amounts.append(val)
        t = (text or "").lower()
        result: Dict[str, float] = {}
        if not amounts:
            return result
        if any(w in t for w in ("deposit", "advance")):
            result["deposit_amount"] = amounts[0]
        elif any(w in t for w in ("salary", "wage", "unpaid")):
            result["unpaid_amount"] = amounts[0]
        elif any(w in t for w in ("stole", "stolen", "robbed")):
            result["stolen_value"] = amounts[0]
        elif any(w in t for w in ("cheated", "fraud", "scam")):
            result["cheated_amount"] = amounts[0]
        elif any(w in t for w in ("medical", "hospital", "treatment", "injury")):
            result["medical_bills"] = amounts[0]
        else:
            result["amount"] = max(amounts)
        return result

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", text or "")
        if not m:
            return None
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # ── Strategy builder ───────────────────────────────────────────────
    def build(self, description: str, domain: str = None, incident_date: str = None,
              facts: Dict[str, Any] = None) -> Dict[str, Any]:
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
            deadline = self.deadlines.calculate_deadline(deadline_type, incident_date)

        checklist = self.evidence.get_checklist(domain_key)

        strengths, weaknesses = self._assess(description, facts, compensation)

        return {
            "domain": domain_key,
            "summary": self._summary(domain_key, description, compensation),
            "assessment": {"strengths": strengths, "weaknesses": weaknesses},
            "legal_route": {
                "criminal": domain_key in ("criminal", "cyber", "defamation"),
                "civil": True,
                "forums": _FORUMS.get(domain_key, _FORUMS["civil"]),
            },
            "compensation_estimate": compensation,
            "evidence_checklist": checklist,
            "deadline": deadline,
            "action_plan": _ACTIONS.get(domain_key, _ACTIONS["civil"]),
            "disclaimer": _DISCLAIMER,
        }

    def _summary(self, domain_key: str, description: str, compensation) -> str:
        cap = ("your recoverable range is estimated at "
               f"Rs {compensation['min_amount']:,.0f} - Rs {compensation['max_amount']:,.0f}")
        return (
            f"This looks like a {domain_key.replace('_', ' ')} matter. "
            f"Based on the facts, {cap} on a civil remedy, "
            "with the evidence and steps below to protect your position."
        )

    @staticmethod
    def _assess(description: str, facts: Dict[str, float], compensation) -> tuple:
        strengths, weaknesses = [], []
        if any(k in facts for k in ("deposit_amount", "unpaid_amount", "amount",
                                    "stolen_value", "cheated_amount", "medical_bills")):
            strengths.append("There is a clear monetary amount that can be proved.")
        if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}", description or ""):
            strengths.append("A specific date is mentioned, which helps fix the limitation start point.")
        if compensation.get("max_amount", 0) > 0:
            strengths.append("The claimed value is quantifiable, giving the court a definite relief to award.")
        if any(w in (description or "").lower() for w in ("witness", "receipt", "screenshot", "whatsapp", "cctv", "police", "fir")):
            strengths.append("Documentary/witness evidence is already mentioned.")
        if not any(k in facts for k in ("deposit_amount", "unpaid_amount", "amount",
                                        "stolen_value", "cheated_amount", "medical_bills")):
            weaknesses.append("No monetary amount stated - quantify your loss before filing.")
        if not re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}", description or ""):
            weaknesses.append("No incident date stated - the limitation period may be an issue; add it.")
        if len((description or "").strip()) < 60:
            weaknesses.append("Very brief description - add what happened, with whom, and the sequence.")
        return strengths, weaknesses


if __name__ == "__main__":
    agent = CaseStrategyAgent()
    case = ("My landlord Mr Sharma is not returning my Rs 50,000 security deposit after I "
            "vacated on 01-08-2024. He cut the water supply on 15-07-2024. I have the lease "
            "and WhatsApp messages where he admits cutting the water.")
    import json
    result = agent.build(case)
    print(json.dumps({
        "domain": result["domain"],
        "summary": result["summary"],
        "compensation": result["compensation_estimate"],
        "deadline": result["deadline"],
        "action_plan": result["action_plan"],
    }, ensure_ascii=False, indent=2))
