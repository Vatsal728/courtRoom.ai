"""
agent.py - Agentic Tool Layer for courtRoom.ai.
Supports the Top 6 Core Indian Legal Tools:
1. legal_notice - Legal Demand Notice (Landlord-tenant, breach of contract, money recovery)
2. rti_application - RTI Section 6 Form A Application
3. cyber_fraud - Cyber Fraud Emergency & Bank Account Freeze Requisition
4. police_complaint - Formal Written Police Complaint to SHO under BNSS 2023 / BNS 2023
5. labour_dues - Unpaid Salary, FnF Settlement & Wage Claim Recovery under Code on Wages 2019
6. case_strategy - Case Strategy Legal Brief & Roadmap
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.agents.notice_agent import LegalNoticeAgent
from src.agents.rti_agent import RTIApplicationAgent
from src.agents.cyber_fraud_agent import CyberFraudAgent
from src.agents.police_complaint_agent import PoliceComplaintAgent
from src.agents.labour_dues_agent import LabourDuesAgent
from src.agents.strategy_agent import CaseStrategyAgent
from src.document_factory import infer_bns_offence_sections, infer_notice_statutory_sections
from src.groq_drafter import normalize_tool_params
from src.session_manager import get_session_manager

load_dotenv(override=True)

# ── Tool Registry ────────────────────────────────────────────────────────
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "legal_notice": {
        "name": "legal_notice",
        "description": (
            "Draft a formal legal notice (written demand letter) against a landlord, tenant, business, "
            "seller, or person for issues like unpaid rent, security deposit refund, defective product, "
            "breach of agreement, or money recovery. Provides downloadable PDF and editable Word DOCX."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sender_name": {"type": "string", "description": "Full name of the sender"},
                "sender_address": {"type": "string", "description": "Address of the sender"},
                "sender_phone": {"type": "string"},
                "sender_email": {"type": "string"},
                "recipient_name": {"type": "string", "description": "Name of the recipient / opposite party"},
                "recipient_address": {"type": "string", "description": "Address of the recipient"},
                "recipient_email": {"type": "string"},
                "issue_type": {"type": "string", "description": "Dispute title, e.g. Security Deposit Refund"},
                "issue_description": {"type": "string", "description": "Facts: dates, amounts, what went wrong"},
                "applicable_section": {"type": "string", "description": "Law or section relied on"},
                "demand_amount": {"type": "string", "description": "Money demanded in rupees"},
            },
            "required": ["recipient_name", "issue_description"],
        },
    },
    "rti_application": {
        "name": "rti_application",
        "description": (
            "Draft a Right to Information (RTI) Act 2005 Section 6 Form A application asking a "
            "public authority (municipality, police, university, transport, health) for records. "
            "Provides downloadable official Form A PDF, editable Word DOCX, and submission text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pio_office": {"type": "string", "description": "Government office / public authority name"},
                "pio_address": {"type": "string", "description": "Address of the PIO office"},
                "information_sought": {"type": "string", "description": "The specific information or records requested"},
                "applicant_name": {"type": "string", "description": "Full name of the applicant"},
                "applicant_address": {"type": "string"},
                "applicant_phone": {"type": "string"},
                "applicant_email": {"type": "string"},
            },
            "required": ["information_sought"],
        },
    },
    "cyber_fraud": {
        "name": "cyber_fraud",
        "description": (
            "Draft an emergency Bank Account Freeze & Lien letter (under RBI circular guidelines) "
            "and 1930 National Cyber Crime Portal complaint for UPI fraud, fake QR codes, KYC scams, "
            "unauthorized bank debits, OTP scams, and phishing. Provides downloadable PDF and DOCX."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "victim_name": {"type": "string", "description": "Full name of the account holder"},
                "victim_phone": {"type": "string"},
                "victim_bank_name": {"type": "string", "description": "Bank name where victim holds account"},
                "victim_bank_branch": {"type": "string"},
                "account_number": {"type": "string"},
                "amount_lost": {"type": "string", "description": "Defrauded amount in rupees"},
                "transaction_id_utr": {"type": "string", "description": "UTR number or Transaction ID"},
                "beneficiary_upi_vpa_or_account": {"type": "string", "description": "Fraudulent receiving VPA or account"},
                "fraud_type": {"type": "string", "description": "Type of scam: UPI fraud, fake QR code, KYC scam, OTP scam, phishing"},
                "facts_summary": {"type": "string", "description": "Brief description of the scam"},
            },
            "required": ["amount_lost", "facts_summary"],
        },
    },
    "police_complaint": {
        "name": "police_complaint",
        "description": (
            "Draft a formal written complaint to the Police Station / Station House Officer (SHO) "
            "under Sections 173 / 175 of Bharatiya Nagarik Suraksha Sanhita (BNSS 2023) and BNS 2023 "
            "for snatching, theft, robbery, physical threats, cheating, harassment, stalking, or assault, "
            "complete with police station receiving counterfoil. Provides downloadable PDF and DOCX."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "complainant_name": {"type": "string", "description": "Full name of the complainant"},
                "complainant_address": {"type": "string"},
                "complainant_phone": {"type": "string"},
                "police_station_name": {"type": "string", "description": "Name of the local police station"},
                "incident_date_time": {"type": "string"},
                "incident_location": {"type": "string"},
                "accused_details": {"type": "string", "description": "Name/details of accused or 'Unknown persons'"},
                "incident_facts": {"type": "string", "description": "Full factual description of the criminal occurrence"},
                "applicable_section": {"type": "string", "description": "Operative BNS 2023 sections"},
                "evidence_available": {"type": "string", "description": "CCTV, WhatsApp chats, call recordings, witnesses"},
            },
            "required": ["incident_facts"],
        },
    },
    "labour_dues": {
        "name": "labour_dues",
        "description": (
            "Draft a formal Legal Demand Notice to Company HR & Directors and claim petition to "
            "District Labour Commissioner under Code on Wages 2019 & Payment of Wages Act 1936 "
            "for unpaid monthly salary, withheld Full & Final (FnF) settlement, gratuity, or illegal termination. "
            "Provides downloadable PDF and DOCX."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "employee_name": {"type": "string", "description": "Full name of the employee"},
                "designation": {"type": "string"},
                "employee_id": {"type": "string"},
                "employee_phone": {"type": "string"},
                "company_name": {"type": "string", "description": "Name of the employer company"},
                "office_address": {"type": "string"},
                "hr_email": {"type": "string"},
                "total_dues_amount": {"type": "string", "description": "Total pending salary or FnF in rupees"},
                "breakdown": {"type": "string", "description": "Breakdown: unpaid salary, leave encashment, gratuity, notice pay"},
                "last_working_day": {"type": "string"},
            },
            "required": ["company_name", "total_dues_amount"],
        },
    },
    "case_strategy": {
        "name": "case_strategy",
        "description": (
            "Build a step-by-step case strategy: legal route, forums, evidence checklist, "
            "filing deadlines and an ESTIMATED compensation range. Provides downloadable "
            "Legal Opinion Memo in PDF and DOCX, plus interactive chat view."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "case_description": {"type": "string", "description": "Full description: what happened, with whom, when, amounts, current status"},
                "incident_date": {"type": "string", "description": "Incident date in DD-MM-YYYY format, if known"},
                "domain": {"type": "string", "enum": ["rent", "consumer", "labor", "criminal", "cyber", "defamation", "family", "commercial", "civil"], "description": "The legal area, if clear"},
            },
            "required": ["case_description"],
        },
    },
}

# ── Layer 1: Deterministic Detection Regexes ─────────────────────────────
_GREET_WORDS = r"(hi+|hello+|hey+|namaste|namaskar|hola|yo|good\s*(morning|afternoon|evening))"

_SMALLTALK_RE = re.compile(
    r"^\s*" + _GREET_WORDS + r"(\s+(there|guys|everyone|dude|bro|friends?))?[!?.,\s]*$"
    r"|^\s*(who\s+are\s+you|what\s+are\s+you|how\s+are\s+you|what(can|do)\s+you\s+do|whats?\s*up|"
    r"are\s+you\s+(there|an?\s+ai|real|human)|what\s+is\s+your\s+(name|purpose)|who\s+made\s+you|"
    r"thank(s|\s+you|\s+you\s+so\s+much)|thanks\s+a\s+lot|thx|ok|okay|got\s+it|understand(ed)?|"
    r"bye|goodbye|see\s+you(\s+(later|soon|tomorrow))?|later|help(\s+me)?\s*$|how\s+can\s+you\s+help)\b[!?.,\s]*$"
    r"|^\s*(hi+|hello+|hey+|namaste|namaskar)([,\s.!]+)(\w+\s+)*how\s+are\s+you\b[^!?]*[!?.,\s]*$"
    r"|^\s*(how\s+are\s+you(\s+doing(\s+today)?)?|how\s+is\s+it\s+going|how\s+are\s+things)\b[!?.,\s]*$",
    re.I,
)

_NOTICE_RE = re.compile(r"\b(legal\s*(?:demand\s*)?notice|demand\s*(?:letter|notice))\b", re.I)
_RTI_RE = re.compile(r"\b(rti\b|right\s*to\s*information|form\s*a\s*application|pio\s*application)", re.I)
_CYBER_RE = re.compile(
    r"\b(cyber\s*fraud|upi\s*(?:fraud|scam)|online\s*scam|(?:bank|account)\s*freeze|freeze\s*notice|"
    r"phishing|debited\s*(?:fraudulently|from\s+my\s+.*?(?:bank|account|upi))|cyber\s*crime|fake\s*qr|"
    r"fraudster\s*(?:id|upi|vpa|account)|kyc\s*(?:scam|fraud|link)|utr\b|upi/[0-9]+)\b",
    re.I,
)
_POLICE_RE = re.compile(
    r"\b(police\s*complaint|complaint\s*(?:to|for)\s*(?:the\s*)?sho|fir\s*application|sho\s*complaint|"
    r"police\s*station|file\s*an?\s*fir|stolen\s*(?:phone|laptop|bike|chain|wallet)|theft\s*complaint|"
    r"snatched\s+my|chain\s*snatching)\b",
    re.I,
)
_LABOUR_RE = re.compile(
    r"\b(unpaid\s*salary|salary\s*(?:not\s*paid|dues)|not\s*released\s*my\s*.*?salary|fnf|full\s*and\s*final|"
    r"wage\s*claim|labour\s*(?:court|dues|commissioner)|fired\s*without\s*salary|withheld\s*salary|"
    r"pending\s*salary)\b",
    re.I,
)
_STRATEGY_RE = re.compile(
    r"\b(case\s*strategy|build\s*(?:me\s*)?a?\s*case\s*strategy|litigation\s*(?:strategy|roadmap)|"
    r"legal\s*strategy|draft\s*case\s*strategy)\b",
    re.I,
)

_DEFINITION_QUESTION_RE = re.compile(
    r"^\s*(what\s+is|what\s+are|define|meaning\s+of|how\s+does|how\s+do|what\s+do)\b", re.I
)

_TOOL_KEYWORD_RE = re.compile(
    r"\b(notice|demand\s*letter|rti|right\s*to\s*information|draft|generate|write|prepare|"
    r"application|form|complaint|letter|document|strategy|plan|roadmap|chances|steps?|"
    r"cyber|police|sho|fir|salary|dues|wages|recover|compensation|what\s+(should|can)\s+i\s+do)\b",
    re.I,
)


def detect_smalltalk(query: str) -> bool:
    return bool(_SMALLTALK_RE.match((query or "").strip()))


def detect_explicit_tool(query: str) -> Optional[str]:
    """Detect specific citizen legal tools from query text.
    Domain-specific tools (RTI, Cyber Fraud, Police Complaint, Labour Dues, Case Strategy)
    take priority over generic 'legal_notice' so 'Draft a legal demand notice for unpaid salary'
    or 'Draft a bank freeze notice' routes to the specialized statutory tool."""
    q = query or ""
    if _DEFINITION_QUESTION_RE.match(q.strip()):
        return None
    if _RTI_RE.search(q):
        return "rti_application"
    if _CYBER_RE.search(q):
        return "cyber_fraud"
    if _POLICE_RE.search(q):
        return "police_complaint"
    if _LABOUR_RE.search(q):
        return "labour_dues"
    if _STRATEGY_RE.search(q):
        return "case_strategy"
    if _NOTICE_RE.search(q):
        return "legal_notice"
    return None


def is_tool_relevant(query: str) -> bool:
    return bool(_TOOL_KEYWORD_RE.search(query or ""))


# ── Layer 2: Native Tool-Calling ─────────────────────────────────────────
def _tool_defs() -> List[Dict[str, Any]]:
    return [{"type": "function", "function": spec} for spec in TOOL_SPECS.values()]


_TOOL_ROUTER_PROMPT = (
    "You are an Indian legal assistant tool router. If the user clearly wants to generate a document or take action, "
    "call ONE appropriate tool from the available tools and fill its parameters from their message. "
    "If no tool applies (e.g. asking a pure legal explanation question), reply with a plain text answer without calling any tool."
)


def _parse_tool_calls(resp) -> Optional[Dict[str, Any]]:
    choices = getattr(resp, "choices", None)
    msg = getattr(choices[0], "message", None) if choices else getattr(resp, "message", None)
    calls = getattr(msg, "tool_calls", None)
    if not calls:
        return None
    tc = calls[0]
    fn = getattr(tc, "function", None)
    name = getattr(fn, "name", None)
    args_raw = getattr(fn, "arguments", None) or "{}"
    try:
        arguments = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw or {})
    except json.JSONDecodeError:
        arguments = {}
    if name and name in TOOL_SPECS:
        return {"name": name, "arguments": arguments}
    return None


def _detect_groq(query: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    if os.getenv("GROQ_ENABLED", "1") != "1" or not api_key:
        return None
    from groq import Groq

    client = Groq(api_key=api_key, max_retries=1)
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": _TOOL_ROUTER_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=_tool_defs(),
        tool_choice="auto",
        temperature=0.0,
        max_tokens=300,
        timeout=float(os.getenv("GROQ_GENERATION_TIMEOUT", "60")),
    )
    return _parse_tool_calls(resp)


def _detect_ollama(query: str) -> Optional[Dict[str, Any]]:
    import httpx

    payload = {
        "model": os.getenv("OLLAMA_GENERATION_MODEL", "qwen2.5:3b"),
        "messages": [
            {"role": "system", "content": _TOOL_ROUTER_PROMPT},
            {"role": "user", "content": query},
        ],
        "tools": _tool_defs(),
        "stream": False,
        "options": {"temperature": 0.0},
    }
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    with httpx.Client(timeout=float(os.getenv("OLLAMA_GENERATION_TIMEOUT", "60")), verify=False) as client:
        resp = client.post(f"{base}/api/chat", json=payload)
        if resp.status_code != 200:
            return None
        body = resp.json()
        message = body.get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            return None
        fn = calls[0].get("function") or {}
        name = fn.get("name")
        if name and name in TOOL_SPECS:
            arguments = fn.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            return {"name": name, "arguments": arguments}
    return None


def detect_tool_call(query: str) -> Optional[Dict[str, Any]]:
    try:
        hit = _detect_groq(query)
        if hit:
            return hit
    except Exception:
        pass
    try:
        return _detect_ollama(query)
    except Exception:
        return None


# ── Dynamic Indian Currency & Entity Extraction ──────────────────────────
# Supports Indian comma grouping (1,40,000 / 2,50,000), Western grouping (100,000),
# plain digits (50000), decimals (1.5), and multipliers (lakh, lac, crore, thousand, k).
_INDIAN_MONEY_PATTERN = re.compile(
    r"(?:(?:rs\.?|inr|rupees?|₹)\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(lakh|lac|crore|cr|thousand|k)?\b"
    r"|"
    r"\b(\d[\d,]*(?:\.\d+)?)\s*(lakh|lac|crore|cr|thousand|k)?\s*(?:rs\.?|inr|rupees?|/-)\b)",
    re.I,
)

_MULTIPLIERS = {
    "lakh": 100_000,
    "lac": 100_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "thousand": 1_000,
    "k": 1_000,
}


def format_indian_Grouping(val: float) -> str:
    """Format a numeric rupee value using Indian comma grouping (e.g. 175000 -> '1,75,000')."""
    n = int(round(val))
    s = str(abs(n))
    if len(s) <= 3:
        res = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        groups.reverse()
        res = ",".join(groups) + "," + last3
    return f"-{res}" if n < 0 else res


def parse_indian_rupee_amounts(text: str) -> List[Tuple[float, str]]:
    """Extract all (numeric_value, formatted_string) rupee amounts from text."""
    if not text:
        return []
    results: List[Tuple[float, str]] = []
    for m in _INDIAN_MONEY_PATTERN.finditer(text):
        num_str = m.group(1) or m.group(3) or ""
        unit = (m.group(2) or m.group(4) or "").lower().rstrip("s")
        try:
            val = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if unit in _MULTIPLIERS:
            val *= _MULTIPLIERS[unit]
        if val > 0:
            results.append((val, format_indian_Grouping(val)))
    return results


def _extract_rupees(text: str, sum_multiple: bool = False) -> str:
    """Extract rupee amount from text. When sum_multiple=True and multiple distinct
    monetary heads are stated (e.g. 'salary of Rs 1,40,000 and gratuity of Rs 35,000'),
    returns their formatted sum ('1,75,000')."""
    parsed = parse_indian_rupee_amounts(text)
    if not parsed:
        return ""
    if sum_multiple and len(parsed) > 1:
        total = sum(v for v, _ in parsed)
        return format_indian_Grouping(total)
    return parsed[0][1]


_PHONE_RE = re.compile(r"(?<![/A-Za-z0-9_-])(?:\+91[\s-]?)?([6-9]\d{9})\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_UPI_VPA_RE = re.compile(r"\b([a-zA-Z0-9._-]+@[a-zA-Z0-9]{2,20})\b")
_UTR_RE = re.compile(
    r"\b(?:UPI/[A-Za-z0-9/-]{4,30}|UTR[\s:#-]*([A-Za-z0-9/-]{6,30})|(?:transaction\s*(?:id|ref(?:erence)?)|ref(?:erence)?\s*(?:no|id|number)?)\s*(?:is|[:#-])?\s*([A-Za-z0-9/-]{5,30}))\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?)\b",
    re.I,
)

_NOTICE_ISSUE_TYPES = [
    (("deposit", "security deposit"), "Security Deposit Refund"),
    (("rent", "tenant", "arrears", "lease"), "Arrears of Rent & Tenancy Dispute"),
    (("salary", "wage", "unpaid", "fnf", "gratuity"), "Non-payment of Salary & Dues"),
    (("evict", "eviction"), "Illegal Eviction"),
    (("product", "defective", "warranty", "replacement"), "Defective Product & Consumer Refund"),
    (("loan", "repay", "borrowed", "lent"), "Recovery of Friendly / Personal Loan"),
    (("cheque", "bounce", "dishonour"), "Dishonour of Cheque (Sec 138 NI Act)"),
    (("breach", "contract", "agreement", "advance"), "Breach of Contract & Recovery"),
]


def _is_bare_intent_prompt(q: str) -> bool:
    """Return True when the query is just a bare request to trigger a tool without substantive facts
    (e.g. 'I want to send a legal notice.' or 'Draft a legal notice')."""
    cleaned = re.sub(
        r"\b(i|want|would|like|to|please|can|you|help|me|send|draft|make|create|prepare|write|file|generate|a|an|the|formal|legal|demand|notice|letter|application|complaint|rti)\b",
        " ",
        (q or "").lower(),
    )
    cleaned = re.sub(r"[^a-z0-9]", " ", cleaned).strip()
    return len(cleaned.split()) <= 2


def _fill_notice(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("sender_name") or "").strip():
        if str(profile.get("name") or "").strip():
            args["sender_name"] = profile["name"]
        else:
            m = re.search(
                r"(?:sender\s+is|my\s*name\s*is|i\s*am)\s+(?:[Mm][Rr][Ss]?\.?\s*|[Dd][Rr]\.?\s*)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
                q,
            )
            if m:
                args["sender_name"] = re.sub(
                    r"\s+\b(and|the|with|to|landlord|tenant|against|for|who)\b.*$",
                    "",
                    m.group(1),
                    flags=re.I,
                ).strip()
    if not str(args.get("sender_email") or "").strip():
        if str(profile.get("email") or "").strip():
            args["sender_email"] = profile["email"]
    if not str(args.get("recipient_name") or "").strip():
        m = re.search(
            r"(?i:to|against|for|towards|landlord\s*(?:is|:)?|tenant\s*(?:is|:)?|recipient\s*(?:is|:)?)\s+"
            r"(?:(?:my|the|our)\s+(?:landlord|tenant|employer|builder|seller|friend|client|neighbour|neighbor|contractor|vendor)\s+)?"
            r"((?:(?i:mr|mrs|ms|dr|shri|smt)\.?\s*)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
            q,
        )
        if m:
            recip = re.sub(r"^(?i:landlord|tenant|employer|builder|seller)\s+", "", m.group(1).strip(), flags=re.I)
            recip = re.sub(
                r"\s+\b(and|the|with|to|against|for|demanding|who|because|regarding|on)\b.*$",
                "",
                recip,
                flags=re.I,
            ).strip()
            if recip.lower() not in ("send", "draft", "write", "file", "prepare", "legal", "notice"):
                args["recipient_name"] = recip
    if not str(args.get("demand_amount") or "").strip():
        amt = _extract_rupees(q, sum_multiple=True)
        if amt:
            args["demand_amount"] = amt
    if not str(args.get("issue_type") or "").strip():
        low = q.lower()
        for kws, label in _NOTICE_ISSUE_TYPES:
            if any(k in low for k in kws):
                args["issue_type"] = label
                break
    if not str(args.get("issue_description") or "").strip() and not _is_bare_intent_prompt(q):
        m = re.search(r"(?:issue\s+is\s*[:,-]?\s*|about\s+|regarding\s+|because\s*[:,-]?\s*|who\s+has\s+|for\s+(?:not\s+|failing\s+))(.+)$", q, re.I)
        if m and len(m.group(1).strip()) > 10:
            args["issue_description"] = q.strip()[:1200]
        elif len(q.strip()) > 25:
            args["issue_description"] = q.strip()[:1200]
    if str(args.get("issue_description") or "").strip() and not str(args.get("applicable_section") or "").strip():
        args["applicable_section"] = infer_notice_statutory_sections(
            args.get("issue_type", ""), args.get("issue_description", "")
        )
    return args


def _fill_rti(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("applicant_name") or "").strip():
        if str(profile.get("name") or "").strip():
            args["applicant_name"] = profile["name"]
        else:
            m = re.search(
                r"(?:applicant|my\s+name\s+is)\s+((?:mr|mrs|ms|dr)\.?\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
                q,
                re.I,
            )
            if m:
                args["applicant_name"] = (m.group(1) or "") + m.group(2)
    if not str(args.get("applicant_email") or "").strip():
        if str(profile.get("email") or "").strip():
            args["applicant_email"] = profile["email"]
        else:
            m = _EMAIL_RE.search(q)
            if m:
                args["applicant_email"] = m.group(0)
    if not str(args.get("pio_office") or "").strip():
        m = re.search(
            r"(?:to\s+the\s+|to\s+|from\s+the\s+|from\s+)(?!get\s|ask\s|receive\s|obtain\s|file\s|submit\s|request\s)([A-Za-z][A-Za-z\s&'-]{3,60}?)(?=$|\s+(?:for|about|regarding|seeking|,))",
            q,
            re.I,
        )
        if m:
            args["pio_office"] = m.group(1).strip().title()
    if not str(args.get("information_sought") or "").strip() and not _is_bare_intent_prompt(q):
        m = re.search(
            r"(?:for|about|regarding|seeking|requesting|asking\s+for)\s+(.+?)(?=(?:,?\s*(?:applicant|phone|contact|email)\b)|$)",
            q,
            re.I,
        )
        if m:
            args["information_sought"] = m.group(1).strip()[:1200]
        elif len(q.strip()) > 25:
            args["information_sought"] = q.strip()[:1200]
    return args


def _fill_cyber(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("victim_name") or "").strip():
        args["victim_name"] = profile.get("name", "") or "Aggrieved Account Holder"
    if not str(args.get("victim_email") or "").strip():
        args["victim_email"] = profile.get("email", "")
    if not str(args.get("victim_phone") or "").strip():
        pm = _PHONE_RE.search(q)
        if pm:
            args["victim_phone"] = pm.group(1)
    if not str(args.get("amount_lost") or "").strip():
        amt = _extract_rupees(q, sum_multiple=True)
        if amt:
            args["amount_lost"] = amt
    if not str(args.get("victim_bank_name") or "").strip():
        m = re.search(
            r"\b(sbi|state\s*bank\s*of\s*india|hdfc|icici|axis|pnb|punjab\s*national|bob|bank\s*of\s*baroda|kotak|canara|union\s*bank|indusind|yes\s*bank|idfc|federal\s*bank)\b",
            q,
            re.I,
        )
        if m:
            raw_bank = m.group(0).upper()
            args["victim_bank_name"] = raw_bank if "BANK" in raw_bank else f"{raw_bank} Bank"
    if not str(args.get("beneficiary_upi_vpa_or_account") or "").strip():
        vpa_m = _UPI_VPA_RE.search(q)
        if vpa_m:
            args["beneficiary_upi_vpa_or_account"] = vpa_m.group(1)
    if not str(args.get("transaction_id_utr") or "").strip():
        utr_m = _UTR_RE.search(q)
        if utr_m:
            args["transaction_id_utr"] = (utr_m.group(1) or utr_m.group(2) or utr_m.group(0)).strip()
    if not str(args.get("fraud_type") or "").strip():
        low = q.lower()
        if "kyc" in low:
            args["fraud_type"] = "KYC Update / Phishing Link UPI Fraud (Sec 66D IT Act & Sec 318(4) BNS 2023)"
        elif "qr" in low:
            args["fraud_type"] = "Fake QR Code UPI Scam (Sec 66D IT Act & Sec 318(4) BNS 2023)"
        elif "otp" in low:
            args["fraud_type"] = "OTP Vishing / Unauthorized Electronic Debit (Sec 43/66D IT Act)"
        else:
            args["fraud_type"] = "Unauthorized UPI / Electronic Banking Fraud (Sec 43/66D IT Act & Sec 318(4) BNS 2023)"
    if not str(args.get("facts_summary") or "").strip() and not _is_bare_intent_prompt(q):
        args["facts_summary"] = q.strip()[:1200]
    return normalize_tool_params("cyber_fraud", args)


def _fill_police(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("complainant_name") or "").strip():
        args["complainant_name"] = profile.get("name", "") or "Complainant"
    if not str(args.get("complainant_phone") or "").strip():
        m = _PHONE_RE.search(q)
        if m:
            args["complainant_phone"] = m.group(1)
    if not str(args.get("police_station_name") or "").strip():
        m = re.search(r"\b(?:at|to|in|before)\s+([A-Z][A-Za-z\s]{2,25}?\s*police\s*station)\b", q, re.I)
        if not m:
            m = re.search(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+police\s*station)\b", q, re.I)
        if m:
            args["police_station_name"] = m.group(1).strip().title()
    if not str(args.get("incident_location") or "").strip():
        loc_m = re.search(
            r"\b(?:near|at|outside|in\s+front\s+of|around)\s+([A-Z][A-Za-z0-9\s,.-]{2,40}?)(?=\s+(?:and\s+fled|on\s+\d|yesterday|today|last\s+night|where|when|\.|,|$))",
            q,
            re.I,
        )
        if loc_m:
            loc_str = loc_m.group(1).strip()
            if "police station" not in loc_str.lower():
                args["incident_location"] = loc_str.title()
                if not str(args.get("police_station_name") or "").strip():
                    first_place = loc_str.split()[0].title()
                    args["police_station_name"] = f"{first_place} Police Station"
    if not str(args.get("accused_details") or "").strip():
        acc_m = re.search(
            r"^((?:two|three|four|some|unknown|\d+)\s+(?:men|persons|individuals|boys|bikers|people)[^.]{0,60}?)(?=\s+(?:snatched|stole|robbed|attacked|assaulted|threatened|cheated|hit|took)\b)",
            q.strip(),
            re.I,
        )
        if acc_m:
            args["accused_details"] = acc_m.group(1).strip()
    if not str(args.get("incident_facts") or "").strip() and not _is_bare_intent_prompt(q):
        args["incident_facts"] = q.strip()[:1200]
    if str(args.get("incident_facts") or "").strip() and not str(args.get("applicable_section") or "").strip():
        args["applicable_section"] = infer_bns_offence_sections(args["incident_facts"])
    return normalize_tool_params("police_complaint", args)


def _fill_labour(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("employee_name") or "").strip():
        if str(profile.get("name") or "").strip():
            args["employee_name"] = profile["name"]
        else:
            m = re.search(
                r"(?:employee\s*name\s*is|my\s*name\s*is|i\s*am)\s+(?:[Mm][Rr][Ss]?\.?\s*|[Dd][Rr]\.?\s*)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
                q,
            )
            if m and m.group(1).lower() not in ("resigned", "working", "employed", "terminated", "fired"):
                args["employee_name"] = m.group(1).strip()

    # Extract company_name dynamically (supports "resigned from TechCorp Pvt Ltd", "employer Infotech", "at Apex Tech Solutions Pvt Ltd", etc.)
    if not str(args.get("company_name") or "").strip():
        patterns = [
            r"(?:resigned\s+from|working\s+(?:at|with|for)|employed\s+(?:at|with|by)|terminated\s+(?:by|from)|fired\s+(?:by|from)|joined)\s+([A-Z][A-Za-z0-9&.,'-]*(?:\s+[A-Z][A-Za-z0-9&.,'-]*){0,5})",
            r"(?:my\s+)?(?:company|employer|firm|startup|organization)\s+(?:is\s+|named\s+)?([A-Z][A-Za-z0-9&.,'-]*(?:\s+[A-Z][A-Za-z0-9&.,'-]*){0,5}?)(?=\s+(?:and|is|has|have|not|refusing|withheld|owes|on|since|for|where|which|who|,|\.)|$)",
            r"(?:against|from|at|with)\s+([A-Z][A-Za-z0-9&.,'-]+(?:\s+(?:Pvt\.?|Private|Ltd\.?|Limited|LLP|Inc\.?|Corp(?:oration)?|Tech(?:nologies)?|Solutions|Services|Systems|India|[A-Z][a-zA-Z0-9&.-]+)){0,4})",
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                cand = re.sub(
                    r"\s+\b(on|since|from|in|who|which|has|have|is|are|not|and|for|after)\b.*$",
                    "",
                    m.group(1).strip(),
                    flags=re.I,
                ).strip(" .,")
                if cand and cand.lower() not in ("August", "September", "October", "January", "February", "March", "April", "May", "June", "July", "November", "December"):
                    args["company_name"] = cand
                    break

    # Extract and sum multiple monetary heads (e.g. Rs 1,40,000 salary + Rs 35,000 gratuity = Rs 1,75,000)
    amounts = parse_indian_rupee_amounts(q)
    if not str(args.get("total_dues_amount") or "").strip():
        if amounts:
            total_val = sum(v for v, _ in amounts)
            args["total_dues_amount"] = format_indian_Grouping(total_val)
    if not str(args.get("breakdown") or "").strip() and len(amounts) > 1:
        parts = []
        low = q.lower()
        if "salary" in low:
            parts.append(f"Unpaid salary (Rs. {amounts[0][1]}/-)")
        if "gratuity" in low and len(amounts) >= 2:
            parts.append(f"FnF Gratuity (Rs. {amounts[1][1]}/-)")
        elif len(amounts) >= 2:
            parts.append(f"Additional FnF / settlement dues (Rs. {amounts[1][1]}/-)")
        if parts:
            args["breakdown"] = " + ".join(parts)
    if not str(args.get("last_working_day") or "").strip():
        dm = _DATE_RE.search(q)
        if dm:
            args["last_working_day"] = dm.group(1).strip()

    return normalize_tool_params("labour_dues", args)


def _fill_strategy(args: Dict[str, Any], q: str) -> Dict[str, Any]:
    args = dict(args)
    if not str(args.get("case_description") or "").strip() and not _is_bare_intent_prompt(q):
        args["case_description"] = q.strip()
    if not str(args.get("incident_date") or "").strip():
        m = _DATE_RE.search(q)
        if m:
            args["incident_date"] = m.group(1)
    return args


def _backfill_args(
    name: str,
    args: Dict[str, Any],
    query: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = args or {}
    if name == "legal_notice":
        return _fill_notice(args, query or "", profile)
    if name == "rti_application":
        return _fill_rti(args, query or "", profile)
    if name == "cyber_fraud":
        return _fill_cyber(args, query or "", profile)
    if name == "police_complaint":
        return _fill_police(args, query or "", profile)
    if name == "labour_dues":
        return _fill_labour(args, query or "", profile)
    if name == "case_strategy":
        return _fill_strategy(args, query or "")
    return args


# ── Multi-Turn Continuation / Answering ──────────────────────────────────
_LEADING_STOPWORDS = re.compile(
    r"^(he|she|i|we|they|it|the|my|our|your|his|her|its|their|rent|landlord|tenant|sir|madam|"
    r"this|that|there|me|you|us|send|draft|write)\b",
    re.I,
)


def _extract_recipient_name(answer: str, sender_stmt: bool) -> str:
    if not answer:
        return ""
    m = re.search(
        r"(?i:to|against|for|towards|landlord\s*(?:is|:)?|tenant\s*(?:is|:)?|recipient\s*(?:is|:)?|party\s*(?:is|:)?)\s+"
        r"(?:(?:my|the|our)\s+(?:landlord|tenant|employer|builder|seller|friend|client|neighbour|neighbor|contractor|vendor)\s+)?"
        r"((?:(?i:mr|mrs|ms|dr|shri|smt)\.?\s*)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
        answer,
    )
    if m:
        recip = re.sub(r"^(?i:landlord|tenant|employer|builder|seller)\s+", "", m.group(1).strip(), flags=re.I)
        recip = re.sub(r"\s+\b(and|the|with|to|against|for|demanding|who|because|on|in)\b.*$", "", recip, flags=re.I).strip()
        if recip and recip.lower() not in ("send", "draft", "write", "it"):
            return recip
    if sender_stmt or _LEADING_STOPWORDS.match(answer):
        return ""
    m = re.match(r"^((?:(?:mr|mrs|ms|dr|shri|smt)\.?\s*)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})(?=[,;:\s]|$)", answer)
    return m.group(1) if m else ""


def merge_tool_answer(
    name: str,
    args: Dict[str, Any],
    answer: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dynamically merge user's follow-up turn into pending session parameters."""
    args = dict(args or {})
    answer = (answer or "").strip()
    if not answer:
        return args
    profile = profile or {}

    if name == "legal_notice":
        sender_stmt = bool(re.search(r"(?:my\s+name\s+is|i\s*'?m\s+(?:called\s+)?|call\s+me)", answer, re.I))
        if not str(args.get("recipient_name") or "").strip():
            recipient = _extract_recipient_name(answer, sender_stmt)
            if recipient:
                args["recipient_name"] = recipient
        if not str(args.get("sender_name") or "").strip():
            if str(profile.get("name") or "").strip():
                args["sender_name"] = profile["name"]
            else:
                m = re.search(
                    r"(?:my\s+name\s+is|i\s*'?m\s+(?:called\s+)?|call\s+me)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
                    answer,
                    re.I,
                )
                if m:
                    args["sender_name"] = re.sub(
                        r"\s+\b(and|the|with|to|landlord|tenant|against|for)\b.*$",
                        "",
                        m.group(1),
                        flags=re.I,
                    ).strip()
        if not str(args.get("demand_amount") or "").strip():
            amt = _extract_rupees(answer, sum_multiple=True)
            if amt:
                args["demand_amount"] = amt

        # Replace empty or bare-intent issue_description with the substantive facts from the follow-up turn
        existing_desc = str(args.get("issue_description") or "").strip()
        if (not existing_desc or _is_bare_intent_prompt(existing_desc)) and not sender_stmt:
            args["issue_description"] = answer[:1200]

        # Dynamically update issue_type and applicable_section from the merged facts
        low = f"{args.get('issue_description', '')} {answer}".lower()
        if not str(args.get("issue_type") or "").strip():
            for kws, label in _NOTICE_ISSUE_TYPES:
                if any(k in low for k in kws):
                    args["issue_type"] = label
                    break
        args["applicable_section"] = infer_notice_statutory_sections(
            args.get("issue_type", ""),
            args.get("issue_description", ""),
            args.get("applicable_section", ""),
        )
        return args

    if name == "rti_application":
        return _fill_rti(args, answer, profile)

    if name == "cyber_fraud":
        return _fill_cyber(args, answer, profile)

    if name == "police_complaint":
        return _fill_police(args, answer, profile)

    if name == "labour_dues":
        return _fill_labour(args, answer, profile)

    if name == "case_strategy":
        return _fill_strategy(args, answer)

    return args


# ── Tool Execution Functions ─────────────────────────────────────────────
def execute_legal_notice(
    args: Dict[str, Any],
    user_id: str,
    storage,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = normalize_tool_params("legal_notice", args)
    profile = profile or {}
    if not str(args.get("sender_name") or "").strip():
        args["sender_name"] = profile.get("name", "")
    # Treat bare intent strings as missing issue_description
    if _is_bare_intent_prompt(str(args.get("issue_description") or "")):
        args["issue_description"] = ""

    critical = ["recipient_name", "issue_description"]
    missing = [c for c in critical if not str(args.get(c) or "").strip()]
    if missing:
        asks = []
        if "recipient_name" in missing:
            asks.append("Who should the notice be sent to? (Name, and address if available)")
        if "issue_description" in missing:
            asks.append("What happened? Briefly describe the facts (dates, amounts, what went wrong)")
        return {
            "type": "legal_notice",
            "status": "needs_input",
            "title": "Legal Notice",
            "message": " ".join(asks),
            "missing_fields": missing,
            "params": args,
        }

    if not str(args.get("sender_name") or "").strip():
        args["sender_name"] = "Aggrieved Party"
    args["applicable_section"] = infer_notice_statutory_sections(
        args.get("issue_type", ""),
        args.get("issue_description", ""),
        args.get("applicable_section", ""),
    )

    res = LegalNoticeAgent().generate_both(args)
    pdf_id = storage.store_pdf_notice(
        user_id,
        "chat_tool",
        {
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "pdf_path": res["pdf_path"],
            "docx_path": res["docx_path"],
            "sender_name": args.get("sender_name", ""),
            "sender_email": args.get("sender_email", ""),
            "recipient_name": args.get("recipient_name", ""),
            "recipient_email": args.get("recipient_email", ""),
            "issue_type": args.get("issue_type", "Legal Dispute"),
            "demand_amount": args.get("demand_amount", ""),
        },
    )
    sender = args.get("sender_name", "") or "Sender"
    recipient = args.get("recipient_name", "") or "Recipient"
    sec = args.get("applicable_section", "")
    return {
        "type": "legal_notice",
        "status": "ready",
        "title": "Legal Notice drafted",
        "message": (
            f"Formal Legal Notice from {sender} to {recipient} under {sec} "
            f"is ready to download in PDF or editable Word (.docx)."
        ),
        "pdf_id": pdf_id,
        "filename": res["filename"],
        "docx_filename": res["docx_filename"],
        "download_url": f"/download/{pdf_id}?format=pdf",
        "docx_download_url": f"/download/{pdf_id}?format=docx",
        "params": args,
    }


def execute_rti_application(
    args: Dict[str, Any],
    user_id: str,
    storage,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not str(args.get("information_sought") or "").strip():
        asks = ["Which specific information or records do you want to request?"]
        if not str(args.get("pio_office") or "").strip():
            asks.append("Which government office/department should this be addressed to?")
        return {
            "type": "rti_application",
            "status": "needs_input",
            "title": "RTI Application",
            "message": " ".join(asks),
            "missing_fields": ["information_sought"],
            "params": args,
        }

    res = RTIApplicationAgent().generate_both(args)
    pdf_id = storage.store_pdf_notice(
        user_id,
        "chat_tool",
        {
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "pdf_path": res["pdf_path"],
            "docx_path": res["docx_path"],
            "sender_name": args.get("applicant_name", ""),
            "recipient_name": args.get("pio_office", ""),
            "issue_type": "RTI Application (Section 6)",
            "demand_amount": "Rs 10 application fee",
        },
    )
    office = args.get("pio_office") or "the public authority"
    return {
        "type": "rti_application",
        "status": "ready",
        "title": "RTI Application (Form A) drafted",
        "message": f"RTI Application to {office} (Section 6(1) & 6(3), RTI Act 2005) is ready to download in printable PDF or Word (.docx).",
        "pdf_id": pdf_id,
        "filename": res["filename"],
        "docx_filename": res["docx_filename"],
        "download_url": f"/download/{pdf_id}?format=pdf",
        "docx_download_url": f"/download/{pdf_id}?format=docx",
        "application": res["application_text"],
        "params": args,
    }


def execute_cyber_fraud(
    args: Dict[str, Any],
    user_id: str,
    storage,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = normalize_tool_params("cyber_fraud", args)
    profile = profile or {}
    if not str(args.get("victim_name") or "").strip():
        args["victim_name"] = profile.get("name", "") or "Aggrieved Account Holder"
    critical = ["amount_lost", "facts_summary"]
    missing = [c for c in critical if not str(args.get(c) or "").strip()]
    if missing:
        asks = []
        if "amount_lost" in missing:
            asks.append("How much money was defrauded/debited?")
        if "facts_summary" in missing:
            asks.append("Briefly explain how the fraud occurred (e.g. UPI QR code, fake KYC link, unauthorized debit).")
        return {
            "type": "cyber_fraud",
            "status": "needs_input",
            "title": "Cyber Fraud Emergency",
            "message": " ".join(asks),
            "missing_fields": missing,
            "params": args,
        }

    res = CyberFraudAgent().generate_filing(args)
    pdf_id = storage.store_pdf_notice(
        user_id,
        "chat_tool",
        {
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "pdf_path": res["pdf_path"],
            "docx_path": res["docx_path"],
            "sender_name": args.get("victim_name", ""),
            "recipient_name": args.get("victim_bank_name", "Bank"),
            "issue_type": "Cyber Fraud & Account Freeze",
            "demand_amount": args.get("amount_lost", ""),
        },
    )
    amount = args.get("amount_lost", "0")
    bank = args.get("victim_bank_name") or "your bank"
    return {
        "type": "cyber_fraud",
        "status": "ready",
        "title": "Cyber Fraud & Bank Freeze Notice ready",
        "message": (
            f"Emergency Bank Freeze Requisition to {bank} for Rs. {amount}/- (under RBI Circular "
            f"DBR.No.Leg.BC.78/09.07.005/2017-18, Section 43/66D IT Act 2000 & Section 318(4) BNS 2023) "
            f"is ready to download in PDF and Word (.docx). Immediately call 1930 and submit this notice to your branch."
        ),
        "pdf_id": pdf_id,
        "filename": res["filename"],
        "docx_filename": res["docx_filename"],
        "download_url": f"/download/{pdf_id}?format=pdf",
        "docx_download_url": f"/download/{pdf_id}?format=docx",
        "params": args,
    }


def execute_police_complaint(
    args: Dict[str, Any],
    user_id: str,
    storage,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = normalize_tool_params("police_complaint", args)
    profile = profile or {}
    if not str(args.get("complainant_name") or "").strip():
        args["complainant_name"] = profile.get("name", "") or "Complainant"
    if not str(args.get("incident_facts") or "").strip():
        return {
            "type": "police_complaint",
            "status": "needs_input",
            "title": "Police Complaint (BNSS 2023)",
            "message": "Please describe what incident happened and which police station or location it pertains to.",
            "missing_fields": ["incident_facts"],
            "params": args,
        }

    if not str(args.get("applicable_section") or "").strip():
        args["applicable_section"] = infer_bns_offence_sections(args.get("incident_facts", ""))

    res = PoliceComplaintAgent().generate_complaint(args)
    pdf_id = storage.store_pdf_notice(
        user_id,
        "chat_tool",
        {
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "pdf_path": res["pdf_path"],
            "docx_path": res["docx_path"],
            "sender_name": args.get("complainant_name", ""),
            "recipient_name": args.get("police_station_name", "Police Station"),
            "issue_type": "Written Police Complaint (BNSS 2023)",
            "demand_amount": "",
        },
    )
    ps = args.get("police_station_name") or "the local police station"
    bns_sec = args.get("applicable_section", "BNS 2023")
    return {
        "type": "police_complaint",
        "status": "ready",
        "title": "Police Complaint to SHO ready",
        "message": (
            f"Formal written complaint to SHO ({ps}) under Section 173 BNSS 2023 and {bns_sec} "
            f"with mandatory receiving counterfoil is ready to download in PDF and Word (.docx)."
        ),
        "pdf_id": pdf_id,
        "filename": res["filename"],
        "docx_filename": res["docx_filename"],
        "download_url": f"/download/{pdf_id}?format=pdf",
        "docx_download_url": f"/download/{pdf_id}?format=docx",
        "params": args,
    }


def execute_labour_dues(
    args: Dict[str, Any],
    user_id: str,
    storage,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = normalize_tool_params("labour_dues", args)
    profile = profile or {}
    if not str(args.get("employee_name") or "").strip():
        args["employee_name"] = profile.get("name", "") or "Aggrieved Employee"
    critical = ["company_name", "total_dues_amount"]
    missing = [c for c in critical if not str(args.get(c) or "").strip()]
    if missing:
        asks = []
        if "company_name" in missing:
            asks.append("What is the name of your employer / company?")
        if "total_dues_amount" in missing:
            asks.append("What is the total pending salary or FnF dues amount?")
        return {
            "type": "labour_dues",
            "status": "needs_input",
            "title": "Salary & Dues Recovery",
            "message": " ".join(asks),
            "missing_fields": missing,
            "params": args,
        }

    res = LabourDuesAgent().generate_demand(args)
    pdf_id = storage.store_pdf_notice(
        user_id,
        "chat_tool",
        {
            "filename": res["filename"],
            "docx_filename": res["docx_filename"],
            "pdf_path": res["pdf_path"],
            "docx_path": res["docx_path"],
            "sender_name": args.get("employee_name", ""),
            "recipient_name": args.get("company_name", ""),
            "issue_type": "Salary & Labour Dues Claim",
            "demand_amount": args.get("total_dues_amount", ""),
        },
    )
    comp = args.get("company_name", "the employer")
    amt = args.get("total_dues_amount", "0")
    return {
        "type": "labour_dues",
        "status": "ready",
        "title": "Salary Demand Notice ready",
        "message": (
            f"Legal demand notice to {comp} for Rs. {amt}/- under Sections 17 & 18 of the Code on Wages, 2019 "
            f"and Section 15 of the Payment of Wages Act, 1936 (with 7-day cure deadline) is ready to download in PDF and DOCX."
        ),
        "pdf_id": pdf_id,
        "filename": res["filename"],
        "docx_filename": res["docx_filename"],
        "download_url": f"/download/{pdf_id}?format=pdf",
        "docx_download_url": f"/download/{pdf_id}?format=docx",
        "params": args,
    }


def execute_case_strategy(args: Dict[str, Any], user_id: str, storage) -> Dict[str, Any]:
    description = str(args.get("case_description") or "").strip()
    if not description:
        return {
            "type": "case_strategy",
            "status": "needs_input",
            "title": "Case Strategy",
            "message": "Describe your situation in a few sentences - what happened, with whom, when - and I'll build a strategy.",
            "missing_fields": ["case_description"],
            "params": args,
        }
    auto = CaseStrategyAgent._detect_domain(description)
    domain_arg = args.get("domain")
    domain = auto if auto != "civil" else domain_arg
    strategy = CaseStrategyAgent().build(
        description,
        domain=domain,
        incident_date=args.get("incident_date"),
    )
    comp = strategy["compensation_estimate"]

    pdf_id = ""
    if strategy.get("pdf_path") and storage:
        pdf_id = storage.store_pdf_notice(
            user_id,
            "chat_tool",
            {
                "filename": strategy.get("filename", ""),
                "docx_filename": strategy.get("docx_filename", ""),
                "pdf_path": strategy.get("pdf_path", ""),
                "docx_path": strategy.get("docx_path", ""),
                "sender_name": "Case Brief",
                "recipient_name": strategy.get("domain", ""),
                "issue_type": "Case Strategy & Legal Opinion",
                "demand_amount": f"{comp.get('min_amount', 0)}-{comp.get('max_amount', 0)}",
            },
        )

    message = (
        f"{strategy['summary']} Estimated compensation range Rs {comp['min_amount']:,.0f} - "
        f"Rs {comp['max_amount']:,.0f} ({comp['currency']}). Opinion memo ready to download."
    )
    return {
        "type": "case_strategy",
        "status": "ready",
        "title": "Case Strategy & Opinion Memo",
        "message": message,
        "strategy": strategy,
        "view": "strategy",
        "pdf_id": pdf_id,
        "filename": strategy.get("filename"),
        "docx_filename": strategy.get("docx_filename"),
        "download_url": f"/download/{pdf_id}?format=pdf" if pdf_id else None,
        "docx_download_url": f"/download/{pdf_id}?format=docx" if pdf_id else None,
        "disclaimer": strategy["disclaimer"],
    }


def execute_tool(
    name: str,
    args: Dict[str, Any],
    user_id: str,
    storage,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if name == "legal_notice":
        return execute_legal_notice(args or {}, user_id, storage, profile)
    if name == "rti_application":
        return execute_rti_application(args or {}, user_id, storage, profile)
    if name == "cyber_fraud":
        return execute_cyber_fraud(args or {}, user_id, storage, profile)
    if name == "police_complaint":
        return execute_police_complaint(args or {}, user_id, storage, profile)
    if name == "labour_dues":
        return execute_labour_dues(args or {}, user_id, storage, profile)
    if name == "case_strategy":
        return execute_case_strategy(args or {}, user_id, storage)
    return {
        "type": name,
        "status": "error",
        "title": name.replace("_", " ").title(),
        "message": "This tool is not available.",
    }


# ── Top-Level Routing ────────────────────────────────────────────────────
def route_chat_tools(
    query: str,
    user_id: str = "anonymous",
    storage=None,
    allow_llm: bool = True,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    explicit = detect_explicit_tool(query)
    if explicit:
        arguments = _backfill_args(explicit, {}, query, profile)
        return {
            "tools": [execute_tool(explicit, arguments, user_id, storage, profile)] if storage else [],
            "mode": "regex",
        }

    if allow_llm and is_tool_relevant(query):
        hit = detect_tool_call(query)
        if hit:
            arguments = _backfill_args(hit["name"], hit["arguments"], query, profile)
            return {
                "tools": [execute_tool(hit["name"], arguments, user_id, storage, profile)] if storage else [],
                "mode": "llm",
            }

    return {"tools": [], "mode": "none"}


def chat_response(query: str, text: str = None) -> Dict[str, Any]:
    reply = (text or "").strip()
    if not reply:
        from src.llm_router import CHAT_FALLBACK_REPLY

        reply = CHAT_FALLBACK_REPLY
    return {
        "query": query,
        "status": "success",
        "response_type": "chat",
        "confidence_score": 0.0,
        "short_answer": reply,
        "full_response": reply,
        "response": reply,
        "sources": [],
        "applicable_laws": {},
        "tools": [],
        "cached": False,
    }
