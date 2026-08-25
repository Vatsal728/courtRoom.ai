"""
agent.py - Agentic tool layer (hybrid routing).

Layer 1 (deterministic, zero-cost): greeting detection + explicit tool
  phrases (regex) -> no LLM call.
Layer 2 (native tool-calling): keyword gate first; if the query looks
  tool-relevant, ask the LLM (Groq chat.completions `tools` -> local Ollama
  /api/chat `tools` fallback) whether it wants one of the registered tools,
  and parse its `tool_calls`.
Execution: when a tool is resolved, run the matching agent and return an
  artifact payload the chat UI renders as a card.

The pure legal-RAG path is untouched; tools are only detected when the query
is tool-relevant so every normal query stays cheap.
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.agents.notice_agent import LegalNoticeAgent
from src.agents.rti_agent import RTIApplicationAgent

load_dotenv(override=True)

# ── Tool registry ────────────────────────────────────────────────────────
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "legal_notice": {
        "name": "legal_notice",
        "description": (
            "Draft a formal legal notice (written demand letter) against a person, "
            "business or government body for disputes such as landlord-tenant, money "
            "recovery, consumer complaint, cheque bounce, defamation, workplace issue "
            "or breach of contract. Use when the user wants to send a legal notice, "
            "demand letter, or formal complaint document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sender_name": {"type": "string", "description": "Full name of the sender"},
                "sender_address": {"type": "string", "description": "Address of the sender"},
                "recipient_name": {"type": "string", "description": "Name of the recipient / opposite party"},
                "recipient_address": {"type": "string", "description": "Address of the recipient"},
                "issue_type": {"type": "string", "description": "Short label of the dispute, e.g. Non-payment of rent"},
                "issue_description": {"type": "string", "description": "Facts of the dispute: dates, amounts, what happened"},
                "applicable_section": {"type": "string", "description": "Law/section relied on, if known"},
                "demand_amount": {"type": "string", "description": "Money demanded in rupees, if any"},
            },
            "required": ["recipient_name", "issue_description"],
        },
    },
    "rti_application": {
        "name": "rti_application",
        "description": (
            "Draft a Right to Information (RTI) Act 2005 Section 6 application asking a "
            "public authority (municipality, police, education, health, etc.) for "
            "information. Use when the user wants to file an RTI or request information "
            "from a government body."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pio_office": {"type": "string", "description": "Government office / public authority name"},
                "pio_address": {"type": "string", "description": "Address of the PIO office"},
                "information_sought": {"type": "string", "description": "The specific information requested"},
                "applicant_name": {"type": "string", "description": "Full name of the applicant"},
                "applicant_address": {"type": "string"},
                "applicant_phone": {"type": "string"},
                "applicant_email": {"type": "string"},
            },
            "required": ["information_sought"],
        },
    },
    "case_strategy": {
        "name": "case_strategy",
        "description": (
            "Build a step-by-step case strategy: legal route, forums, evidence checklist, "
            "filing deadlines and an ESTIMATED compensation range. Use when the user wants "
            "a plan, strategy, roadmap, 'what to do next', their chances, or how much they "
            "can recover for a legal situation."
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
    "document_audit": {
        "name": "document_audit",
        "description": (
            "Audit a document the user pasted or uploaded (rent agreement, appointment "
            "letter, contract, complaint) against a checklist of required clauses and flag "
            "the missing ones. Use when the user asks to check, review or audit a document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_text": {"type": "string", "description": "The full text of the document to check"},
                "domain": {"type": "string", "enum": ["rent", "consumer", "labor", "criminal", "cyber", "defamation", "family", "commercial", "civil"], "description": "Document type/domain, if clear"},
            },
            "required": ["document_text"],
        },
    },
}

# ── Layer 1: deterministic detection ─────────────────────────────────────
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

_NOTICE_RE = re.compile(r"\b(legal\s*notice|demand\s*letter)\b", re.I)
_RTI_RE = re.compile(r"\b(rti\b|right\s*to\s*information|information\s*request)", re.I)

_DEFINITION_QUESTION_RE = re.compile(
    r"^\s*(what\s+is|what\s+are|define|meaning\s+of|how\s+does|how\s+do|what\s+do)\b", re.I
)

_TOOL_KEYWORD_RE = re.compile(
    r"\b(notice|demand\s*letter|rti|right\s*to\s*information|draft|generate|write|prepare|"
    r"application|form|complaint|letter|document|strategy|plan|roadmap|chances|steps?|"
    r"audit|review|recover|compensation|what\s+(should|can)\s+i\s+do)\b",
    re.I,
)


def detect_smalltalk(query: str) -> bool:
    """True for greetings / chat-openers / acknowledgements (no RAG needed)."""
    return bool(_SMALLTALK_RE.match((query or "").strip()))


def detect_explicit_tool(query: str) -> Optional[str]:
    q = query or ""
    if _DEFINITION_QUESTION_RE.match(q.strip()):
        return None
    if _NOTICE_RE.search(q):
        return "legal_notice"
    if _RTI_RE.search(q):
        return "rti_application"
    return None


def is_tool_relevant(query: str) -> bool:
    return bool(_TOOL_KEYWORD_RE.search(query or ""))


# ── Layer 2: native tool-calling ─────────────────────────────────────────
def _tool_defs() -> List[Dict[str, Any]]:
    return [{"type": "function", "function": spec} for spec in TOOL_SPECS.values()]


_TOOL_ROUTER_PROMPT = (
    "You are a tool router. If the user clearly wants one of the available tools, call it "
    "with a single tool_call and fill its parameters from the user's message. If no tool "
    "applies, reply with a plain text answer (never call a tool). Do not call more than "
    "one tool."
)


def _parse_tool_calls(resp) -> Optional[Dict[str, Any]]:
    calls = getattr(getattr(resp, "message", None), "tool_calls", None)
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
    """Groq native tool-calling, then local Ollama fallback."""
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


# ── Deterministic argument backfill ──────────────────────────────────────
# qwen2.5:3b is a small model: it reliably picks the RIGHT tool but often
# returns empty/partial `arguments`. Backfill missing fields from the user's
# query with regexes so a fully-described request still executes end-to-end.

_RUPEES_RE = re.compile(r"(?:rs\.?|inr|rupees?)\s*([\d]+(?:,\d{3})*(?:\.\d+)?)", re.I)
_PERSON_RE = re.compile(
    r"\b((?:mr|mrs|ms|dr|shri|shrimati|smt)\.?\s*)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b"
)
_PHONE_RE = re.compile(r"\b(\+?[\d\s-]{10,14})\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_DATE_RE = re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b")

_NOTICE_ISSUE_TYPES = [
    (("deposit", "refund"), "Security Deposit Refund"),
    (("salary", "wage", "unpaid"), "Non-payment of Salary/Wages"),
    (("evict", "eviction"), "Illegal Eviction"),
    (("product", "refund", "defective"), "Defective Product Refund"),
    (("cheque", "bounce", "chequebounce"), "Cheque Bounce"),
    (("breach", "contract"), "Breach of Contract"),
]


def _fill_notice(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("sender_name") or "").strip():
        if str(profile.get("name") or "").strip():
            args["sender_name"] = profile["name"]
        else:
            m = re.search(r"(?:sender|from|my\s*name\s*is)\s+(?:[Mm][Rr][Ss]?\.?\s*|[Dd][Rr]\.?\s*)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})", q)
            if m:
                args["sender_name"] = m.group(1)
    if not str(args.get("sender_email") or "").strip():
        if str(profile.get("email") or "").strip():
            args["sender_email"] = profile["email"]
    if not str(args.get("recipient_name") or "").strip():
        m = re.search(r"(?i:to|against|for)\s+(?:landlord\s+)?((?i:mr|mrs|ms|dr)\.?\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}", q)
        if m:
            args["recipient_name"] = re.sub(r"^(?i:to|against|for)\s+", "", m.group(0).strip())
    if not str(args.get("demand_amount") or "").strip():
        m = _RUPEES_RE.search(q)
        if m:
            args["demand_amount"] = m.group(1)
    if not str(args.get("issue_type") or "").strip():
        low = q.lower()
        for kws, label in _NOTICE_ISSUE_TYPES:
            if any(k in low for k in kws):
                args["issue_type"] = label
                break
    if not str(args.get("issue_description") or "").strip():
        m = re.search(r"(?:issue\s+is\s*[:,-]?\s*|about\s+|regarding\s+|because\s*[:,-]?\s*)(.+)$", q, re.I)
        if not m:
            m = re.search(r",\s*(.+)$", q)
        if m:
            args["issue_description"] = m.group(1).strip()[:1200]
    return args


def _fill_rti(args: Dict[str, Any], q: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("applicant_name") or "").strip():
        if str(profile.get("name") or "").strip():
            args["applicant_name"] = profile["name"]
        else:
            m = re.search(r"(?:applicant|name)\s+((?:mr|mrs|ms|dr)\.?\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})", q, re.I)
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
        m = re.search(r"(?:to\s+the\s+|to\s+|from\s+the\s+|from\s+)(?!get\s|ask\s|receive\s|obtain\s|file\s|submit\s|request\s)([A-Za-z][A-Za-z\s&'-]{3,60}?)(?=$|\s+(?:for|about|regarding|seeking|,))", q)
        if m:
            args["pio_office"] = m.group(1).strip()
    if not str(args.get("information_sought") or "").strip():
        m = re.search(r"(?:for|about|regarding|seeking|requesting|asking\s+for)\s+(.+?)(?=(?:,?\s*(?:applicant|phone|contact|email)\b)|$)", q, re.I)
        args["information_sought"] = (m.group(1).strip() if m else "")[:1200]
    if not str(args.get("applicant_phone") or "").strip():
        m = re.search(r"(?:phone|contact|mobile|tel)[:\s]*([\d+\-\s]{10,})", q, re.I)
        if not m:
            m = _PHONE_RE.search(q)
        if m:
            args["applicant_phone"] = m.group(1).strip()[-12:]
    return args


def _fill_strategy(args: Dict[str, Any], q: str) -> Dict[str, Any]:
    args = dict(args)
    if not str(args.get("case_description") or "").strip():
        args["case_description"] = q.strip()
    if not str(args.get("incident_date") or "").strip():
        m = _DATE_RE.search(q)
        if m:
            args["incident_date"] = m.group(1)
    return args


def _fill_audit(args: Dict[str, Any], q: str) -> Dict[str, Any]:
    args = dict(args)
    if not str(args.get("document_text") or "").strip():
        args["document_text"] = q.strip()
    return args


def _backfill_args(name: str, args: Dict[str, Any], query: str,
                   profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = args or {}
    if name == "legal_notice":
        return _fill_notice(args, query or "", profile)
    if name == "rti_application":
        return _fill_rti(args, query or "", profile)
    if name == "case_strategy":
        return _fill_strategy(args, query or "")
    if name == "document_audit":
        return _fill_audit(args, query or "")
    return args


_BARE_NAME_RE = re.compile(
    r"^(?:(?:mr|mrs|ms|dr|shri|shrimati|smt|prof|capt|col)\.?\s+)?"
    r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}$"
)

_SENDER_STMT_RE = re.compile(r"(?:my\s+name\s+is|i\s*'?m\s+(?:called\s+)?|call\s+me|i\s+am\s+named)", re.I)

_LEADING_STOPWORDS = re.compile(
    r"^(he|she|i|we|they|it|the|my|our|your|his|her|its|their|rent|landlord|sir|madam|"
    r"this|that|there|me|you|us|my)\b",
    re.I,
)

_LEADING_NAME_RE = re.compile(
    r"^((?:(?:mr|mrs|ms|dr|shri|shrimati|smt)\.?\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})(?=[,;:\s]|$)"
)


def _is_bare_name(text: str) -> bool:
    return bool(_BARE_NAME_RE.match((text or "").strip()))


def _extract_recipient_name(answer: str, sender_stmt: bool) -> str:
    """Pull a recipient name from a follow-up answer without stealing the
    sender's own name or generic sentence-openers like 'he'/'the'."""
    if not answer:
        return ""
    m = re.search(r"(?i:to|against|for|towards)\s+(?:landlord\s+)?((?i:mr|mrs|ms|dr)\.?\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}", answer)
    if m:
        return re.sub(r"^(?i:to|against|for|towards)\s+", "", m.group(0).strip())
    if sender_stmt or _LEADING_STOPWORDS.match(answer):
        return ""
    m = _LEADING_NAME_RE.match(answer)
    return m.group(1) if m else ""


def merge_tool_answer(name: str, args: Dict[str, Any], answer: str,
                      profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge a user's follow-up answer into a pending tool's partial args.

    Only missing fields are filled, so info gathered across turns accumulates.
    A bare name fills the recipient; anything richer fills the description.
    """
    args = dict(args or {})
    answer = (answer or "").strip()
    if not answer:
        return args
    profile = profile or {}
    if name == "legal_notice":
        sender_stmt = bool(_SENDER_STMT_RE.search(answer))
        if not str(args.get("recipient_name") or "").strip():
            recipient = _extract_recipient_name(answer, sender_stmt)
            if recipient:
                args["recipient_name"] = recipient
        if not str(args.get("sender_name") or "").strip():
            if str(profile.get("name") or "").strip():
                args["sender_name"] = profile["name"]
            else:
                m = re.search(r"(?:my\s+name\s+is|i\s*'?m\s+(?:called\s+)?|call\s+me)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})", answer, re.I)
                if m:
                    args["sender_name"] = m.group(1)
        if not str(args.get("sender_email") or "").strip():
            if str(profile.get("email") or "").strip():
                args["sender_email"] = profile["email"]
            elif _EMAIL_RE.search(answer):
                args["sender_email"] = _EMAIL_RE.search(answer).group(0)
        if not str(args.get("demand_amount") or "").strip():
            m = _RUPEES_RE.search(answer)
            if m:
                args["demand_amount"] = m.group(1)
        if not str(args.get("issue_type") or "").strip():
            low = answer.lower()
            for kws, label in _NOTICE_ISSUE_TYPES:
                if any(k in low for k in kws):
                    args["issue_type"] = label
                    break
        if not str(args.get("issue_description") or "").strip() and not _is_bare_name(answer) and not sender_stmt:
            desc = answer
            recipient = args.get("recipient_name", "") or ""
            if recipient and desc.startswith(recipient):
                desc = desc[len(recipient):].lstrip(",;: \t-")
            if desc:
                args["issue_description"] = desc[:1200]
        return args
    if name == "rti_application":
        if not str(args.get("applicant_name") or "").strip():
            if str(profile.get("name") or "").strip():
                args["applicant_name"] = profile["name"]
        if not str(args.get("applicant_email") or "").strip():
            if str(profile.get("email") or "").strip():
                args["applicant_email"] = profile["email"]
        if not str(args.get("pio_office") or "").strip():
            m = re.search(r"(?:from|to|of|office\s+of)\s+(?:the\s+)?([A-Za-z][A-Za-z\s&'-]{3,60}?)(?=$|\s+(?:for|about|regarding|seeking))", answer)
            if m:
                args["pio_office"] = m.group(1).strip()
            else:
                m = re.search(r"(?:the\s+)?([A-Za-z][A-Za-z\s&'-]{2,60}?(?:board|department|office|corporation|municipality|police|authority|commission))\b", answer, re.I)
                if m:
                    args["pio_office"] = m.group(1).strip()
        if not str(args.get("information_sought") or "").strip():
            office = args.get("pio_office") or ""
            if not (office and _is_bare_name(answer)):
                args["information_sought"] = answer[:1200]
        return args
    if name == "case_strategy":
        if not str(args.get("case_description") or "").strip():
            args["case_description"] = answer
        return args
    if name == "document_audit":
        if not str(args.get("document_text") or "").strip():
            args["document_text"] = answer
        return args
    return args


# ── Tool execution ───────────────────────────────────────────────────────
def execute_legal_notice(args: Dict[str, Any], user_id: str, storage,
                         profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = dict(args)
    profile = profile or {}
    if not str(args.get("sender_name") or "").strip():
        args["sender_name"] = profile.get("name", "")
    if not str(args.get("sender_email") or "").strip():
        args["sender_email"] = profile.get("email", "")
    critical = ["recipient_name", "issue_description"]
    if not str(args.get("sender_name") or "").strip():
        critical.append("sender_name")
    missing = [c for c in critical if not str(args.get(c) or "").strip()]
    if missing:
        asks = []
        if "sender_name" in missing:
            asks.append("What is your name? (or sign in so I can fill it from your profile)")
        if "recipient_name" in missing:
            asks.append("Who should the notice be sent to? (name, and address if you have it)")
        if "issue_description" in missing:
            asks.append("What happened? Describe the facts briefly (dates, amounts, what went wrong)")
        return {
            "type": "legal_notice",
            "status": "needs_input",
            "title": "Legal Notice",
            "message": " ".join(asks),
            "missing_fields": missing,
            "params": args,
        }
    filename = (
        f"Legal_Notice_{str(args['sender_name']).replace(' ', '_')[:40] or 'Sender'}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    args.setdefault("issue_type", "Legal Dispute")
    args.setdefault("demand_amount", "")
    pdf_path = LegalNoticeAgent().generate_notice(args, output_path=f"output/{filename}")
    pdf_id = storage.store_pdf_notice(user_id, "chat_tool", {
        "filename": filename,
        "pdf_path": pdf_path,
        "sender_name": args.get("sender_name", ""),
        "sender_email": args.get("sender_email", ""),
        "recipient_name": args.get("recipient_name", ""),
        "recipient_email": args.get("recipient_email", ""),
        "issue_type": args.get("issue_type", ""),
        "demand_amount": args.get("demand_amount", ""),
    })
    sender = args.get("sender_name", "") or "Sender"
    recipient = args.get("recipient_name", "") or "Recipient"
    return {
        "type": "legal_notice",
        "status": "ready",
        "title": "Legal Notice drafted",
        "message": f"Legal notice from {sender} to {recipient} is ready to download or edit.",
        "pdf_id": pdf_id,
        "filename": filename,
        "download_url": f"/pdf/{pdf_id}/download",
        "params": args,
    }


def execute_rti_application(args: Dict[str, Any], user_id: str, storage,
                            profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not str(args.get("information_sought") or "").strip():
        asks = ["Which government office do you want to ask (e.g. municipality, police, education department)?"]
        if not str(args.get("pio_office") or "").strip():
            asks.insert(0, "Which information do you want from the government office?")
        return {
            "type": "rti_application",
            "status": "needs_input",
            "title": "RTI Application",
            "message": " ".join(asks),
            "missing_fields": ["information_sought"],
            "params": args,
        }
    application = RTIApplicationAgent().generate_rti_application(args)
    office = args.get("pio_office") or "the public authority"
    applicant = args.get("applicant_name") or "you"
    return {
        "type": "rti_application",
        "status": "ready",
        "title": "RTI Application drafted",
        "message": f"RTI application by {applicant} to {office} (Section 6, RTI Act 2005) is ready to copy.",
        "filename": f"RTI_Application_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "application": application,
        "params": args,
    }


def execute_case_strategy(args: Dict[str, Any], user_id: str, storage) -> Dict[str, Any]:
    from src.agents.strategy_agent import CaseStrategyAgent

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
    if auto != "civil":
        domain = auto
    else:
        domain = domain_arg
    strategy = CaseStrategyAgent().build(
        description,
        domain=domain,
        incident_date=args.get("incident_date"),
    )
    comp = strategy["compensation_estimate"]
    message = (
        f"{strategy['summary']} Estimated compensation range Rs {comp['min_amount']:,.0f} - "
        f"Rs {comp['max_amount']:,.0f} ({comp['currency']}). Open the strategy view for the full plan."
    )
    return {
        "type": "case_strategy",
        "status": "ready",
        "title": "Case Strategy",
        "message": message,
        "strategy": strategy,
        "view": "strategy",
        "disclaimer": strategy["disclaimer"],
    }


def execute_document_audit(args: Dict[str, Any], user_id: str, storage) -> Dict[str, Any]:
    from src.agents.audit_agent import DocumentAuditAgent

    text = str(args.get("document_text") or "").strip()
    if not text:
        return {
            "type": "document_audit",
            "status": "needs_input",
            "title": "Document Audit",
            "message": "Paste the document text (or upload it and ask me to audit it) and I'll check it against a required-clause checklist.",
            "missing_fields": ["document_text"],
            "params": args,
        }
    audit = DocumentAuditAgent().audit(text, domain=args.get("domain") or "civil")
    message = (
        f"{audit['document_type']}: {audit['present_count']}/{audit['total_checks']} clauses "
        f"found ({audit['score']}% - {audit['risk']} risk)."
    )
    if audit["issues"]:
        message += " Missing: " + "; ".join(i["label"] for i in audit["issues"][:5])
    return {
        "type": "document_audit",
        "status": "ready",
        "title": "Document Audit",
        "message": message,
        "audit": audit,
        "view": "audit",
    }


def execute_tool(name: str, args: Dict[str, Any], user_id: str, storage,
                 profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if name == "legal_notice":
        return execute_legal_notice(args or {}, user_id, storage, profile)
    if name == "rti_application":
        return execute_rti_application(args or {}, user_id, storage, profile)
    if name == "case_strategy":
        return execute_case_strategy(args or {}, user_id, storage)
    if name == "document_audit":
        return execute_document_audit(args or {}, user_id, storage)
    return {
        "type": name,
        "status": "error",
        "title": name.replace("_", " ").title(),
        "message": "This tool is not available yet.",
    }


# ── Top-level routing ────────────────────────────────────────────────────
def route_chat_tools(query: str, user_id: str = "anonymous", storage=None,
                     allow_llm: bool = True,
                     profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect tool intent (Layer 1 regex -> Layer 2 LLM) and execute.

    Returns {"tools": [...], "mode": "regex"|"llm"|"none"}. Storage may be
    None for read-only callers; then tools that need a store return
    needs_input/error instead of ready.
    """
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


# ── Small-talk response (LLM-generated, no RAG) ─────────────────────────
def chat_response(query: str, text: str = None) -> Dict[str, Any]:
    """Shape the LLM's small-talk reply into the standard result dict.

    `text` is the model's reply (from LLMRouter.generate_chat_reply /
    stream_chat_reply). When None (model unavailable), a short fallback is
    used so the UI still gets a graceful message.
    """
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


if __name__ == "__main__":
    for q in ["hi", "hello there", "who are you", "ok", "thanks",
              "generate legal notice for my landlord", "file an rti against the municipality",
              "what is a legal notice", "my landlord is not returning my deposit",
              "someone hacked my website", "help me with my landlord"]:
        print(q, "->",
              "smalltalk" if detect_smalltalk(q) else
              ("regex:" + str(detect_explicit_tool(q)) if detect_explicit_tool(q) else
               "tool-relevant:" + str(is_tool_relevant(q))))
