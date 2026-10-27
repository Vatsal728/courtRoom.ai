"""
fact_extractor.py - Structured fact/issue extraction from a user query.

Legal answers depend on facts. This module pulls the facts the query *does*
state and flags the facts it *needs but lacks*, so the pipeline can either
retrieve against what is known or ask clarifying questions instead of
fabricating specifics.

Rule-based (regex + config): fast, deterministic, free.
"""
import re
from typing import Dict, List

from src.domain_config import get_config

_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "new delhi", "bengaluru", "bangalore", "mumbai", "pune", "chennai",
    "kolkata", "hyderabad", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur",
    "patna", "indore", "bhopal", "nagpur", "thane", "coimbatore", "vadodara",
    "juhu", "bandra", "andheri", "dadar", "borivali", "powai", "worli", "colaba",
    "indiranagar", "whitefield", "koramangala", "electronic city", "gurgaon",
    "gurugram", "noida", "ghaziabad", "faridabad", "navi mumbai", "secunderabad",
]

_RUPEE = chr(0x20B9)
_MONEY_RE = re.compile(
    r"(?:{rupee}|rs\.?|rupees?|inr)\s*([\d][\d,]*(?:\.[\d]+)?)\s*(lakh|lac|crore|cr|thousand|k)?\b"
    r"|([\d][\d,]*(?:\.[\d]+)?)\s*(lakh|lac|crore|cr|thousand|k)?\s*(?:rupees?|rs\.?|inr|/-)".format(
        rupee=re.escape(_RUPEE)
    ),
    re.IGNORECASE,
)

_MULT = {
    "lakh": 100_000,
    "lac": 100_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "thousand": 1_000,
    "k": 1_000,
}

_STATE_HINT_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _STATES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_ACTOR_PATTERNS = {
    "landlord": ["landlord", "owner of the flat", "owner of the house", "flat owner"],
    "tenant": ["tenant", "i rent", "i rented", "renter"],
    "employer": ["employer", "manager", "company", "recruiter", "recruitment agent", "consultancy", "pvt ltd", "limited", "hr"],
    "employee": ["employee", "worker", "i was employed", "i resigned", "my salary"],
    "husband": ["husband", "my spouse"],
    "wife": ["wife"],
    "seller": ["seller", "shopkeeper", "shop owner", "trader", "vendor", "store", "builder"],
    "attacker": ["assaulted", "attacked", "robber", "snatcher", "thief", "someone", "men on a"],
    "bank": ["bank", "banker", "hdfc", "sbi", "icici", "axis", "kotak", "pnb"],
    "unknown_third_party": ["someone", "somebody", "a person", "unknown", "stranger", "fraudster"],
}

_EVENT_PATTERNS = {
    "assault": ["assault", "assaulted", "hit", "beat", "beaten", "beating", "slap", "slapped", "attacked"],
    "theft": ["stole", "stolen", "steal", "stealing", "pickpocket", "snatch", "snatched", "robbed", "rob", "robbery"],
    "hacking": ["hack", "hacked", "hacking", "breach", "leaked", "unauthorised access", "unauthorized access"],
    "eviction": ["evict", "eviction", "evicting", "throw", "thrown", "kicked out", "vacate", "lock", "cutting water", "cut off utilities"],
    "fraud": ["fraud", "fraudster", "scam", "scammed", "cheat", "cheated", "deceived", "duped", "fake", "false promise", "debited", "kyc"],
    "non_payment": ["didn't pay", "did not pay", "not paying", "unpaid", "hasn't paid", "has not paid", "not paid", "salary not", "not released"],
    "termination": ["fired", "terminated", "sacked", "laid off", "layoff", "retrenched", "dismissed", "resigned"],
    "cheque_bounce": ["bounce", "bounced", "dishonour", "dishonor", "cheque", "returned unpaid"],
    "divorce": ["divorce", "divorced", "dissolution of marriage", "separat"],
    "dowry": ["dowry", "demanded dowry", "asked for dowry"],
    "defective_goods": ["defective", "defect", "not working", "stopped working", "broke down", "damaged"],
    "property_dispute": ["property", "sale deed", "registry", "encroach", "boundary", "my land", "my flat sold"],
    "threat": ["threat", "threaten", "threatened", "threatening", "intimidat", "harass"],
    "custody": ["custody", "guardianship", "my child"],
    "maintenance": ["maintenance", "alimony", "support"],
}


def _detect_issue(name: str, config: dict, text: str) -> bool:
    keywords = config.get(name, {}).get("keywords", [])
    for k in keywords:
        if len(k) <= 4:
            if re.search(rf"\b{re.escape(k)}\b", text):
                return True
        else:
            if re.search(rf"\b{re.escape(k)}", text):
                return True
    return False


def extract_facts(query: str) -> Dict:
    """Extract structured facts/actors/events/issues from a single query."""
    cfg = get_config()
    q = query.lower()

    issues = [
        name
        for name in (cfg.get("legal_elements") or {})
        if _detect_issue(name, cfg.get("legal_elements") or {}, q)
    ]
    events = [name for name, kws in _EVENT_PATTERNS.items() if any(k in q for k in kws)]
    actors = [name for name, kws in _ACTOR_PATTERNS.items() if any(k in q for k in kws)]

    location = None
    m = _STATE_HINT_RE.search(q)
    if m:
        location = m.group(1).title()
    location_known = location is not None

    amount = None
    amounts_found: List[float] = []
    for m in _MONEY_RE.finditer(q):
        num_str = m.group(1) or m.group(3) or ""
        unit = (m.group(2) or m.group(4) or "").lower().rstrip("s")
        try:
            val = float(num_str.replace(",", ""))
            if unit in _MULT:
                val *= _MULT[unit]
            if val > 0:
                amounts_found.append(val)
        except ValueError:
            continue
    if amounts_found:
        amount = sum(amounts_found) if len(amounts_found) > 1 else amounts_found[0]

    facts = []
    for sentence in re.split(r"(?<=[.!?]) ", query):
        if len(sentence.split()) >= 3 and any(
            kw in sentence.lower()
            for kw in ("i ", "my ", "we ", "the ", "my landlord", "my employer", "someone", "a person", "two men", "they ")
        ):
            facts.append(sentence.strip())
    facts = facts[:6]

    missing_cfg = cfg.get("missing_facts") or {}
    relevant = [name for name in issues if name in missing_cfg]
    jurisdiction_required = any(
        "state" in (missing_cfg.get(name) or {}).get("required", []) for name in relevant
    )
    missing_facts: List[str] = []
    blocking = False
    needs_facts: List[str] = []

    for name in relevant:
        spec = missing_cfg[name]
        required = spec.get("required", [])
        have = set()
        if "state" in required and location_known:
            have.add("state")
        missing_here: List[str] = []
        for req in required:
            if req in have:
                continue
            if req == "state":
                if not location_known:
                    missing_here.append("state")
                continue
            if req in ("amount_paid", "amount") and amount is not None:
                continue
            hint = {
                "tenancy_type": ["rent agreement", "lease", "written agreement", "oral", "verbal", "registered agreement"],
                "notice": ["notice", "notice served", "legal notice", "eviction notice"],
                "reason": ["not paying rent", "non-payment", "personal use", "sale", "own use", "repair"],
                "amount_paid": ["paid", "paid ", "deposit", "security deposit", "fee", "debited", "rs", "inr"],
                "mode": ["upi", "bank transfer", "neft", "cash", "cheque", "rtgs", "imps", "online"],
                "proof": ["receipt", "message", "whatsapp", "email", "agreement", "screenshot", "bank statement", "utr", "reference"],
                "injuries": ["injured", "injuries", "bleeding", "fracture", "hospital"],
                "fir": ["fir", "police", "complaint", "reported", "report", "sho"],
                "notice_sent": ["notice", "sent", "legal notice"],
            }.get(req)
            if hint and any(h in q for h in hint):
                continue
            missing_here.append(req)
        missing_facts.extend(missing_here)
        if spec.get("blocking") and "state" in missing_here:
            blocking = True
        if missing_here:
            needs_facts.extend(spec.get("ask", []))

    needs_facts = _dedupe(needs_facts)

    return {
        "facts": facts,
        "actors": actors,
        "events": events,
        "issues": issues,
        "amount": amount,
        "location": location,
        "location_known": location_known,
        "jurisdiction_required": jurisdiction_required,
        "missing_facts": missing_facts,
        "blocking": blocking,
        "needs_facts": needs_facts,
    }


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
