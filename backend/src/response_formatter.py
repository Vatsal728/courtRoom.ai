import json
import logging
import re
from typing import List, Dict, Any, Tuple

from src.domain_config import get_config

logger = logging.getLogger("courtroom-response-formatter")

_CJK_RE = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)

_CITATION_MARKER_RE = re.compile(
    r"\b(?:section|sec\.?|s\.|art\.?|article|rule|sch\.?|schedule|bnss?|ipc|crpc|bsa|iea)\b",
    re.IGNORECASE,
)

_PROCESSING_ERR = "The AI response could not be processed. Please try again."


def _extract_json_object(text: str) -> dict:
    """Tolerantly extract a JSON object from LLM output."""
    if not text or not text.strip():
        return {}
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()

    candidates = []
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])
    if first != -1 and last != -1:
        head = text[first : last + 1]
        cut = head.rfind(",")
        while cut > 0:
            trimmed = head[:cut] + "}"
            if trimmed.count("{") == trimmed.count("}"):
                candidates.append(trimmed)
            cut = head.rfind(",", 0, cut)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


def _clean_text(s: str) -> str:
    """Strip markdown characters from LLM field text for plain-text display"""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.M)
    s = re.sub(r"^\s*[>\u2192\u21d2]+\s*", "", s, flags=re.M)
    return s.strip()


def _get_str(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return _clean_text(v)
    return default


def _get_list(d: dict, *keys: str, default: list = None) -> list:
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v:
            return [_clean_text(x) for x in v if str(x).strip()]
    return default or []


# Multi-target alias groups so both current enactments (BNS, BNSS, BSA, Code on Wages)
# and historical acts match their respective retrieved sources accurately.
_ACT_ALIASES = {
    "bnss": ("bharatiya nagarik suraksha sanhita", "code of criminal procedure"),
    "crpc": ("code of criminal procedure", "bharatiya nagarik suraksha sanhita"),
    "bns": ("bharatiya nyaya sanhita", "indian penal code"),
    "ipc": ("indian penal code", "bharatiya nyaya sanhita"),
    "bsa": ("bharatiya sakshya adhiniyam", "indian evidence act"),
    "iea": ("indian evidence act", "bharatiya sakshya adhiniyam"),
    "code on wages": ("code on wages", "payment of wages", "minimum wages"),
    "nia": ("negotiable instruments act",),
    "hsa": ("hindu succession act",),
    "hma": ("hindu marriage act",),
    "cpa": ("consumer protection act",),
    "ida": ("industrial disputes act", "industrial relations code"),
    "mva": ("motor vehicles act",),
    "pwa": ("payment of wages act", "code on wages"),
    "mwa": ("minimum wages act", "code on wages"),
    "coi": ("constitution of india",),
    "rti": ("right to information",),
    "it act": ("information technology act",),
    "dpdp": ("digital personal data protection",),
    "tpa": ("transfer of property act",),
}


def _filter_sections(sections: list, sources: List[Dict]) -> list:
    """Keep only LLM-cited sections that actually exist in the retrieved sources.
    Drops hallucinated section numbers and numberless vagueness."""
    if not sections:
        return sections
    valid = []
    for s in sources:
        act = str(s.get("act_name") or s.get("source_act") or "").lower().strip()
        num = re.sub(r"[^0-9]", "", str(s.get("section_number") or s.get("section") or ""))
        if act and num:
            valid.append((act, num))

    def acts_of(sec_lower: str) -> Tuple[str, ...]:
        # Check longer keys first (e.g. 'bnss' before 'bns')
        for alias in sorted(_ACT_ALIASES.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", sec_lower):
                return _ACT_ALIASES[alias]
        for vact, _ in valid:
            first = vact.split()[0]
            if first in sec_lower:
                return (vact,)
        return ()

    kept = []
    seen = set()
    for sec in sections:
        plain = re.sub(r"\(.*?\)", "", sec.lower())
        m = re.search(r"(?:section|sec\.?|article|art\.?)\s*(\d{1,4})\b", plain) or re.search(
            r"\b(\d{1,4})(?:[a-z]|)\b", plain
        )
        if not m:
            continue
        num = m.group(1)
        candidate_acts = acts_of(plain)
        matches = [
            (va, vn)
            for va, vn in valid
            if vn == num
            and (
                not candidate_acts
                or any(va.startswith(ca) or ca in va for ca in candidate_acts)
            )
        ]
        if matches:
            key = (matches[0][0], num)
            if key in seen:
                continue
            seen.add(key)
            kept.append(sec)
    return kept


def _filter_citations(items: list, sources: List[Dict]) -> list:
    """Keep only entries that either cite no section number, or cite a section
    number that exists in the retrieved sources."""
    if not items:
        return items
    valid_nums = set()
    for s in sources:
        num = re.sub(r"[^0-9]", "", str(s.get("section_number") or s.get("section") or ""))
        if num:
            valid_nums.add(num)
    kept = []
    for it in items:
        nums = [
            n
            for n in re.findall(r"\b\d{1,4}\b", it)
            if not (n.isdigit() and 1900 <= int(n) <= 2100)
        ]
        if nums and _CITATION_MARKER_RE.search(it) and not all(n in valid_nums for n in nums):
            continue
        kept.append(it)
    return kept


def _get_penalties(cr: dict) -> list:
    raw = cr.get("penalties") if isinstance(cr, dict) else None
    if isinstance(raw, dict) and raw:
        out = []
        for k, v in raw.items():
            if str(v).strip():
                out.append(f"{k}: {_clean_text(str(v))}")
        return out
    return _get_list(cr, "penalties")


def _get_criminal_route(data: dict, sources: List[Dict]) -> Dict[str, Any]:
    cr = data.get("criminal_route", {})
    if not isinstance(cr, dict):
        cr = {}
    sections = _filter_sections(_get_list(cr, "applicable_sections"), sources)
    if not sections:
        raw_pens = cr.get("penalties")
        if isinstance(raw_pens, dict) and raw_pens:
            sections = _filter_sections([str(k) for k in raw_pens.keys()], sources)
    return {
        "applicable_sections": sections,
        "penalties": _filter_citations(_get_penalties(cr), sources),
        "procedure": _filter_citations(_get_list(cr, "procedure"), sources),
    }


def _get_civil_route(data: dict) -> Dict[str, Any]:
    cr = data.get("civil_route", {})
    if not isinstance(cr, dict):
        cr = {}
    return {
        "remedies": _get_list(cr, "remedies"),
        "compensation_range": _get_str(cr, "compensation_range", "compensation"),
        "procedure": _get_list(cr, "procedure"),
    }


def _format_markdown(
    short_answer: str,
    is_this_illegal: str,
    criminal_route: dict,
    civil_route: dict,
    compensation_claims: list,
    evidence_needed: list,
    practical_steps: list,
) -> str:
    def numbered(items: list, indent: str = "   ") -> str:
        return "\n".join(f"{indent}{i + 1}. {item}" for i, item in enumerate(items))

    headers = get_config().get("response_shaping", {}).get("route_headers", {})
    crim_header = headers.get("criminal", "CRIMINAL ROUTE")
    civ_header = headers.get("civil", "CIVIL ROUTE")

    crim_proc = numbered(criminal_route["procedure"])
    civ_proc = numbered(civil_route["procedure"])
    comp = "\n".join(f"• {c}" for c in compensation_claims)
    evid = "\n".join(f"• {e}" for e in evidence_needed)
    steps = numbered(practical_steps, indent="")
    sections = ", ".join(criminal_route["applicable_sections"])
    penalties = criminal_route["penalties"]
    remedies = ", ".join(civil_route["remedies"])

    parts = [f"SHORT ANSWER\n{short_answer}", f"IS THIS ILLEGAL?\n{is_this_illegal}"]

    crim_body = []
    if sections:
        crim_body.append(f"• Applicable Sections: {sections}")
    if penalties:
        crim_body.append("• Penalties:\n" + numbered(penalties))
    if crim_proc:
        crim_body.append("• Procedure:\n" + crim_proc)
    if crim_body:
        parts.append(crim_header + "\n" + "\n".join(crim_body))

    civ_body = []
    if remedies:
        civ_body.append(f"• Remedies: {remedies}")
    if civil_route["compensation_range"]:
        civ_body.append(f"• Compensation Range: {civil_route['compensation_range']}")
    if civ_proc:
        civ_body.append("• Procedure:\n" + civ_proc)
    if civ_body:
        parts.append(civ_header + "\n" + "\n".join(civ_body))

    if comp:
        parts.append("COMPENSATION CLAIMS\n" + comp)
    if evid:
        parts.append("EVIDENCE CHECKLIST\n" + evid)
    if steps:
        parts.append("PRACTICAL STEPS (ACTION PLAN)\n" + steps)

    return "\n\n".join(parts)


def _build_applicable_laws(sources: List[Dict]) -> Dict[str, list]:
    laws = {}
    for s in sources:
        act = s.get("act_name") or s.get("source_act") or s.get("source") or "General Law"
        sec = s.get("section_number") or s.get("section") or "General Provision"
        laws.setdefault(act, []).append(sec)
    return laws


def _build_formatted_sources(sources: List[Dict]) -> List[Dict]:
    return [
        {
            "section": s.get("section_number") or s.get("section") or "Unknown",
            "section_title": s.get("section_title") or "",
            "topic": s.get("topic") or "General",
            "source_act": s.get("act_name") or s.get("source_act") or s.get("source") or "Unknown Act",
            "courts": s.get("applicable_courts") or s.get("courts") or ["District Court"],
            "keywords": s.get("keywords") or [],
            "status": s.get("status") or "active",
            "effective_from": s.get("effective_from") or "",
            "effective_until": s.get("effective_until") or "",
            "replaced_by": s.get("replaced_by") or "",
            "content_preview": (s.get("content") or s.get("text") or "")[:300],
            "content": s.get("content") or s.get("text") or "",
        }
        for s in sources
    ]


class ResponseFormatter:
    """Format LLM JSON response into structured NyayGuru-style output"""

    def _apply_shaping(self, data: dict, domain: str) -> dict:
        shaping = get_config().get("response_shaping", {})
        suppress_crim = shaping.get("suppress_criminal_route", [])
        suppress_civ = shaping.get("suppress_civil_route", [])
        if domain in suppress_crim:
            data.pop("criminal_route", None)
        if domain in suppress_civ:
            data.pop("civil_route", None)
        return data

    def format_response(
        self,
        query: str,
        llm_response: str,
        sources: List[Dict],
        domain: str,
        confidence: float,
        response_type: str = None,
    ) -> Dict[str, Any]:
        data = {}
        if llm_response and llm_response.strip():
            data = _extract_json_object(llm_response)

        garbled = bool(data) and bool(_CJK_RE.search(json.dumps(data, ensure_ascii=False)))
        if garbled:
            logger.warning("LLM output contained CJK script; using graceful fallback")
            data = {}
            llm_response = ""

        data = self._apply_shaping(data, domain)

        short_answer = _get_str(
            data,
            "short_answer",
            default=_PROCESSING_ERR if (garbled or (llm_response and not data)) else "",
        )
        is_this_illegal = _get_str(data, "is_this_illegal")
        criminal_route = _get_criminal_route(data, sources)
        civil_route = _get_civil_route(data)
        compensation_claims = _get_list(data, "compensation_claims")
        evidence_needed = _get_list(data, "evidence_needed")
        practical_steps = _get_list(data, "practical_steps")[:6]

        markdown = _format_markdown(
            short_answer,
            is_this_illegal,
            criminal_route,
            civil_route,
            compensation_claims,
            evidence_needed,
            practical_steps,
        )

        return {
            "query": query,
            "response_type": response_type or domain,
            "confidence_score": confidence,
            "short_answer": short_answer,
            "full_response": markdown,
            "response": markdown,
            "is_this_illegal": is_this_illegal,
            "criminal_route": criminal_route,
            "civil_route": civil_route,
            "practical_steps": practical_steps,
            "compensation_claims": compensation_claims,
            "evidence_needed": evidence_needed,
            "applicable_laws": _build_applicable_laws(sources),
            "sources": _build_formatted_sources(sources),
            "status": "success",
        }


def format_legal_response(
    query: str,
    llm_response: str,
    sources: List[Dict],
    domain: str,
    confidence: float,
    response_type: str = None,
) -> Dict[str, Any]:
    return ResponseFormatter().format_response(query, llm_response, sources, domain, confidence, response_type)
