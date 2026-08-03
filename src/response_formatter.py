import json
import re
from typing import List, Dict, Any


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


_ACT_ALIASES = {
    "bnss": "code of criminal procedure", "crpc": "code of criminal procedure",
    "bns": "bharatiya nyaya sanhita", "ipc": "indian penal code",
    "bsa": "indian evidence act", "iea": "indian evidence act",
    "nia": "negotiable instruments act", "hsa": "hindu succession act",
    "hma": "hindu marriage act", "cpa": "consumer protection act",
    "ida": "industrial disputes act", "mva": "motor vehicles act",
    "pwa": "payment of wages act", "mwa": "minimum wages act",
    "coi": "constitution of india", "rti": "right to information",
    "it act": "information technology act",
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

    def act_of(sec_lower: str) -> str:
        for alias, full in _ACT_ALIASES.items():
            if alias in sec_lower:
                return full
        for vact, _ in valid:
            first = vact.split()[0]
            if first in sec_lower:
                return vact
        return ""

    kept = []
    for sec in sections:
        m = re.search(r"\b(\d{1,4})(?:[a-z]|)\b", sec.lower())
        if not m:
            continue
        num = m.group(1)
        act = act_of(sec.lower())
        matches = [(va, vn) for va, vn in valid if vn == num and (not act or va.startswith(act) or act in va)]
        if matches:
            kept.append(sec)
    return kept


def _filter_citations(items: list, sources: List[Dict]) -> list:
    """Keep only entries that either cite no section number, or cite a section
    number that exists in the retrieved sources. Kills hallucinated
    numbers in free-text fields like penalties."""
    if not items:
        return items
    valid_nums = set()
    for s in sources:
        num = re.sub(r"[^0-9]", "", str(s.get("section_number") or s.get("section") or ""))
        if num:
            valid_nums.add(num)
    kept = []
    for it in items:
        nums = re.findall(r"\b\d{1,4}\b", it)
        if nums and not all(n in valid_nums for n in nums):
            continue
        kept.append(it)
    return kept


def _get_criminal_route(data: dict, sources: List[Dict]) -> Dict[str, Any]:
    cr = data.get("criminal_route", {})
    if not isinstance(cr, dict):
        cr = {}
    sections = _filter_sections(_get_list(cr, "applicable_sections"), sources)
    if not sections:
        # LLM often cites sections in prose but leaves the array empty;
        # fall back to the top penal-act sources (all real, from retrieval).
        for s in sources:
            act = str(s.get("act_name") or s.get("source_act") or "")
            low = act.lower()
            if any(k in low for k in (
                "bharatiya nyaya", "indian penal", "negotiable instruments",
                "information technology", "code of criminal procedure",
            )):
                num = s.get("section_number") or s.get("section")
                if num:
                    num = re.sub(r"^(?:section|sec\.?|s\.?)\s*", "", str(num), flags=re.I)
                    sections.append(f"{act.split(' (')[0]} Section {num}")
            if len(sections) >= 3:
                break
    return {
        "applicable_sections": sections,
        "penalties": _filter_citations(_get_list(cr, "penalties"), sources),
        "procedure": _filter_citations(_get_list(cr, "procedure"), sources)
    }


def _get_civil_route(data: dict) -> Dict[str, Any]:
    cr = data.get("civil_route", {})
    if not isinstance(cr, dict):
        cr = {}
    return {
        "remedies": _get_list(cr, "remedies"),
        "compensation_range": _get_str(cr, "compensation_range", "compensation"),
        "procedure": _get_list(cr, "procedure")
    }


def _format_markdown(short_answer: str, is_this_illegal: str, criminal_route: dict, civil_route: dict,
                     compensation_claims: list, evidence_needed: list, practical_steps: list) -> str:
    def numbered(items: list, indent: str = "   ") -> str:
        return "\n".join(f"{indent}{i + 1}. {item}" for i, item in enumerate(items))

    crim_proc = numbered(criminal_route["procedure"])
    civ_proc = numbered(civil_route["procedure"])
    comp = "\n".join(f"• {c}" for c in compensation_claims)
    evid = "\n".join(f"• {e}" for e in evidence_needed)
    steps = numbered(practical_steps, indent="")
    sections = ", ".join(criminal_route["applicable_sections"])
    penalties = ", ".join(criminal_route["penalties"])
    remedies = ", ".join(civil_route["remedies"])

    parts = [f"SHORT ANSWER\n{short_answer}", f"IS THIS ILLEGAL?\n{is_this_illegal}"]

    parts.append("CRIMINAL ROUTE")
    if sections:
        parts.append(f"• Applicable Sections: {sections}")
    if penalties:
        parts.append(f"• Penalties: {penalties}")
    if crim_proc:
        parts.append("• Procedure:\n" + crim_proc)

    parts.append("CIVIL ROUTE")
    if remedies:
        parts.append(f"• Remedies: {remedies}")
    if civil_route["compensation_range"]:
        parts.append(f"• Compensation Range: {civil_route['compensation_range']}")
    if civ_proc:
        parts.append("• Procedure:\n" + civ_proc)

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
    return [{
        "section": s.get("section_number") or s.get("section") or "Unknown",
        "section_title": s.get("section_title") or "",
        "topic": s.get("topic") or "General",
        "source_act": s.get("act_name") or s.get("source_act") or s.get("source") or "Unknown Act",
        "courts": s.get("applicable_courts") or s.get("courts") or ["District Court"],
        "keywords": s.get("keywords") or [],
        "content_preview": (s.get("content") or s.get("text") or "")[:300],
        "content": s.get("content") or s.get("text") or ""
    } for s in sources]


class ResponseFormatter:
    """Format LLM JSON response into structured NyayGuru-style output"""

    def format_response(self, query: str, llm_response: str, sources: List[Dict],
                        domain: str, confidence: float) -> Dict[str, Any]:
        data = {}
        if llm_response and llm_response.strip():
            try:
                cleaned = re.sub(r'```(?:json)?\n?', '', llm_response).strip()
                data = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                pass

        short_answer = _get_str(data, "short_answer", default=llm_response[:500] if llm_response else "")
        is_this_illegal = _get_str(data, "is_this_illegal")
        criminal_route = _get_criminal_route(data, sources)
        civil_route = _get_civil_route(data)
        compensation_claims = _get_list(data, "compensation_claims")
        evidence_needed = _get_list(data, "evidence_needed")
        practical_steps = _get_list(data, "practical_steps")[:6]

        markdown = _format_markdown(short_answer, is_this_illegal, criminal_route, civil_route,
                                    compensation_claims, evidence_needed, practical_steps)

        return {
            "query": query,
            "response_type": domain,
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
            "status": "success"
        }


def format_legal_response(query: str, llm_response: str, sources: List[Dict],
                          domain: str, confidence: float) -> Dict[str, Any]:
    return ResponseFormatter().format_response(query, llm_response, sources, domain, confidence)
