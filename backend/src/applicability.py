"""
applicability.py - Rule-based applicability gate.

Retrieval relevance is NOT legal applicability. A vector search can surface a
semantically similar provision that does not apply to the facts (e.g. a cruelty
section retrieved for a wallet theft, or an IT Act access/search-power section
retrieved for hacking).

This module compares each retrieved candidate's text against the *legal
elements* raised by the query. A candidate is marked `applicable` only when it
supports at least one element the query actually raises; otherwise it is
`rejected` with a human-readable reason. The generation step may only cite
`applicable` sources.

Fully config-driven: element -> keyword list lives in
config/domain_config.json under "legal_elements". Elements flagged
`"generic": true` (e.g. bare 'computer') never rescue a candidate on their
own when the query raises a specific element the candidate does not support.
"""
import re
from typing import Dict, List

from src.domain_config import get_config


def _contains_any(text: str, keywords: List[str]) -> bool:
    tl = text.lower()
    return any(k in tl for k in keywords)


def _query_elements(normalized_query: str) -> List[str]:
    """Return the legal-element names raised by the query.

    Config-driven suppressions (config/domain_config.json ->
    "element_suppressions"): when the query raises a digital element
    (e.g. unauthorized_access / data_breach / data_theft), physical-property
    elements like "theft" are dropped, because "stole my company's database"
    is data theft (IT Act 43/66), not theft of movable property (BNS 303).
    """
    cfg = get_config().get("legal_elements", {})
    raised = [name for name, spec in cfg.items() if _contains_any(normalized_query, spec.get("keywords", []))]
    raised_set = set(raised)
    for name, suppress_when in (get_config().get("element_suppressions") or {}).items():
        if name in raised_set and any(s in raised_set for s in suppress_when):
            raised_set.discard(name)
    return [e for e in raised if e in raised_set]


def _source_elements(source: dict) -> List[str]:
    """Return the legal-element names supported by a candidate's text/title."""
    cfg = get_config().get("legal_elements", {})
    text = " ".join([
        str(source.get("section_title") or ""),
        str(source.get("section_number") or ""),
        str(source.get("content") or source.get("text") or ""),
    ])
    return [name for name, spec in cfg.items() if _contains_any(text, spec.get("keywords", []))]


def _is_generic(name: str) -> bool:
    cfg = get_config().get("legal_elements", {})
    return bool(cfg.get(name, {}).get("generic"))


def _known_section_for(q_elements: List[str], source: dict) -> List[str]:
    """Return the query elements a source satisfies via an authoritative
    (act, section) mapping, used for terse cross-reference sections whose
    body text carries no element keywords (e.g. IT Act s.66 -> 'does any act
    referred to in s.43')."""
    cfg = get_config().get("known_sections", {})
    act = str(source.get("act_name") or source.get("source_act") or "").lower()
    num = re.sub(r"^[^0-9]+", "", re.sub(r"[^0-9a-z]", "", str(source.get("section_number") or "").lower()))
    matched = []
    for element in q_elements:
        for rule in cfg.get(element, []):
            if rule.get("act") in act and num in [str(n).lower() for n in rule.get("section", [])]:
                matched.append(element)
                break
    return matched


def gate_sources(query: str, normalized_query: str, sources: List[Dict]) -> Dict[str, list]:
    """Split candidates into `applicable` and `rejected`, each with a reason.

    Returns {"applicable": [...], "rejected": [...]} preserving retrieval order
    within each bucket. When the query raises no detectable legal element the
    gate is conservative: it keeps every candidate (so answers are never
    starved) and reports low applicability confidence via the returned ratio.
    """
    q_elements = _query_elements(normalized_query or query)

    if not q_elements:
        for s in sources:
            s["applicable"] = True
            s["matched_elements"] = []
            s["reason"] = "No specific legal element detected in the query; candidate kept for review."
        return {"applicable": list(sources), "rejected": []}

    applicable, rejected = [], []
    for s in sources:
        src_elements = _source_elements(s)
        overlap = [e for e in src_elements if e in q_elements]
        specific_overlap = [e for e in overlap if not _is_generic(e)]
        known = _known_section_for(q_elements, s)

        if specific_overlap:
            s["applicable"] = True
            s["matched_elements"] = specific_overlap
            s["reason"] = ("Provision supports legal element(s): %s, matching the stated facts."
                           % ", ".join(specific_overlap))
            applicable.append(s)
            continue

        if known:
            s["applicable"] = True
            s["matched_elements"] = known
            s["reason"] = ("Known %s provision for the legal element(s): %s."
                           % (str(s.get("act_name") or "legal"), ", ".join(known)))
            applicable.append(s)
            continue

        if overlap:
            # Only generic overlap (e.g. bare 'computer'/'data') while the
            # query also raises a specific element this source does not support:
            # treat as a lexical near-miss, not applicability.
            s["applicable"] = False
            s["matched_elements"] = overlap
            s["reason"] = ("Provision shares only generic term(s) (%s) with the facts; it does not "
                           "establish the specific rule the query raises (%s)."
                           % (", ".join(overlap), ", ".join(q_elements)))
            rejected.append(s)
            continue

        if src_elements:
            s["applicable"] = False
            s["matched_elements"] = src_elements
            s["reason"] = ("Provision concerns %s, which the stated facts do not raise (%s)."
                           % (", ".join(src_elements), ", ".join(q_elements)))
        else:
            s["applicable"] = False
            s["matched_elements"] = []
            s["reason"] = ("No legal element from the query is found in this provision's text or title.")
        rejected.append(s)

    return {"applicable": applicable, "rejected": rejected}


def applicability_confidence(applicable: List[dict], candidates: List[dict]) -> float:
    """Fraction of candidates that survived the gate (0..1)."""
    if not candidates:
        return 0.0
    return round(len(applicable) / len(candidates), 2)
