"""
domain_config.py - Config-driven domain classification, source grounding and
response shaping. Loads config/domain_config.json with a TTL cache so edits
take effect without a server restart. Falls back to built-in defaults if the
file is missing or invalid.
"""

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "domain_config.json"

_REFRESH_TTL = float(os.getenv("DOMAIN_CONFIG_TTL", "60"))

_lock = threading.Lock()
_cache = {"loaded_at": 0.0, "config": None}


def _defaults() -> dict:
    """Built-in defaults so the app still works without a config file."""
    return {
        "domains": {
            "criminal": {"keywords": ["fir", "police", "crime", "criminal", "theft", "robbery"]},
            "civil": {"keywords": ["injunction", "damages", "compensation", "suit", "claim"]},
            "rent": {"keywords": ["rent", "landlord", "tenant", "eviction"]},
            "labor": {"keywords": ["wages", "salary", "employment", "employer"], "nb_class": "labour"},
            "family": {"keywords": ["marriage", "divorce", "custody"]},
            "defamation": {"keywords": ["defame", "slander", "libel"]},
            "cyber": {"keywords": ["online", "hack", "cyber"]},
            "consumer": {"keywords": ["consumer", "defective", "refund"]},
            "commercial": {"keywords": ["business", "company", "invoice"]},
        },
        "nb_class_map": {"labor": "labour"},
        "idioms": [],
        "negation_patterns": [],
        "negation_domains": ["criminal", "defamation", "cyber"],
        "classifier": {
            "keyword_weight": 0.6,
            "nb_weight": 0.4,
            "min_confidence": 0.3,
            "max_confidence": 0.95,
            "no_keyword_fallback_domain": "civil",
            "no_keyword_max_combined": 0.35,
            "secondary_min_keywords": 2,
        },
        "rules": [],
        "element_suppressions": {
            "theft": ["unauthorized_access", "data_breach", "data_theft", "financial_fraud"],
        },
        "response_types": {
            "primary_rules": {
                "criminal_and_civil": ["defamation", "cyber_defamation", "both_criminal_civil"],
                "criminal_only": ["criminal"],
                "civil_only": ["civil"],
            },
            "secondary_rules": {},
            "default": "primary",
        },
        "grounding": {"min_sources": 2, "domains": {}},
        "response_shaping": {"suppress_criminal_route": [], "suppress_civil_route": []},
    }


def _load() -> dict:
    cfg = _defaults()
    try:
        if _DEFAULT_CONFIG_PATH.exists():
            raw = json.loads(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cfg = _deep_merge(cfg, raw)
    except Exception as e:
        print(f"[domain_config] Failed to load {_DEFAULT_CONFIG_PATH}: {e}; using defaults")
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def get_config() -> dict:
    """Return the domain config, refreshing the file on a TTL interval."""
    now = time.monotonic()
    with _lock:
        cached = _cache["config"]
        if cached is None or now - _cache["loaded_at"] > _REFRESH_TTL:
            cached = _cache["config"] = _load()
            _cache["loaded_at"] = now
        return cached


def normalize_query(query: str) -> str:
    """Apply config idioms (e.g. 'wage theft' -> 'unpaid wages') to a query."""
    if not query:
        return query
    q = query
    idioms = sorted(
        (i for i in get_config().get("idioms", []) if i.get("match")),
        key=lambda i: len(i["match"]),
        reverse=True,
    )
    for idiom in idioms:
        q = re.sub(
            rf"\b{re.escape(idiom['match'])}\b",
            idiom.get("replace", ""),
            q,
            flags=re.IGNORECASE,
        )
    return q


def filter_sources_by_domain(sources: List[Dict], domain: str) -> List[Dict]:
    """Rank sources by domain fit. Domain is a SIGNAL, not a hard gate.

    In-domain sources always come first, in retrieval order. When there are
    not enough of them, the list is padded with the highest-ranked out-of-
    domain sources so the generation context is never empty but off-topic
    acts can never dominate or displace the in-domain ones (the old all-or-
    nothing fallback leaked unrelated acts into criminal queries, and the
    later return-[] behaviour starved the context whenever classification
    was imperfect, e.g. a job-fraud query labelled 'labor').

    `_grounded` is stamped on each source so callers can trace which sources
    matched the domain rule and which were only padding.
    """
    if not sources:
        return sources
    cfg = get_config()
    grounding = cfg.get("grounding", {})
    rule = grounding.get("domains", {}).get(domain)
    if not rule:
        for s in sources:
            s["_grounded"] = False
        return sources

    min_sources = int(grounding.get("min_sources", 2))
    soft = bool(grounding.get("soft", True))
    soft_max = int(grounding.get("soft_max_sources", 5))
    acts = [str(a).lower() for a in rule.get("acts", [])]
    topics = [str(t).lower() for t in rule.get("topics", [])]

    def matches(s: dict) -> bool:
        act = str(s.get("act_name") or s.get("source_act") or "").lower()
        topic = str(s.get("topic") or "").lower()
        if acts and any(a in act for a in acts):
            return True
        if topics and any(t in topic for t in topics):
            return True
        return False

    in_domain = [s for s in sources if matches(s)]
    for s in in_domain:
        s["_grounded"] = True

    out_of_domain = [s for s in sources if not matches(s)]
    for s in out_of_domain:
        s["_grounded"] = False

    if not soft:
        if not in_domain:
            return []
        if len(in_domain) >= min_sources:
            return in_domain
        return in_domain + out_of_domain[: max(0, min_sources - len(in_domain))]

    # Soft mode: keep in-domain first, pad to soft_max_sources total with the
    # best out-of-domain sources (never empty, never dominated by off-topic).
    return (in_domain + out_of_domain)[:soft_max]
