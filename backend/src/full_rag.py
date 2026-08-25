import os
import sys
import io
import json
import re
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline as ImprovedRAGPipeline
from src.llm_router import LLMRouter
from src.domain_classifier import DomainClassifier
from src.domain_config import filter_sources_by_domain
from src.applicability import gate_sources, applicability_confidence
from src.fact_extractor import extract_facts
from src.response_formatter import format_legal_response
from typing import Dict, Any

try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = None


# ── Legal-status filtering ─────────────────────────────────────────────
# A current question must never be answered with a repealed law. These
# helpers detect historical questions (explicit year or old-act name) and,
# for current questions, demote historical sources in favour of their
# replacement act.

_CURRENT_ACT_NAME_HINTS = (
    "bharatiya nyaya sanhita", "bns ", "bnss", "bharatiya nagarik", "bharatiya sakshya",
    "bsa ", "code on wages", "code on social security", "occupational safety", "oshwc",
    "industrial relations code", "consumer protection act 2019", "digital personal data protection",
    "dpdp", "contract act",
)

_HISTORICAL_ACT_PATTERNS = (
    r"\bindian penal code\b", r"\bipc\b", r"\bcrpc\b", r"\bcode of criminal procedure\b",
    r"\bindian evidence act\b", r"\biea\b", r"\bpayment of wages act\b",
    r"\bminimum wages act\b", r"\bindustrial disputes act\b",
)

_YEAR_RE = re.compile(r"\b(19\d{2}|20[0-2]\d)\b")


def _is_historical_query(query: str) -> bool:
    """True when the query explicitly targets an old law or a past point in time."""
    q = query.lower()
    if any(hint in q for hint in _CURRENT_ACT_NAME_HINTS):
        return False
    if _YEAR_RE.search(query):
        return True
    return any(re.search(p, q) for p in _HISTORICAL_ACT_PATTERNS)


def _status_label(s: dict) -> str:
    status = s.get("status") or "active"
    if status == "historical":
        rb = s.get("replaced_by")
        return f"HISTORICAL - replaced by {rb}" if rb else "HISTORICAL"
    if status == "pending":
        ef = s.get("effective_from")
        return f"PENDING - not yet in force (effective {ef})" if ef else "PENDING"
    return "ACTIVE"


def _apply_legal_filter(sources: list, query: str, pipeline) -> list:
    """Two-stage retrieval: demote repealed/historical sources on current
    questions and pull in the replacement act's sections."""
    for s in sources:
        s["_label"] = _status_label(s)

    if _is_historical_query(query):
        return sources

    active = [s for s in sources if (s.get("status") or "active") in ("active", "pending")]
    hist = [s for s in sources if (s.get("status") or "active") == "historical"]
    if not hist:
        return sources

    replacements = []
    seen_keys = {(s.get("act_name"), s.get("section_number")) for s in active}
    for s in hist:
        rb = s.get("replaced_by")
        if not rb:
            continue
        for r in pipeline.retrieve_by_act(query, rb, top_k=2):
            key = (r.get("act_name"), r.get("section_number"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            r["_label"] = _status_label(r)
            r["_replacing"] = s.get("act_name")
            replacements.append(r)

    merged = active + replacements if (active or replacements) else sources
    return merged[:5]


class RAGError(Exception):
    def __init__(self, message: str, user_message: str = None):
        self.user_message = user_message or "The legal research system encountered an issue."
        super().__init__(message)
class RetrievalError(RAGError):
    def __init__(self, message: str):
        super().__init__(message, "Could not retrieve relevant legal documents. The knowledge base may be unavailable.")


class ClassificationError(RAGError):
    def __init__(self, message: str):
        super().__init__(message, "Could not classify the legal domain of your query.")


class LLMError(RAGError):
    def __init__(self, message: str):
        super().__init__(message, "The AI legal assistant is temporarily unavailable. Please try again.")


class FullRAGSystem:
    """Complete RAG pipeline: NLP -> Classify -> Retrieve -> Generate"""

    def __init__(self):
        self.domain_classifier = DomainClassifier()
        self.improved_rag = ImprovedRAGPipeline()
        self.llm_router = LLMRouter()
        self._cache = TTLCache(maxsize=100, ttl=3600) if TTLCache else None

        os.environ["CHROMA_TELEMETRY"] = "false"
        os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"

    def build_knowledge_base(self):
        return self.improved_rag.build_knowledge_base()

    def _cache_key(self, query: str) -> str:
        return query.lower().strip()

    @staticmethod
    def _debug(*parts):
        if os.getenv("RAG_DEBUG", "0") == "1":
            print("[RAG-DEBUG]", *parts, flush=True)

    def prepare_context(self, query: str) -> Dict[str, Any]:
        """Shared analysis pipeline: facts -> classify -> retrieve -> gate.

        Returns a dict holding the normalized query, domain/confidence, the
        extracted facts, the verified (applicable) sources only, the rejected
        candidates with reasons, the LLM context/available strings, and a
        `needs_more_info` flag with clarifying questions when the facts are
        insufficient to identify the applicable law.
        """
        normalized = self.domain_classifier.normalize_query(query)
        facts = extract_facts(normalized)

        self._debug("QUERY:", normalized)
        self._debug("FACTS:", json.dumps(facts, ensure_ascii=False, default=str))

        if facts.get("blocking"):
            self._debug("BLOCKING: jurisdiction-critical facts missing")
            return {
                "query": normalized,
                "needs_more_info": True,
                "facts": facts,
                "questions": facts.get("needs_facts") or [],
                "missing_facts": facts.get("missing_facts") or [],
            }

        domain, domain_confidence, secondary = self.domain_classifier.classify(normalized)
        self._debug("CLASSIFICATION:", domain, domain_confidence, secondary)

        try:
            if facts.get("location_known") and facts.get("location"):
                candidates = self.improved_rag.search_by_jurisdiction(
                    normalized, facts["location"], top_k=5)
            else:
                candidates = self.improved_rag.retrieve_with_metadata(normalized, top_k=5)
        except Exception as e:
            raise RetrievalError(str(e))
        self._debug("RETRIEVED:", [
            f"{c.get('act_name')} {c.get('section_number')} ({c.get('topic')})" for c in candidates
        ])

        # Domain is a signal, not a gate: rank in-domain first, never empty.
        candidates = filter_sources_by_domain(candidates, domain)
        # Current-law demotion + replacement-act mapping on current questions.
        candidates = _apply_legal_filter(candidates, normalized, self.improved_rag)

        gated = gate_sources(normalized, normalized, candidates)
        applicable = gated["applicable"]
        rejected = gated["rejected"]
        self._debug("APPLICABLE:", [f"{c.get('act_name')} {c.get('section_number')}" for c in applicable])
        self._debug("REJECTED:", [
            f"{c.get('act_name')} {c.get('section_number')}: {c.get('reason')}" for c in rejected
        ])

        context = "\n\n".join(
            f"[{s.get('act_name') or s.get('source_act')} Section "
            f"{s.get('section_number') or s.get('section')} - {s.get('section_title') or ''} "
            f"({s.get('_label') or _status_label(s)})]\n{s['content']}"
            for s in applicable
        )
        if any((s.get('status') or 'active') == 'historical' for s in applicable):
            context += ("\n\n[NOTE] Some context sources are historical/repealed acts, included "
                        "only because the question references them or their era. Where the "
                        "context also contains the current replacement law, prefer it.")
        available = "\n".join(
            f"{s.get('act_name') or s.get('source_act')} Section "
            f"{s.get('section_number') or s.get('section')}: {s.get('section_title') or ''}"
            for s in applicable
        )

        app_conf = applicability_confidence(applicable, candidates)
        return {
            "query": normalized,
            "needs_more_info": False,
            "facts": facts,
            "domain": domain,
            "domain_confidence": domain_confidence,
            "secondary": secondary,
            "response_type": self.domain_classifier.get_response_type(domain, secondary),
            "sources": applicable,
            "rejected": rejected,
            "all_candidates": candidates,
            "context": context,
            "available": available,
            "confidence_details": {
                "domain_confidence": domain_confidence,
                "applicability_confidence": app_conf,
                "overall_confidence": round((domain_confidence + app_conf) / 2, 2),
            },
        }

    @staticmethod
    def _needs_more_info_response(prep: Dict[str, Any]) -> Dict[str, Any]:
        questions = prep.get("questions") or []
        short = "I need a few more details before I can identify the exact law that applies."
        body = short + ("\n\n" + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)) if questions else "")
        return {
            "query": prep.get("query", ""),
            "status": "needs_more_info",
            "response_type": "needs_more_info",
            "confidence_score": 0.0,
            "short_answer": short,
            "full_response": body,
            "response": body,
            "missing_facts": prep.get("missing_facts") or [],
            "questions": questions,
            "sources": [],
            "applicable_laws": {},
            "cached": False,
        }

    @staticmethod
    def _no_applicable_response(prep: Dict[str, Any]) -> Dict[str, Any]:
        short = ("The retrieved provisions do not establish an applicable rule for these facts. "
                 "The specific law may need more detail from you, or it may not be in the knowledge base.")
        body = short
        rejected = prep.get("rejected") or []
        if rejected:
            body += "\n\nRetrieved candidates were checked and did not match the stated facts:"
            for r in rejected[:5]:
                body += f"\n- {r.get('act_name')} {r.get('section_number')}: {r.get('reason')}"
        return {
            "query": prep.get("query", ""),
            "status": "success",
            "response_type": prep.get("domain") or "general",
            "confidence_score": prep.get("confidence_details", {}).get("overall_confidence", 0.0),
            "short_answer": short,
            "full_response": body,
            "response": body,
            "sources": [],
            "applicable_laws": {},
            "cached": False,
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            prep = self.prepare_context(query)
            cache_key = self._cache_key(prep["query"])
            if self._cache and cache_key in self._cache:
                print("1\ufe0f\u20e3  Cache hit!")
                cached = self._cache[cache_key]
                cached["cached"] = True
                return cached

            if prep["needs_more_info"]:
                print("\u2139\ufe0f  Insufficient facts; asking clarifying questions.")
                return self._needs_more_info_response(prep)

            sources = prep["sources"]
            if not sources:
                print("\u26a0\ufe0f  No applicable provision after the gate.")
                return self._no_applicable_response(prep)

            print("3\ufe0f\u20e3  Generating legal analysis...")
            try:
                llm_response = self.llm_router.generate_response(prep["context"], prep["query"], prep["available"])
            except Exception as e:
                raise LLMError(str(e))

            print("4\ufe0f\u20e3  Formatting response (NyayGuru-style)...")
            try:
                formatted_response = format_legal_response(
                    query=prep["query"],
                    llm_response=llm_response,
                    sources=sources,
                    domain=prep["domain"],
                    confidence=prep["domain_confidence"],
                    response_type=prep["response_type"]
                )
            except Exception as e:
                raise RAGError(str(e), "Failed to format the legal response.")

            print("5\ufe0f\u20e3  Adding metadata...")
            if not formatted_response.get("response"):
                formatted_response["response"] = formatted_response.get("full_response") or formatted_response.get("short_answer")
            formatted_response["domain"] = formatted_response.get("response_type")
            formatted_response["confidence"] = formatted_response.get("confidence_score")
            formatted_response["confidence_details"] = prep["confidence_details"]
            formatted_response["applicability"] = {
                "applicable": [f"{s.get('act_name')} {s.get('section_number')}" for s in sources],
                "rejected": [
                    {"source": f"{s.get('act_name')} {s.get('section_number')}", "reason": s.get("reason")}
                    for s in prep["rejected"]
                ],
            }
            formatted_response["missing_facts"] = prep["facts"].get("missing_facts") or []
            formatted_response["stored"] = True

            if self._cache and not self.llm_router.last_was_error:
                self._cache[cache_key] = formatted_response

            print("\u2705 Response complete!")
            return formatted_response

        except RetrievalError as e:
            return {"error": e.user_message, "status": "failed"}
        except ClassificationError as e:
            return {"error": e.user_message, "status": "failed"}
        except LLMError as e:
            return {"error": e.user_message, "status": "failed"}
        except RAGError as e:
            return {"error": e.user_message, "status": "failed"}
        except Exception as e:
            return {"error": "An unexpected error occurred.", "status": "failed"}


if __name__ == "__main__":
    system = FullRAGSystem()

    test_queries = [
        "Someone hacked my website, stole my company's database, and published it online.",
        "A person assaulted me outside the market and stole my wallet.",
        "The landlord is trying to evict me."
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = system.process_query(query)
        print(f"Result: {result}")
