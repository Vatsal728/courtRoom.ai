import os
import sys
import io
import json
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
from src.response_formatter import format_legal_response
from typing import Dict, Any

try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = None


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

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            cache_key = self._cache_key(query)
            if self._cache and cache_key in self._cache:
                print("1\ufe0f\u20e3  Cache hit!")
                cached = self._cache[cache_key]
                cached["cached"] = True
                return cached

            print("1\ufe0f\u20e3  Classifying domain...")
            domain, domain_confidence, secondary = self.domain_classifier.classify(query)

            print("2\ufe0f\u20e3  Retrieving relevant laws...")
            try:
                sources = self.improved_rag.retrieve_with_metadata(query, top_k=5)
            except Exception as e:
                raise RetrievalError(str(e))

            print("3\ufe0f\u20e3  Generating legal analysis...")
            context = "\n\n".join(
                f"[{s.get('act_name') or s.get('source_act')} Section "
                f"{s.get('section_number') or s.get('section')} - {s.get('section_title') or ''}]\n{s['content']}"
                for s in sources
            )
            available = "\n".join(
                f"{s.get('act_name') or s.get('source_act')} Section "
                f"{s.get('section_number') or s.get('section')}: {s.get('section_title') or ''}"
                for s in sources
            )
            try:
                llm_response = self.llm_router.generate_response(context, query, available)
            except Exception as e:
                raise LLMError(str(e))

            print("4\ufe0f\u20e3  Formatting response (NyayGuru-style)...")
            try:
                formatted_response = format_legal_response(
                    query=query,
                    llm_response=llm_response,
                    sources=sources,
                    domain=domain,
                    confidence=domain_confidence
                )
            except Exception as e:
                raise RAGError(str(e), "Failed to format the legal response.")

            print("5\ufe0f\u20e3  Adding metadata...")
            if not formatted_response.get("response"):
                formatted_response["response"] = formatted_response.get("full_response") or formatted_response.get("short_answer")
            formatted_response["domain"] = formatted_response.get("response_type")
            formatted_response["confidence"] = formatted_response.get("confidence_score")
            formatted_response["stored"] = True

            if self._cache:
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
