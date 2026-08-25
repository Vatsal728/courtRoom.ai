"""
embedding_provider.py - Pluggable embedding providers.

The master knowledge base stays on Ollama nomic-embed-text and is never
re-embedded. Per-user uploaded documents use Google Gemini embeddings
(gemini-embedding-001 by default) through this interface, so the two vector
spaces never mix inside one collection.

Switch with EMBEDDING_PROVIDER=google|ollama in .env
(default: google for per-user documents).
"""
import os
import threading
import time
import urllib.request as _req
from abc import ABC, abstractmethod
from typing import List

from dotenv import load_dotenv

load_dotenv(override=True)

_GOOGLE_DOC_TASK = "RETRIEVAL_DOCUMENT"
_GOOGLE_QUERY_TASK = "RETRIEVAL_QUERY"
_MAX_CHARS = 24000  # well under gemini-embedding-001's 8K-token context
_RATE_LIMIT_RETRY_SLEEP = 2.0


class EmbeddingProvider(ABC):
    """Minimal embed interface (duck-compatible with LangChain embedders)."""

    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text (list length preserved)."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Return a single query vector."""


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embeddings (free tier) for per-user uploaded documents.

    - Batch embed with TaskType RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY.
    - Exact-text caching to avoid burning the free daily quota.
    - Rate-limit backoff on 429 / quota errors.
    """

    name = "google"

    def __init__(self, api_key: str = None, model: str = None, batch_size: int = None):
        import google.generativeai as genai

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GoogleEmbeddingProvider")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model or os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")
        self.batch_size = batch_size or int(os.getenv("GOOGLE_EMBED_BATCH", "64"))
        self.dimension = int(os.getenv("GOOGLE_EMBED_DIM", "3072"))
        self._cache = {}
        self._lock = threading.Lock()

    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        out: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        missing: List[str] = []
        missing_idx: List[int] = []
        with self._lock:
            for i, t in enumerate(texts):
                key = (task, t)
                cached = self._cache.get(key)
                if cached is not None:
                    out[i] = cached
                else:
                    missing_idx.append(i)
                    missing.append(t)
            pending = list(zip(missing_idx, missing))
        if not pending:
            return out

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            payload = [text[: _MAX_CHARS] for _, text in batch]
            while True:
                try:
                    res = self._genai.embed_content(
                        model=self.model, content=payload, task_type=task
                    )
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "quota" in msg or "resource exhausted" in msg:
                        time.sleep(_RATE_LIMIT_RETRY_SLEEP)
                        continue
                    raise
            vectors = res["embedding"]
            for (idx, text), vec in zip(batch, vectors):
                v = list(vec)
                out[idx] = v
                with self._lock:
                    self._cache[(task, text)] = v
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts), _GOOGLE_DOC_TASK)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], _GOOGLE_QUERY_TASK)[0]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama nomic-embed-text (used for the master KB / explicit opt-in)."""

    name = "ollama"

    def __init__(self, base_url: str = None, model: str = None, batch_size: int = 50):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.batch_size = batch_size

    @staticmethod
    def _call(base_url: str, model: str, texts: List[str]) -> List[List[float]]:
        body = ({"model": model, "input": texts}).__str__().encode()
        # urllib needs a JSON body; build it explicitly
        import json
        body = json.dumps({"model": model, "input": texts}).encode()
        req = _req.Request(
            f"{base_url}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _req.urlopen(req, timeout=120) as resp:
            import json as _json
            return _json.loads(resp.read())["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(texts)
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = ["search_document: " + t for t in texts[i:i + self.batch_size]]
            results.extend(self._call(self.base_url, self.model, batch))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._call(self.base_url, self.model, ["search_query: " + text])[0]


def get_embedding_provider(name: str = None) -> EmbeddingProvider:
    """Factory. Defaults to Google unless EMBEDDING_PROVIDER=ollama."""
    selected = (name or os.getenv("EMBEDDING_PROVIDER", "google")).strip().lower()
    if selected == "ollama":
        return OllamaEmbeddingProvider()
    return GoogleEmbeddingProvider()
