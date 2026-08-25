"""google_translator.py - Fast translations via the free Google Translate endpoint.

Groq can be intermittently blocked by corporate web filters, and local Ollama is
too slow for long answers. This client talks directly to
translate.googleapis.com (no API key, no rate-tier) and is used as the
reliable mid-tier in the FastTranslator chain:

    Groq (best quality) -> Google (fast + always reachable) -> Ollama -> original

Keeps the same interface as GroqClient (translate / translate_fields).
"""
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("courtroom-google-translator")

_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

# Free endpoint silently truncates very long queries; keep calls well under it.
_MAX_CHUNK_CHARS = 4500

_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_MAX_CACHE = 500


class GoogleClient:
    """Minimal client for the free Google Translate endpoint."""

    def __init__(self):
        self.enabled = os.getenv("GOOGLE_TRANSLATION_ENABLED", "1") == "1"
        self.timeout = float(os.getenv("GOOGLE_TRANSLATION_TIMEOUT", "15"))
        self.verify = os.getenv("GOOGLE_SSL_VERIFY", os.getenv("GROQ_SSL_VERIFY", "1")) == "1"
        self._cooldown = float(os.getenv("GOOGLE_TRANSLATION_COOLDOWN", "30"))
        self._down_until = 0.0
        self._failures = 0

    @property
    def down(self) -> bool:
        return self.enabled and time.monotonic() < self._down_until

    def _mark_down(self):
        self._failures += 1
        self._down_until = time.monotonic() + self._cooldown
        logger.warning(
            "Google translation unavailable (%s failures); falling back for %ss",
            self._failures, int(self._cooldown),
        )

    @staticmethod
    def _chunk(text: str) -> List[str]:
        """Split long text on paragraph/sentence boundaries, preserving structure."""
        if len(text) <= _MAX_CHUNK_CHARS:
            return [text]
        chunks = []
        for part in re.split(r"(\n\s*\n)", text):
            if not part.strip():
                chunks.append(part)
                continue
            while len(part) > _MAX_CHUNK_CHARS:
                cut = part.rfind(" ", 0, _MAX_CHUNK_CHARS)
                if cut < _MAX_CHUNK_CHARS // 2:
                    cut = _MAX_CHUNK_CHARS
                chunks.append(part[:cut])
                part = part[cut:]
            if part.strip():
                chunks.append(part)
        return chunks or [text]

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "en") -> Optional[str]:
        """Translate a single text. Returns None when it cannot (caller falls back)."""
        if not self.enabled:
            return None
        if tgt_lang in (None, "", "en") or src_lang == tgt_lang or not text.strip():
            return text
        if self.down:
            return None

        cache_key = (text, src_lang, tgt_lang)
        cached = _CACHE.get(cache_key)
        if cached is not None:
            _CACHE.move_to_end(cache_key)
            return cached

        parts = []
        for chunk in self._chunk(text):
            if not chunk.strip():
                parts.append(chunk)
                continue
            translated = self._translate_chunk(chunk, src_lang, tgt_lang)
            if translated is None:
                return None
            parts.append(translated)

        result = "".join(parts)
        if len(_CACHE) >= _MAX_CACHE:
            _CACHE.popitem(last=False)
        _CACHE[cache_key] = result
        return result

    def translate_fields(self, fields: Dict[str, str], tgt_lang: str) -> Dict[str, str]:
        """Translate a dict of fields. Each field is one quick HTTP call (no key,
        no rate-tier), so batching here is just a loop. Returns only the fields
        that succeeded; caller keeps original text for the rest."""
        if not self.enabled or self.down or tgt_lang in (None, "", "en") or not fields:
            return {}
        out = {}
        for key, text in fields.items():
            if not text or not text.strip():
                continue
            translated = self.translate(text, "en", tgt_lang)
            if translated is not None:
                out[key] = translated
        return out

    def _translate_chunk(self, chunk: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        params = {
            "client": "gtx",
            "sl": src_lang,
            "tl": tgt_lang,
            "dt": "t",
            "q": chunk,
        }
        try:
            with httpx.Client(timeout=self.timeout, verify=self.verify) as client:
                resp = client.get(_TRANSLATE_URL, params=params)
            if resp.status_code != 200:
                logger.warning(
                    "Google translate HTTP %s (%s->%s)", resp.status_code, src_lang, tgt_lang
                )
                self._mark_down()
                return None
            data = resp.json()
            segments = data[0] if isinstance(data, list) and data else None
            if not segments:
                logger.warning("Google translate returned empty segments")
                self._mark_down()
                return None
            translated = "".join(str(seg[0]) for seg in segments if isinstance(seg, list) and seg)
            if not translated.strip():
                return None
            return translated
        except Exception as e:
            logger.warning("Google translate failed (%s->%s): %s", src_lang, tgt_lang, e)
            self._mark_down()
            return None


_google_instance = None
_google_lock = None


def get_google_translator() -> GoogleClient:
    global _google_instance, _google_lock
    if _google_instance is None:
        if _google_lock is None:
            import threading
            _google_lock = threading.Lock()
        with _google_lock:
            if _google_instance is None:
                _google_instance = GoogleClient()
    return _google_instance
