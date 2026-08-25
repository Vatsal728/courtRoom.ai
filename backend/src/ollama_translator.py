"""ollama_translator.py - Fast translations via Groq (primary) + Google Translate (fallback).

Groq Cloud (llama-3.1-8b-instant) is the primary translation engine.
Google Translate is the fast, always-reachable mid-tier.
Original text is returned as final fallback.
"""
import logging
import threading
import time
from typing import Dict, Optional

from src.google_translator import get_google_translator
from src.groq_translator import get_groq_translator
from src.translator import has_non_latin_script

logger = logging.getLogger("courtroom-ollama-translator")


class FastTranslator:
    """Composite translator: Groq primary, Google fallback, original text final.

    Groq gives the best legal quality but can be intermittently blocked by
    corporate web filters; Google Translate is fast and always reachable.
    Exposes batch_translate for efficient multi-field translation.
    """

    def __init__(self):
        self._groq = get_groq_translator()
        self._google = get_google_translator()

    @property
    def engine(self) -> str:
        if self._groq.enabled and not self._groq.down:
            return "groq"
        if self._google.enabled and not self._google.down:
            return "google"
        return "original"

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "en") -> str:
        """Translate single text via Groq -> Google -> original."""
        if tgt_lang in (None, "", "en") or src_lang == tgt_lang or not text.strip():
            return text
        for engine in (self._groq, self._google):
            if engine.enabled and not engine.down:
                translated = engine.translate(text, src_lang, tgt_lang)
                if translated is not None:
                    return translated
        return text

    def answer_in_language(self, text: str, tgt_lang: str) -> str:
        """Translate an English answer into the requested language."""
        return self.translate(text, "en", tgt_lang)

    def query_to_english(self, text: str, query_lang: str) -> str:
        """Translate a user query to English if it is not already English."""
        if query_lang == "en":
            return text
        if not has_non_latin_script(text):
            return text
        return self.translate(text, query_lang, "en")

    def batch_translate(self, fields: Dict[str, str], tgt_lang: str, deadline: Optional[float] = None) -> Dict[str, str]:
        """Batch translate multiple fields via Groq -> Google -> original.

        Args:
            fields: Dict of {field_name: english_text}
            tgt_lang: Target language code
            deadline: Optional time.monotonic() deadline for translation budget

        Returns:
            Dict of translated fields (same keys). Untranslated fields keep original.
        """
        if tgt_lang in (None, "", "en") or not fields:
            return fields.copy()

        result = {}
        remaining = {k: v for k, v in fields.items() if v and v.strip()}

        # 1. Groq batch (best quality; 1 call for all fields).
        if self._groq.enabled and not self._groq.down:
            groq_result = self._groq.translate_fields(remaining, tgt_lang)
            if groq_result:
                result.update(groq_result)
                remaining = {k: v for k, v in remaining.items() if k not in groq_result}
                logger.debug("Groq batch translated %d/%d fields to %s", len(groq_result), len(fields), tgt_lang)

        # 2. Google Translate per-field (fast, always reachable).
        if remaining and self._google.enabled and not self._google.down:
            if deadline and time.monotonic() > deadline:
                logger.debug("Translation deadline exceeded before Google fallback")
            else:
                google_result = self._google.translate_fields(remaining, tgt_lang)
                if google_result:
                    result.update(google_result)
                    remaining = {k: v for k, v in remaining.items() if k not in google_result}
                    logger.debug("Google translated %d/%d fields to %s", len(google_result), len(fields), tgt_lang)

        # 3. Original text for any still remaining.
        for key, text in remaining.items():
            if key not in result:
                result[key] = text

        return result


_fast_instance = None
_fast_lock = threading.Lock()


def get_fast_translator() -> FastTranslator:
    global _fast_instance
    if _fast_instance is None:
        with _fast_lock:
            if _fast_instance is None:
                _fast_instance = FastTranslator()
    return _fast_instance