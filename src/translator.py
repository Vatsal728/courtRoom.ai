"""translator.py - Multi-language support via Meta's NLLB-200 (Phase 10)

Supports English -> Indian languages and back using
facebook/nllb-200-distilled-600M. Lazy-loads the model on first use so the
API server does not pay the load cost at startup.
"""
import logging
import re
import threading
from collections import OrderedDict

logger = logging.getLogger("courtroom-translator")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

FLORES_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}

# Unicode script ranges used to decide whether a query was typed in a
# non-English script (in which case it must be translated to English first).
_SCRIPT_RANGES = {
    "gu": r"[\u0a80-\u0aff]",
    "hi": r"[\u0900-\u097f]",
    "mr": r"[\u0900-\u097f]",
    "pa": r"[\u0a00-\u0a7f]",
    "bn": r"[\u0980-\u09ff]",
    "ta": r"[\u0b80-\u0bff]",
    "te": r"[\u0c00-\u0c7f]",
    "kn": r"[\u0c80-\u0cff]",
    "ml": r"[\u0d00-\u0d7f]",
    "ur": r"[\u0600-\u06ff]",
}


def _has_script(text: str, lang: str) -> bool:
    if lang not in _SCRIPT_RANGES:
        return False
    return bool(re.search(_SCRIPT_RANGES[lang], text))


def has_non_latin_script(text: str) -> bool:
    """True if the text contains any non-Latin (Indian) script characters."""
    return any(_has_script(text, lang) for lang in _SCRIPT_RANGES)


def detect_language(text: str) -> str:
    """Best-effort detection of the Indian language script in a query.

    Returns a supported language code, or 'en' when the text has no
    non-Latin (Indian) script characters. Devanagari is shared by Hindi
    and Marathi, so it resolves to 'hi' (the most common).
    """
    if not text:
        return "en"
    for lang in ("gu", "pa", "bn", "ta", "te", "kn", "ml", "ur"):
        if _has_script(text, lang):
            return lang
    if _has_script(text, "hi"):
        return "hi"
    return "en"


# Small LRU-style cache so repeated translations (e.g. source previews that
# recur across queries) skip the expensive model inference.
_TRANSLATION_CACHE = OrderedDict()
_MAX_CACHE_ENTRIES = 200


class NLLBTranslator:
    """Thread-safe singleton wrapper around facebook/nllb-200-distilled-600M."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._tokenizer = None
                    obj._model = None
                    obj._device = "cpu"
                    cls._instance = obj
        return cls._instance

    def _ensure_loaded(self):
        if self._tokenizer is not None:
            return
        with self._lock:
            if self._tokenizer is not None:
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading NLLB-200 translation model on %s...", device)
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            self._model.to(device)
            self._model.eval()
            self._device = device
            try:
                self._model.generation_config.max_length = None
            except Exception:
                pass
            logger.info("NLLB-200 translation model ready")

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "en") -> str:
        """Translate text between supported languages. Returns original on any failure."""
        if not text or not isinstance(text, str) or not text.strip():
            return text
        if src_lang == tgt_lang:
            return text

        src_flores = FLORES_CODES.get(src_lang)
        tgt_flores = FLORES_CODES.get(tgt_lang)
        if not src_flores or not tgt_flores:
            return text

        cache_key = (text, src_lang, tgt_lang)
        cached = _TRANSLATION_CACHE.get(cache_key)
        if cached is not None:
            _TRANSLATION_CACHE.move_to_end(cache_key)
            return cached

        try:
            self._ensure_loaded()
            import torch

            self._tokenizer.src_lang = src_flores
            tgt_token_id = self._tokenizer.convert_tokens_to_ids(tgt_flores)

            chunks = self._chunk(text)
            translated_chunks = []
            for chunk in chunks:
                if not chunk.strip():
                    translated_chunks.append(chunk)
                    continue
                inputs = self._tokenizer(
                    chunk, return_tensors="pt", truncation=True, max_length=600
                ).to(self._device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_token_id,
                        max_new_tokens=600,
                        num_beams=1,
                        do_sample=False,
                    )
                translated = self._tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True
                )[0]
                translated_chunks.append(translated)
            result_text = "".join(translated_chunks)
            if len(_TRANSLATION_CACHE) >= _MAX_CACHE_ENTRIES:
                _TRANSLATION_CACHE.popitem(last=False)
            _TRANSLATION_CACHE[cache_key] = result_text
            return result_text
        except Exception as e:
            logger.warning("NLLB translation failed (%s->%s): %s", src_lang, tgt_lang, e)
            return text

    def query_to_english(self, text: str, query_lang: str) -> str:
        """Translate a user query to English if it is not already English."""
        if query_lang == "en":
            return text
        if not has_non_latin_script(text):
            return text
        return self.translate(text, src_lang=query_lang, tgt_lang="en")

    def answer_in_language(self, text: str, tgt_lang: str) -> str:
        """Translate an English answer into the requested language."""
        if tgt_lang in ("en", None, ""):
            return text
        return self.translate(text, src_lang="en", tgt_lang=tgt_lang)

    @staticmethod
    def _chunk(text: str):
        """Split text on sentence/newline boundaries while keeping chunks small
        enough for the NLLB context window."""
        sentences = re.split(r"(?<=[.!?])\s+|(?<=\n)", text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > 500 and current:
                chunks.append(current)
                current = sentence
            elif sentence:
                current += sentence + " "
        if current.strip():
            chunks.append(current)
        return chunks or [text]


_nllb = None


def get_translator() -> NLLBTranslator:
    global _nllb
    if _nllb is None:
        _nllb = NLLBTranslator()
    return _nllb


# Backwards-compatible helpers retained from the original stub
def translate_to_english(guj_text: str) -> str:
    return get_translator().translate(guj_text, src_lang="gu", tgt_lang="en")


def translate_to_gujarati(eng_text: str) -> str:
    return get_translator().translate(eng_text, src_lang="en", tgt_lang="gu")