"""groq_translator.py - Fast translations via Groq Cloud (llama-3.1-8b-instant).

Primary translation engine. Returns None when rate-limited or unavailable so
the FastTranslator can fall through to Google, then original text.
"""
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import httpx
from groq import Groq

logger = logging.getLogger("courtroom-groq-translator")

_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}

_PROMPT_TEMPLATE = """You are a professional legal translator. Translate the following English text into {tgt_lang}.

Rules:
- Output ONLY the {tgt_lang} translation. No explanations, no quotes, no prefixes.
- Keep numbers, section references, bullet markers (like "-" or "1.") and any formatting intact.
- Use natural, plain {tgt_lang} that a layperson can understand.

English text:
{source}"""

_MAX_CHUNK_CHARS = 450

_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_MAX_CACHE = 500

_FILTER_BLOCK_COOLDOWN = float(os.getenv("GROQ_FILTER_BLOCK_COOLDOWN", "300"))


def _brief_error(err) -> str:
    """Short, log-friendly version of an exception message (never the body)."""
    text = " ".join(str(err).strip().split())
    return text[:120] + ("…" if len(text) > 120 else "")


def _is_filter_blocked(err) -> bool:
    """True when the error looks like a web-filter / 403 block page."""
    msg = str(err).lower()
    return any(h in msg for h in ("403", "blocked", "forbidden", "fortiguard", "web filter"))


class GroqClient:
    """Thin client for Groq chat completions used as the primary translation engine."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_TRANSLATE_MODEL", "llama-3.1-8b-instant")
        self.enabled = os.getenv("GROQ_TRANSLATION_ENABLED", "1") == "1" and bool(self.api_key)
        self.timeout = float(os.getenv("GROQ_TRANSLATION_TIMEOUT", "30"))
        self._cooldown = float(os.getenv("GROQ_TRANSLATION_COOLDOWN", "60"))
        self._down_until = 0.0
        self._failures = 0
        if self.enabled:
            http_client = httpx.Client(timeout=self.timeout, verify=os.getenv("GROQ_SSL_VERIFY", "1") == "1")
            self._client = Groq(
                api_key=self.api_key,
                http_client=http_client,
                max_retries=int(os.getenv("GROQ_MAX_RETRIES", "1")),
            )
        else:
            self._client = None

    @property
    def down(self) -> bool:
        return self.enabled and time.monotonic() < self._down_until

    def _mark_down(self, cooldown: Optional[float] = None):
        self._failures += 1
        self._down_until = time.monotonic() + (cooldown if cooldown is not None else self._cooldown)
        logger.warning(
            "Groq translation unavailable (%s failures); falling back for %ss",
            self._failures, int(self._down_until - time.monotonic()),
        )

    @staticmethod
    def _num_predict(text: str) -> int:
        return min(2048, max(64, int(len(text) * 1.5) + 64))

    @staticmethod
    def _chunk(text: str) -> List[str]:
        """Split on paragraph/sentence boundaries into small callable pieces.

        Blank-line separators are preserved as their own entries so the joined
        translation keeps the markdown structure.
        """
        import re
        chunks = []
        for part in re.split(r"(\n\s*\n)", text):
            if not part.strip():
                chunks.append(part)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", part)
            current = ""
            for sentence in sentences:
                if current and len(current) + len(sentence) > _MAX_CHUNK_CHARS:
                    chunks.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current.strip():
                chunks.append(current)
        return chunks or [text]

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "en") -> Optional[str]:
        """Translate via Groq. Returns None when it cannot (caller falls back)."""
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
            translated = self._translate_chunk(chunk, tgt_lang)
            if translated is None:
                return None
            parts.append(translated)

        result = "".join(parts)
        if len(_CACHE) >= _MAX_CACHE:
            _CACHE.popitem(last=False)
        _CACHE[cache_key] = result
        return result

    def _translate_chunk(self, chunk: str, tgt_lang: str) -> Optional[str]:
        prompt = _PROMPT_TEMPLATE.format(
            tgt_lang=_LANG_NAMES.get(tgt_lang, tgt_lang), source=chunk
        )
        last_error = None
        marked = False
        for attempt in range(2):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=self._num_predict(chunk),
                    stream=False,
                    timeout=self.timeout,
                )
                translated = str(resp.choices[0].message.content or "").strip()
                if not translated:
                    last_error = "empty response"
                    continue
                return translated
            except Exception as e:
                last_error = str(e)
                if _is_filter_blocked(e):
                    self._mark_down(_FILTER_BLOCK_COOLDOWN)
                    marked = True
                    break
                if "429" in str(e) or "rate limit" in str(e).lower():
                    self._mark_down()
                    marked = True
                    break
        if not marked:
            self._mark_down()
        logger.warning("Groq translation failed (%s): %s", tgt_lang, _brief_error(last_error))
        return None

    @staticmethod
    def _parse_json_obj(raw: str):
        """Parse model output as JSON, tolerating code fences and stray text."""
        if not raw:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, TypeError):
            return None

    def translate_fields(self, fields: Dict[str, str], tgt_lang: str) -> Dict[str, str]:
        """Batch translate multiple fields in a single Groq call.

        Returns dict of translated fields; failed fields are omitted (caller keeps original).
        """
        if not self.enabled or self.down or tgt_lang in (None, "", "en"):
            return {}
        if not fields:
            return {}

        source_json = json.dumps(fields, ensure_ascii=False)
        tgt_name = _LANG_NAMES.get(tgt_lang, tgt_lang)
        prompt = f"""You are a professional legal translator. Translate the following JSON object from English to {tgt_name}.

Rules:
- Output ONLY a valid JSON object with the same keys.
- Keep numbers, section references, bullet markers (like "-" or "1.") and any formatting intact.
- Use natural, plain {tgt_name} that a layperson can understand.

English JSON:
{source_json}"""

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
                stream=False,
                timeout=self.timeout,
            )
            content = str(resp.choices[0].message.content or "").strip()
            if not content:
                return {}
            data = self._parse_json_obj(content)
            if not isinstance(data, dict):
                return {}
            return {k: str(v) for k, v in data.items() if k in fields}
        except Exception as e:
            blocked = _is_filter_blocked(e)
            logger.warning("Groq batch translate failed (%s): %s", tgt_lang, _brief_error(e))
            if blocked:
                self._mark_down(_FILTER_BLOCK_COOLDOWN)
            elif "429" in str(e) or "rate limit" in str(e).lower():
                self._mark_down()
            return {}


_groq_instance = None
_groq_lock = None


def get_groq_translator() -> GroqClient:
    global _groq_instance, _groq_lock
    if _groq_instance is None:
        if _groq_lock is None:
            import threading
            _groq_lock = threading.Lock()
        with _groq_lock:
            if _groq_instance is None:
                _groq_instance = GroqClient()
    return _groq_instance