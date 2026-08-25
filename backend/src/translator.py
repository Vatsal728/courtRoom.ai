"""translator.py - Language detection and script helpers (Phase 10+)

Provides lightweight script-based language detection for Indian languages.
Translation is handled by groq_translator.py (primary) and ollama_translator.py (fallback).
"""
import logging
import re

logger = logging.getLogger("courtroom-translator")

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


# Language names (English + native scripts) used to honor explicit requests
# like "answer in gujarati" even when the query itself is typed in English.
LANGUAGE_NAME_MAP = {
    "en": ("english", "અંગ્રેજી", "अंग्रेज़ी"),
    "hi": ("hindi", "हिन्दी", "हिंदी"),
    "gu": ("gujarati", "ગુજરાતી"),
    "mr": ("marathi", "मराठी"),
    "ta": ("tamil", "தமிழ்"),
    "te": ("telugu", "తెలుగు"),
    "kn": ("kannada", "ಕನ್ನಡ"),
    "bn": ("bengali", "bangla", "বাংলা"),
    "ml": ("malayalam", "മലയാളം"),
    "pa": ("punjabi", "ਪੰਜਾਬੀ"),
    "ur": ("urdu", "اردو"),
}


def detect_requested_language(text: str) -> str:
    """Detect an explicit language request like 'help me in gujarati'.

    Returns a supported language code, or 'en' when none is mentioned.
    """
    if not text:
        return "en"
    for lang, aliases in LANGUAGE_NAME_MAP.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                return lang
    return "en"