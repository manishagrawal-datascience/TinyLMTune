"""
Text cleaning and preprocessing utilities.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalise and clean raw text for TinyBERT tokenisation."""
    if not isinstance(text, str):
        return ""
    
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def truncate_text(text: str, max_words: int = 256) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
