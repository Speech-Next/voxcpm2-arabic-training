"""Arabic text clean-up shared by the data pipeline and the evaluation.

This is orthographic unification only (letter variants, tatweel, spacing); reading numbers and symbols aloud is the
inference library's text normalization (issue #4).
"""

from __future__ import annotations

import re

# Persian / Urdu code points that dialect transcripts use for Arabic letters
_LETTERS = str.maketrans({
    "ک": "ك",  # keheh -> kaf
    "ی": "ي",  # farsi yeh -> yeh
    "ھ": "ه",  # heh doachashmee -> heh
    "ہ": "ه",  # heh goal -> heh
    "ە": "ه",  # ae -> heh
    "ـ": None,      # tatweel
})
_TAG = re.compile(r"^\s*\[[A-Za-z]{2,3}\]\s*")
# non-speech events the transcripts mark inline ("... صومعته sigh", "[exhale]")
_EVENT = re.compile(r"\[?\b(?:sigh|laugh(?:s|ter)?|exhale|inhale|breath|cough|noise|music|applause|silence)\b\]?",
                    re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def strip_dialect_tag(text: str) -> str:
    """'[EG] إزيك' -> 'إزيك' (the lahgtna corpus prefixes every text with its dialect)."""
    return _TAG.sub("", text)


def clean(text: str) -> str:
    """Unify letter variants, drop tatweel, the dialect tag and non-speech event marks, collapse whitespace."""
    return _SPACE.sub(" ", _EVENT.sub(" ", strip_dialect_tag(text).translate(_LETTERS))).strip()
