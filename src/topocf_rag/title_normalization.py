"""Conservative title normalization and boundary-safe mention matching."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Return a deterministic comparison form without discarding punctuation.

    HotpotQA titles occasionally use underscores as spaces.  Beyond that
    equivalence, normalization is intentionally conservative: parentheses,
    hyphens, apostrophes, and other punctuation remain significant.
    """

    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title).__name__}")
    normalized = unicodedata.normalize("NFKC", title).replace("_", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip().casefold()
    if not normalized:
        raise ValueError("title must not be empty after normalization")
    return normalized


def normalize_mention_text(text: str) -> str:
    """Normalize prose with the same transformations used for titles."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFKC", text).replace("_", " ")
    return _WHITESPACE_RE.sub(" ", normalized).strip().casefold()


@lru_cache(maxsize=16_384)
def _mention_pattern(normalized_title: str) -> re.Pattern[str]:
    # \w is Unicode-aware in Python.  Requiring non-word boundaries prevents
    # e.g. title "York" from matching "Yorkshire" while allowing possessives.
    return re.compile(rf"(?<!\w){re.escape(normalized_title)}(?!\w)")


def contains_title_mention(text: str, title: str) -> bool:
    """Return whether *text* contains an exact normalized title mention."""

    normalized_title = normalize_title(title)
    normalized_text = normalize_mention_text(text)
    return _mention_pattern(normalized_title).search(normalized_text) is not None


def mentioning_sentence_indices(sentences: Iterable[str], title: str) -> tuple[int, ...]:
    """Return sentence indices containing a boundary-safe title mention."""

    indices: list[int] = []
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, str):
            raise TypeError(
                f"sentence at index {index} must be str, got {type(sentence).__name__}"
            )
        if contains_title_mention(sentence, title):
            indices.append(index)
    return tuple(indices)
