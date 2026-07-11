"""Label-free lexical baselines for question-local candidate ranking.

This module deliberately treats questions and path serializations as opaque
strings.  It tokenizes them in memory and retains only aggregate token counts;
it never writes source text or serialized candidates to disk.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass


_ENGLISH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", re.ASCII)


def tokenize_english(text: str) -> tuple[str, ...]:
    """Tokenize an English string after deterministic NFKC case-folding.

    Hyphens, underscores, arrows, and other punctuation are separators.  The
    path serialization still retains those markers for order-aware models;
    BM25 is intentionally a bag-of-words semantic baseline.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_ENGLISH_TOKEN_PATTERN.findall(normalized))


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Frozen Okapi BM25 hyperparameters."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be finite and between zero and one")


class PerQuestionBM25:
    """A BM25 index fitted once over one question's candidate universe.

    Call :meth:`fit` with every candidate that can be ranked for a question,
    then call :meth:`score` once with that question.  Fitting separately for
    each positive/negative pair would change document-frequency statistics and
    is not a valid use of this baseline.
    """

    __slots__ = (
        "_average_document_length",
        "_candidate_ids",
        "_config",
        "_document_frequencies",
        "_document_lengths",
        "_term_frequencies",
    )

    def __init__(
        self,
        *,
        candidate_ids: tuple[str, ...],
        term_frequencies: tuple[Counter[str], ...],
        document_lengths: tuple[int, ...],
        document_frequencies: Mapping[str, int],
        average_document_length: float,
        config: BM25Config,
    ) -> None:
        self._candidate_ids = candidate_ids
        self._term_frequencies = term_frequencies
        self._document_lengths = document_lengths
        self._document_frequencies = dict(document_frequencies)
        self._average_document_length = average_document_length
        self._config = config

    @classmethod
    def fit(
        cls,
        serialized_candidates: Mapping[str, str],
        *,
        config: BM25Config | None = None,
    ) -> "PerQuestionBM25":
        """Fit one question-local index without retaining candidate text."""

        if not isinstance(serialized_candidates, Mapping):
            raise TypeError("serialized_candidates must be a mapping")
        if not serialized_candidates:
            raise ValueError("serialized_candidates must not be empty")
        resolved_config = config if config is not None else BM25Config()
        if not isinstance(resolved_config, BM25Config):
            raise TypeError("config must be a BM25Config")

        for candidate_id, text in serialized_candidates.items():
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("candidate IDs must be non-empty strings")
            if not isinstance(text, str):
                raise TypeError("serialized candidate values must be strings")

        # Sorting makes scores independent of mapping insertion order.  Only
        # counters and lengths survive this method; raw serializations do not.
        candidate_ids = tuple(sorted(serialized_candidates))
        term_frequencies = tuple(
            Counter(tokenize_english(serialized_candidates[candidate_id]))
            for candidate_id in candidate_ids
        )
        document_lengths = tuple(sum(counts.values()) for counts in term_frequencies)
        document_frequencies: Counter[str] = Counter()
        for counts in term_frequencies:
            document_frequencies.update(counts.keys())

        mean_length = sum(document_lengths) / len(document_lengths)
        # Empty serializations are valid opaque inputs.  A unit fallback keeps
        # length normalization finite; every term frequency remains zero.
        average_document_length = mean_length if mean_length > 0 else 1.0
        return cls(
            candidate_ids=candidate_ids,
            term_frequencies=term_frequencies,
            document_lengths=document_lengths,
            document_frequencies=document_frequencies,
            average_document_length=average_document_length,
            config=resolved_config,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self._candidate_ids

    @property
    def candidate_count(self) -> int:
        return len(self._candidate_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._document_frequencies)

    @property
    def config(self) -> BM25Config:
        return self._config

    def score(self, question: str) -> dict[str, float]:
        """Score every fitted candidate against ``question``.

        The IDF is the positive Lucene variant
        ``log(1 + (N - df + 0.5) / (df + 0.5))``.  This avoids undocumented
        negative-IDF flooring on the small, highly overlapping per-question
        candidate collections used by the kill test.
        """

        query_counts = Counter(tokenize_english(question))
        scores = [0.0] * self.candidate_count
        candidate_count = self.candidate_count
        k1 = self._config.k1
        b = self._config.b

        for term, query_frequency in query_counts.items():
            document_frequency = self._document_frequencies.get(term)
            if document_frequency is None:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (candidate_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, term_counts in enumerate(self._term_frequencies):
                term_frequency = term_counts.get(term, 0)
                if term_frequency == 0:
                    continue
                length_ratio = (
                    self._document_lengths[index] / self._average_document_length
                )
                denominator = term_frequency + k1 * (1.0 - b + b * length_ratio)
                scores[index] += (
                    query_frequency
                    * inverse_document_frequency
                    * term_frequency
                    * (k1 + 1.0)
                    / denominator
                )

        return dict(zip(self._candidate_ids, scores, strict=True))
