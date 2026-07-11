"""Deterministic manifests for real retrieval-error path pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Literal, Mapping, Sequence

from .graph import CandidatePath, QueryDocumentGraph, build_query_document_graph
from .serialization import (
    SERIALIZER_VERSION,
    count_serialization_tokens,
    serialize_evidence_topology,
)
from .topology import EvidenceDocument, EvidenceTopology, TypedEdge


NATURAL_PAIR_SCHEMA_VERSION = 2
NATURAL_STRATUM = "natural_retrieval_error"
TOKEN_MATCH_THRESHOLD = 0.05
NATURAL_MATCHING_POLICY = "exact-serialized-token-distance-v1"

ZeroReason = Literal[
    "no_positive_candidate",
    "no_negative_candidate",
    "unusable_sentence_payload",
]


class NaturalPairInvariantError(ValueError):
    """Raised when a real-path pair violates its content-free schema."""


def _stable_hexdigest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _real_path_edges() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "relation": "retrieval",
            "source": "q",
            "target": "d0",
            "observed": True,
        },
        {
            "relation": "title_mention",
            "source": "d0",
            "target": "d1",
            "observed": True,
        },
    )


@dataclass(frozen=True, slots=True)
class RealPathPayload:
    """Content-free metadata for one observed q->source->target path."""

    context_indices: tuple[int, int]
    sentence_indices: tuple[tuple[int, ...], tuple[int, ...]]
    retrieval_score: float
    retrieval_rank: int
    serialized_token_count: int
    serialization_sha256: str
    exceeds_max_length: bool

    def __post_init__(self) -> None:
        source_index, target_index = self.context_indices
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or not isinstance(target_index, int)
            or isinstance(target_index, bool)
            or source_index < 0
            or target_index < 0
            or source_index == target_index
        ):
            raise NaturalPairInvariantError(
                "path context indices must be distinct non-negative integers"
            )
        if len(self.sentence_indices) != 2:
            raise NaturalPairInvariantError("path must select source and target sentences")
        source_sentences, target_sentences = self.sentence_indices
        for indices in (source_sentences, target_sentences):
            if not indices or tuple(sorted(set(indices))) != indices:
                raise NaturalPairInvariantError(
                    "sentence indices must be non-empty, sorted, and unique"
                )
            if any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in indices
            ):
                raise NaturalPairInvariantError(
                    "sentence indices must be non-negative integers"
                )
        if source_sentences[0] != 0 or target_sentences != (0,):
            raise NaturalPairInvariantError(
                "path sentence policy requires source first/evidence and target first"
            )
        if (
            not isinstance(self.retrieval_score, (int, float))
            or isinstance(self.retrieval_score, bool)
            or not math.isfinite(float(self.retrieval_score))
        ):
            raise NaturalPairInvariantError("retrieval score must be finite")
        if (
            not isinstance(self.retrieval_rank, int)
            or isinstance(self.retrieval_rank, bool)
            or self.retrieval_rank < 1
        ):
            raise NaturalPairInvariantError("retrieval rank must be a positive integer")
        if (
            not isinstance(self.serialized_token_count, int)
            or isinstance(self.serialized_token_count, bool)
            or self.serialized_token_count < 1
        ):
            raise NaturalPairInvariantError(
                "serialized token count must be a positive integer"
            )
        if len(self.serialization_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.serialization_sha256
        ):
            raise NaturalPairInvariantError(
                "serialization SHA256 must be lowercase hexadecimal"
            )
        if not isinstance(self.exceeds_max_length, bool):
            raise NaturalPairInvariantError("exceeds_max_length must be bool")

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "context_indices": list(self.context_indices),
            "sentence_indices": {
                "d0": list(self.sentence_indices[0]),
                "d1": list(self.sentence_indices[1]),
            },
            "typed_edges": list(_real_path_edges()),
            "retrieval_score": float(self.retrieval_score),
            "retrieval_rank": self.retrieval_rank,
            "serialized_token_count": self.serialized_token_count,
            "serialization_sha256": self.serialization_sha256,
            "exceeds_max_length": self.exceeds_max_length,
        }


@dataclass(frozen=True, slots=True)
class NaturalRetrievalPair:
    pair_id: str
    qid: str
    positive: RealPathPayload
    negative: RealPathPayload
    candidate_score_std: float
    absolute_score_gap: float
    normalized_score_gap: float
    absolute_token_gap: int
    relative_token_gap: float
    selection_distance: float

    def __post_init__(self) -> None:
        if not self.pair_id.startswith("natural-pair-"):
            raise NaturalPairInvariantError("pair ID must use the natural-pair prefix")
        if not isinstance(self.qid, str) or not self.qid:
            raise NaturalPairInvariantError("qid must be a non-empty string")
        for value_name, value in (
            ("candidate score std", self.candidate_score_std),
            ("absolute score gap", self.absolute_score_gap),
            ("normalized score gap", self.normalized_score_gap),
            ("relative token gap", self.relative_token_gap),
            ("selection distance", self.selection_distance),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise NaturalPairInvariantError(f"{value_name} must be finite and non-negative")
        if self.absolute_token_gap < 0:
            raise NaturalPairInvariantError("absolute token gap must be non-negative")

        expected_score_gap = abs(
            self.positive.retrieval_score - self.negative.retrieval_score
        )
        expected_token_gap = abs(
            self.positive.serialized_token_count
            - self.negative.serialized_token_count
        )
        score_scale = max(self.candidate_score_std, 1e-6)
        expected_normalized_score_gap = expected_score_gap / score_scale
        expected_relative_token_gap = expected_token_gap / max(
            self.positive.serialized_token_count, 1
        )
        expected_distance = (
            expected_normalized_score_gap + expected_relative_token_gap
        )
        if not math.isclose(self.absolute_score_gap, expected_score_gap):
            raise NaturalPairInvariantError("stored absolute score gap is inconsistent")
        if self.absolute_token_gap != expected_token_gap:
            raise NaturalPairInvariantError("stored absolute token gap is inconsistent")
        if not math.isclose(
            self.normalized_score_gap, expected_normalized_score_gap
        ):
            raise NaturalPairInvariantError("stored normalized score gap is inconsistent")
        if not math.isclose(self.relative_token_gap, expected_relative_token_gap):
            raise NaturalPairInvariantError("stored relative token gap is inconsistent")
        if not math.isclose(self.selection_distance, expected_distance):
            raise NaturalPairInvariantError("stored selection distance is inconsistent")

    @property
    def token_within_five_percent(self) -> bool:
        return self.relative_token_gap <= TOKEN_MATCH_THRESHOLD

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "qid": self.qid,
            "stratum": NATURAL_STRATUM,
            "positive": self.positive.to_manifest_record(),
            "negative": self.negative.to_manifest_record(),
            "matching": {
                "candidate_score_std": self.candidate_score_std,
                "absolute_score_gap": self.absolute_score_gap,
                "normalized_score_gap": self.normalized_score_gap,
                "absolute_token_gap": self.absolute_token_gap,
                "relative_token_gap": self.relative_token_gap,
                "selection_distance": self.selection_distance,
                "token_within_five_percent": self.token_within_five_percent,
            },
        }


@dataclass(frozen=True, slots=True)
class NaturalSelection:
    pair: NaturalRetrievalPair | None
    zero_reason: ZeroReason | None
    positive_candidate_count: int
    negative_candidate_count: int
    candidate_combination_count: int

    def __post_init__(self) -> None:
        if (self.pair is None) == (self.zero_reason is None):
            raise NaturalPairInvariantError(
                "selection must contain exactly one of pair or zero reason"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (
                self.positive_candidate_count,
                self.negative_candidate_count,
                self.candidate_combination_count,
            )
        ):
            raise NaturalPairInvariantError("candidate counts must be non-negative")


def _path_payload(
    path: CandidatePath,
    graph: QueryDocumentGraph,
    tokenizer: Any,
    *,
    max_length: int,
) -> RealPathPayload | None:
    source = graph.documents[path.source_index]
    target = graph.documents[path.target_index]
    if not source.sentences or not target.sentences:
        return None
    source_sentence_indices = tuple(
        sorted({0, *path.mention_sentence_indices})
    )
    if not path.mention_sentence_indices:
        raise NaturalPairInvariantError("candidate path lacks mention evidence")
    documents = (
        EvidenceDocument(
            alias="d0",
            title=source.title,
            sentences=tuple(
                source.sentences[index] for index in source_sentence_indices
            ),
        ),
        EvidenceDocument(
            alias="d1",
            title=target.title,
            sentences=(target.sentences[0],),
        ),
    )
    topology = EvidenceTopology(
        question_id=graph.question_id,
        question=graph.question,
        documents=documents,
        edges=(
            TypedEdge("retrieval", "q", "d0", True),
            TypedEdge("title_mention", "d0", "d1", True),
        ),
    )
    serialization = serialize_evidence_topology(topology)
    token_stats = count_serialization_tokens(
        serialization, tokenizer, max_length=max_length
    )
    return RealPathPayload(
        context_indices=(path.source_index, path.target_index),
        sentence_indices=(source_sentence_indices, (0,)),
        retrieval_score=float(path.retrieval_score),
        retrieval_rank=path.retrieval_rank,
        serialized_token_count=token_stats.untruncated_token_count,
        serialization_sha256=serialization.sha256,
        exceeds_max_length=token_stats.truncated,
    )


def select_natural_retrieval_pair(
    example: Mapping[str, Any],
    retrieval_scores: Sequence[float],
    tokenizer: Any,
    *,
    tokenizer_fingerprint_sha256: str,
    retrieval_top_k: int = 10,
    max_length: int = 1024,
) -> NaturalSelection:
    """Select at most one positive/real-error pair for a question.

    The preregistered distance is normalized retrieval-score gap plus relative
    exact serialized-token gap. Population standard deviation is computed over the
    retrieval scores of all real candidate paths for the question. Ties break
    by absolute token gap, absolute score gap, then context indices.
    """

    if not isinstance(retrieval_top_k, int) or retrieval_top_k < 1:
        raise NaturalPairInvariantError("retrieval_top_k must be a positive integer")
    if not callable(tokenizer):
        raise NaturalPairInvariantError("tokenizer must be callable")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise NaturalPairInvariantError("max_length must be a positive integer")
    if len(tokenizer_fingerprint_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in tokenizer_fingerprint_sha256
    ):
        raise NaturalPairInvariantError(
            "tokenizer fingerprint must be lowercase SHA256 hexadecimal"
        )

    graph = build_query_document_graph(
        example,
        retrieval_scores,
        retrieval_top_k=min(retrieval_top_k, len(example["context"])),
    )
    positives = tuple(graph.gold_paths)
    negatives = tuple(graph.natural_negative_paths)
    if not positives:
        return NaturalSelection(None, "no_positive_candidate", 0, len(negatives), 0)
    if not negatives:
        return NaturalSelection(None, "no_negative_candidate", len(positives), 0, 0)

    candidate_scores = [path.retrieval_score for path in graph.candidate_paths]
    candidate_score_std = (
        statistics.pstdev(candidate_scores) if len(candidate_scores) > 1 else 0.0
    )
    score_scale = max(candidate_score_std, 1e-6)
    payload_by_path: dict[tuple[int, int], RealPathPayload | None] = {}

    def payload(path: CandidatePath) -> RealPathPayload | None:
        key = (path.source_index, path.target_index)
        if key not in payload_by_path:
            payload_by_path[key] = _path_payload(
                path, graph, tokenizer, max_length=max_length
            )
        return payload_by_path[key]

    combinations: list[
        tuple[
            tuple[float, int, float, int, int, int, int],
            RealPathPayload,
            RealPathPayload,
        ]
    ] = []
    for positive_path in positives:
        positive = payload(positive_path)
        if positive is None:
            continue
        for negative_path in negatives:
            negative = payload(negative_path)
            if negative is None:
                continue
            absolute_score_gap = abs(
                positive.retrieval_score - negative.retrieval_score
            )
            absolute_token_gap = abs(
                positive.serialized_token_count
                - negative.serialized_token_count
            )
            relative_token_gap = absolute_token_gap / max(
                positive.serialized_token_count, 1
            )
            distance = absolute_score_gap / score_scale + relative_token_gap
            key = (
                distance,
                absolute_token_gap,
                absolute_score_gap,
                positive.context_indices[0],
                positive.context_indices[1],
                negative.context_indices[0],
                negative.context_indices[1],
            )
            combinations.append((key, positive, negative))

    if not combinations:
        return NaturalSelection(
            None,
            "unusable_sentence_payload",
            len(positives),
            len(negatives),
            0,
        )

    key, positive, negative = min(combinations, key=lambda item: item[0])
    distance, absolute_token_gap, absolute_score_gap, *_indices = key
    normalized_score_gap = absolute_score_gap / score_scale
    relative_token_gap = absolute_token_gap / max(
        positive.serialized_token_count, 1
    )
    qid = str(example["_id"])
    pair_id = "natural-pair-" + _stable_hexdigest(
        {
            "qid": qid,
            "positive_context_indices": positive.context_indices,
            "negative_context_indices": negative.context_indices,
            "sentence_policy": "source_first_plus_mention_evidence_target_first",
            "matching_policy": NATURAL_MATCHING_POLICY,
            "serializer_version": SERIALIZER_VERSION,
            "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        }
    )[:24]
    pair = NaturalRetrievalPair(
        pair_id=pair_id,
        qid=qid,
        positive=positive,
        negative=negative,
        candidate_score_std=float(candidate_score_std),
        absolute_score_gap=float(absolute_score_gap),
        normalized_score_gap=float(normalized_score_gap),
        absolute_token_gap=int(absolute_token_gap),
        relative_token_gap=float(relative_token_gap),
        selection_distance=float(distance),
    )
    return NaturalSelection(
        pair,
        None,
        len(positives),
        len(negatives),
        len(combinations),
    )


def summarize_natural_selections(
    selections: Sequence[NaturalSelection],
) -> dict[str, Any]:
    pairs = [selection.pair for selection in selections if selection.pair is not None]
    zero_reasons = Counter(
        selection.zero_reason
        for selection in selections
        if selection.zero_reason is not None
    )
    return {
        "frozen_question_count": len(selections),
        "questions_with_positive_candidate": sum(
            selection.positive_candidate_count > 0 for selection in selections
        ),
        "questions_with_negative_candidate": sum(
            selection.negative_candidate_count > 0 for selection in selections
        ),
        "questions_with_positive_and_negative_candidates": sum(
            selection.positive_candidate_count > 0
            and selection.negative_candidate_count > 0
            for selection in selections
        ),
        "eligible_pair_count": len(pairs),
        "zero_pair_reason_counts": {
            reason: zero_reasons.get(reason, 0)
            for reason in (
                "no_positive_candidate",
                "no_negative_candidate",
                "unusable_sentence_payload",
            )
        },
        "selected_pairs_with_exact_token_gap_lte_5_percent": sum(
            pair.token_within_five_percent for pair in pairs
        ),
        "selected_positive_paths_exceeding_max_length": sum(
            pair.positive.exceeds_max_length for pair in pairs
        ),
        "selected_negative_paths_exceeding_max_length": sum(
            pair.negative.exceeds_max_length for pair in pairs
        ),
        "selected_pairs_with_either_path_exceeding_max_length": sum(
            pair.positive.exceeds_max_length or pair.negative.exceeds_max_length
            for pair in pairs
        ),
    }


def build_natural_pair_manifest(
    selections: Sequence[NaturalSelection],
    *,
    official_split: str,
    source_sha256: str,
    ids_sha256: str,
    retrieval_cache_sha256: str,
    tokenizer_fingerprint_sha256: str,
    retrieval_top_k: int,
    max_length: int,
) -> dict[str, Any]:
    if (
        not isinstance(retrieval_top_k, int)
        or isinstance(retrieval_top_k, bool)
        or retrieval_top_k < 1
        or not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length < 1
    ):
        raise NaturalPairInvariantError(
            "retrieval_top_k and max_length must be positive integers"
        )
    pairs = sorted(
        (selection.pair for selection in selections if selection.pair is not None),
        key=lambda pair: (pair.qid, pair.pair_id),
    )
    if len({pair.qid for pair in pairs}) != len(pairs):
        raise NaturalPairInvariantError("manifest may contain at most one pair per qid")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise NaturalPairInvariantError("manifest contains duplicate pair IDs")
    for name, digest in (
        ("source", source_sha256),
        ("IDs", ids_sha256),
        ("retrieval cache", retrieval_cache_sha256),
        ("tokenizer files", tokenizer_fingerprint_sha256),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise NaturalPairInvariantError(
                f"{name} SHA256 must be lowercase hexadecimal"
            )

    return {
        "schema_version": NATURAL_PAIR_SCHEMA_VERSION,
        "dataset": "HotpotQA",
        "official_split": official_split,
        "stratum": NATURAL_STRATUM,
        "retrieval_top_k": retrieval_top_k,
        "max_length": max_length,
        "schema": {
            "path_shape": "q->d0 retrieval, d0->d1 title_mention",
            "all_edges_observed": True,
            "manual_edge_edits": False,
            "serializer_version": SERIALIZER_VERSION,
            "matching_policy": NATURAL_MATCHING_POLICY,
            "distance": (
                "abs(retrieval_score_gap)/max(population_std_of_question_candidate_"
                "path_scores,1e-6) + abs(untruncated_serialized_token_count_gap)/"
                "max(positive_untruncated_serialized_token_count,1)"
            ),
            "tie_break": (
                "absolute token gap, absolute score gap, positive source/target "
                "indices, negative source/target indices"
            ),
            "sentence_selection": (
                "independently per path and without supporting-fact sentence labels: "
                "source first sentence union real mention-evidence sentences; target "
                "first sentence; sorted and deduplicated"
            ),
            "token_counting": (
                "Hugging Face tokenizer on the final EvidenceTopology serialization "
                "with special tokens, no padding, and truncation disabled; max_length "
                "is diagnostic only and never changes matching"
            ),
            "truncation_applied": False,
        },
        "provenance": {
            "source_sha256": source_sha256,
            "ids_sha256": ids_sha256,
            "retrieval_cache_sha256": retrieval_cache_sha256,
            "tokenizer_files_sha256": tokenizer_fingerprint_sha256,
            "serializer_version": SERIALIZER_VERSION,
        },
        "counts": summarize_natural_selections(selections),
        "pairs": [pair.to_manifest_record() for pair in pairs],
    }
