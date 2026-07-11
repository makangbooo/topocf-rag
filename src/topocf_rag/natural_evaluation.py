"""Reconstruction and reporting for real natural retrieval-error paths."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import (
    EvaluationInvariantError,
    PreparedCandidate,
    PreparedPair,
    PreparedSplit,
    ScoreBundle,
    canonical_signature_text,
    edge_reality,
    evaluate_prepared_split,
    sha256_file,
    stable_hash,
    stream_selected_source_examples,
)
from .graph import validate_hotpot_example
from .serialization import (
    SERIALIZER_VERSION,
    SerializationTokenStats,
    count_serialization_tokens,
    serialize_evidence_topology,
)
from .title_normalization import contains_title_mention, normalize_title
from .topology import EvidenceDocument, EvidenceTopology, TypedEdge


NATURAL_STRATUM = "natural_retrieval_error"
NATURAL_VARIANT = "natural"
TOKEN_MATCH_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class NaturalMatching:
    candidate_score_std: float
    absolute_score_gap: float
    normalized_score_gap: float
    absolute_token_gap: int
    relative_token_gap: float
    selection_distance: float
    token_within_five_percent: bool


@dataclass(frozen=True, slots=True)
class PreparedNaturalSplit:
    prepared: PreparedSplit
    matching_by_pair_id: Mapping[str, NaturalMatching]
    selection_counts: Mapping[str, Any]
    tokenizer_files_sha256: str
    serializer_version: str
    manifest_max_length: int


def load_natural_manifest(path: str | Path) -> dict[str, Any]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise EvaluationInvariantError("unsupported natural manifest schema")
    if payload.get("stratum") != NATURAL_STRATUM:
        raise EvaluationInvariantError("natural manifest has an unexpected stratum")
    if not isinstance(payload.get("pairs"), list):
        raise EvaluationInvariantError("natural manifest must contain a pairs list")
    if not isinstance(payload.get("counts"), Mapping):
        raise EvaluationInvariantError("natural manifest must contain coverage counts")
    return payload


def natural_manifest_qids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    qids: set[str] = set()
    pair_ids: set[str] = set()
    for record in manifest.get("pairs", []):
        if not isinstance(record, Mapping):
            raise EvaluationInvariantError("natural pair records must be objects")
        qid = record.get("qid")
        pair_id = record.get("pair_id")
        if not isinstance(qid, str) or not qid:
            raise EvaluationInvariantError("natural pair qid must be non-empty")
        if not isinstance(pair_id, str) or not pair_id:
            raise EvaluationInvariantError("natural pair ID must be non-empty")
        if qid in qids:
            raise EvaluationInvariantError("natural manifest may contain one pair per qid")
        if pair_id in pair_ids:
            raise EvaluationInvariantError("natural manifest contains duplicate pair IDs")
        qids.add(qid)
        pair_ids.add(pair_id)
    return tuple(sorted(qids))


def _selected_documents(
    example: Mapping[str, Any], member: Mapping[str, Any]
) -> tuple[tuple[EvidenceDocument, ...], tuple[int, int]]:
    context_indices = member.get("context_indices")
    selections = member.get("sentence_indices")
    if (
        not isinstance(context_indices, list)
        or len(context_indices) != 2
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in context_indices
        )
        or context_indices[0] == context_indices[1]
    ):
        raise EvaluationInvariantError("natural path context indices are invalid")
    if not isinstance(selections, Mapping) or tuple(selections) != ("d0", "d1"):
        raise EvaluationInvariantError("natural path sentence selections must be d0,d1")

    documents: list[EvidenceDocument] = []
    for alias, context_index in zip(("d0", "d1"), context_indices, strict=True):
        if context_index >= len(example["context"]):
            raise EvaluationInvariantError("natural path context index is out of range")
        title, sentences = example["context"][context_index]
        indices = selections[alias]
        if (
            not isinstance(indices, list)
            or not indices
            or indices != sorted(set(indices))
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(sentences)
                for index in indices
            )
        ):
            raise EvaluationInvariantError("natural selected sentence indices are invalid")
        documents.append(
            EvidenceDocument(
                alias=alias,
                title=title,
                sentences=tuple(sentences[index] for index in indices),
            )
        )
    return tuple(documents), (context_indices[0], context_indices[1])


def _observed_edges(member: Mapping[str, Any]) -> tuple[TypedEdge, ...]:
    payloads = member.get("typed_edges")
    if not isinstance(payloads, list) or len(payloads) != 2:
        raise EvaluationInvariantError("natural path must contain exactly two edges")
    edges: list[TypedEdge] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise EvaluationInvariantError("natural typed edge must be an object")
        edges.append(
            TypedEdge(
                relation=payload.get("relation"),
                source=payload.get("source"),
                target=payload.get("target"),
                observed=payload.get("observed"),
            )
        )
    edges.sort(key=lambda edge: (edge.relation != "retrieval", edge.source, edge.target))
    expected = {
        ("retrieval", "q", "d0"),
        ("title_mention", "d0", "d1"),
    }
    if {edge.structural_key for edge in edges} != expected or not all(
        edge.observed for edge in edges
    ):
        raise EvaluationInvariantError(
            "natural path edges must be observed q->d0 retrieval and d0->d1 mention"
        )
    return tuple(edges)


def _prepare_member(
    *,
    qid: str,
    question: str,
    example: Mapping[str, Any],
    member: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    retrieval_top_k: int,
    manifest_max_length: int,
    token_cache: dict[str, SerializationTokenStats],
    fail_on_truncation: bool,
) -> tuple[PreparedCandidate, tuple[int, int], float, int, int]:
    documents, context_indices = _selected_documents(example, member)
    topology = EvidenceTopology(
        question_id=qid,
        question=question,
        documents=documents,
        edges=_observed_edges(member),
    )
    if not any(
        contains_title_mention(sentence, documents[1].title)
        for sentence in documents[0].sentences
    ):
        raise EvaluationInvariantError(
            "natural observed mention lacks selected source sentence evidence"
        )
    retrieval_score = member.get("retrieval_score")
    retrieval_rank = member.get("retrieval_rank")
    serialized_token_count = member.get("serialized_token_count")
    if (
        isinstance(retrieval_score, bool)
        or not isinstance(retrieval_score, (int, float))
        or not math.isfinite(float(retrieval_score))
    ):
        raise EvaluationInvariantError("natural retrieval score must be finite")
    if (
        not isinstance(retrieval_rank, int)
        or isinstance(retrieval_rank, bool)
        or not 1 <= retrieval_rank <= retrieval_top_k
    ):
        raise EvaluationInvariantError("natural retrieval rank is outside top-k")
    if (
        not isinstance(serialized_token_count, int)
        or isinstance(serialized_token_count, bool)
        or serialized_token_count < 1
    ):
        raise EvaluationInvariantError("natural serialized token count must be positive")

    serialization = serialize_evidence_topology(topology)
    token_stats = token_cache.get(serialization.sha256)
    if token_stats is None:
        token_stats = count_serialization_tokens(
            serialization, tokenizer, max_length=max_length
        )
        token_cache[serialization.sha256] = token_stats
    if member.get("serialization_sha256") != serialization.sha256:
        raise EvaluationInvariantError("natural serialization SHA256 is inconsistent")
    if serialized_token_count != token_stats.untruncated_token_count:
        raise EvaluationInvariantError("natural serialized token count is inconsistent")
    stored_exceeds_max_length = member.get("exceeds_max_length")
    if (
        not isinstance(stored_exceeds_max_length, bool)
        or stored_exceeds_max_length
        != (token_stats.untruncated_token_count > manifest_max_length)
    ):
        raise EvaluationInvariantError("natural max-length diagnostic is inconsistent")
    if fail_on_truncation and token_stats.truncated:
        raise EvaluationInvariantError(
            "natural serialized path exceeds max_length; truncation is disabled"
        )
    candidate = PreparedCandidate(
        topology=topology,
        serialization=serialization,
        token_stats=token_stats,
        canonical_signature=canonical_signature_text(topology),
        edge_reality=edge_reality(topology),
    )
    return (
        candidate,
        context_indices,
        float(retrieval_score),
        retrieval_rank,
        serialized_token_count,
    )


def _matching(
    record: Mapping[str, Any],
    positive_score: float,
    negative_score: float,
    positive_serialized_tokens: int,
    negative_serialized_tokens: int,
) -> NaturalMatching:
    payload = record.get("matching")
    if not isinstance(payload, Mapping):
        raise EvaluationInvariantError("natural pair lacks matching diagnostics")
    try:
        result = NaturalMatching(
            candidate_score_std=float(payload["candidate_score_std"]),
            absolute_score_gap=float(payload["absolute_score_gap"]),
            normalized_score_gap=float(payload["normalized_score_gap"]),
            absolute_token_gap=int(payload["absolute_token_gap"]),
            relative_token_gap=float(payload["relative_token_gap"]),
            selection_distance=float(payload["selection_distance"]),
            token_within_five_percent=payload["token_within_five_percent"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationInvariantError("natural matching diagnostics are invalid") from error
    numeric = (
        result.candidate_score_std,
        result.absolute_score_gap,
        result.normalized_score_gap,
        result.relative_token_gap,
        result.selection_distance,
    )
    if (
        any(not math.isfinite(value) or value < 0 for value in numeric)
        or result.absolute_token_gap < 0
        or not isinstance(result.token_within_five_percent, bool)
    ):
        raise EvaluationInvariantError("natural matching diagnostics must be non-negative")
    expected_score_gap = abs(positive_score - negative_score)
    expected_token_gap = abs(
        positive_serialized_tokens - negative_serialized_tokens
    )
    expected_relative_gap = expected_token_gap / max(positive_serialized_tokens, 1)
    expected_normalized_gap = expected_score_gap / max(result.candidate_score_std, 1e-6)
    if not (
        math.isclose(result.absolute_score_gap, expected_score_gap)
        and result.absolute_token_gap == expected_token_gap
        and math.isclose(result.relative_token_gap, expected_relative_gap)
        and math.isclose(result.normalized_score_gap, expected_normalized_gap)
        and math.isclose(
            result.selection_distance,
            expected_relative_gap + expected_normalized_gap,
        )
        and result.token_within_five_percent
        == (expected_relative_gap <= TOKEN_MATCH_THRESHOLD)
    ):
        raise EvaluationInvariantError("natural matching diagnostics are inconsistent")
    return result


def prepare_natural_manifest(
    manifest: Mapping[str, Any],
    examples: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = 1024,
    fail_on_truncation: bool = True,
    expected_tokenizer_files_sha256: str | None = None,
) -> PreparedNaturalSplit:
    qids = natural_manifest_qids(manifest)
    if set(examples) != set(qids):
        raise EvaluationInvariantError("natural source IDs do not match manifest qids")
    counts = manifest.get("counts")
    frozen_question_count = counts.get("frozen_question_count")
    if (
        not isinstance(frozen_question_count, int)
        or isinstance(frozen_question_count, bool)
        or frozen_question_count < len(qids)
        or counts.get("eligible_pair_count") != len(qids)
    ):
        raise EvaluationInvariantError("natural manifest coverage counts are inconsistent")
    retrieval_top_k = manifest.get("retrieval_top_k")
    if not isinstance(retrieval_top_k, int) or retrieval_top_k < 1:
        raise EvaluationInvariantError("natural retrieval top-k must be positive")
    manifest_max_length = manifest.get("max_length")
    if (
        not isinstance(manifest_max_length, int)
        or isinstance(manifest_max_length, bool)
        or manifest_max_length < 1
    ):
        raise EvaluationInvariantError("natural manifest max_length must be positive")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise EvaluationInvariantError("natural manifest provenance is missing")
    tokenizer_files_sha256 = provenance.get("tokenizer_files_sha256")
    serializer_version = provenance.get("serializer_version")
    if (
        not isinstance(tokenizer_files_sha256, str)
        or len(tokenizer_files_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in tokenizer_files_sha256
        )
    ):
        raise EvaluationInvariantError("natural tokenizer fingerprint is invalid")
    if serializer_version != SERIALIZER_VERSION:
        raise EvaluationInvariantError("natural serializer version is incompatible")
    if (
        expected_tokenizer_files_sha256 is not None
        and tokenizer_files_sha256 != expected_tokenizer_files_sha256
    ):
        raise EvaluationInvariantError("natural tokenizer fingerprint does not match")

    prepared_pairs: list[PreparedPair] = []
    questions: dict[str, str] = {}
    matching_by_pair_id: dict[str, NaturalMatching] = {}
    token_cache: dict[str, SerializationTokenStats] = {}
    for record in manifest["pairs"]:
        if record.get("stratum") != NATURAL_STRATUM:
            raise EvaluationInvariantError("natural pair has an unexpected stratum")
        qid = record["qid"]
        pair_id = record["pair_id"]
        example = examples[qid]
        validate_hotpot_example(example)
        if example.get("_id") != qid:
            raise EvaluationInvariantError("natural source qid mismatch")
        question = example["question"]
        questions[qid] = question

        positive_member = record.get("positive")
        negative_member = record.get("negative")
        if not isinstance(positive_member, Mapping) or not isinstance(
            negative_member, Mapping
        ):
            raise EvaluationInvariantError("natural pair members must be objects")
        positive, positive_indices, positive_score, _positive_rank, positive_tokens = (
            _prepare_member(
                qid=qid,
                question=question,
                example=example,
                member=positive_member,
                tokenizer=tokenizer,
                max_length=max_length,
                retrieval_top_k=retrieval_top_k,
                manifest_max_length=manifest_max_length,
                token_cache=token_cache,
                fail_on_truncation=fail_on_truncation,
            )
        )
        negative, negative_indices, negative_score, _negative_rank, negative_tokens = (
            _prepare_member(
                qid=qid,
                question=question,
                example=example,
                member=negative_member,
                tokenizer=tokenizer,
                max_length=max_length,
                retrieval_top_k=retrieval_top_k,
                manifest_max_length=manifest_max_length,
                token_cache=token_cache,
                fail_on_truncation=fail_on_truncation,
            )
        )

        normalized_context_titles = [
            normalize_title(document[0]) for document in example["context"]
        ]
        supporting_titles = {
            normalize_title(title) for title, _sentence_index in example["supporting_facts"]
        }
        gold_indices = {
            index
            for index, title in enumerate(normalized_context_titles)
            if title in supporting_titles
        }
        if len(supporting_titles) != 2 or len(gold_indices) != 2:
            raise EvaluationInvariantError("natural bridge question must have two gold titles")
        if set(positive_indices) != gold_indices:
            raise EvaluationInvariantError("natural positive path does not cover both golds")
        if gold_indices.issubset(set(negative_indices)):
            raise EvaluationInvariantError("natural negative path covers both golds")

        matching = _matching(
            record,
            positive_score,
            negative_score,
            positive_tokens,
            negative_tokens,
        )
        matching_by_pair_id[pair_id] = matching
        class_signatures = [
            positive.canonical_signature,
            negative.canonical_signature,
        ]
        prepared_pairs.append(
            PreparedPair(
                pair_id=pair_id,
                base_id="natural-base-" + stable_hash([qid, pair_id])[:16],
                qid=qid,
                stratum=NATURAL_STRATUM,
                variant=NATURAL_VARIANT,
                positive=positive,
                negative=negative,
                canonical_class="canonical-" + stable_hash(class_signatures)[:16],
            )
        )

    prepared = PreparedSplit(
        official_split=str(manifest["official_split"]),
        frozen_question_count=frozen_question_count,
        questions=questions,
        pairs=tuple(prepared_pairs),
    )
    return PreparedNaturalSplit(
        prepared=prepared,
        matching_by_pair_id=matching_by_pair_id,
        selection_counts=dict(counts),
        tokenizer_files_sha256=tokenizer_files_sha256,
        serializer_version=serializer_version,
        manifest_max_length=manifest_max_length,
    )


def prepare_natural_split_from_files(
    manifest_path: str | Path,
    source_path: str | Path,
    tokenizer: Any,
    *,
    max_length: int = 1024,
    fail_on_truncation: bool = True,
    expected_tokenizer_files_sha256: str | None = None,
) -> tuple[PreparedNaturalSplit, dict[str, str]]:
    manifest = load_natural_manifest(manifest_path)
    source_digest = sha256_file(source_path)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("source_sha256") != source_digest:
        raise EvaluationInvariantError("natural source SHA256 does not match manifest")
    examples = stream_selected_source_examples(
        source_path, natural_manifest_qids(manifest)
    )
    return (
        prepare_natural_manifest(
            manifest,
            examples,
            tokenizer,
            max_length=max_length,
            fail_on_truncation=fail_on_truncation,
            expected_tokenizer_files_sha256=expected_tokenizer_files_sha256,
        ),
        {
            "manifest_sha256": sha256_file(manifest_path),
            "source_sha256": source_digest,
        },
    )


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _describe(values: Sequence[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": float(statistics.median(ordered)),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def evaluate_natural_split(
    natural: PreparedNaturalSplit, scores: ScoreBundle
) -> dict[str, Any]:
    report = evaluate_prepared_split(natural.prepared, scores)
    matching = [
        natural.matching_by_pair_id[pair.pair_id]
        for pair in natural.prepared.pairs
    ]
    exact_absolute_gaps: list[int] = []
    exact_relative_gaps: list[float] = []
    for pair in natural.prepared.pairs:
        positive_tokens = pair.positive.token_stats.untruncated_token_count
        negative_tokens = pair.negative.token_stats.untruncated_token_count
        absolute_gap = abs(positive_tokens - negative_tokens)
        exact_absolute_gaps.append(absolute_gap)
        exact_relative_gaps.append(absolute_gap / max(positive_tokens, 1))

    report["selection_coverage"] = dict(natural.selection_counts)
    report["natural_manifest_contract"] = {
        "tokenizer_files_sha256": natural.tokenizer_files_sha256,
        "serializer_version": natural.serializer_version,
        "manifest_max_length": natural.manifest_max_length,
    }
    report["matching_diagnostics"] = {
        "manifest_selection_matching": {
            "candidate_score_std": _describe(
                [item.candidate_score_std for item in matching]
            ),
            "absolute_retrieval_score_gap": _describe(
                [item.absolute_score_gap for item in matching]
            ),
            "normalized_retrieval_score_gap": _describe(
                [item.normalized_score_gap for item in matching]
            ),
            "absolute_serialized_token_gap": _describe(
                [item.absolute_token_gap for item in matching]
            ),
            "relative_serialized_token_gap": _describe(
                [item.relative_token_gap for item in matching]
            ),
            "selection_distance": _describe(
                [item.selection_distance for item in matching]
            ),
            "within_five_percent_count": sum(
                item.token_within_five_percent for item in matching
            ),
            "within_five_percent_rate": (
                sum(item.token_within_five_percent for item in matching) / len(matching)
                if matching
                else None
            ),
        },
        "exact_serialization_token_matching": {
            "absolute_token_gap": _describe(exact_absolute_gaps),
            "relative_token_gap": _describe(exact_relative_gaps),
            "within_five_percent_count": sum(
                gap <= TOKEN_MATCH_THRESHOLD for gap in exact_relative_gaps
            ),
            "within_five_percent_rate": (
                sum(gap <= TOKEN_MATCH_THRESHOLD for gap in exact_relative_gaps)
                / len(exact_relative_gaps)
                if exact_relative_gaps
                else None
            ),
        },
    }
    return report
