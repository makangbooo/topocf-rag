"""Leakage-safe split and in-memory data contract for TopoCF-Bind-Repair.

The frozen inner split is constructed from content-free Phase 1 manifest
metadata. Question and document text enter only after source reconstruction and
are never written by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .hotpot import proportional_allocation, sha256_file
from .serialization import (
    SerializedTopology,
    all_four_document_relabelings,
    deterministic_four_document_permutation,
    relabel_evidence_topology,
    serialize_evidence_topology,
)
from .topology import EvidenceTopology


INNER_SPLIT_SCHEMA_VERSION = 1
INNER_SPLIT_SEED = 20260718
INNER_VALIDATION_QUESTION_COUNT = 32
METHOD_DATA_SEED = 20260718
PRIMARY_STRATUM = "synthetic_common"
PRIMARY_VARIANT = "t3"


class MethodDataInvariantError(ValueError):
    """Raised when method split or training data violates the frozen contract."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodDataInvariantError(f"{field} must be an object")
    return value


def _require_ids(value: Any, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        raise MethodDataInvariantError(
            f"{field} must be a sorted unique non-empty string list"
        )
    return tuple(value)


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MethodDataInvariantError(f"{field} must be lowercase SHA256")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MethodDataInvariantError(f"{field} must be a non-negative integer")
    return value


def _selection_sha256(ids: Sequence[str]) -> str:
    encoded = json.dumps(
        list(ids), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair_count_bucket(pair_count: int) -> str:
    if (
        not isinstance(pair_count, int)
        or isinstance(pair_count, bool)
        or pair_count < 1
    ):
        raise MethodDataInvariantError("per-question pair count must be positive")
    if pair_count == 1:
        return "1"
    if pair_count == 2:
        return "2"
    if pair_count <= 4:
        return "3-4"
    return "5+"


def _selection_rank(qid: str, *, bucket: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{bucket}\0{qid}".encode("utf-8")).digest()


def _primary_records(pair_manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if pair_manifest.get("schema_version") != 1:
        raise MethodDataInvariantError("unsupported Phase 1 pair manifest schema")
    if pair_manifest.get("official_split") != "train":
        raise MethodDataInvariantError("inner split must come from official train")
    records = pair_manifest.get("pairs")
    if not isinstance(records, list):
        raise MethodDataInvariantError("pair manifest must contain a pairs list")
    selected: list[Mapping[str, Any]] = []
    pair_ids: set[str] = set()
    for record in records:
        payload = _require_mapping(record, "pair record")
        pair_id = payload.get("pair_id")
        qid = payload.get("qid")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise MethodDataInvariantError("pair IDs must be unique non-empty strings")
        if not isinstance(qid, str) or not qid:
            raise MethodDataInvariantError("pair qid must be a non-empty string")
        pair_ids.add(pair_id)
        if (
            payload.get("stratum") == PRIMARY_STRATUM
            and payload.get("variant") == PRIMARY_VARIANT
        ):
            selected.append(payload)
    if not selected:
        raise MethodDataInvariantError("pair manifest contains no primary T3 pairs")
    return tuple(selected)


def build_inner_split_manifest(
    pair_manifest: Mapping[str, Any],
    *,
    pair_manifest_sha256: str,
    validation_question_count: int = INNER_VALIDATION_QUESTION_COUNT,
    seed: int = INNER_SPLIT_SEED,
) -> dict[str, Any]:
    """Build a deterministic question-disjoint train-only inner split."""

    source_digest = _require_sha256(
        pair_manifest_sha256, "pair_manifest_sha256"
    )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise MethodDataInvariantError("seed must be an integer")
    if seed != INNER_SPLIT_SEED:
        raise MethodDataInvariantError("inner split seed is frozen")
    if (
        not isinstance(validation_question_count, int)
        or isinstance(validation_question_count, bool)
        or validation_question_count < 1
    ):
        raise MethodDataInvariantError(
            "validation_question_count must be a positive integer"
        )
    if validation_question_count != INNER_VALIDATION_QUESTION_COUNT:
        raise MethodDataInvariantError("inner validation size is frozen")

    records = _primary_records(pair_manifest)
    pairs_by_qid: dict[str, list[str]] = defaultdict(list)
    for record in records:
        pairs_by_qid[str(record["qid"])].append(str(record["pair_id"]))
    if validation_question_count >= len(pairs_by_qid):
        raise MethodDataInvariantError(
            "inner validation must leave at least one fitting question"
        )

    ids_by_bucket: dict[str, list[str]] = defaultdict(list)
    for qid, pair_ids in pairs_by_qid.items():
        ids_by_bucket[_pair_count_bucket(len(pair_ids))].append(qid)
    bucket_counts = {
        bucket: len(qids) for bucket, qids in sorted(ids_by_bucket.items())
    }
    allocation = proportional_allocation(bucket_counts, validation_question_count)
    validation_ids: list[str] = []
    for bucket in sorted(ids_by_bucket):
        ranked = sorted(
            ids_by_bucket[bucket],
            key=lambda qid: (_selection_rank(qid, bucket=bucket, seed=seed), qid),
        )
        validation_ids.extend(ranked[: allocation[bucket]])
    validation_ids = sorted(validation_ids)
    fit_ids = sorted(set(pairs_by_qid).difference(validation_ids))
    if set(fit_ids).intersection(validation_ids):
        raise RuntimeError("inner fitting and validation IDs overlap")
    if set(fit_ids).union(validation_ids) != set(pairs_by_qid):
        raise RuntimeError("inner split does not partition all primary questions")

    def role_summary(ids: Sequence[str]) -> dict[str, Any]:
        pair_histogram = Counter(len(pairs_by_qid[qid]) for qid in ids)
        bucket_histogram = Counter(
            _pair_count_bucket(len(pairs_by_qid[qid])) for qid in ids
        )
        return {
            "question_count": len(ids),
            "pair_count": sum(len(pairs_by_qid[qid]) for qid in ids),
            "pair_count_histogram": {
                str(count): frequency
                for count, frequency in sorted(pair_histogram.items())
            },
            "bucket_question_counts": dict(sorted(bucket_histogram.items())),
            "id_sequence_sha256": _selection_sha256(ids),
        }

    manifest = {
        "schema_version": INNER_SPLIT_SCHEMA_VERSION,
        "task": "topocf_bind_repair_train_only_inner_split",
        "dataset": "HotpotQA",
        "official_source_split": "train",
        "source_pair_manifest": {
            "path": "data/phase1/hotpot_train_phase1_pairs.json",
            "sha256": source_digest,
        },
        "selector": {
            "stratum": PRIMARY_STRATUM,
            "variant": PRIMARY_VARIANT,
        },
        "protocol": {
            "seed": seed,
            "unit": "question_id",
            "validation_question_count": validation_question_count,
            "stratification": "primary T3 pair-count buckets: 1, 2, 3-4, 5+",
            "allocation": "Hamilton proportional allocation across buckets",
            "rank": "ascending SHA256(seed NUL bucket NUL question_id), then ID",
            "official_dev_used": False,
            "confirmatory_pairs_used": False,
        },
        "population": {
            "question_count": len(pairs_by_qid),
            "pair_count": len(records),
            "bucket_question_counts": bucket_counts,
            "validation_allocation": dict(sorted(allocation.items())),
        },
        "fit": {**role_summary(fit_ids), "ids": fit_ids},
        "validation": {
            **role_summary(validation_ids),
            "ids": validation_ids,
        },
        "integrity": {
            "fit_validation_overlap_count": 0,
            "population_partitioned_exactly": True,
            "pair_group_crossing_count": 0,
        },
    }
    validate_inner_split_manifest(
        manifest, expected_pair_manifest_sha256=source_digest
    )
    return manifest


def validate_inner_split_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_pair_manifest_sha256: str,
) -> None:
    if manifest.get("schema_version") != INNER_SPLIT_SCHEMA_VERSION:
        raise MethodDataInvariantError("unsupported inner split schema")
    if manifest.get("task") != "topocf_bind_repair_train_only_inner_split":
        raise MethodDataInvariantError("inner split task changed")
    if manifest.get("official_source_split") != "train":
        raise MethodDataInvariantError("inner split is not official-train-only")
    source = _require_mapping(
        manifest.get("source_pair_manifest"), "source_pair_manifest"
    )
    if source.get("sha256") != _require_sha256(
        expected_pair_manifest_sha256, "expected_pair_manifest_sha256"
    ):
        raise MethodDataInvariantError("inner split source hash changed")
    selector = _require_mapping(manifest.get("selector"), "selector")
    if (selector.get("stratum"), selector.get("variant")) != (
        PRIMARY_STRATUM,
        PRIMARY_VARIANT,
    ):
        raise MethodDataInvariantError("inner split selector changed")
    protocol = _require_mapping(manifest.get("protocol"), "protocol")
    if protocol.get("seed") != INNER_SPLIT_SEED:
        raise MethodDataInvariantError("inner split seed changed")
    if (
        protocol.get("validation_question_count")
        != INNER_VALIDATION_QUESTION_COUNT
    ):
        raise MethodDataInvariantError("inner validation size changed")
    if protocol.get("official_dev_used") is not False:
        raise MethodDataInvariantError("official dev must not be used")
    if protocol.get("confirmatory_pairs_used") is not False:
        raise MethodDataInvariantError("confirmatory pairs must not be used")

    fit = _require_mapping(manifest.get("fit"), "fit")
    validation = _require_mapping(manifest.get("validation"), "validation")
    fit_ids = _require_ids(fit.get("ids"), "fit.ids")
    validation_ids = _require_ids(validation.get("ids"), "validation.ids")
    if set(fit_ids).intersection(validation_ids):
        raise MethodDataInvariantError("fit and validation IDs overlap")
    population = _require_mapping(manifest.get("population"), "population")
    if len(fit_ids) + len(validation_ids) != population.get("question_count"):
        raise MethodDataInvariantError("inner IDs do not cover the population")
    fit_pair_count = _require_nonnegative_int(
        fit.get("pair_count"), "fit.pair_count"
    )
    validation_pair_count = _require_nonnegative_int(
        validation.get("pair_count"), "validation.pair_count"
    )
    population_pair_count = _require_nonnegative_int(
        population.get("pair_count"), "population.pair_count"
    )
    if fit_pair_count + validation_pair_count != population_pair_count:
        raise MethodDataInvariantError("inner pair counts do not cover the population")
    for name, payload, ids in (
        ("fit", fit, fit_ids),
        ("validation", validation, validation_ids),
    ):
        if payload.get("question_count") != len(ids):
            raise MethodDataInvariantError(f"{name} question count changed")
        if payload.get("id_sequence_sha256") != _selection_sha256(ids):
            raise MethodDataInvariantError(f"{name} ID sequence hash changed")
        histogram = _require_mapping(
            payload.get("pair_count_histogram"), f"{name}.pair_count_histogram"
        )
        try:
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in histogram.values()
            ):
                raise ValueError
            histogram_question_count = sum(histogram.values())
            histogram_pair_count = sum(
                int(pair_count) * frequency
                for pair_count, frequency in histogram.items()
            )
        except (TypeError, ValueError) as exc:
            raise MethodDataInvariantError(
                f"{name} pair-count histogram is invalid"
            ) from exc
        if histogram_question_count != len(ids):
            raise MethodDataInvariantError(
                f"{name} pair-count histogram question total changed"
            )
        if histogram_pair_count != payload.get("pair_count"):
            raise MethodDataInvariantError(
                f"{name} pair-count histogram pair total changed"
            )
    integrity = _require_mapping(manifest.get("integrity"), "integrity")
    if integrity != {
        "fit_validation_overlap_count": 0,
        "population_partitioned_exactly": True,
        "pair_group_crossing_count": 0,
    }:
        raise MethodDataInvariantError("inner split integrity markers changed")


def load_inner_split_manifest(
    path: str | Path,
    *,
    expected_pair_manifest_sha256: str,
) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = _require_mapping(payload, "inner split manifest")
    validate_inner_split_manifest(
        manifest,
        expected_pair_manifest_sha256=expected_pair_manifest_sha256,
    )
    return manifest


def inner_role_ids(manifest: Mapping[str, Any], role: str) -> tuple[str, ...]:
    if role not in {"fit", "validation"}:
        raise MethodDataInvariantError("role must be fit or validation")
    payload = _require_mapping(manifest.get(role), role)
    return _require_ids(payload.get("ids"), f"{role}.ids")


def _binding_matrix(topology: EvidenceTopology) -> tuple[tuple[int, ...], ...]:
    if len(topology.documents) != 4:
        raise MethodDataInvariantError("binding matrix requires four documents")
    matrix = [[0] * 4 for _ in range(4)]
    sources: set[str] = set()
    targets: set[str] = set()
    mention_count = 0
    for edge in topology.edges:
        if edge.relation != "title_mention":
            continue
        source_index = int(edge.source[1:])
        target_index = int(edge.target[1:])
        matrix[source_index][target_index] = 1
        sources.add(edge.source)
        targets.add(edge.target)
        mention_count += 1
    if mention_count != 2 or len(sources) != 2 or len(targets) != 2:
        raise MethodDataInvariantError(
            "primary T3 binding must be a two-by-two one-to-one assignment"
        )
    return tuple(tuple(row) for row in matrix)


def _retrieval_root_mask(topology: EvidenceTopology) -> tuple[int, int, int, int]:
    mask = [0] * 4
    for edge in topology.edges:
        if edge.relation == "retrieval":
            mask[int(edge.target[1:])] = 1
    if sum(mask) != 2:
        raise MethodDataInvariantError("primary T3 must have two retrieval roots")
    return (mask[0], mask[1], mask[2], mask[3])


@dataclass(frozen=True, slots=True)
class CounterfactualTrainingExample:
    """One private in-memory paired example; metadata is not a model input."""

    pair_id: str
    qid: str
    question: str
    epoch: int
    old_aliases_in_new_order: tuple[str, str, str, str]
    positive_topology: EvidenceTopology
    negative_topology: EvidenceTopology
    positive: SerializedTopology
    negative: SerializedTopology
    positive_binding_target: tuple[tuple[int, ...], ...]
    negative_binding: tuple[tuple[int, ...], ...]
    retrieval_root_mask: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CounterfactualEvaluationExample:
    """The exact synchronized 24-permutation orbit for one paired example."""

    pair_id: str
    qid: str
    question: str
    positive_topology_orbit: tuple[EvidenceTopology, ...]
    negative_topology_orbit: tuple[EvidenceTopology, ...]
    positive_orbit: tuple[SerializedTopology, ...]
    negative_orbit: tuple[SerializedTopology, ...]
    positive_binding_targets: tuple[tuple[tuple[int, ...], ...], ...]
    negative_bindings: tuple[tuple[tuple[int, ...], ...], ...]


def _selected_prepared_pairs(prepared: Any, ids: Sequence[str]) -> tuple[Any, ...]:
    if getattr(prepared, "official_split", None) != "train":
        raise MethodDataInvariantError("inner method data must be official train")
    requested = set(ids)
    if not requested or len(requested) != len(ids):
        raise MethodDataInvariantError("selected IDs must be unique and non-empty")
    selected = tuple(
        pair
        for pair in getattr(prepared, "pairs", ())
        if pair.qid in requested
        and pair.stratum == PRIMARY_STRATUM
        and pair.variant == PRIMARY_VARIANT
    )
    seen = {pair.qid for pair in selected}
    if seen != requested:
        raise MethodDataInvariantError(
            "prepared primary pairs do not cover the selected question IDs"
        )
    return tuple(sorted(selected, key=lambda pair: pair.pair_id))


def build_training_examples(
    prepared: Any,
    inner_split_manifest: Mapping[str, Any],
    *,
    epoch: int,
    seed: int = METHOD_DATA_SEED,
) -> tuple[CounterfactualTrainingExample, ...]:
    """Build one S4 augmentation per fit pair; validation IDs are inaccessible."""

    examples: list[CounterfactualTrainingExample] = []
    fit_ids = inner_role_ids(inner_split_manifest, "fit")
    for pair in _selected_prepared_pairs(prepared, fit_ids):
        permutation = deterministic_four_document_permutation(
            seed=seed, epoch=epoch, item_key=pair.pair_id
        )
        positive = relabel_evidence_topology(
            pair.positive.topology, permutation
        )
        negative = relabel_evidence_topology(
            pair.negative.topology, permutation
        )
        if positive.canonical_signature != negative.canonical_signature:
            raise MethodDataInvariantError("primary pair canonical topology changed")
        if positive.degree_profile != negative.degree_profile:
            raise MethodDataInvariantError("primary pair degree profile changed")
        if positive.document_text_multiset != negative.document_text_multiset:
            raise MethodDataInvariantError("primary pair document payload changed")
        if _retrieval_root_mask(positive) != _retrieval_root_mask(negative):
            raise MethodDataInvariantError("primary pair retrieval roots changed")
        examples.append(
            CounterfactualTrainingExample(
                pair_id=pair.pair_id,
                qid=pair.qid,
                question=positive.question,
                epoch=epoch,
                old_aliases_in_new_order=permutation,
                positive_topology=positive,
                negative_topology=negative,
                positive=serialize_evidence_topology(positive),
                negative=serialize_evidence_topology(negative),
                positive_binding_target=_binding_matrix(positive),
                negative_binding=_binding_matrix(negative),
                retrieval_root_mask=_retrieval_root_mask(positive),
            )
        )
    return tuple(examples)


def build_evaluation_examples(
    prepared: Any,
    inner_split_manifest: Mapping[str, Any],
) -> tuple[CounterfactualEvaluationExample, ...]:
    """Build the synchronized exact S4 orbit for inner-validation IDs only."""

    examples: list[CounterfactualEvaluationExample] = []
    validation_ids = inner_role_ids(inner_split_manifest, "validation")
    for pair in _selected_prepared_pairs(prepared, validation_ids):
        positives = all_four_document_relabelings(pair.positive.topology)
        negatives = all_four_document_relabelings(pair.negative.topology)
        if len(positives) != 24 or len(negatives) != 24:
            raise RuntimeError("S4 orbit size changed")
        for positive, negative in zip(positives, negatives, strict=True):
            if positive.document_text_multiset != negative.document_text_multiset:
                raise MethodDataInvariantError(
                    "evaluation orbit relabelings are not synchronized"
                )
        examples.append(
            CounterfactualEvaluationExample(
                pair_id=pair.pair_id,
                qid=pair.qid,
                question=pair.positive.topology.question,
                positive_topology_orbit=positives,
                negative_topology_orbit=negatives,
                positive_orbit=tuple(
                    serialize_evidence_topology(item) for item in positives
                ),
                negative_orbit=tuple(
                    serialize_evidence_topology(item) for item in negatives
                ),
                positive_binding_targets=tuple(
                    _binding_matrix(item) for item in positives
                ),
                negative_bindings=tuple(
                    _binding_matrix(item) for item in negatives
                ),
            )
        )
    return tuple(examples)


def source_pair_manifest_sha256(path: str | Path) -> str:
    """Expose the common streaming SHA helper under the method data API."""

    return sha256_file(path)
