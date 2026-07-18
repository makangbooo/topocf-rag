"""Deterministic question-level splits for the all-observed TopoCF task."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .graph import HotpotInvariantError, validate_hotpot_example
from .hotpot import proportional_allocation, validate_hotpot_record
from .pairs import generate_question_pairs


SPLIT_SCHEMA_VERSION = 1
SPLIT_SEED = 20260719
INNER_VALIDATION_QUESTION_COUNT = 457


class AllObservedSplitInvariantError(ValueError):
    """Raised when split inputs violate the frozen design contract."""


@dataclass(frozen=True, slots=True)
class EligibleQuestion:
    """Content-free attributes needed for question-level stratification."""

    qid: str
    level: str
    pair_count: int


def _pair_count_bucket(pair_count: int) -> str:
    if not isinstance(pair_count, int) or isinstance(pair_count, bool):
        raise AllObservedSplitInvariantError("pair_count must be an integer")
    if pair_count < 1:
        raise AllObservedSplitInvariantError("pair_count must be positive")
    if pair_count == 1:
        return "1"
    if pair_count == 2:
        return "2"
    if pair_count <= 4:
        return "3-4"
    return "5+"


def _stratum(question: EligibleQuestion) -> str:
    return f"{question.level}|{_pair_count_bucket(question.pair_count)}"


def _selection_rank(question: EligibleQuestion, *, seed: int) -> bytes:
    payload = (
        f"{seed}\0{question.level}\0{_pair_count_bucket(question.pair_count)}"
        f"\0{question.qid}"
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _id_sequence_sha256(ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(ids), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_all_observed_questions(
    records: Iterable[Mapping[str, Any]],
) -> tuple[tuple[EligibleQuestion, ...], dict[str, int]]:
    """Collect only IDs and content-free strata for eligible bridge questions."""

    seen_ids: set[str] = set()
    questions: list[EligibleQuestion] = []
    record_count = 0
    bridge_count = 0
    graph_eligible_count = 0
    graph_ineligible_count = 0
    pair_count = 0

    for record in records:
        validate_hotpot_record(record)
        qid = str(record["_id"])
        if qid in seen_ids:
            raise AllObservedSplitInvariantError("source contains duplicate IDs")
        seen_ids.add(qid)
        record_count += 1
        if record.get("type") != "bridge":
            continue
        bridge_count += 1
        try:
            validate_hotpot_example(record)
        except HotpotInvariantError:
            graph_ineligible_count += 1
            continue
        graph_eligible_count += 1
        context = record["context"]
        scores = [0.0] * len(context)
        pairs = tuple(
            pair
            for pair in generate_question_pairs(
                record,
                scores,
                retrieval_top_k=len(context),
            )
            if pair.stratum == "all_observed_rewire"
            and pair.variant == "t3_all_observed"
        )
        if not pairs:
            continue
        if any(
            not edge.observed
            for pair in pairs
            for edge in (*pair.positive_edges, *pair.negative_edges)
        ):
            raise AllObservedSplitInvariantError(
                "all-observed selection contains a non-observed edge"
            )
        level = record.get("level")
        if not isinstance(level, str) or not level:
            raise AllObservedSplitInvariantError("level must be non-empty")
        questions.append(
            EligibleQuestion(qid=qid, level=level, pair_count=len(pairs))
        )
        pair_count += len(pairs)

    if record_count < 1:
        raise AllObservedSplitInvariantError("source contains no records")
    if graph_eligible_count + graph_ineligible_count != bridge_count:
        raise RuntimeError("graph eligibility does not cover bridge questions")
    questions.sort(key=lambda question: question.qid)
    return tuple(questions), {
        "record_count": record_count,
        "bridge_question_count": bridge_count,
        "graph_eligible_bridge_question_count": graph_eligible_count,
        "graph_ineligible_bridge_question_count": graph_ineligible_count,
        "all_observed_question_count": len(questions),
        "all_observed_pair_count": pair_count,
    }


def _question_summary(questions: Sequence[EligibleQuestion]) -> dict[str, Any]:
    ids = sorted(question.qid for question in questions)
    pair_histogram = Counter(question.pair_count for question in questions)
    level_histogram = Counter(question.level for question in questions)
    bucket_histogram = Counter(_pair_count_bucket(q.pair_count) for q in questions)
    stratum_histogram = Counter(_stratum(question) for question in questions)
    return {
        "question_count": len(questions),
        "pair_count": sum(question.pair_count for question in questions),
        "pair_count_histogram": {
            str(count): frequency
            for count, frequency in sorted(pair_histogram.items())
        },
        "level_question_counts": dict(sorted(level_histogram.items())),
        "pair_count_bucket_question_counts": dict(sorted(bucket_histogram.items())),
        "stratum_question_counts": dict(sorted(stratum_histogram.items())),
        "id_sequence_sha256": _id_sequence_sha256(ids),
    }


def partition_train_questions(
    questions: Sequence[EligibleQuestion],
    *,
    validation_question_count: int,
    seed: int,
) -> tuple[tuple[EligibleQuestion, ...], tuple[EligibleQuestion, ...], dict[str, int]]:
    """Partition questions with Hamilton allocation over frozen strata."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AllObservedSplitInvariantError("seed must be an integer")
    if (
        not isinstance(validation_question_count, int)
        or isinstance(validation_question_count, bool)
        or not 1 <= validation_question_count < len(questions)
    ):
        raise AllObservedSplitInvariantError(
            "validation size must leave non-empty fit and validation roles"
        )
    qids = [question.qid for question in questions]
    if len(qids) != len(set(qids)):
        raise AllObservedSplitInvariantError("eligible question IDs must be unique")

    by_stratum: defaultdict[str, list[EligibleQuestion]] = defaultdict(list)
    for question in questions:
        by_stratum[_stratum(question)].append(question)
    allocation = proportional_allocation(
        {name: len(items) for name, items in by_stratum.items()},
        validation_question_count,
    )
    validation: list[EligibleQuestion] = []
    for name in sorted(by_stratum):
        ranked = sorted(
            by_stratum[name],
            key=lambda question: (
                _selection_rank(question, seed=seed),
                question.qid,
            ),
        )
        validation.extend(ranked[: allocation[name]])
    validation_ids = {question.qid for question in validation}
    fit = [question for question in questions if question.qid not in validation_ids]
    fit.sort(key=lambda question: question.qid)
    validation.sort(key=lambda question: question.qid)
    if {question.qid for question in fit}.intersection(validation_ids):
        raise RuntimeError("fit and validation questions overlap")
    if len(fit) + len(validation) != len(questions):
        raise RuntimeError("train roles do not partition the population")
    return tuple(fit), tuple(validation), dict(sorted(allocation.items()))


def build_train_manifest(
    questions: Sequence[EligibleQuestion],
    *,
    source_path: str,
    source_sha256: str,
    population_audit_path: str,
    population_audit_sha256: str,
) -> dict[str, Any]:
    """Build the frozen official-train fit/validation ID manifest."""

    if len(questions) != 2286:
        raise AllObservedSplitInvariantError(
            "official-train eligible question count changed"
        )
    if sum(question.pair_count for question in questions) != 5682:
        raise AllObservedSplitInvariantError("official-train pair count changed")
    fit, validation, allocation = partition_train_questions(
        questions,
        validation_question_count=INNER_VALIDATION_QUESTION_COUNT,
        seed=SPLIT_SEED,
    )
    fit_ids = sorted(question.qid for question in fit)
    validation_ids = sorted(question.qid for question in validation)
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": "topocf_all_observed_train_only_inner_split",
        "dataset": "HotpotQA",
        "official_source_split": "train",
        "source": {"path": source_path, "sha256": source_sha256},
        "population_audit": {
            "path": population_audit_path,
            "sha256": population_audit_sha256,
        },
        "selector": {
            "stratum": "all_observed_rewire",
            "variant": "t3_all_observed",
        },
        "protocol": {
            "seed": SPLIT_SEED,
            "unit": "question_id",
            "validation_question_count": INNER_VALIDATION_QUESTION_COUNT,
            "fit_question_count": len(fit),
            "stratification": "HotpotQA level x pair-count bucket: 1, 2, 3-4, 5+",
            "allocation": "Hamilton proportional allocation across joint strata",
            "rank": (
                "ascending SHA256(seed NUL level NUL pair-count bucket NUL "
                "question_id), then ID"
            ),
            "official_dev_used": False,
            "all_eligible_train_questions_partitioned": True,
        },
        "population": {
            **_question_summary(questions),
            "validation_allocation": allocation,
        },
        "fit": {**_question_summary(fit), "ids": fit_ids},
        "validation": {
            **_question_summary(validation),
            "ids": validation_ids,
        },
        "integrity": {
            "fit_validation_overlap_count": 0,
            "population_partitioned_exactly": True,
            "pair_group_crossing_count": 0,
            "official_dev_used": False,
        },
    }
    validate_train_manifest(manifest)
    return manifest


def validate_train_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the frozen role sizes and question-disjointness."""

    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise AllObservedSplitInvariantError("unsupported train manifest schema")
    if manifest.get("task") != "topocf_all_observed_train_only_inner_split":
        raise AllObservedSplitInvariantError("train manifest task changed")
    if manifest.get("official_source_split") != "train":
        raise AllObservedSplitInvariantError("train manifest source changed")
    fit = manifest.get("fit")
    validation = manifest.get("validation")
    if not isinstance(fit, Mapping) or not isinstance(validation, Mapping):
        raise AllObservedSplitInvariantError("train roles must be objects")
    fit_ids = fit.get("ids")
    validation_ids = validation.get("ids")
    for name, ids, expected in (
        ("fit", fit_ids, 1829),
        ("validation", validation_ids, INNER_VALIDATION_QUESTION_COUNT),
    ):
        if (
            not isinstance(ids, list)
            or ids != sorted(ids)
            or len(ids) != expected
            or len(ids) != len(set(ids))
        ):
            raise AllObservedSplitInvariantError(f"{name} IDs changed")
        if manifest[name].get("id_sequence_sha256") != _id_sequence_sha256(ids):
            raise AllObservedSplitInvariantError(f"{name} ID hash changed")
    if set(fit_ids).intersection(validation_ids):
        raise AllObservedSplitInvariantError("train role IDs overlap")
    if fit.get("pair_count", 0) + validation.get("pair_count", 0) != 5682:
        raise AllObservedSplitInvariantError("train role pair counts changed")


def build_dev_manifest(
    questions: Sequence[EligibleQuestion],
    *,
    source_path: str,
    source_sha256: str,
    population_audit_path: str,
    population_audit_sha256: str,
) -> dict[str, Any]:
    """Freeze every eligible official-dev question for one-shot evaluation."""

    if len(questions) != 162:
        raise AllObservedSplitInvariantError(
            "official-dev eligible question count changed"
        )
    if sum(question.pair_count for question in questions) != 374:
        raise AllObservedSplitInvariantError("official-dev pair count changed")
    ids = sorted(question.qid for question in questions)
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": "topocf_all_observed_official_dev_one_shot",
        "dataset": "HotpotQA",
        "official_source_split": "dev_distractor",
        "source": {"path": source_path, "sha256": source_sha256},
        "population_audit": {
            "path": population_audit_path,
            "sha256": population_audit_sha256,
        },
        "selector": {
            "stratum": "all_observed_rewire",
            "variant": "t3_all_observed",
        },
        "protocol": {
            "unit": "question_id",
            "selection": "all eligible official-dev questions",
            "model_selection_allowed": False,
            "one_shot_evaluation_only": True,
        },
        "evaluation": {**_question_summary(questions), "ids": ids},
        "integrity": {
            "all_eligible_dev_questions_reserved": True,
            "model_selection_allowed": False,
        },
    }
