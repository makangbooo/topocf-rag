"""Aggregate-only readiness audit for untouched official 2Wiki test data."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from typing import Any

from .musique_test_readiness import EXPECTED_PRIMARY_CONFIG
from .twowiki import (
    DATASET_NAME,
    QUESTION_TYPES,
    TwoWikiInvariantError,
    assess_twowiki_eligibility,
)
from .twowiki_graph_retrieval import FROZEN_HOTPOT_TRANSFER_CONFIG


TWOWIKI_TEST_READINESS_SCHEMA_VERSION = 1
EXPECTED_TWOWIKI_TEST_SHA256 = (
    "48b196d4ba8557343abb9bd1ad03566bc02762ecd734617ff910027c33821b04"
)
EXPECTED_MUSIQUE_STAGE_G1_REPORT_SHA256 = (
    "f3cb0f2183cff8c22de7a16991e96c249d6a103553b9af2e6637c3130b678f70"
)
EXPECTED_TWOWIKI_TRAIN_MANIFEST_SHA256 = (
    "812fd21e8280b25ee90a392d1c40e72f2d7111585e2cb6ac2ae0f9a9b48e7028"
)
EXPECTED_TWOWIKI_DEV_MANIFEST_SHA256 = (
    "e0137937c2780618715686574e6002e9b75768dbccfc39894984d92cb272d207"
)
MINIMUM_ELIGIBLE_PER_TYPE = 100


class TwoWikiTestReadinessError(ValueError):
    """Raised when the untouched 2Wiki test audit cannot be bound."""


def _string_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda value: str(value))
    }


def audit_twowiki_test_records(
    records: Iterable[Any],
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Inspect schema and eligibility without serializing record content."""

    record_count = 0
    non_object_count = 0
    invalid_id_count = 0
    duplicate_id_count = 0
    schema_invalid_count = 0
    validatable_record_count = 0
    eligible_count = 0
    keysets: Counter[tuple[str, ...]] = Counter()
    field_types: defaultdict[str, Counter[str]] = defaultdict(Counter)
    context_lengths: Counter[int | None] = Counter()
    supporting_fact_counts: Counter[int | None] = Counter()
    question_type_counts: Counter[str] = Counter()
    eligible_by_type: defaultdict[str, list[str]] = defaultdict(list)
    eligibility_exclusions: Counter[str] = Counter()
    eligibility_reason_sets: Counter[tuple[str, ...]] = Counter()
    gold_document_counts: Counter[int] = Counter()
    ids: list[str] = []
    seen_ids: set[str] = set()

    audited_fields = (
        "_id",
        "question",
        "answer",
        "type",
        "context",
        "supporting_facts",
    )
    for record in records:
        record_count += 1
        if not isinstance(record, Mapping):
            non_object_count += 1
            continue
        keysets[tuple(sorted(str(key) for key in record))] += 1
        for field in audited_fields:
            field_types[field][type(record.get(field)).__name__] += 1
        context = record.get("context")
        supporting_facts = record.get("supporting_facts")
        context_lengths[len(context) if isinstance(context, list) else None] += 1
        supporting_fact_counts[
            len(supporting_facts) if isinstance(supporting_facts, list) else None
        ] += 1
        question_type = record.get("type")
        if isinstance(question_type, str):
            question_type_counts[question_type] += 1
        qid = record.get("_id")
        if not isinstance(qid, str) or not qid:
            invalid_id_count += 1
            continue
        if qid in seen_ids:
            duplicate_id_count += 1
        else:
            seen_ids.add(qid)
            ids.append(qid)
        try:
            assessment = assess_twowiki_eligibility(record)
        except (TwoWikiInvariantError, KeyError, TypeError, ValueError):
            schema_invalid_count += 1
            continue
        validatable_record_count += 1
        gold_document_counts[assessment.gold_document_count] += 1
        if assessment.eligible:
            eligible_count += 1
            eligible_by_type[assessment.question_type].append(qid)
        else:
            eligibility_reason_sets[assessment.reasons] += 1
            eligibility_exclusions.update(assessment.reasons)

    public = {
        "record_count": record_count,
        "non_object_count": non_object_count,
        "invalid_id_count": invalid_id_count,
        "unique_id_count": len(seen_ids),
        "duplicate_id_count": duplicate_id_count,
        "schema_invalid_count": schema_invalid_count,
        "validatable_record_count": validatable_record_count,
        "eligible_count": eligible_count,
        "eligible_rate": eligible_count / record_count if record_count else None,
        "field_contract": {
            "keysets": [
                {"keys": list(keys), "count": count}
                for keys, count in sorted(keysets.items())
            ],
            "field_type_histograms": {
                field: _string_counter(counter)
                for field, counter in sorted(field_types.items())
            },
        },
        "histograms": {
            "context_length": _string_counter(context_lengths),
            "supporting_fact_count": _string_counter(supporting_fact_counts),
            "question_type": _string_counter(question_type_counts),
            "gold_document_count": _string_counter(gold_document_counts),
            "eligible_count_by_question_type": {
                question_type: len(eligible_by_type.get(question_type, ()))
                for question_type in QUESTION_TYPES
            },
        },
        "eligibility_exclusions": {
            "reason_occurrence_count": _string_counter(
                eligibility_exclusions
            ),
            "reason_set_count": [
                {"reasons": list(reasons), "count": count}
                for reasons, count in sorted(eligibility_reason_sets.items())
            ],
        },
    }
    return (
        public,
        tuple(ids),
        {
            question_type: tuple(eligible_by_type.get(question_type, ()))
            for question_type in QUESTION_TYPES
        },
    )


def _manifest_ids(
    manifest: Mapping[str, Any], *, expected_split: str
) -> tuple[str, ...]:
    if manifest.get("dataset") != DATASET_NAME:
        raise TwoWikiTestReadinessError("2Wiki manifest dataset changed")
    if manifest.get("official_split") != expected_split:
        raise TwoWikiTestReadinessError("2Wiki manifest split changed")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or any(
        not isinstance(qid, str) or not qid for qid in ids
    ):
        raise TwoWikiTestReadinessError("2Wiki manifest IDs are invalid")
    if len(ids) != len(set(ids)):
        raise TwoWikiTestReadinessError("2Wiki manifest IDs are not unique")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or selection.get("sample_size") != len(ids):
        raise TwoWikiTestReadinessError("2Wiki manifest sample size changed")
    return tuple(ids)


def _validate_musique_g1_failure(
    report: Mapping[str, Any], report_sha256: str
) -> Mapping[str, Any]:
    if report_sha256 != EXPECTED_MUSIQUE_STAGE_G1_REPORT_SHA256:
        raise TwoWikiTestReadinessError("MuSiQue Stage G1 SHA256 changed")
    if report.get("task") != (
        "musique_unused_official_train_holdout_readiness_audit"
    ):
        raise TwoWikiTestReadinessError("MuSiQue Stage G1 task changed")
    gate = report.get("readiness_gate")
    hypotheses = report.get("frozen_holdout_hypotheses")
    if not isinstance(gate, Mapping) or gate.get("passed") is not False:
        raise TwoWikiTestReadinessError("MuSiQue Stage G1 must remain failed")
    if gate.get("status") != "stop_before_remainder_scoring":
        raise TwoWikiTestReadinessError("MuSiQue Stage G1 status changed")
    if not isinstance(hypotheses, Mapping):
        raise TwoWikiTestReadinessError("MuSiQue hypotheses are missing")
    return hypotheses


def build_twowiki_test_readiness_report(
    *,
    source_path: str,
    source_sha256: str,
    source_size_bytes: int,
    audit: Mapping[str, Any],
    test_ids: tuple[str, ...],
    eligible_ids_by_type: Mapping[str, tuple[str, ...]],
    selected_train_manifest: Mapping[str, Any],
    selected_train_manifest_path: str,
    selected_train_manifest_sha256: str,
    selected_dev_manifest: Mapping[str, Any],
    selected_dev_manifest_path: str,
    selected_dev_manifest_sha256: str,
    musique_g1_report: Mapping[str, Any],
    musique_g1_report_path: str,
    musique_g1_report_sha256: str,
) -> dict[str, Any]:
    """Freeze cross-dataset hypotheses before any 2Wiki test scoring."""

    frozen_musique_hypotheses = _validate_musique_g1_failure(
        musique_g1_report, musique_g1_report_sha256
    )
    train_ids = _manifest_ids(
        selected_train_manifest, expected_split="train"
    )
    dev_ids = _manifest_ids(selected_dev_manifest, expected_split="dev")
    test_set = set(test_ids)
    train_set = set(train_ids)
    dev_set = set(dev_ids)
    test_train_overlap = test_set.intersection(train_set)
    test_dev_overlap = test_set.intersection(dev_set)
    train_dev_overlap = train_set.intersection(dev_set)
    eligible_counts = {
        question_type: len(eligible_ids_by_type.get(question_type, ()))
        for question_type in QUESTION_TYPES
    }
    eligible_ids = {
        qid for values in eligible_ids_by_type.values() for qid in values
    }
    outgoing = replace(EXPECTED_PRIMARY_CONFIG, direction="outgoing")
    incoming = replace(EXPECTED_PRIMARY_CONFIG, direction="incoming")
    checks = {
        "official_test_source_hash": (
            source_sha256 == EXPECTED_TWOWIKI_TEST_SHA256
        ),
        "record_count_positive": bool(audit.get("record_count", 0)),
        "all_records_are_objects": audit.get("non_object_count") == 0,
        "ids_valid_and_unique": (
            audit.get("invalid_id_count") == 0
            and audit.get("duplicate_id_count") == 0
            and len(test_ids) == audit.get("record_count")
        ),
        "all_records_have_validatable_gold_schema": (
            audit.get("validatable_record_count") == audit.get("record_count")
            and audit.get("schema_invalid_count") == 0
        ),
        "all_four_question_types_have_eligible_records": all(
            eligible_counts[question_type] >= MINIMUM_ELIGIBLE_PER_TYPE
            for question_type in QUESTION_TYPES
        ),
        "eligible_ids_unique": (
            len(eligible_ids) == sum(eligible_counts.values())
        ),
        "test_has_no_selected_train_overlap": not test_train_overlap,
        "test_has_no_selected_dev_overlap": not test_dev_overlap,
        "selected_train_dev_do_not_overlap": not train_dev_overlap,
        "selected_train_manifest_hash_frozen": (
            selected_train_manifest_sha256
            == EXPECTED_TWOWIKI_TRAIN_MANIFEST_SHA256
        ),
        "selected_dev_manifest_hash_frozen": (
            selected_dev_manifest_sha256
            == EXPECTED_TWOWIKI_DEV_MANIFEST_SHA256
        ),
        "musique_failed_holdout_chain_bound": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": TWOWIKI_TEST_READINESS_SCHEMA_VERSION,
        "task": "2wiki_untouched_official_test_readiness_audit",
        "scope": "official 2WikiMultiHopQA test controlled ten-document pools",
        "musique_g1": {
            "path": musique_g1_report_path,
            "sha256": musique_g1_report_sha256,
            "gate_passed": False,
            "status": "stop_before_remainder_scoring",
        },
        "protocol": {
            "declared_after_musique_test_and_remainder_were_unavailable": True,
            "performed_before_any_2wiki_test_embedding_or_retrieval_scoring": True,
            "test_used_for_hyperparameter_selection": False,
            "sampling_performed": False,
            "test_pool_rule": (
                "all records satisfying the frozen 2Wiki eligibility predicate"
            ),
            "minimum_eligible_per_question_type": MINIMUM_ELIGIBLE_PER_TYPE,
            "primary_aggregate": "equal-weight macro across four question types",
            "secondary_aggregate": "eligible-test-distribution micro",
            "configuration_selection_after_test_forbidden": True,
            "rerun_after_observing_test_forbidden": True,
            "claim_status": (
                "untouched official-test controlled-pool replication; 2Wiki "
                "train/dev were previously observed; not fully dataset-"
                "independent or full-corpus GraphRAG"
            ),
        },
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "expected_sha256": EXPECTED_TWOWIKI_TEST_SHA256,
            "size_bytes": source_size_bytes,
            **dict(audit),
        },
        "split_separation": {
            "train_manifest_path": selected_train_manifest_path,
            "train_manifest_sha256": selected_train_manifest_sha256,
            "train_selected_count": len(train_ids),
            "dev_manifest_path": selected_dev_manifest_path,
            "dev_manifest_sha256": selected_dev_manifest_sha256,
            "dev_selected_count": len(dev_ids),
            "test_train_overlap_count": len(test_train_overlap),
            "test_dev_overlap_count": len(test_dev_overlap),
            "train_dev_overlap_count": len(train_dev_overlap),
        },
        "eligible_test_pool": {
            "eligible_count": len(eligible_ids),
            "eligible_count_by_question_type": eligible_counts,
            "manifest_order": "official test source order after eligibility filtering",
        },
        "frozen_test_hypotheses": {
            "musique_primary_zero_shot": {
                "config": asdict(EXPECTED_PRIMARY_CONFIG),
                "baseline": "dense",
                "success": (
                    "positive four-type paired-bootstrap 95% lower bound for "
                    "macro mean complete-evidence-rate delta"
                ),
            },
            "same_config_degree_control": {
                "config": asdict(EXPECTED_PRIMARY_CONFIG),
                "success": (
                    "positive paired-bootstrap 95% lower bound versus the "
                    "same-config degree prior"
                ),
            },
            "direction_mechanism": {
                "outgoing_config": asdict(outgoing),
                "incoming_config": asdict(incoming),
                "confirmatory_contrast": "outgoing minus incoming",
                "required_mean_complete_delta": 0.02,
                "required_paired_bootstrap_95_low": 0.0,
                "outgoing_minus_undirected": "descriptive only",
            },
            "legacy_hotpot_transfer_secondary": {
                "config": asdict(FROZEN_HOTPOT_TRANSFER_CONFIG),
                "baseline": "dense_bm25_rrf",
                "confirmatory": False,
            },
            "source_hypotheses": dict(frozen_musique_hypotheses),
        },
        "readiness_gate": {
            "name": "2wiki_untouched_official_test_readiness_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "inspect_counts_then_freeze_2wiki_test_manifest"
                if passed
                else "stop_before_2wiki_test_scoring"
            ),
            "interpretation": (
                "Passing verifies official source identity, complete gold-label "
                "schema, frozen eligibility, per-type capacity, and split "
                "separation. It authorizes manifest materialization only."
            ),
        },
        "content_contract": (
            "aggregate counts, configurations, paths, and hashes only; no IDs, "
            "questions, answers, titles, sentences, evidence text, or supporting "
            "facts"
        ),
    }
