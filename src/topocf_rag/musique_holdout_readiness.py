"""Readiness audit for an untouched MuSiQue official-train remainder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .musique_splits import CandidatePool, all_cells


MUSIQUE_HOLDOUT_READINESS_SCHEMA_VERSION = 1
EXPECTED_STAGE_C_REPORT_SHA256 = (
    "4904e10eb7cb118cf5ed8bfe26516d24b8f4aca9fc6122364688944aada29a5c"
)
EXPECTED_STAGE_G0_REPORT_SHA256 = (
    "929b47215c23518c0cfc330469c0ade9914b2eab6eb92ca79409fd8b41b4837c"
)
MINIMUM_REMAINDER_PER_CELL = 100


class MuSiQueHoldoutReadinessError(ValueError):
    """Raised when the untouched-remainder contract cannot be verified."""


def _manifest_ids(
    manifest: Mapping[str, Any], *, expected_split: str
) -> tuple[str, ...]:
    if manifest.get("dataset") != "musique_ans_v1.0":
        raise MuSiQueHoldoutReadinessError("manifest dataset changed")
    if manifest.get("split") != expected_split:
        raise MuSiQueHoldoutReadinessError("manifest split changed")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or any(
        not isinstance(qid, str) or not qid for qid in ids
    ):
        raise MuSiQueHoldoutReadinessError("manifest IDs are invalid")
    if len(ids) != len(set(ids)):
        raise MuSiQueHoldoutReadinessError("manifest IDs are not unique")
    if manifest.get("selected_count") != len(ids):
        raise MuSiQueHoldoutReadinessError("manifest selected count changed")
    return tuple(ids)


def _validate_stage_c(
    report: Mapping[str, Any], report_sha256: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if report_sha256 != EXPECTED_STAGE_C_REPORT_SHA256:
        raise MuSiQueHoldoutReadinessError("Stage C report SHA256 changed")
    if report.get("task") != (
        "musique_hop_collision_stratified_split_materialization"
    ):
        raise MuSiQueHoldoutReadinessError("Stage C task changed")
    gate = report.get("gate")
    train = report.get("train")
    artifacts = report.get("artifacts")
    if not isinstance(gate, Mapping) or not gate.get("passed"):
        raise MuSiQueHoldoutReadinessError("Stage C gate did not pass")
    if not isinstance(train, Mapping) or not isinstance(artifacts, Mapping):
        raise MuSiQueHoldoutReadinessError("Stage C sections are missing")
    return train, artifacts


def _validate_stage_g0_failure(
    report: Mapping[str, Any], report_sha256: str
) -> Mapping[str, Any]:
    if report_sha256 != EXPECTED_STAGE_G0_REPORT_SHA256:
        raise MuSiQueHoldoutReadinessError("Stage G0 report SHA256 changed")
    if report.get("task") != (
        "musique_answerable_test_final_holdout_readiness_audit"
    ):
        raise MuSiQueHoldoutReadinessError("Stage G0 task changed")
    gate = report.get("readiness_gate")
    source = report.get("source")
    hypotheses = report.get("frozen_test_hypotheses")
    if not isinstance(gate, Mapping) or gate.get("passed") is not False:
        raise MuSiQueHoldoutReadinessError(
            "Stage G0 must contain the frozen failed gate"
        )
    if gate.get("status") != "stop_before_test_scoring":
        raise MuSiQueHoldoutReadinessError("Stage G0 failure status changed")
    if not isinstance(source, Mapping) or not isinstance(hypotheses, Mapping):
        raise MuSiQueHoldoutReadinessError("Stage G0 sections are missing")
    field_contract = source.get("field_contract")
    if not isinstance(field_contract, Mapping):
        raise MuSiQueHoldoutReadinessError(
            "Stage G0 field contract is missing"
        )
    keysets = field_contract.get("keysets")
    if keysets != [
        {"count": 2459, "keys": ["id", "paragraphs", "question"]}
    ]:
        raise MuSiQueHoldoutReadinessError(
            "Stage G0 unlabeled-test schema changed"
        )
    return hypotheses


def build_musique_holdout_readiness_report(
    *,
    train_candidate_pool: CandidatePool,
    train_source_path: str,
    train_source_sha256: str,
    stage_c_report: Mapping[str, Any],
    stage_c_report_path: str,
    stage_c_report_sha256: str,
    stage_g0_report: Mapping[str, Any],
    stage_g0_report_path: str,
    stage_g0_report_sha256: str,
    selected_train_manifest: Mapping[str, Any],
    selected_train_manifest_path: str,
    selected_train_manifest_sha256: str,
    selected_dev_manifest: Mapping[str, Any],
    selected_dev_manifest_path: str,
    selected_dev_manifest_sha256: str,
) -> dict[str, Any]:
    """Audit unused labeled train candidates without scoring any of them."""

    stage_c_train, stage_c_artifacts = _validate_stage_c(
        stage_c_report, stage_c_report_sha256
    )
    frozen_hypotheses = _validate_stage_g0_failure(
        stage_g0_report, stage_g0_report_sha256
    )
    selected_train_ids = _manifest_ids(
        selected_train_manifest, expected_split="train"
    )
    selected_dev_ids = _manifest_ids(
        selected_dev_manifest, expected_split="dev"
    )
    candidate_ids = set(train_candidate_pool.ordered_ids)
    selected_train_set = set(selected_train_ids)
    selected_dev_set = set(selected_dev_ids)
    selected_not_in_pool = selected_train_set.difference(candidate_ids)
    train_dev_overlap = selected_train_set.intersection(selected_dev_set)
    candidate_dev_overlap = candidate_ids.intersection(selected_dev_set)

    remainder_ids_by_cell = {
        cell: tuple(
            qid
            for qid in train_candidate_pool.ids_by_cell.get(cell, ())
            if qid not in selected_train_set
        )
        for cell in all_cells()
    }
    remainder_counts = {
        cell: len(remainder_ids_by_cell[cell]) for cell in all_cells()
    }
    remainder_count = sum(remainder_counts.values())
    stage_c_cell_counts = stage_c_train.get("candidate_count_by_cell")
    if not isinstance(stage_c_cell_counts, Mapping):
        raise MuSiQueHoldoutReadinessError(
            "Stage C candidate cell counts are missing"
        )

    checks = {
        "train_source_hash_matches_stage_c": (
            train_source_sha256 == stage_c_train.get("source_sha256")
        ),
        "train_source_record_count_matches_stage_c": (
            train_candidate_pool.record_count
            == stage_c_train.get("source_record_count")
        ),
        "candidate_count_matches_stage_c": (
            len(train_candidate_pool.ordered_ids)
            == stage_c_train.get("candidate_count")
        ),
        "candidate_cell_counts_match_stage_c": all(
            len(train_candidate_pool.ids_by_cell.get(cell, ()))
            == stage_c_cell_counts.get(cell)
            for cell in all_cells()
        ),
        "selected_train_manifest_hash_matches_stage_c": (
            selected_train_manifest_sha256
            == stage_c_artifacts.get("train_manifest_sha256")
        ),
        "selected_dev_manifest_hash_matches_stage_c": (
            selected_dev_manifest_sha256
            == stage_c_artifacts.get("dev_manifest_sha256")
        ),
        "every_selected_train_id_is_candidate_eligible": (
            not selected_not_in_pool
        ),
        "selected_train_and_dev_do_not_overlap": not train_dev_overlap,
        "train_candidate_and_selected_dev_do_not_overlap": (
            not candidate_dev_overlap
        ),
        "remainder_count_conserves_candidates": (
            remainder_count + len(selected_train_ids)
            == len(train_candidate_pool.ordered_ids)
        ),
        "every_remainder_cell_meets_frozen_minimum": all(
            remainder_counts[cell] >= MINIMUM_REMAINDER_PER_CELL
            for cell in all_cells()
        ),
        "stage_g0_unlabeled_test_failure_bound": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": MUSIQUE_HOLDOUT_READINESS_SCHEMA_VERSION,
        "task": "musique_unused_official_train_holdout_readiness_audit",
        "scope": (
            "all occurrence-aware exact-20 official-train candidates not used "
            "in the frozen 1800-question graph selection split"
        ),
        "stage_c": {
            "path": stage_c_report_path,
            "sha256": stage_c_report_sha256,
            "gate_passed": True,
        },
        "stage_g0": {
            "path": stage_g0_report_path,
            "sha256": stage_g0_report_sha256,
            "gate_passed": False,
            "status": "stop_before_test_scoring",
            "failure_reason": (
                "official answerable test contains only id, question, and "
                "paragraphs; retrieval gold labels are unavailable"
            ),
        },
        "protocol": {
            "declared_after_official_test_was_found_unlabeled": True,
            "performed_before_any_remainder_embedding_or_retrieval_scoring": True,
            "holdout_used_for_hyperparameter_selection": False,
            "sampling_performed": False,
            "holdout_pool_rule": (
                "all remaining Stage-C-eligible official-train IDs in source order"
            ),
            "minimum_remainder_per_cell": MINIMUM_REMAINDER_PER_CELL,
            "primary_aggregate": "equal-weight macro across nine hop-collision cells",
            "secondary_aggregate": "remainder-distribution micro",
            "configurations": "exactly the hypotheses frozen before Stage G0",
            "configuration_selection_after_holdout_forbidden": True,
            "rerun_after_observing_holdout_forbidden": True,
            "claim_status": (
                "post-primary untouched within-dataset replication, not an "
                "official test-set result or preregistered external validation"
            ),
        },
        "source": {
            "path": train_source_path,
            "sha256": train_source_sha256,
            "record_count": train_candidate_pool.record_count,
        },
        "selected_splits": {
            "train_manifest_path": selected_train_manifest_path,
            "train_manifest_sha256": selected_train_manifest_sha256,
            "train_selected_count": len(selected_train_ids),
            "dev_manifest_path": selected_dev_manifest_path,
            "dev_manifest_sha256": selected_dev_manifest_sha256,
            "dev_selected_count": len(selected_dev_ids),
            "selected_train_ids_missing_from_candidate_pool_count": len(
                selected_not_in_pool
            ),
            "selected_train_dev_overlap_count": len(train_dev_overlap),
            "train_candidate_dev_overlap_count": len(candidate_dev_overlap),
        },
        "candidate_pool": {
            "candidate_count": len(train_candidate_pool.ordered_ids),
            "candidate_count_by_cell": {
                cell: len(train_candidate_pool.ids_by_cell.get(cell, ()))
                for cell in all_cells()
            },
            "exclusion_histogram": dict(
                sorted(train_candidate_pool.exclusion_histogram.items())
            ),
        },
        "untouched_remainder": {
            "question_count": remainder_count,
            "question_count_by_cell": remainder_counts,
            "minimum_cell_count": min(remainder_counts.values()),
            "maximum_cell_count": max(remainder_counts.values()),
            "manifest_order": "official source order after selected-train exclusion",
        },
        "frozen_holdout_hypotheses": dict(frozen_hypotheses),
        "readiness_gate": {
            "name": "musique_unused_train_remainder_readiness_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "inspect_counts_then_freeze_remainder_manifest"
                if passed
                else "stop_before_remainder_scoring"
            ),
            "interpretation": (
                "Passing verifies that the labeled remainder is disjoint from "
                "all selected train/dev records, reproduces the Stage C pool, "
                "and has at least 100 untouched records in every frozen cell. "
                "It authorizes manifest materialization only."
            ),
        },
        "content_contract": (
            "aggregate counts, configurations, paths, and hashes only; no IDs, "
            "questions, answers, titles, paragraph text, decomposition text, or "
            "supporting labels"
        ),
    }
