"""Readiness audit and frozen hypotheses for untouched MuSiQue test data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

from .graph_retrieval import GraphMethodConfig
from .musique_splits import CandidatePool, all_cells


MUSIQUE_TEST_READINESS_SCHEMA_VERSION = 1
EXPECTED_MUSIQUE_ANSWERABLE_TEST_SHA256 = (
    "92ee3067957b8ebce885baa71156a7c0c29f3bc72cca04510aef506911dd768f"
)
EXPECTED_STAGE_F_REPORT_SHA256 = (
    "051a9ff57d26cca05963c301b28cf3748e9513bab16317c912780058e404901e"
)
EXPECTED_PRIMARY_CONFIG = GraphMethodConfig(
    family="one_hop_max",
    seed="dense",
    direction="undirected",
    graph_weight=0.75,
)


class MuSiQueTestReadinessError(ValueError):
    """Raised when the final-test audit cannot be bound safely."""


def _validate_manifest(
    manifest: Mapping[str, Any], *, expected_split: str
) -> tuple[str, ...]:
    if manifest.get("dataset") != "musique_ans_v1.0":
        raise MuSiQueTestReadinessError("manifest dataset changed")
    if manifest.get("split") != expected_split:
        raise MuSiQueTestReadinessError("manifest split changed")
    ids = manifest.get("ids")
    if not isinstance(ids, list) or any(
        not isinstance(qid, str) or not qid for qid in ids
    ):
        raise MuSiQueTestReadinessError("manifest IDs are invalid")
    if len(ids) != len(set(ids)):
        raise MuSiQueTestReadinessError("manifest IDs are not unique")
    if manifest.get("selected_count") != len(ids):
        raise MuSiQueTestReadinessError("manifest selected count changed")
    return tuple(ids)


def _stage_f_contract(
    stage_f_report: Mapping[str, Any], stage_f_report_sha256: str
) -> tuple[GraphMethodConfig, str, Mapping[str, Any], Mapping[str, Any]]:
    if stage_f_report_sha256 != EXPECTED_STAGE_F_REPORT_SHA256:
        raise MuSiQueTestReadinessError("Stage F report SHA256 changed")
    if stage_f_report.get("task") != (
        "musique_graph_retrieval_post_primary_robustness_validation"
    ):
        raise MuSiQueTestReadinessError("Stage F task changed")
    gate = stage_f_report.get("gate")
    reproduction = stage_f_report.get("stage_e_reproduction")
    provenance = stage_f_report.get("provenance")
    if not isinstance(gate, Mapping) or not gate.get("passed"):
        raise MuSiQueTestReadinessError("Stage F gate did not pass")
    if not isinstance(reproduction, Mapping):
        raise MuSiQueTestReadinessError("Stage F reproduction is missing")
    if not isinstance(provenance, Mapping):
        raise MuSiQueTestReadinessError("Stage F provenance is missing")
    train_provenance = provenance.get("train")
    dev_provenance = provenance.get("dev")
    if not isinstance(train_provenance, Mapping) or not isinstance(
        dev_provenance, Mapping
    ):
        raise MuSiQueTestReadinessError("Stage F split provenance is missing")
    config_payload = reproduction.get("selected_graph_config")
    baseline = reproduction.get("selected_baseline")
    if not isinstance(config_payload, Mapping):
        raise MuSiQueTestReadinessError("Stage F primary config is missing")
    config = GraphMethodConfig(**config_payload)
    if config != EXPECTED_PRIMARY_CONFIG:
        raise MuSiQueTestReadinessError("Stage F primary config changed")
    if baseline != "dense":
        raise MuSiQueTestReadinessError("Stage F primary baseline changed")
    return config, baseline, train_provenance, dev_provenance


def build_musique_test_readiness_report(
    *,
    schema_audit: Mapping[str, Any],
    duplicate_audit: Mapping[str, Any],
    candidate_pool: CandidatePool,
    stage_f_report: Mapping[str, Any],
    stage_f_report_sha256: str,
    train_manifest: Mapping[str, Any],
    train_manifest_path: str,
    train_manifest_sha256: str,
    dev_manifest: Mapping[str, Any],
    dev_manifest_path: str,
    dev_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind an aggregate-only test audit before any test scoring."""

    (
        primary_config,
        primary_baseline,
        train_provenance,
        dev_provenance,
    ) = _stage_f_contract(stage_f_report, stage_f_report_sha256)
    train_ids = _validate_manifest(train_manifest, expected_split="train")
    dev_ids = _validate_manifest(dev_manifest, expected_split="dev")

    schema_sha = schema_audit.get("sha256")
    duplicate_sha = duplicate_audit.get("sha256")
    if candidate_pool.record_count != schema_audit.get("record_count"):
        raise MuSiQueTestReadinessError(
            "candidate and schema record counts differ"
        )
    if candidate_pool.record_count != duplicate_audit.get("record_count"):
        raise MuSiQueTestReadinessError(
            "candidate and duplicate-audit record counts differ"
        )
    if schema_sha != duplicate_sha:
        raise MuSiQueTestReadinessError(
            "schema and duplicate audits use different sources"
        )

    policy_counts = duplicate_audit.get("policy_eligible_count")
    if not isinstance(policy_counts, Mapping):
        raise MuSiQueTestReadinessError(
            "duplicate-audit policy counts are missing"
        )
    expected_candidate_count = policy_counts.get(
        "occurrence_fanout_label_unambiguous__exactly_20_paragraphs"
    )
    cell_counts = {
        cell: len(candidate_pool.ids_by_cell.get(cell, ()))
        for cell in all_cells()
    }
    candidate_ids = set(candidate_pool.ordered_ids)
    train_overlap = candidate_ids.intersection(train_ids)
    dev_overlap = candidate_ids.intersection(dev_ids)
    train_dev_overlap = set(train_ids).intersection(dev_ids)

    outgoing = replace(primary_config, direction="outgoing")
    incoming = replace(primary_config, direction="incoming")
    checks = {
        "official_test_source_hash": (
            schema_sha == EXPECTED_MUSIQUE_ANSWERABLE_TEST_SHA256
        ),
        "json_valid": schema_audit.get("json_error_count") == 0,
        "ids_unique": schema_audit.get("duplicate_id_count") == 0,
        "record_count_positive": bool(schema_audit.get("record_count", 0)),
        "candidate_count_matches_duplicate_policy": (
            len(candidate_pool.ordered_ids) == expected_candidate_count
        ),
        "candidate_ids_unique": (
            len(candidate_pool.ordered_ids) == len(candidate_ids)
        ),
        "all_nine_cells_nonempty": all(
            cell_counts[cell] > 0 for cell in all_cells()
        ),
        "test_has_no_train_manifest_overlap": not train_overlap,
        "test_has_no_dev_manifest_overlap": not dev_overlap,
        "train_and_dev_manifests_do_not_overlap": not train_dev_overlap,
        "train_manifest_hash_matches_stage_f": (
            train_manifest_sha256 == train_provenance.get("ids_sha256")
        ),
        "dev_manifest_hash_matches_stage_f": (
            dev_manifest_sha256 == dev_provenance.get("ids_sha256")
        ),
        "train_manifest_count_matches_stage_f": (
            len(train_ids) == train_provenance.get("question_count")
        ),
        "dev_manifest_count_matches_stage_f": (
            len(dev_ids) == dev_provenance.get("question_count")
        ),
        "stage_f_gate_bound_and_passed": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": MUSIQUE_TEST_READINESS_SCHEMA_VERSION,
        "task": "musique_answerable_test_final_holdout_readiness_audit",
        "scope": "official untouched MuSiQue v1.0 answerable test paragraph pools",
        "stage_f": {
            "sha256": stage_f_report_sha256,
            "task": stage_f_report.get("task"),
            "gate_passed": True,
        },
        "protocol": {
            "declared_after_stage_f_dev_validation": True,
            "performed_before_any_test_retrieval_scoring": True,
            "test_used_for_hyperparameter_selection": False,
            "test_scoring_performed": False,
            "sampling_performed": False,
            "test_pool_rule": (
                "all occurrence-structurally-eligible exact-20 records with "
                "no exact-text mixed-support-label title collision"
            ),
            "primary_aggregate": "equal-weight macro across nine hop-collision cells",
            "secondary_aggregate": "official-distribution micro",
            "top_k_rule": "gold document count plus frozen extra budgets 0, 1, 3",
            "configuration_selection_after_test_forbidden": True,
            "rerun_after_observing_test_forbidden": True,
        },
        "source": {
            "path": schema_audit.get("path"),
            "size_bytes": schema_audit.get("size_bytes"),
            "sha256": schema_sha,
            "expected_sha256": EXPECTED_MUSIQUE_ANSWERABLE_TEST_SHA256,
            "record_count": schema_audit.get("record_count"),
            "json_error_count": schema_audit.get("json_error_count"),
            "duplicate_id_count": schema_audit.get("duplicate_id_count"),
            "eligible_count_under_strict_unique_title": schema_audit.get(
                "eligible_count"
            ),
            "histograms": schema_audit.get("histograms"),
            "exclusions": schema_audit.get("exclusions"),
            "field_contract": schema_audit.get("field_contract"),
        },
        "duplicate_title_audit": {
            "record_status_histogram": duplicate_audit.get(
                "record_status_histogram"
            ),
            "record_status_by_hop": duplicate_audit.get(
                "record_status_by_hop"
            ),
            "duplicate_group_aggregates": duplicate_audit.get(
                "duplicate_group_aggregates"
            ),
            "policy_eligible_count": policy_counts,
            "policy_eligible_hop_histogram": duplicate_audit.get(
                "policy_eligible_hop_histogram"
            ),
        },
        "candidate_pool": {
            "record_count": candidate_pool.record_count,
            "candidate_count": len(candidate_pool.ordered_ids),
            "candidate_count_by_cell": cell_counts,
            "exclusion_histogram": dict(
                sorted(candidate_pool.exclusion_histogram.items())
            ),
        },
        "split_separation": {
            "train_manifest_path": train_manifest_path,
            "train_manifest_sha256": train_manifest_sha256,
            "train_manifest_count": len(train_ids),
            "test_train_overlap_count": len(train_overlap),
            "dev_manifest_path": dev_manifest_path,
            "dev_manifest_sha256": dev_manifest_sha256,
            "dev_manifest_count": len(dev_ids),
            "test_dev_overlap_count": len(dev_overlap),
            "train_dev_overlap_count": len(train_dev_overlap),
        },
        "frozen_test_hypotheses": {
            "primary_method": {
                "name": primary_config.name,
                "config": asdict(primary_config),
                "control": primary_baseline,
                "success": (
                    "nine-cell paired-bootstrap 95% lower bound for the mean "
                    "complete-evidence-rate delta is greater than zero"
                ),
            },
            "same_config_degree_control": {
                "config": asdict(primary_config),
                "success": (
                    "primary graph has a positive nine-cell paired-bootstrap "
                    "95% lower bound against the same-config degree prior"
                ),
            },
            "direction_mechanism": {
                "outgoing_config": asdict(outgoing),
                "incoming_config": asdict(incoming),
                "primary_config": asdict(primary_config),
                "confirmatory_contrast": "outgoing minus incoming",
                "required_mean_complete_delta": 0.02,
                "required_paired_bootstrap_95_low": 0.0,
                "outgoing_minus_undirected": "descriptive only; never a selection rule",
            },
            "heterogeneity": {
                "confirmatory": False,
                "report_by_hop_collision_and_each_cell": True,
                "uniform_gain_required": False,
            },
        },
        "readiness_gate": {
            "name": "musique_untouched_test_readiness_gate_v1",
            "checks": checks,
            "passed": passed,
            "status": (
                "inspect_counts_then_freeze_test_manifest"
                if passed
                else "stop_before_test_scoring"
            ),
            "interpretation": (
                "Passing verifies source identity, structural auditability, "
                "nine-cell coverage, and split separation. It authorizes only "
                "review and manifest freezing, not test scoring."
            ),
        },
        "content_contract": (
            "aggregate counts, configurations, paths, and hashes only; no IDs, "
            "questions, answers, titles, paragraph text, decomposition text, or "
            "supporting labels"
        ),
    }
