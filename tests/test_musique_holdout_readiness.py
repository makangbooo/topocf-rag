from __future__ import annotations

import json

import pytest

from topocf_rag.musique_holdout_readiness import (
    EXPECTED_STAGE_C_REPORT_SHA256,
    EXPECTED_STAGE_G0_REPORT_SHA256,
    MuSiQueHoldoutReadinessError,
    build_musique_holdout_readiness_report,
)
from topocf_rag.musique_splits import CandidatePool, all_cells


def _pool(per_cell: int = 101) -> CandidatePool:
    by_cell = {
        cell: tuple(f"{cell}-private-{index}" for index in range(per_cell))
        for cell in all_cells()
    }
    ordered = tuple(qid for cell in all_cells() for qid in by_cell[cell])
    return CandidatePool(
        record_count=len(ordered),
        ordered_ids=ordered,
        ids_by_cell=by_cell,
        exclusion_histogram={},
    )


def _train_manifest(pool: CandidatePool) -> dict:
    ids = [pool.ids_by_cell[cell][0] for cell in all_cells()]
    return {
        "dataset": "musique_ans_v1.0",
        "split": "train",
        "selected_count": len(ids),
        "ids": ids,
    }


def _dev_manifest(ids: list[str] | None = None) -> dict:
    values = ids or ["dev-private"]
    return {
        "dataset": "musique_ans_v1.0",
        "split": "dev",
        "selected_count": len(values),
        "ids": values,
    }


def _stage_c(pool: CandidatePool) -> dict:
    return {
        "task": "musique_hop_collision_stratified_split_materialization",
        "gate": {"passed": True},
        "artifacts": {
            "train_manifest_sha256": "a" * 64,
            "dev_manifest_sha256": "b" * 64,
        },
        "train": {
            "source_sha256": "c" * 64,
            "source_record_count": pool.record_count,
            "candidate_count": len(pool.ordered_ids),
            "candidate_count_by_cell": {
                cell: len(pool.ids_by_cell[cell]) for cell in all_cells()
            },
        },
    }


def _stage_g0() -> dict:
    return {
        "task": "musique_answerable_test_final_holdout_readiness_audit",
        "readiness_gate": {
            "passed": False,
            "status": "stop_before_test_scoring",
        },
        "source": {
            "field_contract": {
                "keysets": [
                    {
                        "count": 2459,
                        "keys": ["id", "paragraphs", "question"],
                    }
                ]
            }
        },
        "frozen_test_hypotheses": {
            "primary_method": {"name": "frozen-private-config"}
        },
    }


def _build(
    *,
    pool: CandidatePool | None = None,
    dev_manifest: dict | None = None,
) -> dict:
    candidate_pool = pool or _pool()
    return build_musique_holdout_readiness_report(
        train_candidate_pool=candidate_pool,
        train_source_path="/private/train.jsonl",
        train_source_sha256="c" * 64,
        stage_c_report=_stage_c(candidate_pool),
        stage_c_report_path="/private/stage-c.json",
        stage_c_report_sha256=EXPECTED_STAGE_C_REPORT_SHA256,
        stage_g0_report=_stage_g0(),
        stage_g0_report_path="/private/stage-g0.json",
        stage_g0_report_sha256=EXPECTED_STAGE_G0_REPORT_SHA256,
        selected_train_manifest=_train_manifest(candidate_pool),
        selected_train_manifest_path="/private/train-ids.json",
        selected_train_manifest_sha256="a" * 64,
        selected_dev_manifest=dev_manifest or _dev_manifest(),
        selected_dev_manifest_path="/private/dev-ids.json",
        selected_dev_manifest_sha256="b" * 64,
    )


def test_unused_remainder_passes_without_scoring_or_content_leakage() -> None:
    report = _build()
    assert report["readiness_gate"]["passed"]
    assert report["untouched_remainder"]["minimum_cell_count"] == 100
    assert report["untouched_remainder"]["question_count"] == 900
    assert report["protocol"][
        "performed_before_any_remainder_embedding_or_retrieval_scoring"
    ]
    encoded = json.dumps(report, sort_keys=True)
    assert "unique_titles-private" not in encoded
    assert "dev-private" not in encoded


def test_unused_remainder_gate_rejects_small_cell() -> None:
    pool = _pool(per_cell=100)
    report = _build(pool=pool)
    assert not report["readiness_gate"]["passed"]
    assert report["untouched_remainder"]["minimum_cell_count"] == 99
    assert not report["readiness_gate"]["checks"][
        "every_remainder_cell_meets_frozen_minimum"
    ]


def test_unused_remainder_gate_detects_dev_overlap() -> None:
    pool = _pool()
    overlap_id = pool.ids_by_cell[all_cells()[0]][-1]
    report = _build(pool=pool, dev_manifest=_dev_manifest([overlap_id]))
    assert not report["readiness_gate"]["passed"]
    assert report["selected_splits"]["train_candidate_dev_overlap_count"] == 1


def test_unused_remainder_rejects_changed_g0_hash() -> None:
    pool = _pool()
    with pytest.raises(MuSiQueHoldoutReadinessError, match="G0 report SHA256"):
        build_musique_holdout_readiness_report(
            train_candidate_pool=pool,
            train_source_path="/private/train.jsonl",
            train_source_sha256="c" * 64,
            stage_c_report=_stage_c(pool),
            stage_c_report_path="/private/stage-c.json",
            stage_c_report_sha256=EXPECTED_STAGE_C_REPORT_SHA256,
            stage_g0_report=_stage_g0(),
            stage_g0_report_path="/private/stage-g0.json",
            stage_g0_report_sha256="0" * 64,
            selected_train_manifest=_train_manifest(pool),
            selected_train_manifest_path="/private/train-ids.json",
            selected_train_manifest_sha256="a" * 64,
            selected_dev_manifest=_dev_manifest(),
            selected_dev_manifest_path="/private/dev-ids.json",
            selected_dev_manifest_sha256="b" * 64,
        )
