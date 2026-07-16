from __future__ import annotations

import json

import pytest

from topocf_rag.musique_splits import CandidatePool, all_cells
from topocf_rag.musique_test_readiness import (
    EXPECTED_MUSIQUE_ANSWERABLE_TEST_SHA256,
    EXPECTED_STAGE_F_REPORT_SHA256,
    MuSiQueTestReadinessError,
    build_musique_test_readiness_report,
)


def _candidate_pool(*, overlap_id: str | None = None) -> CandidatePool:
    ids_by_cell = {
        cell: (overlap_id if overlap_id and index == 0 else f"test-{index}",)
        for index, cell in enumerate(all_cells())
    }
    ordered = tuple(values[0] for values in ids_by_cell.values())
    return CandidatePool(
        record_count=9,
        ordered_ids=ordered,
        ids_by_cell=ids_by_cell,
        exclusion_histogram={},
    )


def _schema() -> dict:
    return {
        "path": "/private/musique_ans_v1.0_test.jsonl",
        "size_bytes": 123,
        "sha256": EXPECTED_MUSIQUE_ANSWERABLE_TEST_SHA256,
        "record_count": 9,
        "json_error_count": 0,
        "duplicate_id_count": 0,
        "eligible_count": 9,
        "histograms": {"paragraph_count": {"20": 9}},
        "exclusions": {},
        "field_contract": {"keysets": []},
    }


def _duplicates() -> dict:
    return {
        "sha256": EXPECTED_MUSIQUE_ANSWERABLE_TEST_SHA256,
        "record_count": 9,
        "record_status_histogram": {"unique_titles": 9},
        "record_status_by_hop": {},
        "duplicate_group_aggregates": {},
        "policy_eligible_count": {
            "occurrence_fanout_label_unambiguous__exactly_20_paragraphs": 9
        },
        "policy_eligible_hop_histogram": {},
    }


def _stage_f() -> dict:
    return {
        "task": "musique_graph_retrieval_post_primary_robustness_validation",
        "gate": {"passed": True},
        "provenance": {
            "train": {"ids_sha256": "a" * 64, "question_count": 1},
            "dev": {"ids_sha256": "b" * 64, "question_count": 1},
        },
        "stage_e_reproduction": {
            "selected_baseline": "dense",
            "selected_graph_config": {
                "family": "one_hop_max",
                "seed": "dense",
                "direction": "undirected",
                "graph_weight": 0.75,
                "restart_probability": None,
            },
        },
    }


def _manifest(split: str, ids: list[str]) -> dict:
    return {
        "dataset": "musique_ans_v1.0",
        "split": split,
        "selected_count": len(ids),
        "ids": ids,
    }


def _build(*, pool: CandidatePool | None = None) -> dict:
    return build_musique_test_readiness_report(
        schema_audit=_schema(),
        duplicate_audit=_duplicates(),
        candidate_pool=pool or _candidate_pool(),
        stage_f_report=_stage_f(),
        stage_f_report_sha256=EXPECTED_STAGE_F_REPORT_SHA256,
        train_manifest=_manifest("train", ["train-private"]),
        train_manifest_path="/private/train-ids.json",
        train_manifest_sha256="a" * 64,
        dev_manifest=_manifest("dev", ["dev-private"]),
        dev_manifest_path="/private/dev-ids.json",
        dev_manifest_sha256="b" * 64,
    )


def test_readiness_gate_freezes_hypotheses_without_test_scoring() -> None:
    report = _build()
    assert report["readiness_gate"]["passed"]
    assert report["protocol"]["test_scoring_performed"] is False
    assert report["protocol"]["configuration_selection_after_test_forbidden"]
    hypotheses = report["frozen_test_hypotheses"]
    assert hypotheses["primary_method"]["name"] == (
        "one_hop_max__dense__undirected__w0.75"
    )
    assert hypotheses["direction_mechanism"]["required_mean_complete_delta"] == 0.02
    encoded = json.dumps(report, sort_keys=True)
    assert "test-0" not in encoded
    assert "train-private" not in encoded
    assert "dev-private" not in encoded


def test_readiness_gate_detects_test_train_overlap() -> None:
    report = _build(pool=_candidate_pool(overlap_id="train-private"))
    assert not report["readiness_gate"]["passed"]
    assert report["split_separation"]["test_train_overlap_count"] == 1
    assert not report["readiness_gate"]["checks"][
        "test_has_no_train_manifest_overlap"
    ]


def test_readiness_rejects_changed_stage_f_hash() -> None:
    with pytest.raises(MuSiQueTestReadinessError, match="SHA256 changed"):
        build_musique_test_readiness_report(
            schema_audit=_schema(),
            duplicate_audit=_duplicates(),
            candidate_pool=_candidate_pool(),
            stage_f_report=_stage_f(),
            stage_f_report_sha256="0" * 64,
            train_manifest=_manifest("train", ["train-private"]),
            train_manifest_path="/private/train-ids.json",
            train_manifest_sha256="a" * 64,
            dev_manifest=_manifest("dev", ["dev-private"]),
            dev_manifest_path="/private/dev-ids.json",
            dev_manifest_sha256="b" * 64,
        )
