from __future__ import annotations

import json

import pytest

from topocf_rag.twowiki import DATASET_NAME, QUESTION_TYPES
from topocf_rag.twowiki_test_readiness import (
    EXPECTED_MUSIQUE_STAGE_G1_REPORT_SHA256,
    EXPECTED_TWOWIKI_DEV_MANIFEST_SHA256,
    EXPECTED_TWOWIKI_TEST_SHA256,
    EXPECTED_TWOWIKI_TRAIN_MANIFEST_SHA256,
    TwoWikiTestReadinessError,
    audit_twowiki_test_records,
    build_twowiki_test_readiness_report,
)


def _record(qid: str, question_type: str) -> dict[str, object]:
    context = [
        [f"Private title {index}", [f"Private sentence {index}."]]
        for index in range(10)
    ]
    gold_count = 4 if question_type == "bridge_comparison" else 2
    return {
        "_id": qid,
        "question": "Private question?",
        "answer": "Private answer",
        "type": question_type,
        "context": context,
        "supporting_facts": [
            [f"Private title {index}", 0] for index in range(gold_count)
        ],
    }


def _records() -> list[dict[str, object]]:
    return [
        _record(f"private-{question_type}-{index}", question_type)
        for question_type in QUESTION_TYPES
        for index in range(100)
    ]


def _manifest(split: str, count: int) -> dict[str, object]:
    ids = [f"selected-{split}-{index}" for index in range(count)]
    return {
        "dataset": DATASET_NAME,
        "official_split": split,
        "selection": {"sample_size": count},
        "ids": ids,
    }


def _musique_g1() -> dict[str, object]:
    return {
        "task": "musique_unused_official_train_holdout_readiness_audit",
        "readiness_gate": {
            "passed": False,
            "status": "stop_before_remainder_scoring",
        },
        "frozen_holdout_hypotheses": {
            "primary_method": {"name": "private-frozen-method"}
        },
    }


def _build(
    records: list[dict[str, object]] | None = None,
    *,
    train_manifest: dict[str, object] | None = None,
    train_manifest_sha256: str = EXPECTED_TWOWIKI_TRAIN_MANIFEST_SHA256,
    musique_g1_sha256: str = EXPECTED_MUSIQUE_STAGE_G1_REPORT_SHA256,
) -> dict:
    audit, test_ids, eligible = audit_twowiki_test_records(
        _records() if records is None else records
    )
    return build_twowiki_test_readiness_report(
        source_path="/private/test.json",
        source_sha256=EXPECTED_TWOWIKI_TEST_SHA256,
        source_size_bytes=123,
        audit=audit,
        test_ids=test_ids,
        eligible_ids_by_type=eligible,
        selected_train_manifest=train_manifest or _manifest("train", 10),
        selected_train_manifest_path="/private/train-ids.json",
        selected_train_manifest_sha256=train_manifest_sha256,
        selected_dev_manifest=_manifest("dev", 8),
        selected_dev_manifest_path="/private/dev-ids.json",
        selected_dev_manifest_sha256=(
            EXPECTED_TWOWIKI_DEV_MANIFEST_SHA256
        ),
        musique_g1_report=_musique_g1(),
        musique_g1_report_path="/private/musique-g1.json",
        musique_g1_report_sha256=musique_g1_sha256,
    )


def test_readiness_passes_without_scoring_or_content_leakage() -> None:
    report = _build()
    assert report["readiness_gate"]["passed"]
    assert report["eligible_test_pool"]["eligible_count"] == 400
    assert report["protocol"][
        "performed_before_any_2wiki_test_embedding_or_retrieval_scoring"
    ]
    encoded = json.dumps(report, sort_keys=True)
    assert "private-compositional-0" not in encoded
    assert "Private question" not in encoded
    assert "Private title" not in encoded


def test_readiness_rejects_missing_gold_schema() -> None:
    records = _records()
    records[0].pop("supporting_facts")
    report = _build(records)
    assert not report["readiness_gate"]["passed"]
    assert report["source"]["schema_invalid_count"] == 1
    assert not report["readiness_gate"]["checks"][
        "all_records_have_validatable_gold_schema"
    ]


def test_readiness_rejects_selected_split_overlap() -> None:
    manifest = _manifest("train", 10)
    manifest["ids"][0] = "private-comparison-0"  # type: ignore[index]
    report = _build(train_manifest=manifest)
    assert not report["readiness_gate"]["passed"]
    assert report["split_separation"]["test_train_overlap_count"] == 1


def test_readiness_rejects_changed_musique_g1_hash() -> None:
    with pytest.raises(TwoWikiTestReadinessError, match="Stage G1 SHA256"):
        _build(musique_g1_sha256="0" * 64)


def test_readiness_rejects_changed_selected_manifest_hash() -> None:
    report = _build(train_manifest_sha256="0" * 64)
    assert not report["readiness_gate"]["passed"]
    assert not report["readiness_gate"]["checks"][
        "selected_train_manifest_hash_frozen"
    ]
