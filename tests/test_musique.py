from __future__ import annotations

import json

from topocf_rag.musique import (
    assess_musique_record,
    audit_musique_file,
    build_musique_schema_report,
)


def _record(qid: str = "2hop__1_2") -> dict:
    return {
        "id": qid,
        "question": "Composed question?",
        "question_decomposition": [
            {
                "id": 1,
                "question": "First?",
                "answer": "A",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "Second using #1?",
                "answer": "B",
                "paragraph_support_idx": 2,
            },
        ],
        "answerable": True,
        "answer": "B",
        "answer_aliases": [],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Alpha",
                "paragraph_text": "Alpha mentions Beta.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Distractor",
                "paragraph_text": "Unrelated text.",
                "is_supporting": False,
            },
            {
                "idx": 2,
                "title": "Beta",
                "paragraph_text": "The final evidence.",
                "is_supporting": True,
            },
        ],
    }


def test_structurally_valid_record_is_eligible() -> None:
    assessment = assess_musique_record(_record())
    assert assessment.eligible
    assert assessment.hop_count == 2
    assert assessment.paragraph_count == 3
    assert assessment.supporting_paragraph_count == 2


def test_support_mismatch_and_normalized_duplicate_are_rejected() -> None:
    record = _record()
    record["paragraphs"][1]["title"] = "  ALPHA  "
    record["paragraphs"][2]["is_supporting"] = False
    assessment = assess_musique_record(record)
    assert not assessment.eligible
    assert "duplicate_normalized_title" in assessment.reasons
    assert "support_flag_decomposition_mismatch" in assessment.reasons


def test_bad_paragraph_order_and_duplicate_step_support_are_rejected() -> None:
    record = _record()
    record["paragraphs"][1]["idx"] = 2
    record["question_decomposition"][1]["paragraph_support_idx"] = 0
    assessment = assess_musique_record(record)
    assert "duplicate_paragraph_index" in assessment.reasons
    assert "paragraph_index_order_mismatch" in assessment.reasons
    assert "duplicate_decomposition_support_index" in assessment.reasons


def test_streaming_audit_is_aggregate_and_content_free(tmp_path) -> None:
    path = tmp_path / "dev.jsonl"
    records = [_record("private-one"), _record("private-two")]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    audit = audit_musique_file(path)
    assert audit["record_count"] == 2
    assert audit["eligible_count"] == 2
    assert audit["histograms"]["hop_count"] == {"2": 2}
    assert "private-one" not in str(audit)
    assert "Composed question" not in str(audit)


def test_schema_gate_binds_hashes_without_authorizing_scoring() -> None:
    train = {
        "sha256": (
            "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
        ),
        "record_count": 10,
        "json_error_count": 0,
        "duplicate_id_count": 0,
        "eligible_count": 9,
    }
    dev = {
        "sha256": (
            "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
        ),
        "record_count": 5,
        "json_error_count": 0,
        "duplicate_id_count": 0,
        "eligible_count": 5,
    }
    report = build_musique_schema_report(train, dev)
    assert report["schema_gate"]["passed"]
    assert report["protocol"]["sampling_or_scoring_performed"] is False
    assert report["schema_gate"]["status"] == (
        "inspect_histograms_then_freeze_sampling_protocol"
    )
