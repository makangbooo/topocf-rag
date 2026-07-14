from __future__ import annotations

from topocf_rag.musique_duplicates import (
    build_duplicate_audit_report,
    diagnose_duplicate_titles,
)


def _record() -> dict:
    return {
        "id": "2hop__1_2",
        "question": "Question?",
        "answerable": True,
        "answer": "Answer",
        "answer_aliases": [],
        "question_decomposition": [
            {
                "id": 1,
                "question": "First?",
                "answer": "A",
                "paragraph_support_idx": 0,
            },
            {
                "id": 2,
                "question": "Second?",
                "answer": "B",
                "paragraph_support_idx": 2,
            },
        ],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Alpha",
                "paragraph_text": "Same text.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": " alpha ",
                "paragraph_text": "Same text.",
                "is_supporting": False,
            },
            {
                "idx": 2,
                "title": "Beta",
                "paragraph_text": "Second support.",
                "is_supporting": True,
            },
        ],
    }


def test_exact_text_mixed_label_collision_is_identified() -> None:
    diagnostic = diagnose_duplicate_titles(_record())
    assert diagnostic.occurrence_structurally_eligible
    assert not diagnostic.strict_unique_title_eligible
    assert diagnostic.duplicate_group_count == 1
    assert diagnostic.exact_text_group_count == 1
    assert diagnostic.exact_text_mixed_support_label_group_count == 1
    assert diagnostic.status == "exact_text_mixed_support_label"


def test_mixed_text_support_collision_is_separate() -> None:
    record = _record()
    record["paragraphs"][1]["paragraph_text"] = "Different text."
    diagnostic = diagnose_duplicate_titles(record)
    assert diagnostic.mixed_text_group_count == 1
    assert diagnostic.exact_text_mixed_support_label_group_count == 0
    assert diagnostic.mixed_text_supporting_group_count == 1
    assert diagnostic.status == "mixed_text_support_title_collision"


def test_distractor_only_collision_remains_primary_pool_eligible() -> None:
    record = _record()
    record["paragraphs"][1]["title"] = "Gamma"
    record["paragraphs"].append(
        {
            "idx": 3,
            "title": " gamma ",
            "paragraph_text": "Other distractor.",
            "is_supporting": False,
        }
    )
    diagnostic = diagnose_duplicate_titles(record)
    assert diagnostic.status == "distractor_only_title_collision"
    assert diagnostic.exact_text_mixed_support_label_group_count == 0


def test_stage_b_report_binds_stage_a_without_authorizing_scoring() -> None:
    split = {
        "sha256": "a" * 64,
        "record_count": 10,
        "json_error_count": 0,
    }
    schema = {
        "task": "musique_answerable_v1_schema_and_eligibility_audit",
        "schema_gate": {"passed": True},
        "splits": {
            "train": {"sha256": "a" * 64, "record_count": 10},
            "dev": {"sha256": "a" * 64, "record_count": 10},
        },
    }
    report = build_duplicate_audit_report(
        split,
        split,
        schema_report=schema,
        schema_report_sha256="b" * 64,
    )
    assert report["consistency_gate"]["passed"]
    assert report["protocol"]["sampling_or_scoring_performed"] is False
    assert report["protocol"]["planned_primary_pool"] == (
        "occurrence_fanout_label_unambiguous__exactly_20_paragraphs"
    )
