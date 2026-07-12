import json

import pytest

from topocf_rag.natural_error_audit import (
    NaturalErrorAuditInvariantError,
    audit_question,
    build_private_sample_record,
    build_public_audit_report,
    wilson_interval,
)


def _example(qid: str) -> dict[str, object]:
    return {
        "_id": qid,
        "type": "bridge",
        "level": "hard",
        "question": "Which sentinel answer follows the bridge?",
        "answer": "Zeta Sentinel",
        "context": [
            [
                "Gold Alpha",
                [
                    "Gold Alpha mentions Gold Beta.",
                    "The local fact contains Zeta Sentinel.",
                ],
            ],
            ["Gold Beta", ["Gold Beta supplies the second local fact."]],
            ["Distractor X", ["Unrelated sentinel distractor X."]],
            ["Distractor Y", ["Unrelated sentinel distractor Y."]],
        ],
        "supporting_facts": [
            ["Gold Alpha", 1],
            ["Gold Beta", 0],
        ],
    }


def _audit(qid: str, scores: list[float]):
    return audit_question(
        _example(qid),
        scores,
        top_ks=[2, 3],
        primary_top_k=2,
    )


def test_question_audit_uses_stable_score_then_context_index_ranking() -> None:
    audit = _audit("q-tie", [0.8, 0.1, 0.8, 0.5])
    assert audit.ranked_indices == (0, 2, 3, 1)

    outcome = audit.outcome_for(2)
    assert outcome.retrieved_indices == (0, 2)
    assert outcome.gold_support_document_count == 1
    assert outcome.retained_supporting_fact_count == 1
    assert outcome.partial_gold_evidence_proxy
    assert outcome.failure
    assert outcome.answer_string_check_eligible
    assert outcome.answer_string_present is True


def test_public_report_aggregates_failures_and_keeps_text_out() -> None:
    audits = [
        _audit("q-partial", [0.9, 0.1, 0.8, 0.7]),
        _audit("q-none", [0.2, 0.1, 0.9, 0.8]),
        _audit("q-complete", [0.9, 0.8, 0.2, 0.1]),
    ]
    report, sample = build_public_audit_report(
        audits,
        audit_id="unit-audit",
        dataset="hotpotqa",
        official_split="dev_distractor",
        top_ks=[2, 3],
        primary_top_k=2,
        sample_size=100,
        sample_seed=20260713,
        screening_threshold=0.15,
        source_sha256="a" * 64,
        ids_sha256="b" * 64,
        retrieval_cache_sha256="c" * 64,
        config_sha256="d" * 64,
    )

    counts = report["by_top_k"]["2"]["counts"]
    rates = report["by_top_k"]["2"]["rates"]
    assert counts["eligible_question_count"] == 3
    assert counts["complete_gold_evidence_count"] == 1
    assert counts["retrieval_failure_count"] == 2
    assert counts["partial_gold_evidence_proxy_count"] == 1
    assert counts["no_gold_support_document_count"] == 1
    assert rates["partial_proxy_rate_among_retrieval_failures"] == 0.5
    assert report["screening_gate"]["passed"] is True
    assert report["screening_gate"]["formal_g0"] is False
    assert {audit.qid for audit in sample} == {"q-partial", "q-none"}

    answer_strata = report["by_top_k"]["2"][
        "answer_string_by_retrieval_status"
    ]
    assert answer_strata["retrieval_failure"] == {
        "eligible_count": 2,
        "present_count": 1,
        "absent_count": 1,
        "present_rate": 0.5,
    }
    assert answer_strata["partial_gold_evidence_proxy"]["present_rate"] == 1.0
    assert answer_strata["no_gold_support_document"]["present_rate"] == 0.0

    encoded = json.dumps(report, sort_keys=True)
    for forbidden in (
        "Which sentinel answer",
        "Zeta Sentinel",
        "Gold Alpha",
        "Gold Beta",
        "q-partial",
        "q-none",
        "q-complete",
    ):
        assert forbidden not in encoded


def test_failure_sample_is_hash_deterministic_and_failure_only() -> None:
    audits = [
        _audit(f"q-{index}", [0.9, 0.1, 0.8, 0.7])
        for index in range(10)
    ]
    kwargs = dict(
        audit_id="unit-audit",
        dataset="hotpotqa",
        official_split="dev_distractor",
        top_ks=[2, 3],
        primary_top_k=2,
        sample_size=4,
        sample_seed=17,
        screening_threshold=0.15,
        source_sha256="a" * 64,
        ids_sha256="b" * 64,
        retrieval_cache_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    report_a, sample_a = build_public_audit_report(audits, **kwargs)
    report_b, sample_b = build_public_audit_report(list(reversed(audits)), **kwargs)

    assert [audit.qid for audit in sample_a] == [audit.qid for audit in sample_b]
    assert report_a["private_sample"] == report_b["private_sample"]
    assert len(sample_a) == 4
    assert all(audit.outcome_for(2).failure for audit in sample_a)


def test_private_sample_contains_text_and_blank_human_annotation() -> None:
    example = _example("q-private")
    audit = _audit("q-private", [0.9, 0.1, 0.8, 0.7])
    record = build_private_sample_record(example, audit, primary_top_k=2)

    assert record["privacy"] == "private_text_bearing_artifact_do_not_commit"
    assert record["question"] == example["question"]
    assert record["answer"] == example["answer"]
    assert len(record["retrieved_documents"]) == 2
    assert len(record["gold_supporting_facts_for_adjudication"]) == 2
    assert record["human_annotation"]["label"] is None
    assert "locally_true_globally_incomplete" in record["human_annotation"][
        "allowed_labels"
    ]


def test_audit_rejects_score_count_and_top_k_contract_violations() -> None:
    with pytest.raises(
        NaturalErrorAuditInvariantError,
        match="retrieval score count",
    ):
        audit_question(
            _example("q-short"),
            [0.1],
            top_ks=[2, 3],
            primary_top_k=2,
        )
    with pytest.raises(NaturalErrorAuditInvariantError, match="unique"):
        audit_question(
            _example("q-duplicate-k"),
            [0.9, 0.8, 0.7, 0.6],
            top_ks=[2, 2],
            primary_top_k=2,
        )


def test_answer_occurrence_check_excludes_yes_no_answers() -> None:
    example = _example("q-yes")
    example["answer"] = "yes"
    audit = audit_question(
        example,
        [0.9, 0.1, 0.8, 0.7],
        top_ks=[2],
        primary_top_k=2,
    )
    outcome = audit.outcome_for(2)
    assert outcome.answer_string_check_eligible is False
    assert outcome.answer_string_present is None


def test_wilson_interval_handles_empty_and_known_half_rate() -> None:
    assert wilson_interval(0, 0) is None
    interval = wilson_interval(5, 10)
    assert interval is not None
    assert interval["low"] == pytest.approx(0.236593, abs=1e-6)
    assert interval["high"] == pytest.approx(0.763407, abs=1e-6)
