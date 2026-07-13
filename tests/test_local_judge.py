import json

import pytest

from topocf_rag.local_judge import (
    ALLOWED_LABELS,
    LocalJudgeInvariantError,
    build_judge_messages,
    label_histogram,
    parse_judgment_text,
    prompt_template_sha256,
    stable_json_sha256,
    validate_private_audit_record,
)


def _private_record(qid: str = "q-private") -> dict[str, object]:
    return {
        "schema_version": 2,
        "privacy": "private_text_bearing_artifact_do_not_commit",
        "question_id": qid,
        "question": "Which answer follows from the two-hop evidence?",
        "answer": "Reference Answer",
        "primary_top_k": 5,
        "proxy": {
            "gold_support_document_count": 1,
            "retained_supporting_fact_count": 1,
            "partial_gold_evidence_proxy": True,
            "answer_string_check_eligible": True,
            "answer_string_present": False,
        },
        "retrieved_documents": [
            {
                "context_index": 0,
                "retrieval_rank": 1,
                "retrieval_score": 0.9,
                "title": "Retrieved Title",
                "sentences": ["A true local fact."],
            }
        ],
        "gold_supporting_facts_for_adjudication": [
            {
                "context_index": 0,
                "retrieved": True,
                "title": "Retrieved Title",
                "sentence_index": 0,
                "sentence": "A true local fact.",
            },
            {
                "context_index": 1,
                "retrieved": False,
                "title": "Missing Title",
                "sentence_index": 0,
                "sentence": "The missing bridge fact.",
            },
        ],
        "human_annotation": {
            "label": None,
            "allowed_labels": list(ALLOWED_LABELS),
            "retrieved_local_facts_true": None,
            "globally_sufficient": None,
            "alternative_proof_present": None,
            "notes": "",
        },
    }


def _valid_judgment(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": "locally_true_globally_incomplete",
        "retrieved_local_facts_true": True,
        "globally_sufficient": False,
        "alternative_proof_present": False,
        "confidence": "high",
        "rationale": "One true local step is present but the bridge is missing.",
        "missing_requirement": "The relation supplied by the missing document.",
    }
    payload.update(updates)
    return payload


def test_private_record_validation_and_prompt_are_deterministic() -> None:
    record = _private_record()
    assert validate_private_audit_record(record) is record
    messages_a = build_judge_messages(record)
    messages_b = build_judge_messages(record)

    assert messages_a == messages_b
    assert [message["role"] for message in messages_a] == ["system", "user"]
    assert "Retrieved Title" in messages_a[1]["content"]
    assert "Missing Title" in messages_a[1]["content"]
    assert "facts marked retrieved=false are NOT available" in messages_a[0][
        "content"
    ]
    assert prompt_template_sha256() == prompt_template_sha256()
    assert len(prompt_template_sha256()) == 64


def test_input_hash_changes_with_private_text() -> None:
    first = _private_record()
    second = _private_record()
    second["question"] = "A different private question"
    assert stable_json_sha256(first) != stable_json_sha256(second)


def test_strict_judgment_parser_accepts_consistent_object() -> None:
    payload = _valid_judgment()
    judgment, errors = parse_judgment_text(json.dumps(payload))
    assert errors == ()
    assert judgment is not None
    assert judgment["label"] == "locally_true_globally_incomplete"


def test_strict_judgment_parser_rejects_markdown_and_extra_text() -> None:
    encoded = json.dumps(_valid_judgment())
    for output in (f"```json\n{encoded}\n```", f"Result: {encoded}"):
        judgment, errors = parse_judgment_text(output)
        assert judgment is None
        assert errors == ("model output is not strict JSON",)


@pytest.mark.parametrize(
    "updates",
    [
        {"retrieved_local_facts_true": False},
        {"globally_sufficient": True},
        {"alternative_proof_present": True},
    ],
)
def test_target_label_requires_consistent_boolean_fields(
    updates: dict[str, object],
) -> None:
    judgment, errors = parse_judgment_text(json.dumps(_valid_judgment(**updates)))
    assert judgment is None
    assert any("inconsistent" in error for error in errors)


def test_alternative_proof_label_requires_sufficient_and_alternative() -> None:
    payload = _valid_judgment(
        label="complete_via_alternative_proof",
        globally_sufficient=True,
        alternative_proof_present=True,
        missing_requirement="",
    )
    judgment, errors = parse_judgment_text(json.dumps(payload))
    assert errors == ()
    assert judgment is not None

    payload["alternative_proof_present"] = False
    judgment, errors = parse_judgment_text(json.dumps(payload))
    assert judgment is None
    assert any("inconsistent" in error for error in errors)


def test_parser_rejects_missing_extra_and_oversized_fields() -> None:
    payload = _valid_judgment()
    del payload["confidence"]
    payload["unexpected"] = "value"
    payload["rationale"] = "x" * 501
    judgment, errors = parse_judgment_text(json.dumps(payload))
    assert judgment is None
    assert any("missing keys" in error for error in errors)
    assert any("unexpected keys" in error for error in errors)
    assert any("at most 500" in error for error in errors)


def test_private_record_rejects_wrong_privacy_marker_and_empty_evidence() -> None:
    record = _private_record()
    record["privacy"] = "public"
    with pytest.raises(LocalJudgeInvariantError, match="privacy marker"):
        validate_private_audit_record(record)

    record = _private_record()
    record["retrieved_documents"] = []
    with pytest.raises(LocalJudgeInvariantError, match="must not be empty"):
        validate_private_audit_record(record)


def test_label_histogram_ignores_invalid_records() -> None:
    records = [
        {"judgment": _valid_judgment()},
        {
            "judgment": _valid_judgment(
                label="complete_via_alternative_proof",
                globally_sufficient=True,
                alternative_proof_present=True,
            )
        },
        {"judgment": None},
    ]
    histogram = label_histogram(records)
    assert histogram["locally_true_globally_incomplete"] == 1
    assert histogram["complete_via_alternative_proof"] == 1
    assert sum(histogram.values()) == 2
