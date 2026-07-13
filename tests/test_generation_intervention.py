import json

import pytest

from topocf_rag.generation_intervention import (
    CONDITIONS,
    GenerationInterventionInvariantError,
    answer_exact_match,
    answer_f1,
    build_generation_messages,
    build_public_generation_report,
    construct_evidence_conditions,
    generation_prompt_template_sha256,
    normalize_answer,
    parse_generation_text,
    select_confirmed_records,
)
from topocf_rag.human_audit import HumanAuditBundle


def _source(qid: str = "q-private") -> dict[str, object]:
    context = [
        ["Gold Alpha", ["Alpha links to Beta.", "Alpha local fact."]],
        ["Gold Beta", ["Beta gives the Sentinel Answer."]],
    ]
    context.extend(
        [f"Distractor {index}", [f"Distractor sentence {index}."]]
        for index in range(2, 10)
    )
    return {
        "_id": qid,
        "type": "bridge",
        "level": "hard",
        "question": "What answer follows from Alpha through Beta?",
        "answer": "Sentinel Answer",
        "context": context,
        "supporting_facts": [["Gold Alpha", 0], ["Gold Beta", 0]],
    }


def _private(source: dict[str, object]) -> dict[str, object]:
    retrieved_indices = [0, 2, 3, 4, 5]
    context = source["context"]
    return {
        "schema_version": 2,
        "privacy": "private_text_bearing_artifact_do_not_commit",
        "question_id": source["_id"],
        "question": source["question"],
        "answer": source["answer"],
        "primary_top_k": 5,
        "proxy": {
            "partial_gold_evidence_proxy": True,
            "answer_string_present": False,
        },
        "retrieved_documents": [
            {
                "context_index": context_index,
                "retrieval_rank": rank,
                "retrieval_score": 1.0 / rank,
                "title": context[context_index][0],
                "sentences": context[context_index][1],
            }
            for rank, context_index in enumerate(retrieved_indices, start=1)
        ],
        "gold_supporting_facts_for_adjudication": [
            {
                "context_index": 0,
                "retrieved": True,
                "title": "Gold Alpha",
                "sentence_index": 0,
                "sentence": "Alpha links to Beta.",
            },
            {
                "context_index": 1,
                "retrieved": False,
                "title": "Gold Beta",
                "sentence_index": 0,
                "sentence": "Beta gives the Sentinel Answer.",
            },
        ],
        "human_annotation": {"label": None},
    }


def _prelabel() -> dict[str, object]:
    return {
        "status": "valid",
        "judgment": {"label": "locally_true_globally_incomplete"},
    }


def _bundle(records: tuple[dict[str, object], ...]) -> HumanAuditBundle:
    return HumanAuditBundle(
        records=records,
        prelabels={str(record["question_id"]): _prelabel() for record in records},
        input_sha256="a" * 64,
        prelabels_sha256="b" * 64,
        run_fingerprint="run",
    )


def _events(qid: str, final_label: str) -> list[dict[str, object]]:
    return [
        {
            "event_type": "blind_decision",
            "question_id": qid,
            "label": "locally_true_globally_incomplete",
        },
        {
            "event_type": "final_decision",
            "question_id": qid,
            "label": final_label,
            "blind_label_at_reveal": "locally_true_globally_incomplete",
            "prelabel_label_at_reveal": "locally_true_globally_incomplete",
        },
    ]


def test_select_confirmed_records_uses_final_target_label_only() -> None:
    first = _private(_source("q1"))
    second = _private(_source("q2"))
    bundle = _bundle((first, second))
    events = _events("q1", "locally_true_globally_incomplete")
    events.extend(_events("q2", "ambiguous_or_annotation_issue"))
    selected = select_confirmed_records(bundle, events)
    assert [record["question_id"] for record in selected] == ["q1"]


def test_construct_conditions_preserves_budget_and_repairs_missing_gold() -> None:
    source = _source()
    private = _private(source)
    conditions = construct_evidence_conditions(private, source)

    assert tuple(conditions) == CONDITIONS
    partial_indices = [doc["context_index"] for doc in conditions["partial_at_5"]]
    repaired_indices = [doc["context_index"] for doc in conditions["repaired_at_5"]]
    oracle_indices = [doc["context_index"] for doc in conditions["oracle_gold_2"]]
    assert partial_indices == [0, 2, 3, 4, 5]
    assert repaired_indices == [0, 2, 3, 4, 1]
    assert oracle_indices == [0, 1]
    assert len(partial_indices) == len(repaired_indices) == 5
    assert conditions["repaired_at_5"][-1]["origin"] == "missing_gold_repair"
    assert [doc["slot"] for doc in conditions["partial_at_5"] if doc["is_gold"]] == [1]
    assert [doc["slot"] for doc in conditions["repaired_at_5"] if doc["is_gold"]] == [1, 5]


def test_construct_conditions_rejects_source_or_partial_contract_drift() -> None:
    source = _source()
    private = _private(source)
    private["question"] = "Changed question"
    with pytest.raises(GenerationInterventionInvariantError, match="question"):
        construct_evidence_conditions(private, source)

    private = _private(source)
    context = source["context"]
    private["retrieved_documents"][-1] = {
        "context_index": 1,
        "retrieval_rank": 5,
        "retrieval_score": 0.1,
        "title": context[1][0],
        "sentences": context[1][1],
    }
    with pytest.raises(GenerationInterventionInvariantError, match="exactly one"):
        construct_evidence_conditions(private, source)


def test_prompt_is_condition_blind_and_deterministic() -> None:
    conditions = construct_evidence_conditions(_private(_source()), _source())
    messages_a = build_generation_messages(
        "Private question", conditions["partial_at_5"]
    )
    messages_b = build_generation_messages(
        "Private question", conditions["partial_at_5"]
    )
    assert messages_a == messages_b
    assert "partial_at_5" not in json.dumps(messages_a)
    assert "Gold Alpha" in messages_a[1]["content"]
    assert "strictly evidence-grounded" in messages_a[0]["content"]
    assert len(generation_prompt_template_sha256()) == 64


def test_strict_response_parser_accepts_answer_and_abstention() -> None:
    answered = {
        "answer": "Sentinel Answer",
        "abstain": False,
        "supporting_document_numbers": [1, 2],
        "confidence": "high",
    }
    response, errors = parse_generation_text(
        json.dumps(answered), document_count=2
    )
    assert errors == ()
    assert response == answered

    abstained = {
        "answer": "",
        "abstain": True,
        "supporting_document_numbers": [1],
        "confidence": "low",
    }
    response, errors = parse_generation_text(
        json.dumps(abstained), document_count=5
    )
    assert errors == ()
    assert response == abstained


@pytest.mark.parametrize(
    "payload,error_fragment",
    [
        (
            {
                "answer": "Sentinel",
                "abstain": True,
                "supporting_document_numbers": [1],
                "confidence": "low",
            },
            "empty answer",
        ),
        (
            {
                "answer": "Sentinel",
                "abstain": False,
                "supporting_document_numbers": [3],
                "confidence": "high",
            },
            "invalid",
        ),
        (
            {
                "answer": "Sentinel",
                "abstain": False,
                "supporting_document_numbers": [],
                "confidence": "high",
            },
            "at least one",
        ),
    ],
)
def test_response_parser_rejects_inconsistent_payloads(
    payload: dict[str, object], error_fragment: str
) -> None:
    response, errors = parse_generation_text(json.dumps(payload), document_count=2)
    assert response is None
    assert any(error_fragment in error for error in errors)


def test_response_parser_rejects_markdown() -> None:
    response, errors = parse_generation_text(
        '```json\n{"answer":"x"}\n```', document_count=2
    )
    assert response is None
    assert errors == ("model output is not strict JSON",)


def test_hotpot_answer_metrics_follow_normalized_token_overlap() -> None:
    assert normalize_answer("The Sentinel-Answer!") == "sentinelanswer"
    assert answer_exact_match("The Sentinel Answer", "sentinel answer") == 1.0
    assert answer_f1("Sentinel Answer Extra", "Sentinel Answer") == 0.8
    assert answer_f1("yes", "no") == 0.0


def test_public_report_scores_paired_repair_without_private_content() -> None:
    private = _private(_source())
    qid = str(private["question_id"])
    responses = {
        "partial_at_5": {
            "answer": "",
            "abstain": True,
            "supporting_document_numbers": [1],
            "confidence": "medium",
        },
        "repaired_at_5": {
            "answer": "Sentinel Answer",
            "abstain": False,
            "supporting_document_numbers": [1, 5],
            "confidence": "high",
        },
        "oracle_gold_2": {
            "answer": "Sentinel Answer",
            "abstain": False,
            "supporting_document_numbers": [1, 2],
            "confidence": "high",
        },
    }
    records = [
        {
            "question_id": qid,
            "condition": condition,
            "status": "valid",
            "response": response,
            "gold_document_numbers": (
                [1] if condition == "partial_at_5" else [1, 5]
                if condition == "repaired_at_5"
                else [1, 2]
            ),
        }
        for condition, response in responses.items()
    ]
    report = build_public_generation_report(
        records,
        [private],
        artifact_hashes={"private_output_sha256": "a" * 64},
        model={"name": "Qwen3-8B"},
        protocol={"do_sample": False},
        runtime={"elapsed_seconds": 1.0},
    )
    assert report["conditions"]["partial_at_5"]["metrics"]["answer_em"] == 0.0
    assert report["conditions"]["partial_at_5"]["metrics"][
        "abstain_rate_among_valid"
    ] == 1.0
    assert report["conditions"]["repaired_at_5"]["metrics"]["answer_em"] == 1.0
    assert report["conditions"]["repaired_at_5"]["metrics"][
        "mean_gold_document_citation_recall"
    ] == 1.0
    assert report["conditions"]["repaired_at_5"]["metrics"][
        "correct_with_full_available_gold_citation_count"
    ] == 1
    paired = report["paired_comparisons"][
        "repaired_at_5_minus_partial_at_5"
    ]
    assert paired["mean_answer_f1_delta"] == 1.0
    assert paired["em_transition_counts"]["improved"] == 1
    encoded = json.dumps(report)
    for private_text in (
        qid,
        "What answer follows",
        "Sentinel Answer",
        "Gold Alpha",
        "Gold Beta",
    ):
        assert private_text not in encoded


def test_public_report_requires_exactly_three_conditions() -> None:
    private = _private(_source())
    records = [
        {
            "question_id": private["question_id"],
            "condition": "partial_at_5",
            "response": None,
        }
    ]
    with pytest.raises(GenerationInterventionInvariantError, match="exactly one"):
        build_public_generation_report(
            records,
            [private],
            artifact_hashes={},
            model={},
            protocol={},
            runtime={},
        )
