import json
from pathlib import Path

import pytest

from topocf_rag.human_audit import (
    HumanAuditInvariantError,
    build_annotation_state,
    build_item_payload,
    build_public_human_audit_report,
    load_annotation_events,
    load_human_audit_bundle,
    record_blind_decision,
    record_final_decision,
    sha256_file,
    write_public_human_audit_report,
)
from topocf_rag.local_judge import stable_json_sha256


def _private_record(qid: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "privacy": "private_text_bearing_artifact_do_not_commit",
        "question_id": qid,
        "question": f"Private question text {qid}",
        "answer": "Private reference answer",
        "primary_top_k": 5,
        "proxy": {
            "partial_gold_evidence_proxy": True,
            "answer_string_present": False,
        },
        "retrieved_documents": [
            {
                "context_index": 0,
                "retrieval_rank": 1,
                "retrieval_score": 0.9,
                "title": "Retrieved Title",
                "sentences": ["A private retrieved sentence."],
            }
        ],
        "gold_supporting_facts_for_adjudication": [
            {
                "context_index": 0,
                "retrieved": True,
                "title": "Retrieved Title",
                "sentence_index": 0,
                "sentence": "A private retrieved sentence.",
            },
            {
                "context_index": 1,
                "retrieved": False,
                "title": "Missing Title",
                "sentence_index": 0,
                "sentence": "A private missing bridge sentence.",
            },
        ],
        "human_annotation": {"label": None, "notes": ""},
    }


def _judgment(label: str = "locally_true_globally_incomplete") -> dict[str, object]:
    return {
        "label": label,
        "retrieved_local_facts_true": True,
        "globally_sufficient": False,
        "alternative_proof_present": False,
        "confidence": "medium",
        "rationale": "A private model rationale.",
        "missing_requirement": "A private missing relation.",
    }


def _prelabel(record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "privacy": "private_text_bearing_artifact_do_not_commit",
        "question_id": record["question_id"],
        "input_sha256": stable_json_sha256(record),
        "run_fingerprint": "run-fingerprint",
        "status": "valid",
        "judgment": _judgment(),
        "validation_errors": [],
        "raw_output": "private raw model output",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "elapsed_seconds": 1.0,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def audit_files(tmp_path: Path) -> tuple[Path, Path, Path, object]:
    source = [_private_record("q1"), _private_record("q2")]
    prelabels = [_prelabel(record) for record in source]
    input_path = tmp_path / "input.jsonl"
    prelabels_path = tmp_path / "prelabels.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    _write_jsonl(input_path, source)
    _write_jsonl(prelabels_path, prelabels)
    bundle = load_human_audit_bundle(input_path, prelabels_path)
    return input_path, prelabels_path, annotations_path, bundle


def test_bundle_requires_exact_record_order_and_matching_hashes(tmp_path: Path) -> None:
    source = [_private_record("q1"), _private_record("q2")]
    input_path = tmp_path / "input.jsonl"
    prelabels_path = tmp_path / "prelabels.jsonl"
    _write_jsonl(input_path, source)
    _write_jsonl(prelabels_path, [_prelabel(source[1]), _prelabel(source[0])])
    with pytest.raises(HumanAuditInvariantError, match="input order"):
        load_human_audit_bundle(input_path, prelabels_path)

    bad = [_prelabel(record) for record in source]
    bad[0]["input_sha256"] = "0" * 64
    _write_jsonl(prelabels_path, bad)
    with pytest.raises(HumanAuditInvariantError, match="input hash"):
        load_human_audit_bundle(input_path, prelabels_path)


def test_prelabel_is_hidden_until_blind_decision(audit_files: tuple) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    state = build_annotation_state([], bundle)
    payload = build_item_payload(bundle, state, 0)

    assert payload["prelabel"] is None
    assert "proxy" not in payload
    assert "human_annotation" not in payload
    assert payload["question"] == "Private question text q1"

    record_blind_decision(
        bundle,
        annotations,
        question_id="q1",
        label="locally_true_globally_incomplete",
        created_at_utc="2026-07-13T00:00:00Z",
    )
    events = load_annotation_events(annotations, bundle)
    payload = build_item_payload(bundle, build_annotation_state(events, bundle), 0)
    assert payload["prelabel"]["judgment"]["label"] == (
        "locally_true_globally_incomplete"
    )
    assert "raw_output" not in payload["prelabel"]


def test_final_decision_cannot_precede_blind(audit_files: tuple) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    with pytest.raises(HumanAuditInvariantError, match="must be locked"):
        record_final_decision(
            bundle,
            annotations,
            question_id="q1",
            label="other",
        )


def test_blind_decision_is_immutable_and_private(audit_files: tuple) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    record_blind_decision(
        bundle,
        annotations,
        question_id="q1",
        label="other",
        notes="private human note",
    )
    assert annotations.stat().st_mode & 0o777 == 0o600
    with pytest.raises(HumanAuditInvariantError, match="already locked"):
        record_blind_decision(
            bundle,
            annotations,
            question_id="q1",
            label="locally_true_globally_incomplete",
        )


def test_final_decision_records_reveal_snapshots_and_may_be_revised(
    audit_files: tuple,
) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    record_blind_decision(
        bundle,
        annotations,
        question_id="q1",
        label="other",
        created_at_utc="2026-07-13T00:00:00Z",
    )
    first = record_final_decision(
        bundle,
        annotations,
        question_id="q1",
        label="locally_true_globally_incomplete",
        created_at_utc="2026-07-13T00:01:00Z",
    )
    second = record_final_decision(
        bundle,
        annotations,
        question_id="q1",
        label="ambiguous_or_annotation_issue",
        created_at_utc="2026-07-13T00:02:00Z",
    )
    assert first["blind_label_at_reveal"] == "other"
    assert first["prelabel_label_at_reveal"] == (
        "locally_true_globally_incomplete"
    )
    events = load_annotation_events(annotations, bundle)
    state = build_annotation_state(events, bundle)
    assert state["q1"]["final"] == second
    assert len(events) == 3


def test_annotation_artifacts_are_bound_to_input_and_prelabels(
    audit_files: tuple, tmp_path: Path
) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    record_blind_decision(
        bundle,
        annotations,
        question_id="q1",
        label="other",
    )
    altered_source = [_private_record("q1"), _private_record("q2")]
    altered_source[0]["question"] = "Changed private question"
    altered_input = tmp_path / "altered-input.jsonl"
    altered_prelabels = tmp_path / "altered-prelabels.jsonl"
    _write_jsonl(altered_input, altered_source)
    _write_jsonl(altered_prelabels, [_prelabel(record) for record in altered_source])
    altered_bundle = load_human_audit_bundle(altered_input, altered_prelabels)
    with pytest.raises(HumanAuditInvariantError, match="input hash"):
        load_annotation_events(annotations, altered_bundle)


def test_public_report_contains_only_aggregate_results(audit_files: tuple) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    record_blind_decision(
        bundle,
        annotations,
        question_id="q1",
        label="other",
        notes="private human note",
    )
    record_final_decision(
        bundle,
        annotations,
        question_id="q1",
        label="locally_true_globally_incomplete",
    )
    events = load_annotation_events(annotations, bundle)
    report = build_public_human_audit_report(
        bundle, events, annotations_sha256=sha256_file(annotations)
    )
    encoded = json.dumps(report)

    assert report["counts"] == {
        "eligible_question_count": 2,
        "annotation_event_count": 2,
        "blind_completed_count": 1,
        "final_completed_count": 1,
    }
    assert report["agreement"]["blind_vs_qwen"]["agreement_rate"] == 0.0
    assert report["agreement"]["final_vs_qwen"]["agreement_rate"] == 1.0
    assert report["agreement"]["changed_after_qwen_reveal"]["changed_rate"] == 1.0
    for private_text in (
        "q1",
        "Private question",
        "Private reference answer",
        "private human note",
        "private model rationale",
    ):
        assert private_text not in encoded


def test_report_writer_supports_empty_and_partial_annotation_files(
    audit_files: tuple, tmp_path: Path
) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    report_path = tmp_path / "report.json"
    empty = write_public_human_audit_report(bundle, annotations, report_path)
    assert empty["counts"]["blind_completed_count"] == 0
    assert empty["artifacts"]["private_annotations_sha256"] is None

    record_blind_decision(
        bundle,
        annotations,
        question_id="q1",
        label="other",
    )
    partial = write_public_human_audit_report(bundle, annotations, report_path)
    assert partial["counts"]["blind_completed_count"] == 1
    assert partial["artifacts"]["private_annotations_sha256"] == sha256_file(
        annotations
    )
    assert json.loads(report_path.read_text()) == partial


def test_invalid_label_and_oversized_notes_are_rejected(audit_files: tuple) -> None:
    _input, _prelabels, annotations, bundle = audit_files
    with pytest.raises(HumanAuditInvariantError, match="not allowed"):
        record_blind_decision(
            bundle, annotations, question_id="q1", label="made-up-label"
        )
    with pytest.raises(HumanAuditInvariantError, match="at most 2000"):
        record_blind_decision(
            bundle,
            annotations,
            question_id="q1",
            label="other",
            notes="x" * 2001,
        )
