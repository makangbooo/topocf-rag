import json

import httpx

from topocf_rag.api_judge import (
    api_judge_prompt_template_sha256,
    api_judge_run_fingerprint,
    build_public_api_judge_report,
    execute_api_judge_call,
    select_final_human_records,
)
from topocf_rag.human_audit import HumanAuditBundle
from topocf_rag.local_judge import stable_json_sha256


def _judgment(label: str = "locally_true_globally_incomplete") -> dict[str, object]:
    return {
        "label": label,
        "retrieved_local_facts_true": True,
        "globally_sufficient": False,
        "alternative_proof_present": False,
        "confidence": "high",
        "rationale": "One relevant local step is present.",
        "missing_requirement": "The bridge-to-answer step is missing.",
    }


def _private(qid: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "privacy": "private_text_bearing_artifact_do_not_commit",
        "question_id": qid,
        "question": f"Private question {qid}",
        "answer": "Private answer",
        "retrieved_documents": [
            {"title": "Private document", "sentences": ["Private sentence."]}
        ],
        "gold_supporting_facts_for_adjudication": [
            {
                "title": "Private gold",
                "sentence": "Private gold sentence.",
                "retrieved": False,
            }
        ],
        "proxy": {"partial_gold_evidence_proxy": True},
    }


def _bundle() -> HumanAuditBundle:
    records = (_private("q1"), _private("q2"))
    return HumanAuditBundle(
        records=records,
        prelabels={
            str(record["question_id"]): {
                "judgment": _judgment(),
                "status": "valid",
            }
            for record in records
        },
        input_sha256="a" * 64,
        prelabels_sha256="b" * 64,
        run_fingerprint="qwen-run",
    )


def _events() -> list[dict[str, object]]:
    return [
        {
            "event_type": "blind_decision",
            "question_id": "q1",
            "label": "locally_true_globally_incomplete",
        },
        {
            "event_type": "final_decision",
            "question_id": "q1",
            "label": "locally_true_globally_incomplete",
            "blind_label_at_reveal": "locally_true_globally_incomplete",
            "prelabel_label_at_reveal": "locally_true_globally_incomplete",
        },
    ]


class _Response:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _Client:
    def __init__(self, response: _Response):
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_execute_api_judge_call_is_deterministic_and_parses_strict_json() -> None:
    assert len(api_judge_prompt_template_sha256()) == 64
    client = _Client(
        _Response(
            200,
            {
                "choices": [{"message": {"content": json.dumps(_judgment())}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )
    )
    result = execute_api_judge_call(
        client,  # type: ignore[arg-type]
        base_url="https://private.invalid/v1/",
        api_key="secret",
        model="judge-model",
        messages=[{"role": "user", "content": "private prompt"}],
        max_completion_tokens=512,
        clock=iter((1.0, 2.5)).__next__,
    )
    assert result["status_code"] == 200
    assert result["judgment"]["label"] == "locally_true_globally_incomplete"
    request = client.calls[0]
    assert request["url"] == "https://private.invalid/v1/chat/completions"
    assert request["json"]["temperature"] == 0
    assert request["json"]["max_tokens"] == 512


def test_run_fingerprint_binds_timeout_model_and_dataset() -> None:
    base = {
        "dataset_name": "hotpotqa",
        "dataset_split": "dev_distractor",
        "model": "judge-model",
        "private_input_sha256": "a" * 64,
        "private_annotations_sha256": "b" * 64,
        "selected_sequence_sha256": "c" * 64,
        "max_completion_tokens": 512,
        "timeout_seconds": 120.0,
    }
    fingerprint = api_judge_run_fingerprint(**base)
    assert len(fingerprint) == 64
    for key, value in (
        ("timeout_seconds", 300.0),
        ("model", "different-model"),
        ("dataset_split", "train"),
    ):
        changed = dict(base)
        changed[key] = value
        assert api_judge_run_fingerprint(**changed) != fingerprint


def test_execute_api_judge_call_never_persists_http_error_body() -> None:
    client = _Client(_Response(429, {"error": "private provider error"}))
    result = execute_api_judge_call(
        client,  # type: ignore[arg-type]
        base_url="https://private.invalid",
        api_key="secret",
        model="judge-model",
        messages=[{"role": "user", "content": "private prompt"}],
        max_completion_tokens=10,
        clock=iter((1.0, 1.5)).__next__,
    )
    assert result["raw_output"] == ""
    assert result["validation_errors"] == ["http_status:429"]
    assert "private provider error" not in json.dumps(result)


def test_select_final_human_records_uses_only_completed_final_decisions() -> None:
    selected = select_final_human_records(_bundle(), _events())
    assert [record["question_id"] for record in selected] == ["q1"]


def test_public_report_is_content_free_and_aligns_four_labels() -> None:
    bundle = _bundle()
    source = bundle.records[0]
    record = {
        "question_id": "q1",
        "input_sha256": stable_json_sha256(source),
        "status": "valid",
        "status_code": 200,
        "latency_seconds": 1.5,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "validation_errors": [],
        "judgment": _judgment(),
        "raw_output": "private model rationale",
    }
    report = build_public_api_judge_report(
        [record],
        bundle,
        _events(),
        artifact_hashes={"private_api_output_sha256": "c" * 64},
        model="judge-model",
        dataset={"name": "hotpotqa", "split": "dev_distractor"},
        protocol={"run_fingerprint": "run"},
        runtime={"invocation_elapsed_seconds": 1.5},
    )
    assert report["counts"]["api_valid_count"] == 1
    assert report["agreement"]["api_vs_final_human"]["agreement_rate"] == 1.0
    assert report["agreement"]["four_way_unanimous"]["unanimous_count"] == 1
    encoded = json.dumps(report)
    for private_text in (
        "q1",
        "Private question",
        "Private answer",
        "Private document",
        "private model rationale",
    ):
        assert private_text not in encoded


def test_request_error_is_sanitized() -> None:
    class FailingClient:
        def post(self, *_args: object, **_kwargs: object) -> object:
            request = httpx.Request("POST", "https://private.invalid")
            raise httpx.ConnectError("private network detail", request=request)

    result = execute_api_judge_call(
        FailingClient(),  # type: ignore[arg-type]
        base_url="https://private.invalid",
        api_key="secret",
        model="judge-model",
        messages=[{"role": "user", "content": "private prompt"}],
        max_completion_tokens=10,
        clock=iter((1.0, 1.1)).__next__,
    )
    assert result["validation_errors"] == ["request_error:ConnectError"]
    assert "private network detail" not in json.dumps(result)
