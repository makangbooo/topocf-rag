"""Independent OpenAI-compatible adjudication of private retrieval errors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
import time
from typing import Any, Callable

import httpx

from .human_audit import HumanAuditBundle, build_annotation_state
from .local_judge import (
    ALLOWED_LABELS,
    LABEL_INSTRUCTIONS,
    SYSTEM_PROMPT,
    parse_judgment_text,
    stable_json_sha256,
)


API_JUDGE_SCHEMA_VERSION = 1
API_JUDGE_PROTOCOL_VERSION = "independent-evidence-sufficiency-judge-v1"
API_JUDGE_PROMPT_VERSION = "natural-error-independent-judge-v1"


class ApiJudgeInvariantError(ValueError):
    """Raised when an API judgment artifact violates the frozen protocol."""


def api_judge_prompt_template_sha256() -> str:
    """Hash the shared message text under an API-specific protocol version."""

    return stable_json_sha256(
        {
            "prompt_version": API_JUDGE_PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "labels": LABEL_INSTRUCTIONS,
        }
    )


def select_final_human_records(
    bundle: HumanAuditBundle,
    annotation_events: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Select records with completed final human decisions in input order."""

    state = build_annotation_state(annotation_events, bundle)
    selected = tuple(
        record
        for record in bundle.records
        if state[str(record["question_id"])]["final"] is not None
    )
    if not selected:
        raise ApiJudgeInvariantError("no completed final human decisions are available")
    return selected


def _response_shape() -> dict[str, Any]:
    return {
        "json_object": False,
        "choices_list": False,
        "choices_count": None,
        "first_choice_object": False,
        "message_object": False,
        "content_string": False,
    }


def _numeric_usage(value: Any) -> Any:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            normalized = _numeric_usage(child)
            if normalized is not None and normalized != {}:
                clean[key] = normalized
        return clean
    return None


def execute_api_judge_call(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_completion_tokens: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Make one call without retries or persistence of headers/error bodies."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise ApiJudgeInvariantError("base_url must be a non-empty string")
    if not isinstance(api_key, str) or not api_key:
        raise ApiJudgeInvariantError("api_key must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ApiJudgeInvariantError("model must be a non-empty string")
    if max_completion_tokens < 1:
        raise ApiJudgeInvariantError("max_completion_tokens must be positive")
    if not messages or any(
        not isinstance(message, Mapping)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ApiJudgeInvariantError("messages violate the chat contract")

    started = clock()
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": list(messages),
                "temperature": 0,
                "max_tokens": max_completion_tokens,
            },
        )
    except httpx.RequestError as error:
        return {
            "status_code": None,
            "latency_seconds": max(0.0, clock() - started),
            "usage": {},
            "response_shape": _response_shape(),
            "raw_output": "",
            "judgment": None,
            "validation_errors": [f"request_error:{type(error).__name__}"],
        }

    latency = max(0.0, clock() - started)
    status_code = int(response.status_code)
    if not 200 <= status_code < 300:
        return {
            "status_code": status_code,
            "latency_seconds": latency,
            "usage": {},
            "response_shape": _response_shape(),
            "raw_output": "",
            "judgment": None,
            "validation_errors": [f"http_status:{status_code}"],
        }

    try:
        payload = response.json()
    except ValueError:
        payload = None
    shape = _response_shape()
    usage: dict[str, Any] = {}
    raw_output = ""
    if isinstance(payload, Mapping):
        shape["json_object"] = True
        clean_usage = _numeric_usage(payload.get("usage"))
        if isinstance(clean_usage, dict):
            usage = clean_usage
        choices = payload.get("choices")
        if isinstance(choices, list):
            shape["choices_list"] = True
            shape["choices_count"] = len(choices)
            if choices and isinstance(choices[0], Mapping):
                shape["first_choice_object"] = True
                message = choices[0].get("message")
                if isinstance(message, Mapping):
                    shape["message_object"] = True
                    content = message.get("content")
                    if isinstance(content, str):
                        shape["content_string"] = True
                        raw_output = content

    if not shape["content_string"]:
        judgment = None
        errors = ("response content is not a string",)
    else:
        judgment, errors = parse_judgment_text(raw_output)
    return {
        "status_code": status_code,
        "latency_seconds": latency,
        "usage": usage,
        "response_shape": shape,
        "raw_output": raw_output,
        "judgment": judgment,
        "validation_errors": list(errors),
    }


def _complete_histogram(counter: Counter[str]) -> dict[str, int]:
    return {label: counter[label] for label in ALLOWED_LABELS}


def _wilson_95(successes: int, total: int) -> dict[str, float | None]:
    if total == 0:
        return {"low": None, "high": None}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return {"low": center - half_width, "high": center + half_width}


def _cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    if len(first) != len(second):
        raise ApiJudgeInvariantError("paired labels have different lengths")
    if not first:
        return None
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] for label in ALLOWED_LABELS
    ) / (len(first) ** 2)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def _agreement(first: Sequence[str], second: Sequence[str]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ApiJudgeInvariantError("paired labels have different lengths")
    count = len(first)
    agreements = sum(a == b for a, b in zip(first, second, strict=True))
    return {
        "eligible_count": count,
        "agreement_count": agreements,
        "agreement_rate": agreements / count if count else None,
        "agreement_wilson_95": _wilson_95(agreements, count),
        "cohen_kappa": _cohen_kappa(first, second),
    }


def _confusion_matrix(
    reference: Sequence[str], prediction: Sequence[str]
) -> dict[str, dict[str, int]]:
    if len(reference) != len(prediction):
        raise ApiJudgeInvariantError("paired labels have different lengths")
    matrix = {
        label: {predicted: 0 for predicted in ALLOWED_LABELS}
        for label in ALLOWED_LABELS
    }
    for expected, predicted in zip(reference, prediction, strict=True):
        matrix[expected][predicted] += 1
    return matrix


def build_public_api_judge_report(
    records: Sequence[Mapping[str, Any]],
    bundle: HumanAuditBundle,
    annotation_events: Sequence[Mapping[str, Any]],
    *,
    artifact_hashes: Mapping[str, str],
    model: str,
    dataset: Mapping[str, str],
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a content-free report aligned to completed API judgments."""

    selected = select_final_human_records(bundle, annotation_events)
    selected_ids = {str(record["question_id"]) for record in selected}
    state = build_annotation_state(annotation_events, bundle)
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        qid = str(record.get("question_id"))
        if qid not in selected_ids or qid in by_id:
            raise ApiJudgeInvariantError("API records contain an unknown/duplicate ID")
        by_id[qid] = record

    selected_final_histogram: Counter[str] = Counter(
        str(state[str(record["question_id"])]["final"]["label"])
        for record in selected
    )
    api_histogram: Counter[str] = Counter()
    blind_labels: list[str] = []
    final_labels: list[str] = []
    qwen_labels: list[str] = []
    api_labels: list[str] = []
    status_codes: Counter[str] = Counter()
    validation_errors: Counter[str] = Counter()
    latencies: list[float] = []
    usage_totals: Counter[str] = Counter()

    for source in selected:
        qid = str(source["question_id"])
        record = by_id.get(qid)
        if record is None:
            continue
        code = record.get("status_code")
        status_codes["none" if code is None else str(code)] += 1
        latency = record.get("latency_seconds")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latencies.append(float(latency))
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage_totals[key] += value
        for error in record.get("validation_errors", []):
            if isinstance(error, str):
                validation_errors[error] += 1
        if record.get("status") != "valid":
            continue
        judgment = record.get("judgment")
        api_label = judgment.get("label") if isinstance(judgment, Mapping) else None
        if api_label not in ALLOWED_LABELS:
            raise ApiJudgeInvariantError("valid API record has an invalid label")
        blind = state[qid]["blind"]
        final = state[qid]["final"]
        qwen = bundle.prelabels[qid].get("judgment")
        qwen_label = qwen.get("label") if isinstance(qwen, Mapping) else None
        if blind is None or final is None or qwen_label not in ALLOWED_LABELS:
            raise ApiJudgeInvariantError("aligned reference label is unavailable")
        api_histogram[str(api_label)] += 1
        blind_labels.append(str(blind["label"]))
        final_labels.append(str(final["label"]))
        qwen_labels.append(str(qwen_label))
        api_labels.append(str(api_label))

    valid_count = len(api_labels)
    invalid_count = len(records) - valid_count
    unanimous = sum(
        len({blind, final, qwen, api}) == 1
        for blind, final, qwen, api in zip(
            blind_labels, final_labels, qwen_labels, api_labels, strict=True
        )
    )
    return {
        "schema_version": API_JUDGE_SCHEMA_VERSION,
        "content_contract": (
            "aggregate counts, rates, model/protocol metadata, and hashes only; "
            "no question IDs, questions, answers, evidence text, rationales, raw "
            "outputs, credentials, URLs, or request headers"
        ),
        "task": "independent_evidence_sufficiency_adjudication",
        "dataset": dict(dataset),
        "model": {"id": model, "provider_interface": "openai_compatible"},
        "protocol": dict(protocol),
        "artifacts": dict(artifact_hashes),
        "counts": {
            "eligible_final_human_count": len(selected),
            "api_completed_count": len(records),
            "api_valid_count": valid_count,
            "api_invalid_count": invalid_count,
        },
        "label_histograms": {
            "eligible_final_human": _complete_histogram(selected_final_histogram),
            "api_valid": _complete_histogram(api_histogram),
        },
        "agreement": {
            "api_vs_blind_human": _agreement(api_labels, blind_labels),
            "api_vs_final_human": _agreement(api_labels, final_labels),
            "api_vs_qwen": _agreement(api_labels, qwen_labels),
            "four_way_unanimous": {
                "eligible_count": valid_count,
                "unanimous_count": unanimous,
                "unanimous_rate": unanimous / valid_count if valid_count else None,
            },
        },
        "confusion_matrices": {
            "final_human_rows_api_columns": _confusion_matrix(
                final_labels, api_labels
            ),
            "qwen_rows_api_columns": _confusion_matrix(qwen_labels, api_labels),
        },
        "api_diagnostics": {
            "status_code_histogram": dict(sorted(status_codes.items())),
            "validation_error_histogram": dict(sorted(validation_errors.items())),
            "usage_totals": dict(sorted(usage_totals.items())),
            "mean_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else None
            ),
        },
        "runtime": dict(runtime),
    }
