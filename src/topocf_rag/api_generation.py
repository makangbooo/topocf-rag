"""Credential-safe helpers for OpenAI-compatible generation workflows."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

import httpx


SMOKE_PROMPT = "Return exactly OK."
SMOKE_MAX_TOKENS = 8
SMOKE_TIMEOUT_SECONDS = 60.0


class ApiGenerationInvariantError(ValueError):
    """Raised when an API smoke-test input violates the frozen contract."""


def _empty_response_shape() -> dict[str, Any]:
    return {
        "json_object": False,
        "choices_list": False,
        "choices_count": None,
        "first_choice_object": False,
        "message_object": False,
        "content_string": False,
    }


def _numeric_usage_only(value: Any) -> Any:
    """Recursively retain mappings and finite numeric usage values only."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        sanitized = {}
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            clean_child = _numeric_usage_only(child)
            if clean_child is not None and clean_child != {}:
                sanitized[key] = clean_child
        return sanitized
    return None


def _parse_success_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any], bool]:
    shape = _empty_response_shape()
    if not isinstance(payload, Mapping):
        return shape, {}, False
    shape["json_object"] = True
    usage = _numeric_usage_only(payload.get("usage"))
    if not isinstance(usage, dict):
        usage = {}

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return shape, usage, False
    shape["choices_list"] = True
    shape["choices_count"] = len(choices)
    if not choices or not isinstance(choices[0], Mapping):
        return shape, usage, False
    shape["first_choice_object"] = True
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return shape, usage, False
    shape["message_object"] = True
    content = message.get("content")
    if not isinstance(content, str):
        return shape, usage, False
    shape["content_string"] = True
    return shape, usage, content == "OK"


def execute_smoke_call(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute exactly one deterministic Chat Completions smoke request.

    The returned report intentionally excludes the URL, model, headers, response
    body, and error body.  The caller must not retry this function automatically.
    """

    if not isinstance(base_url, str) or not base_url.strip():
        raise ApiGenerationInvariantError("base_url must be a non-empty string")
    if not isinstance(api_key, str) or not api_key:
        raise ApiGenerationInvariantError("api_key must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ApiGenerationInvariantError("model must be a non-empty string")

    url = f"{base_url.rstrip('/')}/chat/completions"
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "temperature": 0,
        "max_tokens": SMOKE_MAX_TOKENS,
    }
    started = clock()
    try:
        response = client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )
    except httpx.RequestError:
        latency = max(0.0, clock() - started)
        return {
            "status_code": None,
            "latency_seconds": latency,
            "usage": {},
            "response_shape": _empty_response_shape(),
            "exact_match": False,
        }
    latency = max(0.0, clock() - started)
    status_code = int(response.status_code)
    if not 200 <= status_code < 300:
        return {
            "status_code": status_code,
            "latency_seconds": latency,
            "usage": {},
            "response_shape": _empty_response_shape(),
            "exact_match": False,
        }
    try:
        payload = response.json()
    except ValueError:
        payload = None
    shape, usage, exact_match = _parse_success_payload(payload)
    return {
        "status_code": status_code,
        "latency_seconds": latency,
        "usage": usage,
        "response_shape": shape,
        "exact_match": exact_match,
    }
