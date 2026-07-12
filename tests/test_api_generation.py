import json

import httpx
import pytest

from topocf_rag.api_generation import (
    ApiGenerationInvariantError,
    execute_smoke_call,
)


def _clock(values: list[float]):
    iterator = iter(values)
    return lambda: next(iterator)


def test_smoke_call_sends_exact_frozen_request_once_and_sanitizes_report() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-id-must-not-be-recorded",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 1,
                    "total_tokens": 7,
                    "provider_text": "must-not-be-recorded",
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        report = execute_smoke_call(
            client,
            base_url="https://gateway.invalid/v1/",
            api_key="unit-test-secret",
            model="unit-test-model",
            clock=_clock([10.0, 10.25]),
        )

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://gateway.invalid/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer unit-test-secret"
    assert json.loads(request.content) == {
        "model": "unit-test-model",
        "messages": [{"role": "user", "content": "Return exactly OK."}],
        "temperature": 0,
        "max_tokens": 8,
    }
    assert report == {
        "status_code": 200,
        "latency_seconds": 0.25,
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 1,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
        "response_shape": {
            "json_object": True,
            "choices_list": True,
            "choices_count": 1,
            "first_choice_object": True,
            "message_object": True,
            "content_string": True,
        },
        "exact_match": True,
    }
    encoded = json.dumps(report, sort_keys=True)
    for forbidden in (
        "unit-test-secret",
        "unit-test-model",
        "gateway.invalid",
        "response-id-must-not-be-recorded",
        "provider_text",
    ):
        assert forbidden not in encoded


def test_smoke_call_does_not_strip_or_normalize_content() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK\n"}}]},
        )
    )
    with httpx.Client(transport=transport) as client:
        report = execute_smoke_call(
            client,
            base_url="https://gateway.invalid/v1",
            api_key="secret",
            model="model",
            clock=_clock([1.0, 2.0]),
        )
    assert report["exact_match"] is False
    assert report["response_shape"]["content_string"] is True


def test_non_success_response_is_not_parsed_or_recorded() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401,
            json={"error": "secret provider error body"},
        )
    )
    with httpx.Client(transport=transport) as client:
        report = execute_smoke_call(
            client,
            base_url="https://gateway.invalid/v1",
            api_key="secret",
            model="model",
            clock=_clock([3.0, 3.5]),
        )
    assert report["status_code"] == 401
    assert report["usage"] == {}
    assert report["exact_match"] is False
    assert "secret provider error body" not in json.dumps(report)


def test_transport_failure_returns_safe_report_without_retry() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("provider detail must not be stored", request=request)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        report = execute_smoke_call(
            client,
            base_url="https://gateway.invalid/v1",
            api_key="secret",
            model="model",
            clock=_clock([5.0, 6.0]),
        )
    assert call_count == 1
    assert report["status_code"] is None
    assert report["exact_match"] is False
    assert "provider detail" not in json.dumps(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [("base_url", ""), ("api_key", ""), ("model", "")],
)
def test_smoke_call_rejects_missing_inputs(field: str, value: str) -> None:
    kwargs = {
        "base_url": "https://gateway.invalid/v1",
        "api_key": "secret",
        "model": "model",
    }
    kwargs[field] = value
    with httpx.Client(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ApiGenerationInvariantError):
            execute_smoke_call(client, **kwargs)
