import json
from collections.abc import Mapping

import pytest

from angularai.errors import KeyRejected, SubstrateUnavailable
from angularai.models import Workspace
from angularai.provider import Policy
from angularai.substrate import (
    Answer,
    Anthropic,
    OpenAICompatible,
    Transport,
    error_for_status,
    provider_error,
)

KEY = "sk-test-secret-value-xyz"
LEAK = "sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv"


def openai(code: str | None, kind: str, message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": kind, "code": code}}).encode()


def anthropic(kind: str, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()


QUOTA = openai(
    "insufficient_quota",
    "insufficient_quota",
    "You exceeded your current quota, please check your plan and billing details.",
)
BILLING = openai(
    "billing_not_active", "invalid_request_error", "Your account is not active, check billing."
)
RATE = openai("rate_limit_exceeded", "requests", "Rate limit reached for requests per min.")


@pytest.mark.parametrize(
    ("status", "body", "headers", "expected"),
    [
        (
            429,
            QUOTA,
            {},
            "The substrate refused the pass for billing: insufficient_quota. "
            "Check the plan and credits for this key. Waiting will not help.",
        ),
        (429, BILLING, {}, "The substrate refused the pass for billing: billing_not_active."),
        (400, BILLING, {}, "The substrate refused the pass for billing: billing_not_active."),
        (
            400,
            anthropic("invalid_request_error", "Your credit balance is too low."),
            {},
            "The substrate returned 400. Your credit balance is too low.",
        ),
        (402, b"", {}, "The substrate refused the pass for billing: payment_required."),
        (
            429,
            RATE,
            {"retry-after": "20"},
            "The substrate is rate limiting. Try again in 20 seconds.",
        ),
        (
            429,
            RATE,
            {"retry-after-ms": "1500"},
            "The substrate is rate limiting. Try again in 2 seconds.",
        ),
        (429, RATE, {}, "The substrate is rate limiting. Wait and try again."),
        (
            429,
            anthropic("rate_limit_error", "Number of request tokens has exceeded your limit."),
            {"Retry-After": "3"},
            "The substrate is rate limiting. Try again in 3 seconds.",
        ),
        (429, b"", {"retry-after": "soon"}, "Wait and try again."),
        (429, b"not json", {}, "The substrate is rate limiting. Wait and try again."),
        (
            529,
            anthropic("overloaded_error", "Overloaded"),
            {},
            "The substrate is overloaded. Try again later.",
        ),
        (
            503,
            anthropic("overloaded_error", "Overloaded"),
            {},
            "The substrate is overloaded. Try again later.",
        ),
        (
            404,
            openai("model_not_found", "invalid_request_error", "The model `x` does not exist."),
            {},
            "The substrate returned 404. Check the model and the base URL. "
            "The model `x` does not exist.",
        ),
        (
            500,
            openai(None, "server_error", "The server had an error."),
            {},
            "The substrate is unavailable. The server had an error.",
        ),
        (503, b"", {}, "The substrate is unavailable."),
        (
            400,
            openai(None, "invalid_request_error", "Unsupported parameter: 'response_format'."),
            {},
            "The substrate returned 400. Unsupported parameter: 'response_format'.",
        ),
        (422, json.dumps({"detail": "bad field"}).encode(), {}, "returned 422. bad field"),
        (400, json.dumps({"error": "model is required"}).encode(), {}, "model is required"),
        (418, json.dumps({"message": "teapot"}).encode(), {}, "The substrate returned 418. teapot"),
    ],
)
def test_a_provider_error_maps_to_one_sentence(
    status: int, body: bytes, headers: dict[str, str], expected: str
) -> None:
    error = error_for_status(status, body, KEY, headers)
    assert isinstance(error, SubstrateUnavailable)
    assert expected in str(error)


def test_billing_is_not_a_rate_limit() -> None:
    message = str(error_for_status(429, QUOTA, KEY, {"retry-after": "20"}))
    assert "rate limiting" not in message
    assert "insufficient_quota" in message


def test_key_errors() -> None:
    rejected = error_for_status(401, openai("invalid_api_key", "invalid_request_error", KEY), KEY)
    assert isinstance(rejected, KeyRejected)
    assert str(rejected) == "The substrate rejected the API key."
    forbidden = error_for_status(
        403,
        openai("unsupported_country_region_territory", "request_forbidden", "Not supported."),
        KEY,
    )
    assert isinstance(forbidden, KeyRejected)
    assert str(forbidden) == "The substrate does not allow this key to do that. Not supported."


def test_a_provider_message_is_redacted_and_capped() -> None:
    body = openai(None, "invalid_request_error", f"bad key {KEY} and {LEAK} " + "x" * 400)
    message = str(error_for_status(400, body, KEY))
    assert KEY not in message
    assert LEAK not in message
    assert "[redacted]" in message
    assert len(message) <= len("The substrate returned 400. ") + 200
    assert message.endswith("...")


def test_provider_error_reads_each_shape() -> None:
    assert provider_error(QUOTA)[0] == frozenset({"insufficient_quota"})
    assert provider_error(anthropic("rate_limit_error", "slow")) == (
        frozenset({"rate_limit_error"}),
        "slow",
    )
    assert provider_error(json.dumps({"error": "plain"}).encode()) == (frozenset(), "plain")
    assert provider_error(b"\xff") == (frozenset(), "")
    assert provider_error(b"[1]") == (frozenset(), "")


def _workspace() -> Workspace:
    return Workspace(
        system="SYS",
        user="hello",
        input_text="hello",
        trace_ids=(),
        estimated_tokens=10,
        budget=2000,
        traces_shown=0,
        traces_held=0,
        episodes_shown=0,
        episodes_held=0,
    )


def _fake(answer: Answer) -> Transport:
    def transport(
        url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> Answer:
        del url, headers, payload, timeout
        return answer

    return transport


def test_openai_quota_over_http() -> None:
    model = OpenAICompatible(
        model="demo",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        json_mode=False,
        completion_field="max_completion_tokens",
        transport=_fake((429, QUOTA, {"retry-after": "1"})),
    )
    with pytest.raises(SubstrateUnavailable, match="billing: insufficient_quota"):
        model.integrate(_workspace())


def test_anthropic_rate_limit_over_http() -> None:
    body = anthropic("rate_limit_error", "slow down")
    model = Anthropic(
        model="demo",
        base_url="https://api.anthropic.com/v1",
        api_key=KEY,
        transport=_fake((429, body, {"retry-after": "9"})),
        policy=Policy(attempts=1),
    )
    with pytest.raises(SubstrateUnavailable, match="Try again in 9 seconds"):
        model.integrate(_workspace())


def test_a_two_part_answer_still_maps() -> None:
    model = Anthropic(
        model="demo",
        base_url="https://api.anthropic.com/v1",
        api_key=KEY,
        transport=_fake((529, b"")),
        policy=Policy(attempts=1),
    )
    with pytest.raises(SubstrateUnavailable, match="overloaded"):
        model.integrate(_workspace())
