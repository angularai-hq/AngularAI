import json
from collections.abc import Mapping
from pathlib import Path
from urllib.request import Request

import pytest

from angularai.errors import KeyRejected, SubstrateUnavailable, SubstrateUnreadable
from angularai.http import _NoRedirect
from angularai.models import Workspace
from angularai.pilot import pass_turn
from angularai.store import init_cortex, open_store
from angularai.substrate import Anthropic, OpenAICompatible, error_for_status

KEY = "sk-test-secret-value"


def workspace() -> Workspace:
    return Workspace(
        system="SYS",
        user="USER",
        input_text="USER",
        trace_ids=(),
        estimated_tokens=2,
        budget=80,
        traces_shown=0,
        traces_held=0,
        episodes_shown=0,
        episodes_held=0,
    )


def reply(text: str) -> bytes:
    body = {"reply": text, "remember": [], "supersede": []}
    payload = {"choices": [{"message": {"content": json.dumps(body)}}]}
    return json.dumps(payload).encode()


def test_openai_sends_only_the_workspace() -> None:
    captured: list[Mapping[str, object]] = []

    def transport(
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, timeout
        assert url == "https://api.openai.com/v1/chat/completions"
        captured.append(payload)
        return 200, reply("Hello.")

    model = OpenAICompatible(
        model="demo",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        json_mode=True,
        completion_field="max_completion_tokens",
        transport=transport,
    )
    integration = model.integrate(workspace())
    assert integration.reply == "Hello."
    sent = captured[0]
    messages = sent["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"] == "SYS"
    assert messages[1]["content"] == "USER"
    assert sent["max_completion_tokens"] == 2048
    assert sent["response_format"] == {"type": "json_object"}
    assert KEY not in json.dumps(sent)


def test_a_bad_reply_is_asked_once_more() -> None:
    captured: list[Mapping[str, object]] = []

    def transport(
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, timeout
        captured.append(payload)
        if len(captured) == 1:
            payload_body = {"choices": [{"message": {"content": "not json"}}]}
            return 200, json.dumps(payload_body).encode()
        return 200, reply("Hello.")

    model = OpenAICompatible(
        model="demo",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        json_mode=True,
        completion_field="max_completion_tokens",
        transport=transport,
    )
    assert model.integrate(workspace()).reply == "Hello."
    assert len(captured) == 2
    messages = captured[1]["messages"]
    assert isinstance(messages, list)
    assert "Return only the JSON object." in messages[1]["content"]


def test_json_mode_rejection_retries_without_a_third_call() -> None:
    captured: list[Mapping[str, object]] = []

    def transport(
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, timeout
        captured.append(payload)
        if len(captured) == 1:
            return 400, b'{"error":{"message":"response_format unsupported"}}'
        return 200, b'{"choices":[{"message":{"content":"still not json"}}]}'

    model = OpenAICompatible(
        model="demo",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        json_mode=True,
        completion_field="max_completion_tokens",
        transport=transport,
    )
    with pytest.raises(SubstrateUnreadable):
        model.integrate(workspace())
    assert len(captured) == 2
    assert "response_format" not in captured[1]


def test_rejected_key_is_not_echoed() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, payload, timeout
        return 401, json.dumps({"error": {"message": KEY}}).encode()

    model = OpenAICompatible(
        model="demo",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        json_mode=False,
        completion_field="max_completion_tokens",
        transport=transport,
    )
    with pytest.raises(KeyRejected) as caught:
        model.integrate(workspace())
    assert KEY not in str(caught.value)


def test_a_provider_error_does_not_store_the_key(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)

    def transport(
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, payload, timeout
        return 400, json.dumps({"error": {"message": f"bad {KEY}"}}).encode()

    model = OpenAICompatible(
        model="demo",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        json_mode=False,
        completion_field="max_completion_tokens",
        transport=transport,
    )
    with pytest.raises(SubstrateUnavailable):
        pass_turn(store, model, "Hello there.")
    log = (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    assert KEY not in log
    assert "[redacted]" in log


def test_anthropic_posts_to_messages() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> tuple[int, bytes]:
        del timeout
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == KEY
        assert payload["system"] == "SYS"
        body = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"reply": "Hello.", "remember": [], "supersede": []}),
                }
            ]
        }
        return 200, json.dumps(body).encode()

    model = Anthropic(
        model="demo",
        base_url="https://api.anthropic.com/v1/",
        api_key=KEY,
        transport=transport,
    )
    assert model.integrate(workspace()).reply == "Hello."


def test_status_errors_are_specific() -> None:
    assert "rate limiting" in str(error_for_status(429, b"", KEY))
    assert str(error_for_status(503, b"", KEY)) == "The substrate is unavailable."
    assert KEY not in str(error_for_status(401, json.dumps({"error": KEY}).encode(), KEY))


def test_redirects_are_not_followed() -> None:
    handler = _NoRedirect()
    assert (
        handler.redirect_request(
            Request("http://example.test"),
            None,
            302,
            "Found",
            {},
            "https://evil.test",
        )
        is None
    )
