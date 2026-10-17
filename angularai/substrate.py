import json
from collections.abc import Callable, Mapping
from typing import Protocol, assert_never

from angularai.budget import estimate_tokens
from angularai.errors import (
    InputRejected,
    KeyMissing,
    KeyRejected,
    SubstrateUnavailable,
    SubstrateUnreadable,
)
from angularai.http import http_json
from angularai.models import Config, Integration, SubstrateKind, Workspace
from angularai.parse import parse_integration
from angularai.secrets import environment_secrets, redact

Transport = Callable[[str, dict[str, str], Mapping[str, object], float], tuple[int, bytes]]
_PARSE_NUDGE = "\n\nReturn only the JSON object."


class Substrate(Protocol):
    def integrate(self, workspace: Workspace) -> Integration:
        """One pass on the wired model. At most two HTTP calls."""


class OpenAICompatible:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        json_mode: bool,
        completion_field: str,
        transport: Transport = http_json,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._json_mode = json_mode
        self._completion_field = completion_field
        self._transport = transport

    def integrate(self, workspace: Workspace) -> Integration:
        try:
            text = self._complete(workspace, self._json_mode, workspace.user)
        except SubstrateUnavailable as exc:
            if not self._json_mode or "returned 400" not in str(exc):
                raise
            text = self._complete(workspace, False, workspace.user)
            return parse_integration(text)
        try:
            return parse_integration(text)
        except SubstrateUnreadable:
            extra = workspace.user + _PARSE_NUDGE
            projected = estimate_tokens(workspace.system) + estimate_tokens(extra)
            if projected > workspace.budget:
                raise
            return parse_integration(self._complete(workspace, False, extra))

    def _complete(self, workspace: Workspace, json_mode: bool, user: str) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            self._completion_field: 2048,
            "messages": [
                {"role": "system", "content": workspace.system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "angularai/0.1.0",
        }
        return openai_text(_post(self._transport, url, headers, payload, self._api_key))


class Anthropic:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        transport: Transport = http_json,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport

    def integrate(self, workspace: Workspace) -> Integration:
        text = self._complete(workspace.system, workspace.user)
        try:
            return parse_integration(text)
        except SubstrateUnreadable:
            extra = workspace.user + _PARSE_NUDGE
            projected = estimate_tokens(workspace.system) + estimate_tokens(extra)
            if projected > workspace.budget:
                raise
            return parse_integration(self._complete(workspace.system, extra))

    def _complete(self, system: str, user: str) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        url = f"{self._base_url}/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "User-Agent": "angularai/0.1.0",
        }
        return anthropic_text(_post(self._transport, url, headers, payload, self._api_key))


def build_substrate(config: Config, environ: Mapping[str, str]) -> Substrate:
    if config.substrate is None or config.model is None:
        raise InputRejected(
            "This cortex has no model. Wire one with: angularai wire openai --model <name>"
        )
    kind = config.substrate
    env_name = key_env(kind)
    key = environ.get(env_name, "").strip()
    if not key:
        raise KeyMissing(f"{env_name} is not set. The key stays in the environment.")
    if kind == "openai":
        return OpenAICompatible(
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=key,
            json_mode=config.json_mode,
            completion_field="max_completion_tokens",
        )
    if kind == "anthropic":
        return Anthropic(
            model=config.model,
            base_url=config.base_url or "https://api.anthropic.com/v1",
            api_key=key,
        )
    if kind == "compatible":
        if config.base_url is None:
            raise InputRejected("A compatible substrate needs --base-url.")
        return OpenAICompatible(
            model=config.model,
            base_url=config.base_url,
            api_key=key,
            json_mode=config.json_mode,
            completion_field="max_tokens",
        )
    assert_never(kind)


def secret_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Every value in the environment whose variable name says it is a secret."""

    return environment_secrets(environ)


def key_env(kind: SubstrateKind) -> str:
    if kind == "openai":
        return "OPENAI_API_KEY"
    if kind == "anthropic":
        return "ANTHROPIC_API_KEY"
    if kind == "compatible":
        return "ANGULARAI_API_KEY"
    assert_never(kind)


def openai_text(payload: Mapping[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SubstrateUnreadable("The substrate returned no reply. Nothing was written.")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise SubstrateUnreadable("The substrate returned no reply. Nothing was written.")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise SubstrateUnreadable("The substrate returned no reply. Nothing was written.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise SubstrateUnreadable("The substrate returned no reply. Nothing was written.")
    return content


def anthropic_text(payload: Mapping[str, object]) -> str:
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        raise SubstrateUnreadable("The substrate returned no reply. Nothing was written.")
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        text = block.get("text")
        if block.get("type") == "text" and isinstance(text, str):
            parts.append(text)
    reply = "".join(parts).strip()
    if not reply:
        raise SubstrateUnreadable("The substrate returned no reply. Nothing was written.")
    return reply


def error_for_status(status: int, body: bytes, secret: str) -> SubstrateUnavailable | KeyRejected:
    if status in {401, 403}:
        return KeyRejected("The substrate rejected the API key.")
    if status == 429:
        return SubstrateUnavailable("The substrate is rate limiting. Wait and try again.")
    if status >= 500:
        return SubstrateUnavailable("The substrate is unavailable.")
    message = redact(provider_message(body), (secret,))
    if message:
        return SubstrateUnavailable(f"The substrate returned {status}. {message}")
    return SubstrateUnavailable(f"The substrate returned {status}.")


def provider_message(body: bytes) -> str:
    try:
        payload: object = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return " ".join(message.split())[:200]
    if isinstance(error, str):
        return " ".join(error.split())[:200]
    return ""


def _post(
    transport: Transport,
    url: str,
    headers: dict[str, str],
    payload: Mapping[str, object],
    secret: str,
) -> dict[str, object]:
    status, body = transport(url, headers, payload, 90)
    if status < 200 or status >= 300:
        raise error_for_status(status, body, secret)
    return _object(body)


def _object(body: bytes) -> dict[str, object]:
    try:
        payload: object = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SubstrateUnreadable("The substrate returned something that is not JSON.") from exc
    if not isinstance(payload, dict):
        raise SubstrateUnreadable("The substrate returned something that is not JSON.")
    result: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise SubstrateUnreadable("The substrate returned something that is not JSON.")
        result[key] = value
    return result
