import html
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from secrets import token_hex
from typing import Protocol, assert_never

from angularai.budget import PARSE_NUDGE, estimate_tokens
from angularai.errors import (
    AngularAIError,
    AuthFailed,
    BadResponse,
    Forbidden,
    InputRejected,
    KeyMissing,
    KeyRejected,
    NotFound,
    Overloaded,
    QuotaExceeded,
    RateLimited,
    RequestRejected,
    ServerError,
    SubstrateError,
    SubstrateUnavailable,
    SubstrateUnreadable,
    Timeout,
)
from angularai.http import http_json
from angularai.models import Config, Integration, SubstrateKind, Workspace
from angularai.parse import TOOL_SPECS, integration_from_calls, parse_integration
from angularai.pricing import Usage, usage_of
from angularai.profiles import KEY_ENVS, Vault, effective_config, saved_key
from angularai.provider import (
    Clock,
    Deadline,
    Policy,
    policy_from_env,
    retry_after,
    wait_before_retry,
)
from angularai.secrets import environment_secrets, redact

Answer = tuple[int, bytes] | tuple[int, bytes, Mapping[str, str]]
# (url, headers, payload, read timeout in seconds) -> answer
Transport = Callable[[str, dict[str, str], Mapping[str, object], float], Answer]
_NO_REPLY = "returned no reply. Nothing was written."


class Metered(Protocol):
    """A substrate that reports what it was billed."""

    def billed(self) -> Usage: ...


class Substrate(Protocol):
    def integrate(self, workspace: Workspace) -> Integration:
        """One pass on the wired model. At most two calls, each retried within the deadline."""


def default_transport(policy: Policy) -> Transport:
    def send(
        url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> Answer:
        return http_json(url, headers, payload, timeout, connect_timeout=policy.connect_timeout)

    return send


class _Caller:
    """Sends one request, retrying what may be retried, within one pass deadline.

    A retry resends the same request inside integrate(), before any reply exists,
    so the pass is still written once and counted once.
    """

    def __init__(
        self,
        *,
        transport: Transport | None,
        policy: Policy | None,
        clock: Clock | None,
        provider: str,
        secret: str,
    ) -> None:
        self.policy = policy or Policy()
        self.transport = transport or default_transport(self.policy)
        self.clock = clock or Clock()
        self.provider = provider
        self.secret = secret
        # Every token billed through this caller, failed passes included.
        self.usage = Usage()

    def deadline(self) -> Deadline:
        return Deadline(self.policy.deadline, self.clock)

    def post(
        self,
        deadline: Deadline,
        url: str,
        headers: dict[str, str],
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline.remaining()
            if remaining <= 0:
                raise self._out_of_time(deadline)
            try:
                answer = self.transport(
                    url, headers, payload, min(self.policy.read_timeout, remaining)
                )
                status, body = answer[0], answer[1]
                if status < 200 or status >= 300:
                    found = answer[2] if len(answer) == 3 else {}
                    raise error_for_status(
                        status,
                        body,
                        self.secret,
                        found,
                        provider=self.provider,
                        now=self.clock.wall(),
                    )
                parsed = _object(body, self.secret, self.provider)
                self.usage = self.usage + usage_of(parsed)
                return parsed
            except SubstrateError as exc:
                if attempt >= self.policy.attempts:
                    raise
                wait = wait_before_retry(exc, attempt, self.policy, self.clock)
                if wait is None:
                    raise
                if wait >= deadline.remaining():
                    raise self._out_of_time(deadline) from exc
                self.clock.sleep(wait)

    def _out_of_time(self, deadline: Deadline) -> Timeout:
        return Timeout(
            f"{_subject(self.provider)} did not answer within {deadline.seconds:g} seconds."
        )


class OpenAICompatible:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        json_mode: bool,
        completion_field: str,
        transport: Transport | None = None,
        policy: Policy | None = None,
        clock: Clock | None = None,
        provider: str = "",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._json_mode = json_mode
        self._completion_field = completion_field
        self._caller = _Caller(
            transport=transport, policy=policy, clock=clock, provider=provider, secret=api_key
        )

    def billed(self) -> Usage:
        """Every token this substrate has been billed for."""

        return self._caller.usage

    def integrate(self, workspace: Workspace) -> Integration:
        deadline = self._caller.deadline()
        if workspace.native_tools:
            try:
                return self._native(deadline, workspace)
            except RequestRejected as exc:
                if exc.status != 400:
                    raise
                # A server without tool calling: the JSON act protocol below.
        workspace = inline_exchange(workspace)
        try:
            text = self._complete(deadline, workspace, self._json_mode, workspace.user)
        except RequestRejected as exc:
            if not self._json_mode or exc.status != 400:
                raise
            text = self._complete(deadline, workspace, False, workspace.user)
            return parse_integration(text)
        try:
            return parse_integration(text)
        except SubstrateUnreadable:
            extra = workspace.user + PARSE_NUDGE
            projected = estimate_tokens(workspace.system) + estimate_tokens(extra)
            if projected > workspace.budget:
                raise
            return parse_integration(self._complete(deadline, workspace, False, extra))

    def _native(self, deadline: Deadline, workspace: Workspace) -> Integration:
        payload: dict[str, object] = {
            "model": self._model,
            self._completion_field: 2048,
            "messages": native_messages(workspace),
            "tools": list(TOOL_SPECS),
            "tool_choice": "auto",
        }
        url = f"{self._base_url}/chat/completions"
        reply = self._caller.post(deadline, url, _openai_headers(self._api_key), payload)
        content, calls = openai_message(reply, self._caller.provider)
        if calls:
            return integration_from_calls(content, calls)
        try:
            return parse_integration(content)
        except SubstrateUnreadable:
            # Plain text and no tool call: idle, and the model is told so.
            return integration_from_calls(content, ())

    def ping(self) -> None:
        payload: dict[str, object] = {
            "model": self._model,
            self._completion_field: 16,
            "messages": [{"role": "user", "content": "ping"}],
        }
        url = f"{self._base_url}/chat/completions"
        self._caller.post(self._caller.deadline(), url, _openai_headers(self._api_key), payload)

    def _complete(
        self, deadline: Deadline, workspace: Workspace, json_mode: bool, user: str
    ) -> str:
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
        reply = self._caller.post(deadline, url, _openai_headers(self._api_key), payload)
        return openai_text(reply, self._caller.provider)


def _openai_headers(api_key: str) -> dict[str, str]:
    """A local server without a key gets no Authorization header at all."""

    headers = {"Content-Type": "application/json", "User-Agent": "angularai/0.1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "User-Agent": "angularai/0.1.0",
    }


class Anthropic:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        transport: Transport | None = None,
        policy: Policy | None = None,
        clock: Clock | None = None,
        provider: str = "",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._caller = _Caller(
            transport=transport, policy=policy, clock=clock, provider=provider, secret=api_key
        )

    def billed(self) -> Usage:
        """Every token this substrate has been billed for."""

        return self._caller.usage

    def integrate(self, workspace: Workspace) -> Integration:
        deadline = self._caller.deadline()
        workspace = inline_exchange(workspace)
        text = self._complete(deadline, workspace.system, workspace.user)
        try:
            return parse_integration(text)
        except SubstrateUnreadable:
            extra = workspace.user + PARSE_NUDGE
            projected = estimate_tokens(workspace.system) + estimate_tokens(extra)
            if projected > workspace.budget:
                raise
            return parse_integration(self._complete(deadline, workspace.system, extra))

    def ping(self) -> None:
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        url = f"{self._base_url}/messages"
        self._caller.post(self._caller.deadline(), url, _anthropic_headers(self._api_key), payload)

    def _complete(self, deadline: Deadline, system: str, user: str) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        url = f"{self._base_url}/messages"
        reply = self._caller.post(deadline, url, _anthropic_headers(self._api_key), payload)
        return anthropic_text(reply, self._caller.provider)


def build_substrate(
    config: Config, environ: Mapping[str, str], *, vault: Vault | None = None
) -> Substrate:
    """The substrate a pass calls. See angularai.profiles for which model and key win."""

    config, profile = effective_config(config, environ)
    if config.substrate is None or config.model is None:
        raise InputRejected(
            "This cortex has no model. Set one up with: angularai setup, "
            "or wire one with: angularai wire openai --model <name>"
        )
    kind = config.substrate
    env_name = key_env(kind)
    key = environ.get(env_name, "").strip()
    if not key and profile is not None:
        key = (saved_key(profile, environ, vault) or "").strip()
        if not key and profile.storage != "none":
            raise KeyMissing(
                f"The key of saved model {profile.name} is gone. "
                f"Run angularai setup again, or set {env_name}."
            )
        if not key and kind != "compatible":
            raise KeyMissing(f"{env_name} is not set. Saved model {profile.name} keeps no key.")
    elif not key:
        raise KeyMissing(f"{env_name} is not set. The key stays in the environment.")
    return substrate_for(kind, config.model, config.base_url, key, policy_from_env(environ))


def substrate_for(
    kind: SubstrateKind,
    model: str,
    base_url: str | None,
    key: str,
    policy: Policy,
    *,
    json_mode: bool | None = None,
    transport: Transport | None = None,
    clock: Clock | None = None,
) -> "OpenAICompatible | Anthropic":
    json_on = kind == "openai" if json_mode is None else json_mode
    if kind == "openai":
        return OpenAICompatible(
            policy=policy,
            provider="OpenAI",
            model=model,
            base_url=base_url or OPENAI_URL,
            api_key=key,
            json_mode=json_on,
            completion_field="max_completion_tokens",
            transport=transport,
            clock=clock,
        )
    if kind == "anthropic":
        return Anthropic(
            policy=policy,
            provider="Anthropic",
            model=model,
            base_url=base_url or ANTHROPIC_URL,
            api_key=key,
            transport=transport,
            clock=clock,
        )
    if kind == "compatible":
        if base_url is None:
            raise InputRejected("A compatible substrate needs --base-url.")
        return OpenAICompatible(
            policy=policy,
            provider="compatible",
            model=model,
            base_url=base_url,
            api_key=key,
            json_mode=json_on,
            completion_field="max_tokens",
            transport=transport,
            clock=clock,
        )
    assert_never(kind)


OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
PROBE_POLICY = Policy(connect_timeout=5.0, read_timeout=60.0, deadline=60.0, attempts=1)


def probe(
    kind: SubstrateKind,
    model: str,
    base_url: str | None,
    key: str,
    *,
    policy: Policy = PROBE_POLICY,
    transport: Transport | None = None,
    clock: Clock | None = None,
) -> None:
    """One tiny call: a one-word prompt, a few tokens back. Raises the provider's error.

    Any 2xx JSON object passes: the key, the model, and the endpoint all answered.
    """

    substrate = substrate_for(kind, model, base_url, key, policy, transport=transport, clock=clock)
    substrate.ping()


def secret_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Every value in the environment whose variable name says it is a secret, and the
    saved key of the saved model in use, so a pass redacts it like any other."""

    found = environment_secrets(environ)
    try:
        key = _saved_in_use(environ)
    except AngularAIError:
        key = None
    if key and key not in found:
        return (*found, key)
    return found


def _saved_in_use(environ: Mapping[str, str]) -> str | None:
    from angularai.profiles import load_book

    book = load_book(environ)
    picked = environ.get("ANGULARAI_MODEL", "").strip() or book.active
    profile = None if picked is None else book.get(picked)
    if profile is None or profile.storage == "none":
        return None
    return saved_key(profile, environ)


def key_env(kind: SubstrateKind) -> str:
    return KEY_ENVS[kind]


def openai_text(payload: Mapping[str, object], provider: str = "") -> str:
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise SubstrateUnreadable(f"{_subject(provider)} {_NO_REPLY}")
    return content


def inline_exchange(workspace: Workspace) -> Workspace:
    """The workspace for the JSON act protocol: the last call and output in the text."""

    exchange = workspace.exchange
    if exchange is None:
        return workspace
    marker = token_hex(6)
    while marker in exchange.output:
        marker = token_hex(6)
    block = (
        f"\n\nLAST {exchange.tool} {exchange.arguments}\n"
        f"BEGIN TOOL OUTPUT {marker}\n{exchange.output}\nEND TOOL OUTPUT {marker}"
    )
    return replace(workspace, user=workspace.user + block, exchange=None)


def native_messages(workspace: Workspace) -> list[dict[str, object]]:
    """System, user, and the last tool call with its output as an assistant/tool pair."""

    messages: list[dict[str, object]] = [
        {"role": "system", "content": workspace.system},
        {"role": "user", "content": workspace.user},
    ]
    exchange = workspace.exchange
    if exchange is not None:
        call = {
            "id": "call_last",
            "type": "function",
            "function": {"name": exchange.tool, "arguments": exchange.arguments},
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        marker = token_hex(6)
        while marker in exchange.output:
            marker = token_hex(6)
        content = f"BEGIN TOOL OUTPUT {marker}\n{exchange.output}\nEND TOOL OUTPUT {marker}"
        messages.append({"role": "tool", "tool_call_id": "call_last", "content": content})
    return messages


def openai_message(payload: Mapping[str, object], provider: str = "") -> tuple[str, list[object]]:
    """The content and the tool calls of a chat completion. One of them must be there."""

    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, Mapping) else None
    if not isinstance(message, Mapping):
        raise SubstrateUnreadable(f"{_subject(provider)} {_NO_REPLY}")
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    raw = message.get("tool_calls")
    calls = list(raw) if isinstance(raw, list) else []
    if not text.strip() and not calls:
        raise SubstrateUnreadable(f"{_subject(provider)} {_NO_REPLY}")
    return text, calls


def anthropic_text(payload: Mapping[str, object], provider: str = "") -> str:
    blocks = payload.get("content")
    parts: list[str] = []
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, Mapping):
            continue
        text = block.get("text")
        if block.get("type") == "text" and isinstance(text, str):
            parts.append(text)
    reply = "".join(parts).strip()
    if not reply:
        raise SubstrateUnreadable(f"{_subject(provider)} {_NO_REPLY}")
    return reply


_BILLING = frozenset(
    {
        "insufficient_quota",
        "billing_not_active",
        "billing_error",
        "billing_hard_limit_reached",
        "payment_required",
        "credit_balance_too_low",
    }
)
_RATE = frozenset({"rate_limit_exceeded", "rate_limit_error", "rate_limited", "requests", "tokens"})
_OVERLOADED = frozenset({"overloaded_error", "overloaded", "server_overloaded"})
_MESSAGE_CAP = 200


def _subject(provider: str) -> str:
    return f"The substrate ({provider})" if provider else "The substrate"


def error_for_status(
    status: int,
    body: bytes,
    secret: str,
    headers: Mapping[str, str] | None = None,
    *,
    provider: str = "",
    now: float | None = None,
) -> SubstrateUnavailable | KeyRejected:
    """One sentence an operator can act on. The provider's words are redacted and capped.

    The class says what kind of failure it is and whether a retry may help.
    """

    the = _subject(provider)
    labels, raw_message = provider_error(body)
    message = _said(raw_message or _page_text(body), secret)
    billing = _billing(status, labels)
    if billing is not None:
        return QuotaExceeded(
            f"{the} refused the pass for billing: {billing}. "
            "Check the plan and credits for this key. Waiting will not help.",
            code=billing,
        )
    if status == 401:
        return AuthFailed(f"{the} rejected the API key.")
    if status == 403:
        return Forbidden(_with(f"{the} does not allow this key to do that.", message))
    if status == 429 or labels & _RATE:
        wait = retry_after(headers or {}, _now(now))
        if wait is not None:
            shown = max(1, math.ceil(wait))
            return RateLimited(f"{the} is rate limiting. Try again in {shown} seconds.", wait)
        return RateLimited(f"{the} is rate limiting. Wait and try again.")
    if status == 529 or labels & _OVERLOADED:
        return Overloaded(f"{the} is overloaded. Try again later.")
    if status == 404:
        return NotFound(_with(f"{the} returned 404. Check the model and the base URL.", message))
    if status >= 500:
        return ServerError(_with(f"{the} is unavailable.", message), status)
    return RequestRejected(_with(f"{the} returned {status}.", message), status)


def _now(now: float | None) -> float:
    return time.time() if now is None else now


def provider_message(body: bytes) -> str:
    return provider_error(body)[1]


def provider_error(body: bytes) -> tuple[frozenset[str], str]:
    """The error labels and message of an OpenAI, Anthropic, or compatible body.

    OpenAI: {"error": {"message", "type", "code"}}. Anthropic: {"type": "error",
    "error": {"type", "message"}}. A compatible server may send {"error": "text"},
    {"message": ...}, or {"detail": ...}. Anything else has no labels and no message.
    """

    try:
        payload: object = json.loads(body)
    except (ValueError, RecursionError):
        return frozenset(), ""
    if not isinstance(payload, dict):
        return frozenset(), ""
    labels: set[str] = set()
    message = ""
    error = payload.get("error")
    if isinstance(error, dict):
        for field in ("code", "type", "status"):
            value = error.get(field)
            if isinstance(value, str) and value.strip():
                labels.add(value.strip().lower())
        found = error.get("message")
        if isinstance(found, str):
            message = found
    elif isinstance(error, str):
        message = error
    if not message:
        for field in ("message", "detail"):
            found = payload.get(field)
            if isinstance(found, str):
                message = found
                break
    return frozenset(labels), " ".join(message.split())


_TAG = re.compile(r"<(script|style)\b.*?</\1\s*>|<[^>]*>", re.IGNORECASE | re.DOTALL)


def _page_text(body: bytes) -> str:
    """The readable text of a body that is not provider JSON: an HTML page or plain text."""

    text = body[: _MESSAGE_CAP * 20].decode("utf-8", errors="replace")
    if text.lstrip().startswith(("{", "[")):
        return ""
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def _said(message: str, secret: str) -> str:
    cleaned = redact(message, (secret,) if secret else ())
    if len(cleaned) > _MESSAGE_CAP:
        cleaned = cleaned[: _MESSAGE_CAP - 3] + "..."
    return cleaned


def _with(sentence: str, message: str) -> str:
    return f"{sentence} {message}" if message else sentence


def _billing(status: int, labels: frozenset[str]) -> str | None:
    for label in sorted(labels):
        if label in _BILLING or "billing" in label or "quota" in label:
            return label
    if status == 402:
        return "payment_required"
    return None


def _object(body: bytes, secret: str = "", provider: str = "") -> dict[str, object]:
    """The JSON object of a 2xx body. Anything else is BadResponse with a redacted snippet."""

    the = _subject(provider)
    if not body.strip():
        raise BadResponse(f"{the} returned an empty body. Nothing was written.")
    try:
        payload: object = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise BadResponse(
            _with(f"{the} returned something that is not JSON.", _snippet(body, secret))
        ) from exc
    if not isinstance(payload, dict):
        raise BadResponse(
            _with(f"{the} returned something that is not JSON.", _snippet(body, secret))
        )
    if "error" in payload and not ("choices" in payload or "content" in payload):
        _, said = provider_error(body)
        raise BadResponse(
            _with(f"{the} answered with an error and a success status.", _said(said, secret))
        )
    return {str(key): value for key, value in payload.items()}


def _snippet(body: bytes, secret: str) -> str:
    text = _page_text(body) or " ".join(body[: _MESSAGE_CAP * 2].decode("utf-8", "replace").split())
    said = _said(text, secret)
    return f"It began: {said}" if said else ""
