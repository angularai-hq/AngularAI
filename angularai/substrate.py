import html
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from typing import Protocol, assert_never

from angularai.budget import estimate_tokens
from angularai.errors import (
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
from angularai.parse import parse_integration
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
_PARSE_NUDGE = "\n\nReturn only the JSON object."
_NO_REPLY = "returned no reply. Nothing was written."


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
                return _object(body, self.secret, self.provider)
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

    def integrate(self, workspace: Workspace) -> Integration:
        deadline = self._caller.deadline()
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
            extra = workspace.user + _PARSE_NUDGE
            projected = estimate_tokens(workspace.system) + estimate_tokens(extra)
            if projected > workspace.budget:
                raise
            return parse_integration(self._complete(deadline, workspace, False, extra))

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
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "angularai/0.1.0",
        }
        reply = self._caller.post(deadline, url, headers, payload)
        return openai_text(reply, self._caller.provider)


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

    def integrate(self, workspace: Workspace) -> Integration:
        deadline = self._caller.deadline()
        text = self._complete(deadline, workspace.system, workspace.user)
        try:
            return parse_integration(text)
        except SubstrateUnreadable:
            extra = workspace.user + _PARSE_NUDGE
            projected = estimate_tokens(workspace.system) + estimate_tokens(extra)
            if projected > workspace.budget:
                raise
            return parse_integration(self._complete(deadline, workspace.system, extra))

    def _complete(self, deadline: Deadline, system: str, user: str) -> str:
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
        reply = self._caller.post(deadline, url, headers, payload)
        return anthropic_text(reply, self._caller.provider)


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
    policy = policy_from_env(environ)
    if kind == "openai":
        return OpenAICompatible(
            policy=policy,
            provider="OpenAI",
            model=config.model,
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=key,
            json_mode=config.json_mode,
            completion_field="max_completion_tokens",
        )
    if kind == "anthropic":
        return Anthropic(
            policy=policy,
            provider="Anthropic",
            model=config.model,
            base_url=config.base_url or "https://api.anthropic.com/v1",
            api_key=key,
        )
    if kind == "compatible":
        if config.base_url is None:
            raise InputRejected("A compatible substrate needs --base-url.")
        return OpenAICompatible(
            policy=policy,
            provider="compatible",
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


def openai_text(payload: Mapping[str, object], provider: str = "") -> str:
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise SubstrateUnreadable(f"{_subject(provider)} {_NO_REPLY}")
    return content


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
