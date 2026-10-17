from datetime import UTC, datetime
from uuid import uuid4

from angularai.assemble import SYSTEM, assemble
from angularai.errors import InputRejected, SubstrateError
from angularai.lock import Thinker, ValueLock
from angularai.models import (
    Integration,
    Outcome,
    PassRecord,
    Trace,
    TraceDraft,
    TurnResult,
    Workspace,
)
from angularai.secrets import contains_secret, redact
from angularai.store import Store
from angularai.substrate import Substrate

_MAX_INPUT_CHARS = 20000


def pass_turn(
    store: Store,
    substrate: Substrate,
    user_text: str,
    secrets: tuple[str, ...] = (),
    system: str | None = None,
) -> TurnResult:
    text = user_text.strip()
    if not text:
        raise InputRejected("Say something first.")
    if len(text) > _MAX_INPUT_CHARS:
        raise InputRejected("That input is too long to enter the cortex.")
    if contains_secret(text, secrets):
        raise InputRejected(
            "That input looks like a secret. It was not sent and nothing was written."
        )

    config = store.config()
    prompt = SYSTEM if system is None else system
    workspace = assemble(store.view(), text, config, prompt)
    try:
        integration = substrate.integrate(workspace)
    except SubstrateError as exc:
        store.append_pass(_error_pass(text, workspace, exc, secrets))
        raise

    lock = ValueLock()
    verdict = lock.judge(integration.reply, config.forbid, secrets)
    reply = Thinker(lock).release(verdict)
    if not verdict.allowed:
        store.append_pass(_held_pass(text, reply, workspace, secrets))
        return TurnResult(reply, True, workspace.estimated_tokens, workspace.budget, None)

    drafts = tuple(
        draft for draft in integration.remember if not contains_secret(draft.text, secrets)
    )
    store.append_pass(_ok_pass(text, reply, drafts, integration, workspace, secrets))
    return TurnResult(
        reply, False, workspace.estimated_tokens, workspace.budget, integration.act
    )


def _ok_pass(
    text: str,
    reply: str,
    drafts: tuple[TraceDraft, ...],
    integration: Integration,
    workspace: Workspace,
    secrets: tuple[str, ...],
) -> PassRecord:
    now = _now()
    stored_user = _stored_user(text, secrets)
    traces = tuple(
        Trace(
            id=_trace_id(),
            text=draft.text,
            wavelengths=draft.wavelengths,
            source=stored_user[:500],
            at=now,
            state="current",
            superseded_by=None,
        )
        for draft in drafts
    )
    visible = set(workspace.trace_ids)
    supersede = tuple((trace_id, now) for trace_id in integration.supersede if trace_id in visible)
    return _pass(
        now,
        "ok",
        stored_user,
        reply,
        traces,
        supersede,
        None,
        workspace,
    )


def _held_pass(text: str, reply: str, workspace: Workspace, secrets: tuple[str, ...]) -> PassRecord:
    return _pass(
        _now(),
        "held",
        _stored_user(text, secrets),
        reply,
        (),
        (),
        None,
        workspace,
    )


def _error_pass(
    text: str,
    workspace: Workspace,
    exc: SubstrateError,
    secrets: tuple[str, ...],
) -> PassRecord:
    message = redact(str(exc), secrets)
    message = " ".join(message.split())[:300]
    return _pass(
        _now(),
        "error",
        _stored_user(text, secrets),
        None,
        (),
        (),
        message,
        workspace,
    )


def _pass(
    at: str,
    outcome: Outcome,
    user: str,
    reply: str | None,
    remember: tuple[Trace, ...],
    supersede: tuple[tuple[str, str], ...],
    error: str | None,
    workspace: Workspace,
) -> PassRecord:
    return PassRecord(
        at=at,
        outcome=outcome,
        user=user,
        reply=reply,
        remember=remember,
        supersede=supersede,
        error=error,
        budget=workspace.budget,
        used=workspace.estimated_tokens,
        traces_shown=workspace.traces_shown,
        traces_held=workspace.traces_held,
        episodes_shown=workspace.episodes_shown,
        episodes_held=workspace.episodes_held,
    )


def _stored_user(text: str, secrets: tuple[str, ...]) -> str:
    stored = redact(text, secrets).strip()
    if not stored:
        return "[redacted]"
    return stored


def _now() -> str:
    return f"{datetime.now(UTC).isoformat()}-{uuid4().hex[:4]}"


def _trace_id() -> str:
    return "t_" + uuid4().hex[:12]
