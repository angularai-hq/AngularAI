from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from angularai.assemble import SYSTEM, assemble
from angularai.errors import InputRejected, InputTooLong, SecretRefused, SubstrateError
from angularai.lock import Thinker, ValueLock
from angularai.models import (
    Action,
    Integration,
    Outcome,
    PassRecord,
    Peer,
    Trace,
    TraceDraft,
    TurnResult,
    Workspace,
)
from angularai.secrets import contains_secret, redact
from angularai.semantic import SemanticRecall
from angularai.store import Store
from angularai.substrate import Substrate

_MAX_INPUT_CHARS = 20000


def pass_turn(
    store: Store,
    substrate: Substrate,
    user_text: str,
    secrets: tuple[str, ...] = (),
    system: str | None = None,
    record: str | None = None,
    recall: SemanticRecall | None = None,
    peer: Peer | None = None,
) -> TurnResult:
    """One pass. record, when given, is what the log keeps in place of the input.

    recall, when given, adds traces that match by meaning. It never stops the pass.
    peer, when given, is written with the pass. When the input came from the peer
    (answered), a trace names the peer as its source, and nothing is superseded:
    another cortex can add to what this one knows, never make it forget.
    """

    text = screen(user_text, secrets)

    config = store.config()
    prompt = SYSTEM if system is None else system
    view = store.view()
    meaning = None if recall is None else recall.ranks(view, text)
    workspace = assemble(view, text, config, prompt, meaning)
    try:
        integration = substrate.integrate(workspace)
    except SubstrateError as exc:
        failed = _error_pass(text if record is None else record, workspace, exc, secrets)
        store.append_pass(replace(failed, peer=peer))
        raise

    kept = text if record is None else record
    lock = ValueLock()
    act = () if integration.act is None else _act_texts(integration.act)
    verdict = lock.judge(integration.reply, config.forbid, secrets, act)
    reply = Thinker(lock).release(verdict)
    if not verdict.allowed:
        store.append_pass(replace(_held_pass(kept, reply, workspace, secrets), peer=peer))
        return TurnResult(reply, True, workspace.estimated_tokens, workspace.budget, None)

    # A trace is judged like the reply: a secret or a locked phrase is never remembered.
    drafts = tuple(
        draft
        for draft in integration.remember
        if lock.judge(draft.text, config.forbid, secrets).allowed
    )
    if peer is not None and peer.role == "answered":
        integration = replace(integration, supersede=())
    written = _ok_pass(kept, reply, drafts, integration, workspace, secrets, peer)
    store.append_pass(written)
    if recall is not None:
        recall.index(written.remember)
    return TurnResult(reply, False, workspace.estimated_tokens, workspace.budget, integration.act)


def screen(user_text: str, secrets: tuple[str, ...] = ()) -> str:
    """Refuse input before anything else runs. No key, no model, and no disk are needed."""

    text = user_text.strip()
    if not text:
        raise InputRejected("Say something first.")
    if len(text) > _MAX_INPUT_CHARS:
        raise InputTooLong("That input is too long to enter the cortex.")
    if contains_secret(text, secrets):
        raise SecretRefused(
            "That input looks like a secret. It was not sent and nothing was written."
        )
    return text


def _act_texts(act: Action) -> tuple[str, ...]:
    return tuple(
        part for part in (act.path, act.query, act.old, act.new, act.text, act.command) if part
    )


def _ok_pass(
    text: str,
    reply: str,
    drafts: tuple[TraceDraft, ...],
    integration: Integration,
    workspace: Workspace,
    secrets: tuple[str, ...],
    peer: Peer | None = None,
) -> PassRecord:
    now = _now()
    stored_user = _stored_user(text, secrets)
    source = stored_user
    if peer is not None and peer.role == "answered":
        source = f"cortex {peer.name}: {stored_user}"
    traces = tuple(
        Trace(
            id=_trace_id(),
            text=draft.text,
            wavelengths=draft.wavelengths,
            source=source[:500],
            at=now,
            state="current",
            superseded_by=None,
        )
        for draft in drafts
    )
    visible = set(workspace.trace_ids)
    supersede = tuple((trace_id, now) for trace_id in integration.supersede if trace_id in visible)
    written = _pass(
        now,
        "ok",
        stored_user,
        reply,
        traces,
        supersede,
        None,
        workspace,
    )
    return replace(written, peer=peer)


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
