"""Memory without a model: note a trace, rank traces for a query, forget one.

A note is a trace written by an operator or a tool, not by a model. It passes the
same gates as a trace the model writes: screen refuses a secret or an input over
the cap, parse's trace rules shape it, and the value lock judges it. It is one pass
in the log, with the note as its input and no reply.

The MCP server in packages/angularai-mcp is a thin layer over these functions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from angularai.attention import score
from angularai.budget import estimate_tokens
from angularai.errors import InputRejected, InputTooLong, LockHeld, TraceNotFound
from angularai.lock import ValueLock
from angularai.models import CortexView, PassRecord, Trace
from angularai.parse import trace_draft, valid_trace_id
from angularai.pilot import screen
from angularai.semantic import SemanticRecall, drop_vectors
from angularai.store import Store

MAX_NOTE = 240
MAX_WAVELENGTHS = 4
MAX_SOURCE = 500
Match = Literal["lexical", "meaning"]


@dataclass(frozen=True)
class Noted:
    trace: Trace
    created: bool


@dataclass(frozen=True)
class Forgot:
    trace: Trace
    vectors: int


@dataclass(frozen=True)
class Ranked:
    trace: Trace
    score: float
    match: Match


@dataclass(frozen=True)
class Recalled:
    """Traces in rank order whose texts fit max_tokens, and what that cost."""

    found: tuple[Ranked, ...]
    matched: int
    used: int
    max_tokens: int


def note(
    store: Store,
    text: str,
    wavelengths: Sequence[str] = (),
    secrets: tuple[str, ...] = (),
    source: str = "note",
    recall: SemanticRecall | None = None,
) -> Noted:
    """Write one trace. A current trace with the same text is returned, not written twice."""

    cleaned = screen(text, secrets)
    if len(" ".join(cleaned.split())) > MAX_NOTE:
        raise InputTooLong(f"A note is at most {MAX_NOTE} characters. Nothing was written.")
    if len(wavelengths) > MAX_WAVELENGTHS:
        raise InputRejected(f"A note has at most {MAX_WAVELENGTHS} tags. Nothing was written.")
    draft = trace_draft(cleaned, wavelengths)
    if draft is None:
        raise InputRejected(
            "A tag is a lowercase slug: a letter, then up to 31 letters, digits, or -. "
            "Nothing was written."
        )
    origin = " ".join(source.split())[:MAX_SOURCE] or "note"
    config = store.config()
    lock = ValueLock()
    for part in (draft.text, *draft.wavelengths, origin):
        if not lock.judge(part, config.forbid, secrets).allowed:
            raise LockHeld("Value lock held that note. Nothing was written.")
    view = store.view()
    for existing in view.current_traces():
        if existing.text == draft.text:
            return Noted(existing, False)
    now = f"{datetime.now(UTC).isoformat()}-{uuid4().hex[:4]}"
    trace = Trace(
        id="t_" + uuid4().hex[:12],
        text=draft.text,
        wavelengths=draft.wavelengths,
        source=origin,
        at=now,
        state="current",
        superseded_by=None,
    )
    store.append_pass(
        PassRecord(
            at=now,
            outcome="ok",
            user=draft.text,
            reply=None,
            remember=(trace,),
            supersede=(),
            error=None,
            budget=config.workspace_budget,
            used=0,
            traces_shown=0,
            traces_held=len(view.current_traces()),
            episodes_shown=0,
            episodes_held=len(view.episodes),
        )
    )
    if recall is not None:
        recall.index((trace,))
    return Noted(trace, True)


def rank(
    view: CortexView,
    query: str,
    meaning: Mapping[str, float] | None = None,
    *,
    include_superseded: bool = False,
) -> tuple[Ranked, ...]:
    """Every trace that matches, best first. Lexical when it shares a word, else by meaning.

    This is the order angularai recall prints. Ties keep the order on disk.
    """

    similar = meaning or {}
    pool = view.traces if include_superseded else view.current_traces()
    found: list[Ranked] = []
    for trace in pool:
        document = " ".join((trace.text, " ".join(trace.wavelengths), trace.source))
        lexical = score(query, document)
        if lexical > 0:
            found.append(Ranked(trace, lexical, "lexical"))
        elif trace.id in similar:
            found.append(Ranked(trace, similar[trace.id], "meaning"))
    found.sort(key=lambda item: item.score, reverse=True)
    return tuple(found)


def recall_within(
    view: CortexView,
    query: str,
    max_tokens: int,
    limit: int,
    meaning: Mapping[str, float] | None = None,
    *,
    include_superseded: bool = False,
) -> Recalled:
    """The best traces, at most limit, whose texts together fit max_tokens.

    A trace that does not fit is skipped and a smaller one after it may still be taken,
    as the workspace is packed.
    """

    ranked = rank(view, query, meaning, include_superseded=include_superseded)
    taken: list[Ranked] = []
    used = 0
    for item in ranked:
        if len(taken) >= limit:
            break
        cost = estimate_tokens(item.trace.text)
        if used + cost > max_tokens:
            continue
        taken.append(item)
        used += cost
    return Recalled(tuple(taken), len(ranked), used, max_tokens)


def forget(store: Store, trace_id: str) -> Forgot:
    """Remove the trace with exactly this id, from the log and from embeddings.jsonl."""

    if not valid_trace_id(trace_id):
        raise TraceNotFound(
            "A trace id is t_ and 12 lowercase hex digits, exactly as recall shows it. "
            "Nothing was removed."
        )
    gone = store.forget(trace_id)
    return Forgot(gone, drop_vectors(store.root, frozenset({trace_id})))
