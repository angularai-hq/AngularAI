"""The v1 and v2 arms: the history goes into a real v1 cortex, and the question is one v1 pass.

Each user turn and the assistant turn after it become one episode in passes.jsonl.
Nothing is sent to the model while the history is written: no model-written traces,
so recall is v1's lexical recall over episodes. The question then runs through
angularai.pilot.pass_turn at the given workspace budget, exactly as `angularai say`.
The v2 arm is the same pass with a cortex v2 index (angularai.index): the log is
indexed on the first sync, as a v1 cortex migrates, with local vectors when an
embedder is given.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from angularai.evals.dataset import Question, Turn
from angularai.index import CortexIndex, task_prefixes
from angularai.models import PassRecord
from angularai.pilot import pass_turn
from angularai.semantic import Embedder
from angularai.store import default_config, init_cortex, open_store
from angularai.substrate import Substrate

_USER_LIMIT = 20000
_REPLY_LIMIT = 8000
_DATE = "%Y/%m/%d (%a) %H:%M"


@dataclass(frozen=True)
class V1Answer:
    reply: str
    used: int
    budget: int
    episodes: int
    episodes_shown: int
    chunks: int = 0


def at(date: str, step: int) -> str:
    """An ISO time for turn step of a session dated date. Order within a session holds."""

    try:
        moment = datetime.strptime(date, _DATE)
    except ValueError:
        moment = datetime(2000, 1, 1)
    return (moment + timedelta(seconds=step)).isoformat()


def episodes(question: Question) -> tuple[PassRecord, ...]:
    records: list[PassRecord] = []
    for session in question.sessions:
        for step, (user, reply) in enumerate(_pairs(session.turns)):
            records.append(
                PassRecord(
                    at=at(session.date, step),
                    outcome="ok",
                    user=_clip(user, _USER_LIMIT) or "(no user turn)",
                    reply=_clip(reply, _REPLY_LIMIT) if reply else None,
                    remember=(),
                    supersede=(),
                    error=None,
                    budget=0,
                    used=0,
                    traces_shown=0,
                    traces_held=0,
                    episodes_shown=0,
                    episodes_held=0,
                )
            )
    return tuple(records)


def v1_input(question: Question) -> str:
    return f"Current Date: {question.date}\nQuestion: {question.question}"


def answer_v1(
    root: Path,
    question: Question,
    substrate: Substrate,
    *,
    budget: int,
    model: str,
    base_url: str,
    v2: bool = False,
    embedder: Embedder | None = None,
    context: int = 0,
) -> V1Answer:
    init_cortex(root)
    store = open_store(root)
    base = default_config()
    store.save_config(
        replace(
            base,
            max_input_tokens=min(base.max_input_tokens, budget),
            substrate="compatible",
            model=model,
            base_url=base_url,
            workspace_budget=budget,
            json_mode=True,
        )
    )
    written = episodes(question)
    for record in written:
        store.append_pass(record)
    index = None
    if v2:
        document, query = task_prefixes(embedder.model if embedder is not None else "")
        index = CortexIndex(
            root, embedder, document_prefix=document, query_prefix=query, context=context
        )
    try:
        result = pass_turn(store, substrate, v1_input(question), index=index)
        chunks = 0 if index is None else index.chunks()
    finally:
        if index is not None:
            index.close()
    last = store.last_pass()
    shown = 0 if last is None else last.episodes_shown
    return V1Answer(
        reply=result.reply,
        used=result.used,
        budget=result.budget,
        episodes=len(written),
        episodes_shown=shown,
        chunks=chunks,
    )


def _pairs(turns: tuple[Turn, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for turn in turns:
        if turn.role == "user":
            if pending is not None:
                pairs.append((pending, ""))
            pending = turn.content.strip()
        else:
            pairs.append((pending or "", turn.content.strip()))
            pending = None
    if pending is not None:
        pairs.append((pending, ""))
    return [(user, reply) for user, reply in pairs if user or reply]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]
