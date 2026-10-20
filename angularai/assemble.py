import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from angularai.attention import score
from angularai.budget import estimate_tokens
from angularai.errors import InputDoesNotFit
from angularai.models import Config, CortexView, Episode, Trace, Workspace

SYSTEM = (
    "You are AngularAI. This workspace is the only context for this pass. "
    "TRACES are what you know. RECENT is the part of the conversation that still fits. "
    "Answer INPUT. "
    "remember holds every durable fact the next pass would lose, one fact each, "
    "including identifiers, replacements, phrases, and constraints. "
    "supersede holds trace ids from TRACES that this input contradicts. "
    "Do not remember questions, guesses, API keys, or private keys. "
    "Reply with one JSON object and nothing else: "
    '{"reply":"...","remember":[{"text":"...","wavelengths":["slug"]}],"supersede":["id"]}'
)


@dataclass(frozen=True)
class _Candidate:
    kind: Literal["trace", "episode"]
    rank: float
    at: str
    line: str
    trace_id: str | None


def assemble(
    view: CortexView,
    user_text: str,
    config: Config,
    system: str = SYSTEM,
    meaning: Mapping[str, float] | None = None,
) -> Workspace:
    """Pack L2. Anything that does not fit stays on disk and is not sent.

    meaning maps a trace id to a rank by meaning (a cosine). A trace ranks by the
    higher of that and its lexical score. Either way it is sent only if it fits.
    """

    text = user_text.strip()
    if estimate_tokens(text) > config.max_input_tokens:
        raise InputDoesNotFit(
            "That input does not fit the workspace. "
            f"The input cap is {config.max_input_tokens} estimated tokens."
        )

    head = lock_block(config.forbid) + "\n\nTRACES\n"
    recent = "\nRECENT\n"
    tail = f"\nINPUT\n{text}"

    def used(trace_lines: list[str], episode_lines: list[str]) -> int:
        body = head + "".join(trace_lines) + recent + "".join(episode_lines) + tail
        return estimate_tokens(system) + estimate_tokens(body)

    if used([], []) > config.workspace_budget:
        raise InputDoesNotFit(
            f"That input does not fit the workspace budget of {config.workspace_budget}. "
            "Shorten it, or raise the budget with angularai budget."
        )

    ranked = sorted(
        _candidates(view, text, meaning or {}), key=lambda item: (item.rank, item.at), reverse=True
    )
    chosen_traces: list[_Candidate] = []
    chosen_episodes: list[_Candidate] = []
    for candidate in ranked:
        if candidate.rank <= 0:
            continue
        trial_traces = [*chosen_traces, candidate] if candidate.kind == "trace" else chosen_traces
        trial_episodes = (
            [*chosen_episodes, candidate] if candidate.kind == "episode" else chosen_episodes
        )
        if used(_lines(trial_traces), _lines(trial_episodes)) <= config.workspace_budget:
            if candidate.kind == "trace":
                chosen_traces.append(candidate)
            else:
                chosen_episodes.append(candidate)

    chosen_episodes = _with_continuity(
        view, chosen_episodes, chosen_traces, used, config.workspace_budget, text
    )
    episode_lines = _lines(sorted(chosen_episodes, key=lambda item: item.at))
    trace_lines = _lines(chosen_traces)
    body = head + "".join(trace_lines) + recent + "".join(episode_lines) + tail
    total = estimate_tokens(system) + estimate_tokens(body)
    shown_episodes = {item.at for item in chosen_episodes}
    return Workspace(
        system=system,
        user=body,
        input_text=text,
        trace_ids=tuple(item.trace_id for item in chosen_traces if item.trace_id is not None),
        estimated_tokens=total,
        budget=config.workspace_budget,
        traces_shown=len(chosen_traces),
        traces_held=len(view.current_traces()) - len(chosen_traces),
        episodes_shown=len(shown_episodes),
        episodes_held=len(view.episodes) - len(shown_episodes),
    )


def _with_continuity(
    view: CortexView,
    chosen_episodes: list[_Candidate],
    chosen_traces: list[_Candidate],
    used: Callable[[list[str], list[str]], int],
    budget: int,
    query: str,
) -> list[_Candidate]:
    if not view.episodes:
        return chosen_episodes
    newest = max(view.episodes, key=lambda episode: episode.at)
    if any(item.at == newest.at for item in chosen_episodes):
        return chosen_episodes
    candidate = _episode_candidate(newest, 0.0, query)
    trial = [*chosen_episodes, candidate]
    if used(_lines(chosen_traces), _lines(trial)) <= budget:
        return trial
    return chosen_episodes


def _candidates(
    view: CortexView, query: str, meaning: Mapping[str, float]
) -> tuple[_Candidate, ...]:
    items: list[_Candidate] = []
    for trace in view.current_traces():
        document = " ".join((trace.text, " ".join(trace.wavelengths), trace.source))
        rank = max(score(query, document), meaning.get(trace.id, 0.0))
        items.append(_Candidate("trace", rank, trace.at, _trace_line(trace), trace.id))
    for episode in view.episodes:
        items.append(_episode_candidate(episode, score(query, _episode_document(episode)), query))
    return tuple(items)


def _episode_candidate(episode: Episode, rank: float, query: str) -> _Candidate:
    return _Candidate("episode", rank, episode.at, _episode_line(episode, query), None)


def _episode_document(episode: Episode) -> str:
    if episode.reply is None:
        return episode.user
    return f"{episode.user}\n{episode.reply}"


def _lines(items: list[_Candidate]) -> list[str]:
    return [item.line for item in items]


def lock_block(forbid: tuple[str, ...]) -> str:
    if not forbid:
        return "VALUE LOCK\nnone"
    lines = "\n".join(f"- {phrase}" for phrase in forbid)
    return f"VALUE LOCK\n{lines}"


def _trace_line(trace: Trace) -> str:
    waves = ",".join(trace.wavelengths) if trace.wavelengths else "-"
    return f"{trace.id}  {waves}  {_clip(trace.text, 240)}\n"


def _episode_line(episode: Episode, query: str) -> str:
    asker, answerer = speakers(episode)
    you = f"{asker}  {focus(episode.user, query, 700)}\n"
    if episode.reply is None:
        return you
    return you + f"{answerer}  {focus(episode.reply, query, 400)}\n"


def speakers(episode: Episode) -> tuple[str, str]:
    """Who said each half. A peer is named, so its words are never taken for yours."""

    peer = episode.peer
    if peer is None or peer.role == "opened":
        return "you", "angularai"
    if peer.role == "asked":
        return "you", f"cortex {peer.name}"
    return f"cortex {peer.name}", "angularai"


def focus(text: str, query: str, limit: int) -> str:
    """The slice of a long episode that overlaps the question."""

    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", flat) if part]
    if not sentences or not any(score(query, sentence) > 0 for sentence in sentences):
        return _clip(flat, limit)
    chosen: list[int] = []
    used = 0
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (score(query, sentences[index]), -index),
        reverse=True,
    )
    for index in ranked:
        if score(query, sentences[index]) <= 0:
            break
        piece = sentences[index]
        extra = len(piece) + (1 if used else 0)
        if used + extra > limit:
            if not chosen:
                return _window(piece, query, limit)
            continue
        chosen.append(index)
        used += extra
    chosen.sort()
    return " ".join(sentences[index] for index in chosen)


def _window(text: str, query: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    best = text[:limit]
    best_score = score(query, best)
    step = max(40, limit // 4)
    start = 0
    while start < len(text):
        piece = text[start : start + limit]
        rank = score(query, piece)
        if rank > best_score:
            best = piece
            best_score = rank
        if start + limit >= len(text):
            break
        start += step
    return _clip(best.strip(), limit)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
