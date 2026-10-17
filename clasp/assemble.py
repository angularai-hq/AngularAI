from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from clasp.attention import score
from clasp.budget import estimate_tokens
from clasp.errors import InputDoesNotFit
from clasp.models import Config, CortexView, Episode, Trace, Workspace

SYSTEM = (
    "You are Clasp. This workspace is the only context for this pass. "
    "TRACES are what you know. RECENT is the part of the conversation that still fits. "
    "Answer INPUT. "
    "remember holds durable facts the next pass would lose, in plain words, one fact each. "
    "supersede holds trace ids from TRACES that this input contradicts. "
    "Do not remember questions, guesses, or secrets. "
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


def assemble(view: CortexView, user_text: str, config: Config) -> Workspace:
    """Pack L2. Anything that does not fit stays on disk and is not sent."""

    text = user_text.strip()
    if estimate_tokens(text) > config.max_input_tokens:
        raise InputDoesNotFit(
            "That input does not fit the workspace. "
            f"The input cap is {config.max_input_tokens} estimated tokens."
        )

    head = _lock_block(config.forbid) + "\n\nTRACES\n"
    recent = "\nRECENT\n"
    tail = f"\nINPUT\n{text}"

    def used(trace_lines: list[str], episode_lines: list[str]) -> int:
        body = head + "".join(trace_lines) + recent + "".join(episode_lines) + tail
        return estimate_tokens(SYSTEM) + estimate_tokens(body)

    if used([], []) > config.workspace_budget:
        raise InputDoesNotFit(
            f"That input does not fit the workspace budget of {config.workspace_budget}. "
            "Shorten it, or raise the budget with clasp budget."
        )

    ranked = sorted(_candidates(view, text), key=lambda item: (item.rank, item.at), reverse=True)
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
        view, chosen_episodes, chosen_traces, used, config.workspace_budget
    )
    episode_lines = _lines(sorted(chosen_episodes, key=lambda item: item.at))
    trace_lines = _lines(chosen_traces)
    body = head + "".join(trace_lines) + recent + "".join(episode_lines) + tail
    total = estimate_tokens(SYSTEM) + estimate_tokens(body)
    shown_episodes = {item.at for item in chosen_episodes}
    return Workspace(
        system=SYSTEM,
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
) -> list[_Candidate]:
    if not view.episodes:
        return chosen_episodes
    newest = max(view.episodes, key=lambda episode: episode.at)
    if any(item.at == newest.at for item in chosen_episodes):
        return chosen_episodes
    candidate = _episode_candidate(newest, 0.0)
    trial = [*chosen_episodes, candidate]
    if used(_lines(chosen_traces), _lines(trial)) <= budget:
        return trial
    return chosen_episodes


def _candidates(view: CortexView, query: str) -> tuple[_Candidate, ...]:
    items: list[_Candidate] = []
    for trace in view.current_traces():
        document = " ".join((trace.text, " ".join(trace.wavelengths), trace.source))
        items.append(
            _Candidate("trace", score(query, document), trace.at, _trace_line(trace), trace.id)
        )
    for episode in view.episodes:
        items.append(_episode_candidate(episode, score(query, _episode_document(episode))))
    return tuple(items)


def _episode_candidate(episode: Episode, rank: float) -> _Candidate:
    return _Candidate("episode", rank, episode.at, _episode_line(episode), None)


def _episode_document(episode: Episode) -> str:
    if episode.reply is None:
        return episode.user
    return f"{episode.user}\n{episode.reply}"


def _lines(items: list[_Candidate]) -> list[str]:
    return [item.line for item in items]


def _lock_block(forbid: tuple[str, ...]) -> str:
    if not forbid:
        return "VALUE LOCK\nnone"
    lines = "\n".join(f"- {phrase}" for phrase in forbid)
    return f"VALUE LOCK\n{lines}"


def _trace_line(trace: Trace) -> str:
    waves = ",".join(trace.wavelengths) if trace.wavelengths else "-"
    return f"{trace.id}  {waves}  {_one_line(trace.text, 240)}\n"


def _episode_line(episode: Episode) -> str:
    you = f"you  {_one_line(episode.user, 400)}\n"
    if episode.reply is None:
        return you
    return you + f"clasp  {_one_line(episode.reply, 400)}\n"


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."
