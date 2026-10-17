from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from angularai.errors import InputRejected
from angularai.models import Action
from angularai.pilot import pass_turn
from angularai.store import Store
from angularai.substrate import Substrate
from angularai.tools import execute

_DEFAULT_STEPS = 8
_MAX_STEPS = 24
_LAST_CHARS = 1400

AGENT_SYSTEM = (
    "You are AngularAI, working inside one directory. "
    "This workspace is the only context for this pass. "
    "TRACES are what you know. RECENT is the part of the work that still fits. "
    "TASK is the job. LAST is the previous tool, when there is one. "
    "Use one tool in act, or null when the task is done. "
    "Tools: list path; read path; search query and path; "
    "edit path old new; write path text; run command. "
    "run has no shell. Stay inside the directory. "
    "remember holds facts the next pass would lose. "
    "Do not remember questions, guesses, API keys, or private keys. "
    "Reply with one JSON object and nothing else: "
    '{"reply":"...","act":null,"remember":[{"text":"...","wavelengths":["slug"]}],'
    '"supersede":[]}. '
    'An act looks like {"tool":"read","path":"src/app.py"}.'
)

Stop = Literal["done", "budget", "held"]


@dataclass(frozen=True)
class AgentResult:
    reply: str
    steps: int
    stop: Stop


def run_agent(
    store: Store,
    substrate: Substrate,
    task: str,
    root: Path,
    secrets: tuple[str, ...] = (),
    steps: int = _DEFAULT_STEPS,
    on_step: Callable[[str], None] | None = None,
    allow_run: bool = False,
) -> AgentResult:
    if steps < 1 or steps > _MAX_STEPS:
        raise InputRejected(f"Steps must be from 1 to {_MAX_STEPS}.")
    directory = root.expanduser().resolve()
    if not directory.is_dir():
        raise InputRejected("The agent root is not a directory.")
    last: str | None = None
    reply = ""
    for index in range(1, steps + 1):
        result = pass_turn(
            store,
            substrate,
            _step_input(task, index, steps, last),
            secrets,
            AGENT_SYSTEM,
        )
        reply = result.reply
        if result.held or result.act is None:
            stop: Stop = "held" if result.held else "done"
            return AgentResult(reply, index, stop)
        if on_step is not None:
            on_step(f"{index}  {_detail(result.act)}")
        last = f"{_detail(result.act)}\n{execute(directory, result.act, secrets, allow_run)}"
    return AgentResult(reply, steps, "budget")


def _step_input(task: str, index: int, steps: int, last: str | None) -> str:
    body = f"TASK\n{task.strip()}"
    if last is None:
        return body
    clipped = last if len(last) <= _LAST_CHARS else last[: _LAST_CHARS - 3] + "..."
    return f"{body}\n\nSTEP {index} of {steps}\n{clipped}"


def _detail(action: Action) -> str:
    if action.tool == "search":
        target = action.query or action.path
    elif action.tool == "run":
        target = action.command
    elif action.tool == "write":
        target = action.path
    else:
        target = action.path or action.command
    flat = " ".join(target.split())
    if len(flat) > 80:
        flat = flat[:77] + "..."
    return f"{action.tool}  {flat}".rstrip()
