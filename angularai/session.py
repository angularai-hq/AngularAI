from collections.abc import Mapping
from typing import Literal

from angularai.errors import AngularAIError
from angularai.present import format_locks, format_recall, format_status, format_traces
from angularai.profiles import effective_config, key_source
from angularai.store import Store
from angularai.substrate import key_env

SessionAction = Literal["quit", "continue", "speak", "clear"]

COMMANDS = (
    ("/clear", "clear the screen"),
    ("/cortex", "traces on disk"),
    ("/recall", "find without a model"),
    ("/status", "budget and the last pass"),
    ("/locks", "phrases a reply cannot contain"),
    ("/budget", "the workspace limit"),
    ("/help", "these commands"),
    ("/quit", "leave"),
)

HELP = "\n".join(f"{name:<10}{hint}" for name, hint in COMMANDS)
_CLEARED = "Screen cleared. The cortex is unchanged."


def session_command(store: Store, line: str, status_text: str) -> tuple[SessionAction, str | None]:
    """A slash command, a line to speak, or nothing. Speaking stays with the caller."""

    stripped = line.strip()
    if not stripped:
        return "continue", None
    if stripped in {"/quit", "/exit"}:
        return "quit", None
    if stripped in {"/clear", "/new"}:
        return "clear", _CLEARED
    if stripped == "/help":
        return "continue", HELP
    if stripped == "/cortex":
        return "continue", format_traces(store.view())
    if stripped == "/status":
        return "continue", status_text
    if stripped == "/locks":
        return "continue", format_locks(store.config().forbid)
    if stripped == "/budget":
        budget = store.config().workspace_budget
        return "continue", f"{budget} estimated tokens"
    if stripped.startswith("/recall"):
        query = stripped.removeprefix("/recall").strip()
        if not query:
            return "continue", "/recall needs a query."
        return "continue", format_recall(store.view(), query)
    if stripped.startswith("/"):
        return "continue", "Unknown command. /help lists them."
    return "speak", stripped


def cortex_status(store: Store, environ: Mapping[str, str]) -> str:
    stored = store.config()
    try:
        config, profile = effective_config(stored, environ)
    except AngularAIError:
        config, profile = stored, None
    key_name = key_env(config.substrate) if config.substrate is not None else None
    key_set = bool(key_name and environ.get(key_name, "").strip())
    text = format_status(
        root=store.root,
        config=config,
        view=store.view(),
        last=store.last_pass(),
        key_name=None if profile is not None else key_name,
        key_set=key_set,
    )
    if profile is None:
        return text
    head, _, rest = text.partition("\n")
    saved = f"saved model {profile.name}\nkey  {key_source(profile.provider, profile, environ)}"
    return f"{head}\n{saved}\n{rest}"
