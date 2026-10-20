"""The setup and model screens as pure state: keys in, state and one effect out.

Nothing here draws or does I/O. ``setup_step`` and ``manage_step`` are reducers:
each takes a state and an event and returns the next state and, at most, one
effect for the caller to run (a test call, a model listing, a save). ``render_*``
turn a state into lines of styled segments. The Textual layer in
``angularai.setup_tui`` only maps keys to events, runs effects, and paints lines.

The key lives in ``SetupState.key`` and nowhere else. It is left out of the repr,
and every rendering shows bullets while it is typed and ``mask`` after.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from angularai.errors import InputRejected
from angularai.models import SubstrateKind
from angularai.onboard import CHOICES, OLLAMA_URL, Choice
from angularai.profiles import KEY_ENVS, check_key, check_model, check_url, mask

Step = Literal["provider", "base_url", "key", "model", "test", "saved", "cancelled"]
Test = Literal["running", "ok", "failed", "saving"]
Role = Literal["text", "dim", "accent", "selected", "ok", "error", "title", "field"]
Segment = tuple[str, Role]
Line = tuple[Segment, ...]

SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-5-mini", "gpt-5", "gpt-4.1-mini", "gpt-4o-mini"),
    "anthropic": ("claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"),
    "ollama": ("llama3.2", "qwen2.5", "mistral"),
    "compatible": (),
}
BULLET = "•"
_MAX_BULLETS = 32
_MAX_FIELD = 4096
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# --- events and effects -----------------------------------------------------


@dataclass(frozen=True)
class Key:
    """A key: its name ("up", "enter", "ctrl+c") and the character it types, if any."""

    name: str
    char: str = ""


@dataclass(frozen=True)
class Paste:
    text: str = field(repr=False)


@dataclass(frozen=True)
class Fetched:
    models: tuple[str, ...]


@dataclass(frozen=True)
class Probed:
    ok: bool
    message: str = ""
    seconds: float = 0.0


@dataclass(frozen=True)
class Saved:
    name: str
    where: str


@dataclass(frozen=True)
class SaveFailed:
    message: str


Event = Key | Paste | Fetched | Probed | Saved | SaveFailed


@dataclass(frozen=True)
class FetchModels:
    choice: Choice
    base_url: str
    key: str = field(repr=False)


@dataclass(frozen=True)
class Probe:
    choice: Choice
    model: str
    base_url: str | None
    key: str = field(repr=False)


@dataclass(frozen=True)
class Save:
    choice: Choice
    model: str
    base_url: str | None
    key: str | None = field(repr=False)


@dataclass(frozen=True)
class Finish:
    """Leave the screen: saved, or cancelled."""

    saved: bool


Effect = FetchModels | Probe | Save | Finish


# --- setup ------------------------------------------------------------------


@dataclass(frozen=True)
class SetupState:
    step: Step = "provider"
    cursor: int = 0
    base_url: str = ""
    key: str = field(default="", repr=False)
    models: tuple[str, ...] = ()
    model_cursor: int = 0
    custom: str = ""
    fetching: bool = False
    listed: bool = False
    model: str = ""
    test: Test = "running"
    message: str = ""
    seconds: float = 0.0
    option: int = 0
    error: str = ""
    env_kinds: frozenset[SubstrateKind] = frozenset()
    saved_as: str = ""
    saved_where: str = ""
    spin: int = 0

    @property
    def choice(self) -> Choice:
        return CHOICES[self.cursor]

    @property
    def env_key(self) -> bool:
        return self.choice.kind in self.env_kinds

    @property
    def options(self) -> tuple[str, ...]:
        if self.test == "ok":
            return ("Save", "Edit", "Cancel")
        if self.test == "failed":
            return ("Retry", "Edit", "Save anyway", "Cancel")
        return ()

    @property
    def model_rows(self) -> int:
        """Listed models, then the free-text row."""

        return len(self.models) + 1

    @property
    def on_custom(self) -> bool:
        return self.model_cursor >= len(self.models)


def start(env_kinds: frozenset[SubstrateKind] = frozenset()) -> SetupState:
    return SetupState(env_kinds=env_kinds)


def setup_step(state: SetupState, event: Event) -> tuple[SetupState, Effect | None]:
    """The next state, and the effect the caller must run."""

    if isinstance(event, Key) and event.name == "ctrl+c":
        return replace(state, step="cancelled"), Finish(saved=False)
    if isinstance(event, Fetched):
        return _fetched(state, event), None
    if isinstance(event, Probed):
        return _probed(state, event), None
    if isinstance(event, Saved):
        done = replace(state, step="saved", saved_as=event.name, saved_where=event.where)
        return done, Finish(saved=True)
    if isinstance(event, SaveFailed):
        return replace(state, test="failed", message=event.message, option=0), None
    handler = {
        "provider": _provider,
        "base_url": _base_url,
        "key": _key,
        "model": _model,
        "test": _test,
    }.get(state.step)
    if handler is None:
        return state, None
    return handler(state, event)


def _move(cursor: int, rows: int, event: Event) -> int | None:
    """The new cursor for a move key, or None when the event is not a move."""

    if not isinstance(event, Key):
        return None
    if event.name in {"up", "k"} or event.char == "k":
        return max(0, cursor - 1)
    if event.name in {"down", "j"} or event.char == "j":
        return min(rows - 1, cursor + 1)
    if event.name == "home":
        return 0
    if event.name == "end":
        return rows - 1
    return None


def _provider(state: SetupState, event: Event) -> tuple[SetupState, Effect | None]:
    moved = _move(state.cursor, len(CHOICES), event)
    if moved is not None:
        return replace(state, cursor=moved, error=""), None
    if not isinstance(event, Key):
        return state, None
    if event.char.isdigit() and 1 <= int(event.char) <= len(CHOICES):
        return _chosen(replace(state, cursor=int(event.char) - 1))
    if event.name in {"escape", "q"} or event.char == "q":
        return replace(state, step="cancelled"), Finish(saved=False)
    if event.name == "enter":
        return _chosen(state)
    return state, None


def _chosen(state: SetupState) -> tuple[SetupState, Effect | None]:
    choice = state.choice
    fresh = replace(state, error="", models=(), listed=False, model_cursor=0)
    if choice.flag == "compatible":
        return replace(fresh, step="base_url"), None
    if choice.flag == "ollama":
        return _to_model(replace(fresh, base_url=choice.base_url or OLLAMA_URL, key=""))
    if choice.base_url is None:
        fresh = replace(fresh, base_url="")
    return replace(fresh, step="key"), None


def _edit(text: str, event: Event, *, single_line: bool = True) -> str | None:
    """The field after a typing event, or None when the event does not edit."""

    if isinstance(event, Paste):
        pasted = event.text.replace("\r", "").replace("\n", "") if single_line else event.text
        return (text + pasted)[:_MAX_FIELD]
    if not isinstance(event, Key):
        return None
    if event.name == "backspace":
        return text[:-1]
    if event.name == "ctrl+u":
        return ""
    if event.char and event.char.isprintable() and len(event.char) == 1:
        return (text + event.char)[:_MAX_FIELD]
    return None


def _base_url(state: SetupState, event: Event) -> tuple[SetupState, Effect | None]:
    edited = _edit(state.base_url, event)
    if edited is not None:
        return replace(state, base_url=edited, error=""), None
    if not isinstance(event, Key):
        return state, None
    if event.name == "escape":
        return replace(state, step="provider", error=""), None
    if event.name == "enter":
        try:
            url = check_url(state.base_url)
        except InputRejected as exc:
            return replace(state, error=str(exc)), None
        return replace(state, base_url=url, step="key", error=""), None
    return state, None


def _key(state: SetupState, event: Event) -> tuple[SetupState, Effect | None]:
    edited = _edit(state.key, event)
    if edited is not None:
        return replace(state, key=edited, error=""), None
    if not isinstance(event, Key):
        return state, None
    if event.name == "escape":
        back: Step = "base_url" if state.choice.flag == "compatible" else "provider"
        return replace(state, step=back, error=""), None
    if event.name != "enter":
        return state, None
    if not state.key.strip():
        if state.choice.needs_key and not state.env_key:
            return replace(state, error=f"{state.choice.label} needs a key."), None
        return _to_model(replace(state, key="", error=""))
    try:
        cleaned = check_key(state.key)
    except InputRejected as exc:
        return replace(state, error=str(exc)), None
    return _to_model(replace(state, key=cleaned, error=""))


def _to_model(state: SetupState) -> tuple[SetupState, Effect | None]:
    choice = state.choice
    suggestions = SUGGESTIONS.get(choice.flag, ())
    moved = replace(state, step="model", error="")
    if state.listed:
        return moved, None
    if choice.kind == "compatible":
        fetching = replace(moved, models=suggestions, fetching=True, model_cursor=0)
        return fetching, FetchModels(choice, state.base_url, state.key)
    return replace(moved, models=suggestions, model_cursor=0, listed=True), None


def _fetched(state: SetupState, event: Fetched) -> SetupState:
    if not state.fetching:
        return state
    if event.models:
        return replace(state, models=event.models, fetching=False, listed=True, model_cursor=0)
    cursor = 0 if state.models else len(state.models)
    return replace(state, fetching=False, listed=True, model_cursor=cursor)


def _model(state: SetupState, event: Event) -> tuple[SetupState, Effect | None]:
    if state.on_custom:
        edited = _edit(state.custom, event)
        # j and k type into the free-text row; only the arrows leave it.
        if edited is not None and not (isinstance(event, Key) and event.name in {"up", "down"}):
            return replace(state, custom=edited, error=""), None
        if isinstance(event, Key) and event.name == "up":
            return replace(state, model_cursor=max(0, state.model_cursor - 1), error=""), None
    else:
        moved = _move(state.model_cursor, state.model_rows, event)
        if moved is not None:
            return replace(state, model_cursor=moved, error=""), None
    if not isinstance(event, Key):
        return state, None
    if event.name == "escape":
        back: Step = "provider" if state.choice.flag == "ollama" else "key"
        return replace(state, step=back, error=""), None
    if event.name in {"q"} or (event.char == "q" and not state.on_custom):
        return replace(state, step="cancelled"), Finish(saved=False)
    if event.name != "enter":
        return state, None
    picked = state.custom if state.on_custom else state.models[state.model_cursor]
    try:
        model = check_model(picked)
    except InputRejected as exc:
        return replace(state, error=str(exc)), None
    return _run_test(replace(state, model=model))


def _run_test(state: SetupState) -> tuple[SetupState, Effect | None]:
    base_url = state.base_url or None
    running = replace(state, step="test", test="running", message="", option=0, error="")
    return running, Probe(state.choice, state.model, base_url, state.key)


def _probed(state: SetupState, event: Probed) -> SetupState:
    if state.step != "test" or state.test != "running":
        return state
    return replace(
        state,
        test="ok" if event.ok else "failed",
        message=event.message,
        seconds=event.seconds,
        option=0,
    )


def _test(state: SetupState, event: Event) -> tuple[SetupState, Effect | None]:
    if state.test in {"running", "saving"} or not isinstance(event, Key):
        return state, None
    options = state.options
    moved = _move(state.option, len(options), event)
    if moved is not None:
        return replace(state, option=moved), None
    shortcuts = {"r": "Retry", "e": "Edit", "s": "Save", "c": "Cancel", "q": "Cancel"}
    wanted = shortcuts.get(event.char or event.name)
    if wanted == "Save" and state.test == "failed":
        wanted = "Save anyway"
    if event.name == "escape":
        return replace(state, step="model", error=""), None
    if event.name == "enter":
        wanted = options[state.option]
    if wanted is None or wanted not in options:
        return state, None
    if wanted == "Retry":
        return _run_test(state)
    if wanted == "Edit":
        return replace(state, step="provider", error=""), None
    if wanted == "Cancel":
        return replace(state, step="cancelled"), Finish(saved=False)
    key = state.key or None
    saving = replace(state, test="saving")
    return saving, Save(state.choice, state.model, state.base_url or None, key)


def tick(state: SetupState) -> SetupState:
    return replace(state, spin=(state.spin + 1) % len(SPINNER))


# --- manage -----------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    name: str
    provider: str
    model: str
    key: str
    base_url: str
    active: bool


@dataclass(frozen=True)
class Use:
    name: str


@dataclass(frozen=True)
class Remove:
    name: str


@dataclass(frozen=True)
class Add:
    pass


@dataclass(frozen=True)
class Leave:
    pass


ManageEffect = Use | Remove | Add | Leave


@dataclass(frozen=True)
class ManageState:
    rows: tuple[Row, ...] = ()
    cursor: int = 0
    confirm: str | None = None
    message: str = ""
    override: str = ""


@dataclass(frozen=True)
class Refreshed:
    rows: tuple[Row, ...]
    message: str = ""


def manage_start(rows: Sequence[Row], override: str = "") -> ManageState:
    active = next((index for index, row in enumerate(rows) if row.active), 0)
    return ManageState(rows=tuple(rows), cursor=active, override=override)


def manage_step(
    state: ManageState, event: Key | Refreshed
) -> tuple[ManageState, ManageEffect | None]:
    if isinstance(event, Refreshed):
        cursor = min(state.cursor, max(0, len(event.rows) - 1))
        return replace(state, rows=event.rows, cursor=cursor, message=event.message), None
    if event.name == "ctrl+c":
        return state, Leave()
    if state.confirm is not None:
        if event.char in {"y", "Y"}:
            return replace(state, confirm=None), Remove(state.confirm)
        return replace(state, confirm=None, message="Nothing was removed."), None
    moved = _move(state.cursor, max(1, len(state.rows)), event)
    if moved is not None:
        return replace(state, cursor=moved, message=""), None
    current = state.rows[state.cursor] if state.rows else None
    if event.name in {"escape"} or event.char == "q":
        return state, Leave()
    if event.char == "a":
        return replace(state, message=""), Add()
    if current is None:
        return state, None
    if event.name == "enter" or event.char in {"u", " "}:
        return state, Use(current.name)
    if event.name in {"delete", "backspace"} or event.char in {"d", "x"}:
        return replace(state, confirm=current.name, message=""), None
    return state, None


# --- rendering --------------------------------------------------------------


_STEPS = ("provider", "base_url", "key", "model", "test")
_TITLES = {
    "provider": "Provider",
    "base_url": "Base URL",
    "key": "API key",
    "model": "Model",
    "test": "Connection test",
}


def _line(*segments: Segment) -> Line:
    return tuple(segments)


def _steps_for(choice: Choice) -> tuple[str, ...]:
    if choice.flag == "ollama":
        return ("provider", "model", "test")
    if choice.flag == "compatible":
        return _STEPS
    return ("provider", "key", "model", "test")


def bullets(key: str) -> str:
    return BULLET * min(len(key), _MAX_BULLETS)


def render_setup(state: SetupState, *, first_use: bool = False) -> list[Line]:
    """The body of the setup screen. The key shows as bullets, then as its mask."""

    lines: list[Line] = []
    if first_use and state.step == "provider":
        lines.append(_line(("No model is set up yet. ", "dim"), ("Pick one to start.", "dim")))
        lines.append(())
    steps = _steps_for(state.choice)
    shown = state.step if state.step in steps else steps[-1]
    number = steps.index(shown) + 1
    lines.append(_line(("Set up a model", "title")))
    lines.append(_line((f"step {number} of {len(steps)}  ·  {_TITLES[shown]}", "dim")))
    lines.append(())
    lines.extend(_summary(state))
    body = {
        "provider": _render_provider,
        "base_url": _render_base_url,
        "key": _render_key,
        "model": _render_model,
        "test": _render_test,
        "saved": _render_saved,
        "cancelled": _render_saved,
    }[state.step](state)
    lines.extend(body)
    if state.error:
        lines.append(())
        lines.append(_line((state.error, "error")))
    return lines


def _summary(state: SetupState) -> list[Line]:
    """What is already chosen, above the current step."""

    if state.step == "provider":
        return []
    done: list[Line] = [_line(("provider  ", "dim"), (state.choice.label, "text"))]
    if state.step != "base_url" and state.choice.kind == "compatible" and state.base_url:
        done.append(_line(("base url  ", "dim"), (state.base_url, "text")))
    if state.step in {"model", "test", "saved"} and state.choice.flag != "ollama":
        if state.key:
            done.append(_line(("key       ", "dim"), (mask(state.key), "text")))
        elif state.env_key:
            done.append(_line(("key       ", "dim"), (f"{KEY_ENVS[state.choice.kind]}", "text")))
        else:
            done.append(_line(("key       ", "dim"), ("none", "text")))
    if state.step in {"test", "saved"}:
        done.append(_line(("model     ", "dim"), (state.model, "text")))
    done.append(())
    return done


def _row(selected: bool, text: str, hint: str = "") -> Line:
    mark: Segment = ("› ", "accent") if selected else ("  ", "text")
    role: Role = "selected" if selected else "text"
    if hint:
        return _line(mark, (f"{text:<22}", role), (hint, "dim"))
    return _line(mark, (text, role))


_HINTS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "local, no key",
    "compatible": "LM Studio, vLLM, a proxy",
}


def _render_provider(state: SetupState) -> list[Line]:
    rows = []
    for index, choice in enumerate(CHOICES):
        label = choice.label.split(" (")[0]
        rows.append(_row(index == state.cursor, label, _HINTS[choice.flag]))
    return rows


def _field(value: str, *, placeholder: str = "") -> Line:
    if not value and placeholder:
        return _line(("› ", "accent"), ("▏", "accent"), (placeholder, "dim"))
    return _line(("› ", "accent"), (value, "field"), ("▏", "accent"))


def _render_base_url(state: SetupState) -> list[Line]:
    return [
        _line(("The endpoint's OpenAI-compatible root, ending in /v1.", "dim")),
        (),
        _field(state.base_url, placeholder="http://127.0.0.1:1234/v1"),
    ]


def _render_key(state: SetupState) -> list[Line]:
    choice = state.choice
    env_name = KEY_ENVS[choice.kind]
    if choice.needs_key and state.env_key:
        hint = f"Hidden as you type. Enter with none uses {env_name} and saves no key."
    elif choice.needs_key:
        hint = "Hidden as you type. Paste works. It is never shown again."
    else:
        hint = "Hidden as you type. Enter with none for a server without a key."
    return [_line((hint, "dim")), (), _field(bullets(state.key), placeholder="key")]


def _render_model(state: SetupState) -> list[Line]:
    lines: list[Line] = []
    if state.fetching:
        lines.append(_line((f"{SPINNER[state.spin]} listing models at {state.base_url}", "dim")))
    elif state.choice.kind == "compatible" and state.listed and not state.models:
        lines.append(_line(("No model list at that endpoint. Type a model id.", "dim")))
    else:
        lines.append(_line(("Pick one, or type another.", "dim")))
    lines.append(())
    for index, name in enumerate(state.models):
        lines.append(_row(index == state.model_cursor, name))
    if state.on_custom:
        lines.append(_line(("› ", "accent"), (state.custom, "field"), ("▏", "accent")))
    else:
        lines.append(_line(("  ", "text"), ("other: type a model id", "dim")))
    return lines


def _render_test(state: SetupState) -> list[Line]:
    where = state.base_url or ("api.openai.com" if state.choice.kind == "openai" else "")
    where = where or "api.anthropic.com"
    if state.test == "running":
        return [_line((f"{SPINNER[state.spin]} ", "accent"), (f"testing {where}", "dim"))]
    if state.test == "saving":
        return [_line((f"{SPINNER[state.spin]} ", "accent"), ("saving", "dim"))]
    if state.test == "ok":
        head = _line(("✓ ", "ok"), (f"Connected in {state.seconds:.1f} s.", "ok"))
    else:
        head = _line(("✗ ", "error"), (state.message or "The test failed.", "error"))
    lines = [head, ()]
    for index, option in enumerate(state.options):
        lines.append(_row(index == state.option, option))
    return lines


def _render_saved(state: SetupState) -> list[Line]:
    if state.step == "cancelled":
        return [_line(("Nothing was saved.", "dim"))]
    return [_line(("✓ ", "ok"), (f"Saved {state.saved_as}. {state.saved_where}", "ok"))]


SETUP_KEYS: dict[str, str] = {
    "provider": "↑↓ move   enter select   esc quit",
    "base_url": "enter next   esc back   ctrl+u clear",
    "key": "enter next   esc back   ctrl+u clear",
    "model": "↑↓ move   enter test   esc back",
    "test": "↑↓ move   enter choose   r retry   e edit   esc back",
    "saved": "",
    "cancelled": "",
}


def render_manage(state: ManageState) -> list[Line]:
    lines: list[Line] = [_line(("Models", "title"))]
    if not state.rows:
        lines += [(), _line(("No saved models. Press a to add one.", "dim"))]
    else:
        lines.append(_line(("● active  ·  the key shows as its last four characters", "dim")))
        lines.append(())
        widths = [
            max(len(row.name) for row in state.rows),
            max(len(row.provider) for row in state.rows),
            max(len(row.model) for row in state.rows),
            max(len(row.key) for row in state.rows),
        ]
        for index, row in enumerate(state.rows):
            selected = index == state.cursor
            mark: Segment = ("› ", "accent") if selected else ("  ", "text")
            dot: Segment = ("● ", "accent") if row.active else ("  ", "text")
            role: Role = "selected" if selected else "text"
            lines.append(
                _line(
                    mark,
                    dot,
                    (f"{row.name:<{widths[0]}}  ", role),
                    (f"{row.provider:<{widths[1]}}  ", "dim"),
                    (f"{row.model:<{widths[2]}}  ", role),
                    (f"{row.key:<{widths[3]}}  ", "dim"),
                    (row.base_url, "dim"),
                )
            )
    if state.override:
        lines += [(), _line((state.override, "dim"))]
    if state.confirm is not None:
        lines += [(), _line((f"Remove {state.confirm} and its key? ", "error"), ("y / n", "text"))]
    elif state.message:
        lines += [(), _line((state.message, "dim"))]
    return lines


MANAGE_KEYS = "↑↓ move   enter use   a add   d remove   q quit"


def plain(lines: Sequence[Line]) -> str:
    """The text of rendered lines, without styles. Tests read frames through this."""

    return "\n".join("".join(text for text, _role in line) for line in lines)
