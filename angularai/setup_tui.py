"""The setup and model screens on Textual, in the session's look.

Same framework, palette, and frame as ``angularai.tui``: a dim top line, the body,
and a status line under a rule. The brand pink marks the cursor. Under NO_COLOR,
Textual draws it all in monochrome and the roles fall back to bold and dim.

The screens hold no logic of their own. A key becomes an event for the reducers
in ``angularai.setup_state``; an effect they return (a test call, a model listing,
a save) runs here, on a worker thread when it may block, and its result comes
back as another event. Textual owns the terminal: the alternate screen, raw keys,
bracketed paste, and putting the terminal back on every exit.
"""

import http.client
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Static

from angularai import __version__
from angularai.errors import AngularAIError, SubstrateError
from angularai.http import connection
from angularai.onboard import Choice, Probe, _default_probe, _explain
from angularai.profiles import (
    KEY_ENVS,
    MODEL_ENV,
    Profile,
    Vault,
    add_profile,
    free_name,
    key_source,
    load_book,
    remove_profile,
    storage_words,
    use_profile,
)
from angularai.setup_state import (
    MANAGE_KEYS,
    SETUP_KEYS,
    Add,
    Effect,
    Event,
    Fetched,
    FetchModels,
    Finish,
    Key,
    Leave,
    Line,
    ManageEffect,
    ManageState,
    Paste,
    Probed,
    Refreshed,
    Role,
    Row,
    Save,
    Saved,
    SaveFailed,
    SetupState,
    Use,
    manage_start,
    manage_step,
    render_manage,
    render_setup,
    setup_step,
    start,
    tick,
)
from angularai.setup_state import Probe as ProbeEffect

PINK = "#ec4a7f"
_LIST_SECONDS = 3.0
_LIST_BYTES = 1 << 20
_LIST_MAX = 50

CSS = """
SetupFrame {
    background: #0c0c0c;
    color: #e8e6e3;
}

#setup-top {
    height: 1;
    padding: 0 1;
    color: #6f6a64;
    background: #0c0c0c;
}

#setup-body {
    height: 1fr;
    padding: 1 2 0 2;
    background: #0c0c0c;
    scrollbar-size: 1 1;
    scrollbar-color: #2a2a2a;
    scrollbar-background: #0c0c0c;
}

#setup-page {
    height: auto;
}

#setup-dock {
    height: 2;
    border-top: solid #2a2a2a;
    background: #0c0c0c;
}

#setup-keys {
    height: 1;
    padding: 0 1;
    background: #141414;
    color: #7a756e;
}
"""

_COLOR: dict[Role, Style] = {
    "text": Style(color="#e8e6e3"),
    "dim": Style(color="#6f6a64"),
    "accent": Style(color=PINK, bold=True),
    "selected": Style(color="#ffffff", bold=True),
    "ok": Style(color="#6cc58c"),
    "error": Style(color="#ef6b6b"),
    "title": Style(color="#e8e6e3", bold=True),
    "field": Style(color="#e8e6e3", bgcolor="#191919"),
}
_MONO: dict[Role, Style] = {
    "text": Style(),
    "dim": Style(dim=True),
    "accent": Style(bold=True),
    "selected": Style(bold=True),
    "ok": Style(bold=True),
    "error": Style(bold=True),
    "title": Style(bold=True, underline=True),
    "field": Style(reverse=True),
}


def color_on(environ: Mapping[str, str]) -> bool:
    return "NO_COLOR" not in environ


def paint(lines: list[Line], color: bool) -> Text:
    styles = _COLOR if color else _MONO
    text = Text()
    for index, line in enumerate(lines):
        if index:
            text.append("\n")
        for part, role in line:
            text.append(part, styles[role])
    return text


def key_event(event: events.Key) -> Key:
    char = event.character if event.is_printable and event.character else ""
    return Key(event.key, char)


def top_line(width: int, right: str = __version__) -> str:
    room = max(width - 2, 0)
    gap = room - len("angularai") - len(right)
    return f"angularai{' ' * gap}{right}" if gap >= 1 else "angularai"[:room]


# --- model listing ----------------------------------------------------------


def list_models(
    choice: Choice, base_url: str, key: str, timeout: float = _LIST_SECONDS
) -> tuple[str, ...]:
    """Model ids the endpoint offers, or () on any failure. Ollama's /api/tags, else /models."""

    root = base_url.rstrip("/")
    if choice.flag == "ollama":
        url = (root[: -len("/v1")] if root.endswith("/v1") else root) + "/api/tags"
    else:
        url = root + "/models"
    headers = {"User-Agent": f"angularai/{__version__}", "Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        body = _get(url, headers, timeout)
        payload: object = json.loads(body)
    except (OSError, ValueError, http.client.HTTPException, RecursionError):
        return ()
    return _model_ids(payload)


def _get(url: str, headers: dict[str, str], timeout: float) -> bytes:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise OSError("not http")
    conn, target, extra = connection(parts, timeout)
    try:
        conn.request("GET", target, headers={**headers, **extra})
        response = conn.getresponse()
        if response.status != 200:
            raise OSError("not 200")
        return response.read(_LIST_BYTES)
    finally:
        conn.close()


def _model_ids(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    entries = payload.get("models", payload.get("data"))
    if not isinstance(entries, list):
        return ()
    found: list[str] = []
    for entry in entries:
        name = entry.get("name", entry.get("id")) if isinstance(entry, dict) else None
        if (
            isinstance(name, str)
            and 0 < len(name) <= 128
            and all(c.isprintable() and not c.isspace() for c in name)
            and name not in found
        ):
            found.append(name)
    return tuple(sorted(found)[:_LIST_MAX])


Lister = Callable[[Choice, str, str], tuple[str, ...]]


# --- results ----------------------------------------------------------------


@dataclass(frozen=True)
class SetupResult:
    profile: Profile
    where: str
    env_note: str


def save_profile(
    environ: Mapping[str, str],
    vault: Vault,
    effect: Save,
    *,
    make_active: bool,
) -> SetupResult:
    """Save under the provider's name. The same model replaces it; another gets -2, -3."""

    book = load_book(environ)
    base = effect.choice.flag
    existing = book.get(base)
    name = base if existing is None or existing.model == effect.model else free_name(book, base)
    profile = Profile(name, effect.choice.kind, effect.model, effect.base_url, "none", "")
    saved = add_profile(environ, vault, profile, effect.key, make_active=make_active)
    env_name = KEY_ENVS[effect.choice.kind]
    env_set = bool(environ.get(env_name, "").strip())
    if effect.key is not None:
        where = f"Key {saved.hint} in {storage_words(saved.storage, environ)}."
        note = f"{env_name} is set in this shell and wins over the saved key." if env_set else ""
    elif env_set:
        where, note = f"No key saved. {env_name} is read from the environment.", ""
    else:
        where, note = "No key saved.", ""
    return SetupResult(saved, where, note)


# --- screens ----------------------------------------------------------------


class SetupFrame(Screen[object]):
    """The session's frame: top line, body, keys under a rule.

    Styles ride on the screen so they hold inside the session app too.
    """

    DEFAULT_CSS = CSS

    def __init__(self, color: bool) -> None:
        super().__init__()
        self._color = color

    def compose(self) -> ComposeResult:
        yield Static("", id="setup-top", markup=False)
        with VerticalScroll(id="setup-body"):
            yield Static("", id="setup-page", markup=False)
        with Vertical(id="setup-dock"):
            yield Static("", id="setup-keys", markup=False)

    def on_resize(self, _event: events.Resize) -> None:
        self.refresh_page()

    def show(self, lines: list[Line], keys: str, right: str = __version__) -> None:
        self.query_one("#setup-top", Static).update(top_line(self.size.width, right))
        self.query_one("#setup-page", Static).update(paint(lines, self._color))
        self.query_one("#setup-keys", Static).update(keys)

    def refresh_page(self) -> None:  # pragma: no cover - overridden
        pass


class SetupScreen(SetupFrame):
    """Provider, base URL, key, model, test. Dismisses with a SetupResult or None."""

    def __init__(
        self,
        environ: Mapping[str, str],
        vault: Vault,
        *,
        tester: Probe = _default_probe,
        lister: Lister = list_models,
        first_use: bool = False,
        make_active: bool = True,
        color: bool = True,
    ) -> None:
        super().__init__(color)
        kinds = frozenset(kind for kind, name in KEY_ENVS.items() if environ.get(name, "").strip())
        self.state: SetupState = start(kinds)
        self._environ = environ
        self._vault = vault
        self._tester = tester
        self._lister = lister
        self._first_use = first_use
        self._make_active = make_active
        self._result: SetupResult | None = None
        self._spinner: Timer | None = None

    def on_mount(self) -> None:
        self._spinner = self.set_interval(0.08, self._tick, pause=True)
        self.refresh_page()

    def refresh_page(self) -> None:
        lines = render_setup(self.state, first_use=self._first_use)
        self.show(lines, SETUP_KEYS.get(self.state.step, ""), "setup")

    def on_key(self, event: events.Key) -> None:
        event.stop()
        event.prevent_default()
        self.dispatch(key_event(event))

    def on_paste(self, event: events.Paste) -> None:
        event.stop()
        self.dispatch(Paste(event.text))

    def dispatch(self, event: Event) -> None:
        self.state, effect = setup_step(self.state, event)
        busy = self.state.fetching or (
            self.state.step == "test" and self.state.test in {"running", "saving"}
        )
        if self._spinner is not None:
            (self._spinner.resume if busy else self._spinner.pause)()
        self.refresh_page()
        if effect is not None:
            self._run(effect)

    def _tick(self) -> None:
        self.state = tick(self.state)
        self.refresh_page()

    def _run(self, effect: Effect) -> None:
        if isinstance(effect, Finish):
            self.dismiss(self._result if effect.saved else None)
            return
        self.run_worker(lambda: self._work(effect), thread=True, exit_on_error=False)

    def _work(self, effect: Effect) -> None:
        result: Event
        if isinstance(effect, FetchModels):
            result = Fetched(self._lister(effect.choice, effect.base_url, effect.key))
        elif isinstance(effect, ProbeEffect):
            result = self._probe(effect)
        elif isinstance(effect, Save):
            result = self._save(effect)
        else:  # pragma: no cover - Finish never reaches a worker
            return
        self.app.call_from_thread(self.dispatch, result)

    def _probe(self, effect: ProbeEffect) -> Probed:
        key = effect.key or self._environ.get(KEY_ENVS[effect.choice.kind], "").strip()
        where = effect.base_url or (
            "https://api.openai.com/v1"
            if effect.choice.kind == "openai"
            else "https://api.anthropic.com/v1"
        )
        started = time.monotonic()
        try:
            self._tester(effect.choice.kind, effect.model, effect.base_url, key)
        except SubstrateError as exc:
            return Probed(False, _explain(exc, where))
        except AngularAIError as exc:
            return Probed(False, str(exc))
        except Exception:
            return Probed(False, "The test failed. Nothing was saved.")
        return Probed(True, seconds=time.monotonic() - started)

    def _save(self, effect: Save) -> Saved | SaveFailed:
        try:
            self._result = save_profile(
                self._environ, self._vault, effect, make_active=self._make_active
            )
        except AngularAIError as exc:
            return SaveFailed(str(exc))
        except OSError:
            return SaveFailed("The model could not be saved. Nothing was changed.")
        return Saved(self._result.profile.name, self._result.where)


def manage_rows(environ: Mapping[str, str]) -> tuple[Row, ...]:
    book = load_book(environ)
    return tuple(
        Row(
            item.name,
            item.provider,
            item.model,
            key_source(item.provider, item, environ),
            item.base_url or "",
            item.name == book.active,
        )
        for item in book.profiles
    )


def _override(environ: Mapping[str, str]) -> str:
    picked = environ.get(MODEL_ENV, "").strip()
    return f"{MODEL_ENV} picks {picked} in this shell." if picked else ""


class ManageScreen(SetupFrame):
    """Saved models: switch, add, remove. Dismisses with None."""

    def __init__(
        self,
        environ: Mapping[str, str],
        vault: Vault,
        *,
        tester: Probe = _default_probe,
        lister: Lister = list_models,
        color: bool = True,
    ) -> None:
        super().__init__(color)
        self._environ = environ
        self._vault = vault
        self._tester = tester
        self._lister = lister
        self.state: ManageState = manage_start(manage_rows(environ), _override(environ))

    def on_mount(self) -> None:
        self.refresh_page()

    def refresh_page(self) -> None:
        self.show(render_manage(self.state), MANAGE_KEYS, "models")

    def on_key(self, event: events.Key) -> None:
        event.stop()
        event.prevent_default()
        self.dispatch(key_event(event))

    def dispatch(self, event: Key | Refreshed) -> None:
        self.state, effect = manage_step(self.state, event)
        self.refresh_page()
        if effect is not None:
            self._run(effect)

    def _run(self, effect: ManageEffect) -> None:
        if isinstance(effect, Leave):
            self.dismiss(None)
            return
        if isinstance(effect, Add):
            first = not self.state.rows
            screen = SetupScreen(
                self._environ,
                self._vault,
                tester=self._tester,
                lister=self._lister,
                make_active=first,
                color=self._color,
            )
            self.app.push_screen(screen, self._added)
            return
        try:
            if isinstance(effect, Use):
                found = use_profile(self._environ, effect.name)
                message = f"Active model: {found.name}  {found.provider}  {found.model}"
            else:
                gone, was_active = remove_profile(self._environ, self._vault, effect.name)
                message = f"Removed {gone.name}" + (
                    " and its key." if gone.storage != "none" else "."
                )
                if was_active:
                    message += " No model is active."
        except AngularAIError as exc:
            message = str(exc)
        self.dispatch(Refreshed(manage_rows(self._environ), message))

    def _added(self, result: object) -> None:
        message = ""
        if isinstance(result, SetupResult):
            message = f"Saved {result.profile.name}. {result.where}"
        self.dispatch(Refreshed(manage_rows(self._environ), message))


# --- apps -------------------------------------------------------------------


class _Host(App[object]):
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, make: Callable[[], Screen[object]]) -> None:
        super().__init__()
        self._make = make
        self.title = "angularai"

    def on_mount(self) -> None:
        self.push_screen(self._make(), self.exit)


def run_setup_tui(
    environ: Mapping[str, str] | None = None,
    vault: Vault | None = None,
    *,
    tester: Probe = _default_probe,
    lister: Lister = list_models,
    first_use: bool = False,
    make_active: bool = True,
) -> SetupResult | None:
    env = os.environ if environ is None else environ
    keys = vault or Vault(env)

    def make() -> Screen[object]:
        return SetupScreen(
            env,
            keys,
            tester=tester,
            lister=lister,
            first_use=first_use,
            make_active=make_active,
            color=color_on(env),
        )

    result = _Host(make).run()
    return result if isinstance(result, SetupResult) else None


def run_manage_tui(
    environ: Mapping[str, str] | None = None,
    vault: Vault | None = None,
    *,
    tester: Probe = _default_probe,
    lister: Lister = list_models,
) -> None:
    env = os.environ if environ is None else environ
    keys = vault or Vault(env)

    def make() -> Screen[object]:
        return ManageScreen(env, keys, tester=tester, lister=lister, color=color_on(env))

    _Host(make).run()
