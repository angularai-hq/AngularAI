"""The setup and model screens on Textual: flows, masking, colour, the session's /model."""

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from textual import events
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import Input, Static

from angularai.errors import AuthFailed, InputRejected, Unreachable
from angularai.models import SubstrateKind
from angularai.onboard import CHOICES
from angularai.profiles import Vault, load_book, mask
from angularai.setup_state import Save
from angularai.setup_tui import (
    ManageScreen,
    SetupResult,
    SetupScreen,
    _Host,
    color_on,
    list_models,
    manage_rows,
    paint,
    save_profile,
    top_line,
)
from angularai.store import init_cortex, open_store
from angularai.tui import AngularApp
from tests.setupfakes import KEY, FakeKeyring

OPENAI, ANTHROPIC, OLLAMA, COMPATIBLE = CHOICES
_WAIT = 10.0


class FakeTester:
    def __init__(self, *errors: Exception) -> None:
        self.calls: list[tuple[SubstrateKind, str, str | None, str]] = []
        self.errors = list(errors)

    def __call__(self, kind: SubstrateKind, model: str, base_url: str | None, key: str) -> None:
        self.calls.append((kind, model, base_url, key))
        if self.errors:
            raise self.errors.pop(0)


def lister(models: tuple[str, ...]) -> Callable[..., tuple[str, ...]]:
    def listed(_choice: object, _url: str, _key: str) -> tuple[str, ...]:
        return models

    return listed


def env_for(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"ANGULARAI_HOME": str(tmp_path / "home"), **extra}


def page(app: _Host) -> str:
    return str(app.screen.query_one("#setup-page", Static).content)


def keys_line(app: _Host) -> str:
    return str(app.screen.query_one("#setup-keys", Static).content)


async def until(pilot: Pilot[Any], check: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _WAIT
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if check():
            return
    raise AssertionError("the screen never got there")


def host(make: Callable[[], Screen[object]]) -> _Host:
    return _Host(make)


# --- setup screen -------------------------------------------------------------


def test_openai_flow_saves_masks_and_never_draws_the_key(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    tester = FakeTester()
    frames: list[str] = []
    app = host(lambda: SetupScreen(environ, Vault(environ), tester=tester, first_use=True))

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert "No model is set up yet." in page(app)
            assert "↑↓ move" in keys_line(app)
            assert "setup" in str(app.screen.query_one("#setup-top", Static).content)
            await pilot.press("enter")
            await until(pilot, lambda: "API key" in page(app))
            await pilot.press("s", "k")
            app.screen.post_message(events.Paste(KEY[2:]))
            await until(pilot, lambda: "•" * len(KEY) in page(app))
            frames.append(app.export_screenshot())
            await pilot.press("enter")
            await until(pilot, lambda: "gpt-5-mini" in page(app))
            frames.append(app.export_screenshot())
            await pilot.press("enter")
            await until(pilot, lambda: "Connected in" in page(app))
            frames.append(app.export_screenshot())
            await pilot.press("enter")
            await until(pilot, lambda: app.return_value is not None)

    asyncio.run(run())
    assert tester.calls == [("openai", "gpt-5-mini", None, KEY)]
    result = app.return_value
    assert isinstance(result, SetupResult)
    assert result.profile.name == "openai" and result.profile.model == "gpt-5-mini"
    assert mask(KEY) in result.where
    assert load_book(environ).active == "openai"
    for shown in frames:
        assert KEY not in shown
        assert KEY[3:15] not in shown
    assert KEY not in repr(result)


def test_compatible_flow_lists_models_and_a_failed_test_can_be_retried(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    tester = FakeTester(AuthFailed("The key was refused."))
    app = host(
        lambda: SetupScreen(
            environ, Vault(environ), tester=tester, lister=lister(("demo-large", "demo-model"))
        )
    )

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("end", "enter")
            await until(pilot, lambda: "Base URL" in page(app))
            app.screen.post_message(events.Paste("http://127.0.0.1:9/v1"))
            await pilot.press("enter")
            await until(pilot, lambda: "API key" in page(app))
            await pilot.press("enter")
            await until(pilot, lambda: "demo-model" in page(app))
            await pilot.press("j", "enter")
            await until(pilot, lambda: "✗" in page(app))
            assert "Retry" in page(app) and "Save anyway" in page(app)
            await pilot.press("r")
            await until(pilot, lambda: "Connected in" in page(app))
            await pilot.press("enter")
            await until(pilot, lambda: app.return_value is not None)

    asyncio.run(run())
    assert [call[1] for call in tester.calls] == ["demo-model", "demo-model"]
    result = app.return_value
    assert isinstance(result, SetupResult)
    assert result.profile.base_url == "http://127.0.0.1:9/v1"
    assert result.where == "No key saved."


def test_ollama_saves_anyway_after_a_failure_and_the_mapping_names_the_host(
    tmp_path: Path,
) -> None:
    environ = env_for(tmp_path)
    tester = FakeTester(Unreachable("down"))
    app = host(lambda: SetupScreen(environ, Vault(environ), tester=tester, lister=lister(())))
    seen: list[str] = []

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("3")
            await until(pilot, lambda: "llama3.2" in page(app))
            await pilot.press("enter")
            await until(pilot, lambda: "✗" in page(app))
            seen.append(page(app))
            await pilot.press("s")
            await until(pilot, lambda: app.return_value is not None)

    asyncio.run(run())
    assert "127.0.0.1:11434" in seen[0]
    result = app.return_value
    assert isinstance(result, SetupResult) and result.profile.name == "ollama"


@pytest.mark.parametrize("leave", ["escape", "q", "ctrl+c"])
def test_leaving_setup_returns_nothing(tmp_path: Path, leave: str) -> None:
    environ = env_for(tmp_path)
    app = host(lambda: SetupScreen(environ, Vault(environ), tester=FakeTester()))

    async def run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(leave)
            await pilot.pause()

    asyncio.run(run())
    assert app.return_value is None
    assert load_book(environ).profiles == ()


def test_unexpected_test_errors_and_save_errors_stay_on_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = env_for(tmp_path)
    tester = FakeTester(InputRejected("odd model"), RuntimeError(KEY))
    app = host(lambda: SetupScreen(environ, Vault(environ), tester=tester))
    seen: list[str] = []

    def broken(*_args: object, **_kw: object) -> SetupResult:
        raise OSError("disk")

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("enter")
            app.screen.post_message(events.Paste(KEY))
            await pilot.press("enter")
            await until(pilot, lambda: "gpt-5-mini" in page(app))
            await pilot.press("enter")
            await until(pilot, lambda: "odd model" in page(app))
            await pilot.press("r")
            await until(pilot, lambda: "The test failed. Nothing was saved." in page(app))
            seen.append(app.export_screenshot())
            monkeypatch.setattr("angularai.setup_tui.save_profile", broken)
            await pilot.press("s")
            await until(pilot, lambda: "could not be saved" in page(app))
            await pilot.press("c")
            await pilot.pause()

    asyncio.run(run())
    assert app.return_value is None
    assert KEY not in seen[0]


def test_resize_redraws_the_frame(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    app = host(lambda: SetupScreen(environ, Vault(environ), tester=FakeTester()))

    async def run() -> None:
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(120, 40)
            await until(
                pilot, lambda: len(str(app.screen.query_one("#setup-top", Static).content)) == 118
            )
            await pilot.press("q")

    asyncio.run(run())


def test_no_color_draws_monochrome() -> None:
    assert color_on({}) and not color_on({"NO_COLOR": "1"})
    lines = [(("› ", "accent"), ("x", "selected"))]
    colored = paint(lines, True)  # type: ignore[arg-type]
    mono = paint(lines, False)  # type: ignore[arg-type]
    assert any(span.style and "#ec4a7f" in str(span.style) for span in colored.spans)
    assert all("#" not in str(span.style) for span in mono.spans)
    assert top_line(30, "setup").endswith("setup") and len(top_line(30, "setup")) == 28
    assert top_line(5) == "ang"


# --- manage screen ------------------------------------------------------------


def seeded(environ: dict[str, str]) -> None:
    vault = Vault(environ)
    for choice, model, key in ((OPENAI, "gpt-5", KEY), (OLLAMA, "llama3.2", None)):
        effect = Save(choice, model, choice.base_url, key)
        save_profile(environ, vault, effect, make_active=choice is OPENAI)


def test_manage_lists_switches_and_removes_with_confirmation(tmp_path: Path) -> None:
    environ = env_for(tmp_path, ANGULARAI_MODEL="ollama")
    seeded(environ)
    shots: list[str] = []
    app = host(lambda: ManageScreen(environ, Vault(environ), color=False))

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            shown = page(app)
            assert "● openai" in shown and mask(KEY) in shown
            assert "ANGULARAI_MODEL picks ollama" in shown
            shots.append(app.export_screenshot())
            await pilot.press("j", "enter")
            await until(pilot, lambda: "Active model: ollama" in page(app))
            await pilot.press("k", "d")
            await until(pilot, lambda: "Remove openai and its key?" in page(app))
            await pilot.press("n")
            await until(pilot, lambda: "Nothing was removed." in page(app))
            await pilot.press("d", "y")
            await until(pilot, lambda: "Removed openai and its key." in page(app))
            await pilot.press("d", "y")
            await until(pilot, lambda: "No model is active." in page(app))
            await pilot.press("u")
            await pilot.press("q")

    asyncio.run(run())
    assert KEY not in shots[0]
    assert load_book(environ).profiles == ()


def test_manage_adds_through_the_setup_screen(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    app = host(lambda: ManageScreen(environ, Vault(environ), tester=FakeTester()))

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert "No saved models" in page(app)
            await pilot.press("a")
            await until(pilot, lambda: isinstance(app.screen, SetupScreen))
            await pilot.press("escape")
            await until(pilot, lambda: isinstance(app.screen, ManageScreen))
            await pilot.press("a")
            await until(pilot, lambda: isinstance(app.screen, SetupScreen))
            await pilot.press("3", "enter")
            await until(pilot, lambda: "Connected in" in page(app))
            await pilot.press("enter")
            await until(pilot, lambda: isinstance(app.screen, ManageScreen))
            await until(pilot, lambda: "Saved ollama. No key saved." in page(app))
            assert "● ollama" in page(app)
            await pilot.press("escape")

    asyncio.run(run())


def test_manage_rows_say_where_each_key_lives(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    seeded(environ)
    found = {row.name: row for row in manage_rows(environ)}
    assert found["openai"].active and mask(KEY) in found["openai"].key
    assert found["ollama"].base_url and not found["ollama"].active


# --- saving -------------------------------------------------------------------


def test_save_profile_names_by_provider_and_notes_the_environment(tmp_path: Path) -> None:
    environ = env_for(tmp_path, OPENAI_API_KEY="from-env-not-shown")
    vault = Vault(environ)
    first = save_profile(environ, vault, Save(OPENAI, "gpt-5", None, KEY), make_active=True)
    again = save_profile(environ, vault, Save(OPENAI, "gpt-5", None, KEY), make_active=True)
    other = save_profile(environ, vault, Save(OPENAI, "gpt-4o", None, None), make_active=False)
    assert (first.profile.name, again.profile.name, other.profile.name) == (
        "openai",
        "openai",
        "openai-2",
    )
    assert "wins over the saved key" in first.env_note
    assert other.where == "No key saved. OPENAI_API_KEY is read from the environment."
    assert "from-env-not-shown" not in repr(first) + repr(other)


def test_a_saved_key_goes_to_the_keyring_and_reads_back(
    tmp_path: Path, fake_keyring_and_home: FakeKeyring
) -> None:
    environ = env_for(tmp_path)
    result = save_profile(
        environ, Vault(environ), Save(ANTHROPIC, "c", None, KEY), make_active=True
    )
    assert result.profile.storage == "keyring"
    assert KEY in fake_keyring_and_home.store.values()
    assert not result.env_note


# --- model listing ------------------------------------------------------------


class _Models(BaseHTTPRequestHandler):
    seen: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        type(self).seen.append(self.headers.get("Authorization"))
        body: object
        if self.path == "/api/tags":
            body = {"models": [{"name": "llama3.2"}, {"name": "qwen2.5"}, {"name": "bad name"}]}
        elif self.path == "/v1/models":
            body = {"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}, {"id": 7}, "x"]}
        elif self.path == "/list/models":
            body = []
        elif self.path == "/text/models":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"not json")
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def models_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Models)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_list_models_reads_ollama_tags_and_v1_models(
    models_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    _Models.seen.clear()
    assert list_models(OLLAMA, models_url + "/v1", "") == ("llama3.2", "qwen2.5")
    assert list_models(COMPATIBLE, models_url + "/v1", KEY) == ("a", "b")
    assert _Models.seen == [None, f"Bearer {KEY}"]
    assert list_models(COMPATIBLE, models_url + "/list", "") == ()
    assert list_models(COMPATIBLE, models_url + "/text", "") == ()
    assert list_models(COMPATIBLE, models_url + "/missing", "") == ()
    assert list_models(COMPATIBLE, "ftp://h/v1", "") == ()
    assert list_models(COMPATIBLE, "http://127.0.0.1:1/v1", "", timeout=0.5) == ()


# --- the session's /model -----------------------------------------------------


def test_the_session_opens_the_model_screen_and_comes_back(tmp_path: Path) -> None:
    init_cortex(tmp_path / "cortex")
    store = open_store(tmp_path / "cortex")
    app = AngularApp(store, lambda _store, _text: "unused")

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#input", Input)
            field.value = "/model"
            await pilot.press("enter")
            await until(pilot, lambda: isinstance(app.screen, ManageScreen))
            assert not app.check_action("quit", ())
            assert "No saved models" in str(app.screen.query_one("#setup-page", Static).content)
            await pilot.press("ctrl+c")
            await until(pilot, lambda: not isinstance(app.screen, ManageScreen))
            assert app.check_action("quit", ())
            assert app.focused is app.query_one("#input", Input)

    asyncio.run(run())
