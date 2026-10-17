import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, Static

from angularai.cli import main, repl_line
from angularai.errors import InputRejected, SubstrateUnavailable
from angularai.models import Integration, Workspace
from angularai.pilot import pass_turn
from angularai.store import Store, init_cortex, open_store
from angularai.tui import AngularApp, Block, Speaker, run_tui, status_line
from tests.scripted import Scripted, noted


def test_status_line_names_the_budget_and_drops_what_does_not_fit(tmp_path: Path) -> None:
    store = _cortex(tmp_path)
    quiet = status_line(store, 400, busy=False)
    assert "unwired" in quiet
    assert "—/2000" in quiet
    assert "0 traces" in quiet
    assert str(tmp_path) in quiet
    assert "integrating" in status_line(store, 120, busy=True)
    assert status_line(store, 1, busy=False) == ""
    narrow = status_line(store, 24, busy=False)
    assert "\n" not in narrow
    assert len(narrow) <= 22
    assert str(tmp_path) not in narrow


def test_a_terminal_opens_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cortex(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    opened: list[Store] = []

    def fake(store: Store, speak: object) -> int:
        opened.append(store)
        assert speak is not None
        return 0

    monkeypatch.setattr("angularai.cli.run_tui", fake)
    assert main(["--cortex", str(tmp_path)]) == 0
    assert opened[0].root == tmp_path


def test_without_a_terminal_the_line_session_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cortex(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    lines = iter(["/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    assert main(["--cortex", str(tmp_path)]) == 0
    assert "angularai" in capsys.readouterr().out


def test_run_tui_returns_when_the_session_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _cortex(tmp_path)
    monkeypatch.setattr(AngularApp, "run", lambda self: None)
    assert run_tui(store, lambda _store, _text: "x") == 0


def test_the_session_shows_a_reply_and_remembers_the_line(tmp_path: Path) -> None:
    store = _cortex(tmp_path)
    release = threading.Event()

    def speak(_store: Store, text: str) -> str:
        assert text == "I have a blue notebook."
        release.wait(3)
        return "Noted."

    asyncio.run(_session(store, speak, release))


def test_a_failed_pass_stays_on_screen_and_a_crash_hides_its_detail(tmp_path: Path) -> None:
    store = _cortex(tmp_path)
    asyncio.run(_failures(store))


def test_slash_commands_and_an_earlier_pass(tmp_path: Path) -> None:
    store = _cortex(tmp_path)
    pass_turn(store, Scripted(noted("The notebook is blue.", "place")), "I have a blue notebook.")
    asyncio.run(_commands(store))


def test_clear_leaves_the_cortex_and_the_menu_runs_the_highlighted_command(
    tmp_path: Path,
) -> None:
    store = _cortex(tmp_path)
    pass_turn(store, Scripted(noted("The notebook is blue.", "place")), "I have a blue notebook.")
    assert repl_line(store, "/clear")[1] == "Screen cleared. The cortex is unchanged."
    assert repl_line(store, "/new")[1] == "Screen cleared. The cortex is unchanged."
    assert repl_line(store, "/exit")[0] == "quit"
    assert repl_line(store, "/budget")[1] == "2000 estimated tokens"
    asyncio.run(_clears(store))
    assert store.passes()


def test_a_long_reply_arrives_in_full(tmp_path: Path) -> None:
    store = _cortex(tmp_path)
    reply = "The cortex keeps this sentence on the screen as it arrives, one stretch at a time."
    asyncio.run(_arrives(store, reply))


def test_a_failed_call_is_replayed_without_a_trace(tmp_path: Path) -> None:
    store = _cortex(tmp_path)

    class Down:
        def integrate(self, workspace: Workspace) -> Integration:
            raise SubstrateUnavailable("The substrate is unreachable.")

    with pytest.raises(SubstrateUnavailable):
        pass_turn(store, Down(), "hello")
    asyncio.run(_sees(store, "unreachable"))


async def _session(store: Store, speak: Speaker, release: threading.Event) -> None:
    app = AngularApp(store, speak)
    async with app.run_test(size=(100, 28)) as pilot:
        empty = str(app.query_one("#empty", Static).content)
        assert "cortex is empty" in empty
        field = app.query_one("#input", Input)
        field.value = "I have a blue notebook."
        await pilot.press("enter")
        assert field.disabled
        assert app.query("#pending")
        status = str(app.query_one("#status", Static).content)
        assert "integrating" in status
        assert any(frame in status for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        release.set()
        await _until(pilot, "Noted.")
        assert not app.query("#pending")
        assert not app.query_one("#input", Input).disabled
        bodies = [block.body_text for block in app.query(Block)]
        assert bodies == ["I have a blue notebook.", "Noted."]
        await pilot.press("up")
        assert app.query_one("#input", Input).value == "I have a blue notebook."
        await pilot.press("down")
        assert app.query_one("#input", Input).value == ""


async def _failures(store: Store) -> None:
    calls = {"n": 0}

    def speak(_store: Store, _text: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise InputRejected(
                "That input looks like a secret. It was not sent and nothing was written."
            )
        raise RuntimeError("sk-this-must-not-render")

    app = AngularApp(store, speak)
    async with app.run_test(size=(100, 28)) as pilot:
        field = app.query_one("#input", Input)
        field.value = "hello"
        await pilot.press("enter")
        await _until(pilot, "nothing was written")
        assert not field.disabled
        field.value = "again"
        await pilot.press("enter")
        await _until(pilot, "Nothing new was written.")
        rendered = " ".join(block.body_text for block in app.query(Block))
        assert "sk-this-must-not-render" not in rendered


async def _commands(store: Store) -> None:
    app = AngularApp(store, lambda _store, _text: "unused")
    async with app.run_test(size=(100, 30)) as pilot:
        bodies = [block.body_text for block in app.query(Block)]
        assert "I have a blue notebook." in bodies
        assert "Noted." in bodies
        assert "traces" in str(app.query_one("#status", Static).content)
        field = app.query_one("#input", Input)
        await pilot.press("enter")
        assert field.value == ""
        await pilot.press("/", "c", "o")
        await pilot.pause()
        hints = " ".join(str(hint.content) for hint in app.query(".hint").results(Static))
        assert "/cortex" in hints
        assert "/quit" not in hints
        await pilot.press("tab")
        assert field.value.startswith("/cortex")
        field.value = "/cortex"
        await pilot.press("enter")
        await _until(pilot, "The notebook is blue.")
        field.value = "/recall"
        await pilot.press("enter")
        await _until(pilot, "/recall needs a query.")
        field.value = "/nope"
        await pilot.press("enter")
        await _until(pilot, "Unknown command")
        field.value = "/help"
        await pilot.press("enter")
        await _until(pilot, "/status")
        field.value = "/status"
        await pilot.press("enter")
        await _until(pilot, "no model")
        field.value = "/"
        await pilot.press("tab")
        assert field.value == "/"


async def _clears(store: Store) -> None:
    app = AngularApp(store, lambda _store, _text: "unused")
    async with app.run_test(size=(100, 30)) as pilot:
        assert app.query(Block)
        field = app.query_one("#input", Input)
        field.value = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        assert not app.query(Block)
        await pilot.press("ctrl+l")
        field.value = "/"
        await pilot.pause()
        assert "/clear" in str(app.query_one(".on", Static).content)
        await pilot.press("down")
        await pilot.pause()
        assert "/cortex" in str(app.query_one(".on", Static).content)
        await pilot.press("enter")
        await _until(pilot, "The notebook is blue.")
        field.value = "/locks"
        await pilot.press("enter")
        await _until(pilot, "No locks yet")


# A deadline, not a count of pauses. Coverage tracing slows the reveal.
_WAIT = 10.0


async def _arrives(store: Store, reply: str) -> None:
    app = AngularApp(store, lambda _store, _text: reply)
    async with app.run_test(size=(88, 24)) as pilot:
        field = app.query_one("#input", Input)
        field.value = "hello"
        await pilot.press("enter")
        await _until(pilot, reply)
        body = app.query(Block)[-1].query_one(".body", Static)
        deadline = time.monotonic() + _WAIT
        while time.monotonic() < deadline:
            await pilot.pause(0.05)
            if str(body.content) == reply:
                return
        raise AssertionError(str(body.content))


async def _sees(store: Store, needle: str) -> None:
    app = AngularApp(store, lambda _store, _text: "unused")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        rendered = " ".join(block.body_text for block in app.query(Block))
        assert needle in rendered


async def _until(pilot: Pilot[None], needle: str) -> None:
    deadline = time.monotonic() + _WAIT
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if any(needle in block.body_text for block in pilot.app.query(Block)):
            return
    raise AssertionError(needle)


def _cortex(path: Path) -> Store:
    init_cortex(path)
    return open_store(path)
