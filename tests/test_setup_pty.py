"""The setup and model screens end to end in a real pseudo-terminal, and how the CLI picks them.

The child runs ``angularai`` with a temporary HOME, a keyring that refuses (so keys go to
the 0600 file under that HOME), no boot animation, and a fake provider on 127.0.0.1.
"""

import argparse
import errno
import fcntl
import json
import os
import re
import struct
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from angularai import onboard
from angularai.cli import main
from angularai.onboard import model_command, offer_setup, setup_command, tui_ok
from angularai.profiles import Profile, Vault, load_book
from angularai.session import session_command
from angularai.setup_tui import SetupResult
from angularai.store import init_cortex, open_store
from tests.setupfakes import KEY

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pty")

_WRAPPER = r"""
import sys, termios
before = termios.tcgetattr(0)
if sys.argv[1:2] == ["--crash"]:
    import angularai.setup_tui as ui
    real = ui.render_setup
    def render(state, **kw):
        if state.step == "key":
            raise RuntimeError("drawing failed")
        return real(state, **kw)
    ui.render_setup = render
    sys.argv.pop(1)
code = None
try:
    from angularai.cli import main
    code = main(sys.argv[1:])
except BaseException as exc:
    print("RAISED", type(exc).__name__)
after = termios.tcgetattr(0)
print("RESTORED" if before == after else "NOT-RESTORED")
print("EXIT", code)
"""
_ANSI = re.compile(r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]|\x1b[=>]|\x1b\][^\x07]*\x07")
_WAIT = 30.0


class _Provider(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"data": [{"id": "demo-model"}, {"id": "demo-large"}]})

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.headers.get("Authorization", "") not in {"", f"Bearer {KEY}"}:
            self._send(401, {"error": {"message": "bad key"}})
            return
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": "pong"}}]})

    def _send(self, status: int, data: object) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def provider() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


class Child:
    """angularai on the slave end of a pty, 100x30. Output is collected from the master."""

    def __init__(self, home: Path, *args: str) -> None:
        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "TERM": "xterm-256color",
            "ANGULARAI_NO_BOOT": "1",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        }
        self.process = subprocess.Popen(
            [sys.executable, "-c", _WRAPPER, *args],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=home,
            start_new_session=True,
        )
        os.close(slave)
        self.master = master
        self.raw = bytearray()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                chunk = os.read(self.master, 65536)
            except OSError as exc:
                if exc.errno in {errno.EIO, errno.EBADF}:
                    return
                raise
            if not chunk:
                return
            self.raw.extend(chunk)

    def text(self) -> str:
        return _ANSI.sub("", bytes(self.raw).decode("utf-8", "replace"))

    def wait_for(self, needle: str) -> None:
        squeezed = "".join(needle.split())
        deadline = time.monotonic() + _WAIT
        while time.monotonic() < deadline:
            if squeezed in "".join(self.text().split()):
                return
            if self.process.poll() is not None:
                time.sleep(0.2)
                if squeezed in "".join(self.text().split()):
                    return
                break
            time.sleep(0.05)
        raise AssertionError(f"never saw {needle!r}")

    def send(self, data: str, pause: float = 0.15) -> None:
        os.write(self.master, data.encode())
        time.sleep(pause)

    def finish(self) -> str:
        try:
            self.process.wait(timeout=_WAIT)
        finally:
            if self.process.poll() is None:
                self.process.kill()
        self._reader.join(timeout=5)
        os.close(self.master)
        return self.text()


def test_first_use_runs_setup_then_the_session_and_the_manage_view(
    tmp_path: Path, provider: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    child = Child(home)
    child.wait_for("No model is set up yet")
    child.wait_for("step 1 of 4")
    child.send("\x1b[F")
    child.send("\r")
    child.wait_for("Base URL")
    child.send(provider)
    child.send("\r")
    child.wait_for("Hidden as you type")
    child.send(KEY)
    child.send("\r")
    child.wait_for("demo-model")
    child.send("j")
    child.send("\r")
    child.wait_for("Connected in")
    child.send("\r")
    child.wait_for("The cortex is empty")
    child.send("/quit")
    child.send("\r")
    out = child.finish()
    raw = bytes(child.raw)
    assert child.process.returncode == 0
    assert "RESTORED" in out and "NOT-RESTORED" not in out
    assert "EXIT 0" in out
    assert "Saved compatible  compatible  demo-model" in out
    assert "Key sk-…a1b2" in out
    assert b"\x1b[?1049l" in raw
    assert KEY.encode() not in raw
    assert KEY[3:15].encode() not in raw
    keys = home / ".angularai" / "keys.json"
    assert oct(keys.stat().st_mode & 0o777) == "0o600"

    manage = Child(home, "model")
    manage.wait_for("Models")
    manage.wait_for("compatible")
    manage.send("d")
    manage.wait_for("Remove compatible and its key?")
    manage.send("y")
    manage.wait_for("Removed compatible and its key.")
    manage.send("q")
    out = manage.finish()
    assert "RESTORED" in out and "NOT-RESTORED" not in out and "EXIT 0" in out
    assert KEY.encode() not in bytes(manage.raw)
    environ = {"HOME": str(home)}
    assert load_book(environ).profiles == ()


@pytest.mark.parametrize("leave", ["\x03", "q"])
def test_leaving_setup_restores_the_terminal(tmp_path: Path, leave: str) -> None:
    home = tmp_path / "home"
    home.mkdir()
    child = Child(home, "setup")
    child.wait_for("step 1 of 4")
    child.send(leave)
    out = child.finish()
    assert "RESTORED" in out and "NOT-RESTORED" not in out
    assert "Nothing was saved." in out
    assert "EXIT 1" in out
    assert b"\x1b[?1049l" in bytes(child.raw)


def test_a_crash_inside_the_screens_still_restores_the_terminal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    child = Child(home, "--crash", "setup")
    child.wait_for("step 1 of 4")
    child.send("\r")
    out = child.finish()
    assert "RESTORED" in out and "NOT-RESTORED" not in out
    assert "drawing failed" in out
    assert b"\x1b[?1049l" in bytes(child.raw)


def test_a_resized_terminal_redraws(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    child = Child(home, "setup")
    child.wait_for("step 1 of 4")
    fcntl.ioctl(child.master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    child.process.send_signal(28)  # SIGWINCH
    before = len(child.raw)
    deadline = time.monotonic() + _WAIT
    while len(child.raw) == before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(child.raw) > before
    child.send("q")
    assert "RESTORED" in child.finish()


# --- which path the CLI takes -------------------------------------------------


def flags(**values: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "provider": None,
        "model_id": None,
        "base_url": None,
        "save_as": None,
        "key_stdin": False,
        "skip_test": False,
        "replace": False,
        "use": False,
        "model_command": None,
        "json": False,
    }
    base.update(values)
    return argparse.Namespace(**base)


class _Tty:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def test_the_screens_need_a_terminal_on_both_ends_and_no_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Tty(True))
    monkeypatch.setattr(sys, "stdout", _Tty(True))
    assert tui_ok({"TERM": "xterm"})
    assert tui_ok({"TERM": "xterm", "CI": "false"})
    assert not tui_ok({"TERM": "xterm", "CI": "true"})
    assert not tui_ok({"TERM": "dumb"})
    assert not tui_ok({})
    monkeypatch.setattr(sys, "stdout", _Tty(False))
    assert not tui_ok({"TERM": "xterm"})


def _saved(name: str = "openai") -> SetupResult:
    return SetupResult(Profile(name, "openai", "gpt-5", None, "file", "sk-…a1b2"), "Key x.", "")


def test_setup_on_a_terminal_opens_the_screens_and_prints_what_was_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    environ = {"ANGULARAI_HOME": str(tmp_path / "home")}
    calls: list[dict[str, object]] = []

    def fake_run(*_args: object, **kw: object) -> SetupResult | None:
        calls.append(kw)
        return results.pop(0)

    results: list[SetupResult | None] = [
        _saved(),
        SetupResult(_saved().profile, "Key x.", "OPENAI_API_KEY is set."),
        None,
    ]
    monkeypatch.setattr(onboard, "tui_ok", lambda _env: True)
    monkeypatch.setattr("angularai.setup_tui.run_setup_tui", fake_run)
    assert setup_command(flags(), environ) == 0
    out = capsys.readouterr().out
    assert "Saved openai  openai  gpt-5" in out
    assert "Switch with: angularai model use openai" in out
    assert model_command(flags(model_command="add"), environ) == 0
    assert "OPENAI_API_KEY is set." in capsys.readouterr().out
    assert calls[1]["make_active"] is True
    assert setup_command(flags(), environ) == 1
    assert "Nothing was saved." in capsys.readouterr().err
    assert len(calls) == 3


def test_flags_keep_the_prompt_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environ = {"ANGULARAI_HOME": str(tmp_path / "home")}
    monkeypatch.setattr(onboard, "tui_ok", lambda _env: True)

    def refuse(*_args: object, **_kw: object) -> None:
        raise AssertionError("the screens opened for a flagged command")

    monkeypatch.setattr("angularai.setup_tui.run_setup_tui", refuse)
    monkeypatch.setattr("angularai.setup_tui.run_manage_tui", refuse)
    assert onboard._flagless(flags())
    for flagged in (
        {"provider": "openai"},
        {"model_id": "m"},
        {"base_url": "http://h"},
        {"save_as": "x"},
        {"key_stdin": True},
    ):
        assert not onboard._flagless(flags(**flagged))
    assert model_command(flags(model_command="list", json=True), environ) == 0


def test_model_on_a_terminal_opens_the_manage_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = {"ANGULARAI_HOME": str(tmp_path / "home")}
    opened: list[object] = []
    monkeypatch.setattr(onboard, "tui_ok", lambda _env: True)
    monkeypatch.setattr("angularai.setup_tui.run_manage_tui", lambda *a, **k: opened.append(a))
    assert model_command(flags(), environ) == 0
    assert len(opened) == 1


def test_first_use_on_a_terminal_chains_into_the_screens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    environ = {"ANGULARAI_HOME": str(tmp_path / "home")}
    root = tmp_path / "cortex"
    seen: list[dict[str, object]] = []
    results: list[SetupResult | None] = [None, _saved()]

    def fake_run(*_args: object, **kw: object) -> SetupResult | None:
        seen.append(kw)
        return results.pop(0)

    monkeypatch.setattr(onboard, "tui_ok", lambda _env: True)
    monkeypatch.setattr("angularai.setup_tui.run_setup_tui", fake_run)
    monkeypatch.setattr(onboard, "system_terminal", lambda: _Interactive())
    assert offer_setup(root, environ) == "cancelled"
    assert not (root / "cortex.json").exists()
    assert offer_setup(root, environ) is None
    assert (root / "cortex.json").exists()
    assert seen[0]["first_use"] is True
    assert f"Cortex is at {root}" in capsys.readouterr().out


class _Interactive:
    interactive = True


def test_a_left_first_use_setup_ends_the_command_with_a_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("angularai.cli.offer_setup", lambda _root: "cancelled")
    monkeypatch.setenv("ANGULARAI_NO_BOOT", "1")
    monkeypatch.setenv("ANGULARAI_CORTEX", str(tmp_path / "c"))
    assert main([]) == 0
    assert "Setup was left." in capsys.readouterr().err
    assert main(["say", "hi"]) == 1


def test_off_a_terminal_nothing_changes(tmp_path: Path) -> None:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    done = subprocess.run(
        [sys.executable, "-m", "angularai", "model"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0
    assert "\x1b[?1049h" not in done.stdout
    assert "No saved models" in done.stdout + done.stderr


def test_the_line_session_lists_models_for_slash_model(tmp_path: Path) -> None:
    init_cortex(tmp_path / "c")
    store = open_store(tmp_path / "c")
    action, text = session_command(store, "/model", "")
    assert action == "continue" and text is not None and "No saved models" in text
    vault = Vault(os.environ)
    from angularai.profiles import add_profile

    add_profile(
        os.environ, vault, Profile("m", "openai", "gpt-5", None, "none", ""), KEY, make_active=True
    )
    _, text = session_command(store, "/model", "")
    assert text is not None and "gpt-5" in text and KEY not in text
