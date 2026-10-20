import io
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import pytest

from angularai import __version__, boot
from angularai.cli import main
from angularai.store import init_cortex

POSIX = sys.platform != "win32"
needs_pty = pytest.mark.skipif(not POSIX, reason="a pty is POSIX only")
TERMINAL = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}
SIZES = [(100, 30), (80, 24), (boot.MARK_COLUMNS + 4 + boot.WORD_COLUMNS + 2, 19), (48, 19)]


@pytest.mark.parametrize(
    ("env", "stdin_tty", "stdout_tty", "platform"),
    [
        (TERMINAL, False, True, "linux"),
        (TERMINAL, True, False, "linux"),
        (TERMINAL, False, False, "linux"),
        ({**TERMINAL, "ANGULARAI_NO_BOOT": "1"}, True, True, "linux"),
        ({**TERMINAL, "CI": "true"}, True, True, "linux"),
        ({**TERMINAL, "CI": "1"}, True, True, "darwin"),
        ({"TERM": "dumb"}, True, True, "linux"),
        ({"TERM": ""}, True, True, "linux"),
        ({}, True, True, "linux"),
        (TERMINAL, True, True, "win32"),
    ],
)
def test_it_does_not_play_where_it_is_not_wanted(
    env: dict[str, str], stdin_tty: bool, stdout_tty: bool, platform: str
) -> None:
    assert not boot.should_play(env, stdin_tty, stdout_tty, platform)


@pytest.mark.parametrize(
    "env",
    [
        TERMINAL,
        {"TERM": "xterm"},
        {**TERMINAL, "ANGULARAI_NO_BOOT": "0"},
        {**TERMINAL, "ANGULARAI_NO_BOOT": ""},
        {**TERMINAL, "CI": "false"},
        {**TERMINAL, "NO_COLOR": "1"},
    ],
)
def test_it_plays_in_an_interactive_terminal(env: dict[str, str]) -> None:
    assert boot.should_play(env, True, True, "linux")
    assert boot.should_play(env, True, True, "darwin")


def test_color_falls_back_from_truecolor_to_256_to_plain() -> None:
    assert boot.color_mode(TERMINAL) == "truecolor"
    assert boot.color_mode({"COLORTERM": "24bit", "TERM": "xterm"}) == "truecolor"
    assert boot.color_mode({"TERM": "xterm-256color"}) == "256"
    assert boot.color_mode({"TERM": "xterm"}) == "plain"
    assert boot.color_mode({**TERMINAL, "NO_COLOR": "1"}) is None
    assert boot.color_mode({**TERMINAL, "NO_COLOR": ""}) == "truecolor"


def test_the_nearest_256_color_is_in_the_cube_or_the_grey_ramp() -> None:
    assert boot.nearest_256(boot.PINK) == 204
    assert boot.nearest_256((0, 0, 0)) == 16
    assert boot.nearest_256((255, 255, 255)) == 231
    assert boot.nearest_256((128, 128, 128)) == 244
    assert boot.nearest_256((12, 12, 12)) == 232


def test_the_lockup_fits_or_is_not_drawn() -> None:
    wide = boot.layout(100, 30)
    assert wide is not None and wide.kind == "wide"
    compact = boot.layout(80, 24)
    assert compact is not None and compact.kind == "compact"
    assert boot.layout(40, 30) is None
    assert boot.layout(100, boot.MARK_ROWS + 1) is None
    assert boot.frames("0.1.0", 40, 30, "truecolor") == []


@pytest.mark.parametrize(("width", "height"), SIZES)
@pytest.mark.parametrize("mode", ["truecolor", "256", "plain", None])
def test_frames_are_deterministic_and_inside_the_terminal(
    width: int, height: int, mode: boot.ColorMode | None
) -> None:
    shown = boot.frames("0.1.0", width, height, mode)
    assert shown == boot.frames("0.1.0", width, height, mode)
    assert len(shown) == boot.frame_count(still=mode is None)
    for frame in shown:
        assert len(frame) == height
        for line in frame:
            assert len(boot.visible(line)) <= width
            assert "\n" not in line and "\r" not in line
            assert not line.endswith(" ")


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_last_frame_is_the_mark_the_wordmark_and_the_version(width: int, height: int) -> None:
    place = boot.layout(width, height)
    assert place is not None
    last = [boot.visible(line) for line in boot.frames("9.8.7", width, height, "truecolor")[-1]]
    text = "\n".join(last)
    assert "v9.8.7" in text
    if place.kind == "compact":
        assert "Angular AI" in text
    else:
        word = last[place.top + 5 : place.top + 10]
        assert all(set(row[place.word_left :]) - {" "} for row in word[:4])
    mark = [row[: place.word_left - 2] for row in last[place.top : place.top + boot.MARK_ROWS]]
    assert sum(row.count("█") + row.count("▀") + row.count("▄") for row in mark) > 200


def test_the_frames_build_up_and_settle() -> None:
    shown = boot.frames("0.1.0", 100, 30, "truecolor")
    inked = [sum(len(boot.visible(line).replace(" ", "")) for line in frame) for frame in shown]
    assert inked[0] < inked[-1] / 4
    assert shown[-1] == shown[-2]
    assert inked[len(inked) // 2] > inked[0]


def test_no_color_is_one_still_frame_without_escape_codes() -> None:
    for width, height in SIZES:
        shown = boot.frames("0.1.0", width, height, None)
        assert len(shown) == 1
        assert not any("\x1b" in line for line in shown[0])
        assert "v0.1.0" in "\n".join(shown[0])
    plain = boot.frames("0.1.0", 100, 30, "plain")
    assert not any("\x1b" in line for frame in plain for line in frame)
    assert boot.duration(still=True) == boot.STILL_SECONDS


def test_color_is_truecolor_pink_or_its_nearest_256() -> None:
    true = "".join(boot.frames("0.1.0", 100, 30, "truecolor")[-1])
    assert "38;2;236;74;127" in true
    assert "38;5;" not in true
    eight = "".join(boot.frames("0.1.0", 100, 30, "256")[-1])
    assert "38;5;204" in eight
    assert ";2;" not in eight


def test_the_animation_takes_under_a_second_and_a_quarter() -> None:
    assert 0.6 <= boot.duration() <= 1.2
    assert boot.frame_count() * boot.FRAME_SECONDS == boot.duration()


def test_a_redraw_writes_only_the_rows_that_changed() -> None:
    assert boot.redraw(["a", "b"], ["a", "b"]) == ""
    change = boot.redraw(["a", "b"], ["a", "c"])
    assert "\x1b[2;1H\x1b[2Kc" in change
    assert "1;1H" not in change
    assert boot.redraw([], ["a"]).count("H\x1b[2K") == 1


def test_visible_drops_escape_codes() -> None:
    assert boot.visible("\x1b[0;38;2;1;2;3mab\x1b[0m c\x1b[?25l") == "ab c"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _keys(pressed_at: int | None, interrupt_at: int | None = None) -> Callable[[float], bool]:
    calls = iter(range(10_000))

    def wait(seconds: float) -> bool:
        assert seconds >= 0
        index = next(calls)
        if interrupt_at is not None and index == interrupt_at:
            raise KeyboardInterrupt
        return pressed_at is not None and index >= pressed_at

    return wait


def test_play_draws_every_frame_and_always_leaves_the_screen() -> None:
    out = io.StringIO()
    shown = boot.frames("0.1.0", 100, 30, "truecolor")
    assert boot.play(out, shown, 0.0, _keys(None), Clock()) == "done"
    text = out.getvalue()
    assert text.startswith(boot.ENTER)
    assert text.endswith(boot.LEAVE)
    assert text.count(boot.LEAVE) == 1


def test_a_key_ends_play_at_once() -> None:
    out = io.StringIO()
    shown = boot.frames("0.1.0", 100, 30, "truecolor")
    assert boot.play(out, shown, 0.0, _keys(2), Clock()) == "skipped"
    assert out.getvalue().endswith(boot.LEAVE)
    assert out.getvalue().count("\x1b[?2026h") == 3


def test_ctrl_c_in_play_still_leaves_the_screen() -> None:
    out = io.StringIO()
    shown = boot.frames("0.1.0", 100, 30, "truecolor")
    with pytest.raises(KeyboardInterrupt):
        boot.play(out, shown, 0.0, _keys(None, interrupt_at=3), Clock())
    assert out.getvalue().endswith(boot.LEAVE)


def test_without_a_terminal_nothing_is_written() -> None:
    out = io.StringIO()
    assert boot.run("0.1.0", TERMINAL, io.StringIO(), out) is None
    assert out.getvalue() == ""


def _settings(fd: int) -> list[object]:
    """The terminal's mode. PENDIN is left out: BSD and macOS set it themselves when
    canonical input comes back on, to retype what is pending. It is not ours to restore.
    """

    import termios

    attrs: list[object] = list(termios.tcgetattr(fd))
    lflag = attrs[3]
    assert isinstance(lflag, int)
    attrs[3] = lflag & ~termios.PENDIN
    return attrs


@dataclass
class Pty:
    stdin: TextIO
    stdout: TextIO
    output: bytearray
    master: int
    slave: int

    def attrs(self) -> list[object]:
        return _settings(self.slave)

    def press(self, key: bytes, after: float) -> threading.Timer:
        timer = threading.Timer(after, os.write, (self.master, key))
        timer.start()
        return timer


@pytest.fixture
def terminal() -> Iterator[Pty]:
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    output = bytearray()
    stop = threading.Event()

    def drain() -> None:
        import select

        while not stop.is_set():
            ready, _, _ = select.select([master], [], [], 0.02)
            if ready:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    return

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    stdin = open(slave, closefd=False)
    stdout = open(slave, "w", closefd=False, encoding="utf-8")
    try:
        yield Pty(stdin, stdout, output, master, slave)
    finally:
        time.sleep(0.05)
        stop.set()
        reader.join()
        stdin.close()
        stdout.close()
        os.close(slave)
        os.close(master)


def _settle(term: Pty) -> bytes:
    time.sleep(0.1)
    return bytes(term.output)


@needs_pty
def test_in_a_pty_it_plays_and_restores_the_terminal(terminal: Pty) -> None:
    before = terminal.attrs()
    start = time.monotonic()
    assert boot.run("0.1.0", TERMINAL, terminal.stdin, terminal.stdout) == "done"
    elapsed = time.monotonic() - start
    assert boot.duration() * 0.9 <= elapsed <= boot.duration() + 1.0
    assert terminal.attrs() == before
    written = _settle(terminal).decode()
    assert written.startswith(boot.ENTER)
    assert written.endswith(boot.LEAVE)
    assert "38;2;236;74;127" in written


@needs_pty
def test_in_a_pty_any_key_skips_and_restores_the_terminal(terminal: Pty) -> None:
    before = terminal.attrs()
    timer = terminal.press(b"q", 0.1)
    start = time.monotonic()
    assert boot.run("0.1.0", TERMINAL, terminal.stdin, terminal.stdout) == "skipped"
    timer.join()
    assert time.monotonic() - start < boot.duration()
    assert terminal.attrs() == before
    assert _settle(terminal).decode().endswith(boot.LEAVE)


@needs_pty
def test_in_a_pty_ctrl_c_restores_the_terminal(
    terminal: Pty, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupting(fd: int) -> Callable[[float], bool]:
        return _keys(None, interrupt_at=2)

    monkeypatch.setattr(boot, "_key_waiter", interrupting)
    before = terminal.attrs()
    with pytest.raises(KeyboardInterrupt):
        boot.run("0.1.0", TERMINAL, terminal.stdin, terminal.stdout)
    assert terminal.attrs() == before
    assert _settle(terminal).decode().endswith(boot.LEAVE)


@needs_pty
def test_in_a_pty_no_color_shows_one_plain_frame(terminal: Pty) -> None:
    env = {**TERMINAL, "NO_COLOR": "1"}
    assert boot.run("0.1.0", env, terminal.stdin, terminal.stdout) == "done"
    body = _settle(terminal).decode().removeprefix(boot.ENTER).removesuffix(boot.LEAVE)
    assert "38;" not in body
    assert "v0.1.0" in body


@needs_pty
def test_in_a_small_or_quiet_pty_nothing_is_written(terminal: Pty) -> None:
    import fcntl
    import struct
    import termios

    assert (
        boot.run("0.1.0", {**TERMINAL, "ANGULARAI_NO_BOOT": "1"}, terminal.stdin, terminal.stdout)
        is None
    )
    assert boot.run("0.1.0", {**TERMINAL, "CI": "true"}, terminal.stdin, terminal.stdout) is None
    fcntl.ioctl(terminal.slave, termios.TIOCSWINSZ, struct.pack("HHHH", 10, 40, 0, 0))
    assert boot.run("0.1.0", TERMINAL, terminal.stdin, terminal.stdout) is None
    assert _settle(terminal) == b""


_CHILD = """
import json, sys, termios
from angularai import boot

def settings():
    attrs = termios.tcgetattr(0)
    attrs[3] &= ~termios.PENDIN
    return attrs

before = settings()
try:
    outcome = boot.run("0.1.0")
except KeyboardInterrupt:
    outcome = "interrupted"
after = settings()
with open(sys.argv[1], "w") as report:
    json.dump({"outcome": outcome, "restored": before == after}, report)
"""


def _child(tmp_path: Path, key: bytes) -> tuple[dict[str, object], bytes]:
    """The boot mark in a real child on a pty that is its controlling terminal."""

    import fcntl
    import pty
    import select
    import struct
    import termios

    report = tmp_path / "report.json"
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "ANGULARAI_NO_BOOT", "NO_COLOR")}
    env.update(TERMINAL)
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - the child
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        os.execve(sys.executable, [sys.executable, "-c", _CHILD, str(report)], env)
    output = bytearray()
    sent = False
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.05)
        if ready:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
        if not sent and boot.ENTER.encode() in output:
            os.write(master, key)
            sent = True
    os.waitpid(pid, 0)
    os.close(master)
    result: dict[str, object] = json.loads(report.read_text())
    return result, bytes(output)


@needs_pty
def test_a_real_key_in_a_real_terminal_skips(tmp_path: Path) -> None:
    result, output = _child(tmp_path, b"x")
    assert result == {"outcome": "skipped", "restored": True}
    assert output.decode().endswith(boot.LEAVE)


@needs_pty
def test_a_real_ctrl_c_in_a_real_terminal_restores_it(tmp_path: Path) -> None:
    result, output = _child(tmp_path, b"\x03")
    assert result == {"outcome": "interrupted", "restored": True}
    assert output.decode().endswith(boot.LEAVE)


def test_the_session_plays_it_and_a_command_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_cortex(tmp_path)
    played: list[str] = []
    monkeypatch.setattr("angularai.boot.run", lambda version: played.append(version))
    monkeypatch.setattr("angularai.cli.run_tui", lambda store, speak: 0)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["--cortex", str(tmp_path), "status"]) == 0
    assert played == []
    assert main(["--cortex", str(tmp_path)]) == 0
    assert played == [__version__]


def test_help_does_not_import_the_boot_mark() -> None:
    code = (
        "import sys\n"
        "from angularai.cli import main\n"
        "try:\n    main(['--help'])\nexcept SystemExit:\n    pass\n"
        "sys.stderr.write(str('angularai.boot' in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stderr == "False"
