"""The boot mark: the knot snaps together, then the wordmark and the version.

It plays once, about a second, before the setup offer and the session, only when stdin and stdout
are a terminal. It is left out under CI, TERM=dumb or no TERM, ANGULARAI_NO_BOOT=1,
on Windows, and in a terminal too small for the lockup. Under NO_COLOR it is one
still frame without color. Any key ends it. Ctrl-C ends it and goes on to quit.

Frames are pure: built from the terminal size and a color mode, never from the
terminal itself. ``play`` is the only part that writes. ``run`` decides whether to
play at all. Nothing here is imported until the session opens.
"""

import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal, TextIO

ColorMode = Literal["truecolor", "256", "plain"]
Rgb = tuple[int, int, int]
Pixel = Rgb | None
Outcome = Literal["done", "skipped"]

FPS = 30
FRAME_SECONDS = 1 / FPS
PINK: Rgb = (0xEC, 0x4A, 0x7F)
_SHADE: Rgb = (0x6E, 0x20, 0x3C)
_LIGHT: Rgb = (0xF7, 0xB6, 0xCB)
_DIM: Rgb = (0x8C, 0x86, 0x7F)
_BACK: Rgb = (0x0C, 0x0C, 0x0C)
_WHITE: Rgb = (0xFF, 0xFF, 0xFF)
_FACE: Rgb = (0x2C, 0x14, 0x1D)
_TONES: dict[str, Rgb] = {"k": PINK, "g": _SHADE, "w": _FACE}
_ESC = "\x1b["
# Terminals that know synchronized output draw a frame at once. Others ignore it.
_SYNC_ON = f"{_ESC}?2026h"
_SYNC_OFF = f"{_ESC}?2026l"
ENTER = f"{_ESC}?1049h{_ESC}?25l{_ESC}2J"
LEAVE = f"{_ESC}0m{_ESC}2J{_ESC}?25h{_ESC}?1049l"

# The v2 knot, traced from brand/v2 and sampled at 30 x 34 square pixels. Two pixel
# rows make one terminal row. k is the knot, g its shaded side,
# w the faces the logo leaves light, drawn dark here, as on a dark terminal.
_MARK = tuple(
    """
..............kk..............
.............kkkk.............
...........kkkkkkkk...........
.........kkkkkkkkkkkk.........
........kkkkkkwkkkkgg.........
......kkkkkkkwwgggggg..k......
....kkkkkkkwwwwggggggkkkkk....
...kkkkkkwwwwwwgggggwkkkkkkk..
...kkkkkwwwwwwkgggwwwwkkkkkk..
...kkkkwwwwwwkkkkkwwwwwwkkkk..
...kkkkggwwwkkkkkkkwwwwwgkkk..
...kkkkgggwwwkkkkkkkkwgggkkk..
...kkkkgggwwwwwkkkkkkggggkkk..
...kkkkgggwwwwwwkkkkkggggkkk..
..ggkkkggkkwwwwwwwkkkggggkkk..
.ggggkkgkkkkkwwwggkkkggggk....
kgggggkkkkkkkkwgggkkkkgg......
kkkggggkkkkkkkkgggkkkkg......k
kkkkgggkkkwkkkkgggkkkk.....kkk
kkkkgggkwwwkkkkgggkkkkkk..kkkk
kkkkgggwwwwkkkkgkkgkkkkkkkkkkk
kkkkgwwwwwwkkkkkkkgggkkkkkkkkk
kkkkwwwwwwgkkkkkkkggwwwkkkkkkk
kkkkkwwwwggkkkkkkkwwwwwwkkkkkk
kkkkkkkgggggwkkkggggwwwwgkkkkk
.kkkkkkggggwwwkggggggwgggkkkk.
...kkkkgggwwwwwgggggkggggkk...
.....kkgggkkwwwgggkkk..ggk....
......kg..kkkwwggkkkk.........
..........kkkkkkkkkkk.........
..........kkkkkkkkkk..........
...........kkkkkkkk...........
.............kkkk.............
...............k..............
""".split()
)
MARK_COLUMNS = len(_MARK[0])
MARK_ROWS = len(_MARK) // 2

# The wordmark, in the same square pixels: ten rows, five terminal rows. _ is the space.
_FONT = """
A       n       g       u       l   a       r     I   _
......  ......  ......  ......  ..  ......  ....  ..  ...
.####.  ......  ......  ......  ##  ......  ....  ##  ...
##..##  ......  ......  ......  ##  ......  ....  ##  ...
##..##  #####.  .#####  ##..##  ##  #####.  .###  ##  ...
######  ##..##  ##..##  ##..##  ##  ....##  ##..  ##  ...
##..##  ##..##  ##..##  ##..##  ##  .#####  ##..  ##  ...
##..##  ##..##  ##..##  ##..##  ##  ##..##  ##..  ##  ...
##..##  ##..##  .#####  .#####  ##  .#####  ##..  ##  ...
......  ......  ....##  ......  ..  ......  ....  ..  ...
......  ......  #####.  ......  ..  ......  ....  ..  ...
"""
WORD = "Angular AI"

_BLOCKS = 3
_TRAVEL = 12
_STAGGER = 3
_FLIGHT = 8
_HOLD = 9
_REVEAL = 7
STILL_SECONDS = 0.5


@dataclass(frozen=True)
class Layout:
    """Where the lockup sits on a terminal, in cells, top-left origin."""

    kind: Literal["wide", "compact"]
    width: int
    height: int
    top: int
    left: int

    @property
    def word_left(self) -> int:
        return self.left + MARK_COLUMNS + 4


def _glyphs() -> dict[str, tuple[str, ...]]:
    header, *rows = (line.split() for line in _FONT.strip().splitlines())
    return {
        " " if name == "_" else name: tuple(row[index] for row in rows)
        for index, name in enumerate(header)
    }


def _word_pixels() -> tuple[str, ...]:
    glyphs = _glyphs()
    rows = [""] * 10
    for index, char in enumerate(WORD):
        joined = index < len(WORD) - 1 and " " not in (char, WORD[index + 1])
        for row, line in enumerate(glyphs[char]):
            rows[row] += line + ("." if joined else "")
    return tuple(rows)


_WORD = _word_pixels()
WORD_COLUMNS = len(_WORD[0])
_TEXT_COLUMNS = len(WORD) + 2


def layout(width: int, height: int) -> Layout | None:
    """The biggest lockup that fits with a margin, or None when nothing does."""

    if height < MARK_ROWS + 2:
        return None
    top = (height - MARK_ROWS) // 2
    wide = MARK_COLUMNS + 4 + WORD_COLUMNS
    if width >= wide + 2:
        return Layout("wide", width, height, top, (width - wide) // 2)
    compact = MARK_COLUMNS + 4 + _TEXT_COLUMNS
    if width >= compact + 2:
        return Layout("compact", width, height, top, (width - compact) // 2)
    return None


def should_play(
    env: Mapping[str, str],
    stdin_tty: bool,
    stdout_tty: bool,
    platform: str = sys.platform,
) -> bool:
    """Only an interactive terminal that has not asked for quiet."""

    if not (stdin_tty and stdout_tty):
        return False
    if platform == "win32":
        return False
    if env.get("ANGULARAI_NO_BOOT", "") not in ("", "0"):
        return False
    if env.get("CI", "") not in ("", "0", "false"):
        return False
    return env.get("TERM", "dumb") not in ("", "dumb")


def color_mode(env: Mapping[str, str]) -> ColorMode | None:
    """Truecolor, else 256, else plain. None means NO_COLOR: one still, plain frame."""

    if env.get("NO_COLOR", ""):
        return None
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if "256color" in env.get("TERM", ""):
        return "256"
    return "plain"


@cache
def nearest_256(rgb: Rgb) -> int:
    """The xterm-256 index closest to rgb: the 6x6x6 cube or the grey ramp."""

    steps = (0, 95, 135, 175, 215, 255)

    def step(value: int) -> int:
        return min(range(6), key=lambda i: abs(steps[i] - value))

    r, g, b = (step(c) for c in rgb)
    cube = (steps[r], steps[g], steps[b])
    grey_index = max(0, min(23, round((sum(rgb) / 3 - 8) / 10)))
    grey = 8 + grey_index * 10

    def distance(other: Rgb) -> int:
        return sum((x - y) ** 2 for x, y in zip(rgb, other, strict=True))

    if distance((grey, grey, grey)) < distance(cube):
        return 232 + grey_index
    return 16 + 36 * r + 6 * g + b


def _ease(p: float) -> float:
    return 1 - (1 - p) ** 3


def _mix(a: Rgb, b: Rgb, t: float) -> Rgb:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


@cache
def _block_of(x: int, y: int) -> int:
    """Which third of the hexagon a pixel is in, clockwise from the top.

    Each third is a rhombus, one face of an isometric cube: three faces snap into one.
    """

    dx = x + 0.5 - MARK_COLUMNS / 2
    dy = y + 0.5 - len(_MARK) / 2
    angle = (math.atan2(dy, dx) + math.pi / 2) % math.tau
    return int(angle / (math.tau / _BLOCKS)) % _BLOCKS


@cache
def _direction(block: int) -> tuple[float, float]:
    angle = (block + 0.5) * math.tau / _BLOCKS - math.pi / 2
    return math.cos(angle), math.sin(angle)


def _assembly_frames() -> int:
    return (_BLOCKS - 1) * _STAGGER + _FLIGHT


def frame_count(still: bool = False) -> int:
    """Frames in the whole animation. A still is one frame."""

    return 1 if still else _assembly_frames() + _REVEAL + _HOLD


def duration(still: bool = False) -> float:
    """Seconds the animation takes, at FPS."""

    return STILL_SECONDS if still else frame_count() * FRAME_SECONDS


def mark_pixels(frame: int, shaded: bool = True) -> list[list[Pixel]]:
    """The mark at a frame, padded by the travel on every side. Later frames are final.

    Without shaded, only the knot is drawn: in one color the faces would be one shape.
    """

    pad = _TRAVEL
    rows = len(_MARK) + 2 * pad
    columns = MARK_COLUMNS + 2 * pad
    canvas: list[list[Pixel]] = [[None] * columns for _ in range(rows)]
    finish = _assembly_frames()
    flash = {finish - 1: 0.45, finish: 0.2}.get(frame, 0.0)
    for y, line in enumerate(_MARK):
        for x, tone in enumerate(line):
            if tone == "." or (tone != "k" and not shaded):
                continue
            block = _block_of(x, y)
            p = min(1.0, max(0.0, (frame - block * _STAGGER) / _FLIGHT))
            if p <= 0:
                continue
            dx, dy = _direction(block)
            left = (1 - _ease(p)) * _TRAVEL
            color = _mix(_BACK, _TONES[tone], min(1.0, 0.35 + p))
            if flash:
                color = _mix(color, _WHITE, flash)
            canvas[y + pad + round(dy * left)][x + pad + round(dx * left)] = color
    return canvas


def word_pixels(frame: int) -> list[list[Pixel]]:
    """The wordmark at a frame: it wipes in from the left behind a light edge."""

    shown = (frame - _word_start()) / _REVEAL * (WORD_COLUMNS + 2)
    canvas: list[list[Pixel]] = []
    for line in _WORD:
        row: list[Pixel] = []
        for x, bit in enumerate(line):
            if x > shown or bit != "#":
                row.append(None)
            else:
                row.append(_LIGHT if x > shown - 3 else PINK)
        canvas.append(row)
    return canvas


def _word_start() -> int:
    return _assembly_frames() - 2


Cell = tuple[str, str]


@cache
def _sgr(rgb: Rgb, mode: ColorMode, ground: int) -> str:
    if mode == "truecolor":
        return f"{ground};2;{rgb[0]};{rgb[1]};{rgb[2]}"
    return f"{ground};5;{nearest_256(rgb)}"


def cells(pixels: Sequence[Sequence[Pixel]], mode: ColorMode | None) -> list[list[Cell]]:
    """Half blocks: two pixel rows to one row of cells. Plain and still have no color."""

    out: list[list[Cell]] = []
    for top_row in range(0, len(pixels) - 1, 2):
        row: list[Cell] = []
        for top, bottom in zip(pixels[top_row], pixels[top_row + 1], strict=True):
            row.append(_cell(top, bottom, mode))
        out.append(row)
    return out


def _cell(top: Pixel, bottom: Pixel, mode: ColorMode | None) -> Cell:
    if top is None and bottom is None:
        return " ", ""
    if top is None or bottom is None:
        char = "▄" if top is None else "▀"
        lit = bottom if top is None else top
        if lit is None or mode is None or mode == "plain":
            return char, ""
        return char, _sgr(lit, mode, 38)
    if mode is None or mode == "plain":
        return "█", ""
    if top == bottom:
        return "█", _sgr(top, mode, 38)
    return "▀", f"{_sgr(top, mode, 38)};{_sgr(bottom, mode, 48)}"


def serialize(row: Sequence[Cell]) -> str:
    """One row of cells as text. A style is set only when it changes, and reset at the end."""

    out: list[str] = []
    last = ""
    end = len(row)
    while end and row[end - 1] == (" ", ""):
        end -= 1
    for char, style in row[:end]:
        if style != last:
            out.append(f"{_ESC}0m" if not style else f"{_ESC}0;{style}m")
            last = style
        out.append(char)
    if last:
        out.append(f"{_ESC}0m")
    return "".join(out)


def _stamp(
    screen: list[list[Pixel]], layer: Sequence[Sequence[Pixel]], top: int, left: int
) -> None:
    for y, line in enumerate(layer):
        for x, pixel in enumerate(line):
            sy, sx = top + y, left + x
            if pixel is not None and 0 <= sy < len(screen) and 0 <= sx < len(screen[sy]):
                screen[sy][sx] = pixel


def _write(row: list[Cell], column: int, text: str, style: str) -> None:
    for offset, char in enumerate(text):
        if 0 <= column + offset < len(row):
            row[column + offset] = (char, style)


def frame_cells(
    version: str, place: Layout, mode: ColorMode | None, frame: int
) -> list[list[Cell]]:
    """Every row of one frame, top to bottom. Rows the lockup never reaches are empty."""

    first = max(0, place.top - (_TRAVEL + 1) // 2)
    last = min(place.height, place.top + MARK_ROWS + (_TRAVEL + 1) // 2)
    reach = max(WORD_COLUMNS, _TEXT_COLUMNS, len(version) + 1)
    width = min(place.width, max(place.left + MARK_COLUMNS + _TRAVEL, place.word_left + reach))
    screen: list[list[Pixel]] = [[None] * width for _ in range((last - first) * 2)]
    mark = mark_pixels(frame, shaded=mode is not None and mode != "plain")
    _stamp(screen, mark, (place.top - first) * 2 - _TRAVEL, place.left - _TRAVEL)
    word_row = place.top + (MARK_ROWS - 6) // 2
    if place.kind == "wide":
        _stamp(screen, word_pixels(frame), (word_row - first) * 2, place.word_left)
    blank: list[list[Cell]] = [[] for _ in range(place.height)]
    grid = blank[:first] + cells(screen, mode) + blank[last:]
    reveal = (frame - _word_start()) / _REVEAL
    word_style = "" if mode is None or mode == "plain" else f"1;{_sgr(PINK, mode, 38)}"
    tag_style = "" if mode is None or mode == "plain" else _sgr(_DIM, mode, 38)
    if place.kind == "compact":
        shown = max(0, min(len(WORD), round(reveal * len(WORD))))
        _write(grid[word_row + 2], place.word_left, WORD[:shown], word_style)
        if reveal >= 1:
            _write(grid[word_row + 3], place.word_left, f"v{version}", tag_style)
    elif reveal >= 1:
        _write(grid[word_row + 5], place.word_left, f"v{version}", tag_style)
    return grid


def frame_lines(version: str, place: Layout, mode: ColorMode | None, frame: int) -> list[str]:
    """Every terminal row of one frame, as text with escape codes."""

    return [serialize(row) for row in frame_cells(version, place, mode, frame)]


def visible(text: str) -> str:
    """text without its escape codes."""

    out: list[str] = []
    skipping = False
    for char in text:
        if char == "\x1b":
            skipping = True
        elif skipping:
            skipping = not char.isalpha()
        else:
            out.append(char)
    return "".join(out)


def frames(version: str, width: int, height: int, mode: ColorMode | None) -> list[list[str]]:
    """Every frame's rows. Empty when the terminal is too small for any lockup."""

    place = layout(width, height)
    if place is None:
        return []
    if mode is None:
        return [frame_lines(version, place, mode, frame_count() - 1)]
    return [frame_lines(version, place, mode, frame) for frame in range(frame_count())]


def redraw(before: Sequence[str], after: Sequence[str]) -> str:
    """The writes that turn one frame into the next: only the rows that changed."""

    rows = [
        f"{_ESC}{row + 1};1H{_ESC}2K{line}"
        for row, line in enumerate(after)
        if row >= len(before) or before[row] != line
    ]
    return f"{_SYNC_ON}{''.join(rows)}{_SYNC_OFF}" if rows else ""


KeyWait = Callable[[float], bool]


def play(
    out: TextIO,
    shown: Sequence[Sequence[str]],
    seconds: float,
    key: KeyWait,
    clock: Callable[[], float] = time.monotonic,
) -> Outcome:
    """Draw each frame on the alternate screen. A key ends it. The screen is always left."""

    out.write(ENTER)
    try:
        start = clock()
        before: Sequence[str] = ()
        for index, frame in enumerate(shown):
            out.write(redraw(before, frame))
            out.flush()
            before = frame
            if key(max(0.0, start + (index + 1) * seconds - clock())):
                return "skipped"
        return "done"
    finally:
        out.write(LEAVE)
        out.flush()


def _key_waiter(fd: int) -> KeyWait:
    import select

    def wait(seconds: float) -> bool:
        ready, _, _ = select.select([fd], [], [], seconds)
        if ready:
            os.read(fd, 1024)
            return True
        return False

    return wait


def run(
    version: str,
    env: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> Outcome | None:
    """Play the boot mark when the terminal wants it. None when it was skipped up front."""

    env = os.environ if env is None else env
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    if not should_play(env, stdin.isatty(), stdout.isatty()):
        return None
    try:
        size = os.get_terminal_size(stdout.fileno())
        fd = stdin.fileno()
    except (OSError, ValueError):
        return None
    mode = color_mode(env)
    shown = frames(version, size.columns, size.lines, mode)
    if not shown:
        return None
    seconds = STILL_SECONDS if mode is None else FRAME_SECONDS
    return _with_cbreak(fd, lambda: play(stdout, shown, seconds, _key_waiter(fd)))


def _with_cbreak(fd: int, body: Callable[[], Outcome]) -> Outcome | None:
    """Run body with keys read one at a time and not echoed. The old mode always returns."""

    import termios
    import tty

    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        return body()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
