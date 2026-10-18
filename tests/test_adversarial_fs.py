"""Filesystem attacks on the agent's tools: links, races, odd files, size."""

import os
import socket
import stat
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from angularai import tools
from angularai.models import Action
from angularai.tools import execute

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX filesystem semantics")
SECRET = "OUTSIDE-SECRET cedar-plank-9"


@pytest.fixture
def base() -> Iterator[Path]:
    # Short and resolved: an AF_UNIX path on macOS must stay under 104 bytes.
    with tempfile.TemporaryDirectory(dir="/tmp") as raw:
        yield Path(raw).resolve()


@pytest.fixture
def root(base: Path) -> Path:
    work = base / "w"
    (work / "sub").mkdir(parents=True)
    (work / "note.txt").write_text("The notebook is blue.\n", encoding="utf-8")
    (work / "sub" / "inner.txt").write_text("inner\n", encoding="utf-8")
    (base / "outside.txt").write_text(SECRET + "\n", encoding="utf-8")
    (base / "away").mkdir()
    (base / "away" / "far.txt").write_text(SECRET + "\n", encoding="utf-8")
    return work


def act(root: Path, tool: str, **fields: object) -> str:
    return execute(root, Action(tool, **fields))  # type: ignore[arg-type]


def every_tool(root: Path, path: str) -> list[str]:
    return [
        act(root, "read", path=path),
        act(root, "list", path=path),
        act(root, "search", path=path, query="cedar"),
        act(root, "edit", path=path, old="cedar", new="oak"),
        act(root, "write", path=path, text="planted", overwrite=True),
        act(root, "write", path=f"{path}/planted.txt", text="planted"),
    ]


def untouched(base: Path) -> None:
    assert (base / "outside.txt").read_text(encoding="utf-8") == SECRET + "\n"
    assert (base / "away" / "far.txt").read_text(encoding="utf-8") == SECRET + "\n"
    assert not (base / "away" / "planted.txt").exists()
    assert sorted(item.name for item in (base / "away").iterdir()) == ["far.txt"]


# Links.


def test_a_chain_of_links_that_ends_outside_is_refused(root: Path, base: Path) -> None:
    (root / "c").symlink_to(base / "outside.txt")
    (root / "b").symlink_to("c")
    (root / "a").symlink_to("b")
    (root / "d").symlink_to(base / "away")
    (root / "e").symlink_to("d")
    for name in ("a", "b", "c", "e", "e/far.txt"):
        for result in every_tool(root, name):
            assert "cedar" not in result
    untouched(base)


def test_a_chain_of_links_that_stays_inside_is_followed(root: Path) -> None:
    (root / "b").symlink_to("note.txt")
    (root / "a").symlink_to("b")
    (root / "s").symlink_to("sub")
    assert "notebook" in act(root, "read", path="a")
    assert "inner" in act(root, "read", path="s/inner.txt")
    assert act(root, "edit", path="a", old="blue", new="green") == "Edited note.txt."
    assert (root / "note.txt").read_text(encoding="utf-8") == "The notebook is green.\n"
    assert (root / "a").is_symlink()


def test_a_link_loop_is_refused_without_hanging(root: Path, base: Path) -> None:
    (root / "loop").symlink_to("loop")
    (root / "p").symlink_to("q")
    (root / "q").symlink_to("p")
    started = time.monotonic()
    for name in ("loop", "p", "q/x"):
        for result in every_tool(root, name):
            assert "cedar" not in result
    assert "notebook" in act(root, "search", path=".", query="notebook")
    assert time.monotonic() - started < 10
    untouched(base)


def test_a_hard_link_is_read_but_never_written_through(root: Path, base: Path) -> None:
    """A hard link cannot be told apart by path. A write replaces the name only."""

    try:
        os.link(base / "outside.txt", root / "hard.txt")
    except OSError:
        pytest.skip("hard links are not supported here")
    os.chmod(root / "hard.txt", 0o640)
    # The read follows it: the content is in the directory the user gave.
    assert "cedar" in act(root, "read", path="hard.txt")
    assert act(root, "edit", path="hard.txt", old="cedar", new="oak") == "Edited hard.txt."
    assert "oak" in (root / "hard.txt").read_text(encoding="utf-8")
    assert stat.S_IMODE((root / "hard.txt").stat().st_mode) == 0o640
    assert (root / "hard.txt").stat().st_nlink == 1
    act(root, "write", path="hard.txt", text="replaced", overwrite=True)
    untouched(base)


# Races between the check and the open.


def _swap_after_check(
    monkeypatch: pytest.MonkeyPatch, swap: Callable[[], None]
) -> list[Path | None]:
    seen: list[Path | None] = []
    original = tools._inside

    def checked_then_swapped(root: Path, raw: str) -> Path | None:
        found = original(root, raw)
        seen.append(found)
        swap()
        return found

    monkeypatch.setattr(tools, "_inside", checked_then_swapped)
    return seen


def test_a_directory_swapped_for_a_link_after_the_check_is_not_followed(
    root: Path, base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def swap() -> None:
        if not (root / "sub").is_symlink():
            (root / "sub").rename(root / "sub.old")
            (root / "sub").symlink_to(base / "away")

    seen = _swap_after_check(monkeypatch, swap)
    result = act(root, "write", path="sub/planted.txt", text="planted")
    assert seen[0] == root / "sub" / "planted.txt"
    assert "link" in result
    untouched(base)
    edited = act(root, "edit", path="sub/far.txt", old="cedar", new="oak")
    assert "cedar" not in edited
    assert "cedar" not in act(root, "read", path="sub/far.txt")
    untouched(base)


@pytest.mark.parametrize(
    ("tool", "fields"),
    [
        ("read", {}),
        ("edit", {"old": "notebook", "new": "oak"}),
        ("write", {"text": "gone", "overwrite": True}),
        ("search", {"query": "cedar"}),
    ],
)
def test_a_file_swapped_for_a_link_after_the_check_is_not_followed(
    root: Path, base: Path, monkeypatch: pytest.MonkeyPatch, tool: str, fields: dict[str, object]
) -> None:
    def swap() -> None:
        target = root / "note.txt"
        if not target.is_symlink():
            target.unlink()
            target.symlink_to(base / "outside.txt")

    _swap_after_check(monkeypatch, swap)
    result = act(root, tool, path="note.txt", **fields)
    assert "cedar" not in result
    if tool != "search":
        assert "link" in result
    untouched(base)


def test_a_file_swapped_for_a_fifo_after_the_check_does_not_hang(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def swap() -> None:
        target = root / "note.txt"
        if not stat.S_ISFIFO(target.lstat().st_mode):
            target.unlink()
            os.mkfifo(target)

    _swap_after_check(monkeypatch, swap)
    assert "FIFO" in within(5, lambda: act(root, "read", path="note.txt"))


# Traversal spellings. None decodes to a parent; each stays a literal name or resolves.


@pytest.mark.parametrize(
    "raw",
    [
        "../outside.txt",
        "%2e%2e/outside.txt",
        "..%2foutside.txt",
        "\u2024\u2024/outside.txt",
        "\uff0e\uff0e/outside.txt",
        "sub/../../outside.txt",
        "sub/./../../away/far.txt",
        "./../w/../outside.txt",
        "../outside.txt/",
        "..//outside.txt",
        "//outside.txt",
        "/../outside.txt",
        "sub/" + "../" * 40 + "outside.txt",
        "../W/../outside.txt",
        "..\\outside.txt",
        "~/../outside.txt",
    ],
)
def test_traversal_spellings_never_leave(root: Path, base: Path, raw: str) -> None:
    for result in every_tool(root, raw):
        assert "cedar" not in result
    untouched(base)


def test_absolute_paths_inside_are_accepted_and_outside_refused(root: Path, base: Path) -> None:
    assert "notebook" in act(root, "read", path=str(root / "note.txt"))
    assert "notebook" in act(root, "read", path=str(root / "sub" / ".." / "note.txt"))
    assert "notebook" in act(root, "read", path="note.txt/")  # a trailing slash is dropped
    assert "leaves the directory" in act(root, "read", path=str(base / "outside.txt"))
    assert "leaves the directory" in act(root, "read", path=str(root) + "/../outside.txt")


# Files that are not regular files.


def within(seconds: float, call: Callable[[], str]) -> str:
    box: list[str] = []
    worker = threading.Thread(target=lambda: box.append(call()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert box, f"the tool did not return within {seconds} s"
    return box[0]


def test_a_fifo_is_refused_and_nothing_hangs(root: Path) -> None:
    os.mkfifo(root / "pipe")
    for result in within(10, lambda: "\n".join(every_tool(root, "pipe"))).split("\n"):
        assert "Wrote" not in result and "Edited" not in result
    assert "notebook" in within(10, lambda: act(root, "search", path=".", query="notebook"))
    assert stat.S_ISFIFO((root / "pipe").lstat().st_mode)


def test_a_socket_is_refused(root: Path) -> None:
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(root / "sock"))
        for result in within(10, lambda: "\n".join(every_tool(root, "sock"))).split("\n"):
            assert "Wrote" not in result and "Edited" not in result
        assert "socket" in act(root, "read", path="sock")
        assert "notebook" in act(root, "search", path=".", query="notebook")
    finally:
        server.close()


def test_a_device_is_outside_or_refused(root: Path) -> None:
    (root / "zero").symlink_to("/dev/zero")
    assert "leaves the directory" in within(5, lambda: act(root, "read", path="zero"))
    assert "leaves the directory" in within(5, lambda: act(root, "read", path="/dev/zero"))


# Size and shape.


def test_a_huge_file_is_read_and_searched_in_bounded_memory(root: Path) -> None:
    line = b"filler line without the word\n"
    with open(root / "huge.txt", "wb") as handle:
        for _ in range(24 * 1024 * 1024 // len(line)):
            handle.write(line)
        handle.write(b"needle at the very end\n")
    tracemalloc.start()
    try:
        read = act(root, "read", path="huge.txt")
        late = act(root, "read", path="huge.txt", start=900_000)
        searched = act(root, "search", path=".", query="needle")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert read.startswith("1|filler") and len(read) <= 1600
    assert "4 MiB" in late
    assert "huge.txt" not in searched  # past the per-file cap; nothing beyond it is read
    assert peak < 48 * 1024 * 1024
    assert "too large to edit" in act(root, "edit", path="huge.txt", old="filler", new="x")


def test_a_very_long_line_is_clipped(root: Path) -> None:
    (root / "long.txt").write_text("x" * 3_000_000, encoding="utf-8")
    result = act(root, "read", path="long.txt")
    assert len(result) <= 1600
    assert result.startswith("1|xxx")


@pytest.mark.parametrize(
    "data",
    [
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        bytes(range(256)) * 8,
        "caf\xe9 au lait\n".encode("latin-1"),
        b"text\n" * 100 + b"\x00",
    ],
)
def test_a_binary_file_is_refused_not_dumped(root: Path, data: bytes) -> None:
    (root / "blob.bin").write_bytes(data)
    result = act(root, "read", path="blob.bin")
    assert result == f"That file is binary ({len(data)} bytes). It was not read."
    assert "\x00" not in act(root, "search", path=".", query="I")
    assert "binary" in act(root, "edit", path="blob.bin", old="text", new="x")


def test_many_files_are_capped(root: Path) -> None:
    many = root / "many"
    many.mkdir()
    for index in range(3000):
        (many / f"f{index:05d}.txt").write_text("needle\n", encoding="utf-8")
    listed = act(root, "list", path="many").splitlines()
    assert len(listed) == 81
    assert listed[-1] == "... 2920 more"
    hits = act(root, "search", path="many", query="needle").splitlines()
    assert len(hits) == 30


def test_a_deep_tree_is_searched_without_recursion(root: Path) -> None:
    folder = root
    for _ in range(300):
        folder = folder / "d"
    folder.mkdir(parents=True)
    (folder / "bottom.txt").write_text("needle\n", encoding="utf-8")
    found = act(root, "search", path=".", query="needle")
    assert found.endswith("bottom.txt:1:needle")
    relative = str(folder.relative_to(root) / "bottom.txt")
    assert "needle" in act(root, "read", path=relative)
    assert act(root, "write", path=relative + ".new", text="x") == f"Wrote {relative}.new."
