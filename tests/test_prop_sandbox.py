"""Properties of the path sandbox and of run's argument check."""

import errno
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from angularai.models import Action
from angularai.tools import _arg_inside, _inside, _refuse, execute

posix = pytest.mark.skipif(os.name != "posix", reason="symlinks and run are POSIX only")


@pytest.fixture(scope="module")
def tree() -> Iterator[Path]:
    """A root with links in and out of it. Shared by every example; never written."""

    with tempfile.TemporaryDirectory() as raw:
        base = Path(raw).resolve()
        root = base / "root"
        (root / "sub" / "deep").mkdir(parents=True)
        (root / "sub" / "deep" / "note.txt").write_text("inside\n", encoding="utf-8")
        (root / "ünï 😀").mkdir()
        (base / "outside").mkdir()
        (base / "outside" / "secret.txt").write_text("OUTSIDE\n", encoding="utf-8")
        (base / "rootkit").mkdir()  # shares a prefix with root
        (root / "out").symlink_to(base / "outside")
        (root / "up").symlink_to("..")
        (root / "etc").symlink_to("/etc")
        (root / "in").symlink_to(root / "sub")
        (root / "rel").symlink_to("sub/deep")
        (root / "loop").symlink_to("loop")
        (root / "dangling").symlink_to(base / "missing")
        yield root


NAMES = ["sub", "deep", "note.txt", "ünï 😀", "out", "up", "etc", "in", "rel", "loop"]
segment = st.one_of(
    st.sampled_from([*NAMES, "dangling", "..", ".", "", "rootkit", "outside", "secret.txt"]),
    st.text(min_size=1, max_size=12).filter(lambda part: "/" not in part),
)
prefix = st.sampled_from(["", "", "/", "./", "../", "//", "~/"])
raw_paths = st.builds(
    lambda head, parts: head + "/".join(parts), prefix, st.lists(segment, max_size=6)
)


def _outside(root: Path, raw: str) -> bool:
    """The independent answer: where the path really lands, every link followed."""

    target = Path(raw) if Path(raw).is_absolute() else root / raw
    return not Path(os.path.realpath(target)).is_relative_to(root)


def _loops(root: Path, raw: str) -> bool:
    """A link loop. Python 3.12 cannot resolve one, so the resolver refuses it."""

    target = Path(raw) if Path(raw).is_absolute() else root / raw
    try:
        os.stat(target)
    except OSError as exc:
        return exc.errno == errno.ELOOP
    return False


@posix
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw=raw_paths)
# Found by Hypothesis: a NUL byte made resolve() raise ValueError out of the tool.
@example(raw="a\x00b")
@example(raw="../root/../outside/secret.txt")
@example(raw="out/secret.txt")
@example(raw="up/outside/secret.txt")
@example(raw="/etc/passwd")
@example(raw="loop")
@example(raw="loop/x")
def test_the_resolver_never_leaves_the_root(tree: Path, raw: str) -> None:
    found = _inside(tree, raw)
    if found is not None:
        assert found.is_relative_to(tree)
        assert not _outside(tree, raw)
    elif raw.strip() and "\x00" not in raw:
        assert _outside(tree, raw) or _loops(tree, raw)


inside_parts = st.lists(
    st.sampled_from(["sub", "deep", "note.txt", "ünï 😀", "in", "rel", "x"]), min_size=1, max_size=4
)


@posix
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(parts=inside_parts, dotted=st.booleans())
def test_a_path_inside_is_always_accepted(tree: Path, parts: list[str], dotted: bool) -> None:
    raw = ("./" if dotted else "") + "/".join(parts)
    found = _inside(tree, raw)
    assert found is not None
    assert found == Path(os.path.realpath(tree / raw))
    assert _arg_inside(tree, raw)


@posix
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(raw=raw_paths, tool=st.sampled_from(["list", "read", "search", "edit"]))
@example(raw="a\x00b", tool="read")
def test_a_tool_never_crashes_or_reads_outside(tree: Path, raw: str, tool: str) -> None:
    action = Action(tool, path=raw, query="OUTSIDE", old="zz-not-there", new="y")
    result = execute(tree, action)
    assert "OUTSIDE" not in result.replace("query", "")


options = st.sampled_from(["-n", "-l", "-r", "-tx", "-o", "--count", "--output=", "-L", "-H", "-R"])
argument = st.one_of(raw_paths, options, st.builds(lambda o, p: o + p, options, raw_paths))
program = st.sampled_from(["cat", "cp", "grep", "sort", "wc", "ls", "head", "mv", "rm"])


@posix
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(name=program, args=st.lists(argument, max_size=5))
@example(name="cat", args=["a\x00b"])
@example(name="sort", args=["--output=../x"])
@example(name="cp", args=["-L", "in", "x"])
def test_an_accepted_run_names_only_paths_inside(tree: Path, name: str, args: list[str]) -> None:
    if _refuse(tree, [name, *args]) is not None:
        return
    for part in args:
        if part.startswith("--"):
            value = part.split("=", 1)[1] if "=" in part else ""
            assert not value or not _outside(tree, value)
        elif part.startswith("-") and len(part) > 1:
            assert "/" not in part and ".." not in part
        else:
            assert not _outside(tree, part)
