import os
import time
from pathlib import Path

import pytest

from angularai import tools
from angularai.agent import run_agent
from angularai.cli import _parser
from angularai.models import Action, Integration
from angularai.store import init_cortex, open_store
from angularai.tools import _run_env, execute
from tests.scripted import Scripted

posix = pytest.mark.skipif(os.name != "posix", reason="run is POSIX only")
SECRET = "OUTSIDE-SECRET cedar-plank-9"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "outside.txt").write_text(SECRET + "\n", encoding="utf-8")
    (tmp_path / "away").mkdir()
    (tmp_path / "away" / "far.txt").write_text(SECRET + "\n", encoding="utf-8")
    (work / "link.txt").symlink_to(tmp_path / "outside.txt")
    (work / "dl").symlink_to(tmp_path / "away")
    (work / "note.txt").write_text("The notebook is blue.\n", encoding="utf-8")
    return work


def run(root: Path, command: str) -> str:
    return execute(root, Action("run", command=command), allow_run=True)


def test_run_is_off_by_default(root: Path) -> None:
    result = execute(root, Action("run", command="cat note.txt"))
    assert "run is off" in result
    assert "notebook" not in result


def test_the_agent_does_not_run_without_the_flag(root: Path, tmp_path: Path) -> None:
    init_cortex(tmp_path / "cortex")
    store = open_store(tmp_path / "cortex")
    act = Action("run", command="cat note.txt")
    scripted = Scripted(script=(Integration("Running.", (), (), act), Integration("Done.", (), ())))
    run_agent(store, scripted, "Show the note.", root)
    assert "run is off" in scripted.seen[1].user
    scripted = Scripted(script=(Integration("Running.", (), (), act), Integration("Done.", (), ())))
    run_agent(store, scripted, "Show the note.", root, allow_run=True)
    assert "The notebook is blue." in scripted.seen[1].user


def test_the_cli_takes_allow_run() -> None:
    assert _parser().parse_args(["agent", "look"]).allow_run is False
    assert _parser().parse_args(["agent", "--allow-run", "look"]).allow_run is True


# R1: search followed a file symlink out of the directory.
def test_search_skips_a_symlink_that_leaves(root: Path) -> None:
    found = execute(root, Action("search", path=".", query="cedar"))
    assert "cedar" not in found
    assert execute(root, Action("search", path=".", query="notebook")).startswith("note.txt:1:")


def test_search_skips_a_directory_symlink_that_leaves(root: Path) -> None:
    assert "cedar" not in execute(root, Action("search", path=".", query="cedar"))
    assert "leaves the directory" in execute(root, Action("search", path="dl", query="cedar"))


def test_search_keeps_a_symlink_that_stays(root: Path) -> None:
    (root / "inner.txt").symlink_to(root / "note.txt")
    assert "inner.txt:1:" in execute(root, Action("search", path=".", query="notebook"))


def test_read_refuses_symlinks_that_leave(root: Path) -> None:
    assert "leaves the directory" in execute(root, Action("read", path="link.txt"))
    assert "leaves the directory" in execute(root, Action("read", path="dl/far.txt"))


def test_list_hides_symlinks_that_leave(root: Path) -> None:
    listed = execute(root, Action("list", path="."))
    assert "note.txt" in listed
    assert "link.txt" not in listed
    assert "dl" not in listed
    assert "leaves the directory" in execute(root, Action("list", path="dl"))


def test_edit_refuses_a_symlink_that_leaves(root: Path, tmp_path: Path) -> None:
    result = execute(root, Action("edit", path="link.txt", old="cedar", new="oak"))
    assert "leaves the directory" in result
    assert (tmp_path / "outside.txt").read_text(encoding="utf-8") == SECRET + "\n"


def test_write_refuses_a_symlink_that_leaves(root: Path, tmp_path: Path) -> None:
    over = Action("write", path="link.txt", text="gone", overwrite=True)
    assert "leaves the directory" in execute(root, over)
    assert (tmp_path / "outside.txt").read_text(encoding="utf-8") == SECRET + "\n"
    into = Action("write", path="dl/new.txt", text="planted")
    assert "leaves the directory" in execute(root, into)
    assert not (tmp_path / "away" / "new.txt").exists()


# R2: python -c read a file outside. R4b: find -exec ran sh.
@posix
@pytest.mark.parametrize(
    "command",
    [
        "python3 -c \"print(open('/etc/passwd').read())\"",
        "python -c \"import pathlib;print(list(pathlib.Path('..').iterdir()))\"",
        "find . -maxdepth 0 -exec sh -c 'ls /' ;",
        "xargs sh",
        "env",
        "sh -c 'echo hi'",
        "bash -c 'echo hi'",
        "perl -e 'print 1'",
        "node -e 1",
        "git status",
        "make",
        "chmod +x note.txt",
        "/bin/cat note.txt",
        "./note.txt",
    ],
)
def test_run_refuses_what_is_not_allowlisted(root: Path, command: str) -> None:
    result = run(root, command)
    assert result.startswith("Refused"), result
    assert "exit" not in result


# R3: the child saw the API keys.
@posix
def test_run_sees_a_scrubbed_environment(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-ABCDEFGHIJKLMNOP1234")
    env = _run_env(root)
    assert set(env) == {"PATH", "HOME", "LANG"}
    assert env["HOME"] == str(root)
    assert "OPENAI_API_KEY" not in env
    echoed = run(root, "echo $OPENAI_API_KEY $HOME")
    assert echoed == "exit 0\n$OPENAI_API_KEY $HOME"
    assert run(root, "pwd") == f"exit 0\n{root.resolve()}"


# R4: write a script, chmod it, run it.
@posix
def test_run_does_not_start_a_file_in_the_directory(root: Path) -> None:
    execute(root, Action("write", path="x", text="#!/bin/sh\necho shell-ran\n"))
    assert run(root, "chmod +x x").startswith("Refused chmod")
    (root / "x").chmod(0o755)
    assert run(root, "./x").startswith("Refused ./x")
    assert run(root, "x").startswith("Refused x")
    (root / "cat").write_text("#!/bin/sh\necho planted\n", encoding="utf-8")
    (root / "cat").chmod(0o755)
    assert run(root, "cat note.txt") == "exit 0\nThe notebook is blue."


# R5: cat read through a symlink.
@posix
@pytest.mark.parametrize(
    "command",
    [
        "cat link.txt",
        "cat -- link.txt",
        "cat dl/far.txt",
        "cat ../outside.txt",
        "head /etc/passwd",
        "grep -f/etc/passwd note.txt",
        "grep --file=/etc/passwd note.txt",
        "cp link.txt copy.txt",
        "ls dl",
    ],
)
def test_run_refuses_a_path_that_leaves(root: Path, command: str) -> None:
    result = run(root, command)
    assert "leaves the directory" in result
    assert "cedar" not in result


@posix
@pytest.mark.parametrize(
    "command",
    [
        "sort --compress-program=sh note.txt",
        "sort --compress-prog=sh note.txt",
        "wc --files0-from=note.txt",
        "cp -rL . ../copy",
        "cp --deref dl here",
        "grep -R cedar .",
        "grep --dereference-recursive cedar .",
    ],
)
def test_run_refuses_options_that_exec_or_follow_links(root: Path, command: str) -> None:
    assert run(root, command).startswith("Refused")


@posix
def test_recursive_grep_does_not_follow_links(root: Path) -> None:
    result = run(root, "grep -r cedar .")
    assert "cedar" not in result
    assert run(root, "grep -r notebook .").startswith("exit 0\n./note.txt:")


@posix
def test_run_output_is_capped(root: Path) -> None:
    (root / "big.txt").write_text("x" * 200_000, encoding="utf-8")
    result = run(root, "cat big.txt")
    assert result.startswith("exit ")
    assert "(output cut at 65536 bytes)" in result.splitlines()[0]
    assert len(result) <= 1700


@posix
def test_run_times_out_and_kills_the_group(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_RUN_TIMEOUT", 1)
    started = time.monotonic()
    assert run(root, "tail -f note.txt") == "The command timed out."
    assert time.monotonic() - started < 5
