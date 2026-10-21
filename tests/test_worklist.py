import json
import os
from pathlib import Path

import pytest

from angularai import run as run_loop
from angularai.errors import InputRejected
from angularai.worklist import summary_problems, unit_text, work_left
from tests.scripted import Scripted
from tests.test_run import act, begin, finished


def module(root: Path, name: str, doc: str, body: str = "return x + 1") -> Path:
    path = root / "src" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'def {name}(x):\n    """{doc}"""\n    {body}\n', encoding="utf-8")
    return path


def age(path: Path, seconds: float) -> None:
    stamp = path.stat().st_mtime - seconds
    os.utime(path, (stamp, stamp))


def test_work_left_counts_every_marker_line_not_twenty(tmp_path: Path) -> None:
    for n in range(45):
        module(tmp_path, f"m{n:02d}", "TODO: describe.")
    (tmp_path / "notes.md").write_text("FIXME one\nfine\nTODO and FIXME\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00TODO\x00")
    left = work_left(tmp_path, ("TODO", "FIXME"), shown=5)
    assert left.count == 47 and len(left.files) == 46 and len(left.lines) == 5
    assert left.lines[0].startswith("notes.md:1:FIXME one")
    assert work_left(tmp_path, ()).count is None
    assert work_left(tmp_path, ("NOPE",)).count == 0


def test_work_left_is_none_when_the_root_is_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for n in range(4):
        (tmp_path / f"f{n}.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr("angularai.worklist.COUNT_FILES", 3)
    assert work_left(tmp_path, ("TODO",)).count is None


def test_unit_text_is_the_docstring_or_the_first_lines() -> None:
    assert unit_text('"""Mod."""\ndef f():\n    """Fn."""\n', ".py") == "Mod."
    assert unit_text('def f():\n    """Fn."""\n', ".py") == "Fn."
    assert unit_text("def broken(:\n", ".py") == "def broken(:"
    assert unit_text("\n# Title\nbody\nmore\nlast\n", ".md") == "# Title body more"


def summary_root(tmp_path: Path) -> Path:
    module(tmp_path, "amber", "Returns x plus 6.")
    module(tmp_path, "basil", "Returns x multiplied by 5.")
    module(tmp_path, "cedar", "Returns x plus 3.")
    return tmp_path


def test_a_specific_summary_written_last_passes(tmp_path: Path) -> None:
    root = summary_root(tmp_path)
    (root / "SUMMARY.md").write_text(
        "# Summary\n- amber.py: adds six to x\n- basil: x times 5\n- `src/cedar.py`: x + 3\n",
        encoding="utf-8",
    )
    assert summary_problems(root, "SUMMARY.md", "src/*.py") == []
    assert summary_problems(root, "SUMMARY.md", "") == []


def test_the_summary_must_exist_be_last_list_all_and_be_specific(tmp_path: Path) -> None:
    root = summary_root(tmp_path)
    assert summary_problems(root, "SUMMARY.md", "src/*.py") == [
        "SUMMARY.md is not written. Write it last."
    ]
    assert summary_problems(root, "../out.md", "src/*.py") == ["../out.md is outside the root."]
    summary = root / "SUMMARY.md"
    summary.write_text(
        "- amber: applies a small arithmetic change\n- basil: multiplies by 5\n", encoding="utf-8"
    )
    age(summary, 60)
    problems = summary_problems(root, "SUMMARY.md", "src/*.py")
    assert problems[0].startswith("SUMMARY.md was written before 3 of the files")
    assert problems[1] == "SUMMARY.md does not list 1 of 3: cedar"
    assert problems[2].startswith("SUMMARY.md is not specific about 1: amber.")


def test_one_line_said_of_different_files_is_not_specific(tmp_path: Path) -> None:
    root = summary_root(tmp_path)
    (root / "SUMMARY.md").write_text(
        "- amber: does arithmetic\n- basil: does arithmetic\n- cedar: does arithmetic\n",
        encoding="utf-8",
    )
    problems = summary_problems(root, "SUMMARY.md", "src/*.py")
    assert problems == [
        "SUMMARY.md is not specific about 3: amber, basil, cedar. "
        "Say what each one does, as its own text says."
    ]


def test_a_finish_is_refused_while_work_is_left_then_checked_then_done(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, native=False, finish_check=True, finish_checks=1)
    work = Path(state.root)
    (work / "a.py").write_text("x = 1  # TODO: name it\n", encoding="utf-8")
    scripted = Scripted(
        script=(
            finished("Done early."),
            act("edit", path="a.py", old="  # TODO: name it", new=""),
            finished("Done."),
            finished("Really done."),
        )
    )
    done = run_loop.advance(runs, state, store, scripted)
    assert (done.status, done.step) == ("done", 4)
    refused = scripted.seen[1].user
    assert "FINISH REFUSED" in refused and "WORK LEFT 1 lines in 1 files" in refused
    assert "a.py:1:" in refused
    assert "MARKERS none found." in scripted.seen[3].user
    steps = [json.loads(line) for line in (runs / state.id / "steps.jsonl").open()]
    details = [s["detail"] for s in steps if s.get("tool") == "finish"]
    assert details == ["finish refused", "finish check"]


def test_refusals_without_a_change_stall_the_run(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, finish_check=True, finish_refusals=3)
    (Path(state.root) / "a.py").write_text("# TODO\n", encoding="utf-8")
    scripted = Scripted(
        script=(
            finished(),
            act("write", path="b.py", text="x = 2\n"),
            finished(),
            finished(),
            finished(),
        )
    )
    stalled = run_loop.advance(runs, state, store, scripted)
    # The write resets the count: refusals at steps 1, 3, 4, then the third in a row stalls.
    assert (stalled.status, stalled.step) == ("stalled", 5)
    assert stalled.note.startswith("FINISH REFUSED")


def test_the_summary_gates_the_finish(tmp_path: Path) -> None:
    runs, state, store = begin(
        tmp_path, finish_check=True, finish_checks=0, summary="SUMMARY.md", summary_of="*.txt"
    )
    scripted = Scripted(
        script=(
            finished(),
            act("write", path="SUMMARY.md", text="- notes: the kettle is green\n"),
            finished("Summarised."),
        )
    )
    done = run_loop.advance(runs, state, store, scripted)
    assert (done.status, done.step, done.note) == ("done", 3, "Summarised.")
    assert "SUMMARY SUMMARY.md is not written. Write it last." in scripted.seen[1].user


def test_markers_off_and_no_summary_keep_the_capped_check(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, finish_check=True, finish_checks=1, markers=())
    (Path(state.root) / "a.py").write_text("# TODO\n", encoding="utf-8")
    done = run_loop.advance(runs, state, store, Scripted(script=(finished(), finished("Ok."))))
    assert (done.status, done.step, done.note) == ("done", 2, "Ok.")


def test_gate_options_are_checked_and_kept(tmp_path: Path) -> None:
    with pytest.raises(InputRejected, match="needs --summary"):
        begin(tmp_path / "a", summary_of="src/*.py")
    with pytest.raises(InputRejected, match="Markers"):
        begin(tmp_path / "b", markers=("",))
    with pytest.raises(InputRejected, match="inside the root"):
        begin(tmp_path / "c", summary="S.md", summary_of="../*.py")
    with pytest.raises(InputRejected, match="refusals"):
        begin(tmp_path / "d", finish_refusals=0)
    runs, state, _ = begin(tmp_path / "e", markers=("XXX",), summary="S.md", summary_of="*.py")
    loaded = run_loop.load(runs, state.id)
    assert (loaded.markers, loaded.summary, loaded.summary_of) == (("XXX",), "S.md", "*.py")


def test_the_cli_takes_the_gate_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from angularai.cli import main

    work = tmp_path / "work"
    work.mkdir()
    (work / "a.py").write_text("# HACK\n", encoding="utf-8")
    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANGULARAI_INDEX", raising=False)
    scripted = Scripted(script=(finished(), finished(), finished()))
    monkeypatch.setattr("angularai.cli.build_substrate", lambda config, environ: scripted)
    cortex = ["--cortex", str(tmp_path / "cortex")]
    assert main([*cortex, "init"]) == 0
    args = [
        "--root",
        str(work),
        "--markers",
        "HACK",
        "--finish-refusals",
        "2",
        "--search-hits",
        "40",
        "--summary",
    ]
    assert main([*cortex, "run", *args, "S.md", "--summary-of", "*.py", "Fix it."]) == 0
    run_id = capsys.readouterr().err.split("run   ")[1].split()[0]
    state = json.loads((tmp_path / "home" / "runs" / run_id / "state.json").read_text())
    assert state["status"] == "stalled" and state["markers"] == ["HACK"]
    assert state["search_hits"] == 40
    assert main([*cortex, "run", "--root", str(work), "--summary-of", "*.py", "x"]) == 1


def test_the_check_quotes_a_batch_of_what_the_summary_misses(tmp_path: Path) -> None:
    from angularai.worklist import summary_check

    root = summary_root(tmp_path)
    module(root, "dune", "Returns x plus 4. " + "Long. " * 40)
    none = summary_check(root, "SUMMARY.md", "src/*.py", quotes=2)
    assert none.quotes == (
        "src/amber.py: Returns x plus 6.",
        "src/basil.py: Returns x multiplied by 5.",
    )
    (root / "SUMMARY.md").write_text(
        "- amber: adds 6\n- basil: [description]\n- cedar: x + 3\n", encoding="utf-8"
    )
    check = summary_check(root, "SUMMARY.md", "src/*.py", quotes=5, quote_chars=40)
    assert check.problems[0] == "SUMMARY.md does not list 1 of 4: dune"
    assert check.quotes == (
        "src/basil.py: Returns x multiplied by 5.",
        "src/dune.py: Returns x plus 4. Long. Long. Long. L...",
    )
    assert summary_check(root, "SUMMARY.md", "src/*.py").quotes == ()


def test_a_refusal_quotes_a_window_sized_batch(tmp_path: Path) -> None:
    assert [run_loop.quote_batch(b) for b in (1000, 2000, 4000)] == [6, 13, 15]
    runs, state, store = begin(
        tmp_path, finish_check=True, budget=1000, summary="S.md", summary_of="src/*.py"
    )
    for n in range(20):
        module(Path(state.root), f"m{n:02d}", f"Returns x plus {n}.")
    scripted = Scripted(script=(finished(), finished()))
    run_loop.advance(runs, state, store, scripted, steps=2)
    refusal = scripted.seen[1].user
    assert "QUOTES what 6 of them say of themselves." in refusal
    assert "src/m00.py: Returns x plus 0." in refusal and "src/m05.py" in refusal
    assert "src/m06.py" not in refusal
    assert scripted.seen[1].estimated_tokens <= 1000


def test_search_hits_are_configurable(tmp_path: Path) -> None:
    from angularai.models import Action
    from angularai.tools import execute

    (tmp_path / "a.txt").write_text("".join(f"hit {n}\n" for n in range(80)), encoding="utf-8")
    search = Action("search", query="hit", path=".")
    assert len(execute(tmp_path, search).splitlines()) == 30
    assert len(execute(tmp_path, search, search_hits=60).splitlines()) == 60
    assert len(execute(tmp_path, search, search_hits=999).splitlines()) == 80
    runs, state, store = begin(tmp_path / "r", search_hits=50)
    (Path(state.root) / "a.txt").write_text("hit\n" * 70, encoding="utf-8")
    scripted = Scripted(script=(act("search", query="hit", path="."), finished()))
    run_loop.advance(runs, state, store, scripted)
    exchange = scripted.seen[1].exchange
    assert exchange is not None and exchange.output.count("a.txt:") == 50
    with pytest.raises(InputRejected, match="Search hits"):
        begin(tmp_path / "s", search_hits=0)
