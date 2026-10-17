from pathlib import Path

from angularai.agent import run_agent
from angularai.configure import add_lock
from angularai.models import Action, Integration
from angularai.parse import parse_integration
from angularai.store import init_cortex, open_store
from tests.scripted import Scripted


def test_parse_keeps_one_act() -> None:
    parsed = parse_integration(
        '{"reply":"Reading.","act":{"tool":"read","path":"README.md"},"remember":[],"supersede":[]}'
    )
    assert parsed.act is not None
    assert parsed.act.tool == "read"
    assert parsed.act.path == "README.md"
    assert parse_integration('{"reply":"Done.","act":null}').act is None


def test_the_agent_reads_one_file_and_leaves_the_other(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    (root / "seen.txt").write_text("The notebook is blue.\n", encoding="utf-8")
    (root / "hidden.txt").write_text("The vault code is cedar-plank-9.\n", encoding="utf-8")
    init_cortex(tmp_path / "cortex")
    store = open_store(tmp_path / "cortex")
    scripted = Scripted(
        script=(
            Integration("Reading.", (), (), Action("read", path="seen.txt")),
            Integration("The notebook is blue.", (), (), None),
        )
    )
    result = run_agent(store, scripted, "What color is the notebook?", root)
    sent = "\n".join(item.user for item in scripted.seen)
    assert result.stop == "done"
    assert result.steps == 2
    assert result.reply == "The notebook is blue."
    assert "The notebook is blue." in sent
    assert "cedar-plank-9" not in sent
    assert all(item.estimated_tokens <= item.budget for item in scripted.seen)


def test_a_held_reply_does_not_touch_the_directory(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    init_cortex(tmp_path / "cortex")
    store = open_store(tmp_path / "cortex")
    add_lock(store, "dosing chart")
    scripted = Scripted(
        Integration(
            "See the dosing chart.",
            (),
            (),
            Action("write", path="note.txt", text="nope"),
        )
    )
    result = run_agent(store, scripted, "Write a note.", root)
    assert result.stop == "held"
    assert not (root / "note.txt").exists()


def test_the_step_budget_stops_the_loop(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    init_cortex(tmp_path / "cortex")
    store = open_store(tmp_path / "cortex")
    act = Action("list", path=".")
    scripted = Scripted(script=tuple(Integration("Looking.", (), (), act) for _ in range(4)))
    result = run_agent(store, scripted, "Look around.", root, steps=2)
    assert result.stop == "budget"
    assert result.steps == 2
    assert len(scripted.seen) == 2
