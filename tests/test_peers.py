"""Cortexes that talk: names, ask, converse, provenance, locks, and caps."""

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from angularai.assemble import assemble
from angularai.cli import main
from angularai.errors import InputRejected, NoCortex, ServerError
from angularai.lock import HELD
from angularai.models import Config, CortexView, Episode, PassRecord, Peer
from angularai.names import (
    check_name,
    cortex_home,
    label,
    list_names,
    named_root,
    valid_name,
)
from angularai.peers import Cortex, Line, Message, ask, converse, peer_system, stop_notice
from angularai.store import LOG, Store, _pass_json, init_cortex, open_store
from tests.peerfakes import Voice, reply

# A key shape the secret scanner knows. It is made up, and not in the environment.
FAKE_KEY = "sk-proj-Zq81mW0xLr5pVt72nKd9"
FAKE_ENV_SECRET = "mellow-Harbor-4471"


def home_env(home: Path) -> dict[str, str]:
    return {"ANGULARAI_HOME": str(home)}


def make(home: Path, name: str, *, budget: int | None = None, locks: tuple[str, ...] = ()) -> Store:
    root = home / name
    init_cortex(root)
    store = open_store(root)
    config = store.config()
    store.save_config(
        Config(
            substrate="openai",
            model="demo",
            base_url=None,
            workspace_budget=budget or config.workspace_budget,
            max_input_tokens=min(config.max_input_tokens, budget or config.workspace_budget),
            forbid=locks,
            json_mode=False,
        )
    )
    return store


def remember(store: Store, text: str) -> None:
    Cortex("seed", store, Voice("seed", [reply("Noted.", text)])).open(text, "nobody")


def log_text(store: Store) -> str:
    return (store.root / LOG).read_text(encoding="utf-8")


def peers(store: Store) -> list[Peer | None]:
    return [record.peer for record in store.passes()]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir(mode=0o700)
    return path


# Names


@pytest.mark.parametrize("name", ["a", "default", "ana-2", "b_c", "0", "x" * 32])
def test_a_valid_name(name: str) -> None:
    assert valid_name(name)
    assert check_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../x",
        "../../etc",
        "a/b",
        "a\\b",
        "/abs",
        "~",
        "~root",
        "Ana",
        "-a",
        "_a",
        "a b",
        "a\n",
        "a\x00",
        "ｂ",
        "é",
        "x" * 33,
        "con",
        "nul",
        "lpt1",
        ".hidden",
    ],
)
def test_a_name_that_could_leave_or_collide_is_refused(name: str, home: Path) -> None:
    assert not valid_name(name)
    with pytest.raises(InputRejected, match="lowercase"):
        named_root(name, home_env(home))


def test_a_name_is_a_child_of_the_home(home: Path) -> None:
    assert named_root("ana", home_env(home)) == home / "ana"
    assert cortex_home({}) == Path.home() / ".angularai"


def test_a_relative_home_is_refused() -> None:
    with pytest.raises(Exception, match="ANGULARAI_HOME must be an absolute path"):
        cortex_home({"ANGULARAI_HOME": "relative/home"})


def test_a_link_in_place_of_a_cortex_is_refused(home: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    init_cortex(elsewhere)
    (home / "ana").symlink_to(elsewhere)
    with pytest.raises(InputRejected, match="is a link"):
        named_root("ana", home_env(home))
    assert list_names(home_env(home)) == ()


def test_list_names_reads_only_names(home: Path) -> None:
    make(home, "ana")
    make(home, "bo")
    (home / "empty").mkdir()
    (home / "Upper").mkdir()
    init_cortex(home / "Upper" / "x")
    assert list_names(home_env(home)) == ("ana", "bo")
    assert list_names({"ANGULARAI_HOME": str(home / "missing")}) == ()


def test_label_falls_back_when_the_directory_is_not_a_name(tmp_path: Path) -> None:
    assert label(tmp_path / "ana") == "ana"
    assert label(tmp_path / "My Cortex") == "cortex"


# Ask


def test_ask_answers_from_the_target_memory_only(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    remember(a, "Ana keeps the roadmap in a red folder.")
    remember(b, "The launch is on Friday.")
    voice = Voice("bo", [reply("The launch is on Friday.")])
    answer = ask(Cortex("ana", a), Cortex("bo", b, voice), "When is the launch?")
    assert answer == type(answer)("The launch is on Friday.", False, voice.seen[0].estimated_tokens)
    shown = voice.sent()
    assert "The launch is on Friday." in shown
    assert "red folder" not in shown
    assert "cortex named bo" in shown and "from the cortex named ana" in shown


def test_ask_writes_one_pass_on_each_side_with_provenance(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    ask(Cortex("ana", a), Cortex("bo", b, Voice("bo", [reply("Friday.")])), "When?")
    (mine,) = a.passes()
    (theirs,) = b.passes()
    assert (mine.user, mine.reply, mine.outcome, mine.peer) == (
        "When?",
        "Friday.",
        "ok",
        Peer("bo", "asked"),
    )
    assert mine.used == 0
    assert (theirs.user, theirs.reply, theirs.peer) == ("When?", "Friday.", Peer("ana", "answered"))


def test_an_answered_pass_cannot_supersede_and_names_its_source(home: Path) -> None:
    b = make(home, "bo")
    remember(b, "The launch is on Friday.")
    (trace,) = b.view().current_traces()
    voice = Voice("bo", [reply("Moved.", "The launch is on Monday.", supersede=(trace.id,))])
    ask(Cortex("ana", make(home, "ana")), Cortex("bo", b, voice), "The launch moved to Monday.")
    assert trace.id in voice.seen[0].trace_ids
    traces = {item.text: item for item in b.view().traces}
    assert traces["The launch is on Friday."].state == "current"
    assert traces["The launch is on Monday."].source.startswith("cortex ana: ")


def test_an_opened_pass_may_supersede(home: Path) -> None:
    a = make(home, "ana")
    remember(a, "The launch is on Friday.")
    (trace,) = a.view().current_traces()
    voice = Voice("ana", [reply("Moved.", supersede=(trace.id,))])
    Cortex("ana", a, voice).open("The launch moved. Tell bo.", "bo")
    assert a.view().traces[0].state == "superseded"
    assert a.passes()[-1].peer == Peer("bo", "opened")


def test_a_secret_in_the_answer_never_reaches_the_asker(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    voice = Voice("bo", [reply(f"The key is {FAKE_KEY}.")])
    answer = ask(Cortex("ana", a), Cortex("bo", b, voice), "What is the key?")
    assert answer.reply == HELD and answer.held
    assert FAKE_KEY not in log_text(a)
    assert FAKE_KEY not in log_text(b)
    assert a.passes()[0].outcome == "held"


def test_an_environment_secret_in_the_answer_is_held(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    secrets = (FAKE_ENV_SECRET,)
    voice = Voice("bo", [reply(f"It is {FAKE_ENV_SECRET}")])
    answer = ask(Cortex("ana", a, secrets=secrets), Cortex("bo", b, voice, secrets), "Which?")
    assert answer.held
    assert FAKE_ENV_SECRET not in log_text(a) + log_text(b)


def test_the_asker_lock_holds_an_answer_the_target_released(home: Path) -> None:
    a = make(home, "ana", locks=("dosing chart",))
    b = make(home, "bo")
    voice = Voice("bo", [reply("See the Dosing  Chart.")])
    answer = ask(Cortex("ana", a), Cortex("bo", b, voice), "Where is it?")
    assert answer.held
    assert "Dosing" not in log_text(a)
    assert b.passes()[0].outcome == "ok"


def test_the_target_lock_holds_its_own_answer(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo", locks=("red folder",))
    voice = Voice("bo", [reply("In the red folder.")])
    assert ask(Cortex("ana", a), Cortex("bo", b, voice), "Where?").held
    assert "red folder" not in log_text(a) + log_text(b)


def test_a_question_held_by_either_lock_writes_nothing(home: Path) -> None:
    a = make(home, "ana", locks=("payroll",))
    b = make(home, "bo", locks=("salary",))
    voice = Voice("bo")
    with pytest.raises(InputRejected, match="Value lock held that question"):
        ask(Cortex("ana", a), Cortex("bo", b, voice), "Read me the payroll.")
    with pytest.raises(InputRejected, match="Cortex bo holds that question"):
        ask(Cortex("ana", a), Cortex("bo", b, voice), "What is the salary?")
    with pytest.raises(InputRejected, match="looks like a secret"):
        ask(Cortex("ana", a), Cortex("bo", b, voice), f"Store {FAKE_KEY}")
    assert voice.seen == []
    assert a.passes() == () and b.passes() == ()


def test_a_failed_answer_is_an_error_pass_on_both_sides(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")

    class Down:
        def integrate(self, workspace: object) -> object:
            raise ServerError(f"The substrate is unavailable. {FAKE_KEY}", 503)

    with pytest.raises(ServerError):
        ask(Cortex("ana", a), Cortex("bo", b, Down()), "Hello?")  # type: ignore[arg-type]
    (mine,) = a.passes()
    assert mine.outcome == "error" and mine.reply is None
    assert mine.error is not None and "unavailable" in mine.error
    assert FAKE_KEY not in log_text(a) + log_text(b)
    assert b.passes()[0].outcome == "error"


def test_a_cortex_cannot_ask_itself(home: Path) -> None:
    a = make(home, "ana")
    with pytest.raises(InputRejected, match="different names"):
        ask(Cortex("ana", a), Cortex("ana", a, Voice("ana")), "Hi")
    with pytest.raises(InputRejected, match="itself"):
        Cortex("ana", a, Voice("ana")).answer(Message("ana", "Hi"))
    with pytest.raises(InputRejected, match="itself"):
        Cortex("ana", a, Voice("ana")).open("Hi", "ana")


def test_a_cortex_without_a_model_does_not_answer(home: Path) -> None:
    with pytest.raises(InputRejected, match="has no model"):
        Cortex("bo", make(home, "bo")).answer(Message("ana", "Hi"))


def test_a_bad_sender_name_never_reaches_the_log(home: Path) -> None:
    b = make(home, "bo")
    with pytest.raises(InputRejected, match="lowercase"):
        Cortex("bo", b, Voice("bo")).answer(Message("../ana", "Hi"))
    with pytest.raises(InputRejected, match="lowercase"):
        Cortex("Bo", b)
    assert b.passes() == ()


def test_the_next_workspace_names_the_peer(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    ask(Cortex("ana", a), Cortex("bo", b, Voice("bo", [reply("Friday.")])), "When is it?")
    view = b.view()
    workspace = assemble(view, "When is it again?", b.config())
    assert "cortex ana  When is it?" in workspace.user
    assert "angularai  Friday." in workspace.user
    mine = assemble(a.view(), "When is it again?", a.config())
    assert "you  When is it?" in mine.user
    assert "cortex bo  Friday." in mine.user


def test_an_opened_episode_reads_as_your_own() -> None:
    view = CortexView(
        traces=(),
        episodes=(Episode("t", "Plan it.", "Bo, plan it.", "ok", Peer("bo", "opened")),),
    )
    config = Config(None, None, None, 2000, 800, (), False)
    assert "you  Plan it.\nangularai  Bo, plan it." in assemble(view, "plan", config).user


# Converse


def test_converse_takes_turns_up_to_the_cap(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    shown: list[Line] = []
    result = converse(
        Cortex("ana", a, Voice("ana")),
        Cortex("bo", b, Voice("bo")),
        "Plan it.",
        5,
        50_000,
        shown.append,
    )
    assert result.stop == "turns"
    assert [line.speaker for line in result.lines] == ["ana", "bo", "ana", "bo", "ana"]
    assert list(result.lines) == shown
    assert result.used == sum(line.used for line in result.lines)
    assert [p.role if p else None for p in peers(a)] == ["opened", "answered", "answered"]
    assert [p.name if p else None for p in peers(a)] == ["bo", "bo", "bo"]
    assert peers(b) == [Peer("ana", "answered")] * 2
    assert a.passes()[0].user == "Plan it."
    assert b.passes()[0].user == result.lines[0].text
    assert "stopped" not in stop_notice(result, 5, 50_000)
    assert "turn cap of 5" in stop_notice(result, 5, 50_000)


def test_converse_never_passes_the_token_cap(home: Path) -> None:
    a = make(home, "ana", budget=1000)
    b = make(home, "bo", budget=1000)
    result = converse(Cortex("ana", a, Voice("ana")), Cortex("bo", b, Voice("bo")), "Go.", 12, 2500)
    assert result.stop == "tokens"
    assert result.used <= 2500
    assert result.used + 1000 > 2500
    assert len(result.lines) < 12
    assert "token cap of 2500" in stop_notice(result, 12, 2500)


def test_converse_refuses_a_cap_under_one_workspace(home: Path) -> None:
    a = make(home, "ana", budget=2000)
    voice = Voice("ana")
    with pytest.raises(InputRejected, match="under one workspace"):
        converse(
            Cortex("ana", a, voice), Cortex("bo", make(home, "bo"), Voice("bo")), "Go.", 4, 1000
        )
    assert voice.seen == [] and a.passes() == ()


@pytest.mark.parametrize(("turns", "tokens"), [(0, 5000), (13, 5000), (4, 255), (4, 100_001)])
def test_converse_refuses_caps_out_of_range(home: Path, turns: int, tokens: int) -> None:
    with pytest.raises(InputRejected):
        converse(
            Cortex("ana", make(home, "ana"), Voice("ana")),
            Cortex("bo", make(home, "bo"), Voice("bo")),
            "Go.",
            turns,
            tokens,
        )


def test_a_secret_in_a_turn_stops_the_conversation(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    bo = Voice("bo", [reply(f"Use {FAKE_KEY}")])
    result = converse(Cortex("ana", a, Voice("ana")), Cortex("bo", b, bo), "Go.", 6, 50_000)
    assert result.stop == "held"
    assert [line.text for line in result.lines][-1] == HELD
    assert FAKE_KEY not in log_text(a) + log_text(b)
    assert len(a.passes()) == 1


def test_the_receiver_lock_stops_a_turn_at_the_border(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo", locks=("red folder",))
    bo = Voice("bo")
    ana = Voice("ana", [reply("It is in the red folder.")])
    result = converse(Cortex("ana", a, ana), Cortex("bo", b, bo), "Go.", 6, 50_000)
    assert result.stop == "held"
    assert bo.seen == [] and b.passes() == ()
    assert "Value lock held a message" in stop_notice(result, 6, 50_000)


def test_a_turn_that_does_not_fit_stops(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    long = "word " * 1200
    result = converse(
        Cortex("ana", a, Voice("ana", [reply(long)])),
        Cortex("bo", b, Voice("bo")),
        "Go.",
        6,
        50_000,
    )
    assert result.stop == "fit"
    assert b.passes() == ()
    assert "did not fit" in stop_notice(result, 6, 50_000)


def test_a_turn_the_receiver_refuses_stops(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    huge = "x" * 25_000
    result = converse(
        Cortex("ana", a, Voice("ana", [reply(huge)])),
        Cortex("bo", b, Voice("bo")),
        "Go.",
        6,
        50_000,
    )
    assert result.stop == "refused"
    assert "refused" in stop_notice(result, 6, 50_000)


def test_a_held_opening_stops_before_crossing(home: Path) -> None:
    a = make(home, "ana", locks=("payroll",))
    bo = Voice("bo")
    result = converse(
        Cortex("ana", a, Voice("ana", [reply("The payroll is out.")])),
        Cortex("bo", make(home, "bo"), bo),
        "Go.",
        4,
        50_000,
    )
    assert result.stop == "held" and len(result.lines) == 1
    assert bo.seen == []


def test_each_side_sees_only_its_own_traces(home: Path) -> None:
    a = make(home, "ana")
    b = make(home, "bo")
    remember(a, "Ana private: the vault code is in drawer nine.")
    remember(b, "Bo private: the spare badge is under the plant.")
    ana, bo = Voice("ana"), Voice("bo")
    converse(Cortex("ana", a, ana), Cortex("bo", b, bo), "drawer plant badge vault", 6, 50_000)
    assert "spare badge is under the plant" not in ana.sent()
    assert "vault code is in drawer nine" not in bo.sent()


def test_peer_system_states_who_speaks() -> None:
    assert "from your operator" in peer_system("ana", "bo", True)
    assert "not your operator" in peer_system("ana", "bo", False)


# The log


def test_a_pass_without_a_peer_is_written_as_before() -> None:
    record = PassRecord("t", "ok", "hi", "yo", (), (), None, 1, 1, 0, 0, 0, 0)
    assert "peer" not in _pass_json(record)


def test_an_old_log_line_reads_with_no_peer(tmp_path: Path) -> None:
    init_cortex(tmp_path / "c")
    line: dict[str, object] = {
        "at": "t",
        "outcome": "ok",
        "user": "hi",
        "reply": "yo",
        "remember": [],
        "supersede": [],
        "error": None,
        "budget": 1,
        "used": 1,
        "traces_shown": 0,
        "traces_held": 0,
        "episodes_shown": 0,
        "episodes_held": 0,
    }
    bad: list[object] = [
        {"name": "../x", "role": "asked"},
        {"name": "bo", "role": "told"},
        {"name": "bo"},
        {"name": "bo", "role": "asked", "path": "/etc"},
        "bo",
        {"name": 3, "role": "asked"},
    ]
    lines = [json.dumps(line)] + [json.dumps({**line, "peer": peer}) for peer in bad]
    lines.append(json.dumps({**line, "peer": {"name": "bo", "role": "answered"}}))
    (tmp_path / "c" / LOG).write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = open_store(tmp_path / "c")
    assert [record.peer for record in store.passes()] == [None, Peer("bo", "answered")]
    assert [item.line for item in store.damage()] == [2, 3, 4, 5, 6, 7]


# The command line


def test_the_default_cortex_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert main(["init"]) == 0
    assert (tmp_path / ".angularai" / "default" / "cortex.json").is_file()
    assert (tmp_path / ".angularai").stat().st_mode & 0o777 == 0o700
    capsys.readouterr()
    assert main(["--name", "default", "status"]) == 0
    assert str(tmp_path / ".angularai" / "default") in capsys.readouterr().out
    assert main(["cortexes"]) == 0
    assert capsys.readouterr().out.strip() == "default"


def test_a_name_selects_a_cortex_in_the_home(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home / "deeper"))
    assert main(["init", "--name", "ana"]) == 0
    assert (home / "deeper" / "ana" / "cortex.json").is_file()
    assert (home / "deeper").stat().st_mode & 0o777 == 0o700
    assert main(["init"]) == 0
    assert (home / "deeper" / "default" / "cortex.json").is_file()
    capsys.readouterr()
    assert main(["cortexes"]) == 0
    assert capsys.readouterr().out.split() == ["ana", "default"]


def test_no_named_cortexes_is_said(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    assert main(["cortexes"]) == 0
    assert "No named cortexes" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["../x", "../../escape", "a/b", "/tmp/x", "..", "X"])
def test_a_traversing_name_creates_nothing(
    name: str,
    tmp_path: Path,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    assert main(["--name", name, "init"]) == 1
    assert main(["--name", "ana", "ask", "--to", name, "hello"]) == 1
    assert main(["--name", "ana", "converse", "--with", name, "hello"]) == 1
    assert "lowercase" in capsys.readouterr().err
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


def test_cortex_and_name_together_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--cortex", str(tmp_path), "--name", "ana", "status"]) == 1
    assert "not both" in capsys.readouterr().err


def test_ask_names_a_missing_cortex(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    make(home, "ana")
    assert main(["--name", "ana", "ask", "--to", "bo", "hello"]) == 1
    assert "No cortex named bo. Run angularai --name bo init." in capsys.readouterr().err


def test_a_cortex_cannot_ask_itself_by_another_path(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    make(home, "ana")
    assert main(["--cortex", str(home / "ana" / "."), "ask", "--to", "ana", "hi"]) == 1
    assert "itself" in capsys.readouterr().err


def test_ask_refuses_a_secret_before_any_key_is_read(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    make(home, "ana")
    make(home, "bo")
    assert main(["--name", "ana", "ask", "--to", "bo", f"keep {FAKE_KEY}"]) == 1
    err = capsys.readouterr().err
    assert "looks like a secret" in err
    assert FAKE_KEY not in err
    assert "OPENAI_API_KEY" not in err


def test_ask_without_the_target_key_names_the_variable(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    make(home, "ana")
    make(home, "bo")
    assert main(["--name", "ana", "ask", "--to", "bo", "hello"]) == 1
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def _wire_fakes(monkeypatch: pytest.MonkeyPatch, voices: dict[int, Voice]) -> None:
    """Each cortex gets the voice keyed by its budget, so the fake knows who it is."""

    def build(config: Config, environ: object) -> Voice:
        return voices[config.workspace_budget]

    monkeypatch.setattr("angularai.cli.build_substrate", build)


def test_ask_on_the_command_line_prints_only_the_reply(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    make(home, "ana", budget=1500)
    b = make(home, "bo", budget=1600)
    _wire_fakes(monkeypatch, {1600: Voice("bo", [reply("Friday.")])})
    assert main(["--name", "ana", "ask", "--to", "bo", "When", "is", "it?"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Friday.\n"
    assert captured.err == ""
    assert b.passes()[0].peer == Peer("ana", "answered")


def test_the_default_cortex_can_ask_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".angularai"
    init_cortex(home / "default")
    make(home, "bo", budget=1600)
    _wire_fakes(monkeypatch, {1600: Voice("bo", [reply("Hello default.")])})
    assert main(["ask", "--to", "bo", "hi"]) == 0
    assert capsys.readouterr().out == "Hello default.\n"
    assert open_store(home / "bo").passes()[0].peer == Peer("default", "answered")


def test_converse_on_the_command_line(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    make(home, "ana", budget=1500)
    make(home, "bo", budget=1600)
    _wire_fakes(monkeypatch, {1500: Voice("ana"), 1600: Voice("bo")})
    code = main(["--name", "ana", "converse", "--with", "bo", "--turns", "3", "Plan", "it."])
    assert code == 0
    captured = capsys.readouterr()
    speakers = [line.split("  ")[0] for line in captured.out.splitlines()]
    assert speakers == ["ana", "bo", "ana"]
    assert "turn cap of 3" in captured.err


def test_help_lists_the_new_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "ask" in out and "converse" in out and "cortexes" in out and "--name" in out


def test_open_named_reraises_with_the_name(home: Path) -> None:
    from angularai.cli import _open_named

    with pytest.raises(NoCortex, match="No cortex named zed"):
        _open_named("zed", home / "zed")


# Two cortexes at once


def _talker(home: str, me: str, other: str, count: int) -> None:
    mine = Store(Path(home) / me)
    theirs = Store(Path(home) / other)
    for index in range(count):
        question = f"{me} asks {index} " + "q" * 2000
        voice = Voice(other, [reply(f"{other} answers {me} {index} " + "r" * 4000)])
        ask(Cortex(me, mine), Cortex(other, theirs, voice), question)


@pytest.mark.skipif(sys.platform == "win32", reason="the lock is a no-op on Windows")
def test_two_cortexes_asking_each_other_at_once_keep_whole_logs(home: Path) -> None:
    for name in ("ana", "bo"):
        make(home, name, budget=4000)
    context = multiprocessing.get_context("spawn")
    count = 15
    workers = [
        context.Process(target=_talker, args=(str(home), me, other, count))
        for me, other in (("ana", "bo"), ("bo", "ana"), ("ana", "bo"), ("bo", "ana"))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
        assert worker.exitcode == 0
    for name, other in (("ana", "bo"), ("bo", "ana")):
        store = open_store(home / name)
        records = store.passes()
        assert store.damage() == ()
        assert len(records) == 4 * count
        asked = [r for r in records if r.peer == Peer(other, "asked")]
        answered = [r for r in records if r.peer == Peer(other, "answered")]
        assert len(asked) == len(answered) == 2 * count
        assert all(r.reply is not None and r.reply.startswith(f"{other} answers") for r in asked)
        mode = os.stat(home / name / LOG).st_mode & 0o777
        assert mode == 0o600
