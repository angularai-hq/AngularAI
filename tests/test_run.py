import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from angularai import run as run_loop
from angularai.budget import estimate_tokens
from angularai.errors import (
    InputRejected,
    InputTooLong,
    ServerError,
    SubstrateError,
    SubstrateUnreadable,
)
from angularai.index import INDEX, CortexIndex
from angularai.models import Action, Exchange, Integration, TraceDraft, Workspace
from angularai.parse import IDLE, REMEMBERED, TOOL_NAMES, TOOL_SPECS, integration_from_calls
from angularai.pilot import pass_turn
from angularai.run import RunState
from angularai.store import Store, init_cortex, open_store
from angularai.substrate import Anthropic, OpenAICompatible, inline_exchange
from tests.scripted import Scripted

TASK = "Read notes.txt and finish."


def act(tool: str, **fields: str) -> Integration:
    return Integration(f"{tool} next.", (), (), Action(tool, **fields))  # type: ignore[arg-type]


def finished(reply: str = "All done.") -> Integration:
    return Integration(reply, (), (), None)


def setup(tmp_path: Path) -> tuple[Path, Path, Store]:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (work / "notes.txt").write_text("The kettle is green.\n", encoding="utf-8")
    cortex = tmp_path / "cortex"
    init_cortex(cortex)
    return tmp_path / "runs", work, open_store(cortex)


def begin(tmp_path: Path, task: str = TASK, **options: object) -> tuple[Path, RunState, Store]:
    runs, work, store = setup(tmp_path)
    options.setdefault("finish_check", False)
    state = run_loop.start(runs, task=task, root=work, cortex=store.root, **options)  # type: ignore[arg-type]
    return runs, state, store


def call(name: str, arguments: object) -> dict[str, object]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": raw}}


# Native tool calls.


def test_tool_specs_name_the_agent_tools_finish_and_remember() -> None:
    names = {spec["function"]["name"] for spec in TOOL_SPECS}  # type: ignore[index]
    assert names == {*TOOL_NAMES, "finish", "remember"}


def test_the_first_tool_call_is_the_act_and_its_note_the_reply() -> None:
    parsed = integration_from_calls(
        "",
        [
            call("read", {"path": "a.py", "note": "Reading a.py next."}),
            call("list", {"path": "."}),
            call("remember", {"text": "a.py holds the parser."}),
        ],
    )
    assert parsed.act == Action("read", path="a.py")
    assert parsed.reply == "Reading a.py next."
    assert parsed.remember == (TraceDraft("a.py holds the parser.", ()),)


def test_arguments_may_be_an_object_and_content_wins_over_the_note() -> None:
    parsed = integration_from_calls(
        "Content.", [{"function": {"name": "list", "arguments": {"path": "src", "note": "n"}}}]
    )
    assert parsed.act == Action("list", path="src")
    assert parsed.reply == "Content."


def test_finish_ends_with_its_summary() -> None:
    parsed = integration_from_calls("", [call("finish", {"summary": "Fixed it."})])
    assert parsed.act is None and parsed.reply == "Fixed it."
    assert integration_from_calls("", [call("finish", "{}")]).reply == "Done."


def test_no_call_is_idle_and_only_remember_is_remembered() -> None:
    assert integration_from_calls("I think so.", ()).act == IDLE
    assert integration_from_calls("", [call("remember", {"text": "x"})]).act == REMEMBERED
    assert integration_from_calls("", [call("remember", {"text": ""})]).act == IDLE


def test_bad_arguments_and_unknown_fields_are_refused_not_repaired() -> None:
    bad = integration_from_calls("", [call("read", "{not json")]).act
    assert bad is not None and bad.problem and bad.tool == "read"
    extra = integration_from_calls("", [call("read", {"path": "a", "mode": "x"})]).act
    assert extra is not None and "mode" in extra.problem
    skipped = integration_from_calls("", ["junk", {"function": {"name": 3}}]).act
    assert skipped == IDLE


# The window.


def test_working_memory_clips_tool_output_to_half_the_budget(tmp_path: Path) -> None:
    runs, state, _ = begin(tmp_path, budget=1000)
    huge = "line of output\n" * 2000
    for native in (True, False):
        loaded = RunState(
            **{
                **state.__dict__,
                "native": native,
                "last_done": "read  big.txt",
                "last_tool": "read",
                "last_args": '{"path": "big.txt"}',
                "last_output": huge,
                "history": tuple(f"{n}  read  f{n}.txt  ->  ok" for n in range(10)),
            }
        )
        text, exchange = run_loop.working_memory(loaded)
        size = estimate_tokens(text) + (0 if exchange is None else exchange.tokens())
        assert size < 500
        assert (exchange is not None) is native
        assert ("BEGIN TOOL OUTPUT" in text) is not native


def test_every_step_fits_the_budget_and_the_window_does_not_grow(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, budget=1200, max_steps=30)
    (Path(state.root) / "big.txt").write_text("word " * 20000, encoding="utf-8")
    scripted = Scripted(Integration("Reading.", (), (), Action("read", path="big.txt")))
    done = run_loop.advance(runs, state, store, scripted)
    assert done.status == "budget" and done.step == 30
    assert all(item.estimated_tokens <= item.budget == 1200 for item in scripted.seen)
    assert max(len(item.user) for item in scripted.seen) <= 1200 * 4
    assert len(done.history) == 10


def test_a_repeated_step_is_flagged(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, native=False)
    scripted = Scripted(
        script=(act("list", path="./"), act("list", path="."), finished()),
    )
    run_loop.advance(runs, state, store, scripted)
    assert "repeats a recent step" in scripted.seen[2].user


# The loop, checkpoints, resume.


def test_a_run_reads_then_finishes_and_saves_each_step(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path)
    states: list[int] = []

    class Peek(Scripted):
        def integrate(self, workspace: Workspace) -> Integration:
            states.append(run_loop.load(runs, state.id).step)
            return super().integrate(workspace)

    scripted = Peek(script=(act("read", path="notes.txt"), finished("The kettle is green.")))
    lines: list[str] = []
    done = run_loop.advance(runs, state, store, scripted, progress=lines.append)
    assert (done.status, done.step, done.note) == ("done", 2, "The kettle is green.")
    assert states == [0, 1]
    second = scripted.seen[1]
    assert second.native_tools and second.exchange is not None
    assert "The kettle is green." in second.exchange.output
    assert second.exchange.arguments == '{"path": "notes.txt"}'
    steps = (runs / state.id / "steps.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in steps] == [1, 2]
    assert json.loads(steps[0])["tool"] == "read" and json.loads(steps[1])["status"] == "done"
    assert lines[0].startswith("1  read  notes.txt")
    assert run_loop.load(runs, state.id) == done


def test_resume_continues_where_the_pause_left(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, max_steps=5)
    first = run_loop.advance(runs, state, store, Scripted(act("read", path="notes.txt")), steps=2)
    assert (first.status, first.step) == ("running", 2)
    resumed = run_loop.load(runs, state.id)
    scripted = Scripted(script=(act("list", path="."), finished()))
    done = run_loop.advance(runs, resumed, store, scripted)
    assert (done.status, done.step) == ("done", 4)
    assert "STEP 3 of 5" in scripted.seen[0].user
    assert scripted.seen[0].exchange is not None
    with pytest.raises(InputRejected, match="nothing to resume"):
        run_loop.advance(runs, done, store, scripted)


def test_a_failed_step_keeps_the_run_resumable(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path)

    class Down:
        def integrate(self, workspace: Workspace) -> Integration:
            raise ServerError("The model server failed.", 503)

    with pytest.raises(ServerError):
        run_loop.advance(runs, state, store, Down())
    saved = run_loop.load(runs, state.id)
    assert (saved.step, saved.status) == (0, "running") and "failed" in saved.error
    done = run_loop.advance(runs, saved, store, Scripted(finished()))
    assert done.status == "done" and done.error == ""


def test_held_stalled_remembered_and_out_of_steps(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, max_steps=6)
    idle = Integration("Hmm.", (), (), IDLE)
    stalled = run_loop.advance(runs, state, store, Scripted(idle))
    assert (stalled.status, stalled.step, stalled.idle) == ("stalled", 3, 3)

    runs2, state2, store2 = begin(tmp_path / "b", max_steps=2)
    kept = Integration("Keep.", (TraceDraft("The kettle is green.", ()),), (), REMEMBERED)
    spent = run_loop.advance(runs2, state2, store2, Scripted(kept))
    assert spent.status == "budget" and spent.history[-1].startswith("2  remember  ->  Kept. (+1")
    assert spent.last_tool == "" and "The kettle is green." in store2.view().traces[0].text
    more = run_loop.raise_limit(spent, 3)
    assert (more.status, more.max_steps) == ("running", 3)
    with pytest.raises(InputRejected):
        run_loop.raise_limit(spent, 2)

    runs3, state3, store3 = begin(tmp_path / "c")
    store3.save_config(
        store3.config().__class__(**{**store3.config().__dict__, "forbid": ("kettle",)})
    )
    held = run_loop.advance(
        runs3, state3, store3, Scripted(act("write", path="kettle.txt", text="x"))
    )
    assert held.status == "held" and not (Path(held.root) / "kettle.txt").exists()


def test_one_process_holds_a_run(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path)
    with run_loop.holding(runs, state.id):
        with pytest.raises(InputRejected, match="already running"):
            run_loop.advance(runs, state, store, Scripted(finished()))


def test_start_and_load_refuse_what_they_cannot_use(tmp_path: Path) -> None:
    runs, work, store = setup(tmp_path)
    start: Callable[..., RunState] = run_loop.start
    for options, message in (
        ({"budget": 10}, "budget"),
        ({"max_steps": 0}, "Steps"),
        ({"root": tmp_path / "missing"}, "not a directory"),
        ({"task": "   "}, "Say what"),
        ({"task": "x" * 1300}, "longer"),
    ):
        arguments = {"task": TASK, "root": work, "cortex": store.root, **options}
        with pytest.raises(InputRejected, match=message):
            start(runs, **arguments)
    with pytest.raises(InputRejected, match="r_ and 12"):
        run_loop.load(runs, "../etc")
    with pytest.raises(InputRejected, match="No run"):
        run_loop.load(runs, "r_000000000000")
    state = start(runs, task=TASK, root=work, cortex=store.root)
    (runs / state.id / "state.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(InputRejected, match="unreadable"):
        run_loop.load(runs, state.id)
    (runs / "notarun").mkdir()
    assert run_loop.listing(runs) == []
    assert run_loop.listing(tmp_path / "none") == []
    with pytest.raises(InputRejected, match="at least 1"):
        run_loop.advance(runs, state, store, Scripted(finished()), steps=0)


def test_runs_live_under_the_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path))
    assert run_loop.runs_dir({"ANGULARAI_HOME": str(tmp_path)}) == tmp_path / "runs"


def test_the_v2_index_recalls_at_every_step(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, "Find what colour the kettle is.")
    remembered = Integration("Noted.", (TraceDraft("The kettle is green.", ()),), (), None)
    pass_turn(store, Scripted(remembered), "The kettle is green.")
    index = CortexIndex(store.root)
    scripted = Scripted(script=(act("list", path="."), finished()))
    run_loop.advance(runs, state, store, scripted, index=index)
    assert (store.root / INDEX).is_file()
    assert all("The kettle is green." in item.user.split("INPUT")[0] for item in scripted.seen)


# The pilot's budget and exchange.


def test_the_exchange_comes_out_of_the_budget(tmp_path: Path) -> None:
    _, _, store = setup(tmp_path)
    exchange = Exchange("read", '{"path": "a"}', "x" * 800)
    scripted = Scripted(finished())
    pass_turn(store, scripted, "Go on.", budget=1000, native_tools=True, exchange=exchange)
    seen = scripted.seen[0]
    assert seen.budget == 1000 and seen.exchange == exchange
    assert seen.estimated_tokens <= 1000 and seen.estimated_tokens >= exchange.tokens()
    with pytest.raises(InputTooLong):
        pass_turn(store, scripted, "Go on.", budget=1000, exchange=Exchange("r", "", "x" * 4000))


# The command.


def test_cli_runs_pauses_lists_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from angularai.cli import main

    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.txt").write_text("The kettle is green.\n", encoding="utf-8")
    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANGULARAI_INDEX", raising=False)
    scripted = Scripted(
        script=(act("read", path="notes.txt"), act("list", path="."), finished("Green."))
    )
    monkeypatch.setattr("angularai.cli.build_substrate", lambda config, environ: scripted)
    cortex = ["--cortex", str(tmp_path / "cortex")]
    assert main([*cortex, "init"]) == 0
    assert (
        main(
            [
                *cortex,
                "run",
                "--root",
                str(work),
                "--steps",
                "1",
                "--files",
                "off",
                "--finish-check",
                "off",
                TASK,
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    run_id = err.split("run   ")[1].split()[0]
    assert "Paused at step 1." in err
    assert main(["run", "--list"]) == 0
    assert f"{run_id}  running  1/50" in capsys.readouterr().out
    assert main(["run", "--resume", run_id, "--max-steps", "9"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Green.\n" and "step  1 of 9" in captured.err
    assert main(["run", "--resume", run_id]) == 1
    assert main(["run", "--resume", run_id, "extra"]) == 1
    assert main(["run"]) == 1
    assert scripted.seen[2].exchange is not None


# The substrate side of native tool calls.


def native_workspace(exchange: Exchange | None = None) -> Workspace:
    return Workspace(
        system="SYS",
        user="USER",
        input_text="USER",
        trace_ids=(),
        estimated_tokens=2,
        budget=80,
        traces_shown=0,
        traces_held=0,
        episodes_shown=0,
        episodes_held=0,
        native_tools=True,
        exchange=exchange,
    )


def compatible(
    answers: list[tuple[int, object]], sent: list[Mapping[str, object]]
) -> OpenAICompatible:
    def transport(
        url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> tuple[int, bytes]:
        sent.append(payload)
        status, body = answers.pop(0)
        return status, json.dumps(body).encode()

    return OpenAICompatible(
        model="demo",
        base_url="http://127.0.0.1:11434/v1",
        api_key="placeholder",
        json_mode=True,
        completion_field="max_tokens",
        transport=transport,
    )


def message(content: str = "", calls: list[object] | None = None) -> dict[str, object]:
    body: dict[str, object] = {"content": content}
    if calls is not None:
        body["tool_calls"] = calls
    return {"choices": [{"message": body}]}


def test_native_sends_tools_and_the_last_call_as_a_tool_turn() -> None:
    sent: list[Mapping[str, object]] = []
    model = compatible([(200, message("", [call("read", {"path": "b.py"})]))], sent)
    exchange = Exchange("read", '{"path": "a.py"}', "print(1)")
    parsed = model.integrate(native_workspace(exchange))
    assert parsed.act == Action("read", path="b.py")
    payload = sent[0]
    assert payload["tools"] == list(TOOL_SPECS) and "response_format" not in payload
    roles = [item["role"] for item in payload["messages"]]  # type: ignore[attr-defined]
    assert roles == ["system", "user", "assistant", "tool"]
    tool = payload["messages"][3]  # type: ignore[index]
    assert "print(1)" in tool["content"] and tool["content"].startswith("BEGIN TOOL OUTPUT")
    assert payload["messages"][2]["tool_calls"][0]["function"]["arguments"] == exchange.arguments  # type: ignore[index]


def test_native_reads_a_json_reply_and_plain_text_is_idle() -> None:
    sent: list[Mapping[str, object]] = []
    json_reply = '{"reply":"Done.","act":null,"remember":[],"supersede":[]}'
    model = compatible([(200, message(json_reply)), (200, message("Thinking aloud."))], sent)
    assert model.integrate(native_workspace()).act is None
    assert model.integrate(native_workspace()).act == IDLE
    with pytest.raises(SubstrateUnreadable):
        compatible([(200, message(""))], []).integrate(native_workspace())
    with pytest.raises(SubstrateUnreadable):
        compatible([(200, {"choices": []})], []).integrate(native_workspace())


def test_a_server_without_tools_gets_the_json_protocol_with_the_output_inline() -> None:
    sent: list[Mapping[str, object]] = []
    json_reply = '{"reply":"Done.","act":null,"remember":[],"supersede":[]}'
    model = compatible(
        [(400, {"error": {"message": "no tools"}}), (200, message(json_reply))], sent
    )
    parsed = model.integrate(native_workspace(Exchange("list", '{"path": "."}', "a.py")))
    assert parsed.reply == "Done."
    assert "tools" not in sent[1]
    user = sent[1]["messages"][1]["content"]  # type: ignore[index]
    assert "LAST list" in user and "a.py" in user and "BEGIN TOOL OUTPUT" in user
    with pytest.raises(SubstrateError):
        compatible([(401, {"error": {"message": "no"}})], []).integrate(native_workspace())


def test_anthropic_sees_the_exchange_inline() -> None:
    sent: list[Mapping[str, object]] = []

    def transport(
        url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> tuple[int, bytes]:
        sent.append(payload)
        text = '{"reply":"Done.","act":null,"remember":[],"supersede":[]}'
        return 200, json.dumps({"content": [{"type": "text", "text": text}]}).encode()

    model = Anthropic(
        model="demo", base_url="https://api.anthropic.com/v1/", api_key="k", transport=transport
    )
    model.integrate(native_workspace(Exchange("read", "{}", "a secret-free line")))
    assert "a secret-free line" in json.dumps(sent[0])
    assert inline_exchange(native_workspace()) == native_workspace()


def test_arguments_keep_only_the_fields_a_tool_takes() -> None:
    assert run_loop.arguments(Action("read", path="a", start=3)) == '{"path": "a", "start": 3}'
    written = run_loop.arguments(Action("write", path="a", text="x", overwrite=True))
    assert json.loads(written) == {"path": "a", "text": "x", "overwrite": True}


def test_a_state_with_a_wrong_status_or_id_is_unreadable(tmp_path: Path) -> None:
    runs, state, _ = begin(tmp_path)
    path = runs / state.id / "state.json"
    for change in ({"status": "lost"}, {"id": "r_111111111111"}, {"format": 9}):
        payload = {**json.loads(path.read_text()), **change}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(InputRejected, match="unreadable"):
            run_loop.load(runs, state.id)


def test_the_log_keeps_the_tool_and_never_its_output(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path)
    scripted = Scripted(script=(act("read", path="notes.txt"), finished("Green.")))
    run_loop.advance(runs, state, store, scripted)
    records = [json.loads(line)["user"] for line in (store.root / "passes.jsonl").open()]
    assert records[1] == f"RUN {state.id} STEP 2\nLAST read  notes.txt"
    assert "kettle" not in (store.root / "passes.jsonl").read_text()


def test_files_mark_reads_and_changes_that_took(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, native=False)
    scripted = Scripted(
        script=(
            act("read", path="./notes.txt"),
            act("read", path="missing.txt"),
            act("edit", path="notes.txt", old="green", new="blue"),
            act("edit", path="notes.txt", old="absent", new="x"),
            act("write", path="out/a.txt", text="a"),
            finished(),
        )
    )
    done = run_loop.advance(runs, state, store, scripted)
    assert done.files == (("notes.txt", 1, 1), ("out/a.txt", 0, 1))
    assert "FILES (read, changed)\nnotes.txt  read 1  changed 1\nout/a.txt" in scripted.seen[5].user
    assert run_loop.load(runs, state.id).files == done.files


def test_files_keep_the_newest_sixteen() -> None:
    files: tuple[tuple[str, int, int], ...] = ()
    for n in range(20):
        files = run_loop._touch(files, Action("write", path=f"f{n}"), f"Wrote f{n}.")
    assert len(files) == 16 and files[0][0] == "f4" and files[-1] == ("f19", 0, 1)
    assert run_loop._touch(files, Action("list", path="."), "x") == files
    assert run_loop._touch((), Action("read", path="e.txt"), "") == (("e.txt", 1, 0),)


def test_dwelling_on_one_file_is_named(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, native=False)
    reads = tuple(
        Integration("r", (), (), Action("read", path="notes.txt", start=n)) for n in range(2, 6)
    )
    scripted = Scripted(script=(*reads, finished()))
    run_loop.advance(runs, state, store, scripted)
    assert "steps were on notes.txt" not in scripted.seen[3].user
    assert "4 of the last 4 steps were on notes.txt" in scripted.seen[4].user


def test_a_repeat_is_the_same_call_not_the_same_file() -> None:
    assert run_loop.call_key(Action("list", path="./src/")) == run_loop.call_key(
        Action("list", path="src")
    )
    assert run_loop.call_key(Action("read", path="a", start=2)) != run_loop.call_key(
        Action("read", path="a")
    )


@pytest.mark.parametrize(
    ("view", "head"),
    [("list", "FILES (read, changed)\n"), ("partial", "Not the full set"), ("off", None)],
)
def test_the_files_view_is_a_choice(tmp_path: Path, view: str, head: str | None) -> None:
    runs, state, store = begin(tmp_path, native=False, files_view=view)
    scripted = Scripted(script=(act("read", path="notes.txt"), finished()))
    done = run_loop.advance(runs, state, store, scripted)
    assert done.files == (("notes.txt", 1, 0),)
    second = scripted.seen[1]
    if head is None:
        assert "notes.txt  read 1" not in second.user and "files read" not in second.system
    else:
        assert head in second.user and "notes.txt  read 1" in second.user
        assert "files read and changed" in second.system
    assert run_loop.load(runs, state.id).files_view == view


def test_an_unknown_files_view_is_unreadable(tmp_path: Path) -> None:
    runs, state, _ = begin(tmp_path)
    path = runs / state.id / "state.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "files_view": "x"}))
    with pytest.raises(InputRejected, match="unreadable"):
        run_loop.load(runs, state.id)


def test_the_first_finish_gets_a_check_and_the_second_ends(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, native=False, finish_check=True)
    (Path(state.root) / "a.py").write_text("x = 1  # TODO: name it\n", encoding="utf-8")
    scripted = Scripted(
        script=(
            act("edit", path="notes.txt", old="green", new="blue"),
            finished("Done early."),
            act("edit", path="a.py", old="  # TODO: name it", new=""),
            finished("Done."),
            finished("Really done."),
        )
    )
    done = run_loop.advance(runs, state, store, scripted)
    assert (done.status, done.step, done.note) == ("done", 5, "Really done.")
    check = scripted.seen[2].user
    assert "FINISH CHECK" in check and f"TASK {TASK}" in check
    assert "CHANGED notes.txt" in check and "a.py:1:" in check
    assert "MARKERS none found." in scripted.seen[4].user
    steps = [json.loads(line) for line in (runs / state.id / "steps.jsonl").open()]
    assert [s["detail"] for s in steps if s.get("tool") == "finish"] == ["finish check"] * 2


def test_without_the_finish_check_the_first_finish_ends(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, finish_check=False)
    done = run_loop.advance(runs, state, store, Scripted(finished()))
    assert run_loop.start(runs, task=TASK, root=Path(state.root), cortex=store.root).finish_check
    assert (done.status, done.step) == ("done", 1)


def test_a_change_is_kept_for_recall_and_never_its_output(tmp_path: Path) -> None:
    runs, state, store = begin(tmp_path, native=False)
    scripted = Scripted(
        script=(
            act("edit", path="notes.txt", old="green", new="a  bright\nblue"),
            act("write", path="b.txt", text="x" * 400),
            act("edit", path="notes.txt", old="absent", new="never"),
            finished(),
            finished(),
        )
    )
    run_loop.advance(runs, state, store, scripted)
    records = [json.loads(line)["user"] for line in (store.root / "passes.jsonl").open()]
    assert records[1].endswith("EDITED notes.txt: a bright blue")
    assert "WROTE b.txt: " + "x" * 297 + "..." in records[2]
    assert "never" not in records[3]
    steps = [json.loads(line) for line in (runs / state.id / "steps.jsonl").open()]
    assert steps[0]["change"] == "EDITED notes.txt: a bright blue" and "change" not in steps[2]

    runs2, state2, store2 = begin(tmp_path / "off", native=False, change_recall=False)
    scripted2 = Scripted(act("edit", path="notes.txt", old="green", new="blue"))
    run_loop.advance(runs2, state2, store2, scripted2, steps=2)
    kept = (store2.root / "passes.jsonl").read_text()
    assert "EDITED" not in kept


@pytest.mark.parametrize(("limit", "ends"), [(0, 2), (1, 4), (2, 6)])
def test_finish_checks_are_limited(tmp_path: Path, limit: int, ends: int) -> None:
    runs, state, store = begin(tmp_path, native=False, finish_check=True, finish_checks=limit)
    work = act("edit", path="notes.txt", old="green", new="green")
    script = (work, finished(), work, finished(), work, finished(), work, finished())
    done = run_loop.advance(runs, state, store, Scripted(script=script))
    assert (done.status, done.step, done.checks) == ("done", ends, limit)
    with pytest.raises(InputRejected, match="Finish checks"):
        run_loop.start(runs, task=TASK, root=Path(state.root), cortex=store.root, finish_checks=21)
