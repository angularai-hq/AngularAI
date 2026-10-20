import json
import os
from pathlib import Path

import pytest

from angularai.errors import SubstrateUnavailable
from angularai.evals import cli
from angularai.evals.client import PAID_KEYS, Chat, EvalError, require_local
from angularai.evals.cortex import answer_v1, at, episodes, v1_input
from angularai.evals.dataset import Question, Session, Turn, parse
from angularai.evals.reduce import reduce, write_subset
from angularai.evals.runner import answer, append, done, judge, markdown, report, run
from angularai.evals.scoring import (
    Scored,
    contains,
    judge_prompt,
    normalize,
    summarize,
    usage,
    verdict,
)
from angularai.models import Workspace
from tests.evalfakes import FakePost, question, raw_question
from tests.scripted import Scripted, said

LOCAL = "http://127.0.0.1:11434/v1"


def chat(post: FakePost, model: str = "small") -> Chat:
    ticks = iter(range(1000))
    return Chat(model=model, base_url=LOCAL, post=post, clock=lambda: float(next(ticks)))


def test_require_local_refuses_remote_hosts() -> None:
    assert require_local(LOCAL + "/") == LOCAL
    assert require_local("http://localhost:8080/v1") == "http://localhost:8080/v1"
    with pytest.raises(EvalError, match="local models only"):
        require_local("https://api.openai.com/v1")
    assert require_local("https://api.example.com/v1", allow_remote=True)
    with pytest.raises(EvalError):
        require_local("ftp://127.0.0.1/")
    with pytest.raises(EvalError):
        Chat(model="m", base_url="https://api.openai.com/v1")


def test_chat_sends_no_key_and_reads_usage() -> None:
    post = FakePost(["  blue  "])
    done_ = chat(post).complete("sys", "user", 32)
    assert done_.text == "blue"
    assert (done_.prompt_tokens, done_.completion_tokens, done_.seconds) == (11, 2, 1.0)
    body = post.bodies[0]
    assert body["temperature"] == 0 and body["max_tokens"] == 32
    assert post.urls == [LOCAL + "/chat/completions"]
    assert "Authorization" not in json.dumps(body)


def test_chat_refuses_errors_and_odd_replies() -> None:
    with pytest.raises(EvalError, match="500"):
        chat(FakePost(status=500)).complete("s", "u")

    def odd(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        return 200, b'{"choices": []}'

    with pytest.raises(EvalError, match="not a chat completion"):
        Chat(model="m", base_url=LOCAL, post=odd).complete("s", "u")

    def empty(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        return 200, b'{"choices": [{"message": {"content": null}}]}'

    with pytest.raises(EvalError, match="no text"):
        Chat(model="m", base_url=LOCAL, post=empty).complete("s", "u")

    def bare(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        return 200, b'{"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": -1}}'

    result = Chat(model="m", base_url=LOCAL, post=bare).complete("s", "u")
    assert result.prompt_tokens is None and result.completion_tokens is None


def test_default_post_reports_a_closed_port() -> None:
    with pytest.raises(EvalError, match="did not answer"):
        Chat(model="m", base_url="http://127.0.0.1:9/v1", timeout=2).complete("s", "u")


def test_episodes_pair_turns_in_order() -> None:
    q = Question(
        id="e",
        type="multi-session",
        question="q?",
        answer="a",
        date="2023/05/30 (Tue) 23:40",
        sessions=(
            Session(
                "s",
                "2023/05/20 (Sat) 02:21",
                (
                    Turn("assistant", "hello"),
                    Turn("user", "one"),
                    Turn("user", "two"),
                    Turn("assistant", "x" * 9000),
                    Turn("user", "three"),
                ),
            ),
        ),
        evidence=("s",),
    )
    records = episodes(q)
    assert [(r.user, r.reply and r.reply[:5]) for r in records] == [
        ("(no user turn)", "hello"),
        ("one", None),
        ("two", "xxxxx"),
        ("three", None),
    ]
    assert len(records[2].reply or "") == 8000
    assert [r.at for r in records] == sorted(r.at for r in records)
    assert at("2023/05/20 (Sat) 02:21", 3) == "2023-05-20T02:21:03"
    assert at("someday", 0) == "2000-01-01T00:00:00"
    assert v1_input(q) == "Current Date: 2023/05/30 (Tue) 23:40\nQuestion: q?"


def test_v1_arm_runs_a_real_pass_within_budget(tmp_path: Path) -> None:
    substrate = Scripted(said("Your notebook is blue."))
    result = answer_v1(
        tmp_path / "c", question(), substrate, budget=1000, model="small", base_url=LOCAL
    )
    assert result.reply == "Your notebook is blue."
    assert result.budget == 1000 and result.used <= 1000
    assert result.episodes == 3 and result.episodes_shown >= 1
    (workspace,) = substrate.seen
    assert isinstance(workspace, Workspace)
    assert "my notebook is blue" in workspace.user
    assert workspace.input_text.startswith("Current Date:")


def test_judge_prompts_are_the_official_text() -> None:
    q = question()
    standard = judge_prompt(q, "It is blue.")
    assert standard == (
        "I will give you a question, a correct answer, and a response from a model. Please "
        "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
        "response is equivalent to the correct answer or contains all the intermediate steps to "
        "get the correct answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no. \n\nQuestion: What color "
        "is my notebook?\n\nCorrect Answer: blue\n\nModel Response: It is blue.\n\nIs the model "
        "response correct? Answer yes or no only."
    )
    temporal = judge_prompt(question(kind="temporal-reasoning"), "r")
    assert "still correct. \n\nQuestion:" in temporal
    assert "required answer.\n\nQuestion:" in judge_prompt(question(kind="knowledge-update"), "r")
    assert "Rubric: blue" in judge_prompt(question(kind="single-session-preference"), "r")
    assert "unanswerable" in judge_prompt(question(qid="x_abs"), "r")
    with pytest.raises(ValueError):
        judge_prompt(question(kind="other"), "r")


def test_verdict_contains_and_normalize() -> None:
    assert verdict("Yes.") and verdict(" yes") and not verdict("No") and not verdict("")
    assert normalize("The  Blue, notebook!") == "blue notebook"
    assert contains(question(), "It's the BLUE one.")
    assert not contains(question(), "bluebell")
    assert contains(question(qid="x_abs"), "blue") is None
    assert contains(question(kind="single-session-preference"), "blue") is None
    assert contains(question(answer="!!"), "x") is None


def test_summarize_and_usage() -> None:
    rows = [
        Scored("a", "multi-session", False, True, True),
        Scored("b", "multi-session", False, False, None),
        Scored("c", "knowledge-update", True, True, None),
    ]
    summary = summarize(rows)
    assert summary["n"] == 3
    assert summary["judge"] == 66.7
    assert summary["contains"] == 100.0
    assert summary["abstention"] == 100.0
    assert summary["per_type"] == {
        "knowledge-update": {"n": 1, "judge": 100.0, "contains": None},
        "multi-session": {"n": 2, "judge": 50.0, "contains": 100.0},
    }
    assert summarize([])["judge"] is None
    stats = usage([{"prompt_tokens": 10, "seconds": 1.0}, {"prompt_tokens": 20, "error": "x"}])
    assert stats == {
        "prompt_tokens_mean": 15.0,
        "prompt_tokens_est_mean": None,
        "seconds_p50": 1.0,
        "errors": 1.0,
    }


def test_answer_arms(tmp_path: Path) -> None:
    q = question()
    full = answer("full", q, chat(FakePost(["blue"])), budget=2000)
    assert full["prediction"] == "blue" and full["budget"] is None
    assert full["prompt_tokens"] == 11 and isinstance(full["prompt_tokens_est"], int)
    trunc = answer("truncated", q, chat(FakePost(["x"])), budget=300)
    assert trunc["budget"] == 300
    oracle_post = FakePost(["blue"])
    answer("oracle", q, chat(oracle_post), budget=2000)
    content = json.dumps(oracle_post.bodies[0])
    assert "Turn 1." not in content
    v1 = answer("v1", q, chat(FakePost()), budget=1000, substrate=Scripted(said("blue")))
    assert v1["prediction"] == "blue" and v1["episodes"] == 3
    missing = answer("v1", q, chat(FakePost()), budget=1000)
    assert "substrate" in str(missing["error"])


def test_v2_arm_indexes_the_history_with_vectors(tmp_path: Path) -> None:
    from tests.test_semantic import Fake

    fake = Fake(model="nomic-test")
    record = answer(
        "v2",
        question(),
        chat(FakePost()),
        budget=1000,
        substrate=Scripted(said("blue")),
        embedder=fake,
        context=1,
    )
    assert record["error"] is None and record["prediction"] == "blue"
    assert record["chunks"] == 8 and record["embed_model"] == "nomic-test"
    assert record["context"] == 1 and record["budget"] == 1000
    assert fake.sent[0][0].startswith("search_document: ")
    lexical = answer("v2", question(), chat(FakePost()), budget=1000, substrate=Scripted(said("b")))
    assert lexical["embed_model"] is None and lexical["chunks"] == 8


def test_cli_v2_builds_a_local_embedder() -> None:
    embedder = cli.local_embedder("aa-nomic-embed", LOCAL)
    assert embedder.model == "aa-nomic-embed"


def test_answer_keeps_an_error_and_moves_on() -> None:
    class Down:
        def integrate(self, workspace: Workspace) -> object:
            raise SubstrateUnavailable("down")

    record = answer("v1", question(), chat(FakePost()), budget=1000, substrate=Down())  # type: ignore[arg-type]
    assert record["prediction"] == "" and "down" in str(record["error"])
    bad = answer("full", question(), chat(FakePost(status=503)), budget=2000)
    assert "503" in str(bad["error"])


def test_run_resumes_and_judge_reports(tmp_path: Path) -> None:
    qs = (question("a"), question("b", kind="temporal-reasoning"), question("c_abs"))
    out = tmp_path / "preds.jsonl"
    lines: list[str] = []
    assert run("full", qs[:2], out, chat(FakePost(["blue"])), budget=2000) == 2
    assert (
        run("full", qs, out, chat(FakePost(["I do not know."])), budget=2000, progress=lines.append)
        == 1
    )
    assert len(lines) == 1 and lines[0].startswith("full c_abs")
    with out.open("a") as handle:
        handle.write('{"torn"')
    assert sorted(done(out)) == ["a", "b", "c_abs"]
    append(out, {"id": "zzz", "error": "x", "prediction": ""})

    judged = tmp_path / "judged.jsonl"
    grader = FakePost(["yes", "no", "yes"])
    assert judge(qs, out, judged, chat(grader, "judge"), progress=lines.append) == 3
    assert judge(qs, out, judged, chat(grader, "judge")) == 0
    rows = done(judged)
    assert rows["a"]["judge"] is True and rows["a"]["contains"] is True
    assert rows["b"]["judge"] is False
    assert rows["c_abs"]["abstention"] is True

    chosen, manifest = reduce(
        parse([raw_question(qid=q) for q in ("a", "b", "c_abs")]), 3, 0, 10_000
    )
    result = report([judged, tmp_path / "missing.jsonl"], manifest, chosen)
    arms = result["arms"]
    assert isinstance(arms, list) and len(arms) == 1
    arm = arms[0]
    assert arm["arm"] == "full" and arm["n"] == 3 and arm["complete"] is True
    assert arm["judge"] == 66.7
    assert "questions" not in result["subset"]  # type: ignore[operator]
    table = markdown(result)
    assert "| full | small | - | 3 | 66.7 |" in table
    assert markdown({}).count("\n") == 2


def test_judge_marks_an_error_wrong_without_a_call(tmp_path: Path) -> None:
    preds = tmp_path / "p.jsonl"
    append(preds, {"id": "a", "type": "multi-session", "prediction": "", "error": "Timeout: x"})
    grader = FakePost(status=500)
    assert judge((question("a"),), preds, tmp_path / "j.jsonl", chat(grader)) == 1
    assert grader.bodies == []
    assert done(tmp_path / "j.jsonl")["a"]["judge"] is False


def test_scrub_removes_paid_keys_without_reading_them() -> None:
    environ = {name: "set" for name in PAID_KEYS} | {"PATH": "/bin"}
    cli.scrub(environ)
    assert environ == {"PATH": "/bin"}
    assert "OPENAI_API_KEY" in PAID_KEYS


def test_cli_end_to_end_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    data = tmp_path / "data"
    monkeypatch.setattr(cli, "fetch", lambda directory: directory / "lme.json")
    monkeypatch.setattr(
        cli, "load", lambda path: parse([raw_question(qid=f"q{i}") for i in range(3)])
    )
    subset = tmp_path / "subset.json"
    assert cli.main(["reduce", "--data", str(data), "--count", "2", "--out", str(subset)]) == 0
    assert "OPENAI_API_KEY" not in os.environ
    assert "2 of 3 eligible" in capsys.readouterr().out

    replies = FakePost(["blue"])

    def local_chat(**kwargs: object) -> Chat:
        return Chat(post=replies, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("angularai.evals.cli.Chat", local_chat)
    preds = tmp_path / "p.jsonl"
    argv = ["run", "--subset", str(subset), "--arm", "truncated", "--model", "m", "--limit", "1"]
    assert cli.main([*argv, "--out", str(preds)]) == 0
    assert len(done(preds)) == 1
    judged = tmp_path / "j.jsonl"
    argv = ["judge", "--subset", str(subset), "--predictions", str(preds), "--model", "j"]
    assert cli.main([*argv, "--out", str(judged)]) == 0
    out, md = tmp_path / "r.json", tmp_path / "r.md"
    argv = ["report", "--subset", str(subset), "--judged", str(judged), "--out", str(out)]
    assert cli.main([*argv, "--markdown", str(md)]) == 0
    assert json.loads(out.read_text())["arms"][0]["n"] == 1
    assert md.read_text().startswith("| arm |")


def test_cli_v1_arm_builds_a_keyless_local_substrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen, manifest = reduce(parse([raw_question()]), 1, 0, 10_000)
    subset = tmp_path / "subset.json"
    write_subset(subset, chosen, manifest)
    built: list[tuple[object, ...]] = []

    def fake_substrate(
        kind: str, model: str, base: str, key: str, policy: object, **kw: object
    ) -> Scripted:
        built.append((kind, model, base, key, kw))
        return Scripted(said("blue"))

    monkeypatch.setattr(cli, "substrate_for", fake_substrate)
    preds = tmp_path / "p.jsonl"
    argv = ["run", "--subset", str(subset), "--arm", "v1", "--model", "m", "--out", str(preds)]
    assert cli.main(argv) == 0
    assert built == [("compatible", "m", LOCAL, "", {"json_mode": True})]
    assert done(preds)["q1"]["prediction"] == "blue"
    v2 = tmp_path / "v2.jsonl"
    argv = ["run", "--subset", str(subset), "--arm", "v2", "--model", "m", "--out", str(v2)]
    monkeypatch.setattr("angularai.evals.cli.local_embedder", lambda model, base: None)
    assert cli.main([*argv, "--embed-model", "e", "--context", "1"]) == 0
    assert done(v2)["q1"]["context"] == 1


def test_cli_reports_errors_in_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["run", "--subset", str(tmp_path / "none.json"), "--arm", "full", "--model", "m"]
    assert cli.main([*argv, "--out", str(tmp_path / "o")]) == 1
    assert capsys.readouterr().err.startswith("angularai-eval: ")
    chosen, manifest = reduce(parse([raw_question()]), 1, 0, 10_000)
    write_subset(tmp_path / "s.json", chosen, manifest)
    argv = ["run", "--subset", str(tmp_path / "s.json"), "--arm", "full", "--model", "m"]
    argv += ["--base-url", "https://api.openai.com/v1", "--out", str(tmp_path / "o")]
    assert cli.main(argv) == 1
    assert "local models only" in capsys.readouterr().err


def test_cli_fetch_and_default_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_EVAL_DATA", str(tmp_path))
    assert cli.default_data() == tmp_path
    monkeypatch.setattr(cli, "fetch", lambda directory: directory / "f.json")
    assert cli.main(["fetch"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "f.json")
