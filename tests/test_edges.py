import json
from dataclasses import replace
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from angularai.attention import score, tokens
from angularai.budget import estimate_tokens
from angularai.cli import _repl, main, repl_line
from angularai.configure import add_lock, remove_lock, set_budget, wire
from angularai.errors import (
    ConfigError,
    InputDoesNotFit,
    InputRejected,
    KeyMissing,
    SubstrateUnavailable,
)
from angularai.http import http_json
from angularai.lock import ValueLock
from angularai.models import Integration
from angularai.pilot import pass_turn
from angularai.present import (
    format_banner,
    format_locks,
    format_recall,
    format_status,
    format_traces,
)
from angularai.store import init_cortex, open_store
from angularai.substrate import build_substrate
from tests.scripted import Scripted, noted


def test_word_forms_and_empty_budget() -> None:
    assert "city" in tokens("cities")
    assert "live" in tokens("lives")
    assert "live" in tokens("lived")
    assert score("lived", "live") > 0
    assert score("", "hello") == 0
    assert score("the", "warehouse") == 0
    assert score("hello", "") == 0
    assert estimate_tokens("") == 0


def test_a_secret_reply_is_held(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    result = pass_turn(
        store,
        Scripted(Integration("the key is sk-live-abcdefghij123", (), ())),
        "Say it back.",
    )
    assert result.held is True
    assert "sk-live" not in (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    leaked = "leaked sk-test-secret-value"
    assert ValueLock().judge(leaked, (), ("sk-test-secret-value",)).allowed is False


def test_input_boundaries(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    with pytest.raises(InputRejected, match="Say something"):
        pass_turn(store, Scripted(), "   ")
    with pytest.raises(InputRejected, match="too long"):
        pass_turn(store, Scripted(), "x" * 20001)
    assert (tmp_path / "passes.jsonl").read_text(encoding="utf-8") == ""


def test_fixed_workspace_can_exceed_the_budget(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    store.save_config(replace(store.config(), workspace_budget=256, max_input_tokens=200))
    with pytest.raises(InputDoesNotFit, match="budget"):
        pass_turn(store, Scripted(), "word " * 160)
    assert (tmp_path / "passes.jsonl").read_text(encoding="utf-8") == ""


def test_continuity_stays_out_when_the_budget_is_full(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    pass_turn(store, Scripted(noted("The notebook is blue.", "place")), "I have a blue notebook.")
    long_note = "warehouse " + ("cedar " * 120)
    pass_turn(store, Scripted(Integration("Ok.", (), ())), long_note)
    store.save_config(replace(store.config(), workspace_budget=256, max_input_tokens=80))
    question = Scripted(Integration("The notebook is blue.", (), ()))
    pass_turn(store, question, "What color is the notebook?")
    assert "The notebook is blue." in question.seen[-1].user
    assert "cedar cedar" not in question.seen[-1].user


def test_an_unanswered_episode_can_return(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)

    class Down:
        def integrate(self, workspace: object) -> Integration:
            raise SubstrateUnavailable("The substrate is unreachable.")

    with pytest.raises(SubstrateUnavailable):
        pass_turn(store, Down(), "I have a blue notebook and the river is long.")
    follow = Scripted(Integration("Blue.", (), ()))
    pass_turn(store, follow, "What color is the notebook?")
    assert "I have a blue notebook" in follow.seen[-1].user
    assert "angularai  " not in follow.seen[-1].user.split("RECENT", 1)[-1].split("INPUT", 1)[0]


def test_present_reads_a_lived_cortex(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    pass_turn(store, Scripted(noted("The notebook is blue.")), "I have a blue notebook.")
    view = store.view()
    assert "The notebook is blue." in format_traces(view)
    assert "The notebook is blue." in format_recall(view, "notebook")
    assert "angularai  unwired" in format_banner(store.config(), view)
    status = format_status(
        root=store.root,
        config=store.config(),
        view=view,
        last=store.last_pass(),
        key_name=None,
        key_set=False,
    )
    assert "last ok" in status
    assert "dosing chart" in format_locks(("dosing chart",))


def test_configure_boundaries(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    with pytest.raises(InputRejected, match="spaces"):
        wire(store, "openai", "bad model", None)
    with pytest.raises(InputRejected, match="base-url"):
        wire(store, "compatible", "local", None)
    with pytest.raises(InputRejected, match="http"):
        wire(store, "openai", "demo", "ftp://example.test/v1")
    wired = wire(store, "openai", "demo", None)
    assert wired.json_mode is True
    shrunk = set_budget(store, 300)
    assert shrunk.max_input_tokens == 150
    add_lock(store, "dosing chart")
    assert add_lock(store, "Dosing Chart").forbid == ("dosing chart",)
    with pytest.raises(InputRejected, match="phrase"):
        add_lock(store, "   ")
    with pytest.raises(InputRejected, match="too long"):
        add_lock(store, "x" * 201)
    full = tuple(f"lock-{index}" for index in range(64))
    store.save_config(replace(store.config(), forbid=full))
    with pytest.raises(InputRejected, match="64"):
        add_lock(store, "one more")
    remove_lock(store, "lock-0")
    with pytest.raises(InputRejected, match="not set"):
        remove_lock(store, "missing")


def test_build_substrate_names_a_missing_key(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    with pytest.raises(InputRejected, match="no model"):
        build_substrate(store.config(), {})
    wire(store, "anthropic", "demo", None)
    with pytest.raises(KeyMissing, match="ANTHROPIC_API_KEY"):
        build_substrate(store.config(), {})
    built = build_substrate(store.config(), {"ANTHROPIC_API_KEY": "sk-ant-test-value"})
    assert built.__class__.__name__ == "Anthropic"


def test_config_rejects_a_bad_file(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    path = tmp_path / "cortex.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        open_store(tmp_path).config()
    path.write_text(json.dumps({"substrate": "nope"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        open_store(tmp_path).config()


def test_http_reads_a_body_and_names_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, status: object, body: object) -> None:
            self.status = status
            self._body = body

        def read(self) -> object:
            return self._body

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class Opener:
        def __init__(self, outcome: object) -> None:
            self._outcome = outcome

        def open(self, request: Request, timeout: float) -> Response:
            del request, timeout
            outcome = self._outcome
            if isinstance(outcome, Exception):
                raise outcome
            assert isinstance(outcome, Response)
            return outcome

    monkeypatch.setattr("angularai.http._OPENER", Opener(Response(200, b"{}")))
    assert http_json("http://example.test", {}, {"a": 1}) == (200, b"{}")

    monkeypatch.setattr("angularai.http._OPENER", Opener(Response(200, bytearray(b"{}"))))
    assert http_json("http://example.test", {}, {})[1] == b"{}"

    monkeypatch.setattr("angularai.http._OPENER", Opener(Response("200", b"{}")))
    with pytest.raises(SubstrateUnavailable, match="unreadable status"):
        http_json("http://example.test", {}, {})

    error = HTTPError("http://example.test", 401, "no", Message(), BytesIO(b"nope"))
    monkeypatch.setattr("angularai.http._OPENER", Opener(error))
    assert http_json("http://example.test", {}, {}) == (401, b"nope", {})

    limited = Message()
    limited["Retry-After"] = "7"
    error = HTTPError("http://example.test", 429, "slow", limited, BytesIO(b"{}"))
    monkeypatch.setattr("angularai.http._OPENER", Opener(error))
    assert http_json("http://example.test", {}, {}) == (429, b"{}", {"retry-after": "7"})

    monkeypatch.setattr("angularai.http._OPENER", Opener(TimeoutError("slow")))
    with pytest.raises(SubstrateUnavailable, match="timed out"):
        http_json("http://example.test", {}, {})

    monkeypatch.setattr("angularai.http._OPENER", Opener(URLError(TimeoutError("slow"))))
    with pytest.raises(SubstrateUnavailable, match="timed out"):
        http_json("http://example.test", {}, {})

    monkeypatch.setattr("angularai.http._OPENER", Opener(URLError("down")))
    with pytest.raises(SubstrateUnavailable, match="unreachable"):
        http_json("http://example.test", {}, {})


def test_repl_and_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--cortex", str(tmp_path), "status"]) == 1
    assert "angularai init" in capsys.readouterr().err
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "budget"]) == 0
    assert "2000" in capsys.readouterr().out
    assert main(["--cortex", str(tmp_path), "lock"]) == 0
    assert "No locks yet" in capsys.readouterr().out
    assert main(["--cortex", str(tmp_path), "lock", "add", "dosing", "chart"]) == 0
    assert main(["--cortex", str(tmp_path), "lock", "remove", "dosing", "chart"]) == 0

    store = open_store(tmp_path)
    assert repl_line(store, "/help")[1] is not None
    assert "last" not in (repl_line(store, "/status")[1] or "")
    assert "Unknown command" in (repl_line(store, "/nope")[1] or "")

    lines = iter(["hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    monkeypatch.setattr("angularai.cli._speak", lambda store, text: "Hello.")
    assert _repl(store) == 0
    assert "Hello." in capsys.readouterr().out

    def end(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end)
    assert _repl(store) == 0

    def refuse(store: object, text: str) -> str:
        raise InputRejected("Say something first.")

    monkeypatch.setattr("angularai.cli._speak", refuse)
    calls = {"n": 0}

    def once(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] > 1:
            raise EOFError
        return "hello"

    monkeypatch.setattr("builtins.input", once)
    assert _repl(store) == 0
    assert "Say something first." in capsys.readouterr().err
