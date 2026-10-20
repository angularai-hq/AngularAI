"""angularai setup and angularai model, with fake providers and a scripted terminal."""

import argparse
import io
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pytest

from angularai.cli import main
from angularai.errors import (
    AuthFailed,
    BadResponse,
    Forbidden,
    InputRejected,
    NotFound,
    QuotaExceeded,
    RateLimited,
    SubstrateError,
    Timeout,
    Unreachable,
)
from angularai.models import SubstrateKind
from angularai.onboard import model_command, offer_setup, setup_command
from angularai.profiles import KEYS_FILE, Vault, load_book
from angularai.provider import Policy
from angularai.session import cortex_status
from angularai.store import init_cortex, open_store
from angularai.substrate import probe
from tests.setupfakes import KEY, FakeKeyring, FakeTerminal


class Recorder:
    def __init__(self, error: SubstrateError | None = None) -> None:
        self.calls: list[tuple[SubstrateKind, str, str | None, str]] = []
        self.error = error

    def __call__(self, kind: SubstrateKind, model: str, base_url: str | None, key: str) -> None:
        self.calls.append((kind, model, base_url, key))
        if self.error is not None:
            raise self.error


def env_for(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"ANGULARAI_HOME": str(tmp_path / "home"), **extra}


def flags(**values: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "provider": None,
        "model_id": None,
        "base_url": None,
        "save_as": None,
        "key_stdin": False,
        "skip_test": False,
        "replace": False,
        "use": False,
        "model_command": None,
        "json": False,
    }
    base.update(values)
    return argparse.Namespace(**base)


def run_setup(
    environ: Mapping[str, str], term: FakeTerminal, tester: Recorder, **values: object
) -> int:
    return setup_command(flags(**values), environ, term.terminal, Vault(environ), tester)


def run_model(
    environ: Mapping[str, str], term: FakeTerminal, command: str | None, **values: object
) -> int:
    args = flags(model_command=command, **values)
    return model_command(args, environ, term.terminal, Vault(environ), Recorder())


def test_interactive_setup_saves_tests_and_never_shows_the_key(
    tmp_path: Path, fake_keyring_and_home: FakeKeyring, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    environ = env_for(tmp_path)
    term = FakeTerminal(answers=["1", "gpt-x", ""], secrets=[KEY])
    tester = Recorder()
    assert run_setup(environ, term, tester) == 0
    assert tester.calls == [("openai", "gpt-x", None, KEY)]
    book = load_book(environ)
    assert book.active == "openai"
    saved = book.get("openai")
    assert saved is not None and saved.storage == "keyring" and saved.hint == "sk-…a1b2"
    assert "Key sk-…a1b2 saved in the OS keyring." in term.lines
    assert "Active model: openai  openai  gpt-x" in term.lines
    assert any("Key (hidden)" in prompt for prompt in term.prompts)
    assert KEY not in term.text
    assert KEY not in caplog.text
    assert KEY not in (tmp_path / "home" / "models.json").read_text()


def test_setup_says_when_the_key_went_to_the_file(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(answers=["anthropic", "claude-x", "work"], secrets=[KEY])
    vault = Vault(environ, lambda: None)
    assert setup_command(flags(), environ, term.terminal, vault, Recorder()) == 0
    assert any("keys.json (mode 0600; no usable OS keyring)" in line for line in term.lines)
    assert load_book(environ).get("work") is not None
    assert KEY not in term.text


def test_interactive_ollama_needs_no_key(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(answers=["3", "", "llama3.2", ""], secrets=[""])
    tester = Recorder()
    assert run_setup(environ, term, tester) == 0
    assert tester.calls == [("compatible", "llama3.2", "http://127.0.0.1:11434/v1", "")]
    saved = load_book(environ).get("ollama")
    assert saved is not None and saved.storage == "none"
    assert "No key saved." in term.lines


def test_interactive_retries_bad_answers(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(
        answers=["9", "4", "ftp://x", "http://127.0.0.1:8000/v1", "bad model", "m", "Bad"],
        secrets=["sk-bad key-zzzz9999qqqq", KEY],
    )
    term.answers.append("ok-name")
    assert run_setup(environ, term, Recorder()) == 0
    assert "Choose 1 to 4." in term.errors
    assert any("http://" in error for error in term.errors)
    assert any("characters a key does not have" in error for error in term.errors)
    assert load_book(environ).get("ok-name") is not None
    assert KEY not in term.text and "zzzz9999qqqq" not in term.text


def test_interactive_gives_up_after_three_bad_answers(tmp_path: Path) -> None:
    term = FakeTerminal(answers=["x", "y", "z"])
    with pytest.raises(InputRejected, match="No provider"):
        run_setup(env_for(tmp_path), term, Recorder())
    term = FakeTerminal(answers=["1", "m"], secrets=["", "", ""])
    with pytest.raises(InputRejected, match="No key entered"):
        run_setup(env_for(tmp_path), term, Recorder())


def test_an_environment_key_can_stand_in_for_a_saved_one(tmp_path: Path) -> None:
    other = "sk-environment-000000000000ffff"
    environ = env_for(tmp_path, OPENAI_API_KEY=other)
    term = FakeTerminal(answers=["1", "gpt-x", ""], secrets=[""])
    tester = Recorder()
    assert run_setup(environ, term, tester) == 0
    assert tester.calls[0][3] == other
    assert "No key saved. OPENAI_API_KEY is read from the environment." in term.lines
    term = FakeTerminal(answers=["1", "gpt-x", "two"], secrets=[KEY])
    assert run_setup(environ, term, Recorder()) == 0
    assert "OPENAI_API_KEY is set in this shell and wins over the saved key." in term.lines
    assert other not in term.text and KEY not in term.text


ERRORS: list[tuple[SubstrateError, str]] = [
    (AuthFailed("The substrate (OpenAI) rejected the API key."), "Check the key"),
    (Forbidden("The substrate (OpenAI) does not allow this key to do that."), "cannot use"),
    (QuotaExceeded("refused the pass for billing: insufficient_quota.", "x"), "insufficient"),
    (NotFound("The substrate (OpenAI) returned 404. Check the model and the base URL."), "404"),
    (Timeout("The substrate timed out."), "in time"),
    (Unreachable("The substrate is unreachable."), "Is the server running"),
    (RateLimited("The substrate (OpenAI) is rate limiting. Wait and try again."), "rate"),
    (BadResponse("The substrate returned something that is not JSON."), "not JSON"),
]


@pytest.mark.parametrize(("error", "words"), ERRORS)
def test_a_failed_test_saves_nothing_and_says_why(
    tmp_path: Path, error: SubstrateError, words: str
) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(interactive=False, stdin=KEY + "\n")
    code = run_setup(
        environ, term, Recorder(error), provider="openai", model_id="gpt-x", key_stdin=True
    )
    assert code == 1
    assert any(words in line for line in term.errors)
    assert "Nothing was saved." in term.errors
    assert load_book(environ).profiles == ()
    assert KEY not in term.text


def test_interactive_can_save_after_a_failed_test(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(answers=["1", "gpt-x", "y", ""], secrets=[KEY])
    assert run_setup(environ, term, Recorder(Unreachable("x"))) == 0
    assert load_book(environ).get("openai") is not None
    term = FakeTerminal(answers=["1", "gpt-x", ""], secrets=[KEY])
    assert run_setup(env_for(tmp_path / "b"), term, Recorder(AuthFailed("no"))) == 1


def test_non_interactive_setup_with_flags_and_stdin(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(interactive=False, stdin=KEY + "\n")
    tester = Recorder()
    code = run_setup(
        environ, term, tester, provider="anthropic", model_id="claude-x", key_stdin=True
    )
    assert code == 0
    assert tester.calls == [("anthropic", "claude-x", None, KEY)]
    assert load_book(environ).active == "anthropic"
    assert term.prompts == []
    assert KEY not in term.text
    # A second one of the same kind gets the next free name, not a silent replace.
    term = FakeTerminal(interactive=False, stdin=KEY + "\n")
    assert (
        run_setup(
            environ,
            term,
            Recorder(),
            provider="anthropic",
            model_id="c2",
            key_stdin=True,
            skip_test=True,
        )
        == 0
    )
    assert load_book(environ).get("anthropic-2") is not None


def test_non_interactive_setup_refuses_without_flags_or_key(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(interactive=False)
    with pytest.raises(InputRejected, match="needs a terminal"):
        run_setup(environ, term, Recorder())
    with pytest.raises(InputRejected, match="needs a key"):
        run_setup(environ, term, Recorder(), provider="openai", model_id="gpt-x")
    with pytest.raises(InputRejected, match="No key arrived"):
        run_setup(environ, term, Recorder(), provider="openai", model_id="gpt-x", key_stdin=True)
    with pytest.raises(InputRejected, match="--base-url"):
        run_setup(environ, term, Recorder(), provider="compatible", model_id="m")
    with pytest.raises(InputRejected, match="exists"):
        run_setup(environ, term, Recorder(), provider="ollama", model_id="m", save_as="x")
        run_setup(environ, term, Recorder(), provider="ollama", model_id="m", save_as="x")
    assert (
        run_setup(
            environ, term, Recorder(), provider="ollama", model_id="m2", save_as="x", replace=True
        )
        == 0
    )
    saved = load_book(environ).get("x")
    assert saved is not None and saved.model == "m2"


def test_interactive_replace_asks(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    assert (
        run_setup(environ, FakeTerminal(answers=["3", "", "a", ""], secrets=[""]), Recorder()) == 0
    )
    term = FakeTerminal(answers=["3", "", "b", "", "n"], secrets=[""])
    with pytest.raises(InputRejected, match="exists"):
        run_setup(environ, term, Recorder())
    term = FakeTerminal(answers=["3", "", "b", "", "y"], secrets=[""])
    assert run_setup(environ, term, Recorder()) == 0
    saved = load_book(environ).get("ollama")
    assert saved is not None and saved.model == "b"


def test_key_stdin_on_a_terminal_is_hidden(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal(secrets=[KEY])
    term.answers.append("")
    code = run_setup(environ, term, Recorder(), provider="openai", model_id="m", key_stdin=True)
    assert code == 0
    assert KEY not in term.text


def test_model_list_use_remove(tmp_path: Path, fake_keyring_and_home: FakeKeyring) -> None:
    environ = env_for(tmp_path)
    term = FakeTerminal()
    assert run_model(environ, term, None) == 0
    assert term.lines == ["No saved models. Run: angularai setup"]
    run_setup(
        environ,
        FakeTerminal(interactive=False, stdin=KEY),
        Recorder(),
        provider="openai",
        model_id="gpt-x",
        key_stdin=True,
    )
    run_model(
        environ, FakeTerminal(interactive=False), "add", provider="ollama", model_id="llama3.2"
    )
    term = FakeTerminal()
    assert run_model(environ, term, "list") == 0
    table = "\n".join(term.lines)
    assert "* openai" in table
    assert "sk-…a1b2 in the OS keyring" in table
    assert "http://127.0.0.1:11434/v1" in table
    assert KEY not in table

    term = FakeTerminal()
    assert run_model(environ, term, "list", json=True) == 0
    data = json.loads(term.lines[0])
    assert data["active"] == "openai"
    assert data["models"][0]["key"] == {"storage": "keyring", "hint": "sk-…a1b2"}
    assert data["models"][1]["key"] == {"storage": "none", "hint": None}
    assert KEY not in term.lines[0]

    term = FakeTerminal()
    assert run_model(environ, term, "use", model_name="ollama") == 0
    assert load_book(environ).active == "ollama"
    term = FakeTerminal()
    picked = {**environ, "ANGULARAI_MODEL": "openai", "OPENAI_API_KEY": "sk-x-00000000000000"}
    assert run_model(picked, term, "use", model_name="openai") == 0
    assert any("wins over the saved key" in line for line in term.lines)
    term = FakeTerminal()
    assert (
        run_model({**environ, "ANGULARAI_MODEL": "ollama"}, term, "use", model_name="openai") == 0
    )
    assert any("picks another" in line for line in term.lines)
    term = FakeTerminal()
    assert run_model({**environ, "ANGULARAI_MODEL": "ollama"}, term, "list") == 0
    assert "ANGULARAI_MODEL picks ollama in this shell." in term.lines

    term = FakeTerminal()
    assert run_model(environ, term, "remove", model_name="openai") == 0
    assert term.lines[0] == "Removed openai and its key in the OS keyring."
    assert "No model is active." in term.lines[1]
    assert fake_keyring_and_home.store == {}
    term = FakeTerminal()
    assert run_model(environ, term, "remove", model_name="ollama") == 0
    assert term.lines == ["Removed ollama."]
    assert run_model(environ, FakeTerminal(), "nope") == 2


def test_model_add_keeps_the_active_one_unless_asked(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    run_setup(environ, FakeTerminal(interactive=False), Recorder(), provider="ollama", model_id="a")
    term = FakeTerminal(interactive=False)
    assert run_model(environ, term, "add", provider="ollama", model_id="b", save_as="b") == 0
    assert load_book(environ).active == "ollama"
    assert "Saved b. Switch with: angularai model use b" in term.lines
    term = FakeTerminal(interactive=False)
    assert (
        run_model(environ, term, "add", provider="ollama", model_id="c", save_as="c", use=True) == 0
    )
    assert load_book(environ).active == "c"


def test_status_names_the_saved_model_and_masks_the_key(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    run_setup(
        environ,
        FakeTerminal(interactive=False, stdin=KEY),
        Recorder(),
        provider="openai",
        model_id="gpt-x",
        key_stdin=True,
    )
    init_cortex(tmp_path / "cortex")
    text = cortex_status(open_store(tmp_path / "cortex"), environ)
    assert text.startswith("openai  gpt-x\nsaved model openai\nkey  sk-…a1b2 in the OS keyring")
    assert KEY not in text
    broken = cortex_status(open_store(tmp_path / "cortex"), {**environ, "ANGULARAI_MODEL": "x"})
    assert broken.startswith("unwired")


# --- first use --------------------------------------------------------------


def test_first_use_offers_setup_and_creates_the_cortex(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    root = tmp_path / "home" / "default"
    term = FakeTerminal(answers=["", "3", "", "llama3.2", ""], secrets=[""])
    offer_setup(root, environ, term.terminal, Vault(environ), Recorder())
    assert term.lines[0] == "No model is set up yet."
    assert (root / "cortex.json").is_file()
    assert load_book(environ).active == "ollama"
    # Configured now: no second offer.
    quiet = FakeTerminal()
    offer_setup(root, environ, quiet.terminal, Vault(environ), Recorder())
    assert quiet.lines == []


def test_first_use_stays_quiet_when_it_should(tmp_path: Path) -> None:
    environ = env_for(tmp_path)
    root = tmp_path / "cortex"
    for env, term in (
        (environ, FakeTerminal(interactive=False)),
        ({**environ, "CI": "true"}, FakeTerminal()),
        ({**environ, "OPENAI_API_KEY": "sk-x-0000000000"}, FakeTerminal()),
    ):
        offer_setup(root, env, term.terminal, Vault(env), Recorder())
        assert term.lines == [] and term.prompts == []
    init_cortex(root)
    assert (
        main(
            [
                "--cortex",
                str(root),
                "wire",
                "compatible",
                "--model",
                "m",
                "--base-url",
                "http://127.0.0.1:1/v1",
            ]
        )
        == 0
    )
    term = FakeTerminal()
    offer_setup(root, environ, term.terminal, Vault(environ), Recorder())
    assert term.prompts == []
    declined = FakeTerminal(answers=["n"])
    offer_setup(tmp_path / "other", environ, declined.terminal, Vault(environ), Recorder())
    assert not (tmp_path / "other").exists()
    failed = FakeTerminal(answers=["", "1", "m", ""], secrets=[KEY])
    offer_setup(
        tmp_path / "other",
        environ,
        failed.terminal,
        Vault(environ),
        Recorder(AuthFailed("rejected")),
    )
    assert not (tmp_path / "other").exists()
    assert KEY not in failed.text


# --- through main -----------------------------------------------------------


def test_main_setup_off_a_terminal_and_model_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path / "home"))
    assert main(["setup"]) == 1
    assert "needs a terminal" in capsys.readouterr().err
    code = main(["setup", "--provider", "ollama", "--model", "m", "--skip-test"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Active model: ollama  compatible  m" in out
    assert main(["model"]) == 0
    assert "* ollama" in capsys.readouterr().out
    assert main(["model", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["active"] == "ollama"


def test_main_has_no_key_argument(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["setup", "--provider", "openai", "--model", "m", "--key", KEY])
    assert caught.value.code == 2
    assert KEY not in capsys.readouterr().out


def test_a_saved_model_runs_a_pass_and_its_key_never_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end on a fake transport: the saved key is sent, and never stored or shown."""

    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path / "home"))
    seen: list[dict[str, str]] = []
    replies = iter(
        [
            {"reply": "The notebook is blue.", "remember": [], "supersede": []},
            {"reply": f"Your key is {KEY}", "remember": [], "supersede": []},
        ]
    )

    def fake_transport(policy: Policy) -> object:
        def send(url: str, headers: dict[str, str], payload: object, timeout: float) -> object:
            seen.append(headers)
            body = {"choices": [{"message": {"content": json.dumps(next(replies))}}]}
            return 200, json.dumps(body).encode()

        return send

    monkeypatch.setattr("angularai.substrate.default_transport", fake_transport)
    monkeypatch.setattr("sys.stdin", io.StringIO(KEY + "\n"))
    assert (
        main(["setup", "--provider", "openai", "--model", "gpt-x", "--key-stdin", "--skip-test"])
        == 0
    )
    assert main(["init"]) == 0
    assert main(["say", "What color is the notebook?"]) == 0
    assert main(["say", "What is my key?"]) == 0
    assert main(["status"]) == 0
    captured = capsys.readouterr()
    assert "The notebook is blue." in captured.out
    assert seen[0]["Authorization"] == f"Bearer {KEY}"
    assert KEY not in captured.out + captured.err
    home = tmp_path / "home"
    for path in home.rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(encoding="utf-8", errors="replace"), path.name
    assert not (home / KEYS_FILE).exists()


# --- the test call ----------------------------------------------------------


def test_probe_sends_one_tiny_call_and_maps_errors_without_the_key() -> None:
    sent: list[tuple[str, dict[str, str], object, float]] = []

    def ok(url: str, headers: dict[str, str], payload: object, timeout: float) -> object:
        sent.append((url, headers, payload, timeout))
        return 200, b'{"choices": []}'

    probe("openai", "gpt-x", None, KEY, transport=ok)  # type: ignore[arg-type]
    url, headers, payload, timeout = sent[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert payload == {
        "model": "gpt-x",
        "max_completion_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    assert timeout <= 60
    probe("anthropic", "claude-x", None, KEY, transport=ok)  # type: ignore[arg-type]
    assert sent[1][0] == "https://api.anthropic.com/v1/messages"
    assert sent[1][2] == {
        "model": "claude-x",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    probe("compatible", "m", "http://127.0.0.1:9/v1", "", transport=ok)  # type: ignore[arg-type]
    assert "Authorization" not in sent[2][1]
    assert sent[2][2] == {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }

    cases: list[tuple[int, bytes, type[SubstrateError]]] = [
        (401, b'{"error": {"message": "Incorrect API key: ' + KEY.encode() + b'"}}', AuthFailed),
        (429, b'{"error": {"code": "insufficient_quota", "message": "quota"}}', QuotaExceeded),
        (404, b'{"error": {"message": "model gpt-x not found"}}', NotFound),
        (403, b'{"error": {"message": "nope ' + KEY.encode() + b'"}}', Forbidden),
    ]
    for status, body, kind in cases:

        def failing(
            url: str,
            headers: dict[str, str],
            payload: object,
            timeout: float,
            status: int = status,
            body: bytes = body,
        ) -> object:
            return status, body, {}

        with pytest.raises(kind) as caught:
            probe("openai", "gpt-x", None, KEY, transport=failing)  # type: ignore[arg-type]
        assert KEY not in str(caught.value)
        assert KEY not in repr(caught.value)

    def down(url: str, headers: dict[str, str], payload: object, timeout: float) -> object:
        raise Unreachable("The substrate is unreachable.")

    with pytest.raises(Unreachable):
        probe("compatible", "m", "http://127.0.0.1:9/v1", "", transport=down)  # type: ignore[arg-type]


def test_main_offers_setup_on_a_terminal_before_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from angularai import cli, onboard

    monkeypatch.setattr(cli, "offer_setup", onboard.offer_setup)
    monkeypatch.setenv("ANGULARAI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CI", raising=False)
    term = FakeTerminal(answers=["", "3", "", "llama3.2", ""], secrets=[""])
    monkeypatch.setattr(onboard, "system_terminal", lambda: term.terminal)
    monkeypatch.setattr(onboard, "probe", lambda *args, **kwargs: None)
    monkeypatch.setattr("angularai.cli.run_tui", lambda store, speak: 0)
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main([]) == 0
    assert term.lines[0] == "No model is set up yet."
    assert (tmp_path / "home" / "default" / "cortex.json").is_file()


def test_end_of_input_ends_setup_with_a_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    from angularai import onboard

    def eof(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr("getpass.getpass", eof)
    term = onboard.system_terminal()
    with pytest.raises(InputRejected, match="Setup ended"):
        term.ask("x")
    with pytest.raises(InputRejected, match="Setup ended"):
        term.secret("x")


def test_key_stdin_from_a_terminal_is_hidden_even_off_a_terminal(tmp_path: Path) -> None:
    class TtyInput(io.StringIO):
        def isatty(self) -> bool:
            return True

    environ = env_for(tmp_path)
    term = FakeTerminal(interactive=False, secrets=[KEY])
    term.terminal.stdin = TtyInput("")
    code = run_setup(environ, term, Recorder(), provider="openai", model_id="m", key_stdin=True)
    assert code == 0
    assert term.prompts == ["Key (hidden): "]
