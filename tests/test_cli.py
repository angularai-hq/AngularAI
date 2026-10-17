from pathlib import Path

import pytest

from angularai.cli import main, repl_line
from angularai.store import init_cortex, open_store

KEY = "sk-test-secret-value-xyz"


def test_init_wire_and_status_keep_the_key_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "wire", "openai", "--model", "demo-model"]) == 0
    config = (tmp_path / "cortex.json").read_text(encoding="utf-8")
    assert KEY not in config
    assert "demo-model" in config
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert KEY not in out
    assert "OPENAI_API_KEY  set" in out
    assert "demo-model" in out


def test_agent_without_a_key_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "wire", "openai", "--model", "demo-model"]) == 0
    capsys.readouterr()
    code = main(["--cortex", str(tmp_path), "agent", "--root", str(tmp_path), "look"])
    assert code == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "sk-" not in err


def test_say_without_a_key_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "wire", "openai", "--model", "demo-model"]) == 0
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "say", "hello"]) == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "sk-" not in err


def test_empty_cortex_and_recall(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "cortex"]) == 0
    assert "No traces yet" in capsys.readouterr().out
    assert main(["--cortex", str(tmp_path), "recall", "notebook"]) == 0
    assert "Nothing in the cortex" in capsys.readouterr().out


def test_budget_and_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "budget", "10"]) == 1
    assert "between" in capsys.readouterr().err
    assert main(["--cortex", str(tmp_path), "budget", "1500"]) == 0
    assert main(["--cortex", str(tmp_path), "lock", "add", "dosing", "chart"]) == 0
    config = (tmp_path / "cortex.json").read_text(encoding="utf-8")
    assert "dosing chart" in config
    assert '"workspace_budget": 1500' in config


def test_compatible_url_cannot_carry_a_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    code = main(
        [
            "--cortex",
            str(tmp_path),
            "wire",
            "compatible",
            "--model",
            "local",
            "--base-url",
            "http://user:pass@127.0.0.1:11434/v1",
        ]
    )
    assert code == 1
    assert "environment" in capsys.readouterr().err
    assert "pass" not in (tmp_path / "cortex.json").read_text(encoding="utf-8")


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_repl_commands(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    assert repl_line(store, "/quit")[0] == "quit"
    assert repl_line(store, "   ")[1] is None
    text = repl_line(store, "/cortex")[1]
    assert text is not None
    assert "No traces yet" in text
    assert repl_line(store, "/recall")[1] == "/recall needs a query."
