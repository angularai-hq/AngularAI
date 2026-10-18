from pathlib import Path

import pytest

from angularai.cli import main, repl_line
from angularai.errors import InputRejected
from angularai.store import init_cortex, open_store

LEAK = "sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv"


def _wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "wire", "openai", "--model", "demo-model"]) == 0
    capsys.readouterr()


def test_say_refuses_a_secret_before_it_looks_for_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wired(tmp_path, monkeypatch, capsys)
    assert main(["--cortex", str(tmp_path), "say", f"my key is {LEAK}"]) == 1
    err = capsys.readouterr().err
    assert "looks like a secret" in err
    assert "OPENAI_API_KEY" not in err
    assert LEAK not in err
    assert (tmp_path / "passes.jsonl").read_text(encoding="utf-8") == ""


def test_say_refuses_a_secret_with_no_model_wired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "say", LEAK]) == 1
    assert "looks like a secret" in capsys.readouterr().err


def test_agent_refuses_a_secret_before_it_looks_for_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wired(tmp_path, monkeypatch, capsys)
    code = main(["--cortex", str(tmp_path), "agent", "--root", str(tmp_path), "use", LEAK])
    assert code == 1
    err = capsys.readouterr().err
    assert "looks like a secret" in err
    assert "OPENAI_API_KEY is not set" not in err
    assert LEAK not in err


def test_the_session_refuses_a_secret_before_it_looks_for_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    with pytest.raises(InputRejected, match="looks like a secret"):
        repl_line(store, f"remember {LEAK}")


def test_a_clean_input_still_names_the_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wired(tmp_path, monkeypatch, capsys)
    assert main(["--cortex", str(tmp_path), "say", "hello"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
