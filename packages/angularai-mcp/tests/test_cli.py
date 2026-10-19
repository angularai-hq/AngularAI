"""angularai-mcp on the command line. No server is left running."""

import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from angularai.names import valid_name
from angularai_mcp import cli
from angularai_mcp.server import AngularAIServer
from angularai_mcp.web import BearerGate

TOKEN = "t0ken-for-tests-0123456789"


class Ran:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []


@pytest.fixture
def ran(monkeypatch: pytest.MonkeyPatch) -> Ran:
    record = Ran()

    def run(self: AngularAIServer, transport: str = "stdio") -> None:
        record.calls.append(("run", transport))

    def serve(self: object) -> None:
        config = self.config  # type: ignore[attr-defined]
        record.calls.append(("http", (config.host, config.port, type(config.app).__name__)))

    monkeypatch.setattr(AngularAIServer, "run", run)
    monkeypatch.setattr("uvicorn.Server.run", serve)
    return record


def test_stdio_is_the_default(cortex: Path, ran: Ran, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--cortex", str(cortex)]) == 0
    assert ran.calls == [("run", "stdio")]
    assert capsys.readouterr().out == ""


def test_the_default_cortex_comes_from_the_home(
    home: Path, cortex: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    assert cli.main([]) == 0
    assert ran.calls == [("run", "stdio")]


def test_init_creates_a_named_cortex_once(
    home: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home / "nested"))
    assert cli.main(["--name", "bo", "--init"]) == 0
    assert (home / "nested" / "bo" / "cortex.json").is_file()
    assert (home / "nested").stat().st_mode & 0o777 == 0o700
    assert cli.main(["--name", "bo", "--init"]) == 0
    out = capsys.readouterr()
    assert out.out == "" and out.err.count("Cortex is at") == 1


@pytest.mark.parametrize(
    "name", ["../evil", "..", ".", "a/b", "/etc", "UPPER", "con", "-dash", "a" * 33, "a\x00b", ""]
)
def test_a_name_that_could_leave_the_home_is_refused(
    home: Path,
    ran: Ran,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    before = sorted(home.parent.rglob("*"))
    assert cli.main([f"--name={name}", "--init"]) == 2
    assert ran.calls == []
    assert sorted(home.parent.rglob("*")) == before
    assert "cortex name" in capsys.readouterr().err


def test_a_link_in_place_of_a_named_cortex_is_refused(
    home: Path, tmp_path: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / "bo").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("ANGULARAI_HOME", str(home))
    assert cli.main(["--name", "bo", "--init"]) == 2
    assert list(outside.iterdir()) == [] and ran.calls == []


def test_a_relative_home_and_both_flags_are_refused(
    tmp_path: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANGULARAI_HOME", "relative/home")
    assert cli.main(["--name", "bo"]) == 2
    monkeypatch.delenv("ANGULARAI_HOME")
    assert cli.main(["--name", "bo", "--cortex", str(tmp_path)]) == 2
    assert ran.calls == []


@given(st.text(max_size=40))
def test_any_name_is_valid_or_refused_before_disk(raw: str) -> None:
    from angularai.errors import InputRejected
    from angularai.names import resolve_root

    home = Path("/nonexistent-angularai-home")
    try:
        root = resolve_root(None, raw, {"ANGULARAI_HOME": str(home)})
    except InputRejected:
        assert not valid_name(raw)
        return
    assert valid_name(raw) and root.parent == home and root.name == raw


def test_http_binds_loopback(cortex: Path, ran: Ran) -> None:
    assert cli.main(["--cortex", str(cortex), "--http", "--port", "9123"]) == 0
    assert ran.calls == [("http", ("127.0.0.1", 9123, "Starlette"))]


def test_http_with_a_token_is_gated(
    cortex: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_MCP_TOKEN", TOKEN)
    assert cli.main(["--cortex", str(cortex), "--http", "--host", "::1"]) == 0
    assert ran.calls == [("http", ("::1", 8765, BearerGate.__name__))]
    err = capsys.readouterr().err
    assert "http://[::1]:8765/mcp" in err and "bearer token required" in err
    assert TOKEN not in err


def test_beyond_loopback_needs_the_flag_and_the_token(
    cortex: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--cortex", str(cortex), "--http", "--host", "0.0.0.0"]
    assert cli.main(base) == 2
    assert "--allow-remote" in capsys.readouterr().err
    assert cli.main([*base, "--allow-remote"]) == 2
    assert "ANGULARAI_MCP_TOKEN" in capsys.readouterr().err
    monkeypatch.setenv("ANGULARAI_MCP_TOKEN", TOKEN)
    assert cli.main([*base, "--allow-remote"]) == 0
    assert ran.calls == [("http", ("0.0.0.0", 8765, "BearerGate"))]


def test_a_short_token_is_refused_without_echo(
    cortex: Path, ran: Ran, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANGULARAI_MCP_TOKEN", "short9")
    assert cli.main(["--cortex", str(cortex), "--http"]) == 2
    assert "short9" not in capsys.readouterr().err
    monkeypatch.setenv("ANGULARAI_MCP_TOKEN", "has a space but is long enough")
    assert cli.main(["--cortex", str(cortex), "--http"]) == 2
    monkeypatch.setenv("ANGULARAI_MCP_TOKEN", "   ")
    assert cli.main(["--cortex", str(cortex), "--http"]) == 0


def test_interrupt_exits_130(cortex: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stop(self: AngularAIServer, transport: str = "stdio") -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(AngularAIServer, "run", stop)
    assert cli.main(["--cortex", str(cortex)]) == 130


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == "angularai-mcp 0.1.0"


def test_the_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setattr("angularai_mcp.cli.main", lambda: 7)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("angularai_mcp", run_name="__main__")
    assert exit_info.value.code == 7
    assert "ANGULARAI_MCP_TOKEN" not in os.environ
