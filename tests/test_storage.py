import json
import multiprocessing
import sys
from pathlib import Path

import pytest

from angularai import store as store_module
from angularai.cli import main
from angularai.durable import append_line, atomic_write
from angularai.errors import ConfigError
from angularai.models import PassRecord
from angularai.store import (
    CORRUPT,
    LOG,
    SCHEMA_VERSION,
    Store,
    init_cortex,
    migrate,
    open_store,
)


def record(user: str, at: str = "2026-10-18T00:00:00+00:00-0000") -> PassRecord:
    return PassRecord(
        at=at,
        outcome="ok",
        user=user,
        reply="Noted.",
        remember=(),
        supersede=(),
        error=None,
        budget=2000,
        used=10,
        traces_shown=0,
        traces_held=0,
        episodes_shown=0,
        episodes_held=0,
    )


def line(user: str) -> str:
    return json.dumps(
        {
            "at": "2026-10-18T00:00:00+00:00-0000",
            "outcome": "ok",
            "user": user,
            "reply": "Noted.",
            "remember": [],
            "supersede": [],
            "error": None,
            "budget": 2000,
            "used": 10,
            "traces_shown": 0,
            "traces_held": 0,
            "episodes_shown": 0,
            "episodes_held": 0,
        }
    )


@pytest.fixture
def cortex(tmp_path: Path) -> Path:
    init_cortex(tmp_path / "cortex")
    return tmp_path / "cortex"


def users(store: Store) -> list[str]:
    return [item.user for item in store.passes()]


# Atomic writes


def test_atomic_write_replaces_whole(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_bytes(b"old")
    atomic_write(path, b"new")
    assert path.read_bytes() == b"new"
    assert path.stat().st_mode & 0o777 == 0o600
    assert sorted(item.name for item in tmp_path.iterdir()) == ["f.json"]


def test_a_failed_atomic_write_leaves_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "f.json"
    path.write_bytes(b"old")

    def broken(source: object, target: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("angularai.durable.os.replace", broken)
    with pytest.raises(OSError, match="disk full"):
        atomic_write(path, b"new")
    assert path.read_bytes() == b"old"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["f.json"]


def test_the_config_is_written_atomically(cortex: Path) -> None:
    store = open_store(cortex)
    store.save_config(store.config())
    names = {item.name for item in cortex.iterdir()}
    assert not any(name.endswith(".tmp") for name in names)
    assert json.loads((cortex / "cortex.json").read_text())["version"] == SCHEMA_VERSION


# Appends and torn lines


def test_a_record_is_one_line(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        append_line(tmp_path / "x.jsonl", b"a\nb")


def test_a_torn_final_line_is_skipped_and_reported(cortex: Path) -> None:
    log = cortex / LOG
    log.write_text(line("first") + "\n" + line("second")[:40], encoding="utf-8")
    notices: list[str] = []
    store = open_store(cortex, warn=notices.append)
    assert users(store) == ["first"]
    assert [(item.line, item.kind) for item in store.damage()] == [(2, "torn")]
    assert "passes.jsonl:2" in notices[0]
    assert "cut off" in notices[0]


def test_an_append_after_a_torn_line_starts_a_new_line(cortex: Path) -> None:
    log = cortex / LOG
    log.write_text(line("first") + "\n" + line("second")[:40], encoding="utf-8")
    store = open_store(cortex)
    store.append_pass(record("third"))
    assert users(store) == ["first", "third"]
    assert [(item.line, item.kind) for item in store.damage()] == [(2, "unreadable")]
    assert log.read_bytes().endswith(b"\n")
    assert len(log.read_bytes().splitlines()) == 3


def test_a_bad_line_anywhere_is_skipped_and_warned_once(cortex: Path) -> None:
    log = cortex / LOG
    lines = [
        line("one"),
        "{not json",
        line("two"),
        '{"outcome": "ok"}',
        "\u00ff not json",
        line("three"),
    ]
    log.write_bytes("\n".join(lines).encode("utf-8") + b"\n")
    notices: list[str] = []
    store = open_store(cortex, warn=notices.append)
    assert users(store) == ["one", "two", "three"]
    assert [item.line for item in store.damage()] == [2, 4, 5]
    store.passes()
    store.view()
    assert len(notices) == 1
    assert "3 unreadable lines: passes.jsonl:2, passes.jsonl:4, passes.jsonl:5" in notices[0]
    assert "angularai repair" in notices[0]


def test_invalid_utf8_is_a_bad_line(cortex: Path) -> None:
    (cortex / LOG).write_bytes(line("one").encode() + b"\n\xff\xfe\n")
    store = open_store(cortex)
    assert users(store) == ["one"]
    assert [item.line for item in store.damage()] == [2]


# Repair


def test_repair_moves_bad_lines_and_keeps_them(cortex: Path) -> None:
    log = cortex / LOG
    log.write_text(
        line("one") + "\n{not json\n" + line("two") + "\n" + line("cut")[:30], encoding="utf-8"
    )
    store = open_store(cortex)
    result = store.repair()
    assert (result.moved, result.kept) == (2, 2)
    assert users(store) == ["one", "two"]
    assert store.damage() == ()
    corrupt = (cortex / CORRUPT).read_text(encoding="utf-8").splitlines()
    assert corrupt == ["{not json", line("cut")[:30]]
    assert (cortex / CORRUPT).stat().st_mode & 0o777 == 0o600
    assert store.repair().moved == 0


def test_the_cli_warns_and_repairs(cortex: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (cortex / LOG).write_text(line("one") + "\n{bad\n", encoding="utf-8")
    assert main(["--cortex", str(cortex), "cortex"]) == 0
    err = capsys.readouterr().err
    assert err.count("unreadable") == 1
    assert main(["--cortex", str(cortex), "repair"]) == 0
    assert "Moved 1 unreadable line" in capsys.readouterr().out
    assert main(["--cortex", str(cortex), "repair"]) == 0
    assert "The log is clean. 1 passes." in capsys.readouterr().out
    assert main(["--cortex", str(cortex), "cortex"]) == 0
    assert "unreadable" not in capsys.readouterr().err


# Concurrency


def _writer(root: str, name: str, count: int) -> None:
    store = Store(Path(root))
    for index in range(count):
        store.append_pass(record(f"{name}-{index}-" + "x" * 15000))


@pytest.mark.skipif(sys.platform == "win32", reason="the lock is a no-op on Windows")
def test_concurrent_writers_do_not_interleave(cortex: Path) -> None:
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_writer, args=(str(cortex), f"w{n}", 40)) for n in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    store = open_store(cortex)
    seen = users(store)
    assert store.damage() == ()
    assert len(seen) == 160
    for n in range(4):
        mine = [user.split("-")[1] for user in seen if user.startswith(f"w{n}-")]
        assert mine == [str(index) for index in range(40)]


# Versions and migration


def test_a_cortex_without_a_version_is_version_1(cortex: Path) -> None:
    path = cortex / "cortex.json"
    payload = json.loads(path.read_text())
    del payload["version"]
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    store = open_store(cortex)
    assert store.config().workspace_budget == 2000
    assert path.read_bytes() == before


def test_a_newer_cortex_is_refused_untouched(cortex: Path) -> None:
    path = cortex / "cortex.json"
    payload = json.loads(path.read_text())
    payload["version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with pytest.raises(ConfigError, match=f"version {SCHEMA_VERSION + 1}. This AngularAI reads"):
        open_store(cortex)
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [0, -1, "1", True, 1.5])
def test_a_bad_version_is_refused(cortex: Path, value: object) -> None:
    path = cortex / "cortex.json"
    payload = json.loads(path.read_text())
    payload["version"] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="version"):
        open_store(cortex)


def test_migrate_is_identity_at_version_1() -> None:
    payload: dict[object, object] = {"model": None, "version": 1}
    migrated, found = migrate(payload)
    assert found == 1
    assert migrated == payload


def test_a_migration_runs_once_and_keeps_the_old_file(
    cortex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def one_to_two(payload: dict[str, object]) -> dict[str, object]:
        calls.append(1)
        return {**payload, "workspace_budget": 3000}

    monkeypatch.setattr(store_module, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(store_module, "MIGRATIONS", {1: one_to_two})
    path = cortex / "cortex.json"
    payload = json.loads(path.read_text())
    payload["version"] = 1
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    store = open_store(cortex)
    assert json.loads(path.read_text())["version"] == 2
    assert store.config().workspace_budget == 3000
    assert (cortex / "cortex.json.v1").read_bytes() == original
    open_store(cortex)
    assert calls == [1]


def test_a_missing_migration_is_named(cortex: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(store_module, "MIGRATIONS", {})
    with pytest.raises(ConfigError, match="No migration from cortex version 1"):
        open_store(cortex)


# The read cache


def test_appends_are_read_incrementally(cortex: Path) -> None:
    store = open_store(cortex)
    other = open_store(cortex)
    store.append_pass(record("one"))
    assert users(store) == ["one"]
    other.append_pass(record("two"))
    other.append_pass(record("three"))
    assert users(store) == ["one", "two", "three"]


def test_a_rewritten_log_is_read_again(cortex: Path) -> None:
    store = open_store(cortex)
    store.append_pass(record("aaaa"))
    store.append_pass(record("bbbb"))
    assert users(store) == ["aaaa", "bbbb"]
    log = cortex / LOG
    data = log.read_bytes().replace(b"bbbb", b"cccc")
    with log.open("r+b") as handle:
        handle.write(data)
    assert users(store) == ["aaaa", "cccc"]
    atomic_write(log, line("only").encode() + b"\n")
    assert users(store) == ["only"]
    log.write_bytes(b"")
    assert users(store) == []
