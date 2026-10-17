import json
from pathlib import Path

import pytest

from angularai.errors import ConfigError, CortexExists, LogDamaged
from angularai.models import Integration, Trace, TraceDraft
from angularai.pilot import pass_turn
from angularai.store import init_cortex, open_store
from tests.scripted import Scripted


def test_init_is_private_and_refuses_to_overwrite(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    assert (tmp_path / "cortex.json").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "passes.jsonl").stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    with pytest.raises(CortexExists):
        init_cortex(tmp_path)


def test_a_damaged_log_is_named(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    with (tmp_path / "passes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(LogDamaged, match="line 1"):
        open_store(tmp_path).view()


def test_an_unknown_config_field_is_refused(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    path = tmp_path / "cortex.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["api_key"] = "sk-test-secret-value"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="api_key"):
        open_store(tmp_path).config()


def test_superseded_text_stays_in_the_log(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    pass_turn(
        store,
        Scripted(Integration("Noted.", (TraceDraft("The notebook is blue.", ("place",)),), ())),
        "I have a blue notebook.",
    )
    trace = store.view().current_traces()[0]
    pass_turn(
        store,
        Scripted(
            Integration("Noted.", (TraceDraft("The notebook is red.", ("object",)),), (trace.id,))
        ),
        "The notebook is red.",
    )
    view = store.view()
    old = next(item for item in view.traces if item.text == "The notebook is blue.")
    assert old.state == "superseded"
    assert isinstance(old, Trace)
    current = view.current_traces()
    assert [item.text for item in current] == ["The notebook is red."]
    assert "The notebook is blue." in (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
