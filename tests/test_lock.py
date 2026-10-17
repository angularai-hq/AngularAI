from dataclasses import replace

import pytest

from angularai.attention import score
from angularai.errors import InputDoesNotFit, InputRejected, LockBypass, SubstrateUnreadable
from angularai.lock import HELD, Thinker, ValueLock
from angularai.models import Integration, TraceDraft, Workspace
from angularai.parse import parse_integration
from angularai.pilot import pass_turn
from angularai.secrets import redact
from angularai.store import init_cortex, open_store
from tests.scripted import Scripted


def test_a_plural_meets_its_singular() -> None:
    assert score("live", "lives") == 1
    assert score("live", "the warehouse counted cedar planks at dawn") == 0


def test_value_lock_holds_the_reply_and_drops_the_trace(tmp_path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    store.save_config(replace(store.config(), forbid=("dosing chart",)))
    result = pass_turn(
        store,
        Scripted(
            Integration(
                "here is a dosing chart",
                (TraceDraft("Take two tablets.", ("med",)),),
                (),
            )
        ),
        "What should I take?",
    )
    log = (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    assert result.held is True
    assert result.reply == HELD
    assert "dosing chart" not in log
    assert "Take two tablets." not in log
    assert store.view().current_traces() == ()


def test_thinker_rejects_a_verdict_from_another_lock() -> None:
    lock = ValueLock()
    other = ValueLock()
    with pytest.raises(LockBypass):
        Thinker(lock).release(other.judge("hello", (), ()))
    assert Thinker(lock).release(lock.judge("hello", (), ())) == "hello"


def test_a_secret_in_the_input_is_not_sent(tmp_path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)

    class Sent:
        def integrate(self, workspace: Workspace) -> Integration:
            raise AssertionError(workspace.input_text)

    with pytest.raises(InputRejected, match="secret"):
        pass_turn(store, Sent(), "the key is sk-live-abcdefghij123")
    assert (tmp_path / "passes.jsonl").read_text(encoding="utf-8") == ""


def test_a_secret_trace_is_dropped(tmp_path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    result = pass_turn(
        store,
        Scripted(
            Integration(
                "Noted.",
                (TraceDraft("key sk-live-abcdefghij123", ("secret",)),),
                (),
            )
        ),
        "Remember this.",
    )
    log = (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    assert result.reply == "Noted."
    assert "sk-live" not in log
    assert store.view().current_traces() == ()


def test_input_over_the_cap_writes_nothing(tmp_path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    store.save_config(replace(store.config(), max_input_tokens=32))
    before = (tmp_path / "passes.jsonl").read_text(encoding="utf-8")
    with pytest.raises(InputDoesNotFit):
        pass_turn(store, Scripted(), "word " * 80)
    assert (tmp_path / "passes.jsonl").read_text(encoding="utf-8") == before


def test_redact_removes_the_key() -> None:
    assert "sk-test" not in redact("rejected sk-test-secret-value", ("sk-test-secret-value",))


def test_parse_accepts_a_fence_and_drops_a_bad_trace() -> None:
    raw = "\n".join(
        (
            "```json",
            '{"reply":"Hello.","remember":[',
            '{"text":"Fine.","wavelengths":["place"]},',
            '{"text":"","wavelengths":[]},',
            '{"text":"nope","wavelengths":["9bad"]}',
            '],"supersede":["nope","t_abc123def456"]}',
            "```",
        )
    )
    parsed = parse_integration(raw)
    assert parsed.reply == "Hello."
    assert parsed.remember == (TraceDraft("Fine.", ("place",)),)
    assert parsed.supersede == ("t_abc123def456",)


def test_a_dense_brief_keeps_more_than_six_facts() -> None:
    facts = ",".join(
        f'{{"text":"Fact number {index} stays.","wavelengths":["fact"]}}' for index in range(10)
    )
    parsed = parse_integration(f'{{"reply":"Noted.","remember":[{facts}],"supersede":[]}}')
    assert len(parsed.remember) == 10
    assert parsed.remember[-1].text == "Fact number 9 stays."


def test_parse_rejects_a_missing_reply() -> None:
    with pytest.raises(SubstrateUnreadable):
        parse_integration('{"remember":[]}')
