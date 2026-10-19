"""Properties of cortexes that talk: names stay home, secrets never cross, caps hold."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from angularai.errors import InputRejected
from angularai.lock import HELD
from angularai.models import Config, Peer
from angularai.names import named_root, valid_name
from angularai.peers import Cortex, ask, converse
from angularai.store import LOG, Store, init_cortex, open_store
from tests.peerfakes import Voice, reply

ALPHABET = "abcxyz019-_./\\~ \n\x00A..é"


def _store(root: Path, budget: int = 2000, locks: tuple[str, ...] = ()) -> Store:
    init_cortex(root)
    store = open_store(root)
    store.save_config(
        Config("openai", "demo", None, budget, min(800, budget), locks, False),
    )
    return store


@given(st.text(alphabet=ALPHABET, max_size=40))
@example("../x")
@example("..")
@example("a/../../b")
@example("default")
def test_a_name_resolves_to_a_child_of_the_home_or_is_refused(name: str) -> None:
    home = Path("/nonexistent-home")
    environ = {"ANGULARAI_HOME": str(home)}
    if valid_name(name):
        root = named_root(name, environ)
        assert root.parent == home
        assert root.name == name
        assert "/" not in name and "\\" not in name and "." not in name
        assert name == name.lower()
    else:
        with pytest.raises(InputRejected):
            named_root(name, environ)


_KEYS = st.sampled_from(
    [
        "sk-proj-Zq81mW0xLr5pVt72nKd9",
        "ghp_Q7w8E9r0T1y2U3i4O5p6A7s8",
        "AKIAQWERTYUIOP123456",
        "xoxb-1234567890-abcdefABCDEF",
        "password=Hunter2!Hunter2",
        "hf_aB3dE5gH7jK9lM1nP3qR5sT7uV9wX1yZ3",
    ]
)
_PROSE = st.text(alphabet="abcdefghij klmnop.,", max_size=60)


@given(_PROSE, _KEYS, _PROSE)
def test_a_secret_in_an_answer_never_reaches_the_asker(left: str, key: str, right: str) -> None:
    with tempfile.TemporaryDirectory() as raw:
        home = Path(raw)
        a = _store(home / "ana")
        b = _store(home / "bo")
        voice = Voice("bo", [reply(f"{left} {key} {right}", f"remember {key}")])
        answer = ask(Cortex("ana", a), Cortex("bo", b, voice), "What do you hold?")
        assert answer.held and answer.reply == HELD
        for store in (a, b):
            assert key not in (store.root / LOG).read_text(encoding="utf-8")


@given(_PROSE, _KEYS)
def test_a_secret_in_a_turn_never_crosses(prose: str, key: str) -> None:
    with tempfile.TemporaryDirectory() as raw:
        home = Path(raw)
        a = _store(home / "ana")
        b = _store(home / "bo")
        bo = Voice("bo")
        ana = Voice("ana", [reply(f"{prose} {key}")])
        result = converse(Cortex("ana", a, ana), Cortex("bo", b, bo), "Go.", 6, 50_000)
        assert result.stop == "held"
        assert bo.seen == []
        assert key not in bo.sent()
        assert b.passes() == ()


@given(
    turns=st.integers(1, 12),
    tokens=st.integers(1000, 12_000),
    budgets=st.tuples(st.integers(256, 1000), st.integers(256, 1000)),
    words=st.lists(st.sampled_from(["plan", "launch", "friday", "badge", "noon"]), max_size=8),
)
def test_a_conversation_never_passes_its_caps(
    turns: int, tokens: int, budgets: tuple[int, int], words: list[str]
) -> None:
    with tempfile.TemporaryDirectory() as raw:
        home = Path(raw)
        a = _store(home / "ana", budgets[0])
        b = _store(home / "bo", budgets[1])
        ana, bo = Voice("ana"), Voice("bo")
        opening = " ".join(["Go", *words])
        result = converse(Cortex("ana", a, ana), Cortex("bo", b, bo), opening, turns, tokens)
        assert 1 <= len(result.lines) <= turns
        assert result.used <= tokens
        assert result.used == sum(line.used for line in result.lines)
        for workspace in ana.seen:
            assert workspace.estimated_tokens <= budgets[0]
        for workspace in bo.seen:
            assert workspace.estimated_tokens <= budgets[1]
        assert len(a.passes()) + len(b.passes()) == len(result.lines)
        assert all(r.peer == Peer("ana", "answered") for r in b.passes())
