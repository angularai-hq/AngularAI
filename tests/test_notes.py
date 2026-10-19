"""Notes, ranked recall, and forget: memory without a model."""

import json
import tempfile
import threading
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from angularai.budget import estimate_tokens
from angularai.cli import main
from angularai.errors import InputRejected, InputTooLong, LockHeld, SecretRefused, TraceNotFound
from angularai.models import Integration, TraceDraft
from angularai.names import resolve_root
from angularai.notes import Forgot, forget, note, rank, recall_within
from angularai.parse import trace_draft, valid_trace_id
from angularai.pilot import pass_turn
from angularai.semantic import VECTORS, SemanticRecall, drop_vectors, load_vectors
from angularai.store import LOG, Store, init_cortex, open_store
from tests.scripted import Scripted, noted
from tests.test_semantic import Fake

KEY = "sk-proj-" + "Ab3dE5gH7jK9mN1pQ3sT5vX7z"


def fresh(root: Path) -> Store:
    init_cortex(root)
    return open_store(root)


def test_a_note_is_one_pass_with_its_trace(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    result = note(store, "  The notebook   is blue. ", ["color", "Notebook"], source="mcp")
    assert result.created
    trace = result.trace
    assert valid_trace_id(trace.id)
    assert trace.text == "The notebook is blue."
    assert trace.wavelengths == ("color", "notebook")
    assert trace.source == "mcp"
    (record,) = open_store(tmp_path).passes()
    assert record.user == "The notebook is blue." and record.reply is None
    assert record.remember == (trace,)


def test_the_same_note_twice_is_written_once(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    first = note(store, "Catering is at noon.")
    again = note(store, "Catering   is at noon.")
    assert not again.created and again.trace == first.trace
    assert len(store.passes()) == 1


@pytest.mark.parametrize(
    ("text", "tags", "error"),
    [
        ("", (), InputRejected),
        ("x" * 241, (), InputTooLong),
        ("x" * 20001, (), InputTooLong),
        (f"my key is {KEY}", (), SecretRefused),
        ("password=hunter2hunter2", (), SecretRefused),
        ("fine", ("Bad Tag",), InputRejected),
        ("fine", ("a", "b", "c", "d", "e"), InputRejected),
    ],
)
def test_a_bad_note_writes_nothing(
    tmp_path: Path, text: str, tags: tuple[str, ...], error: type[Exception]
) -> None:
    store = fresh(tmp_path)
    with pytest.raises(error):
        note(store, text, tags)
    assert (tmp_path / LOG).read_bytes() == b""


def test_a_secret_from_the_environment_is_refused(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    with pytest.raises(SecretRefused):
        note(store, "the code is plain4you", secrets=("plain4you",))


def test_the_value_lock_holds_a_note(tmp_path: Path) -> None:
    assert main(["--cortex", str(tmp_path), "init"]) == 0
    assert main(["--cortex", str(tmp_path), "lock", "add", "project falcon"]) == 0
    store = open_store(tmp_path)
    with pytest.raises(LockHeld):
        note(store, "Project  FALCON ships in May.")
    with pytest.raises(LockHeld):
        note(store, "fine", source="about project falcon")
    assert store.passes() == ()


def test_a_note_is_embedded_when_recall_is_on(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    fake = Fake()
    trace = note(store, "The notebook is blue.", recall=SemanticRecall(tmp_path, fake)).trace
    assert fake.sent == [["The notebook is blue."]]
    assert trace.id in load_vectors(tmp_path, "fake-1")


def test_rank_matches_recall_order_and_marks_meaning(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    blue = note(store, "The notebook is blue.").trace
    noon = note(store, "Catering is at noon.").trace
    view = store.view()
    assert [item.trace for item in rank(view, "notebook")] == [blue]
    found = rank(view, "lunch", {noon.id: 0.7, blue.id: 0.1})
    assert [(item.trace, item.match) for item in found] == [(noon, "meaning"), (blue, "meaning")]
    assert rank(view, "nothing shares") == ()


def test_recall_fits_the_token_cap(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    long = note(store, "alpha " + "word " * 40).trace
    short = note(store, "alpha short").trace
    view = store.view()
    result = recall_within(view, "alpha", max_tokens=10, limit=5)
    assert [item.trace for item in result.found] == [short]
    assert result.matched == 2 and result.used == estimate_tokens(short.text)
    assert recall_within(view, "alpha", max_tokens=1000, limit=1).found[0].trace == long


def test_recall_can_include_superseded_traces(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    pass_turn(store, Scripted(noted("The launch is Friday.")), "Launch Friday")
    old = store.view().traces[0]
    nxt = Integration("Noted.", (TraceDraft("The launch is Monday.", ()),), (old.id,))
    pass_turn(store, Scripted(nxt), "The launch moved, not Friday")
    view = store.view()
    assert [i.trace.id for i in rank(view, "launch")] != [old.id]
    assert old.id not in {i.trace.id for i in rank(view, "launch")}
    assert old.id in {i.trace.id for i in rank(view, "launch", include_superseded=True)}


def test_forget_removes_the_trace_and_its_note(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    keep = note(store, "Catering is at noon.").trace
    gone = note(store, "The notebook is blue.").trace
    (tmp_path / VECTORS).write_text(
        json.dumps({"v": 1, "trace": gone.id}) + "\n" + json.dumps({"v": 1, "trace": keep.id})
    )
    assert forget(store, gone.id) == Forgot(gone, 1)
    raw = (tmp_path / LOG).read_text(encoding="utf-8")
    assert "notebook" not in raw and keep.id in raw
    assert [t.id for t in open_store(tmp_path).view().traces] == [keep.id]
    assert gone.id not in (tmp_path / VECTORS).read_text()
    assert keep.id in (tmp_path / VECTORS).read_text()
    assert (tmp_path / LOG).stat().st_mode & 0o777 == 0o600
    with pytest.raises(TraceNotFound):
        forget(store, gone.id)


def test_forget_keeps_the_episode_of_a_model_pass(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    pass_turn(store, Scripted(noted("Blue.", "color")), "I have a blue notebook.")
    trace = store.view().traces[0]
    forget(store, trace.id)
    (record,) = open_store(tmp_path).passes()
    assert record.remember == () and record.user == "I have a blue notebook."


def test_forget_drops_supersedes_and_keeps_damage(tmp_path: Path) -> None:
    store = fresh(tmp_path)
    gone = note(store, "The launch is Friday.").trace
    log = tmp_path / LOG
    other = json.loads(log.read_text().splitlines()[0])
    other["user"], other["remember"] = "moved", []
    other["supersede"] = [{"id": gone.id, "by": "2026-01-01"}]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("not json\n" + json.dumps(other) + "\n" + '{"torn')
    forget(store, gone.id)
    lines = log.read_text().split("\n")
    assert lines[0] == "not json" and lines[-1] == '{"torn'
    assert json.loads(lines[1])["supersede"] == []


@pytest.mark.parametrize(
    "bad", ["", "t_", "t_ABCDEF123456", "../t_0123456789ab", "t_0123456789ab\n"]
)
def test_forget_needs_an_exact_id(tmp_path: Path, bad: str) -> None:
    store = fresh(tmp_path)
    note(store, "The notebook is blue.")
    before = (tmp_path / LOG).read_bytes()
    with pytest.raises(TraceNotFound):
        forget(store, bad)
    with pytest.raises(TraceNotFound):
        forget(store, "t_0123456789ab")
    assert (tmp_path / LOG).read_bytes() == before


def test_drop_vectors_without_a_file(tmp_path: Path) -> None:
    assert drop_vectors(tmp_path, frozenset({"t_0123456789ab"})) == 0
    (tmp_path / VECTORS).write_text("garbage\n[1]\n")
    assert drop_vectors(tmp_path, frozenset({"t_0123456789ab"})) == 0
    assert (tmp_path / VECTORS).read_text() == "garbage\n[1]\n"


def test_concurrent_notes_all_land(tmp_path: Path) -> None:
    init_cortex(tmp_path)

    def write(n: int) -> None:
        note(open_store(tmp_path), f"Fact number {n} is kept.")

    threads = [threading.Thread(target=write, args=(n,)) for n in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store = open_store(tmp_path)
    assert len(store.view().current_traces()) == 16
    assert store.damage() == ()


def test_resolve_root(tmp_path: Path) -> None:
    env = {"ANGULARAI_HOME": str(tmp_path)}
    assert resolve_root(None, None, env) == tmp_path / "default"
    assert resolve_root(None, "bo", env) == tmp_path / "bo"
    assert resolve_root(tmp_path / "x", None, env) == tmp_path / "x"
    assert resolve_root(None, None, {**env, "ANGULARAI_CORTEX": str(tmp_path / "c")}) == (
        tmp_path / "c"
    )
    with pytest.raises(InputRejected):
        resolve_root(tmp_path, "bo", env)
    with pytest.raises(InputRejected):
        resolve_root(None, "../bo", env)


def test_trace_draft_rules() -> None:
    assert trace_draft("a  b", ["x-1"]) is not None
    assert trace_draft("a", ["a", "b", "c", "d", "e"]) is None
    assert trace_draft("a", ["1abc"]) is None


safe_text = st.text(
    alphabet=st.characters(categories=("Ll", "Lu", "Nd", "Zs"), max_codepoint=0x24F),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip())


@given(st.lists(safe_text, min_size=1, max_size=6, unique_by=lambda s: " ".join(s.split())))
def test_forget_one_keeps_the_rest(texts: list[str]) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "c"
        store = fresh(root)
        written = []
        for text in texts:
            try:
                written.append(note(store, text).trace)
            except InputRejected:
                continue
        if not written:
            return
        forget(store, written[0].id)
        left = [t.id for t in open_store(root).view().traces]
        assert left == [t.id for t in written[1:]]
