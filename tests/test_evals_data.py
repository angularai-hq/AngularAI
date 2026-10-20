import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from angularai.budget import estimate_tokens
from angularai.evals import dataset
from angularai.evals.dataset import DatasetError, Question, Session, Turn, fetch, load, parse
from angularai.evals.reduce import (
    allocate,
    cut,
    history_tokens,
    read_subset,
    reduce,
    write_subset,
)
from angularai.evals.render import (
    SYSTEM,
    full_prompt,
    oracle_prompt,
    session_text,
    truncated_prompt,
)
from tests.evalfakes import question, raw_question


def test_parse_reads_a_question() -> None:
    (q,) = parse([raw_question(answer=7)])
    assert q.id == "q1"
    assert q.answer == "7"
    assert len(q.sessions) == 4
    assert q.sessions[1].turns[0].content.startswith("My notebook is blue.")
    assert q.evidence == ("sq1-1",)
    assert not q.abstention
    assert parse([raw_question(qid="x_abs")])[0].abstention


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question_id", ""),
        ("question_type", "other"),
        ("answer", None),
        ("answer", True),
        ("haystack_dates", ["one"]),
        ("haystack_session_ids", "nope"),
        ("haystack_sessions", [[{"role": "system", "content": "x"}]] * 4),
        ("haystack_sessions", [["turn"]] * 4),
        ("haystack_sessions", ["nope"] * 4),
    ],
)
def test_parse_refuses_a_bad_question(field: str, value: object) -> None:
    item = raw_question()
    item[field] = value
    with pytest.raises(DatasetError):
        parse([item])


def test_parse_refuses_what_is_not_a_list() -> None:
    with pytest.raises(DatasetError):
        parse({"questions": []})
    with pytest.raises(DatasetError):
        parse(["nope"])


def _pin(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    import hashlib

    monkeypatch.setattr(dataset, "SHA256", hashlib.sha256(data).hexdigest())


def test_fetch_downloads_once_and_checks_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = json.dumps([raw_question()]).encode()
    _pin(monkeypatch, data)
    calls: list[str] = []

    def download(url: str, destination: Path) -> None:
        calls.append(url)
        destination.write_bytes(data)

    path = fetch(tmp_path, download)
    assert fetch(tmp_path, download) == path
    assert calls == [dataset.URL]
    assert dataset.REVISION in dataset.URL
    assert load(path)[0].id == "q1"


def test_fetch_refuses_a_file_that_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, b"expected")

    def download(url: str, destination: Path) -> None:
        destination.write_bytes(b"something else")

    with pytest.raises(DatasetError, match="SHA-256"):
        fetch(tmp_path, download)
    assert list(tmp_path.iterdir()) == []


def test_load_refuses_missing_changed_and_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data.json"
    with pytest.raises(DatasetError, match="fetch"):
        load(path)
    path.write_bytes(b"[]")
    with pytest.raises(DatasetError, match="SHA-256"):
        load(path)
    path.write_bytes(b"{not json")
    _pin(monkeypatch, b"{not json")
    with pytest.raises(DatasetError, match="not JSON"):
        load(path)
    with pytest.raises(DatasetError, match="not JSON"):
        load(path, verify=False)


def _parsed(**kwargs: object) -> Question:
    return parse([raw_question(**kwargs)])[0]  # type: ignore[arg-type]


def test_cut_keeps_evidence_and_order_within_the_cap() -> None:
    q = _parsed(sessions=12, evidence=(3, 9), words=60)
    whole = history_tokens(q.sessions)
    cap = whole // 2
    reduced = cut(q, cap, seed=0)
    assert reduced is not None
    ids = [s.id for s in reduced.sessions]
    assert {"sq1-3", "sq1-9"} <= set(ids)
    assert ids == sorted(ids, key=lambda sid: int(sid.split("-")[1]))
    assert history_tokens(reduced.sessions) <= cap
    assert len(ids) < 12
    assert cut(q, cap, seed=0) == reduced
    assert cut(q, whole, seed=0) is not None


def test_cut_refuses_evidence_that_does_not_fit() -> None:
    q = _parsed(sessions=3, evidence=(0,), words=400)
    assert cut(q, 50, seed=0) is None
    assert cut(_parsed(evidence=()), 10_000, seed=0) is None


@given(st.integers(min_value=200, max_value=3000), st.integers(min_value=0, max_value=5))
def test_cut_never_passes_the_cap(cap: int, seed: int) -> None:
    q = _parsed(sessions=9, evidence=(4,), words=30)
    reduced = cut(q, cap, seed)
    if reduced is not None:
        assert history_tokens(reduced.sessions) <= cap
        assert "sq1-4" in {s.id for s in reduced.sessions}


def test_allocate_is_proportional_and_exact() -> None:
    shares = allocate({"a": 10, "b": 30, "c": 0}, 8)
    assert shares == {"a": 2, "b": 6, "c": 0}
    assert sum(allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2
    assert allocate({"a": 0}, 0) == {"a": 0}
    with pytest.raises(DatasetError):
        allocate({"a": 1}, 2)


def test_reduce_is_stratified_seeded_and_round_trips(tmp_path: Path) -> None:
    raw = [raw_question(qid=f"u{i}", kind="single-session-user") for i in range(6)]
    raw += [raw_question(qid=f"m{i}", kind="multi-session") for i in range(6)]
    raw += [raw_question(qid="big", kind="multi-session", words=4000)]
    questions = parse(raw)
    chosen, manifest = reduce(questions, 4, seed=1, haystack_tokens=400)
    assert manifest.eligible == 12
    assert manifest.total == 13
    assert manifest.per_type == {"multi-session": 2, "single-session-user": 2}
    assert [q.id for q in chosen] == sorted(manifest.questions)
    assert reduce(questions, 4, seed=1, haystack_tokens=400)[0] == chosen
    path = tmp_path / "subset.json"
    write_subset(path, chosen, manifest)
    assert read_subset(path) == (chosen, manifest)


def test_read_subset_refuses_other_files(tmp_path: Path) -> None:
    path = tmp_path / "subset.json"
    path.write_text("{}")
    with pytest.raises(DatasetError):
        read_subset(path)
    chosen, manifest = reduce(parse([raw_question()]), 1, 0, 10_000)
    write_subset(path, chosen, manifest)
    payload = json.loads(path.read_text())
    payload["manifest"]["format"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(DatasetError, match="format"):
        read_subset(path)


def test_prompts_carry_history_date_and_question() -> None:
    q = question()
    full = full_prompt(q)
    assert "### Session 1:" in full and "### Session 3:" in full
    assert "Session Date: 2023/05/10 (Wed) 10:00" in full
    assert full.endswith(
        "Current Date: 2023/05/30 (Tue) 23:40\nQuestion: What color is my notebook?\nAnswer:"
    )
    oracle = oracle_prompt(q)
    assert "Turn 0: my notebook is blue." in oracle and "Turn 1." not in oracle


def test_truncated_keeps_the_newest_turns_that_fit() -> None:
    q = question(sessions=3)
    whole = truncated_prompt(q, 100_000)
    assert whole == full_prompt(q)
    tight = estimate_tokens(SYSTEM) + estimate_tokens(full_prompt(q)) - 5
    cut_prompt = truncated_prompt(q, tight)
    assert "Turn 0: my notebook is blue." not in cut_prompt
    assert "Answer 2." in cut_prompt
    nothing = truncated_prompt(q, 10)
    assert "### Session" not in nothing


@given(st.integers(min_value=120, max_value=1200))
def test_truncated_never_passes_its_budget(budget: int) -> None:
    q = Question(
        id="p",
        type="multi-session",
        question="Where is it?",
        answer="here",
        date="2023/01/01 (Sun) 00:00",
        sessions=tuple(
            Session(f"s{i}", "2023/01/01 (Sun) 00:00", (Turn("user", "word " * (i * 7 + 3)),))
            for i in range(8)
        ),
        evidence=("s0",),
    )
    prompt = truncated_prompt(q, budget)
    if "### Session" in prompt:
        assert estimate_tokens(SYSTEM) + estimate_tokens(prompt) <= budget


def test_session_text_numbers_sessions() -> None:
    q = question()
    assert session_text(5, q.sessions[0]).startswith("### Session 5:\n")
