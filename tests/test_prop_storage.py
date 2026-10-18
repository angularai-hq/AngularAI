"""Properties of the pass log: round trips, garbage between records, repair."""

import json
import tempfile
from collections import Counter
from pathlib import Path

from hypothesis import example, given
from hypothesis import strategies as st

from angularai.models import PassRecord, Trace
from angularai.store import CORRUPT, LOG, _pass_json, init_cortex, open_store

text = st.text(min_size=1, max_size=60)
stat = st.integers(0, 1_000_000)
trace = st.builds(
    Trace,
    id=st.text(min_size=1, max_size=20),
    text=st.text(min_size=1, max_size=240),
    wavelengths=st.lists(st.text(max_size=12), max_size=4).map(tuple),
    source=text,
    at=st.text(min_size=1, max_size=40),
    state=st.just("current"),
    superseded_by=st.none(),
)
record = st.builds(
    PassRecord,
    at=st.text(min_size=1, max_size=40),
    outcome=st.sampled_from(["ok", "held", "error"]),
    user=st.text(min_size=1, max_size=400),
    reply=st.none() | text,
    remember=st.lists(trace, max_size=3).map(tuple),
    supersede=st.lists(
        st.tuples(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=40)), max_size=3
    ).map(tuple),
    error=st.none() | st.text(min_size=1, max_size=300),
    budget=stat,
    used=stat,
    traces_shown=stat,
    traces_held=stat,
    episodes_shown=stat,
    episodes_held=stat,
)
# One line of garbage: any bytes but a newline.
garbage = st.binary(max_size=200).map(lambda raw: raw.replace(b"\n", b"")) | st.sampled_from(
    [
        b"[" * 100_000,
        b"1" * 5000,
        b'{"outcome": "ok"}',
        b"{" + b'"a":' * 50 + b"}",
        b"\xff\xfe\x00",
        b" \t ",
        b"null",
    ]
)


def _line(item: PassRecord) -> bytes:
    return json.dumps(_pass_json(item), ensure_ascii=False).encode("utf-8")


@given(st.lists(record, max_size=8))
def test_records_round_trip(records: list[PassRecord]) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        init_cortex(root)
        store = open_store(root)
        for item in records:
            store.append_pass(item)
        assert open_store(root).passes() == tuple(records)
        assert store.passes() == tuple(records)


@given(st.lists(st.one_of(record.map(lambda r: ("good", r)), garbage.map(lambda g: ("bad", g)))))
# Found while writing this property: nesting past the recursion limit, and an int past 4300 digits,
# raised from json.loads instead of marking the line unreadable.
@example([("bad", b"[" * 100_000)])
@example([("bad", b"1" * 5000)])
# Found by Hypothesis: repair dropped a line of only spaces and tabs.
@example([("bad", b" \t ")])
def test_garbage_lines_never_crash_and_good_records_survive(
    lines: list[tuple[str, PassRecord | bytes]],
) -> None:
    good = [item for kind, item in lines if kind == "good"]
    data = b"".join(
        (_line(item) if isinstance(item, PassRecord) else item) + b"\n" for _, item in lines
    )
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        init_cortex(root)
        (root / LOG).write_bytes(data)
        store = open_store(root)
        assert store.passes() == tuple(good)
        before = [line for line in data.split(b"\n") if line]

        report = store.repair()

        kept = [line for line in (root / LOG).read_bytes().split(b"\n") if line]
        moved_path = root / CORRUPT
        moved = (
            [line for line in moved_path.read_bytes().split(b"\n") if line]
            if moved_path.exists()
            else []
        )
        # Every byte of every line is in the log or in the sidecar, in order.
        assert Counter(kept + moved) == Counter(before)
        assert kept == [line for line in before if line in kept]
        assert moved == [line for line in before if line not in kept]
        assert report.moved == len(moved)
        assert open_store(root).passes() == tuple(good)
        assert open_store(root).damage() == ()
