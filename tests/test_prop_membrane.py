"""Properties of the membrane: the token estimate, the assembled workspace, trimming."""

from dataclasses import replace

from hypothesis import example, given
from hypothesis import strategies as st

from angularai.assemble import _clip, assemble, focus
from angularai.budget import estimate_tokens
from angularai.errors import InputDoesNotFit
from angularai.models import CortexView, Episode, Trace
from angularai.store import default_config

# Unicode of every kind, emoji and CJK included, plus long runs.
texts = st.text() | st.text(alphabet="你好世界😀👩‍👩‍👧 \n\t", max_size=400)
huge = st.builds(
    lambda piece, times: piece * times, st.text(min_size=1, max_size=8), st.integers(0, 3000)
)


@given(texts, texts)
def test_the_estimate_is_monotonic_additive_and_never_negative(a: str, b: str) -> None:
    whole = estimate_tokens(a + b)
    assert estimate_tokens(a) >= 0
    assert estimate_tokens(a) <= whole
    assert estimate_tokens(b) <= whole
    # Additive within rounding: each part rounds up by less than one token.
    assert whole <= estimate_tokens(a) + estimate_tokens(b) <= whole + 1
    assert (estimate_tokens(a) == 0) == (a == "")


@given(st.text(), st.integers(0, 50))
def test_a_longer_text_never_counts_less(text: str, more: int) -> None:
    assert estimate_tokens(text + "x" * more) >= estimate_tokens(text)


words = st.text(alphabet="abcdefghij klmn", min_size=1, max_size=60).filter(str.strip)
# Latin and CJK words in one line, so a mixed question still finds mixed traces.
mixed = st.text(alphabet="abcdefg 你好世界 안녕 カタ ＡＢ 😀", min_size=1, max_size=60).filter(
    str.strip
)
traces = st.lists(
    st.builds(
        lambda index, text: Trace(
            id=f"t{index}",
            text=text,
            wavelengths=(),
            source="s",
            at=f"2026-01-01T00:00:{index:02d}Z",
            state="current",
            superseded_by=None,
        ),
        st.integers(0, 59),
        words | mixed | texts,
    ),
    max_size=12,
    unique_by=lambda trace: trace.id,
)
episodes = st.lists(
    st.builds(
        lambda index, user, reply: Episode(
            at=f"2026-01-02T00:00:{index:02d}Z", user=user, reply=reply, outcome="ok"
        ),
        st.integers(0, 59),
        words | mixed | texts | huge,
        st.none() | words | mixed | huge,
    ),
    max_size=8,
    unique_by=lambda episode: episode.at,
)


@given(
    traces,
    episodes,
    texts | huge | words | mixed,
    st.integers(1, 4000),
    st.integers(1, 4000),
    st.lists(words, max_size=3),
)
@example([], [], "", 1, 1, [])
@example([], [], "😀" * 4000, 2000, 4000, [])
@example([], [], "你好 hello 世界 " * 300, 2000, 4000, [])
def test_the_workspace_never_exceeds_the_budget(
    found: list[Trace],
    said: list[Episode],
    text: str,
    budget: int,
    cap: int,
    forbid: list[str],
) -> None:
    config = replace(
        default_config(), workspace_budget=budget, max_input_tokens=cap, forbid=tuple(forbid)
    )
    view = CortexView(tuple(found), tuple(said))
    try:
        workspace = assemble(view, text, config)
    except InputDoesNotFit:
        return
    assert workspace.estimated_tokens <= budget
    real = estimate_tokens(workspace.system) + estimate_tokens(workspace.user)
    assert real == workspace.estimated_tokens
    assert estimate_tokens(workspace.input_text) <= cap
    # The same inputs pack the same workspace.
    assert assemble(view, text, config) == workspace


@given(texts | huge, st.text(alphabet="abc xyz", max_size=20), st.integers(8, 600))
def test_focus_fits_and_is_idempotent(text: str, query: str, limit: int) -> None:
    once = focus(text, query, limit)
    assert len(once) <= limit
    assert focus(once, query, limit) == once


@given(st.text(), st.integers(3, 500))
def test_clip_fits_and_is_idempotent(text: str, limit: int) -> None:
    once = _clip(text, limit)
    assert len(once) <= limit
    assert _clip(once, limit) == once
