"""Properties of mask: at most the last four characters of a key ever show."""

import string

from hypothesis import given
from hypothesis import strategies as st

from angularai.profiles import mask

_PREFIXES = ("sk-ant-", "sk-proj-", "sk-", "")
keys = st.text(alphabet=string.ascii_letters + string.digits + "-_", min_size=1, max_size=200)
prefixed = st.tuples(st.sampled_from(_PREFIXES), keys).map(lambda pair: pair[0] + pair[1])


@given(prefixed)
def test_at_most_four_characters_of_the_key_show(key: str) -> None:
    shown = mask(key)
    prefix, marker, tail = shown.partition("…")
    assert marker == "…"
    assert prefix in _PREFIXES
    assert key.startswith(prefix)
    assert len(tail) <= 4
    assert key.endswith(tail)
    assert key not in shown


@given(prefixed)
def test_a_short_key_shows_less(key: str) -> None:
    tail = mask(key).partition("…")[2]
    if len(key) < 8:
        assert tail == ""
    elif len(key) < 16:
        assert len(tail) <= 2
        assert mask(key).startswith("…")


middles = st.text(alphabet=string.ascii_letters + string.digits, min_size=16, max_size=100)
tails = st.text(alphabet=string.ascii_letters + string.digits, min_size=4, max_size=4)


@given(st.sampled_from(_PREFIXES), middles, middles, tails)
def test_only_the_prefix_and_the_tail_show(prefix: str, one: str, two: str, tail: str) -> None:
    first, second = f"{prefix}0{one}{tail}", f"{prefix}0{two}{tail}"
    assert mask(first) == mask(second) == f"{prefix}…{tail}"
    assert mask(first) == mask(f"  {first}\n")
