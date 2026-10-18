"""Properties of the value lock: normalization, and variants of a locked value."""

import string

from hypothesis import example, given
from hypothesis import strategies as st

from angularai.lock import ValueLock, normalize

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"]
SPACES = [" ", "  ", "\t", "\n", "\r\n", "\u00a0", "\u2003", "\u3000", " \u200b "]
# Letters that look like Latin ones but are Cyrillic or Greek.
HOMOGLYPHS = {
    "a": "а",
    "c": "с",
    "e": "е",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
    "i": "і",
    "A": "Α",
    "B": "В",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
}


@given(st.text())
# Found while writing this property (random search did not reach it): dropping the
# joiner let the accent compose on a second pass.
@example("e\u200d\u0301")
@example("\u00a0\u0301")
def test_normalize_is_idempotent(text: str) -> None:
    once = normalize(text)
    assert normalize(once) == once


word = st.text(alphabet=string.ascii_letters, min_size=2, max_size=10)
locked = st.lists(word, min_size=1, max_size=4).map(" ".join)


def _fullwidth(ch: str) -> str:
    return chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch


def _bold(ch: str) -> str:
    if "A" <= ch <= "Z":
        return chr(0x1D400 + ord(ch) - ord("A"))
    if "a" <= ch <= "z":
        return chr(0x1D41A + ord(ch) - ord("a"))
    return ch


@st.composite
def variant(draw: st.DrawFn, value: str) -> str:
    out: list[str] = []
    for ch in value:
        if ch == " ":
            out.append(draw(st.sampled_from(SPACES)))
            continue
        kind = draw(st.sampled_from(["same", "upper", "lower", "full", "bold", "glyph"]))
        if kind == "upper":
            ch = ch.upper()
        elif kind == "lower":
            ch = ch.lower()
        elif kind == "full":
            ch = _fullwidth(ch)
        elif kind == "bold":
            ch = _bold(ch)
        elif kind == "glyph":
            ch = HOMOGLYPHS.get(ch, ch)
        out.append(ch)
        out.append(draw(st.sampled_from(["", "", *ZERO_WIDTH])))
    return "".join(out)


@given(st.data(), locked, st.text(max_size=40), st.text(max_size=40))
def test_every_variant_of_a_locked_value_is_held(
    data: st.DataObject, value: str, left: str, right: str
) -> None:
    disguised = data.draw(variant(value))
    reply = f"{left} {disguised} {right}"
    verdict = ValueLock().judge(reply, (value,), ())
    assert not verdict.allowed, repr(disguised)


# Found by Hypothesis: a Greek capital alpha passed for a Latin A.
@example("AA", "AΑ")
@example("secret plan", "ѕесrеt\u200b рlаn")
@example("Open Sesame", "ОРЕΝ\u00a0ЅЕЅΑΜЕ")
@given(st.just("never"), st.just("nеvеr"))
def test_found_lookalikes_are_held(value: str, disguised: str) -> None:
    assert not ValueLock().judge(disguised, (value,), ()).allowed


def test_a_cyrillic_phrase_still_matches_cyrillic() -> None:
    assert not ValueLock().judge("Это секрет.", ("секрет",), ()).allowed
    assert ValueLock().judge("An ordinary reply.", ("секрет",), ()).allowed
