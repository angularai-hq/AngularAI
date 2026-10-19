import re
from functools import lru_cache
from unicodedata import east_asian_width

_OUTSIDE_ASCII = re.compile(r"[^\x00-\x7f]+")


def estimate_tokens(text: str) -> int:
    """Membrane size, counted in quarter tokens and rounded up.

    An ASCII character is a quarter token: four to one, as before.
    A wide character (Han, Hiragana, Katakana, Hangul, fullwidth forms, and most emoji,
    by Unicode East Asian Width W or F) is one token, and two above U+FFFF, where
    emoji live.
    Any other character outside ASCII (an accented letter, Cyrillic, Greek, Arabic,
    a combining mark, a joiner) is half a token.

    This is the budget AngularAI enforces. It is not the provider's tokenizer.
    It leans high for non-Latin text, so staying under it keeps every call inside
    a fixed window. ASCII text counts exactly as it did.
    """

    if not text:
        return 0
    quarters = len(text)
    if not text.isascii():
        quarters += sum(map(_extra, "".join(_OUTSIDE_ASCII.findall(text))))
    return (quarters + 3) // 4


@lru_cache(maxsize=8192)
def _extra(char: str) -> int:
    """Quarter tokens a non-ASCII character adds over the one every character has."""

    if east_asian_width(char) not in ("W", "F"):
        return 1
    return 7 if ord(char) > 0xFFFF else 3
