"""Known gaps. Each test fails today and is marked strict, so a fix turns it red."""

import pytest

from angularai.budget import estimate_tokens
from angularai.secrets import looks_secret

gap = pytest.mark.xfail(strict=True, reason="known gap, outside the sandbox fix")


# R6: token families the pattern list does not know.
@gap
@pytest.mark.parametrize(
    "text",
    [
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ",
        "gho_abcdefghijklmnopqrstuvwxyz0123456789",
        "sk_live_51Habcdefghijklmnop",
        "AIzaSyA-abcdefghijklmnopqrstuvwxyz12345",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnop",
        "glpat-abcdefghijklmnopqrst",
        "hf_abcdefghijklmnopqrstuvwxyzABCDEF",
        "password=hunter2hunter2",
    ],
)
def test_a_secret_is_seen(text: str) -> None:
    assert looks_secret(text)


# R7: ordinary words read as an sk- key.
@gap
@pytest.mark.parametrize(
    "text",
    [
        "Put it on the desk-organizer shelf",
        "ask-anything mode",
        "the risk-assessment doc",
    ],
)
def test_an_ordinary_word_is_not_a_secret(text: str) -> None:
    assert not looks_secret(text)


# R8: the membrane counts code points, so CJK and emoji are under-counted.
@gap
def test_the_membrane_does_not_undercount_cjk() -> None:
    assert estimate_tokens("你好世界" * 100) >= 300
