"""The token estimate: ASCII as before, and no undercount for CJK, Hangul, or emoji."""

import pytest

from angularai.budget import estimate_tokens


def test_ascii_is_four_characters_to_a_token() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("The quick brown fox jumps over the lazy dog.") == 11


# R8, once a strict xfail: 400 Han characters counted as 100 tokens.
def test_the_membrane_does_not_undercount_cjk() -> None:
    assert estimate_tokens("你好世界" * 100) == 400


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("今日はいい天気ですね。", 11),  # Han, Hiragana, an ideographic full stop
        ("カタカナ", 4),
        ("안녕하세요", 5),
        ("ＡＢＣ１２３", 6),  # fullwidth forms
        ("😀🎉", 4),  # above U+FFFF: two each
        ("👩\u200d💻", 5),  # two emoji and a joiner at half
        ("⌚", 1),  # a wide emoji inside the BMP
    ],
)
def test_wide_characters_count_one_token_or_two(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


def test_other_scripts_count_half_a_token() -> None:
    assert estimate_tokens("Привет") == 3
    assert estimate_tokens("αβγδ") == 2
    assert estimate_tokens("é") == 1


def test_mixed_text_adds_its_parts() -> None:
    # 8 ASCII (2 tokens), 2 Han (2), 1 emoji (2): quarters add, then round up once.
    assert estimate_tokens("Deploy 服务 🚀") == 6
    assert estimate_tokens("café") == 2  # three quarters and a half
