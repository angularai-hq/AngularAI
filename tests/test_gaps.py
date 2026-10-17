"""Known gaps. Each test fails today and is marked strict, so a fix turns it red."""

import pytest

from angularai.budget import estimate_tokens

gap = pytest.mark.xfail(strict=True, reason="known gap, outside the secrets fix")


# R8: the membrane counts code points, so CJK and emoji are under-counted.
@gap
def test_the_membrane_does_not_undercount_cjk() -> None:
    assert estimate_tokens("你好世界" * 100) >= 300
