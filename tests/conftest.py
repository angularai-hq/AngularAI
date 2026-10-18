import pytest


@pytest.fixture(autouse=True)
def no_real_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry in a test must use an injected clock, never the real sleep."""

    def refuse(seconds: float) -> None:
        raise AssertionError(f"a test waited {seconds} real seconds for a retry")

    monkeypatch.setattr("angularai.provider._sleep", refuse)
