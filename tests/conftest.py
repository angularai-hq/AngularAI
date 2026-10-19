import os
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, settings

# HYPOTHESIS_PROFILE picks one. CI sets "ci": more examples, a fixed seed, no example
# database, so a failure reproduces from the log alone. "dev" is the default.
settings.register_profile(
    "ci",
    max_examples=200,
    derandomize=True,
    database=None,
    deadline=timedelta(seconds=2),
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=50, deadline=timedelta(seconds=1))
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(autouse=True)
def no_real_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry in a test must use an injected clock, never the real sleep."""

    def refuse(seconds: float) -> None:
        raise AssertionError(f"a test waited {seconds} real seconds for a retry")

    monkeypatch.setattr("angularai.provider._sleep", refuse)


@pytest.fixture(autouse=True)
def no_real_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recall by meaning is off in every test unless the test turns it on itself."""

    for name in ("ANGULARAI_EMBEDDINGS", "ANGULARAI_EMBEDDINGS_MODEL", "ANGULARAI_EMBEDDINGS_MIN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_ambient_cortex_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the operator's cortexes or keys. A test sets what it needs.

    Values are removed, never read or printed.
    """

    for name in (
        "ANGULARAI_CORTEX",
        "ANGULARAI_HOME",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANGULARAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
