import os
from collections.abc import Iterator
from datetime import timedelta

import keyring
import pytest
from hypothesis import HealthCheck, settings

from tests.setupfakes import FakeKeyring

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

    for name in (
        "ANGULARAI_EMBEDDINGS",
        "ANGULARAI_EMBEDDINGS_MODEL",
        "ANGULARAI_EMBEDDINGS_MIN",
        "ANGULARAI_EMBEDDINGS_URL",
        "ANGULARAI_INDEX",
    ):
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
        "ANGULARAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def fake_keyring_and_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[FakeKeyring]:
    """No test reaches the OS keyring or the real ~/.angularai.

    keyring gets an in-memory backend, and HOME is a fresh directory, so
    ``Path.home() / ".angularai"`` is never the operator's.
    """

    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.delenv("PYTHON_KEYRING_BACKEND", raising=False)
    previous = keyring.core._keyring_backend
    fake = FakeKeyring()
    keyring.set_keyring(fake)
    try:
        yield fake
    finally:
        keyring.core._keyring_backend = previous


@pytest.fixture(autouse=True)
def no_first_use_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that fakes a terminal never meets the setup offer unless it asks for it."""

    monkeypatch.setattr("angularai.cli.offer_setup", lambda root: None)


@pytest.fixture(autouse=True)
def no_boot_animation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boot mark never plays into a test's terminal. Its own tests pass an env."""

    monkeypatch.setenv("ANGULARAI_NO_BOOT", "1")
