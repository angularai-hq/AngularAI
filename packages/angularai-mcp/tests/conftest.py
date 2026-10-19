import os
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from angularai.store import init_cortex

settings.register_profile(
    "ci",
    max_examples=100,
    derandomize=True,
    database=None,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "dev",
    max_examples=25,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

# Names a test must never inherit. Values are removed, never read or printed.
_AMBIENT = (
    "ANGULARAI_CORTEX",
    "ANGULARAI_HOME",
    "ANGULARAI_EMBEDDINGS",
    "ANGULARAI_EMBEDDINGS_MODEL",
    "ANGULARAI_EMBEDDINGS_MIN",
    "ANGULARAI_MCP_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANGULARAI_API_KEY",
)


@pytest.fixture(autouse=True)
def no_ambient_cortex_or_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the operator's cortexes, keys, or token. A test sets what it needs."""

    for name in _AMBIENT:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_real_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(seconds: float) -> None:
        raise AssertionError(f"a test waited {seconds} real seconds for a retry")

    monkeypatch.setattr("angularai.provider._sleep", refuse)


@pytest.fixture(autouse=True)
def no_real_embeddings_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """An embeddings request that reaches the network fails the test."""

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a test tried to call a real embeddings API")

    monkeypatch.setattr("angularai.semantic.EmbeddingsAPI.embed", refuse)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir(mode=0o700)
    return path


@pytest.fixture
def cortex(home: Path) -> Iterator[Path]:
    root = home / "default"
    init_cortex(root)
    yield root
