"""The bearer gate, as raw ASGI."""

from typing import Any

import pytest

from angularai_mcp.web import BearerGate, is_loopback, token_from

pytestmark = pytest.mark.anyio
TOKEN = "t0ken-for-tests-0123456789"


class Inner:
    def __init__(self) -> None:
        self.scopes: list[str] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.scopes.append(scope["type"])


async def call(gate: BearerGate, scope: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    async def send(message: Any) -> None:
        sent.append(message)

    await gate(scope, receive, send)
    return sent


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Bearer wrong-token-0123456789")],
        [(b"authorization", TOKEN.encode())],
        [(b"authorization", b"Basic " + TOKEN.encode())],
        [(b"authorization", b"Bearer " + TOKEN.encode() + b"x")],
    ],
)
async def test_a_request_without_the_token_gets_401(headers: list[tuple[bytes, bytes]]) -> None:
    inner = Inner()
    sent = await call(BearerGate(inner, TOKEN), {"type": "http", "headers": headers})
    assert inner.scopes == []
    assert sent[0]["status"] == 401
    assert TOKEN.encode() not in sent[1]["body"]


async def test_the_token_passes_and_lifespan_is_untouched() -> None:
    inner = Inner()
    gate = BearerGate(inner, TOKEN)
    good = [(b"Authorization", b"Bearer " + TOKEN.encode())]
    assert await call(gate, {"type": "http", "headers": good}) == []
    assert await call(gate, {"type": "lifespan"}) == []
    assert inner.scopes == ["http", "lifespan"]


def test_loopback() -> None:
    assert all(is_loopback(h) for h in ("127.0.0.1", "127.1.2.3", "::1", "[::1]", "localhost"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "::", "10.0.0.1", "example.com"))


def test_token_from() -> None:
    assert token_from({}) is None
    assert token_from({"ANGULARAI_MCP_TOKEN": f"  {TOKEN} "}) == TOKEN
