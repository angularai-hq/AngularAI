"""Streamable HTTP: loopback by default, an optional bearer token.

The token is ANGULARAI_MCP_TOKEN. When it is set, every HTTP request needs
``Authorization: Bearer <token>``; anything else gets 401 and reaches no tool.
It is compared in constant time and never logged. A host other than loopback
needs --allow-remote and the token.
"""

import hmac
import ipaddress
from collections.abc import Mapping

from starlette.types import ASGIApp, Receive, Scope, Send

TOKEN_ENV = "ANGULARAI_MCP_TOKEN"
MIN_TOKEN = 16
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PATH = "/mcp"


class TokenProblem(ValueError):
    """The token setting cannot be used. The message never holds the token."""


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def token_from(environ: Mapping[str, str]) -> str | None:
    raw = environ.get(TOKEN_ENV)
    if raw is None or not raw.strip():
        return None
    token = raw.strip()
    if len(token) < MIN_TOKEN or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in token):
        raise TokenProblem(
            f"{TOKEN_ENV} must be at least {MIN_TOKEN} printable ASCII characters, no spaces."
        )
    return token


class BearerGate:
    """ASGI middleware: an HTTP request without the token is answered 401."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        given = b""
        for key, value in scope.get("headers", ()):
            if key.lower() == b"authorization":
                given = value
                break
        if hmac.compare_digest(given, self._expected):
            await self._app(scope, receive, send)
            return
        body = b'{"error":"unauthorized","message":"A bearer token is required."}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"www-authenticate", b'Bearer realm="angularai-mcp"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
