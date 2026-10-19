import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.types import CallToolResult

from angularai_mcp.server import build_server


@asynccontextmanager
async def connected(root: Path, environ: Mapping[str, str] | None = None) -> AsyncIterator[Client]:
    env = {"ANGULARAI_HOME": str(root.parent)} if environ is None else dict(environ)
    async with Client(build_server(root, env)) as client:
        yield client


def ok(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, text(result)
    content: dict[str, Any] | None = result.structured_content
    assert content is not None
    return content


def text(result: CallToolResult) -> str:
    parts = [getattr(block, "text", "") for block in result.content]
    return "\n".join(parts)


def failed(result: CallToolResult, code: str) -> str:
    assert result.is_error, result.structured_content
    message = text(result)
    assert message.startswith(f"[{code}] "), message
    assert "Traceback" not in message
    return message


def child_env(home: Path, **extra: str) -> dict[str, str]:
    """The whole environment of a spawned server: no inherited key can reach it."""

    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home.parent)}
    env["ANGULARAI_HOME"] = str(home)
    env.update(extra)
    return env
