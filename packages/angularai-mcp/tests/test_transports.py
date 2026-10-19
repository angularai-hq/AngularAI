"""The server as a client sees it: a child process over stdio, and over streamable HTTP."""

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from tests.helpers import child_env, failed, ok

pytestmark = pytest.mark.anyio
TOKEN = "t0ken-for-tests-0123456789"
KEY = "sk-proj-" + "Ab3dE5gH7jK9mN1pQ3sT5vX7z"


def stdio(home: Path, *args: str, **env: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "angularai_mcp", *args],
        env=child_env(home, **env),
        cwd=str(home),
    )


async def test_stdio_round_trip(home: Path, cortex: Path) -> None:
    async with Client(stdio(home, "--cortex", str(cortex), OPENAI_API_KEY=KEY)) as client:
        assert client.server_info is not None and client.server_info.name == "AngularAI"
        names = {tool.name for tool in (await client.list_tools()).tools}
        noted = ok(await client.call_tool("remember", {"text": "The notebook is blue."}))
        found = ok(await client.call_tool("recall", {"query": "notebook"}))
        refused = await client.call_tool("remember", {"text": f"key {KEY}"})
        status = ok(await client.call_tool("status", {}))
        gone = ok(await client.call_tool("forget", {"id": noted["trace"]["id"]}))
    assert names == {"remember", "recall", "forget", "status", "cortexes"}
    assert found["results"][0]["trace"]["id"] == noted["trace"]["id"]
    assert KEY not in failed(refused, "secret_refused")
    assert status["traces_current"] == 1
    assert gone["forgotten"]["id"] == noted["trace"]["id"]


async def test_stdio_with_a_name_and_init(home: Path) -> None:
    async with Client(stdio(home, "--name", "bo", "--init")) as client:
        status = ok(await client.call_tool("status", {}))
        listed = ok(await client.call_tool("cortexes", {}))
    assert status["cortex"] == "bo" and listed["names"] == ["bo"]
    assert (home / "bo" / "cortex.json").is_file()


def test_a_bad_name_exits_before_serving(home: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "angularai_mcp", "--name", "../evil", "--init"],
        env=child_env(home),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 2 and done.stdout == ""
    assert "cortex name" in done.stderr and not (home.parent / "evil").exists()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@contextmanager
def http_server(home: Path, cortex: Path, **env: str) -> Iterator[str]:
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "angularai_mcp", "--cortex", str(cortex), "--http"]
        + ["--port", str(port)],
        env=child_env(home, **env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError("the HTTP server did not start") from None
                time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            out, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            out, _ = process.communicate()
        assert out == b""


async def test_http_round_trip(home: Path, cortex: Path) -> None:
    with http_server(home, cortex) as url:
        async with Client(url) as client:
            noted = ok(await client.call_tool("remember", {"text": "Catering is at noon."}))
            found = ok(await client.call_tool("recall", {"query": "catering"}))
    assert found["results"][0]["trace"]["id"] == noted["trace"]["id"]


def post(url: str, headers: dict[str, str]) -> int:
    body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in {"Content-Type": "application/json", **headers}.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as answer:
            status: int = answer.status
            return status
    except urllib.error.HTTPError as exc:
        return exc.code


async def test_http_with_a_token(home: Path, cortex: Path) -> None:
    with http_server(home, cortex, ANGULARAI_MCP_TOKEN=TOKEN) as url:
        assert post(url, {}) == 401
        assert post(url, {"Authorization": "Bearer wrong-wrong-wrong-wrong"}) == 401
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {TOKEN}"})
        async with http, Client(streamable_http_client(url, http_client=http)) as client:
            status = ok(await client.call_tool("status", {}))
    assert status["cortex"] == "default"


async def test_http_refuses_a_foreign_host_header(home: Path, cortex: Path) -> None:
    with http_server(home, cortex) as url:
        assert post(url, {"Host": "evil.example:80"}) == 421
