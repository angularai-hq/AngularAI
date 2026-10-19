"""What a hostile or careless client sends. Nothing it sends comes back as a secret."""

import json
import logging
from pathlib import Path

import anyio
import pytest
from mcp import Client

from angularai.store import open_store
from angularai_mcp.server import build_server
from tests.helpers import connected, failed, ok, text

pytestmark = pytest.mark.anyio

KEY = "sk-proj-" + "Ab3dE5gH7jK9mN1pQ3sT5vX7z"
ENV_SECRET = "zz9-plural-z-alpha"


def env_with_secrets(cortex: Path) -> dict[str, str]:
    return {
        "ANGULARAI_HOME": str(cortex.parent),
        "OPENAI_API_KEY": KEY,
        "MY_SERVICE_TOKEN": ENV_SECRET,
    }


@pytest.mark.parametrize(
    "args",
    [
        {"text": f"my key is {KEY}"},
        {"text": "password=hunter2hunter2"},
        {"text": "-----BEGIN RSA PRIVATE KEY-----"},
        {"text": "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2"},
        {"text": f"the service uses {ENV_SECRET} today"},
    ],
)
async def test_a_secret_is_refused_and_never_echoed(
    cortex: Path, args: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    async with connected(cortex, env_with_secrets(cortex)) as client:
        result = await client.call_tool("remember", args)
    message = failed(result, "secret_refused")
    assert KEY not in message and ENV_SECRET not in message and "hunter2" not in message
    assert KEY not in caplog.text and ENV_SECRET not in caplog.text
    assert (cortex / "passes.jsonl").read_bytes() == b""


async def test_a_secret_in_a_tag_or_a_query_is_refused(cortex: Path) -> None:
    async with connected(cortex, env_with_secrets(cortex)) as client:
        tag = await client.call_tool("remember", {"text": "fine", "tags": [KEY]})
        query = await client.call_tool("recall", {"query": f"find {KEY}"})
    assert KEY not in text(tag) and KEY not in text(query)
    failed(tag, "invalid_input")
    failed(query, "secret_refused")
    assert (cortex / "passes.jsonl").read_bytes() == b""


async def test_a_schema_mismatch_names_fields_not_values(cortex: Path) -> None:
    async with connected(cortex) as client:
        wrong = await client.call_tool("remember", {"text": [KEY], "tags": KEY})
        extra = await client.call_tool("recall", {"query": {"nested": KEY}})
        unknown = await client.call_tool("run", {"command": "rm -rf /"})
        edit = await client.call_tool("edit", {"path": "x"})
    for result in (wrong, extra):
        message = failed(result, "invalid_input")
        assert KEY not in message
    assert "tags" in text(wrong) and "text" in text(wrong)
    assert unknown.is_error and edit.is_error


async def test_tool_output_never_holds_an_environment_value(cortex: Path) -> None:
    async with connected(cortex, env_with_secrets(cortex)) as client:
        seen = [
            await client.call_tool("remember", {"text": "The notebook is blue."}),
            await client.call_tool("recall", {"query": "notebook"}),
            await client.call_tool("status", {}),
            await client.call_tool("cortexes", {}),
            await client.call_tool("forget", {"id": "t_000000000000"}),
        ]
        resource = await client.read_resource("angularai://status")
    dumped = json.dumps([r.model_dump(mode="json") for r in seen]) + resource.model_dump_json()
    assert KEY not in dumped and ENV_SECRET not in dumped
    assert "OPENAI_API_KEY" not in dumped and "MY_SERVICE_TOKEN" not in dumped


async def test_an_old_trace_holding_an_environment_value_is_redacted_on_the_way_out(
    cortex: Path,
) -> None:
    async with connected(cortex) as client:
        ok(await client.call_tool("remember", {"text": f"The vault phrase is {ENV_SECRET}."}))
    async with connected(cortex, env_with_secrets(cortex)) as client:
        found = ok(await client.call_tool("recall", {"query": "vault phrase"}))
    assert ENV_SECRET not in json.dumps(found)
    assert "[redacted]" in found["results"][0]["trace"]["text"]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"text": "x" * 241}, "too_long"),
        ({"text": "word " * 200_000}, "too_long"),
        ({"text": "   "}, "invalid_input"),
        ({"text": "fine", "tags": ["a", "b", "c", "d", "e"]}, "invalid_input"),
        ({"text": "fine", "tags": ["x" * 5000]}, "invalid_input"),
        ({"text": "fine", "tags": ["Upper Case"]}, "invalid_input"),
    ],
)
async def test_oversized_or_malformed_notes_write_nothing(
    cortex: Path, args: dict[str, object], code: str
) -> None:
    async with connected(cortex) as client:
        failed(await client.call_tool("remember", args), code)
    assert (cortex / "passes.jsonl").read_bytes() == b""


async def test_oversized_queries_are_refused(cortex: Path) -> None:
    async with connected(cortex) as client:
        failed(await client.call_tool("recall", {"query": "q" * 2001}), "too_long")
        failed(await client.call_tool("recall", {"query": "  "}), "invalid_input")
        failed(await client.call_tool("recall", {"query": "q", "limit": 51}), "invalid_input")
        failed(await client.call_tool("recall", {"query": "q", "max_tokens": 0}), "invalid_input")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "t_",
        "t_ABCDEF012345",
        "t_0123456789a",
        "t_0123456789abc",
        "../../etc/passwd",
        "t_0123456789ab; rm -rf /",
        "t_0123456789ab\n",
        "*",
        "t_" + "0" * 10_000,
    ],
)
async def test_forget_with_a_bad_id_removes_nothing(cortex: Path, bad: str) -> None:
    async with connected(cortex) as client:
        ok(await client.call_tool("remember", {"text": "The notebook is blue."}))
        before = (cortex / "passes.jsonl").read_bytes()
        failed(await client.call_tool("forget", {"id": bad}), "not_found")
    assert (cortex / "passes.jsonl").read_bytes() == before


async def test_forget_takes_no_prefix_of_a_real_id(cortex: Path) -> None:
    async with connected(cortex) as client:
        real = ok(await client.call_tool("remember", {"text": "The notebook is blue."}))
        prefix = real["trace"]["id"][:-1]
        failed(await client.call_tool("forget", {"id": prefix}), "not_found")
        failed(await client.call_tool("forget", {"id": prefix + "0" * 2}), "not_found")
        failed(await client.call_tool("forget", {"id": real["trace"]["id"].upper()}), "not_found")
    assert len(open_store(cortex).view().traces) == 1


async def test_value_lock_holds_a_note(cortex: Path) -> None:
    from angularai.cli import main

    assert main(["--cortex", str(cortex), "lock", "add", "project falcon"]) == 0
    async with connected(cortex) as client:
        message = failed(
            await client.call_tool("remember", {"text": "Project  FALCON ships Friday."}), "locked"
        )
    assert "falcon" not in message.lower()
    assert open_store(cortex).passes() == ()


async def test_concurrent_writes_from_many_calls_and_two_servers(cortex: Path) -> None:
    env = {"ANGULARAI_HOME": str(cortex.parent)}
    async with Client(build_server(cortex, env)) as one, Client(build_server(cortex, env)) as two:
        async with anyio.create_task_group() as group:
            for n in range(40):
                client = one if n % 2 else two
                group.start_soon(client.call_tool, "remember", {"text": f"Fact {n} is kept."})
    store = open_store(cortex)
    texts = sorted(trace.text for trace in store.view().current_traces())
    assert texts == sorted(f"Fact {n} is kept." for n in range(40))
    assert store.damage() == ()
    for line in (cortex / "passes.jsonl").read_text().splitlines():
        json.loads(line)


async def test_concurrent_forget_and_remember(cortex: Path) -> None:
    env = {"ANGULARAI_HOME": str(cortex.parent)}
    async with Client(build_server(cortex, env)) as one, Client(build_server(cortex, env)) as two:
        ids = [
            ok(await one.call_tool("remember", {"text": f"Old fact {n}."}))["trace"]["id"]
            for n in range(10)
        ]
        async with anyio.create_task_group() as group:
            for n, trace_id in enumerate(ids):
                group.start_soon(one.call_tool, "forget", {"id": trace_id})
                group.start_soon(two.call_tool, "remember", {"text": f"New fact {n}."})
    texts = sorted(trace.text for trace in open_store(cortex).view().traces)
    assert texts == sorted(f"New fact {n}." for n in range(10))


async def test_an_unexpected_failure_shows_no_details(
    cortex: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError(f"internal detail {KEY}")

    def disk(*args: object, **kwargs: object) -> object:
        raise PermissionError(13, "denied", str(cortex))

    monkeypatch.setattr("angularai_mcp.server.note", explode)
    monkeypatch.setattr("angularai_mcp.server.recall_within", disk)
    async with connected(cortex) as client:
        crash = await client.call_tool("remember", {"text": "fine"})
        denied = await client.call_tool("recall", {"query": "fine"})
    message = failed(crash, "internal")
    assert KEY not in message and "internal detail" not in message
    assert KEY not in caplog.text and "Traceback" not in caplog.text
    assert "RuntimeError" in caplog.text
    failed(denied, "storage_error")


async def test_a_damaged_config_is_a_typed_error(cortex: Path) -> None:
    (cortex / "cortex.json").write_text("{not json")
    async with connected(cortex) as client:
        failed(await client.call_tool("status", {}), "config_error")


async def test_damaged_log_lines_are_counted_and_skipped(cortex: Path) -> None:
    async with connected(cortex) as client:
        ok(await client.call_tool("remember", {"text": "The notebook is blue."}))
        with (cortex / "passes.jsonl").open("a") as handle:
            handle.write("garbage\n")
        status = ok(await client.call_tool("status", {}))
        found = ok(await client.call_tool("recall", {"query": "notebook"}))
    assert status["damaged_lines"] == 1 and len(found["results"]) == 1
