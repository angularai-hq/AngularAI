"""Every tool through an in-process MCP client."""

import json
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import TextResourceContents

from angularai.cli import main as angularai
from angularai.pilot import pass_turn
from angularai.semantic import VECTORS
from angularai.store import open_store
from angularai_mcp import __version__
from angularai_mcp.server import INSTRUCTIONS, NAME
from tests.fakes import FakeEmbedder
from tests.helpers import connected, failed, ok

pytestmark = pytest.mark.anyio


async def test_the_server_names_itself_and_lists_five_tools(cortex: Path) -> None:
    async with connected(cortex) as client:
        info = client.server_info
        assert info is not None and info.name == NAME == "AngularAI" and info.title == "AngularAI"
        assert info.version == __version__
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"remember", "recall", "forget", "status", "cortexes"}
    for tool in tools.values():
        assert tool.description and len(tool.description) > 60
        assert tool.output_schema is not None
        assert tool.annotations is not None
    assert tools["forget"].annotations is not None
    assert tools["forget"].annotations.destructive_hint is True
    assert (tools["forget"].meta or {}).get("anthropic/requiresUserInteraction") is True
    assert tools["recall"].annotations is not None
    assert tools["recall"].annotations.read_only_hint is True
    schema = tools["remember"].input_schema["properties"]
    assert schema["text"]["maxLength"] == 240 and schema["tags"]["maxItems"] == 4
    assert tools["forget"].input_schema["properties"]["id"]["pattern"] == "^t_[0-9a-f]{12}$"
    assert "secret" in INSTRUCTIONS


async def test_remember_writes_one_trace_with_provenance(cortex: Path) -> None:
    async with connected(cortex) as client:
        found = ok(
            await client.call_tool("remember", {"text": "The notebook is blue.", "tags": ["color"]})
        )
        again = ok(await client.call_tool("remember", {"text": "The notebook is blue."}))
    assert found["created"] is True and again["created"] is False
    trace = found["trace"]
    assert trace["text"] == "The notebook is blue." and trace["tags"] == ["color"]
    assert trace["source"] == "mcp" and trace["state"] == "current"
    assert again["trace"]["id"] == trace["id"]
    assert [t.id for t in open_store(cortex).view().traces] == [trace["id"]]


async def test_recall_ranks_and_reports_provenance(cortex: Path) -> None:
    async with connected(cortex) as client:
        await client.call_tool("remember", {"text": "The notebook is blue.", "tags": ["notebook"]})
        await client.call_tool("remember", {"text": "Catering is at noon."})
        result = ok(await client.call_tool("recall", {"query": "What color is the notebook?"}))
        none = ok(await client.call_tool("recall", {"query": "train station"}))
    assert result["mode"] == "lexical" and result["matched"] == 1
    (hit,) = result["results"]
    assert hit["trace"]["text"] == "The notebook is blue." and hit["match"] == "lexical"
    assert 0 < hit["score"] <= 1 and hit["tokens"] == 6
    assert result["used_tokens"] == 6 and result["max_tokens"] == result["workspace_budget"] == 2000
    assert none["results"] == [] and none["matched"] == 0


async def test_recall_stays_inside_max_tokens_and_limit(cortex: Path) -> None:
    async with connected(cortex) as client:
        for n in range(6):
            await client.call_tool("remember", {"text": f"alpha fact {n} " + "x" * 40})
        capped = ok(await client.call_tool("recall", {"query": "alpha", "max_tokens": 30}))
        limited = ok(await client.call_tool("recall", {"query": "alpha", "limit": 2}))
        over = await client.call_tool("recall", {"query": "alpha", "max_tokens": 999999})
        zero = await client.call_tool("recall", {"query": "alpha", "limit": 0})
    assert capped["used_tokens"] <= 30 and len(capped["results"]) == 2 and capped["matched"] == 6
    assert len(limited["results"]) == 2
    failed(over, "invalid_input")
    failed(zero, "invalid_input")


async def test_recall_by_meaning_with_a_fake_embedder(
    cortex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEmbedder()
    monkeypatch.setattr("angularai_mcp.server.embedder_from_env", lambda config, environ: fake)
    env = {"ANGULARAI_HOME": str(cortex.parent), "ANGULARAI_EMBEDDINGS": "openai"}
    async with connected(cortex, env) as client:
        ok(await client.call_tool("remember", {"text": "Catering is at noon."}))
        by_meaning = ok(await client.call_tool("recall", {"query": "When is lunch?"}))
        lexical = ok(
            await client.call_tool("recall", {"query": "When is lunch?", "lexical_only": True})
        )
        status = ok(await client.call_tool("status", {}))
    assert by_meaning["mode"] == "lexical+meaning"
    assert [h["match"] for h in by_meaning["results"]] == ["meaning"]
    assert lexical["mode"] == "lexical" and lexical["results"] == []
    assert fake.sent[0] == ["Catering is at noon."]
    assert status["embeddings"] == {
        "mode": "openai",
        "model": "text-embedding-3-small",
        "minimum": 0.35,
        "vectors_file": True,
    }


async def test_recall_says_why_meaning_is_off(
    cortex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {"ANGULARAI_HOME": str(cortex.parent), "ANGULARAI_EMBEDDINGS": "openai"}
    async with connected(cortex, env) as client:
        missing = ok(await client.call_tool("recall", {"query": "lunch"}))
    assert missing["mode"] == "lexical" and "OPENAI_API_KEY" in (missing["note"] or "")
    broken = FakeEmbedder(fail=OSError("down"))
    monkeypatch.setattr("angularai_mcp.server.embedder_from_env", lambda config, environ: broken)
    async with connected(cortex, env) as client:
        failing = ok(await client.call_tool("recall", {"query": "lunch"}))
    assert failing["mode"] == "lexical" and "unavailable" in (failing["note"] or "")


async def test_recall_can_show_superseded_traces(cortex: Path) -> None:
    from angularai.models import Integration, TraceDraft
    from tests.helpers import connected as again

    store = open_store(cortex)

    class One:
        def __init__(self, reply: Integration) -> None:
            self.reply = reply

        def integrate(self, workspace: object) -> Integration:
            return self.reply

    first = Integration("Noted.", (TraceDraft("The launch is Friday.", ()),), ())
    pass_turn(store, One(first), "launch Friday")
    old = store.view().traces[0].id
    second = Integration("Noted.", (TraceDraft("The launch is Monday.", ()),), (old,))
    pass_turn(store, One(second), "the launch moved")
    async with again(cortex) as client:
        current = ok(await client.call_tool("recall", {"query": "launch"}))
        every = ok(
            await client.call_tool("recall", {"query": "launch", "include_superseded": True})
        )
    assert old not in {h["trace"]["id"] for h in current["results"]}
    hits = {h["trace"]["id"]: h["trace"] for h in every["results"]}
    assert hits[old]["state"] == "superseded" and hits[old]["superseded_by"]


async def test_forget_removes_exactly_one(cortex: Path) -> None:
    (cortex / VECTORS).write_text("")
    async with connected(cortex) as client:
        keep = ok(await client.call_tool("remember", {"text": "Catering is at noon."}))
        gone = ok(await client.call_tool("remember", {"text": "The notebook is blue."}))
        (cortex / VECTORS).write_text(json.dumps({"v": 1, "trace": gone["trace"]["id"]}) + "\n")
        result = ok(await client.call_tool("forget", {"id": gone["trace"]["id"]}))
        twice = await client.call_tool("forget", {"id": gone["trace"]["id"]})
        left = ok(await client.call_tool("recall", {"query": "notebook catering"}))
    assert result["forgotten"]["text"] == "The notebook is blue." and result["vectors_removed"] == 1
    failed(twice, "not_found")
    assert [h["trace"]["id"] for h in left["results"]] == [keep["trace"]["id"]]
    assert "notebook" not in (cortex / "passes.jsonl").read_text()


async def test_status_counts_and_budget(cortex: Path) -> None:
    assert angularai(["--cortex", str(cortex), "lock", "add", "falcon"]) == 0
    async with connected(cortex) as client:
        empty = ok(await client.call_tool("status", {}))
        await client.call_tool("remember", {"text": "The notebook is blue."})
        status = ok(await client.call_tool("status", {}))
    assert empty["last_pass"] is None and empty["traces_current"] == 0
    assert status["cortex"] == "default" and status["path"] == str(cortex)
    assert status["traces_current"] == 1 and status["episodes"] == 1
    assert status["memory_tokens"] == 6 and status["memory_fits_budget"] is True
    assert status["locks"] == 1 and "falcon" not in json.dumps(status)
    assert status["last_pass"]["outcome"] == "ok"
    assert status["embeddings"]["mode"] == "off"
    assert status["damaged_lines"] == 0 and status["substrate"] is None


async def test_status_reports_a_bad_embeddings_setting(cortex: Path) -> None:
    env = {
        "ANGULARAI_HOME": str(cortex.parent),
        "ANGULARAI_EMBEDDINGS": "nope",
    }
    async with connected(cortex, env) as client:
        status = ok(await client.call_tool("status", {}))
        recall = await client.call_tool("recall", {"query": "x"})
    assert status["embeddings"]["mode"] == "invalid"
    failed(recall, "config_error")
    env["ANGULARAI_EMBEDDINGS"] = "compatible"
    env["ANGULARAI_EMBEDDINGS_MIN"] = "7"
    async with connected(cortex, env) as client:
        status = ok(await client.call_tool("status", {}))
    assert status["embeddings"]["mode"] == "compatible" and status["embeddings"]["minimum"] is None


async def test_cortexes_lists_names_and_the_active_one(home: Path, cortex: Path) -> None:
    assert angularai(["--cortex", str(home / "bo"), "init"]) == 0
    (home / "Not A Name").mkdir()
    async with connected(home / "bo", {"ANGULARAI_HOME": str(home)}) as client:
        result = ok(await client.call_tool("cortexes", {}))
    assert result == {"active": "bo", "home": str(home), "names": ["bo", "default"]}


async def test_resources_status_and_trace(cortex: Path) -> None:
    async with connected(cortex) as client:
        trace = ok(await client.call_tool("remember", {"text": "The notebook is blue."}))["trace"]
        listed = await client.list_resources()
        templates = await client.list_resource_templates()
        status = await client.read_resource("angularai://status")
        one = await client.read_resource(f"angularai://trace/{trace['id']}")
        with pytest.raises(MCPError, match=r"\[not_found\]"):
            await client.read_resource("angularai://trace/t_000000000000")
        with pytest.raises(MCPError, match=r"\[not_found\]"):
            await client.read_resource("angularai://trace/nope")
    assert [str(r.uri) for r in listed.resources] == ["angularai://status"]
    assert [t.uri_template for t in templates.resource_templates] == [
        "angularai://trace/{trace_id}"
    ]
    (content,) = status.contents
    assert isinstance(content, TextResourceContents)
    assert json.loads(content.text)["traces_current"] == 1
    (body,) = one.contents
    assert isinstance(body, TextResourceContents) and json.loads(body.text) == trace


async def test_a_missing_cortex_is_a_typed_error(home: Path) -> None:
    async with connected(home / "default") as client:
        for name, args in (
            ("status", {}),
            ("remember", {"text": "x y"}),
            ("recall", {"query": "x"}),
            ("forget", {"id": "t_000000000000"}),
        ):
            message = failed(await client.call_tool(name, args), "no_cortex")
            assert "--init" in message
        with pytest.raises(MCPError, match=r"\[no_cortex\]"):
            await client.read_resource("angularai://status")
        with pytest.raises(MCPError, match=r"\[no_cortex\]"):
            await client.read_resource("angularai://trace/t_000000000000")
