"""Properties over the tools, through the in-process client."""

import tempfile
from pathlib import Path
from typing import Any

import anyio
from hypothesis import given
from hypothesis import strategies as st

from angularai.secrets import looks_secret
from angularai.store import init_cortex
from tests.helpers import connected

words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=10)
facts = st.lists(
    st.lists(words, min_size=1, max_size=8).map(" ".join),
    min_size=1,
    max_size=6,
    unique=True,
)


def run(body: Any) -> Any:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "default"
        init_cortex(root)
        return anyio.run(body, root)


@given(facts, st.integers(min_value=1, max_value=60), st.integers(min_value=1, max_value=50))
def test_recall_never_passes_its_caps(texts: list[str], max_tokens: int, limit: int) -> None:
    async def body(root: Path) -> None:
        async with connected(root) as client:
            for text in texts:
                await client.call_tool("remember", {"text": text})
            query = " ".join(texts)[:2000]
            result = await client.call_tool(
                "recall", {"query": query, "max_tokens": max_tokens, "limit": limit}
            )
        found = result.structured_content
        assert found is not None and not result.is_error
        assert found["used_tokens"] <= max_tokens and len(found["results"]) <= limit
        assert sum(hit["tokens"] for hit in found["results"]) == found["used_tokens"]
        assert found["matched"] == len(texts)

    run(body)


@given(facts, st.data())
def test_forget_removes_one_and_keeps_every_other(texts: list[str], data: st.DataObject) -> None:
    async def body(root: Path) -> None:
        async with connected(root) as client:
            ids = []
            for text in texts:
                result = await client.call_tool("remember", {"text": text})
                assert result.structured_content is not None
                ids.append(result.structured_content["trace"]["id"])
            target = data.draw(st.sampled_from(ids))
            gone = await client.call_tool("forget", {"id": target})
            again = await client.call_tool("forget", {"id": target})
            status = await client.call_tool("status", {})
        assert not gone.is_error and again.is_error
        assert status.structured_content is not None
        assert status.structured_content["traces_current"] == len(texts) - 1

    run(body)


@given(st.text(min_size=1, max_size=300))
def test_remember_either_keeps_the_text_or_writes_nothing(raw: str) -> None:
    async def body(root: Path) -> None:
        async with connected(root) as client:
            result = await client.call_tool("remember", {"text": raw})
        log = (root / "passes.jsonl").read_bytes()
        if result.is_error:
            assert log == b""
            text = getattr(result.content[0], "text", "")
            assert text.startswith("[")
            if len(raw.strip()) >= 8:
                assert raw.strip() not in text
        else:
            assert result.structured_content is not None
            assert result.structured_content["trace"]["text"] == " ".join(raw.split())
            assert not looks_secret(raw.strip())

    run(body)
