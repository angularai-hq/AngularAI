import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from angularai.budget import estimate_tokens
from angularai.errors import ConfigError, InputDoesNotFit, SubstrateUnavailable
from angularai.index import (
    INDEX,
    CortexIndex,
    IndexUnavailable,
    assemble_indexed,
    chunks_of,
    fts_query,
    index_from_env,
    task_prefixes,
)
from angularai.models import Integration, PassRecord, Peer, TraceDraft, Workspace
from angularai.pilot import pass_turn
from angularai.semantic import EmbeddingsAPI, embedder_from_env
from angularai.store import Store, default_config, init_cortex, open_store
from tests.scripted import Scripted, noted, said
from tests.test_semantic import Fake


def episode(at: str, user: str, reply: str | None = "ok", peer: Peer | None = None) -> PassRecord:
    return PassRecord(
        at=at,
        outcome="ok",
        user=user,
        reply=reply,
        remember=(),
        supersede=(),
        error=None,
        budget=0,
        used=0,
        traces_shown=0,
        traces_held=0,
        episodes_shown=0,
        episodes_held=0,
        peer=peer,
    )


def cortex(root: Path, *records: PassRecord, budget: int = 2000) -> Store:
    init_cortex(root)
    store = open_store(root)
    from dataclasses import replace

    store.save_config(
        replace(default_config(), workspace_budget=budget, max_input_tokens=min(800, budget))
    )
    for record in records:
        store.append_pass(record)
    return store


def test_chunks_of_packs_whole_sentences() -> None:
    assert chunks_of("") == []
    assert chunks_of("One. Two.  Three!") == ["One. Two. Three!"]
    pieces = chunks_of("Alpha beta. " * 50, limit=60)
    assert all(len(p) <= 60 for p in pieces) and len(pieces) > 5
    long = chunks_of("x" * 130 + ". Tail.", limit=50)
    assert long == ["x" * 50, "x" * 50, "x" * 30 + ". Tail."]
    assert chunks_of("a\n\nb") == ["a b"]


@given(st.text(min_size=0, max_size=600), st.integers(min_value=10, max_value=200))
def test_chunks_never_pass_the_limit_and_keep_the_words(text: str, limit: int) -> None:
    pieces = chunks_of(text, limit)
    assert all(0 < len(p) <= limit for p in pieces)
    assert "".join("".join(p.split()) for p in pieces) == "".join(text.split())


def test_fts_query_quotes_words() -> None:
    assert fts_query('What "color" OR NOT notebook*?') == '"color" OR "notebook"'
    assert fts_query("the of and") == ""


def test_sync_indexes_a_v1_log_and_finds_by_bm25(tmp_path: Path) -> None:
    store = cortex(
        tmp_path / "c",
        episode("2023-05-01T10:00:00", "My notebook is blue.", "Noted."),
        episode("2023-05-02T10:00:00", "Lunch was pasta.", "Sounds good."),
    )
    index = CortexIndex(store.root)
    assert index.sync(store) == 4
    assert index.sync(store) == 0
    assert (store.root / INDEX).stat().st_mode & 0o777 == 0o600
    hits = index.search(store.view(), "What color is my notebook?")
    assert hits[0].chunk.text == "My notebook is blue."
    assert hits[0].chunk.speaker == "you" and hits[0].chunk.at.startswith("2023-05-01")
    assert index.search(store.view(), "the") == []
    store.append_pass(episode("2023-05-03T10:00:00", "The notebook is red now."))
    assert index.sync(store) == 2
    assert index.chunks() == 6
    assert index.newest() is not None and index.newest().text == "The notebook is red now."  # type: ignore[union-attr]
    index.close()


def test_a_rewritten_log_is_indexed_again(tmp_path: Path) -> None:
    root = tmp_path / "c"
    init_cortex(root)
    store = open_store(root)
    pass_turn(store, Scripted(noted("The notebook is blue.")), "Note: the notebook is blue.")
    index = CortexIndex(root)
    index.sync(store)
    (trace,) = store.view().current_traces()
    assert any(h.chunk.trace_id == trace.id for h in index.search(store.view(), "notebook blue"))
    store.forget(trace.id)
    index.sync(store)
    assert all(h.chunk.kind != "trace" for h in index.search(store.view(), "notebook blue"))
    index.close()


def test_a_superseded_trace_is_not_returned(tmp_path: Path) -> None:
    root = tmp_path / "c"
    init_cortex(root)
    store = open_store(root)
    pass_turn(store, Scripted(noted("The notebook is blue.")), "Note: the notebook is blue.")
    (old,) = store.view().current_traces()
    swap = Integration("Noted.", (TraceDraft("The notebook is red.", ()),), (old.id,))
    index = CortexIndex(root)
    pass_turn(store, Scripted(swap), "The notebook is red now, not blue.", index=index)
    ids = {h.chunk.trace_id for h in index.search(store.view(), "notebook blue red")}
    assert old.id not in ids
    assert any(t.text == "The notebook is red." for t in store.view().current_traces())
    index.close()


def test_vectors_rank_by_meaning_and_fuse(tmp_path: Path) -> None:
    store = cortex(
        tmp_path / "c",
        episode("2023-05-01T10:00:00", "My journal has a red cover.", None),
        episode("2023-05-02T10:00:00", "Catering is at noon.", None),
    )
    fake = Fake()
    index = CortexIndex(store.root, fake, document_prefix="doc: ", query_prefix="q: ")
    index.sync(store)
    assert fake.sent[0] == ["doc: My journal has a red cover.", "doc: Catering is at noon."]
    hits = index.search(store.view(), "What shade is the notebook?")
    assert hits[0].chunk.text == "My journal has a red cover."
    assert fake.sent[-1] == ["q: What shade is the notebook?"]
    index.close()
    # Another model means other vectors: the index is rebuilt.
    other = CortexIndex(store.root, Fake(model="fake-2"))
    assert other.chunks() == 0
    assert other.sync(store) == 2
    other.close()


def test_a_failed_embedding_never_stops_and_is_retried(tmp_path: Path) -> None:
    store = cortex(tmp_path / "c", episode("2023-05-01T10:00:00", "Blue notebook.", None))
    notices: list[str] = []
    down = Fake(fail=SubstrateUnavailable("down"))
    index = CortexIndex(store.root, down, warn=notices.append)
    index.sync(store)
    assert index.search(store.view(), "notebook")[0].chunk.text == "Blue notebook."
    assert len(notices) == 1 and "lexical" in notices[0]
    index.close()
    healed = Fake()
    again = CortexIndex(store.root, healed)
    again.sync(store)
    assert healed.sent == [["Blue notebook."]]
    vectors = sqlite3.connect(store.root / INDEX).execute("SELECT count(vector) FROM chunks")
    assert vectors.fetchone()[0] == 1
    again.close()


def test_a_vector_of_another_size_is_skipped(tmp_path: Path) -> None:
    store = cortex(tmp_path / "c", episode("2023-05-01T10:00:00", "Blue notebook.", None))
    index = CortexIndex(store.root, Fake())
    index.sync(store)

    class Short(Fake):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    index._embedder = Short()
    assert index._vector("notebook", 5) == []
    index.close()


def test_context_adds_neighbors_from_the_same_turn(tmp_path: Path) -> None:
    long = " ".join(f"Sentence {i} is filler text for the test." for i in range(30))
    store = cortex(tmp_path / "c", episode("2023-05-01T10:00:00", long + " The key is 42.", None))
    plain = CortexIndex(store.root)
    plain.sync(store)
    alone = plain.search(store.view(), "key")
    plain.close()
    wide = CortexIndex(store.root, context=1)
    near = wide.search(store.view(), "key")
    wide.close()
    assert len(alone) == 1 and len(near) == 2
    assert near[1].chunk.id == alone[0].chunk.id - 1 and near[1].rank < near[0].rank


def test_assemble_indexed_packs_within_budget_in_order(tmp_path: Path) -> None:
    records = [
        episode(f"2023-05-{day:02d}T10:00:00", f"Day {day}: the notebook was colour {day}.")
        for day in range(1, 30)
    ]
    store = cortex(tmp_path / "c", *records, budget=400)
    index = CortexIndex(store.root)
    index.sync(store)
    view = store.view()
    hits = index.search(view, "notebook colour")
    workspace = assemble_indexed(view, hits, index.newest(), "notebook colour?", store.config())
    assert workspace.estimated_tokens <= 400
    lines = [line for line in workspace.user.split("RECENT\n")[1].split("\n") if line[:2] == "20"]
    assert lines == sorted(lines) and 0 < len(lines) < 58
    assert workspace.episodes_shown > 0 and workspace.episodes_held >= 0
    with pytest.raises(InputDoesNotFit):
        assemble_indexed(view, hits, None, "word " * 1000, store.config())
    from dataclasses import replace

    tiny = replace(store.config(), workspace_budget=256, max_input_tokens=200)
    with pytest.raises(InputDoesNotFit):
        assemble_indexed(view, hits, None, "word " * 190, tiny)
    index.close()


def test_pass_turn_with_an_index_uses_v2_recall(tmp_path: Path) -> None:
    records = [
        episode(f"2023-04-{d:02d}T09:00:00", f"Filler talk number {d}.") for d in range(1, 28)
    ]
    store = cortex(tmp_path / "c", episode("2023-03-01T09:00:00", "My notebook is blue."), *records)
    substrate = Scripted(said("Blue."))
    index = CortexIndex(store.root, Fake())
    result = pass_turn(store, substrate, "What colour is the notebook?", index=index)
    (workspace,) = substrate.seen
    assert isinstance(workspace, Workspace)
    assert "2023-03-01  you  My notebook is blue." in workspace.user
    assert result.used <= result.budget
    assert index.sync(store) == 0  # the new pass was indexed after it was written
    index.close()


def test_pass_turn_falls_back_to_v1_when_the_index_fails(tmp_path: Path) -> None:
    store = cortex(tmp_path / "c", episode("2023-03-01T09:00:00", "My notebook is blue."))
    index = CortexIndex(store.root)
    index.close()
    substrate = Scripted(said("Blue."))
    pass_turn(store, substrate, "What colour is the notebook?", index=index)
    assert "you  My notebook is blue." in substrate.seen[0].user
    with pytest.raises(IndexUnavailable):
        index.newest()


def test_peer_speakers_are_kept(tmp_path: Path) -> None:
    store = cortex(
        tmp_path / "c",
        episode("2023-03-01T09:00:00", "Ask: notebook?", "Blue.", Peer("ops", "asked")),
    )
    index = CortexIndex(store.root)
    index.sync(store)
    speakers = {h.chunk.speaker for h in index.search(store.view(), "notebook blue")}
    assert speakers == {"you", "cortex ops"}
    index.close()


def test_an_index_of_another_format_is_rebuilt(tmp_path: Path) -> None:
    store = cortex(tmp_path / "c", episode("2023-03-01T09:00:00", "My notebook is blue."))
    first = CortexIndex(store.root)
    first.sync(store)
    first._db.execute("UPDATE meta SET value = '0' WHERE key = 'format'")
    first._db.commit()
    first.close()
    again = CortexIndex(store.root)
    assert again.chunks() == 0 and again.sync(store) == 2
    again.close()


def test_index_unavailable_without_fts5_or_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(IndexUnavailable):
        CortexIndex(tmp_path / "missing" / "dir")
    real = sqlite3.connect

    class NoFts:
        def __init__(self, path: object) -> None:
            self._db = real(":memory:")

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            if "fts5" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            return self._db.execute(sql, *args)  # type: ignore[arg-type]

        def commit(self) -> None:
            self._db.commit()

    init_cortex(tmp_path / "c")
    monkeypatch.setattr("angularai.index.sqlite3.connect", NoFts)
    with pytest.raises(IndexUnavailable, match="FTS5"):
        CortexIndex(tmp_path / "c")


def test_index_from_env_and_prefixes(tmp_path: Path) -> None:
    init_cortex(tmp_path / "c")
    assert index_from_env(tmp_path / "c", {}, None) is None
    assert index_from_env(tmp_path / "c", {"ANGULARAI_INDEX": "off"}, None) is None
    with pytest.raises(ConfigError):
        index_from_env(tmp_path / "c", {"ANGULARAI_INDEX": "maybe"}, None)
    made = index_from_env(tmp_path / "c", {"ANGULARAI_INDEX": "on"}, Fake(model="nomic-x"))
    assert made is not None and made.model == "nomic-x"
    made.close()
    notices: list[str] = []
    gone = index_from_env(
        tmp_path / "nowhere" / "x", {"ANGULARAI_INDEX": "on"}, None, warn=notices.append
    )
    assert gone is None and "v1" in notices[0]
    assert task_prefixes("nomic-embed-text") == ("search_document: ", "search_query: ")
    assert task_prefixes("bge-small") == ("", "")


def test_local_embeddings_send_no_key() -> None:
    embedder = embedder_from_env(default_config(), {"ANGULARAI_EMBEDDINGS": "local"})
    assert isinstance(embedder, EmbeddingsAPI) and embedder.model == "nomic-embed-text"
    seen: list[tuple[str, dict[str, str]]] = []

    def transport(
        url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> tuple[int, bytes]:
        seen.append((url, headers))
        return 200, json.dumps({"data": [{"index": 0, "embedding": [0.1, 0.2]}]}).encode()

    api = EmbeddingsAPI(
        model="m",
        base_url="http://127.0.0.1:11434/v1",
        api_key="",
        dimensions=None,
        provider="local embeddings",
        transport=transport,
    )
    assert api.embed(["x"]) == [[0.1, 0.2]]
    assert seen[0][0] == "http://127.0.0.1:11434/v1/embeddings"
    assert "Authorization" not in seen[0][1]
    named = embedder_from_env(
        default_config(),
        {
            "ANGULARAI_EMBEDDINGS": "local",
            "ANGULARAI_EMBEDDINGS_URL": "http://127.0.0.1:8080/v1",
            "ANGULARAI_EMBEDDINGS_MODEL": "bge",
        },
    )
    assert named is not None and named.model == "bge"


def test_estimate_of_lines_bounds_the_whole() -> None:
    parts = ["2023-05-01  you  a b c\n", "2023-05-02  angularai  d e\n"]
    assert estimate_tokens("".join(parts)) <= sum(estimate_tokens(p) for p in parts)


def _remembered(root: Path, fact: str) -> None:
    init_cortex(root)
    pass_turn(open_store(root), Scripted(noted(fact)), f"Note: {fact}")


def test_say_uses_the_index_when_it_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from angularai.cli import main

    _remembered(tmp_path, "The user has a blue notebook.")
    scripted = Scripted(said("Blue."))
    monkeypatch.setattr("angularai.cli.build_substrate", lambda config, environ: scripted)
    monkeypatch.setattr("angularai.cli.embedder_from_env", lambda config, environ: Fake())
    monkeypatch.setenv("ANGULARAI_INDEX", "on")
    assert main(["--cortex", str(tmp_path), "say", "What shade is the notebook?"]) == 0
    assert capsys.readouterr().out == "Blue.\n"
    assert "The user has a blue notebook." in scripted.seen[0].user.split("RECENT")[0]
    assert (tmp_path / INDEX).is_file()


def test_say_with_the_index_and_no_embedding_key_is_lexical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from angularai.cli import main

    _remembered(tmp_path, "The user has a blue notebook.")
    scripted = Scripted(said("Blue."))
    monkeypatch.setattr("angularai.cli.build_substrate", lambda config, environ: scripted)
    monkeypatch.setenv("ANGULARAI_INDEX", "on")
    monkeypatch.setenv("ANGULARAI_EMBEDDINGS", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--cortex", str(tmp_path), "say", "notebook colour?"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Blue.\n" and "Recall is lexical." in captured.err
    assert "blue notebook" in scripted.seen[0].user
