"""Recall by meaning, with a fake embedder: no key, no network."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from angularai.cli import main
from angularai.errors import (
    AuthFailed,
    BadResponse,
    ConfigError,
    InputRejected,
    KeyMissing,
    Unreachable,
)
from angularai.models import Integration, TraceDraft
from angularai.pilot import pass_turn
from angularai.provider import Clock
from angularai.semantic import (
    VECTORS,
    EmbeddingsAPI,
    SemanticRecall,
    cosine,
    embedder_from_env,
    load_vectors,
    minimum_from_env,
    vectors_in,
)
from angularai.store import default_config, init_cortex, open_store
from tests.scripted import Scripted, noted, said

# Words that share a meaning share an axis. The last axis keeps every vector nonzero.
CONCEPTS = {
    "color": {"blue", "colour", "color", "shade", "red"},
    "notebook": {"notebook", "journal"},
    "food": {"catering", "lunch", "noon", "food"},
}
SECRET = "Zq8vLm2pXw9rTt4Ks7"


def concept_vector(text: str) -> list[float]:
    words = {word.strip("?.,!").lower() for word in text.split()}
    return [float(len(words & group)) for group in CONCEPTS.values()] + [0.1]


class Fake:
    """An embedder that records every text it was sent."""

    def __init__(self, fail: Exception | None = None, model: str = "fake-1") -> None:
        self.sent: list[list[str]] = []
        self.fail = fail
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.sent.append(list(texts))
        if self.fail is not None:
            raise self.fail
        return [concept_vector(text) for text in texts]


def remembered(root: Path, *facts: str) -> None:
    init_cortex(root)
    store = open_store(root)
    for fact in facts:
        pass_turn(store, Scripted(noted(fact)), f"Note: {fact}")


def ranks(root: Path, query: str, fake: Fake | None = None) -> dict[str, float]:
    store = open_store(root)
    return SemanticRecall(root, fake or Fake()).ranks(store.view(), query)


def text_of(root: Path, trace_id: str) -> str:
    return next(t.text for t in open_store(root).view().traces if t.id == trace_id)


@pytest.mark.parametrize("query", ["What shade is it?", "colour", "notebook color", "shade"])
def test_a_paraphrase_finds_the_blue_notebook(tmp_path: Path, query: str) -> None:
    remembered(tmp_path, "The user has a blue notebook.", "Catering is at noon.")
    found = ranks(tmp_path, query)
    assert [text_of(tmp_path, trace_id) for trace_id in found] == ["The user has a blue notebook."]
    assert all(value >= 0.35 for value in found.values())


def test_an_unrelated_question_finds_nothing(tmp_path: Path) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    assert ranks(tmp_path, "Where is the train station?") == {}


def test_recall_is_lexical_unless_embeddings_are_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    monkeypatch.delenv("ANGULARAI_EMBEDDINGS", raising=False)
    fake = Fake()
    monkeypatch.setattr("angularai.cli.embedder_from_env", lambda config, environ: fake)
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "recall", "--lexical", "shade"]) == 0
    assert capsys.readouterr().out == "Nothing in the cortex matches that.\n"
    assert fake.sent == []
    assert main(["--cortex", str(tmp_path), "recall", "shade"]) == 0
    line = capsys.readouterr().out
    assert "~0.71" in line and "The user has a blue notebook." in line
    assert main(["--cortex", str(tmp_path), "recall", "notebook"]) == 0
    assert "  1.00  " in capsys.readouterr().out


def test_with_embeddings_off_recall_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    monkeypatch.setenv("ANGULARAI_EMBEDDINGS", "off")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used-0000")
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "recall", "shade"]) == 0
    assert capsys.readouterr().out == "Nothing in the cortex matches that.\n"
    assert not (tmp_path / VECTORS).exists()


def test_vectors_survive_a_new_process_and_are_not_sent_again(tmp_path: Path) -> None:
    remembered(tmp_path, "The user has a blue notebook.", "Catering is at noon.")
    first = Fake()
    ranks(tmp_path, "shade", first)
    assert sorted(first.sent[0][1:]) == ["Catering is at noon.", "The user has a blue notebook."]
    assert (tmp_path / VECTORS).stat().st_mode & 0o777 == 0o600
    stored = load_vectors(tmp_path, "fake-1")
    assert len(stored) == 2
    for _, vector in stored.values():
        assert len(vector) == len(CONCEPTS) + 1
    again = Fake()
    assert ranks(tmp_path, "lunch", again)
    assert again.sent == [["lunch"]]
    # Another model's vectors are not this model's.
    other = Fake(model="fake-2")
    ranks(tmp_path, "lunch", other)
    assert len(other.sent[0]) == 3


def test_a_damaged_vector_line_is_skipped_and_rebuilt(tmp_path: Path) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    ranks(tmp_path, "shade")
    with (tmp_path / VECTORS).open("ab") as handle:
        handle.write(
            b'{"v": 1, "model": "fake-1", "trace": "t_x", "digest": "d", "vector": "!!"}\n'
        )
        handle.write(b"not json\n[]\n")
        handle.write(b'{"v": 1, "model": "fake-1", "trace": "t_y", "digest": "d", "vector": "AAA"}')
    assert len(load_vectors(tmp_path, "fake-1")) == 1
    (tmp_path / VECTORS).write_bytes(b"\xff\xfe garbage")
    fake = Fake()
    assert ranks(tmp_path, "shade", fake)
    assert len(fake.sent[0]) == 2


def test_a_secret_in_a_trace_is_redacted_before_it_is_embedded(tmp_path: Path) -> None:
    # A trace from an older log that still holds a secret.
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    draft = Integration("Noted.", (TraceDraft(f"The blue notebook code is {SECRET}.", ()),), ())
    pass_turn(store, Scripted(draft), "Note the code.")
    fake = Fake()
    found = SemanticRecall(tmp_path, fake, secrets=(SECRET,)).ranks(store.view(), f"shade {SECRET}")
    assert found
    sent = json.dumps(fake.sent)
    assert SECRET not in sent and "[redacted]" in sent
    assert SECRET.encode() not in (tmp_path / VECTORS).read_bytes()


def test_a_failure_is_reported_once_and_recall_stays_lexical(tmp_path: Path) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    notes: list[str] = []
    fake = Fake(fail=Unreachable(f"The substrate is unreachable. key {SECRET}"))
    recall = SemanticRecall(tmp_path, fake, secrets=(SECRET,), warn=notes.append)
    view = open_store(tmp_path).view()
    assert recall.ranks(view, "shade") == {}
    assert recall.ranks(view, "shade") == {}
    recall.index(view.traces)
    assert len(fake.sent) == 1
    assert len(notes) == 1 and "Recall is lexical." in notes[0] and SECRET not in notes[0]


def test_a_pass_sends_a_trace_found_by_meaning(tmp_path: Path) -> None:
    remembered(tmp_path, "The user has a blue notebook.", "Catering is at noon.")
    store = open_store(tmp_path)
    plain = Scripted(said("I do not know."))
    pass_turn(store, plain, "What shade is it?")
    assert "blue notebook" not in plain.seen[0].user.split("RECENT")[0]
    meant = Scripted(said("Blue."))
    result = pass_turn(store, meant, "What shade is it?", recall=SemanticRecall(tmp_path, Fake()))
    traces = meant.seen[0].user.split("RECENT")[0]
    assert "The user has a blue notebook." in traces
    assert "Catering" not in traces
    assert result.used <= result.budget


def test_a_failing_embedder_never_stops_a_pass(tmp_path: Path) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    store = open_store(tmp_path)
    recall = SemanticRecall(tmp_path, Fake(fail=Unreachable("down")))
    result = pass_turn(
        store, Scripted(noted("Lunch is at one.")), "Lunch is at one.", recall=recall
    )
    assert result.reply == "Noted."
    assert any(t.text == "Lunch is at one." for t in store.view().traces)


def test_a_trace_is_indexed_when_it_is_written(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    fake = Fake()
    pass_turn(
        store,
        Scripted(noted("The user has a blue notebook.")),
        "I have a blue notebook.",
        recall=SemanticRecall(tmp_path, fake),
    )
    assert fake.sent[-1] == ["The user has a blue notebook."]
    assert len(load_vectors(tmp_path, "fake-1")) == 1


def test_say_uses_meaning_when_the_cli_has_an_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    scripted = Scripted(said("Blue."))
    monkeypatch.setattr("angularai.cli.build_substrate", lambda config, environ: scripted)
    monkeypatch.setattr("angularai.cli.embedder_from_env", lambda config, environ: Fake())
    assert main(["--cortex", str(tmp_path), "say", "What shade is it?"]) == 0
    assert capsys.readouterr().out == "Blue.\n"
    assert "The user has a blue notebook." in scripted.seen[0].user.split("RECENT")[0]


def test_a_missing_embeddings_key_is_a_note_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    remembered(tmp_path, "The user has a blue notebook.")
    monkeypatch.setenv("ANGULARAI_EMBEDDINGS", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    capsys.readouterr()
    assert main(["--cortex", str(tmp_path), "recall", "notebook"]) == 0
    captured = capsys.readouterr()
    assert "The user has a blue notebook." in captured.out
    assert "OPENAI_API_KEY is not set" in captured.err and "Recall is lexical." in captured.err


def test_the_environment_picks_the_embedder() -> None:
    config = default_config()
    assert embedder_from_env(config, {}) is None
    assert embedder_from_env(config, {"ANGULARAI_EMBEDDINGS": " OFF "}) is None
    with pytest.raises(ConfigError, match="off, openai, compatible, or local"):
        embedder_from_env(config, {"ANGULARAI_EMBEDDINGS": "cloud"})
    with pytest.raises(KeyMissing, match="OPENAI_API_KEY"):
        embedder_from_env(config, {"ANGULARAI_EMBEDDINGS": "openai"})
    chosen = embedder_from_env(config, {"ANGULARAI_EMBEDDINGS": "openai", "OPENAI_API_KEY": "k"})
    assert chosen is not None and chosen.model == "text-embedding-3-small"
    with pytest.raises(InputRejected, match="--base-url"):
        embedder_from_env(config, {"ANGULARAI_EMBEDDINGS": "compatible"})
    wired = default_config().__class__(**{**config.__dict__, "base_url": "http://127.0.0.1:9/v1"})
    with pytest.raises(KeyMissing, match="ANGULARAI_API_KEY"):
        embedder_from_env(wired, {"ANGULARAI_EMBEDDINGS": "compatible"})
    named = embedder_from_env(
        wired,
        {
            "ANGULARAI_EMBEDDINGS": "compatible",
            "ANGULARAI_API_KEY": "local",
            "ANGULARAI_EMBEDDINGS_MODEL": "nomic-embed-text",
        },
    )
    assert named is not None and named.model == "nomic-embed-text"
    assert minimum_from_env({}) == 0.35
    assert minimum_from_env({"ANGULARAI_EMBEDDINGS_MIN": "0.2"}) == 0.2
    for bad in ("1.5", "-0.1", "nan", "high"):
        with pytest.raises(ConfigError, match="ANGULARAI_EMBEDDINGS_MIN"):
            minimum_from_env({"ANGULARAI_EMBEDDINGS_MIN": bad})


def api(answer: tuple[int, bytes], captured: list[Mapping[str, object]]) -> EmbeddingsAPI:
    def transport(
        url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> tuple[int, bytes] | tuple[int, bytes, Mapping[str, str]]:
        del timeout
        assert url == "https://api.openai.com/v1/embeddings"
        assert headers["Authorization"] == "Bearer sk-test-embed-0000"
        captured.append(payload)
        status, body = answer
        return (status, body) if status < 300 else (status, body, {})

    return EmbeddingsAPI(
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1/",
        api_key="sk-test-embed-0000",
        dimensions=512,
        provider="OpenAI embeddings",
        transport=transport,
        clock=Clock(sleep=lambda seconds: None, random=lambda: 0.0),
    )


def test_the_api_sends_the_texts_and_reads_vectors_by_index() -> None:
    captured: list[Mapping[str, object]] = []
    body = {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1.5, 0]}]}
    model = api((200, json.dumps(body).encode()), captured)
    assert model.embed(["a", "b"]) == [[1.5, 0.0], [0.0, 1.0]]
    assert captured == [{"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 512}]
    assert model.embed([]) == []


def test_the_api_names_a_rejected_key() -> None:
    with pytest.raises(AuthFailed):
        api((401, b'{"error": {"message": "bad key"}}'), []).embed(["a"])


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [{"index": 0, "embedding": []}],
        [{"index": 1, "embedding": [1]}],
        [{"index": True, "embedding": [1]}],
        [{"index": 0, "embedding": ["1"]}],
        [{"index": 0, "embedding": [True]}],
        [{"index": 0, "embedding": [1e999]}],
        ["vector"],
    ],
)
def test_an_answer_without_good_vectors_is_bad(data: object) -> None:
    with pytest.raises(BadResponse):
        vectors_in({"data": data}, 1, "OpenAI")


def test_vectors_of_different_sizes_are_bad() -> None:
    data = [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]
    with pytest.raises(BadResponse, match="different sizes"):
        vectors_in({"data": data}, 2)
    duplicate = [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]
    with pytest.raises(BadResponse):
        vectors_in({"data": duplicate}, 2)


def test_cosine_is_bounded_and_safe_on_zero() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
