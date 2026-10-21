"""Cortex v2 recall: an SQLite index beside the log, with FTS5 and local vectors.

passes.jsonl stays the truth. ``index.sqlite3`` is a view of it and can be deleted:
the next pass rebuilds it. A v1 cortex migrates by being read: the first sync indexes
every pass already in its log.

What is indexed:
- every current trace, whole;
- every episode, cut into chunks of whole sentences of at most 400 characters, the
  user's words and the reply's apart, each with the date of its pass.

Recall is hybrid. FTS5 ranks chunks by BM25 (porter stemming). When an embedder is
given, chunk vectors are ranked by cosine. The two ranks are fused by reciprocal
rank (k = 60). A superseded or forgotten trace is never returned.

Sync is incremental. The index keeps a running digest of the passes it holds. A log
that was appended to is indexed from where it stopped; a log that was rewritten (a
repair, a forget) is indexed again from the start. A failed embedding call never
stops a pass: the chunk is kept without a vector and embedded on a later sync.
"""

import hashlib
import math
import re
import sqlite3
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from angularai.assemble import SYSTEM, focus, lock_block, speakers
from angularai.attention import score, tokens
from angularai.budget import PARSE_NUDGE, estimate_tokens
from angularai.errors import AngularAIError, ConfigError, InputDoesNotFit
from angularai.models import Config, CortexView, Episode, PassRecord, Trace, Workspace
from angularai.secrets import redact
from angularai.semantic import Embedder
from angularai.store import Store

INDEX = "index.sqlite3"
FORMAT = "2"
CHUNK_CHARS = 400
RRF_K = 60
POOL = 60
_BATCH = 64
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

Kind = Literal["trace", "user", "reply", "episode"]
Unit = Literal["sentence", "turn", "episode"]
UNITS: tuple[Unit, ...] = ("sentence", "turn", "episode")
SEP = "\x1f"
USER_CHARS = 700
REPLY_CHARS = 400
Warn = Callable[[str], None]


class IndexUnavailable(AngularAIError):
    """This Python's SQLite has no FTS5, or the index file cannot be opened."""


@dataclass(frozen=True)
class Chunk:
    id: int
    position: int
    at: str
    kind: Kind
    trace_id: str | None
    text: str
    speaker: str = ""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    rank: float


def chunks_of(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Whole sentences, packed up to limit characters. A longer sentence is cut."""

    pieces: list[str] = []
    current = ""
    for sentence in (" ".join(part.split()) for part in _SENTENCE.split(text)):
        if not sentence:
            continue
        while len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:limit])
            sentence = sentence[limit:].lstrip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def task_prefixes(model: str) -> tuple[str, str]:
    """The task prefixes nomic-embed-text was trained with. Other models get none."""

    if "nomic" in model:
        return "search_document: ", "search_query: "
    return "", ""


def index_from_env(
    root: Path,
    environ: Mapping[str, str],
    embedder: Embedder | None,
    secrets: tuple[str, ...] = (),
    warn: Warn | None = None,
) -> "CortexIndex | None":
    """Cortex v2 recall when ANGULARAI_INDEX is on. None when off or unavailable."""

    mode = environ.get("ANGULARAI_INDEX", "").strip().lower() or "off"
    if mode not in ("off", "on"):
        raise ConfigError("ANGULARAI_INDEX must be off or on.")
    if mode == "off":
        return None
    document, query = task_prefixes("" if embedder is None else embedder.model)
    try:
        return CortexIndex(
            root,
            embedder,
            document_prefix=document,
            query_prefix=query,
            secrets=secrets,
            warn=warn,
        )
    except IndexUnavailable as exc:
        if warn is not None:
            warn(f"{exc} Recall is v1.")
        return None


def pass_key(record: PassRecord) -> bytes:
    parts = [record.at, record.outcome, record.user, record.reply or ""]
    parts += [f"{t.id}\x1f{t.text}" for t in record.remember]
    return "\x1e".join(parts).encode("utf-8")


def fts_query(text: str) -> str:
    """An OR of the query's content words, each quoted, so FTS5 syntax cannot leak in."""

    words = sorted(tokens(text))
    return " OR ".join(f'"{word}"' for word in words)


def _pack(vector: Sequence[float]) -> bytes:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return array("f", (value / norm for value in vector)).tobytes()


def _unpack(blob: bytes) -> array[float]:
    values = array("f")
    values.frombytes(blob)
    return values


class CortexIndex:
    def __init__(
        self,
        root: Path,
        embedder: Embedder | None = None,
        *,
        document_prefix: str = "",
        query_prefix: str = "",
        secrets: tuple[str, ...] = (),
        warn: Warn | None = None,
        chunk_chars: int = CHUNK_CHARS,
        context: int = 0,
        unit: Unit = "sentence",
        word_rank: bool = False,
    ) -> None:
        self.root = root
        self._chunk_chars = chunk_chars
        self._context = context
        self._unit = unit
        self._word_rank = word_rank
        self.path = root / INDEX
        self._embedder = embedder
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._secrets = secrets
        self._warn = warn
        self._warned = False
        try:
            self._db = sqlite3.connect(self.path)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._schema()
        except sqlite3.Error as exc:
            raise IndexUnavailable(f"The cortex index cannot be opened: {exc}.") from exc
        self.path.chmod(0o600)

    @property
    def model(self) -> str:
        return "" if self._embedder is None else self._embedder.model

    def close(self) -> None:
        self._db.close()

    def _schema(self) -> None:
        db = self._db
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        found = self._meta("format")
        shape = f"{self.model}|{self._unit}|{self._chunk_chars}|{self._document_prefix}"
        if found not in (None, FORMAT) or self._meta("shape") not in (None, shape):
            db.execute("DROP TABLE IF EXISTS fts")
            db.execute("DROP TABLE IF EXISTS chunks")
            db.execute("DELETE FROM meta")
        db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, position INTEGER "
            "NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL, trace_id TEXT, text TEXT NOT "
            "NULL, speaker TEXT NOT NULL DEFAULT '', vector BLOB)"
        )
        try:
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, content='chunks', "
                "content_rowid='id', tokenize='porter unicode61')"
            )
        except sqlite3.OperationalError as exc:
            raise IndexUnavailable("This Python's SQLite has no FTS5.") from exc
        self._set("format", FORMAT)
        self._set("shape", shape)
        db.commit()

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _set(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def sync(self, store: Store) -> int:
        """Index what the log holds and the index does not. Returns chunks added."""

        try:
            return self._sync(store)
        except sqlite3.Error as exc:
            raise IndexUnavailable(f"The cortex index failed: {exc}.") from exc

    def _sync(self, store: Store) -> int:
        records = store.passes()
        held = int(self._meta("passes") or 0)
        digest = hashlib.sha256()
        start = 0
        if held <= len(records):
            for record in records[:held]:
                digest.update(hashlib.sha256(pass_key(record)).digest())
            if digest.hexdigest() == (self._meta("digest") or hashlib.sha256().hexdigest()):
                start = held
        if start == 0 and held:
            self._clear()
            digest = hashlib.sha256()
        added = 0
        for position in range(start, len(records)):
            record = records[position]
            added += self._add(position, record)
            digest.update(hashlib.sha256(pass_key(record)).digest())
        self._set("passes", str(len(records)))
        self._set("digest", digest.hexdigest())
        self._db.commit()
        self._embed_missing()
        return added

    def _clear(self) -> None:
        self._db.execute("DELETE FROM chunks")
        self._db.execute("INSERT INTO fts (fts) VALUES ('rebuild')")

    def _add(self, position: int, record: PassRecord) -> int:
        rows: list[tuple[int, str, str, str | None, str, str]] = []
        for trace in record.remember:
            rows.append((position, trace.at, "trace", trace.id, trace.text, ""))
        if record.outcome != "error":
            episode = Episode(record.at, record.user, record.reply, record.outcome, record.peer)
            asker, answerer = speakers(episode)
            user = " ".join(record.user.split())
            reply = " ".join((record.reply or "").split())
            if self._unit == "episode":
                text = user + (SEP + reply if reply else "")
                rows.append((position, record.at, "episode", None, text, asker + SEP + answerer))
            elif self._unit == "turn":
                rows.append((position, record.at, "user", None, user, asker))
                if reply:
                    rows.append((position, record.at, "reply", None, reply, answerer))
            else:
                for piece in chunks_of(record.user, self._chunk_chars):
                    rows.append((position, record.at, "user", None, piece, asker))
                for piece in chunks_of(record.reply or "", self._chunk_chars):
                    rows.append((position, record.at, "reply", None, piece, answerer))
        for row in rows:
            cursor = self._db.execute(
                "INSERT INTO chunks (position, at, kind, trace_id, text, speaker) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )
            self._db.execute(
                "INSERT INTO fts (rowid, text) VALUES (?, ?)", (cursor.lastrowid, row[4])
            )
        return len(rows)

    def _embed_missing(self) -> None:
        if self._embedder is None:
            return
        while True:
            rows = self._db.execute(
                "SELECT id, text FROM chunks WHERE vector IS NULL ORDER BY id LIMIT ?", (_BATCH,)
            ).fetchall()
            if not rows:
                return
            texts = [
                self._document_prefix + redact(str(text).replace(SEP, "\n"), self._secrets)
                for _, text in rows
            ]
            try:
                vectors = self._embedder.embed(texts)
            except AngularAIError as exc:
                self._failed(exc)
                return
            for (chunk_id, _), vector in zip(rows, vectors, strict=True):
                self._db.execute(
                    "UPDATE chunks SET vector = ? WHERE id = ?", (_pack(vector), chunk_id)
                )
            self._db.commit()

    def _failed(self, exc: Exception) -> None:
        if self._warn is not None and not self._warned:
            self._warned = True
            self._warn(f"Embeddings are unavailable: {exc} Recall is lexical.")

    def chunks(self) -> int:
        return int(self._db.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def search(self, view: CortexView, query: str, limit: int = POOL) -> list[Hit]:
        """Chunks for query, best first, fused from BM25 and cosine ranks."""

        try:
            return self._search(view, query, limit)
        except sqlite3.Error as exc:
            raise IndexUnavailable(f"The cortex index failed: {exc}.") from exc

    def _search(self, view: CortexView, query: str, limit: int) -> list[Hit]:
        current = {trace.id for trace in view.current_traces()}
        ranks: dict[int, float] = {}
        for order, chunk_id in enumerate(self._lexical(query, limit * 2)):
            ranks[chunk_id] = ranks.get(chunk_id, 0.0) + 1.0 / (RRF_K + order + 1)
        for order, chunk_id in enumerate(self._vector(query, limit * 2)):
            ranks[chunk_id] = ranks.get(chunk_id, 0.0) + 1.0 / (RRF_K + order + 1)
        if self._word_rank:
            for order, chunk_id in enumerate(self._words(query, limit * 2)):
                ranks[chunk_id] = ranks.get(chunk_id, 0.0) + 1.0 / (RRF_K + order + 1)
        hits: list[Hit] = []
        for chunk_id, rank in sorted(ranks.items(), key=lambda item: (-item[1], item[0])):
            chunk = self._chunk(chunk_id)
            if chunk is None or (chunk.kind == "trace" and chunk.trace_id not in current):
                continue
            hits.append(Hit(chunk, rank))
            if len(hits) >= limit:
                break
        return self._with_context(hits) if self._context else hits

    def _with_context(self, hits: list[Hit]) -> list[Hit]:
        """Each hit followed by up to context chunks either side of it in the same turn."""

        out: list[Hit] = []
        seen: set[int] = set()
        for hit in hits:
            if hit.chunk.id in seen:
                continue
            out.append(hit)
            seen.add(hit.chunk.id)
            if hit.chunk.kind == "trace":
                continue
            for step in range(1, self._context + 1):
                for neighbor_id in (hit.chunk.id - step, hit.chunk.id + step):
                    neighbor = self._chunk(neighbor_id)
                    if (
                        neighbor is None
                        or neighbor.id in seen
                        or neighbor.position != hit.chunk.position
                        or neighbor.kind != hit.chunk.kind
                    ):
                        continue
                    out.append(Hit(neighbor, hit.rank / 2))
                    seen.add(neighbor.id)
        return out

    def _lexical(self, query: str, limit: int) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        rows = self._db.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (match, limit)
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _words(self, query: str, limit: int) -> list[int]:
        """v1's attention: the share of the query's words found in the unit."""

        scored: list[tuple[float, int]] = []
        for chunk_id, text in self._db.execute("SELECT id, text FROM chunks"):
            rank = score(query, str(text).replace(SEP, "\n"))
            if rank > 0:
                scored.append((rank, int(chunk_id)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk_id for _, chunk_id in scored[:limit]]

    def _vector(self, query: str, limit: int) -> list[int]:
        if self._embedder is None:
            return []
        try:
            (vector,) = self._embedder.embed([self._query_prefix + redact(query, self._secrets)])
        except (AngularAIError, ValueError) as exc:
            self._failed(exc)
            return []
        probe = _unpack(_pack(vector))
        scored: list[tuple[float, int]] = []
        for chunk_id, blob in self._db.execute(
            "SELECT id, vector FROM chunks WHERE vector IS NOT NULL"
        ):
            values = _unpack(blob)
            if len(values) != len(probe):
                continue
            scored.append((sum(a * b for a, b in zip(probe, values, strict=True)), int(chunk_id)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk_id for _, chunk_id in scored[:limit]]

    def _chunk(self, chunk_id: int) -> Chunk | None:
        row = self._db.execute(
            "SELECT id, position, at, kind, trace_id, text, speaker FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        found = str(row[3])
        kind: Kind = (
            "trace"
            if found == "trace"
            else "user"
            if found == "user"
            else "episode"
            if found == "episode"
            else "reply"
        )
        return Chunk(int(row[0]), int(row[1]), str(row[2]), kind, row[4], str(row[5]), str(row[6]))

    def newest(self) -> Chunk | None:
        """The first chunk of the newest pass, for continuity."""

        try:
            row = self._db.execute(
                "SELECT id FROM chunks WHERE kind != 'trace' ORDER BY position DESC, id ASC LIMIT 1"
            ).fetchone()
            return None if row is None else self._chunk(int(row[0]))
        except sqlite3.Error as exc:
            raise IndexUnavailable(f"The cortex index failed: {exc}.") from exc


def _chunk_line(chunk: Chunk, traces: dict[str, Trace], query: str = "") -> str:
    """One unit as the model reads it: dated, named, and focused on the query as v1 does."""

    if chunk.kind == "trace" and chunk.trace_id in traces:
        trace = traces[chunk.trace_id]
        waves = ",".join(trace.wavelengths) if trace.wavelengths else "-"
        return f"{trace.id}  {waves}  {chunk.text}\n"
    date = chunk.at[:10]
    if chunk.kind == "episode":
        user, _, reply = chunk.text.partition(SEP)
        asker, _, answerer = chunk.speaker.partition(SEP)
        line = f"{date}  {asker}  {focus(user, query, USER_CHARS)}\n"
        if reply:
            line += f"{date}  {answerer}  {focus(reply, query, REPLY_CHARS)}\n"
        return line
    limit = USER_CHARS if chunk.kind == "user" else REPLY_CHARS
    return f"{date}  {chunk.speaker or chunk.kind}  {focus(chunk.text, query, limit)}\n"


def assemble_indexed(
    view: CortexView,
    hits: Sequence[Hit],
    newest: Chunk | None,
    user_text: str,
    config: Config,
    system: str = SYSTEM,
) -> Workspace:
    """Pack L2 from index hits. The same blocks, budget and refusals as assemble.

    Hits are added best first while they fit, short of the repair nudge's room. Traces go under TRACES; chunks under
    RECENT in the order they were said, each with its date. The newest pass's first
    chunk is added for continuity when it fits.
    """

    text = user_text.strip()
    if estimate_tokens(text) > config.max_input_tokens:
        raise InputDoesNotFit(
            "That input does not fit the workspace. "
            f"The input cap is {config.max_input_tokens} estimated tokens."
        )
    head = lock_block(config.forbid) + "\n\nTRACES\n"
    recent = "\nRECENT\n"
    tail = f"\nINPUT\n{text}"
    base = estimate_tokens(system) + estimate_tokens(head + recent + tail)
    if base > config.workspace_budget:
        raise InputDoesNotFit(
            f"That input does not fit the workspace budget of {config.workspace_budget}. "
            "Shorten it, or raise the budget with angularai budget."
        )
    traces = {trace.id: trace for trace in view.current_traces()}
    # Packed to the budget, it still leaves room for the one repair retry.
    room = config.workspace_budget - base - estimate_tokens(PARSE_NUDGE)
    chosen: list[Chunk] = []
    seen: set[int] = set()
    candidates = [hit.chunk for hit in hits]
    if newest is not None:
        candidates.append(newest)
    for chunk in candidates:
        if chunk.id in seen:
            continue
        # Each line is estimated on its own; the sum is never below the whole.
        cost = estimate_tokens(_chunk_line(chunk, traces, text))
        if cost <= room:
            chosen.append(chunk)
            seen.add(chunk.id)
            room -= cost
    shown_traces = [c for c in chosen if c.kind == "trace"]
    said = sorted((c for c in chosen if c.kind != "trace"), key=lambda c: (c.position, c.id))
    body = (
        head
        + "".join(_chunk_line(c, traces, text) for c in shown_traces)
        + recent
        + "".join(_chunk_line(c, traces, text) for c in said)
        + tail
    )
    passes = {c.position for c in said}
    return Workspace(
        system=system,
        user=body,
        input_text=text,
        trace_ids=tuple(c.trace_id for c in shown_traces if c.trace_id is not None),
        estimated_tokens=estimate_tokens(system) + estimate_tokens(body),
        budget=config.workspace_budget,
        traces_shown=len(shown_traces),
        traces_held=len(traces) - len(shown_traces),
        episodes_shown=len(passes),
        episodes_held=max(0, len(view.episodes) - len(passes)),
    )
