"""Recall by meaning: trace vectors from an embeddings API, kept beside the cortex.

Off unless ANGULARAI_EMBEDDINGS names a provider. Nothing is sent while it is off.

- ``off`` (the default): lexical attention only.
- ``openai``: POST {base}/embeddings with OPENAI_API_KEY, ``text-embedding-3-small``
  at 512 dimensions. The base is the cortex's when it is wired to openai with one.
- ``compatible``: the cortex's ``--base-url`` with ANGULARAI_API_KEY.
- ``local``: ANGULARAI_EMBEDDINGS_URL (default Ollama, ``http://127.0.0.1:11434/v1``),
  no key, ``nomic-embed-text``. Free, and nothing leaves the machine.

ANGULARAI_EMBEDDINGS_MODEL names another model. ANGULARAI_EMBEDDINGS_MIN, from 0 to 1,
default 0.35, is the cosine a trace needs. At most 8 traces match by meaning.

Only traces are embedded, never episodes or tool output. A trace's text is redacted
and embedded once, in the first request after it is written. Its vector is a line in
``embeddings.jsonl``: ``{"v": 1, "model", "trace", "digest", "vector"}``, the vector
base64 little-endian float32, and the digest that of the redacted text. A line whose
model or digest does not match is ignored, so the file is a cache: delete it and it
is rebuilt. Lines are appended with durable.append_line under the cortex lock.

A failure (no key, no network, an answer without vectors) never stops a pass. The
reason is reported once, and recall stays lexical.
"""

import base64
import hashlib
import json
import math
import sys
from array import array
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from angularai.durable import append_line, atomic_write, locked
from angularai.errors import AngularAIError, BadResponse, ConfigError, InputRejected, KeyMissing
from angularai.models import Config, CortexView, Trace
from angularai.provider import Clock, Policy
from angularai.secrets import redact
from angularai.substrate import Transport, _Caller

VECTORS = "embeddings.jsonl"
MODES = ("off", "openai", "compatible", "local")
LOCAL_URL = "http://127.0.0.1:11434/v1"
LOCAL_MODEL = "nomic-embed-text"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_MIN = 0.35
TOP = 8
_DIMENSIONS = 512
_FORMAT = 1
_BATCH = 256
_MAX_DIMENSIONS = 8192
# Short, so a slow embeddings API costs a pass seconds, not minutes.
_POLICY = Policy(connect_timeout=5.0, read_timeout=15.0, deadline=20.0, attempts=2, max_wait=5.0)

Warn = Callable[[str], None]


class Embedder(Protocol):
    @property
    def model(self) -> str:
        """The model name stored with each vector."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order."""


class EmbeddingsAPI:
    """The OpenAI embeddings endpoint, or a compatible one."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dimensions: int | None,
        provider: str,
        transport: Transport | None = None,
        clock: Clock | None = None,
        policy: Policy | None = None,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._dimensions = dimensions
        self._caller = _Caller(
            transport=transport,
            policy=policy or _POLICY,
            clock=clock,
            provider=provider,
            secret=api_key,
        )

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": self._model, "input": list(texts)}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        headers = {"Content-Type": "application/json", "User-Agent": "angularai/0.1.0"}
        if self._api_key:
            # A local server without a key gets no Authorization header at all.
            headers["Authorization"] = f"Bearer {self._api_key}"
        reply = self._caller.post(self._caller.deadline(), self._url, headers, payload)
        return vectors_in(reply, len(texts), self._caller.provider)


def vectors_in(reply: Mapping[str, object], count: int, provider: str = "") -> list[list[float]]:
    """The vectors of an embeddings answer, by index. Anything else is BadResponse."""

    the = f"The embeddings API ({provider})" if provider else "The embeddings API"
    data = reply.get("data")
    if not isinstance(data, list) or len(data) != count:
        raise BadResponse(f"{the} did not return one vector per text.")
    found: dict[int, list[float]] = {}
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < count
            or index in found
            or not isinstance(vector, list)
            or not 0 < len(vector) <= _MAX_DIMENSIONS
        ):
            raise BadResponse(f"{the} returned a vector it did not index.")
        numbers: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise BadResponse(f"{the} returned a vector that is not numbers.")
            number = float(value)
            if not math.isfinite(number):
                raise BadResponse(f"{the} returned a vector that is not finite.")
            numbers.append(number)
        found[index] = numbers
    if len({len(vector) for vector in found.values()}) != 1:
        raise BadResponse(f"{the} returned vectors of different sizes.")
    return [found[index] for index in range(count)]


def embedder_from_env(config: Config, environ: Mapping[str, str]) -> Embedder | None:
    """None while ANGULARAI_EMBEDDINGS is off. A bad value is refused by name."""

    mode = environ.get("ANGULARAI_EMBEDDINGS", "").strip().lower() or "off"
    if mode not in MODES:
        raise ConfigError("ANGULARAI_EMBEDDINGS must be off, openai, compatible, or local.")
    if mode == "off":
        return None
    if mode == "local":
        return EmbeddingsAPI(
            model=environ.get("ANGULARAI_EMBEDDINGS_MODEL", "").strip() or LOCAL_MODEL,
            base_url=environ.get("ANGULARAI_EMBEDDINGS_URL", "").strip() or LOCAL_URL,
            api_key="",
            dimensions=None,
            provider="local embeddings",
        )
    model = environ.get("ANGULARAI_EMBEDDINGS_MODEL", "").strip() or DEFAULT_MODEL
    if mode == "openai":
        key = environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise KeyMissing("OPENAI_API_KEY is not set. The key stays in the environment.")
        wired = config.base_url if config.substrate == "openai" else None
        return EmbeddingsAPI(
            model=model,
            base_url=wired or "https://api.openai.com/v1",
            api_key=key,
            dimensions=_DIMENSIONS if model.startswith("text-embedding-3") else None,
            provider="OpenAI embeddings",
        )
    if config.base_url is None:
        raise InputRejected("ANGULARAI_EMBEDDINGS=compatible needs a cortex wired with --base-url.")
    key = environ.get("ANGULARAI_API_KEY", "").strip()
    if not key:
        raise KeyMissing("ANGULARAI_API_KEY is not set. The key stays in the environment.")
    return EmbeddingsAPI(
        model=model,
        base_url=config.base_url,
        api_key=key,
        dimensions=None,
        provider="compatible embeddings",
    )


def minimum_from_env(environ: Mapping[str, str]) -> float:
    raw = environ.get("ANGULARAI_EMBEDDINGS_MIN", "").strip()
    if not raw:
        return DEFAULT_MIN
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0 <= value <= 1:
        raise ConfigError("ANGULARAI_EMBEDDINGS_MIN must be a number from 0 to 1.")
    return value


class SemanticRecall:
    """Cosine ranks for the current traces. Every failure degrades to no ranks."""

    def __init__(
        self,
        root: Path,
        embedder: Embedder,
        *,
        minimum: float = DEFAULT_MIN,
        secrets: tuple[str, ...] = (),
        warn: Warn | None = None,
    ) -> None:
        self.root = root
        self.embedder = embedder
        self.minimum = minimum
        self.secrets = secrets
        self._warn = warn
        self.failure: str | None = None

    def ranks(self, view: CortexView, query: str) -> dict[str, float]:
        """Trace id to cosine, for at most TOP traces at or above the minimum."""

        try:
            return self._ranks(view, query)
        except (AngularAIError, OSError) as exc:
            self._failed(exc)
            return {}

    def index(self, traces: Sequence[Trace]) -> None:
        """Embed traces just written. A failure leaves them for the next request."""

        if not traces or self.failure is not None:
            return
        try:
            known = load_vectors(self.root, self.embedder.model)
            self._embed_missing(traces, known, ())
        except (AngularAIError, OSError) as exc:
            self._failed(exc)

    def _ranks(self, view: CortexView, query: str) -> dict[str, float]:
        if self.failure is not None:
            return {}
        current = view.current_traces()
        known = load_vectors(self.root, self.embedder.model)
        wanted = redact(query, self.secrets)
        asked = self._embed_missing(current, known, (wanted,))[0]
        found: list[tuple[float, str]] = []
        for trace in current:
            stored = known.get(trace.id)
            if stored is None or stored[0] != self._digest(trace) or len(stored[1]) != len(asked):
                continue
            similarity = cosine(asked, stored[1])
            if similarity >= self.minimum:
                found.append((similarity, trace.id))
        found.sort(reverse=True)
        return {trace_id: similarity for similarity, trace_id in found[:TOP]}

    def _embed_missing(
        self,
        traces: Sequence[Trace],
        known: dict[str, tuple[str, list[float]]],
        first: tuple[str, ...],
    ) -> list[list[float]]:
        """Embed first (the query) and every trace without a matching vector, newest first."""

        missing = [
            trace
            for trace in sorted(traces, key=lambda item: item.at, reverse=True)
            if (known.get(trace.id) or ("",))[0] != self._digest(trace)
        ][: _BATCH - len(first)]
        texts = [*first, *(redact(trace.text, self.secrets) for trace in missing)]
        if not texts:
            return []
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(texts):
            raise BadResponse("The embeddings API did not return one vector per text.")
        fresh = vectors[len(first) :]
        if missing:
            lines = [
                _record(self.embedder.model, trace.id, self._digest(trace), vector)
                for trace, vector in zip(missing, fresh, strict=True)
            ]
            with locked(self.root):
                for line in lines:
                    append_line(self.root / VECTORS, line)
            for trace, vector in zip(missing, fresh, strict=True):
                known[trace.id] = (self._digest(trace), vector)
        return vectors[: len(first)]

    def _digest(self, trace: Trace) -> str:
        return digest(redact(trace.text, self.secrets))

    def _failed(self, exc: Exception) -> None:
        if self.failure is not None:
            return
        reason = " ".join(redact(str(exc), self.secrets).split())[:200] or type(exc).__name__
        self.failure = reason
        if self._warn is not None:
            self._warn(f"Embeddings are unavailable: {reason} Recall is lexical.")


def load_vectors(root: Path, model: str) -> dict[str, tuple[str, list[float]]]:
    """Trace id to (digest, vector) for one model. A line that cannot be read is skipped."""

    path = root / VECTORS
    with locked(root, shared=True):
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return {}
    found: dict[str, tuple[str, list[float]]] = {}
    for line in raw.split(b"\n"):
        parsed = _parse(line, model)
        if parsed is not None:
            found[parsed[0]] = (parsed[1], parsed[2])
    return found


def drop_vectors(root: Path, trace_ids: frozenset[str]) -> int:
    """Remove every vector of these traces, for any model. Returns how many lines went.

    The file is rewritten with atomic_write under the cortex lock. A line that cannot be
    read is kept as it is.
    """

    path = root / VECTORS
    with locked(root):
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return 0
        lines = [line for line in raw.split(b"\n") if line]
        kept = [line for line in lines if _trace_of(line) not in trace_ids]
        dropped = len(lines) - len(kept)
        if dropped:
            atomic_write(path, b"".join(line + b"\n" for line in kept))
    return dropped


def _trace_of(line: bytes) -> str | None:
    try:
        payload: object = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    trace = payload.get("trace")
    return trace if isinstance(trace, str) else None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if not norm or not math.isfinite(norm):
        return 0.0
    return max(-1.0, min(1.0, dot / norm))


def digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).hexdigest()


def _record(model: str, trace_id: str, text_digest: str, vector: Sequence[float]) -> bytes:
    packed = array("f", vector)
    if sys.byteorder != "little":
        packed.byteswap()
    payload = {
        "v": _FORMAT,
        "model": model,
        "trace": trace_id,
        "digest": text_digest,
        "vector": base64.b64encode(packed.tobytes()).decode("ascii"),
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _parse(line: bytes, model: str) -> tuple[str, str, list[float]] | None:
    if not line.strip():
        return None
    try:
        payload: object = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("v") != _FORMAT:
        return None
    trace_id, text_digest, vector = (
        payload.get("trace"),
        payload.get("digest"),
        payload.get("vector"),
    )
    if payload.get("model") != model or not isinstance(trace_id, str):
        return None
    if not isinstance(text_digest, str) or not isinstance(vector, str):
        return None
    try:
        data = base64.b64decode(vector, validate=True)
    except ValueError:
        return None
    if not data or len(data) % 4 or len(data) > 4 * _MAX_DIMENSIONS:
        return None
    packed = array("f")
    packed.frombytes(data)
    if sys.byteorder != "little":
        packed.byteswap()
    numbers = packed.tolist()
    if not all(math.isfinite(number) for number in numbers):
        return None
    return trace_id, text_digest, numbers
