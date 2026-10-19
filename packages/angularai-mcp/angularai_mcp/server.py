"""The AngularAI MCP server: one cortex, five tools, two resources.

Every tool is a thin call into the angularai library. Secrets, the value lock,
the budget, durable writes, and the file lock all live there.
"""

import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError, ToolError
from mcp.types import CallToolResult, InputRequiredResult, ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from angularai.budget import estimate_tokens
from angularai.errors import AngularAIError, InputTooLong, KeyMissing, NoCortex, SecretRefused
from angularai.models import Trace
from angularai.names import cortex_home, label, list_names
from angularai.notes import MAX_NOTE, MAX_WAVELENGTHS, note, recall_within
from angularai.notes import forget as forget_trace
from angularai.parse import valid_trace_id
from angularai.secrets import contains_secret, environment_secrets, redact
from angularai.semantic import (
    DEFAULT_MODEL,
    VECTORS,
    SemanticRecall,
    embedder_from_env,
    minimum_from_env,
)
from angularai.store import Store, open_store
from angularai_mcp import __version__
from angularai_mcp.errors import CortexToolError, tool_error

NAME = "AngularAI"
SOURCE = "mcp"
MAX_QUERY = 2000
MAX_LIMIT = 50
INSTRUCTIONS = (
    "AngularAI is a memory on disk. remember keeps one short fact. recall finds facts "
    "for a query, best first, inside a token budget, with where each came from. "
    "forget removes one fact by its exact id. status shows the budget and the counts. "
    "Do not remember secrets, keys, or passwords: they are refused."
)
logger = logging.getLogger("angularai_mcp")
T = TypeVar("T")


class TraceOut(BaseModel):
    """One trace with its provenance."""

    id: str = Field(description="The exact id. forget takes this.")
    text: str
    tags: list[str] = Field(description="Wavelengths: lowercase slugs.")
    source: str = Field(description="What wrote it: 'mcp' for a note, else the input it came from.")
    at: str = Field(description="When it was written, ISO 8601 UTC with a short suffix.")
    state: Literal["current", "superseded"]
    superseded_by: str | None = Field(description="When a later pass superseded it.")


class Remembered(BaseModel):
    cortex: str
    created: bool = Field(description="False when the same fact was already kept.")
    trace: TraceOut


class Hit(BaseModel):
    trace: TraceOut
    score: float = Field(description="Lexical overlap 0-1, or cosine when match is meaning.")
    match: Literal["lexical", "meaning"]
    tokens: int = Field(description="Estimated tokens of the text.")


class Recall(BaseModel):
    cortex: str
    mode: Literal["lexical", "lexical+meaning"]
    results: list[Hit]
    matched: int = Field(description="Traces that matched before the limit and the budget.")
    used_tokens: int
    max_tokens: int
    workspace_budget: int
    note: str | None = Field(description="Why recall by meaning was not used, when it was on.")


class Forgotten(BaseModel):
    cortex: str
    forgotten: TraceOut
    vectors_removed: int = Field(description="Lines removed from embeddings.jsonl.")


class LastPass(BaseModel):
    at: str
    outcome: Literal["ok", "held", "error"]
    used: int
    budget: int


class Embeddings(BaseModel):
    mode: Literal["off", "openai", "compatible", "invalid"]
    model: str | None
    minimum: float | None
    vectors_file: bool


class Status(BaseModel):
    cortex: str
    path: str
    substrate: str | None
    model: str | None
    workspace_budget: int = Field(description="Estimated tokens a workspace may use.")
    max_input_tokens: int
    traces_current: int
    traces_superseded: int
    episodes: int
    memory_tokens: int = Field(description="Estimated tokens of every current trace text.")
    memory_fits_budget: bool
    locks: int = Field(description="How many value-lock phrases are set. The phrases stay out.")
    damaged_lines: int
    last_pass: LastPass | None
    embeddings: Embeddings
    server_version: str


class Cortexes(BaseModel):
    active: str
    home: str
    names: list[str] = Field(description="Named cortexes in the home. The server uses only active.")


class Memory:
    """The one cortex this server serves. Calls are serialized in-process; the file
    lock in angularai keeps other processes apart."""

    def __init__(self, root: Path, environ: Mapping[str, str]) -> None:
        self.root = root
        self.name = label(root)
        self._environ = dict(environ)
        self.secrets = environment_secrets(self._environ)
        self._lock = threading.Lock()
        self._store: Store | None = None

    @property
    def environ(self) -> Mapping[str, str]:
        return self._environ

    def guard(self, work: Callable[[], T]) -> T:
        try:
            with self._lock:
                return work()
        except CortexToolError:
            raise
        except Exception as exc:
            raise tool_error(exc, self.secrets) from None

    def store(self) -> Store:
        if self._store is None or not (self.root / "cortex.json").is_file():
            self._store = None
            try:
                self._store = open_store(self.root, warn=self._warn)
            except NoCortex:
                raise NoCortex(
                    f"No cortex at {self.root}. Run angularai init, "
                    "or start angularai-mcp with --init."
                ) from None
        return self._store

    def semantic(self, store: Store) -> tuple[SemanticRecall | None, str | None]:
        minimum = minimum_from_env(self._environ)
        try:
            embedder = embedder_from_env(store.config(), self._environ)
        except KeyMissing as exc:
            return None, f"Embeddings are unavailable: {exc} Recall is lexical."
        if embedder is None:
            return None, None
        return (
            SemanticRecall(
                self.root, embedder, minimum=minimum, secrets=self.secrets, warn=self._warn
            ),
            None,
        )

    def embeddings(self) -> Embeddings:
        raw = self._environ.get("ANGULARAI_EMBEDDINGS", "").strip().lower() or "off"
        exists = (self.root / VECTORS).is_file()
        if raw == "off":
            return Embeddings(mode="off", model=None, minimum=None, vectors_file=exists)
        if raw not in ("openai", "compatible"):
            return Embeddings(mode="invalid", model=None, minimum=None, vectors_file=exists)
        model = self._environ.get("ANGULARAI_EMBEDDINGS_MODEL", "").strip() or DEFAULT_MODEL
        try:
            minimum: float | None = minimum_from_env(self._environ)
        except AngularAIError:
            minimum = None
        mode: Literal["openai", "compatible"] = "openai" if raw == "openai" else "compatible"
        return Embeddings(mode=mode, model=model, minimum=minimum, vectors_file=exists)

    def out(self, trace: Trace) -> TraceOut:
        """A trace on its way out, redacted once more against the environment."""

        return TraceOut(
            id=trace.id,
            text=redact(trace.text, self.secrets),
            tags=list(trace.wavelengths),
            source=redact(trace.source, self.secrets),
            at=trace.at,
            state=trace.state,
            superseded_by=trace.superseded_by,
        )

    def _warn(self, notice: str) -> None:
        logger.warning("%s", redact(notice, self.secrets))


class AngularAIServer(MCPServer):
    """MCPServer whose argument errors are typed and never echo what was sent."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context[Any, Any] | None = None
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exc:
            cause = exc.__cause__
            if isinstance(cause, CortexToolError):
                raise cause from None
            if isinstance(cause, ValidationError):
                fields = sorted({".".join(str(p) for p in e["loc"]) for e in cause.errors()})
                listed = ", ".join(fields) or "arguments"
                raise CortexToolError(
                    "invalid_input",
                    f"The arguments do not match the input schema of {name}: {listed}.",
                ) from None
            raise


def build_server(root: Path, environ: Mapping[str, str]) -> AngularAIServer:
    """The server for the cortex at root. environ is read once, here."""

    memory = Memory(root, environ)
    server = AngularAIServer(
        NAME,
        title=NAME,
        description="AngularAI's cortex: a memory on disk in front of the model you call.",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/angularai-hq/AngularAI",
        version=__version__,
        log_level="WARNING",
    )

    @server.tool(
        name="remember",
        title="Remember a fact",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    def remember(
        text: Annotated[
            str,
            Field(
                description=(
                    f"One durable fact, in one line of at most {MAX_NOTE} characters. "
                    "Write it so it stands alone: 'The user's notebook is blue.'"
                ),
                json_schema_extra={"minLength": 1, "maxLength": MAX_NOTE},
            ),
        ],
        tags: Annotated[
            list[str],
            Field(
                description=(
                    f"Up to {MAX_WAVELENGTHS} topic slugs: a lowercase letter, then letters, "
                    "digits, or -. They help recall."
                ),
                json_schema_extra={"maxItems": MAX_WAVELENGTHS},
            ),
        ] = [],
    ) -> Remembered:
        """Keep one fact in the AngularAI cortex, on disk, for later recall.

        The fact is refused, and nothing is written, when it looks like a secret (an API
        key, a token, a password), when a value-lock phrase of the cortex is in it, or
        when it is over the length cap. The same fact twice is kept once.
        """

        def work() -> Remembered:
            store = memory.store()
            recall, _ = memory.semantic(store)
            noted = note(store, text, tags, memory.secrets, SOURCE, recall)
            return Remembered(
                cortex=memory.name, created=noted.created, trace=memory.out(noted.trace)
            )

        return memory.guard(work)

    @server.tool(
        name="recall",
        title="Recall facts",
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def recall(
        query: Annotated[
            str,
            Field(
                description=f"What to look for, in words. At most {MAX_QUERY} characters.",
                json_schema_extra={"minLength": 1, "maxLength": MAX_QUERY},
            ),
        ],
        limit: Annotated[
            int,
            Field(
                description=f"Most results, 1 to {MAX_LIMIT}.",
                json_schema_extra={"minimum": 1, "maximum": MAX_LIMIT},
            ),
        ] = 10,
        max_tokens: Annotated[
            int | None,
            Field(
                description=(
                    "Most estimated tokens of result text. Default and ceiling: the "
                    "cortex's workspace budget."
                ),
                json_schema_extra={"minimum": 1},
            ),
        ] = None,
        lexical_only: Annotated[
            bool,
            Field(description="Match words only, even when recall by meaning is on."),
        ] = False,
        include_superseded: Annotated[
            bool,
            Field(description="Also return facts a later pass superseded."),
        ] = False,
    ) -> Recall:
        """Find facts in the AngularAI cortex for a query, best first.

        A fact matches when it shares a word with the query, or, when the server has
        ANGULARAI_EMBEDDINGS on, when it is close in meaning. Results stop at limit and
        at max_tokens estimated tokens (4 ASCII characters to one). Each result carries
        its id, tags, source, time, and score. Nothing is written.
        """

        def work() -> Recall:
            cleaned = query.strip()
            if not cleaned:
                raise AngularAIError("recall needs a query.")
            if len(cleaned) > MAX_QUERY:
                raise InputTooLong(f"A query is at most {MAX_QUERY} characters.")
            if not 1 <= limit <= MAX_LIMIT:
                raise AngularAIError(f"limit must be from 1 to {MAX_LIMIT}.")
            if contains_secret(cleaned, memory.secrets):
                raise SecretRefused(
                    "That query looks like a secret. It was not used and nothing was written."
                )
            store = memory.store()
            config = store.config()
            budget = config.workspace_budget
            cap = budget if max_tokens is None else max_tokens
            if not 1 <= cap <= budget:
                raise AngularAIError(f"max_tokens must be from 1 to the budget, {budget}.")
            view = store.view()
            semantic, notice = (None, None) if lexical_only else memory.semantic(store)
            meaning = None if semantic is None else semantic.ranks(view, cleaned)
            if semantic is not None and semantic.failure is not None:
                notice = f"Embeddings are unavailable: {semantic.failure} Recall is lexical."
            found = recall_within(
                view, cleaned, cap, limit, meaning, include_superseded=include_superseded
            )
            return Recall(
                cortex=memory.name,
                mode="lexical" if meaning is None or notice else "lexical+meaning",
                results=[
                    Hit(
                        trace=memory.out(item.trace),
                        score=round(item.score, 4),
                        match=item.match,
                        tokens=estimate_tokens(item.trace.text),
                    )
                    for item in found.found
                ],
                matched=found.matched,
                used_tokens=found.used,
                max_tokens=cap,
                workspace_budget=budget,
                note=None if notice is None else redact(notice, memory.secrets),
            )

        return memory.guard(work)

    @server.tool(
        name="forget",
        title="Forget one fact",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
        ),
        # Claude Code asks a person on every call of a tool marked so. Others ignore it.
        meta={"anthropic/requiresUserInteraction": True},
    )
    def forget(
        id: Annotated[
            str,
            Field(
                description=(
                    "The exact trace id, as recall or remember returned it: t_ and 12 "
                    "lowercase hex digits. No prefix, pattern, or text is accepted."
                ),
                json_schema_extra={"pattern": "^t_[0-9a-f]{12}$"},
            ),
        ],
    ) -> Forgotten:
        """Remove one fact from the AngularAI cortex for good, by its exact id.

        The fact leaves the log on disk and its vectors leave embeddings.jsonl. This
        cannot be undone. Call recall first and pass the id it shows; an unknown or
        malformed id removes nothing and returns not_found. A conversation episode that
        mentions the fact in its own words is kept.
        """

        def work() -> Forgotten:
            gone = forget_trace(memory.store(), id)
            return Forgotten(
                cortex=memory.name, forgotten=memory.out(gone.trace), vectors_removed=gone.vectors
            )

        return memory.guard(work)

    def status_now() -> Status:
        store = memory.store()
        config = store.config()
        view = store.view()
        current = view.current_traces()
        last = store.last_pass()
        tokens = sum(estimate_tokens(trace.text) for trace in current)
        return Status(
            cortex=memory.name,
            path=str(memory.root),
            substrate=config.substrate,
            model=config.model,
            workspace_budget=config.workspace_budget,
            max_input_tokens=config.max_input_tokens,
            traces_current=len(current),
            traces_superseded=len(view.traces) - len(current),
            episodes=len(view.episodes),
            memory_tokens=tokens,
            memory_fits_budget=tokens <= config.workspace_budget,
            locks=len(config.forbid),
            damaged_lines=len(store.damage()),
            last_pass=None
            if last is None
            else LastPass(at=last.at, outcome=last.outcome, used=last.used, budget=last.budget),
            embeddings=memory.embeddings(),
            server_version=__version__,
        )

    @server.tool(
        name="status",
        title="Cortex status",
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def status() -> Status:
        """The AngularAI cortex this server uses: its name, budget, how much memory it
        holds against that budget, trace and episode counts, the last pass, and whether
        recall by meaning is on. No key or environment value is ever returned."""

        return memory.guard(status_now)

    @server.tool(
        name="cortexes",
        title="List cortexes",
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    )
    def cortexes() -> Cortexes:
        """The named AngularAI cortexes in the home directory, and the one this server
        uses. Another cortex is served by another angularai-mcp started with --name."""

        def work() -> Cortexes:
            return Cortexes(
                active=memory.name,
                home=str(cortex_home(memory.environ)),
                names=list(list_names(memory.environ)),
            )

        return memory.guard(work)

    @server.resource(
        "angularai://status",
        name="status",
        title="Cortex status",
        description="The status tool's answer, as JSON.",
        mime_type="application/json",
    )
    def status_resource() -> str:
        try:
            return memory.guard(status_now).model_dump_json(indent=2)
        except ToolError as exc:
            raise ResourceError(str(exc)) from None

    @server.resource(
        "angularai://trace/{trace_id}",
        name="trace",
        title="One trace",
        description="One trace by its exact id, with its provenance, as JSON.",
        mime_type="application/json",
    )
    def trace_resource(trace_id: str) -> str:
        if not valid_trace_id(trace_id):
            raise ResourceNotFoundError("[not_found] A trace id is t_ and 12 lowercase hex digits.")

        def work() -> TraceOut | None:
            for trace in memory.store().view().traces:
                if trace.id == trace_id:
                    return memory.out(trace)
            return None

        try:
            found = memory.guard(work)
        except ToolError as exc:
            raise ResourceError(str(exc)) from None
        if found is None:
            raise ResourceNotFoundError("[not_found] No trace in this cortex has that id.")
        return found.model_dump_json(indent=2)

    return server
