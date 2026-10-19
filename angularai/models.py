from dataclasses import dataclass
from typing import Literal

SubstrateKind = Literal["openai", "anthropic", "compatible"]
Outcome = Literal["ok", "held", "error"]
TraceState = Literal["current", "superseded"]
# asked: this cortex put the input to the peer, and the reply is the peer's.
# answered: the input came from the peer, and the reply is this cortex's.
# opened: the input came from the operator, and the reply went to the peer.
PeerRole = Literal["asked", "answered", "opened"]


@dataclass(frozen=True)
class Peer:
    """The other cortex in a pass that crossed between two. name is a valid cortex name."""

    name: str
    role: PeerRole


@dataclass(frozen=True)
class Config:
    substrate: SubstrateKind | None
    model: str | None
    base_url: str | None
    workspace_budget: int
    max_input_tokens: int
    forbid: tuple[str, ...]
    json_mode: bool


@dataclass(frozen=True)
class TraceDraft:
    text: str
    wavelengths: tuple[str, ...]


@dataclass(frozen=True)
class Action:
    tool: str
    path: str = ""
    query: str = ""
    old: str = ""
    new: str = ""
    text: str = ""
    command: str = ""
    start: int = 1
    overwrite: bool = False
    # Why parse refused this act. A refused act is never run; the model is told why.
    problem: str = ""


@dataclass(frozen=True)
class Integration:
    reply: str
    remember: tuple[TraceDraft, ...]
    supersede: tuple[str, ...]
    act: Action | None = None


@dataclass(frozen=True)
class Trace:
    id: str
    text: str
    wavelengths: tuple[str, ...]
    source: str
    at: str
    state: TraceState
    superseded_by: str | None


@dataclass(frozen=True)
class Episode:
    at: str
    user: str
    reply: str | None
    outcome: Outcome
    peer: Peer | None = None


@dataclass(frozen=True)
class CortexView:
    traces: tuple[Trace, ...]
    episodes: tuple[Episode, ...]

    def current_traces(self) -> tuple[Trace, ...]:
        return tuple(trace for trace in self.traces if trace.state == "current")


@dataclass(frozen=True)
class Workspace:
    system: str
    user: str
    input_text: str
    trace_ids: tuple[str, ...]
    estimated_tokens: int
    budget: int
    traces_shown: int
    traces_held: int
    episodes_shown: int
    episodes_held: int


@dataclass(frozen=True)
class PassRecord:
    at: str
    outcome: Outcome
    user: str
    reply: str | None
    remember: tuple[Trace, ...]
    supersede: tuple[tuple[str, str], ...]
    error: str | None
    budget: int
    used: int
    traces_shown: int
    traces_held: int
    episodes_shown: int
    episodes_held: int
    peer: Peer | None = None


@dataclass(frozen=True)
class TurnResult:
    reply: str
    held: bool
    used: int
    budget: int
    act: Action | None = None
