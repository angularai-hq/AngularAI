"""The cortex on disk: cortex.json and passes.jsonl.

Durability, repair, and versioning are documented in the README under "On disk".
Writes go through angularai.durable: atomic replace for whole files, one locked,
fsynced line per pass for the log.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from angularai.durable import append_line, atomic_write, fsync_dir, locked
from angularai.errors import ConfigError, CortexExists, LogDamaged, NoCortex
from angularai.models import (
    Config,
    CortexView,
    Episode,
    Outcome,
    PassRecord,
    Peer,
    PeerRole,
    SubstrateKind,
    Trace,
    TraceState,
)
from angularai.names import valid_name
from angularai.urls import validate_base_url

# The layout this code writes. A cortex without a version field is version 1.
SCHEMA_VERSION = 1
Migration = Callable[[dict[str, object]], dict[str, object]]
# MIGRATIONS[n] turns a version-n config into version n + 1. Version 1 is current,
# so there is nothing to run yet. migrate() is the hook every open goes through.
MIGRATIONS: dict[int, Migration] = {}

LOG = "passes.jsonl"
CORRUPT = "passes.jsonl.corrupt"
_CONFIG_FIELDS = frozenset(
    {
        "substrate",
        "model",
        "base_url",
        "workspace_budget",
        "max_input_tokens",
        "forbid",
        "json_mode",
    }
)
_OPTIONAL_FIELDS = frozenset({"version"})
_OUTCOMES = frozenset({"ok", "held", "error"})
_MIN_BUDGET = 256
_MAX_BUDGET = 32000

Warn = Callable[[str], None]


@dataclass(frozen=True)
class Damage:
    line: int
    kind: Literal["unreadable", "torn"]


@dataclass(frozen=True)
class Repair:
    moved: int
    kept: int


def init_cortex(root: Path) -> None:
    if root.exists():
        if not root.is_dir():
            raise CortexExists(f"{root} is not a directory.")
        if any(root.iterdir()):
            raise CortexExists(f"A cortex already lives at {root}.")
    else:
        root.mkdir(parents=True, mode=0o700)
    if root.parent.name == ".angularai":
        root.parent.chmod(0o700)
    root.chmod(0o700)
    write_config(root / "cortex.json", default_config())
    log = root / LOG
    log.touch()
    log.chmod(0o600)
    fsync_dir(root)


def open_store(root: Path, warn: Warn | None = None) -> "Store":
    path = root / "cortex.json"
    if not path.is_file():
        raise NoCortex(f"No cortex at {root}. Run angularai init.")
    _upgrade(root, path)
    return Store(root, warn)


def default_config() -> Config:
    return Config(
        substrate=None,
        model=None,
        base_url=None,
        workspace_budget=2000,
        max_input_tokens=800,
        forbid=(),
        json_mode=False,
    )


def write_config(path: Path, config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_config_json(config), indent=2) + "\n"
    atomic_write(path, text.encode("utf-8"))


def read_config(path: Path) -> Config:
    return _config_from(migrate(_read_payload(path))[0])


def migrate(payload: dict[object, object]) -> tuple[dict[object, object], int]:
    """Bring a config payload to SCHEMA_VERSION. Returns it and the version it had."""

    found = payload.get("version", 1)
    if isinstance(found, bool) or not isinstance(found, int) or found < 1:
        raise ConfigError("The cortex config field version must be a whole number from 1.")
    if found > SCHEMA_VERSION:
        raise ConfigError(
            f"This cortex is version {found}. This AngularAI reads up to version "
            f"{SCHEMA_VERSION}. Upgrade angularai. The cortex was not modified."
        )
    current: dict[object, object] = dict(payload)
    version = found
    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ConfigError(f"No migration from cortex version {version}.")
        upgraded = step({str(key): value for key, value in current.items()})
        current = {}
        current.update(upgraded)
        version += 1
    current["version"] = SCHEMA_VERSION
    return current, found


def _upgrade(root: Path, path: Path) -> None:
    """Write a migrated config back. The old file is kept as cortex.json.v<n>."""

    payload = _read_payload(path)
    migrated, found = migrate(payload)
    if found == SCHEMA_VERSION:
        return
    _config_from(migrated)
    with locked(root):
        raw = path.read_bytes()
        atomic_write(root / f"cortex.json.v{found}", raw)
        text = json.dumps({str(k): v for k, v in migrated.items()}, indent=2) + "\n"
        atomic_write(path, text.encode("utf-8"))


def _read_payload(path: Path) -> dict[object, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"The cortex config at {path} cannot be read.") from exc
    try:
        payload: object = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ConfigError("The cortex config is not valid JSON. It was not modified.") from exc
    if not isinstance(payload, dict):
        raise ConfigError("The cortex config is not an object. It was not modified.")
    unknown = set(payload) - _CONFIG_FIELDS - _OPTIONAL_FIELDS
    if unknown:
        name = sorted(str(item) for item in unknown)[0]
        raise ConfigError(f"The cortex config has an unknown field: {name}.")
    missing = _CONFIG_FIELDS - set(payload)
    if missing:
        name = sorted(missing)[0]
        raise ConfigError(f"The cortex config is missing {name}.")
    return payload


class Store:
    def __init__(self, root: Path, warn: Warn | None = None) -> None:
        self.root = root
        self._warn = warn
        self._warned = False
        self._offset = 0
        self._lines = 0
        self._digest = _digest(b"")
        self._records: list[PassRecord] = []
        self._damage: list[Damage] = []
        self._torn: Damage | None = None

    def config(self) -> Config:
        return read_config(self.root / "cortex.json")

    def save_config(self, config: Config) -> None:
        with locked(self.root):
            write_config(self.root / "cortex.json", config)

    def passes(self) -> tuple[PassRecord, ...]:
        """Every readable pass. A bad line is skipped, kept on disk, and reported once."""

        with locked(self.root, shared=True):
            self._refresh()
        if self._warn is not None and not self._warned and self.damage():
            self._warned = True
            self._warn(damage_notice(self.damage()))
        return tuple(self._records)

    def damage(self) -> tuple[Damage, ...]:
        torn = () if self._torn is None else (self._torn,)
        return (*self._damage, *torn)

    def view(self) -> CortexView:
        return fold(self.passes())

    def last_pass(self) -> PassRecord | None:
        records = self.passes()
        if not records:
            return None
        return records[-1]

    def append_pass(self, record: PassRecord) -> None:
        line = json.dumps(_pass_json(record), ensure_ascii=False)
        with locked(self.root):
            append_line(self.root / LOG, line.encode("utf-8"))

    def repair(self) -> Repair:
        """Move every unreadable line to passes.jsonl.corrupt. Nothing is deleted."""

        path = self.root / LOG
        with locked(self.root):
            raw = _read_log(path)
            good: list[bytes] = []
            bad: list[bytes] = []
            for number, line in enumerate(raw.split(b"\n"), start=1):
                if not line:
                    continue
                if not line.strip():
                    good.append(line)  # whitespace is skipped on read, and kept
                elif _parse(line, number) is None:
                    bad.append(line)
                else:
                    good.append(line)
            if bad:
                for line in bad:
                    append_line(self.root / CORRUPT, line)
                atomic_write(path, b"".join(line + b"\n" for line in good))
        self._reset()
        self._warned = False
        return Repair(moved=len(bad), kept=sum(1 for line in good if line.strip()))

    def _reset(self) -> None:
        self._offset = 0
        self._lines = 0
        self._digest = _digest(b"")
        self._records = []
        self._damage = []
        self._torn = None

    def _refresh(self) -> None:
        """Parse only what was appended since the last read.

        The whole file is read, but only new lines are parsed. The bytes already
        parsed are checked by digest, so a log rewritten or edited in place is
        parsed again from the start.
        """

        data = _read_log(self.root / LOG)
        known = self._offset
        if len(data) < known or _digest(data[:known]) != self._digest:
            self._reset()
            known = 0
        fresh = data[known:]
        end = fresh.rfind(b"\n") + 1
        for raw in fresh[:end].split(b"\n")[:-1]:
            self._lines += 1
            if not raw.strip():
                continue
            parsed = _parse(raw, self._lines)
            if parsed is None:
                self._damage.append(Damage(self._lines, "unreadable"))
            else:
                self._records.append(parsed)
        self._offset = known + end
        self._digest = _digest(data[: self._offset])
        tail = fresh[end:]
        self._torn = Damage(self._lines + 1, "torn") if tail.strip() else None


def damage_notice(damage: tuple[Damage, ...]) -> str:
    places = ", ".join(f"{LOG}:{item.line}" for item in damage[:5])
    more = f" and {len(damage) - 5} more" if len(damage) > 5 else ""
    torn = " The last line was cut off mid-write." if any(d.kind == "torn" for d in damage) else ""
    count = len(damage)
    noun = "line" if count == 1 else "lines"
    return (
        f"The cortex log has {count} unreadable {noun}: {places}{more}.{torn} "
        f"They are skipped and still on disk. angularai repair moves them to {CORRUPT}."
    )


def _read_log(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LogDamaged(f"The cortex log at {path} cannot be read.") from exc


def _digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def _parse(line: bytes, number: int) -> PassRecord | None:
    try:
        return _pass_from_line(line.decode("utf-8"), number)
    except (LogDamaged, UnicodeDecodeError):
        return None


def fold(records: tuple[PassRecord, ...]) -> CortexView:
    traces: dict[str, Trace] = {}
    order: list[str] = []
    episodes: list[Episode] = []
    for record in records:
        episodes.append(Episode(record.at, record.user, record.reply, record.outcome, record.peer))
        for old_id, by in record.supersede:
            previous = traces.get(old_id)
            if previous is None or previous.state != "current":
                continue
            traces[old_id] = Trace(
                id=previous.id,
                text=previous.text,
                wavelengths=previous.wavelengths,
                source=previous.source,
                at=previous.at,
                state="superseded",
                superseded_by=by,
            )
        for trace in record.remember:
            if trace.id not in traces:
                order.append(trace.id)
            traces[trace.id] = trace
    return CortexView(
        traces=tuple(traces[trace_id] for trace_id in order if trace_id in traces),
        episodes=tuple(episodes),
    )


def _config_json(config: Config) -> dict[str, object]:
    return {
        "version": SCHEMA_VERSION,
        "substrate": config.substrate,
        "model": config.model,
        "base_url": config.base_url,
        "workspace_budget": config.workspace_budget,
        "max_input_tokens": config.max_input_tokens,
        "forbid": list(config.forbid),
        "json_mode": config.json_mode,
    }


def _config_from(payload: dict[object, object]) -> Config:
    substrate = _substrate(payload.get("substrate"))
    model = _optional_text(payload.get("model"), "model", 128)
    base_url = _base_url(payload.get("base_url"))
    budget = _bounded_int(
        payload.get("workspace_budget"), "workspace_budget", _MIN_BUDGET, _MAX_BUDGET
    )
    max_input = _bounded_int(payload.get("max_input_tokens"), "max_input_tokens", 32, budget)
    forbid = _forbid(payload.get("forbid"))
    json_mode = payload.get("json_mode")
    if not isinstance(json_mode, bool):
        raise ConfigError("The cortex config field json_mode must be true or false.")
    if substrate is None and model is not None:
        raise ConfigError("The cortex config has a model without a substrate.")
    if substrate is not None and model is None:
        raise ConfigError("The cortex config has a substrate without a model.")
    if substrate == "compatible" and base_url is None:
        raise ConfigError("A compatible substrate needs base_url.")
    return Config(substrate, model, base_url, budget, max_input, forbid, json_mode)


def _substrate(value: object) -> SubstrateKind | None:
    if value is None:
        return None
    if value == "openai" or value == "anthropic" or value == "compatible":
        return value
    raise ConfigError("The cortex config field substrate is not a known API.")


def _base_url(value: object) -> str | None:
    text = _optional_text(value, "base_url", 300)
    if text is None:
        return None
    try:
        return validate_base_url(text)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _optional_text(value: object, field: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ConfigError(f"The cortex config field {field} is not valid.")
    if any(character.isspace() for character in value):
        raise ConfigError(f"The cortex config field {field} cannot contain spaces.")
    return value


def _bounded_int(value: object, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ConfigError(f"The cortex config field {field} must be between {low} and {high}.")
    return value


def _forbid(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError("The cortex config field forbid must be a list.")
    if len(value) > 64:
        raise ConfigError("The cortex config has more than 64 locks.")
    phrases: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ConfigError("A lock phrase must be text.")
        phrase = " ".join(item.split())
        if not phrase or len(phrase) > 200:
            raise ConfigError("A lock phrase is empty or too long.")
        key = phrase.lower()
        if key in seen:
            raise ConfigError(f"The lock {phrase} is repeated.")
        seen.add(key)
        phrases.append(phrase)
    return tuple(phrases)


def _pass_json(record: PassRecord) -> dict[str, object]:
    payload = _base_json(record)
    if record.peer is not None:
        # Only a pass that crossed carries it, so a log of single passes is unchanged.
        payload["peer"] = {"name": record.peer.name, "role": record.peer.role}
    return payload


def _base_json(record: PassRecord) -> dict[str, object]:
    return {
        "at": record.at,
        "outcome": record.outcome,
        "user": record.user,
        "reply": record.reply,
        "remember": [
            {
                "id": trace.id,
                "text": trace.text,
                "wavelengths": list(trace.wavelengths),
                "source": trace.source,
                "at": trace.at,
            }
            for trace in record.remember
        ],
        "supersede": [{"id": old_id, "by": by} for old_id, by in record.supersede],
        "error": record.error,
        "budget": record.budget,
        "used": record.used,
        "traces_shown": record.traces_shown,
        "traces_held": record.traces_held,
        "episodes_shown": record.episodes_shown,
        "episodes_held": record.episodes_held,
    }


def _pass_from_line(line: str, number: int) -> PassRecord:
    try:
        payload: object = json.loads(line)
    except (ValueError, RecursionError) as exc:
        raise LogDamaged(f"The cortex log is damaged at line {number}.") from exc
    if not isinstance(payload, dict):
        raise LogDamaged(f"The cortex log is damaged at line {number}.")
    try:
        return _pass_from(payload)
    except (ConfigError, ValueError) as exc:
        raise LogDamaged(f"The cortex log is damaged at line {number}.") from exc


def _pass_from(payload: dict[object, object]) -> PassRecord:
    outcome = payload.get("outcome")
    if outcome not in _OUTCOMES:
        raise ValueError("outcome")
    at = _required_text(payload.get("at"), 40)
    user = _required_text(payload.get("user"), 20000)
    reply = _nullable_text(payload.get("reply"), 8000)
    error = _nullable_text(payload.get("error"), 300)
    remember = tuple(_trace_from(item) for item in _list(payload.get("remember")))
    supersede = tuple(_pair_from(item) for item in _list(payload.get("supersede")))
    if not isinstance(outcome, str):
        raise ValueError("outcome")
    return PassRecord(
        at=at,
        outcome=_outcome(outcome),
        user=user,
        reply=reply,
        remember=remember,
        supersede=supersede,
        error=error,
        budget=_stat(payload.get("budget")),
        used=_stat(payload.get("used")),
        traces_shown=_stat(payload.get("traces_shown")),
        traces_held=_stat(payload.get("traces_held")),
        episodes_shown=_stat(payload.get("episodes_shown")),
        episodes_held=_stat(payload.get("episodes_held")),
        peer=_peer_from(payload.get("peer")) if "peer" in payload else None,
    )


def _peer_from(value: object) -> Peer:
    if not isinstance(value, dict) or set(value) != {"name", "role"}:
        raise ValueError("peer")
    name = value.get("name")
    if not isinstance(name, str) or not valid_name(name):
        raise ValueError("peer")
    return Peer(name=name, role=_role(value.get("role")))


def _role(value: object) -> PeerRole:
    if value == "asked":
        return "asked"
    if value == "answered":
        return "answered"
    if value == "opened":
        return "opened"
    raise ValueError("role")


def _outcome(value: str) -> Outcome:
    if value == "ok":
        return "ok"
    if value == "held":
        return "held"
    if value == "error":
        return "error"
    raise ValueError("outcome")


def _trace_from(value: object) -> Trace:
    if not isinstance(value, dict):
        raise ValueError("trace")
    wavelengths = value.get("wavelengths")
    if not isinstance(wavelengths, list) or not all(isinstance(item, str) for item in wavelengths):
        raise ValueError("wavelengths")
    state: TraceState = "current"
    return Trace(
        id=_required_text(value.get("id"), 20),
        text=_required_text(value.get("text"), 240),
        wavelengths=tuple(wavelengths),
        source=_required_text(value.get("source"), 500),
        at=_required_text(value.get("at"), 40),
        state=state,
        superseded_by=None,
    )


def _pair_from(value: object) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise ValueError("supersede")
    return (_required_text(value.get("id"), 20), _required_text(value.get("by"), 40))


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("list")
    return list(value)


def _required_text(value: object, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError("text")
    return value


def _nullable_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return _required_text(value, limit)


def _stat(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 1_000_000:
        raise ValueError("stat")
    return value
