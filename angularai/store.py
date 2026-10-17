import json
import os
from pathlib import Path

from angularai.errors import ConfigError, CortexExists, LogDamaged, NoCortex
from angularai.models import (
    Config,
    CortexView,
    Episode,
    Outcome,
    PassRecord,
    SubstrateKind,
    Trace,
    TraceState,
)
from angularai.urls import validate_base_url

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
_OUTCOMES = frozenset({"ok", "held", "error"})
_MIN_BUDGET = 256
_MAX_BUDGET = 32000


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
    log = root / "passes.jsonl"
    log.touch()
    log.chmod(0o600)


def open_store(root: Path) -> "Store":
    if not (root / "cortex.json").is_file():
        raise NoCortex(f"No cortex at {root}. Run angularai init.")
    return Store(root)


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
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_config_json(config), indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def read_config(path: Path) -> Config:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"The cortex config at {path} cannot be read.") from exc
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError("The cortex config is not valid JSON. It was not modified.") from exc
    if not isinstance(payload, dict):
        raise ConfigError("The cortex config is not an object. It was not modified.")
    unknown = set(payload) - _CONFIG_FIELDS
    if unknown:
        name = sorted(str(item) for item in unknown)[0]
        raise ConfigError(f"The cortex config has an unknown field: {name}.")
    missing = _CONFIG_FIELDS - set(payload)
    if missing:
        name = sorted(missing)[0]
        raise ConfigError(f"The cortex config is missing {name}.")
    return _config_from(payload)


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root

    def config(self) -> Config:
        return read_config(self.root / "cortex.json")

    def save_config(self, config: Config) -> None:
        write_config(self.root / "cortex.json", config)

    def passes(self) -> tuple[PassRecord, ...]:
        path = self.root / "passes.jsonl"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LogDamaged(f"The cortex log at {path} cannot be read.") from exc
        records: list[PassRecord] = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            records.append(_pass_from_line(line, number))
        return tuple(records)

    def view(self) -> CortexView:
        return fold(self.passes())

    def last_pass(self) -> PassRecord | None:
        records = self.passes()
        if not records:
            return None
        return records[-1]

    def append_pass(self, record: PassRecord) -> None:
        path = self.root / "passes.jsonl"
        line = json.dumps(_pass_json(record), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o600)


def fold(records: tuple[PassRecord, ...]) -> CortexView:
    traces: dict[str, Trace] = {}
    order: list[str] = []
    episodes: list[Episode] = []
    for record in records:
        episodes.append(Episode(record.at, record.user, record.reply, record.outcome))
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
    except json.JSONDecodeError as exc:
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
    )


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
