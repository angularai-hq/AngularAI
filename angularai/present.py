from collections.abc import Mapping
from pathlib import Path

from angularai.attention import score
from angularai.models import Config, CortexView, PassRecord, Trace


def format_banner(config: Config, view: CortexView) -> str:
    kind = config.substrate or "unwired"
    model = config.model or "no model"
    return (
        f"angularai  {kind}  {model}  budget {config.workspace_budget}  "
        f"traces {len(view.current_traces())}"
    )


def format_traces(view: CortexView) -> str:
    current = view.current_traces()
    if not current:
        return "No traces yet. What you say can be kept after the next pass."
    return "\n".join(_trace_line(trace) for trace in current)


def format_recall(
    view: CortexView,
    query: str,
    limit: int = 20,
    meaning: Mapping[str, float] | None = None,
) -> str:
    """Lexical ranks as 0.50. A trace found only by meaning shows its cosine as ~0.64."""

    similar = meaning or {}
    ranked: list[tuple[float, str, Trace]] = []
    for trace in view.current_traces():
        document = " ".join((trace.text, " ".join(trace.wavelengths), trace.source))
        rank = score(query, document)
        if rank > 0:
            ranked.append((rank, f"{rank:.2f}", trace))
        elif trace.id in similar:
            ranked.append((similar[trace.id], f"~{similar[trace.id]:.2f}", trace))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return "Nothing in the cortex matches that."
    lines: list[str] = []
    for _, shown, trace in ranked[:limit]:
        waves = ",".join(trace.wavelengths) if trace.wavelengths else "-"
        lines.append(f"{trace.id}  {shown}  {waves}  {trace.text}")
    return "\n".join(lines)


def format_locks(forbid: tuple[str, ...]) -> str:
    if not forbid:
        return (
            "No locks yet. A lock is a phrase the reply is not allowed to contain.\n"
            'angularai lock add "the phrase"'
        )
    return "\n".join(forbid)


def format_status(
    *,
    root: Path,
    config: Config,
    view: CortexView,
    last: PassRecord | None,
    key_name: str | None,
    key_set: bool,
) -> str:
    current = len(view.current_traces())
    superseded = sum(1 for trace in view.traces if trace.state == "superseded")
    lines = [
        f"{config.substrate or 'unwired'}  {config.model or 'no model'}",
        str(root),
        f"budget {config.workspace_budget}",
        f"traces {current} current, {superseded} superseded",
        f"episodes {len(view.episodes)}",
    ]
    if key_name is not None:
        lines.append(f"{key_name}  {'set' if key_set else 'missing'}")
    if last is None:
        lines.append("no pass yet")
    else:
        lines.append(
            f"last {last.outcome}  used {last.used} of {last.budget}  "
            f"traces shown {last.traces_shown} held {last.traces_held}  "
            f"episodes shown {last.episodes_shown} held {last.episodes_held}"
        )
        if last.error:
            lines.append(last.error)
    return "\n".join(lines)


def _trace_line(trace: Trace) -> str:
    waves = ",".join(trace.wavelengths) if trace.wavelengths else "-"
    return f"{trace.id}  {waves}  {trace.text}"
