import json
import re
from collections.abc import Sequence

from angularai.errors import SubstrateUnreadable
from angularai.models import Action, Integration, TraceDraft

_SLUG = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_TRACE_ID = re.compile(r"^t_[0-9a-f]{12}$")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_MAX_REPLY = 8000
_MAX_TRACE = 240
_MAX_REMEMBER = 24
_MAX_SUPERSEDE = 8

_UNREADABLE = "The model did not return a cortex reply. Nothing was written."


def parse_integration(raw: str) -> Integration:
    text = _FENCE.sub("", raw.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise SubstrateUnreadable(_UNREADABLE)
    try:
        payload: object = json.loads(text[start : end + 1])
    except (ValueError, RecursionError) as exc:
        raise SubstrateUnreadable(_UNREADABLE) from exc
    if not isinstance(payload, dict):
        raise SubstrateUnreadable(_UNREADABLE)
    reply = _reply(payload.get("reply"))
    remember = _remember(payload.get("remember", []))
    supersede = _supersede(payload.get("supersede", []))
    return Integration(
        reply=reply,
        remember=remember,
        supersede=supersede,
        act=_act(payload.get("act")),
    )


def _reply(value: object) -> str:
    if not isinstance(value, str):
        raise SubstrateUnreadable(_UNREADABLE)
    reply = value.strip()
    if not reply:
        raise SubstrateUnreadable(_UNREADABLE)
    if len(reply) > _MAX_REPLY:
        return reply[:_MAX_REPLY]
    return reply


def _remember(value: object) -> tuple[TraceDraft, ...]:
    if not isinstance(value, list):
        return ()
    drafts: list[TraceDraft] = []
    for item in value:
        if len(drafts) >= _MAX_REMEMBER:
            break
        draft = _draft(item)
        if draft is not None:
            drafts.append(draft)
    return tuple(drafts)


def trace_draft(text: str, wavelengths: Sequence[str] = ()) -> TraceDraft | None:
    """A trace as the model would write it, or None: one line of at most 240 characters,
    and at most 4 wavelengths, each a lowercase slug. Whitespace runs become one space."""

    if len(wavelengths) > 4:
        return None
    return _draft({"text": text, "wavelengths": list(wavelengths)})


def valid_trace_id(value: str) -> bool:
    """A trace id as AngularAI writes it: t_ and 12 hex digits."""

    return _TRACE_ID.fullmatch(value) is not None


def _draft(value: object) -> TraceDraft | None:
    if not isinstance(value, dict):
        return None
    raw_text = value.get("text")
    if not isinstance(raw_text, str):
        return None
    text = " ".join(raw_text.split())
    if not text or len(text) > _MAX_TRACE:
        return None
    wavelengths = _wavelengths(value.get("wavelengths", []))
    if wavelengths is None:
        return None
    return TraceDraft(text=text, wavelengths=wavelengths)


def _wavelengths(value: object) -> tuple[str, ...] | None:
    if value is None:
        return ()
    if not isinstance(value, list):
        return None
    slugs: list[str] = []
    for item in value[:4]:
        if not isinstance(item, str):
            return None
        slug = item.strip().lower()
        if not _SLUG.fullmatch(slug):
            return None
        if slug not in slugs:
            slugs.append(slug)
    return tuple(slugs)


_TEXT_FIELDS = {
    "path": 240,
    "query": 200,
    "old": 4000,
    "new": 8000,
    "text": 80000,
    "command": 400,
}
# note is the model's own progress line, taken by every tool and never run.
_ACT_FIELDS = frozenset({"tool", "start", "overwrite", "append", "note", *_TEXT_FIELDS})
_MAX_START = 100000


def _act(value: object) -> Action | None:
    """One tool call, or None when the work is done.

    Anything malformed is refused, not repaired: a field is never cut to fit, a wrong
    type is never guessed at, and a field no tool takes is not ignored. The refusal
    goes back to the model as the tool's result, and nothing runs.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        return _refused("", "act must be one object with a tool, or null.")
    tool = value.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return _refused("", "act needs a tool name.")
    name = tool.strip().lower()
    if len(name) > 16:
        return _refused("", "That tool name is too long.")
    extra = sorted(str(key) for key in value if key not in _ACT_FIELDS)
    if extra:
        return _refused(name, f"act has a field no tool takes: {extra[0][:40]}.")
    texts: dict[str, str] = {}
    for field, limit in _TEXT_FIELDS.items():
        raw = value.get(field)
        if raw is None:
            texts[field] = ""
        elif not isinstance(raw, str):
            return _refused(name, f"{field} must be text.")
        elif len(raw) > limit:
            return _refused(name, f"{field} is longer than {limit} characters. Nothing was cut.")
        else:
            texts[field] = raw
    start = value.get("start", 1)
    if isinstance(start, bool) or not isinstance(start, int) or start > _MAX_START:
        return _refused(name, f"start must be a whole number up to {_MAX_START}.")
    overwrite = value.get("overwrite", False)
    if not isinstance(overwrite, bool):
        return _refused(name, "overwrite must be true or false.")
    append = value.get("append", False)
    if not isinstance(append, bool):
        return _refused(name, "append must be true or false.")
    if append and overwrite:
        return _refused(name, "append and overwrite cannot both be true.")
    return Action(tool=name, start=max(start, 1), overwrite=overwrite, append=append, **texts)


def _refused(tool: str, problem: str) -> Action:
    return Action(tool=tool, problem=problem)


def _supersede(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    ids: list[str] = []
    for item in value:
        if len(ids) >= _MAX_SUPERSEDE:
            break
        if isinstance(item, str) and _TRACE_ID.fullmatch(item) and item not in ids:
            ids.append(item)
    return tuple(ids)


def _function(
    name: str, description: str, fields: dict[str, str], required: list[str]
) -> dict[str, object]:
    kinds = {"start": "integer", "overwrite": "boolean", "append": "boolean"}
    if name not in ("finish", "remember"):
        fields = {**fields, "note": "Short: what is done so far and what comes next."}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    field: {"type": kinds.get(field, "string"), "description": text}
                    for field, text in fields.items()
                },
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# The agent tools as OpenAI-style function definitions. remember is not a tool that
# runs: it becomes a trace, judged by the lock like any other.
TOOL_SPECS: tuple[dict[str, object], ...] = (
    _function("list", "List a directory inside the root.", {"path": "A directory."}, ["path"]),
    _function(
        "read",
        "Read a text file, 160 lines at a time.",
        {"path": "A file.", "start": "The first line, from 1."},
        ["path"],
    ),
    _function(
        "search",
        "Find lines that contain a text.",
        {"query": "The text.", "path": "A directory or file."},
        ["query"],
    ),
    _function(
        "edit",
        "Replace one exact text in a file.",
        {"path": "A file.", "old": "The exact text.", "new": "Its replacement."},
        ["path", "old", "new"],
    ),
    _function(
        "write",
        "Write a new file, overwrite one when overwrite is true, or add text to the end "
        "of one when append is true.",
        {
            "path": "A file.",
            "text": "The whole content, or the lines to add.",
            "overwrite": "Replace a file.",
            "append": "Add text to the end of the file.",
        },
        ["path", "text"],
    ),
    _function(
        "run",
        "Run one allowlisted program. No shell.",
        {"command": "The program and its arguments."},
        ["command"],
    ),
    _function(
        "finish",
        "End the run: the task is done.",
        {"summary": "What was done."},
        ["summary"],
    ),
    _function(
        "remember",
        "Keep one durable fact for later steps.",
        {"text": "One fact, at most 240 characters."},
        ["text"],
    ),
)
TOOL_NAMES = frozenset({"list", "read", "search", "edit", "write", "run"})


# A step that only kept facts. It runs nothing; the traces are its work.
REMEMBERED = Action(tool="remember")
IDLE = Action(tool="", problem="No tool was called. Call one tool, or finish.")


def integration_from_calls(content: str, calls: Sequence[object]) -> Integration:
    """A native reply: content and tool calls. The first tool call is the act.

    remember calls become traces. finish ends the work: its act is None and its summary
    the reply. Arguments that are not a JSON object refuse the act, as a malformed JSON
    act does. Only remember calls make REMEMBERED. No tool call at all is IDLE: a
    refused act that tells the model to call a tool, so a stray sentence never ends
    a long run.
    """

    act: Action | None = None
    finished = ""
    drafts: list[TraceDraft] = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        raw = function.get("arguments") if isinstance(function, dict) else None
        arguments: object = raw
        if isinstance(raw, str):
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except (ValueError, RecursionError):
                arguments = None
        if not isinstance(name, str):
            continue
        if name == "remember":
            text = arguments.get("text") if isinstance(arguments, dict) else None
            draft = trace_draft(text) if isinstance(text, str) else None
            if draft is not None and len(drafts) < _MAX_REMEMBER:
                drafts.append(draft)
            continue
        if act is not None or finished:
            continue
        if name == "finish":
            summary = arguments.get("summary") if isinstance(arguments, dict) else None
            finished = summary if isinstance(summary, str) and summary.strip() else "Done."
            continue
        if not isinstance(arguments, dict):
            act = _refused(
                name.strip().lower()[:16], "The tool call arguments are not a JSON object."
            )
        else:
            note = arguments.get("note")
            if isinstance(note, str) and note.strip() and not content.strip():
                content = note
            act = _act({**arguments, "tool": name})
    if finished:
        reply = " ".join(finished.split())[:_MAX_REPLY]
        return Integration(reply=reply, remember=tuple(drafts), supersede=(), act=None)
    reply = " ".join(content.split())[:_MAX_REPLY] or "Working."
    if act is None:
        act = REMEMBERED if drafts else IDLE
    return Integration(reply=reply, remember=tuple(drafts), supersede=(), act=act)
