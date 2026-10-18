import json
import re

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
_ACT_FIELDS = frozenset({"tool", "start", "overwrite", *_TEXT_FIELDS})
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
    return Action(tool=name, start=max(start, 1), overwrite=overwrite, **texts)


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
