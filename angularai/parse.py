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


def _act(value: object) -> Action | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SubstrateUnreadable(_UNREADABLE)
    tool = value.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    return Action(
        tool=tool.strip().lower()[:16],
        path=_text(value.get("path"), 240),
        query=_text(value.get("query"), 200),
        old=_text(value.get("old"), 4000),
        new=_text(value.get("new"), 8000),
        text=_text(value.get("text"), 80000),
        command=_text(value.get("command"), 400),
        start=_start(value.get("start")),
        overwrite=value.get("overwrite") is True,
    )


def _text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value[:limit]


def _start(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    if value < 1:
        return 1
    return min(value, 100000)


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
