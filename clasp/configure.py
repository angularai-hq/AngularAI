from dataclasses import replace

from clasp.errors import InputRejected
from clasp.models import Config, SubstrateKind
from clasp.store import Store
from clasp.urls import validate_base_url

_MIN_BUDGET = 256
_MAX_BUDGET = 32000


def wire(store: Store, kind: SubstrateKind, model: str, base_url: str | None) -> Config:
    cleaned = model.strip()
    if not cleaned or len(cleaned) > 128 or any(character.isspace() for character in cleaned):
        raise InputRejected("The model name is empty or has spaces.")
    url = _url(base_url) if base_url else None
    if kind == "compatible" and url is None:
        raise InputRejected(
            "A compatible substrate needs --base-url, for example http://127.0.0.1:11434/v1."
        )
    current = store.config()
    updated = Config(
        substrate=kind,
        model=cleaned,
        base_url=url,
        workspace_budget=current.workspace_budget,
        max_input_tokens=current.max_input_tokens,
        forbid=current.forbid,
        json_mode=kind == "openai",
    )
    store.save_config(updated)
    return updated


def set_budget(store: Store, budget: int) -> Config:
    if budget < _MIN_BUDGET or budget > _MAX_BUDGET:
        raise InputRejected(
            f"Budget must be between {_MIN_BUDGET} and {_MAX_BUDGET} estimated tokens."
        )
    current = store.config()
    max_input = current.max_input_tokens
    if max_input > budget:
        max_input = max(32, budget // 2)
    updated = replace(current, workspace_budget=budget, max_input_tokens=max_input)
    store.save_config(updated)
    return updated


def add_lock(store: Store, phrase: str) -> Config:
    cleaned = " ".join(phrase.split())
    if not cleaned:
        raise InputRejected("A lock needs a phrase.")
    if len(cleaned) > 200:
        raise InputRejected("That lock phrase is too long.")
    current = store.config()
    if cleaned.lower() in {item.lower() for item in current.forbid}:
        return current
    if len(current.forbid) >= 64:
        raise InputRejected("This cortex already has 64 locks.")
    updated = replace(current, forbid=(*current.forbid, cleaned))
    store.save_config(updated)
    return updated


def remove_lock(store: Store, phrase: str) -> Config:
    cleaned = " ".join(phrase.split())
    current = store.config()
    kept = tuple(item for item in current.forbid if item.lower() != cleaned.lower())
    if len(kept) == len(current.forbid):
        raise InputRejected("That lock is not set.")
    updated = replace(current, forbid=kept)
    store.save_config(updated)
    return updated


def _url(value: str) -> str:
    try:
        return validate_base_url(value)
    except ValueError as exc:
        raise InputRejected(str(exc)) from exc
