"""Saved models and their keys.

A saved model is a name, a provider, a model id, and for a compatible endpoint a
base URL. They live in ``<home>/models.json`` with the name of the active one.
That file holds no key: at most a hint, the key's last four characters.

A key lives in the OS keyring (service ``angularai``) when a usable backend
exists. Without one, a headless Linux box for example, it lives in
``<home>/keys.json``, created with mode 0600 from the first byte and replaced
atomically. Every write to either file takes the home's lock.

Which model a pass uses: ``ANGULARAI_MODEL`` (a saved name), else the cortex's
own ``angularai wire``, else the active saved model. Which key: the provider's
variable (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``ANGULARAI_API_KEY``), else
the saved key of the saved model in use. A cortex wired by ``angularai wire``
reads its key from the environment only.

A key is never printed, logged, or put in an exception or a repr. ``mask`` is the
only view of it that leaves this module.
"""

import json
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from angularai.durable import atomic_write, locked
from angularai.errors import ConfigError, InputRejected, KeyMissing
from angularai.models import Config, SubstrateKind
from angularai.names import cortex_home
from angularai.urls import validate_base_url

MODELS_FILE = "models.json"
KEYS_FILE = "keys.json"
SERVICE = "angularai"
MODEL_ENV = "ANGULARAI_MODEL"
KEY_ENVS: Mapping[SubstrateKind, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "compatible": "ANGULARAI_API_KEY",
}
Storage = Literal["keyring", "file", "none"]
_VERSION = 1
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}", re.ASCII)
_MAX_MODELS = 64
_MAX_KEY = 4096
NAME_RULE = (
    "A saved model name is 1 to 32 lowercase letters, digits, - or _, "
    "starting with a letter or a digit."
)
_PREFIXES = ("sk-ant-", "sk-proj-", "sk-")


@dataclass(frozen=True)
class Profile:
    """A saved model. It never holds the key: storage says where the key is."""

    name: str
    provider: SubstrateKind
    model: str
    base_url: str | None
    storage: Storage
    hint: str


@dataclass(frozen=True)
class Book:
    active: str | None
    profiles: tuple[Profile, ...]

    def get(self, name: str) -> Profile | None:
        return next((item for item in self.profiles if item.name == name), None)


class Backend(Protocol):
    """The part of a keyring backend this module uses."""

    def get_password(self, service: str, username: str) -> str | None: ...

    def set_password(self, service: str, username: str, password: str) -> None: ...

    def delete_password(self, service: str, username: str) -> None: ...


def valid_name(name: str) -> bool:
    return _NAME.fullmatch(name) is not None


def check_name(name: str) -> str:
    """The name, or InputRejected. The name is not echoed: it may be anything."""

    if not valid_name(name):
        raise InputRejected(NAME_RULE)
    return name


def check_model(model: str) -> str:
    cleaned = model.strip()
    bad = any(c.isspace() or not c.isprintable() for c in cleaned)
    if not cleaned or len(cleaned) > 128 or bad:
        raise InputRejected("The model id is empty, too long, or has spaces.")
    return cleaned


def check_url(value: str) -> str:
    try:
        return validate_base_url(value)
    except ValueError as exc:
        raise InputRejected(str(exc)) from exc


def check_key(key: str) -> str:
    """A key as typed, stripped. The key is never part of the message."""

    cleaned = key.strip()
    if not cleaned:
        raise InputRejected("The key is empty.")
    if len(cleaned) > _MAX_KEY:
        raise InputRejected("The key is too long.")
    if not cleaned.isascii() or any(c.isspace() or not c.isprintable() for c in cleaned):
        raise InputRejected("The key has spaces or characters a key does not have.")
    return cleaned


def mask(key: str) -> str:
    """At most the last four characters, after a known public prefix.

    ``sk-demo-0000000000000000a1b2`` is ``sk-…a1b2``. A short key shows less: two
    characters under 16, none under 8.
    """

    cleaned = key.strip()
    if len(cleaned) >= 16:
        tail = cleaned[-4:]
    elif len(cleaned) >= 8:
        tail = cleaned[-2:]
    else:
        tail = ""
    prefix = ""
    if len(cleaned) >= 16:
        prefix = next((item for item in _PREFIXES if cleaned.startswith(item)), "")
    return f"{prefix}…{tail}"


def home(environ: Mapping[str, str]) -> Path:
    return cortex_home(environ)


def ensure_home(environ: Mapping[str, str]) -> Path:
    where = home(environ)
    where.mkdir(parents=True, exist_ok=True, mode=0o700)
    if where.name == ".angularai":
        where.chmod(0o700)
    return where


# --- models.json ------------------------------------------------------------


def load_book(environ: Mapping[str, str]) -> Book:
    path = home(environ) / MODELS_FILE
    payload = _read_json(path, MODELS_FILE)
    if payload is None:
        return Book(None, ())
    return _book_from(payload)


def save_book(environ: Mapping[str, str], book: Book) -> None:
    where = ensure_home(environ)
    data = json.dumps(_book_json(book), indent=2) + "\n"
    atomic_write(where / MODELS_FILE, data.encode("utf-8"))


def _book_json(book: Book) -> dict[str, object]:
    return {
        "version": _VERSION,
        "active": book.active,
        "models": [
            {
                "name": item.name,
                "provider": item.provider,
                "model": item.model,
                "base_url": item.base_url,
                "key": item.storage,
                "hint": item.hint,
            }
            for item in book.profiles
        ],
    }


def _book_from(payload: dict[str, object]) -> Book:
    if payload.get("version") != _VERSION:
        raise ConfigError(f"{MODELS_FILE} has a version this AngularAI does not know.")
    raw = payload.get("models")
    if not isinstance(raw, list) or len(raw) > _MAX_MODELS:
        raise ConfigError(f"{MODELS_FILE} has no list of models.")
    profiles: list[Profile] = []
    for entry in raw:
        profile = _profile_from(entry)
        if any(item.name == profile.name for item in profiles):
            raise ConfigError(f"{MODELS_FILE} names one model twice.")
        profiles.append(profile)
    active = payload.get("active")
    if active is not None and (
        not isinstance(active, str) or all(item.name != active for item in profiles)
    ):
        raise ConfigError(f"{MODELS_FILE} marks a model active that it does not have.")
    return Book(active, tuple(profiles))


def _profile_from(entry: object) -> Profile:
    if not isinstance(entry, dict):
        raise ConfigError(f"{MODELS_FILE} has a model that is not an object.")
    try:
        name = check_name(_text(entry.get("name")))
        model = check_model(_text(entry.get("model")))
        url_raw = entry.get("base_url")
        base_url = None if url_raw is None else check_url(_text(url_raw))
    except InputRejected as exc:
        raise ConfigError(f"{MODELS_FILE}: {exc}") from exc
    provider = entry.get("provider")
    if provider != "openai" and provider != "anthropic" and provider != "compatible":
        raise ConfigError(f"{MODELS_FILE} has a model with an unknown provider.")
    if provider == "compatible" and base_url is None:
        raise ConfigError(f"{MODELS_FILE} has a compatible model without base_url.")
    storage = entry.get("key")
    if storage != "keyring" and storage != "file" and storage != "none":
        raise ConfigError(f"{MODELS_FILE} has a model with an unknown key storage.")
    hint = entry.get("hint", "")
    if not isinstance(hint, str) or len(hint) > 16:
        raise ConfigError(f"{MODELS_FILE} has a model with a bad key hint.")
    return Profile(name, provider, model, base_url, storage, hint)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _read_json(path: Path, label: str) -> dict[str, object] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError(f"{label} in {path.parent} is not a regular file.")
    try:
        payload: object = json.loads(path.read_bytes())
    except (ValueError, RecursionError) as exc:
        raise ConfigError(f"{label} in {path.parent} is not valid JSON.") from exc
    except OSError as exc:
        raise ConfigError(f"{label} in {path.parent} cannot be read.") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} in {path.parent} is not a JSON object.")
    return {str(key): value for key, value in payload.items()}


# --- keys -------------------------------------------------------------------


def default_backend() -> Backend | None:
    """The OS keyring, or None when keyring has no backend that can store."""

    try:
        import keyring
        from keyring.backends import chainer, fail
    except ImportError:  # pragma: no cover - keyring is a dependency
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    if isinstance(backend, fail.Keyring) or getattr(backend, "priority", 0) <= 0:
        return None
    if isinstance(backend, chainer.ChainerBackend) and not backend.backends:
        return None
    if type(backend).__module__.startswith("keyring.backends.null"):
        return None
    return backend


class Vault:
    """Where keys are kept. The keyring when it works, else keys.json at 0600."""

    def __init__(
        self,
        environ: Mapping[str, str],
        backend: Callable[[], Backend | None] = default_backend,
    ) -> None:
        self._environ = environ
        self._backend = backend

    def __repr__(self) -> str:
        return "Vault()"

    def _account(self, name: str) -> str:
        return f"{home(self._environ)}::{name}"

    def save(self, name: str, key: str) -> Storage:
        """Store the key. Returns where it went. The keyring is tried first."""

        backend = self._backend()
        if backend is not None:
            try:
                backend.set_password(SERVICE, self._account(name), key)
                stored = backend.get_password(SERVICE, self._account(name)) == key
            except Exception:
                stored = False
            if stored:
                self._drop_file(name)
                return "keyring"
        self._write_file(name, key)
        return "file"

    def load(self, name: str, storage: Storage) -> str | None:
        if storage == "none":
            return None
        if storage == "file":
            return self._read_file().get(name)
        backend = self._backend()
        if backend is None:
            raise KeyMissing(
                f"The key for saved model {name} is in the OS keyring, and no keyring is "
                "usable now. Run angularai setup again, or set the provider's variable."
            )
        # The backend's error is dropped, not chained: a backend may echo what it holds.
        failed = False
        found: str | None = None
        try:
            found = backend.get_password(SERVICE, self._account(name))
        except Exception:
            failed = True
        if failed:
            raise KeyMissing(
                f"The OS keyring did not give the key for saved model {name}. "
                "Unlock it, run angularai setup again, or set the provider's variable."
            )
        return found

    def delete(self, name: str, storage: Storage) -> None:
        if storage == "keyring":
            backend = self._backend()
            if backend is not None:
                try:
                    backend.delete_password(SERVICE, self._account(name))
                except Exception:
                    pass
        elif storage == "file":
            self._drop_file(name)

    def _path(self) -> Path:
        return home(self._environ) / KEYS_FILE

    def _read_file(self) -> dict[str, str]:
        path = self._path()
        try:
            info = path.lstat()
        except FileNotFoundError:
            return {}
        if stat.S_ISREG(info.st_mode) and info.st_mode & 0o077:
            raise ConfigError(f"{path} is readable by others. Run: chmod 600 {path}")
        payload = _read_json(path, KEYS_FILE)
        if payload is None:
            return {}
        keys = payload.get("keys")
        if payload.get("version") != _VERSION or not isinstance(keys, dict):
            raise ConfigError(f"{KEYS_FILE} in {path.parent} is not a key file.")
        return {
            str(name): value
            for name, value in keys.items()
            if isinstance(value, str) and valid_name(str(name))
        }

    def _write_file(self, name: str, key: str) -> None:
        where = ensure_home(self._environ)
        with locked(where):
            keys = self._read_file()
            keys[name] = key
            self._dump(keys)

    def _drop_file(self, name: str) -> None:
        if not self._path().exists():
            return
        with locked(home(self._environ)):
            keys = self._read_file()
            if keys.pop(name, None) is not None:
                self._dump(keys)

    def _dump(self, keys: dict[str, str]) -> None:
        data = json.dumps({"version": _VERSION, "keys": keys}, indent=2) + "\n"
        # atomic_write creates the temporary file with mkstemp, so it is 0600
        # from the first byte, then moves it over the name.
        atomic_write(self._path(), data.encode("utf-8"), mode=0o600)


# --- changes ----------------------------------------------------------------


def add_profile(
    environ: Mapping[str, str],
    vault: Vault,
    profile: Profile,
    key: str | None,
    *,
    make_active: bool,
) -> Profile:
    """Save profile and its key. Replaces a saved model of the same name."""

    where = ensure_home(environ)
    with locked(where):
        book = load_book(environ)
        old = book.get(profile.name)
        if old is None and len(book.profiles) >= _MAX_MODELS:
            raise InputRejected(f"There are already {_MAX_MODELS} saved models.")
    storage: Storage = "none"
    hint = ""
    if key is not None:
        storage = vault.save(profile.name, key)
        hint = mask(key)
    if old is not None and old.storage != storage:
        vault.delete(profile.name, old.storage)
    saved = replace(profile, storage=storage, hint=hint)
    with locked(where):
        book = load_book(environ)
        kept = tuple(item for item in book.profiles if item.name != profile.name)
        active = profile.name if make_active or book.active is None else book.active
        save_book(environ, Book(active, (*kept, saved)))
    return saved


def use_profile(environ: Mapping[str, str], name: str) -> Profile:
    check_name(name)
    where = ensure_home(environ)
    with locked(where):
        book = load_book(environ)
        found = book.get(name)
        if found is None:
            raise InputRejected(f"No saved model named {name}. angularai model lists them.")
        save_book(environ, Book(name, book.profiles))
    return found


def remove_profile(environ: Mapping[str, str], vault: Vault, name: str) -> tuple[Profile, bool]:
    """Remove a saved model and its key. Returns it, and whether it was active."""

    check_name(name)
    where = ensure_home(environ)
    with locked(where):
        book = load_book(environ)
        found = book.get(name)
        if found is None:
            raise InputRejected(f"No saved model named {name}. angularai model lists them.")
        kept = tuple(item for item in book.profiles if item.name != name)
        was_active = book.active == name
        save_book(environ, Book(None if was_active else book.active, kept))
    vault.delete(name, found.storage)
    return found, was_active


def free_name(book: Book, base: str) -> str:
    if book.get(base) is None:
        return base
    for number in range(2, _MAX_MODELS + 2):
        candidate = f"{base[:29]}-{number}"
        if book.get(candidate) is None:
            return candidate
    raise InputRejected(f"There are already {_MAX_MODELS} saved models.")


# --- resolution -------------------------------------------------------------


def chosen_profile(config: Config, environ: Mapping[str, str]) -> Profile | None:
    """The saved model a pass uses, or None when the cortex's own wiring decides."""

    picked = environ.get(MODEL_ENV, "").strip()
    if picked:
        if not valid_name(picked):
            raise ConfigError(f"{MODEL_ENV} must name a saved model. {NAME_RULE}")
        book = load_book(environ)
        found = book.get(picked)
        if found is None:
            raise ConfigError(f"{MODEL_ENV} names {picked}, and no saved model has that name.")
        return found
    if config.substrate is not None:
        return None
    book = load_book(environ)
    return None if book.active is None else book.get(book.active)


def effective_config(config: Config, environ: Mapping[str, str]) -> tuple[Config, Profile | None]:
    """The cortex's config with the saved model in use put in, and that saved model."""

    profile = chosen_profile(config, environ)
    if profile is None:
        return config, None
    return (
        replace(
            config,
            substrate=profile.provider,
            model=profile.model,
            base_url=profile.base_url,
            json_mode=profile.provider == "openai",
        ),
        profile,
    )


def saved_key(
    profile: Profile, environ: Mapping[str, str], vault: Vault | None = None
) -> str | None:
    return (vault or Vault(environ)).load(profile.name, profile.storage)


def key_source(kind: SubstrateKind, profile: Profile | None, environ: Mapping[str, str]) -> str:
    """Where the key comes from, in words, without reading the keyring."""

    env_name = KEY_ENVS[kind]
    if environ.get(env_name, "").strip():
        return f"{env_name} (environment)"
    if profile is None:
        return f"{env_name} missing"
    if profile.storage == "keyring":
        return f"{profile.hint} in the OS keyring"
    if profile.storage == "file":
        return f"{profile.hint} in {KEYS_FILE}"
    return "no key"


def anything_configured(environ: Mapping[str, str]) -> bool:
    """A saved model, a key in the environment, or ANGULARAI_MODEL."""

    if environ.get(MODEL_ENV, "").strip():
        return True
    if any(environ.get(name, "").strip() for name in KEY_ENVS.values()):
        return True
    try:
        return bool(load_book(environ).profiles)
    except ConfigError:
        return True


def storage_words(storage: Storage, environ: Mapping[str, str]) -> str:
    if storage == "keyring":
        return "the OS keyring"
    if storage == "file":
        return f"{home(environ) / KEYS_FILE} (mode 0600; no usable OS keyring)"
    return "nowhere (no key)"
