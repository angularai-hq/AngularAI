"""Cortex names and where a named cortex lives.

A name is 1 to 32 characters: lowercase ASCII letters, digits, ``-`` and ``_``,
starting with a letter or a digit. Lowercase only, so two names never meet on a
case-insensitive disk. A name never holds a separator or a dot, so it cannot
leave the home directory. Device names Windows reserves are refused.

The home is ``ANGULARAI_HOME``, else ``~/.angularai``. Cortex ``NAME`` is
``<home>/NAME``. The default cortex is the one named ``default``.
"""

import re
from collections.abc import Mapping
from pathlib import Path

from angularai.errors import ConfigError, InputRejected

DEFAULT = "default"
HOME_ENV = "ANGULARAI_HOME"
MAX_NAME = 32
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}", re.ASCII)
_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)
RULE = (
    "A cortex name is 1 to 32 lowercase letters, digits, - or _, starting with a letter or a digit."
)


def valid_name(name: str) -> bool:
    return _NAME.fullmatch(name) is not None and name not in _RESERVED


def check_name(name: str) -> str:
    """The name, or InputRejected. The name is never echoed: it may be anything."""

    if not valid_name(name):
        raise InputRejected(RULE)
    return name


def cortex_home(environ: Mapping[str, str]) -> Path:
    raw = environ.get(HOME_ENV, "").strip()
    if not raw:
        return Path.home() / ".angularai"
    home = Path(raw).expanduser()
    if not home.is_absolute():
        raise ConfigError(f"{HOME_ENV} must be an absolute path.")
    return home


def named_root(name: str, environ: Mapping[str, str]) -> Path:
    """The directory of cortex name. A link in its place is refused, not followed."""

    home = cortex_home(environ)
    root = home / check_name(name)
    if root.is_symlink():
        raise InputRejected(f"Cortex {name} is a link. A named cortex is a directory in {home}.")
    return root


def resolve_root(cortex: Path | None, name: str | None, environ: Mapping[str, str]) -> Path:
    """The cortex a command works on: --cortex PATH, else --name NAME, else
    ANGULARAI_CORTEX, else the default cortex in the home. Not both flags."""

    if cortex is not None and name is not None:
        raise InputRejected("Choose --cortex PATH or --name NAME, not both.")
    if cortex is not None:
        return cortex.expanduser()
    if name is not None:
        return named_root(name, environ)
    env = environ.get("ANGULARAI_CORTEX")
    if env:
        return Path(env).expanduser()
    return cortex_home(environ) / DEFAULT


def label(root: Path) -> str:
    """The name a cortex gives in a conversation: its directory's, when that is a name."""

    return root.name if valid_name(root.name) else "cortex"


def list_names(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Every name in the home that holds a cortex. Only the names are read."""

    home = cortex_home(environ)
    try:
        entries = sorted(home.iterdir())
    except OSError:
        return ()
    found: list[str] = []
    for entry in entries:
        if valid_name(entry.name) and not entry.is_symlink() and (entry / "cortex.json").is_file():
            found.append(entry.name)
    return tuple(found)
