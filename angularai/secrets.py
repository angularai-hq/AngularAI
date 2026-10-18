"""Secret detection and redaction.

Three ways a string counts as a secret:

1. A known key shape. Each pattern is bounded on the left, so a word that merely
   contains ``sk-`` (desk-organizer, ask-anything) is not a key.
2. An assignment: ``password=``, ``token:`` and kin, followed by a literal value of
   six or more characters. Only the value is redacted. A value that is code
   (it holds ``( [ { $ <``), a placeholder, or a null-like word is not a secret.
   After a colon, an all-letter value is prose, not a secret.
3. Entropy. A run of 32 or more key characters ``[A-Za-z0-9+/=_-]`` is a secret when
   it mixes lower case, upper case, and digits, holds at least four digits, has
   Shannon entropy of at least 4.2 bits a character, and is not hex. Hex runs
   (git SHAs, sha256, md5) and UUIDs are hex, so they pass. Words, paths, and
   camelCase identifiers fail the mix or the entropy floor. A run that starts like a
   path (``/``, ``./``, ``../``, ``~/``) is judged one segment at a time, so a
   temporary directory such as ``/private/var/folders/…/T/…`` is not a secret.

A value from the environment is a secret too, when its variable name looks like
one (see ``environment_secrets``). Eight characters or more match anywhere. Six or
seven characters match as a whole word, and only when they hold a digit or a
symbol. Anything else is ignored: a short or all-letter value (``local``, ``true``)
is a placeholder or a word, and matching it would erase ordinary text.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping

_REDACTED = "[redacted]"
_LEFT = r"(?<![A-Za-z0-9_\-])"
_RIGHT = r"(?![A-Za-z0-9_\-])"

_KEYS = re.compile(
    "|".join(
        (
            _LEFT + r"sk-(?:ant-|proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{16,}",
            _LEFT + r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,}",
            _LEFT + r"github_pat_[A-Za-z0-9_]{22,}",
            _LEFT + r"gh[pousr]_[A-Za-z0-9]{20,}",
            _LEFT + r"AIza[0-9A-Za-z_\-]{35}" + _RIGHT,
            _LEFT + r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
            _LEFT + r"glpat-[A-Za-z0-9_\-]{20,}",
            _LEFT + r"hf_[A-Za-z0-9]{30,}",
            _LEFT + r"xox[abposre]-[A-Za-z0-9\-]{10,}",
            r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Za-z])",
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
            r"(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----)?",
        )
    )
)

_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?:password|passwd|pwd|passphrase|secret|client[_\-]?secret|token|auth[_\-]?token"
    r"|access[_\-]?token|api[_\-]?key|access[_\-]?key|private[_\-]?key)"
    r"[\"']?\s*(?P<sep>[:=])\s*[\"']?(?P<value>[^\s\"',;]{6,})"
)
_CODE = frozenset("()[]{}$<%")
_PLACEHOLDERS = frozenset(
    {"none", "null", "nil", "true", "false", "undefined", "changeme", "example", "redacted"}
)

_RUN = re.compile(r"(?<![A-Za-z0-9+/=_\-])[A-Za-z0-9+/=_\-]{32,}(?![A-Za-z0-9+/=_\-])")
_HEX = re.compile(r"[0-9a-fA-F\-]+")
_ENTROPY_FLOOR = 4.2

_NAME_WORDS = ("KEY", "KEYS", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
_NAME_PREFIXES = ("API", "ACCESS", "PRIVATE", "AUTH", "CLIENT", "SECRET", "SESSION")
_LONG = 8
_SHORT = 6


def looks_secret(text: str) -> bool:
    if _KEYS.search(text) is not None:
        return True
    if any(_assigned(match) for match in _ASSIGNMENT.finditer(text)):
        return True
    return any(_random(match.group(0)) for match in _RUN.finditer(text))


def redact(text: str, secrets: tuple[str, ...]) -> str:
    """Every secret replaced by [redacted]. Redacting the result again changes nothing.

    A marker next to the remaining text can spell a secret again (a secret
    ``d]abcdef`` in ``d]abcdefabcdef`` leaves ``[redacted]abcdef``), so the passes
    repeat until nothing changes. If a secret still shows after that, the whole text
    is withheld.
    """

    usable = _usable(secrets)
    redacted = text
    for _ in range(_ROUNDS):
        again = _redact_once(redacted, usable)
        if again == redacted:
            break
        redacted = again
    else:
        return _REDACTED
    if any(_literal(secret).search(redacted) for secret in usable):
        return _REDACTED
    return redacted


_ROUNDS = 8


def _redact_once(text: str, secrets: tuple[str, ...]) -> str:
    redacted = text
    for secret in secrets:
        redacted = _literal(secret).sub(_REDACTED, redacted)
    redacted = _KEYS.sub(_REDACTED, redacted)
    redacted = _ASSIGNMENT.sub(_value_redactor, redacted)
    return _RUN.sub(_run_redactor, redacted)


def contains_secret(text: str, secrets: tuple[str, ...]) -> bool:
    if looks_secret(text):
        return True
    return any(_literal(secret).search(text) is not None for secret in _usable(secrets))


def environment_secrets(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Values of every variable whose name says it holds a secret."""

    found: list[str] = []
    for name, raw in environ.items():
        value = raw.strip()
        if secret_name(name) and _worth(value) and value not in found:
            found.append(value)
    return tuple(found)


def secret_name(name: str) -> bool:
    for part in re.split(r"[^A-Z0-9]+", name.upper()):
        if part in _NAME_WORDS or part == "CREDENTIALS":
            return True
        if part.startswith(_NAME_PREFIXES) and part.endswith(_NAME_WORDS):
            return True
    return False


def _usable(secrets: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(secret for secret in secrets if _worth(secret))


def _worth(value: str) -> bool:
    if len(value) < _SHORT or value.isdigit() or value.lower() in _PLACEHOLDERS:
        return False
    return len(value) >= _LONG or not value.isalpha()


def _literal(secret: str) -> re.Pattern[str]:
    escaped = re.escape(secret)
    if len(secret) >= _LONG:
        return re.compile(escaped)
    return re.compile(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])")


def _assigned(match: re.Match[str]) -> bool:
    value = match.group("value")
    if any(character in _CODE for character in value):
        return False
    lowered = value.lower().strip(".")
    if lowered in _PLACEHOLDERS or set(lowered) <= {"*", "x", "."}:
        return False
    if lowered.startswith(("os.", "env.", "process.", "self.", "settings.")):
        return False
    # After a colon, prose is likely ("password: forgotten"). A value there needs a
    # digit or a symbol.
    return match.group("sep") == "=" or not value.isalpha()


def _value_redactor(match: re.Match[str]) -> str:
    if not _assigned(match):
        return match.group(0)
    start = match.start("value") - match.start(0)
    return match.group(0)[:start] + _REDACTED


def _run_redactor(match: re.Match[str]) -> str:
    run = match.group(0)
    if _path(run):
        return "/".join(_REDACTED if _random(part) else part for part in run.split("/"))
    return _REDACTED if _random(run) else run


def _path(run: str) -> bool:
    return run.startswith(("/", "./", "../", "~/"))


def _random(run: str) -> bool:
    if _path(run):
        return any(_random(part) for part in run.split("/") if part)
    if len(run) < 32 or _HEX.fullmatch(run):
        return False
    if not (_has(run, str.islower) and _has(run, str.isupper)):
        return False
    if sum(character.isdigit() for character in run) < 4:
        return False
    return _entropy(run) >= _ENTROPY_FLOOR


def _has(run: str, test: Callable[[str], bool]) -> bool:
    return any(test(character) for character in run)


def _entropy(run: str) -> float:
    counts = Counter(run)
    total = len(run)
    return -sum(count / total * math.log2(count / total) for count in counts.values())
