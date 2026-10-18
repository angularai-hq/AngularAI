"""Timeouts, the pass deadline, and retries for the model API.

Defaults: connect 10 s, read 90 s, a 180 s deadline for the whole pass, three
attempts, and no single wait longer than 30 s. Each can be set in the environment:
ANGULARAI_CONNECT_TIMEOUT, ANGULARAI_READ_TIMEOUT, ANGULARAI_DEADLINE,
ANGULARAI_ATTEMPTS, ANGULARAI_MAX_WAIT.

Only a retryable error is sent again: a real rate limit, a 5xx, an overloaded
provider, a timeout, a reset or failed connection. The wait is the provider's
Retry-After (seconds, an HTTP date, or retry-after-ms) when it gives one, else
exponential backoff with full jitter: a random wait from 0 to min(max wait,
2^attempt seconds). A Retry-After longer than the max wait, or a wait that would
pass the deadline, ends the pass instead.
"""

import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

from angularai.errors import ConfigError, RateLimited, SubstrateError


@dataclass(frozen=True)
class Policy:
    connect_timeout: float = 10.0
    read_timeout: float = 90.0
    deadline: float = 180.0
    attempts: int = 3
    max_wait: float = 30.0
    base_wait: float = 1.0


_BOUNDS: dict[str, tuple[str, float, float]] = {
    "ANGULARAI_CONNECT_TIMEOUT": ("connect_timeout", 1, 120),
    "ANGULARAI_READ_TIMEOUT": ("read_timeout", 1, 600),
    "ANGULARAI_DEADLINE": ("deadline", 1, 1800),
    "ANGULARAI_ATTEMPTS": ("attempts", 1, 6),
    "ANGULARAI_MAX_WAIT": ("max_wait", 0, 120),
}


def policy_from_env(environ: Mapping[str, str]) -> Policy:
    values: dict[str, float] = {}
    for name, (attribute, low, high) in _BOUNDS.items():
        raw = environ.get(name, "").strip()
        if not raw:
            continue
        try:
            number = float(raw)
        except ValueError:
            number = math.nan
        whole = attribute != "attempts" or number.is_integer()
        if not (low <= number <= high) or not whole:
            kind = "a whole number" if attribute == "attempts" else "a number of seconds"
            raise ConfigError(f"{name} must be {kind} from {low:g} to {high:g}.")
        values[attribute] = number
    base = Policy()
    return Policy(
        connect_timeout=values.get("connect_timeout", base.connect_timeout),
        read_timeout=values.get("read_timeout", base.read_timeout),
        deadline=values.get("deadline", base.deadline),
        attempts=int(values.get("attempts", base.attempts)),
        max_wait=values.get("max_wait", base.max_wait),
    )


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


@dataclass(frozen=True)
class Clock:
    """Everything that reads or spends time. Tests pass a fake one."""

    monotonic: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    # Looked up at call time, so the test suite can forbid a real wait.
    sleep: Callable[[float], None] = field(default=lambda seconds: _sleep(seconds))
    random: Callable[[], float] = random.random


class Deadline:
    def __init__(self, seconds: float, clock: Clock) -> None:
        self.seconds = seconds
        self._clock = clock
        self._end = clock.monotonic() + seconds

    def remaining(self) -> float:
        return self._end - self._clock.monotonic()


def retry_after(headers: Mapping[str, str], now: float) -> float | None:
    """Seconds the provider asked for: Retry-After seconds or date, or retry-after-ms."""

    lowered = {name.lower(): value.strip() for name, value in headers.items()}
    millis = lowered.get("retry-after-ms", "")
    if millis:
        try:
            value = float(millis) / 1000
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value >= 0:
            return value
    raw = lowered.get("retry-after", "")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        seconds = math.nan
    if math.isfinite(seconds):
        return max(0.0, seconds)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, when.timestamp() - now)


def wait_before_retry(
    error: SubstrateError, attempt: int, policy: Policy, clock: Clock
) -> float | None:
    """The wait before the next attempt, or None when the error must not be retried.

    attempt counts the attempts already made, from 1.
    """

    if not error.retryable:
        return None
    if isinstance(error, RateLimited) and error.retry_after is not None:
        return error.retry_after if error.retry_after <= policy.max_wait else None
    # The exponent is capped so a large attempt count cannot overflow a float.
    ceiling = min(policy.max_wait, policy.base_wait * 2.0 ** min(max(attempt - 1, 0), 64))
    return clock.random() * ceiling
