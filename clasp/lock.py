from dataclasses import dataclass, field

from clasp.errors import LockBypass
from clasp.secrets import contains_secret

HELD = "Value lock held this reply."


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    text: str
    seal: object = field(repr=False, compare=False)


class ValueLock:
    """L4. A reply is judged here before it can leave."""

    def __init__(self) -> None:
        self.seal = object()

    def judge(self, reply: str, forbid: tuple[str, ...], secrets: tuple[str, ...]) -> Verdict:
        lowered = reply.lower()
        for phrase in forbid:
            if phrase.lower() in lowered:
                return Verdict(False, HELD, self.seal)
        if contains_secret(reply, secrets):
            return Verdict(False, HELD, self.seal)
        return Verdict(True, reply, self.seal)


class Thinker:
    """Independent of the model. A reply with the wrong seal does not leave."""

    def __init__(self, lock: ValueLock) -> None:
        self._lock = lock

    def release(self, verdict: Verdict) -> str:
        if not isinstance(verdict, Verdict) or verdict.seal is not self._lock.seal:
            raise LockBypass("L4 did not judge this reply. Nothing leaves the cortex.")
        return verdict.text
