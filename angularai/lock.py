import re
import unicodedata
from dataclasses import dataclass, field

from angularai.errors import LockBypass
from angularai.secrets import contains_secret

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

    def judge(
        self,
        reply: str,
        forbid: tuple[str, ...],
        secrets: tuple[str, ...],
        also: tuple[str, ...] = (),
    ) -> Verdict:
        """Judge the reply, and every text in also. Only the reply is released."""

        for text in (reply, *also):
            if _forbidden(text, forbid) or _secret(text, secrets):
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


def normalize(text: str) -> str:
    """NFKC, no format characters (zero-width and kin), one space for any run."""

    folded = unicodedata.normalize("NFKC", text)
    visible = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    return _SPACE.sub(" ", visible).strip()


_SPACE = re.compile(r"\s+")


def _forbidden(text: str, forbid: tuple[str, ...]) -> bool:
    if not forbid:
        return False
    seen = normalize(text).casefold()
    return any(normalize(phrase).casefold() in seen for phrase in forbid if phrase.strip())


def _secret(text: str, secrets: tuple[str, ...]) -> bool:
    return contains_secret(text, secrets) or contains_secret(normalize(text), secrets)
