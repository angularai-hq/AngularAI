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
    """No format characters (zero-width and kin), NFKC, one space for any run.

    Dropping a format character can bring a letter and its accent together, so the
    steps repeat until the text stops changing. normalize(normalize(x)) == normalize(x).
    """

    folded = text
    for _ in range(4):
        visible = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
        again = _SPACE.sub(" ", unicodedata.normalize("NFKC", visible)).strip()
        if again == folded:
            break
        folded = again
    return folded


_SPACE = re.compile(r"\s+")

# Cyrillic and Greek letters drawn like Latin ones. NFKC keeps them apart; the lock
# does not. Both the phrase and the reply are folded, so a phrase in Cyrillic still
# matches Cyrillic. The cost is a rare extra hold: a Latin word drawn the same as a
# locked Cyrillic one.
_LOOKALIKE = str.maketrans(
    "аеорсухіјѕԁһԛԝӏАВЕКМНОРСТУХІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοινκρυα",
    "aeopcyxijsdhqwlABEKMHOPCTYXIJSABEZHIKMNOPTYXoivkpua",
)


def _skeleton(text: str) -> str:
    return normalize(text).translate(_LOOKALIKE).casefold().translate(_LOOKALIKE)


def _forbidden(text: str, forbid: tuple[str, ...]) -> bool:
    if not forbid:
        return False
    seen = normalize(text).casefold()
    drawn = _skeleton(text)
    for phrase in forbid:
        if not phrase.strip():
            continue
        if normalize(phrase).casefold() in seen or _skeleton(phrase) in drawn:
            return True
    return False


def _secret(text: str, secrets: tuple[str, ...]) -> bool:
    return contains_secret(text, secrets) or contains_secret(normalize(text), secrets)
