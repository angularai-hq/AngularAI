"""Fake substrates for cortexes that talk. No network, no keys."""

from collections.abc import Sequence

from angularai.models import Integration, TraceDraft, Workspace


class Voice:
    """Replies with a fixed script, then with a line that names what it heard."""

    def __init__(self, tag: str, script: Sequence[Integration] = ()) -> None:
        self.tag = tag
        self.seen: list[Workspace] = []
        self._script = list(script)

    def integrate(self, workspace: Workspace) -> Integration:
        self.seen.append(workspace)
        if self._script:
            return self._script.pop(0)
        heard = " ".join(workspace.input_text.split()[:6])
        return Integration(reply=f"{self.tag} heard {heard}", remember=(), supersede=())

    def sent(self) -> str:
        """Everything this substrate was shown, as one text."""

        return "\n".join(item.system + "\n" + item.user for item in self.seen)


def reply(text: str, *remember: str, supersede: tuple[str, ...] = ()) -> Integration:
    return Integration(
        reply=text,
        remember=tuple(TraceDraft(item, ("fact",)) for item in remember),
        supersede=supersede,
    )
