from collections.abc import Sequence

from angularai.models import Integration, TraceDraft, Workspace


class Scripted:
    def __init__(
        self,
        reply: Integration | None = None,
        script: Sequence[Integration] = (),
    ) -> None:
        self.seen: list[Workspace] = []
        self._reply = reply
        self._script = list(script)

    def integrate(self, workspace: Workspace) -> Integration:
        self.seen.append(workspace)
        if self._script:
            return self._script.pop(0)
        if self._reply is None:
            raise AssertionError("scripted substrate has no reply")
        return self._reply


def noted(text: str, *wavelengths: str) -> Integration:
    return Integration(
        reply="Noted.",
        remember=(TraceDraft(text, wavelengths),),
        supersede=(),
    )


def said(reply: str) -> Integration:
    return Integration(reply=reply, remember=(), supersede=())
