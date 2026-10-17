from clasp.models import Integration, TraceDraft, Workspace


class Scripted:
    def __init__(self, reply: Integration | None = None) -> None:
        self.seen: list[Workspace] = []
        self._reply = reply

    def integrate(self, workspace: Workspace) -> Integration:
        self.seen.append(workspace)
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
