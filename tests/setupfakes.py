"""Fakes for setup: an in-memory keyring and a scripted terminal. No real I/O."""

import io
from collections.abc import Sequence

from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from angularai.onboard import Terminal

# Obviously not a real key. Long enough to be redacted as an sk- key shape.
KEY = "sk-demo-0000000000000000a1b2"


class FakeKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise PasswordDeleteError("not found")
        del self.store[(service, username)]


class BrokenKeyring(FakeKeyring):
    """Refuses every write and read, the way a locked keyring does. It echoes input."""

    def set_password(self, service: str, username: str, password: str) -> None:
        raise RuntimeError(f"locked: {password}")

    def get_password(self, service: str, username: str) -> str | None:
        raise RuntimeError("locked")


class SilentKeyring(FakeKeyring):
    """Accepts a write and keeps nothing, like a chainer with no backends."""

    def set_password(self, service: str, username: str, password: str) -> None:
        return None


class FakeTerminal:
    def __init__(
        self,
        answers: Sequence[str] = (),
        secrets: Sequence[str] = (),
        *,
        interactive: bool = True,
        stdin: str = "",
    ) -> None:
        self.answers = list(answers)
        self.secrets = list(secrets)
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.prompts: list[str] = []
        self.terminal = Terminal(
            interactive=interactive,
            ask=self._ask,
            secret=self._secret,
            stdin=io.StringIO(stdin),
            out=self.lines.append,
            err=self.errors.append,
        )

    def _ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected question: {prompt}")
        return self.answers.pop(0)

    def _secret(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.secrets:
            raise AssertionError(f"unexpected secret prompt: {prompt}")
        return self.secrets.pop(0)

    @property
    def text(self) -> str:
        return "\n".join([*self.prompts, *self.lines, *self.errors])
