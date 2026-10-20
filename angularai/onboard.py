"""``angularai setup`` and ``angularai model``: save a model and its key, test it, pick one.

Interactive on a terminal. Without one, flags say everything, and the key comes on
stdin with ``--key-stdin``. A key is never an argument, so it never reaches the
process list or the shell history. A key is never printed: ``mask`` is all that
is shown.
"""

import argparse
import getpass
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from angularai.errors import (
    AngularAIError,
    AuthFailed,
    Forbidden,
    InputRejected,
    NotFound,
    QuotaExceeded,
    SubstrateError,
    Timeout,
    Unreachable,
)
from angularai.models import SubstrateKind
from angularai.profiles import (
    KEY_ENVS,
    MODEL_ENV,
    Book,
    Profile,
    Vault,
    add_profile,
    anything_configured,
    check_key,
    check_model,
    check_name,
    check_url,
    free_name,
    key_source,
    load_book,
    mask,
    remove_profile,
    storage_words,
    use_profile,
)
from angularai.store import init_cortex, open_store
from angularai.substrate import ANTHROPIC_URL, OPENAI_URL, probe

OLLAMA_URL = "http://127.0.0.1:11434/v1"
Probe = Callable[[SubstrateKind, str, str | None, str], None]


@dataclass(frozen=True)
class Choice:
    flag: str
    label: str
    kind: SubstrateKind
    base_url: str | None
    needs_key: bool
    example: str


CHOICES = (
    Choice("openai", "OpenAI", "openai", None, True, "gpt-4o-mini"),
    Choice("anthropic", "Anthropic", "anthropic", None, True, "claude-sonnet-4-5"),
    Choice("ollama", "Ollama (local, no key)", "compatible", OLLAMA_URL, False, "llama3.2"),
    Choice(
        "compatible",
        "OpenAI-compatible (LM Studio, vLLM, a proxy)",
        "compatible",
        None,
        False,
        "local-model",
    ),
)


@dataclass
class Terminal:
    """Everything setup reads or writes. Tests pass their own."""

    interactive: bool
    ask: Callable[[str], str]
    secret: Callable[[str], str]
    stdin: TextIO
    out: Callable[[str], None]
    err: Callable[[str], None]


def system_terminal() -> Terminal:
    return Terminal(
        interactive=sys.stdin.isatty() and sys.stdout.isatty(),
        ask=_ended(input),
        secret=_ended(getpass.getpass),
        stdin=sys.stdin,
        out=lambda line: print(line, flush=True),
        err=lambda line: print(line, file=sys.stderr, flush=True),
    )


def _ended(read: Callable[[str], str]) -> Callable[[str], str]:
    """End of input (Ctrl-D) ends setup with a sentence, not a traceback."""

    def wrapped(prompt: str) -> str:
        try:
            return read(prompt)
        except EOFError:
            ended = True
        if ended:
            raise InputRejected("Setup ended. Nothing was saved.")
        raise AssertionError("unreachable")  # pragma: no cover

    return wrapped


def _default_probe(kind: SubstrateKind, model: str, base_url: str | None, key: str) -> None:
    probe(kind, model, base_url, key)


# --- argparse ---------------------------------------------------------------


def add_commands(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    setup = sub.add_parser(
        "setup",
        help="Save a model and its key, test it, and make it active.",
        description=(
            "Interactive on a terminal. Without one, give --provider and --model, "
            "and the key on stdin with --key-stdin."
        ),
    )
    _model_flags(setup)

    model = sub.add_parser("model", help="List, add, switch, or remove saved models.")
    model_sub = model.add_subparsers(dest="model_command")
    listing = model_sub.add_parser("list", help="List saved models. The key shows as sk-…a1b2.")
    listing.add_argument("--json", action="store_true", help="One JSON object on stdout.")
    add = model_sub.add_parser("add", help="Save another model. It becomes active with --use.")
    _model_flags(add)
    add.add_argument("--use", action="store_true", help="Make it the active model.")
    use = model_sub.add_parser("use", help="Make a saved model the active one.")
    use.add_argument("model_name", metavar="NAME")
    remove = model_sub.add_parser("remove", help="Remove a saved model and its key.")
    remove.add_argument("model_name", metavar="NAME")


def _model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=[choice.flag for choice in CHOICES], default=None)
    parser.add_argument("--model", dest="model_id", default=None, metavar="ID")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--as", dest="save_as", default=None, metavar="NAME")
    parser.add_argument(
        "--key-stdin",
        action="store_true",
        help="Read the key from stdin. Hidden on a terminal. A key is never an argument.",
    )
    parser.add_argument("--skip-test", action="store_true", help="Save without the test call.")
    parser.add_argument("--replace", action="store_true", help="Replace a saved model.")


# --- commands ---------------------------------------------------------------


def setup_command(
    args: argparse.Namespace,
    environ: Mapping[str, str] = os.environ,
    terminal: Terminal | None = None,
    vault: Vault | None = None,
    tester: Probe = _default_probe,
) -> int:
    if terminal is None and _flagless(args) and tui_ok(environ):
        return _setup_tui(environ, vault, tester, make_active=True)
    term = terminal or system_terminal()
    flow = _Flow(environ, term, vault or Vault(environ), tester)
    return flow.run(args, make_active=True)


def model_command(
    args: argparse.Namespace,
    environ: Mapping[str, str] = os.environ,
    terminal: Terminal | None = None,
    vault: Vault | None = None,
    tester: Probe = _default_probe,
) -> int:
    command = args.model_command
    if terminal is None and command is None and tui_ok(environ):
        from angularai.setup_tui import run_manage_tui

        run_manage_tui(environ, vault, tester=tester)
        return 0
    if terminal is None and command == "add" and _flagless(args) and tui_ok(environ):
        active = load_book(environ).active is None
        return _setup_tui(environ, vault, tester, make_active=bool(args.use) or active)
    term = terminal or system_terminal()
    keys = vault or Vault(environ)
    if command is None or command == "list":
        return _list(environ, term, as_json=bool(getattr(args, "json", False)))
    if command == "add":
        return _Flow(environ, term, keys, tester).run(args, make_active=bool(args.use))
    if command == "use":
        found = use_profile(environ, args.model_name)
        term.out(f"Active model: {found.name}  {found.provider}  {found.model}")
        _note_overrides(found, environ, term)
        return 0
    if command == "remove":
        gone, was_active = remove_profile(environ, keys, args.model_name)
        where = {"keyring": " and its key in the OS keyring", "file": " and its saved key"}
        term.out(f"Removed {gone.name}{where.get(gone.storage, '')}.")
        if was_active:
            term.out("No model is active. Pick one with: angularai model use NAME")
        return 0
    term.err("Unknown model command.")
    return 2


def tui_ok(environ: Mapping[str, str]) -> bool:
    """The setup and model screens: a real terminal on both ends, not CI, not TERM=dumb."""

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if environ.get("CI", "").strip() not in {"", "0", "false"}:
        return False
    return environ.get("TERM", "dumb") not in {"", "dumb"}


def _flagless(args: argparse.Namespace) -> bool:
    """No flag that answers a question: the screens ask them all."""

    return (
        getattr(args, "provider", None) is None
        and getattr(args, "model_id", None) is None
        and getattr(args, "base_url", None) is None
        and getattr(args, "save_as", None) is None
        and not getattr(args, "key_stdin", False)
    )


def _setup_tui(
    environ: Mapping[str, str],
    vault: Vault | None,
    tester: Probe,
    *,
    make_active: bool,
    first_use: bool = False,
) -> int:
    """The setup screens. What was saved is printed after, so it stays in the scrollback."""

    from angularai.setup_tui import run_setup_tui

    result = run_setup_tui(
        environ, vault, tester=tester, first_use=first_use, make_active=make_active
    )
    if result is None:
        print("Nothing was saved.", file=sys.stderr)
        return 1
    saved = result.profile
    print(f"Saved {saved.name}  {saved.provider}  {saved.model}")
    print(result.where)
    if result.env_note:
        print(result.env_note)
    if load_book(environ).active == saved.name:
        print(f"Active model: {saved.name}")
    else:
        print(f"Switch with: angularai model use {saved.name}")
    return 0


def _note_overrides(found: Profile, environ: Mapping[str, str], term: Terminal) -> None:
    picked = environ.get(MODEL_ENV, "").strip()
    if picked and picked != found.name:
        term.out(f"{MODEL_ENV} is set in this shell and picks another saved model.")
    env_name = KEY_ENVS[found.provider]
    if environ.get(env_name, "").strip():
        term.out(f"{env_name} is set in this shell and wins over the saved key.")


def _list(environ: Mapping[str, str], term: Terminal, *, as_json: bool) -> int:
    book = load_book(environ)
    picked = environ.get(MODEL_ENV, "").strip() or None
    if as_json:
        rows = [
            {
                "name": item.name,
                "provider": item.provider,
                "model": item.model,
                "base_url": item.base_url,
                "active": item.name == book.active,
                "key": {"storage": item.storage, "hint": item.hint or None},
                "key_source": key_source(item.provider, item, environ),
            }
            for item in book.profiles
        ]
        term.out(json.dumps({"active": book.active, "override": picked, "models": rows}))
        return 0
    if not book.profiles:
        term.out("No saved models. Run: angularai setup")
        return 0
    term.out(format_book(book, environ))
    if picked:
        term.out(f"{MODEL_ENV} picks {picked} in this shell.")
    return 0


def format_book(book: Book, environ: Mapping[str, str]) -> str:
    """One line a saved model, the active one starred. A key shows as its hint only."""

    rows = [("name", "provider", "model", "key", "base url")]
    for item in book.profiles:
        key = key_source(item.provider, item, environ)
        rows.append((item.name, item.provider, item.model, key, item.base_url or ""))
    widths = [max(len(row[index]) for row in rows) for index in range(5)]
    lines = []
    for number, row in enumerate(rows):
        mark = "*" if number and row[0] == book.active else " "
        cells = "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        lines.append(f"{mark} {cells}".rstrip())
    return "\n".join(lines)


# --- the flow ---------------------------------------------------------------


class _Flow:
    def __init__(
        self, environ: Mapping[str, str], term: Terminal, vault: Vault, tester: Probe
    ) -> None:
        self.environ = environ
        self.term = term
        self.vault = vault
        self.tester = tester

    def run(self, args: argparse.Namespace, *, make_active: bool) -> int:
        if not self.term.interactive and (args.provider is None or args.model_id is None):
            raise InputRejected(
                "angularai setup needs a terminal. Without one, pass --provider and --model "
                "(and --base-url for compatible), with the key on stdin: --key-stdin."
            )
        if self.term.interactive:
            self.term.out("AngularAI setup. The key is hidden as you type, and never shown.")
        choice = self._choice(args.provider)
        base_url = self._base_url(choice, args.base_url)
        model = self._model(choice, args.model_id)
        key = self._key(choice, bool(args.key_stdin))
        env_name = KEY_ENVS[choice.kind]
        env_key = self.environ.get(env_name, "").strip()
        if key is None and choice.needs_key and not env_key:
            raise InputRejected(
                f"{choice.label} needs a key. Pass it on stdin with --key-stdin, "
                f"or set {env_name}. A key is never an argument."
            )
        if not args.skip_test and not self._test(choice, model, base_url, key or env_key):
            return 1
        name = self._name(choice, args.save_as, bool(args.replace))
        profile = Profile(name, choice.kind, model, base_url, "none", "")
        saved = add_profile(self.environ, self.vault, profile, key, make_active=make_active)
        self._report(saved, key is not None, env_name, env_key)
        return 0

    # Each step: a flag when given, else a question on a terminal.

    def _choice(self, flag: str | None) -> Choice:
        if flag is not None:
            return next(choice for choice in CHOICES if choice.flag == flag)
        self.term.out("Provider")
        for number, choice in enumerate(CHOICES, start=1):
            self.term.out(f"  {number}  {choice.label}")
        for _ in range(3):
            answer = self.term.ask("Choose [1]: ").strip() or "1"
            if answer.isdigit() and 1 <= int(answer) <= len(CHOICES):
                return CHOICES[int(answer) - 1]
            matched = [c for c in CHOICES if c.flag == answer.lower()]
            if matched:
                return matched[0]
            self.term.err(f"Choose 1 to {len(CHOICES)}.")
        raise InputRejected("No provider chosen. Nothing was saved.")

    def _base_url(self, choice: Choice, flag: str | None) -> str | None:
        if flag is not None:
            return check_url(flag)
        if choice.kind != "compatible":
            return None
        if not self.term.interactive:
            if choice.base_url is None:
                raise InputRejected("A compatible model needs --base-url.")
            return choice.base_url
        default = choice.base_url
        prompt = f"Base URL [{default}]: " if default else "Base URL (http://host:port/v1): "
        return self._retry(lambda: check_url(self.term.ask(prompt).strip() or (default or "")))

    def _model(self, choice: Choice, flag: str | None) -> str:
        if flag is not None:
            return check_model(flag)
        return self._retry(
            lambda: check_model(self.term.ask(f"Model id (for example {choice.example}): "))
        )

    def _key(self, choice: Choice, from_stdin: bool) -> str | None:
        env_name = KEY_ENVS[choice.kind]
        if from_stdin:
            # A key typed at a terminal is hidden, even when stdout is a pipe.
            if self.term.interactive or self.term.stdin.isatty():
                raw = self.term.secret("Key (hidden): ")
            else:
                raw = self.term.stdin.readline()
            if not raw.strip():
                raise InputRejected("No key arrived on stdin. Nothing was saved.")
            return check_key(raw)
        if not self.term.interactive:
            return None
        env_set = bool(self.environ.get(env_name, "").strip())
        if not choice.needs_key:
            prompt = "Key (hidden, Enter for none): "
        elif env_set:
            prompt = f"Key (hidden, Enter to use {env_name} and save none): "
        else:
            prompt = "Key (hidden): "
        for _ in range(3):
            raw = self.term.secret(prompt)
            if not raw.strip():
                if choice.needs_key and not env_set:
                    self.term.err(f"{choice.label} needs a key.")
                    continue
                return None
            try:
                key = check_key(raw)
            except InputRejected as exc:
                self.term.err(str(exc))
                continue
            self.term.out(f"Key {mask(key)}")
            return key
        raise InputRejected("No key entered. Nothing was saved.")

    def _test(self, choice: Choice, model: str, base_url: str | None, key: str) -> bool:
        where = base_url or (OPENAI_URL if choice.kind == "openai" else ANTHROPIC_URL)
        self.term.out(f"Testing {choice.flag} / {model} at {where} ...")
        started = time.monotonic()
        try:
            self.tester(choice.kind, model, base_url, key)
        except SubstrateError as exc:
            self.term.err(f"The test failed. {_explain(exc, where)}")
            if self.term.interactive and self._yes("Save anyway? [y/N]: ", default=False):
                return True
            self.term.err("Nothing was saved.")
            return False
        self.term.out(f"Connected in {time.monotonic() - started:.1f} s.")
        return True

    def _name(self, choice: Choice, flag: str | None, replace: bool) -> str:
        book = load_book(self.environ)
        if flag is not None:
            name = check_name(flag)
        elif self.term.interactive:
            default = choice.flag
            prompt = f"Save as [{default}]: "
            name = self._retry(lambda: check_name(self.term.ask(prompt).strip() or default))
        else:
            name = free_name(book, choice.flag) if not replace else choice.flag
        if book.get(name) is None or replace:
            return name
        question = f"Replace saved model {name}? [y/N]: "
        if self.term.interactive and self._yes(question, default=False):
            return name
        raise InputRejected(f"A saved model named {name} exists. Pass --replace, or --as NAME.")

    def _report(self, saved: Profile, has_key: bool, env_name: str, env_key: str) -> None:
        if has_key:
            where = storage_words(saved.storage, self.environ)
            self.term.out(f"Key {saved.hint} saved in {where}.")
        elif env_key:
            self.term.out(f"No key saved. {env_name} is read from the environment.")
        else:
            self.term.out("No key saved.")
        if has_key and env_key:
            self.term.out(f"{env_name} is set in this shell and wins over the saved key.")
        book = load_book(self.environ)
        if book.active == saved.name:
            self.term.out(f"Active model: {saved.name}  {saved.provider}  {saved.model}")
        else:
            self.term.out(f"Saved {saved.name}. Switch with: angularai model use {saved.name}")

    def _yes(self, prompt: str, *, default: bool) -> bool:
        answer = self.term.ask(prompt).strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes"}

    def _retry(self, step: Callable[[], str]) -> str:
        for attempt in range(3):
            try:
                return step()
            except InputRejected as exc:
                if attempt == 2:
                    raise
                self.term.err(str(exc))
        raise AssertionError("unreachable")  # pragma: no cover


def _explain(exc: SubstrateError, where: str) -> str:
    """The provider's typed error, and what to do about it. Never the key."""

    if isinstance(exc, AuthFailed):
        return f"{exc} Check the key and enter it again."
    if isinstance(exc, Forbidden):
        return f"{exc} The key is valid but cannot use this model."
    if isinstance(exc, QuotaExceeded):
        return f"{exc}"
    if isinstance(exc, NotFound):
        return f"{exc}"
    if isinstance(exc, Timeout):
        return f"Nothing answered at {where} in time. Is the server running?"
    if isinstance(exc, Unreachable):
        return f"Nothing answered at {where}. Is the server running, and the URL right?"
    return str(exc)


# --- first use --------------------------------------------------------------


def offer_setup(
    root: Path,
    environ: Mapping[str, str] = os.environ,
    terminal: Terminal | None = None,
    vault: Vault | None = None,
    tester: Probe = _default_probe,
) -> str | None:
    """On a terminal with nothing configured, offer setup once before the command runs.

    Nothing configured: no saved model, no provider key or ANGULARAI_MODEL in the
    environment, and a cortex that is missing or not wired. Never in CI. On a real
    terminal it is the setup screens; "cancelled" when they were left without saving.
    """

    term = terminal or system_terminal()
    if not term.interactive or environ.get("CI", "").strip():
        return None
    if anything_configured(environ) or _wired(root):
        return None
    if terminal is None and tui_ok(environ):
        if _setup_tui(environ, vault, tester, make_active=True, first_use=True) != 0:
            return "cancelled"
        if not (root / "cortex.json").exists():
            init_cortex(root)
            print(f"Cortex is at {root}")
        return None
    term.out("No model is set up yet.")
    answer = term.ask("Set one up now? [Y/n]: ").strip().lower()
    if answer not in {"", "y", "yes"}:
        return None
    args = argparse.Namespace(
        provider=None,
        model_id=None,
        base_url=None,
        save_as=None,
        key_stdin=False,
        skip_test=False,
        replace=False,
    )
    if _Flow(environ, term, vault or Vault(environ), tester).run(args, make_active=True) != 0:
        return None
    if not (root / "cortex.json").exists():
        init_cortex(root)
        term.out(f"Cortex is at {root}")
    return None


def _wired(root: Path) -> bool:
    try:
        return open_store(root).config().substrate is not None
    except AngularAIError:
        return False
