import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from angularai import __version__
from angularai.agent import run_agent
from angularai.configure import add_lock, remove_lock, set_budget, wire
from angularai.errors import AngularAIError
from angularai.models import SubstrateKind
from angularai.pilot import pass_turn
from angularai.present import format_banner, format_locks, format_recall, format_traces
from angularai.session import cortex_status, session_command
from angularai.store import Store, init_cortex, open_store
from angularai.substrate import build_substrate, key_env, secret_values
from angularai.tui import run_tui

_DESCRIPTION = (
    "A cortex in front of a language model. "
    "The model sees a bounded workspace. What the agent knows stays on disk."
)
ReplAction = Literal["quit", "continue"]


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        cortex, rest = _extract_cortex(raw)
        args = _parser().parse_args(rest)
        return _dispatch(args, cortex)
    except AngularAIError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace, cortex: Path | None) -> int:
    root = _root(cortex)
    command = args.command
    if command is None:
        return _open_session(open_store(root))
    if command == "init":
        return _init(root)
    if command == "wire":
        return _wire(root, args)
    if command == "say":
        return _say(root, " ".join(args.text))
    if command == "agent":
        return _agent(root, args)
    if command == "cortex":
        print(format_traces(open_store(root).view()))
        return 0
    if command == "recall":
        print(format_recall(open_store(root).view(), " ".join(args.query)))
        return 0
    if command == "status":
        print(_status(open_store(root)))
        return 0
    if command == "budget":
        return _budget(root, args.tokens)
    if command == "lock":
        return _lock(root, args)
    print(f"Unknown command {command}.", file=sys.stderr)
    return 2


def _init(root: Path) -> int:
    init_cortex(root)
    print(f"Cortex is at {root}")
    print("Wire a model with: angularai wire openai --model <name>")
    return 0


def _wire(root: Path, args: argparse.Namespace) -> int:
    kind = _kind(args.kind)
    updated = wire(open_store(root), kind, args.model, args.base_url)
    env_name = key_env(kind)
    model = updated.model or args.model
    print(f"Wired {kind} / {model}")
    print(f"The key is read from {env_name} when you speak. It is not stored.")
    return 0


def _say(root: Path, text: str) -> int:
    store = open_store(root)
    reply = _speak(store, text)
    print(reply)
    return 0


def _agent(root: Path, args: argparse.Namespace) -> int:
    store = open_store(root)
    directory = Path.cwd() if args.root is None else args.root
    print(f"root  {directory.expanduser().resolve()}", file=sys.stderr)
    print(f"run   {'on' if args.allow_run else 'off'}", file=sys.stderr)
    substrate = build_substrate(store.config(), os.environ)
    result = run_agent(
        store,
        substrate,
        " ".join(args.task),
        directory,
        secret_values(os.environ),
        args.steps,
        lambda line: print(line, file=sys.stderr),
        args.allow_run,
    )
    if result.stop == "budget":
        print("The step budget is spent.", file=sys.stderr)
    print(result.reply)
    return 0


def _budget(root: Path, tokens: int | None) -> int:
    store = open_store(root)
    if tokens is None:
        print(f"{store.config().workspace_budget} estimated tokens")
        return 0
    updated = set_budget(store, tokens)
    print(f"Workspace budget is {updated.workspace_budget} estimated tokens.")
    return 0


def _lock(root: Path, args: argparse.Namespace) -> int:
    store = open_store(root)
    if args.lock_command is None:
        print(format_locks(store.config().forbid))
        return 0
    phrase = " ".join(args.phrase)
    if args.lock_command == "add":
        before = len(store.config().forbid)
        updated = add_lock(store, phrase)
        if len(updated.forbid) == before:
            print(f"That lock is already set: {phrase}")
        else:
            print(f"Lock set. The reply cannot contain: {phrase}")
        return 0
    if args.lock_command == "remove":
        remove_lock(store, phrase)
        print("Lock removed.")
        return 0
    print("Unknown lock command.", file=sys.stderr)
    return 2


def _open_session(store: Store) -> int:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return run_tui(store, _speak)
    return _repl(store)


def _repl(store: Store) -> int:
    print(format_banner(store.config(), store.view()))
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        try:
            action, output = repl_line(store, line)
        except AngularAIError as exc:
            print(exc, file=sys.stderr)
            continue
        if output:
            print(output)
        if action == "quit":
            return 0


def repl_line(store: Store, line: str) -> tuple[ReplAction, str | None]:
    action, payload = session_command(store, line, _status(store))
    if action == "speak":
        if payload is None:
            return "continue", None
        print("integrating", file=sys.stderr)
        return "continue", _speak(store, payload)
    if action == "quit":
        return "quit", None
    return "continue", payload


def _speak(store: Store, text: str) -> str:
    substrate = build_substrate(store.config(), os.environ)
    result = pass_turn(store, substrate, text, secret_values(os.environ))
    return result.reply


def _status(store: Store) -> str:
    return cortex_status(store, os.environ)


def _root(cortex: Path | None) -> Path:
    if cortex is not None:
        return cortex.expanduser()
    env = os.environ.get("ANGULARAI_CORTEX")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".angularai" / "default"


def _kind(value: str) -> SubstrateKind:
    if value == "openai":
        return "openai"
    if value == "anthropic":
        return "anthropic"
    if value == "compatible":
        return "compatible"
    raise AngularAIError(f"Unknown substrate {value}.")


def _extract_cortex(argv: list[str]) -> tuple[Path | None, list[str]]:
    pre = argparse.ArgumentParser(prog="angularai", add_help=False)
    pre.add_argument("--cortex", type=Path, default=None)
    known, rest = pre.parse_known_args(argv)
    cortex = known.cortex
    if isinstance(cortex, Path):
        return cortex, rest
    return None, rest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="angularai",
        description=_DESCRIPTION,
        epilog=(
            "With no command, a terminal opens the session. "
            "angularai say is one turn. angularai agent works a directory, one tool a pass. "
            "The reply is the only thing either prints. "
            "--cortex PATH can be placed anywhere. The default cortex is ~/.angularai/default."
        ),
    )
    parser.add_argument("--version", action="version", version=f"angularai {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Create a cortex on disk.")

    wire_parser = sub.add_parser("wire", help="Connect this cortex to a model API.")
    wire_parser.add_argument("kind", choices=("openai", "anthropic", "compatible"))
    wire_parser.add_argument("--model", required=True)
    wire_parser.add_argument("--base-url", default=None)

    say = sub.add_parser("say", help="Speak one turn.")
    say.add_argument("text", nargs="+")

    agent = sub.add_parser("agent", help="Work a task inside a directory.")
    agent.add_argument("task", nargs="+")
    agent.add_argument("--root", type=Path, default=None)
    agent.add_argument("--steps", type=int, default=8)
    agent.add_argument(
        "--allow-run",
        action="store_true",
        help="Let the agent start allowlisted programs. Off by default.",
    )

    sub.add_parser("cortex", help="List what the cortex currently knows.")
    recall = sub.add_parser("recall", help="Find traces without calling a model.")
    recall.add_argument("query", nargs="+")
    sub.add_parser("status", help="Show the budget and the last pass.")

    budget = sub.add_parser("budget", help="Show or set the workspace budget.")
    budget.add_argument("tokens", nargs="?", type=int)

    lock = sub.add_parser("lock", help="Show or change value-lock phrases.")
    lock_sub = lock.add_subparsers(dest="lock_command")
    add = lock_sub.add_parser("add", help="Hold any reply that contains this phrase.")
    add.add_argument("phrase", nargs="+")
    remove = lock_sub.add_parser("remove", help="Remove a lock phrase.")
    remove.add_argument("phrase", nargs="+")
    return parser
