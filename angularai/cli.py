import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from angularai import __version__
from angularai import run as run_loop
from angularai.agent import run_agent
from angularai.configure import add_lock, remove_lock, set_budget, wire
from angularai.errors import AngularAIError, InputRejected, KeyMissing, NoCortex
from angularai.index import CortexIndex, index_from_env
from angularai.models import SubstrateKind
from angularai.names import cortex_home, label, list_names, named_root, resolve_root
from angularai.onboard import add_commands, model_command, offer_setup, setup_command
from angularai.peers import (
    DEFAULT_TOKENS,
    DEFAULT_TURNS,
    Cortex,
    Line,
    ask,
    converse,
    stop_notice,
)
from angularai.pilot import pass_turn, screen
from angularai.present import format_banner, format_locks, format_recall, format_traces
from angularai.pricing import parse_price, price_for
from angularai.profiles import anything_configured
from angularai.semantic import SemanticRecall, embedder_from_env, minimum_from_env
from angularai.session import cortex_status, session_command
from angularai.store import CORRUPT, Store, init_cortex, open_store
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
        cortex, name, rest = _extract_cortex(raw)
        args = _parser().parse_args(rest)
        return _dispatch(args, cortex, name)
    except AngularAIError as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace, cortex: Path | None, name: str | None) -> int:
    root = _root(cortex, name)
    command = args.command
    if command == "setup":
        return setup_command(args)
    if command == "model":
        return model_command(args)
    if command is None:
        _boot(root)
    if command == "run" and (args.resume or args.list):
        return _run(root, args)
    if command in {None, "say", "agent", "run"} and offer_setup(root) == "cancelled":
        print("Setup was left. Run angularai setup when you are ready.", file=sys.stderr)
        return 0 if command is None else 1
    if command is None:
        return _open_session(_open(root))
    if command == "repair":
        return _repair(root)
    if command == "init":
        return _init(root, name)
    if command == "ask":
        return _ask(root, name, args)
    if command == "converse":
        return _converse(root, name, args)
    if command == "cortexes":
        return _cortexes()
    if command == "wire":
        return _wire(root, args)
    if command == "say":
        return _say(root, " ".join(args.text))
    if command == "agent":
        return _agent(root, args)
    if command == "run":
        return _run(root, args)
    if command == "cortex":
        print(format_traces(_open(root).view()))
        return 0
    if command == "recall":
        return _recall(_open(root), " ".join(args.query), args.lexical)
    if command == "status":
        print(_status(_open(root)))
        return 0
    if command == "budget":
        return _budget(root, args.tokens)
    if command == "lock":
        return _lock(root, args)
    print(f"Unknown command {command}.", file=sys.stderr)
    return 2


def _open(root: Path) -> Store:
    return open_store(root, warn=lambda notice: print(notice, file=sys.stderr))


def _repair(root: Path) -> int:
    store = open_store(root)
    result = store.repair()
    if result.moved == 0:
        print(f"The log is clean. {result.kept} passes.")
        return 0
    noun = "line" if result.moved == 1 else "lines"
    moved = f"Moved {result.moved} unreadable {noun} to {store.root / CORRUPT}."
    print(f"{moved} {result.kept} passes kept.")
    return 0


def _init(root: Path, name: str | None) -> int:
    if name is not None:
        cortex_home(os.environ).mkdir(parents=True, exist_ok=True, mode=0o700)
    init_cortex(root)
    print(f"Cortex is at {root}")
    print("Set up a model with: angularai setup")
    return 0


def _wire(root: Path, args: argparse.Namespace) -> int:
    kind = _kind(args.kind)
    updated = wire(_open(root), kind, args.model, args.base_url)
    env_name = key_env(kind)
    model = updated.model or args.model
    print(f"Wired {kind} / {model}")
    print(f"The key is read from {env_name} when you speak. It is not stored.")
    return 0


def _say(root: Path, text: str) -> int:
    store = _open(root)
    reply = _speak(store, text)
    print(reply)
    return 0


def _agent(root: Path, args: argparse.Namespace) -> int:
    store = _open(root)
    directory = Path.cwd() if args.root is None else args.root
    print(f"root  {directory.expanduser().resolve()}", file=sys.stderr)
    print(f"run   {'on' if args.allow_run else 'off'}", file=sys.stderr)
    secrets = secret_values(os.environ)
    task = screen(" ".join(args.task), secrets)
    substrate = build_substrate(store.config(), os.environ)
    result = run_agent(
        store,
        substrate,
        task,
        directory,
        secrets,
        args.steps,
        lambda line: print(line, file=sys.stderr),
        args.allow_run,
    )
    if result.stop == "budget":
        print("The step budget is spent.", file=sys.stderr)
    print(result.reply)
    return 0


def _run(root: Path, args: argparse.Namespace) -> int:
    runs = run_loop.runs_dir(os.environ)
    if args.list:
        for item in run_loop.listing(runs):
            print(f"{item.id}  {item.status}  {item.step}/{item.max_steps}  {item.task[:60]}")
        return 0
    secrets = secret_values(os.environ)
    if args.resume:
        if args.task:
            raise InputRejected("--resume continues a run; it takes no task.")
        state = run_loop.load(runs, args.resume)
        if args.max_steps is not None:
            state = run_loop.raise_limit(state, args.max_steps)
        if args.max_cost is not None:
            state = run_loop.raise_cost(state, args.max_cost)
        root = Path(state.cortex)
    else:
        if not args.task:
            raise InputRejected("Say what the run should do, or --resume a run.")
        directory = Path.cwd() if args.root is None else args.root
        try:
            model = _open(root).config().model or ""
            price = parse_price(args.price) if args.price else price_for(model)
        except ValueError as exc:
            raise InputRejected(str(exc)) from exc
        state = run_loop.start(
            runs,
            task=screen(" ".join(args.task), secrets),
            root=directory,
            cortex=root,
            budget=args.budget,
            max_steps=run_loop.DEFAULT_STEPS if args.max_steps is None else args.max_steps,
            allow_run=args.allow_run,
            native=not args.json_acts,
            files_view=args.files,
            finish_check=args.finish_check == "on",
            finish_checks=args.finish_checks,
            change_recall=args.change_recall == "on",
            price=price,
            max_cost=args.max_cost,
            markers=tuple(m.strip() for m in args.markers.split(",") if m.strip()),
            summary=args.summary or "",
            summary_of=args.summary_of or "",
            finish_refusals=args.finish_refusals,
            search_hits=args.search_hits,
        )
    store = _open(root)
    substrate = build_substrate(store.config(), os.environ)
    print(f"run   {state.id}", file=sys.stderr)
    print(f"root  {state.root}", file=sys.stderr)
    print(f"step  {state.step} of {state.max_steps}, budget {state.budget}", file=sys.stderr)
    state = run_loop.advance(
        runs,
        state,
        store,
        substrate,
        secrets=secrets,
        steps=args.steps,
        index=_index(store, secrets),
        recall=_semantic(store),
        progress=lambda line: print(line, file=sys.stderr),
    )
    if state.status == "running":
        print(f"Paused at step {state.step}. angularai run --resume {state.id}", file=sys.stderr)
    elif state.status == "budget":
        print("The step budget is spent.", file=sys.stderr)
    elif state.status == "cost":
        print(
            f"The cost limit stops the run. angularai run --resume {state.id} --max-cost N",
            file=sys.stderr,
        )
    if state.price is not None:
        print(f"cost  ${state.cost:.4f}", file=sys.stderr)
    print(state.note)
    return 0


def _ask(root: Path, name: str | None, args: argparse.Namespace) -> int:
    asker_store = _open(root)
    target_root = _peer_root(root, args.to)
    target_store = _open_named(args.to, target_root)
    secrets = secret_values(os.environ)
    question = screen(" ".join(args.text), secrets)
    substrate = build_substrate(target_store.config(), os.environ)
    asker = Cortex(name or label(root), asker_store, None, secrets)
    target = Cortex(args.to, target_store, substrate, secrets, _semantic(target_store))
    print(ask(asker, target, question).reply)
    return 0


def _converse(root: Path, name: str | None, args: argparse.Namespace) -> int:
    opener_store = _open(root)
    other_root = _peer_root(root, args.peer)
    other_store = _open_named(args.peer, other_root)
    secrets = secret_values(os.environ)
    opening = screen(" ".join(args.text), secrets)
    opener_substrate = build_substrate(opener_store.config(), os.environ)
    other_substrate = build_substrate(other_store.config(), os.environ)
    opener = Cortex(
        name or label(root), opener_store, opener_substrate, secrets, _semantic(opener_store)
    )
    other = Cortex(args.peer, other_store, other_substrate, secrets, _semantic(other_store))

    def show(line: Line) -> None:
        print(f"{line.speaker}  {line.text}", flush=True)

    result = converse(opener, other, opening, args.turns, args.tokens, show)
    print(stop_notice(result, args.turns, args.tokens), file=sys.stderr)
    return 0


def _peer_root(root: Path, peer: str) -> Path:
    """The named peer's directory. It must not be this cortex under another path."""

    other = named_root(peer, os.environ)
    if other.exists() and root.exists() and os.path.samefile(other, root):
        raise InputRejected("A cortex cannot talk to itself.")
    return other


def _open_named(name: str, root: Path) -> Store:
    try:
        return _open(root)
    except NoCortex as exc:
        raise NoCortex(f"No cortex named {name}. Run angularai --name {name} init.") from exc


def _cortexes() -> int:
    names = list_names(os.environ)
    if not names:
        print(f"No named cortexes in {cortex_home(os.environ)}.")
        return 0
    print("\n".join(names))
    return 0


def _recall(store: Store, query: str, lexical: bool) -> int:
    view = store.view()
    recall = None if lexical else _semantic(store)
    meaning = None if recall is None else recall.ranks(view, query)
    print(format_recall(view, query, meaning=meaning))
    return 0


def _semantic(store: Store) -> SemanticRecall | None:
    """Recall by meaning, when ANGULARAI_EMBEDDINGS turns it on. A missing key is a note."""

    def warn(notice: str) -> None:
        print(notice, file=sys.stderr)

    minimum = minimum_from_env(os.environ)
    try:
        embedder = embedder_from_env(store.config(), os.environ)
    except KeyMissing as exc:
        warn(f"Embeddings are unavailable: {exc} Recall is lexical.")
        return None
    if embedder is None:
        return None
    return SemanticRecall(
        store.root, embedder, minimum=minimum, secrets=secret_values(os.environ), warn=warn
    )


def _budget(root: Path, tokens: int | None) -> int:
    store = _open(root)
    if tokens is None:
        print(f"{store.config().workspace_budget} estimated tokens")
        return 0
    updated = set_budget(store, tokens)
    print(f"Workspace budget is {updated.workspace_budget} estimated tokens.")
    return 0


def _lock(root: Path, args: argparse.Namespace) -> int:
    store = _open(root)
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
    store.passes()
    if sys.stdin.isatty() and sys.stdout.isatty():
        return run_tui(store, _speak)
    return _repl(store)


def _boot(root: Path) -> None:
    """The boot mark, before the setup offer and the session.

    It plays only when a session can follow: the cortex exists, or nothing is set up
    and the offer can make it. Imported only here so no other command pays for it.
    """

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    if not (root / "cortex.json").exists() and anything_configured(os.environ):
        return
    from angularai.boot import run

    run(__version__)


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
    secrets = secret_values(os.environ)
    screen(text, secrets)
    substrate = build_substrate(store.config(), os.environ)
    index = _index(store, secrets)
    try:
        recall = None if index is not None else _semantic(store)
        result = pass_turn(store, substrate, text, secrets, recall=recall, index=index)
    finally:
        if index is not None:
            index.close()
    return result.reply


def _index(store: Store, secrets: tuple[str, ...]) -> CortexIndex | None:
    """Cortex v2 recall, when ANGULARAI_INDEX is on. Embeddings as ANGULARAI_EMBEDDINGS."""

    def warn(notice: str) -> None:
        print(notice, file=sys.stderr)

    if os.environ.get("ANGULARAI_INDEX", "").strip().lower() in ("", "off"):
        return None
    try:
        embedder = embedder_from_env(store.config(), os.environ)
    except KeyMissing as exc:
        warn(f"Embeddings are unavailable: {exc} Recall is lexical.")
        embedder = None
    return index_from_env(store.root, os.environ, embedder, secrets, warn)


def _status(store: Store) -> str:
    return cortex_status(store, os.environ)


def _root(cortex: Path | None, name: str | None = None) -> Path:
    return resolve_root(cortex, name, os.environ)


def _kind(value: str) -> SubstrateKind:
    if value == "openai":
        return "openai"
    if value == "anthropic":
        return "anthropic"
    if value == "compatible":
        return "compatible"
    raise AngularAIError(f"Unknown substrate {value}.")


def _extract_cortex(argv: list[str]) -> tuple[Path | None, str | None, list[str]]:
    pre = argparse.ArgumentParser(prog="angularai", add_help=False)
    pre.add_argument("--cortex", type=Path, default=None)
    pre.add_argument("--name", default=None)
    known, rest = pre.parse_known_args(argv)
    cortex = known.cortex if isinstance(known.cortex, Path) else None
    name = known.name if isinstance(known.name, str) else None
    return cortex, name, rest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="angularai",
        description=_DESCRIPTION,
        epilog=(
            "With no command, a terminal opens the session. "
            "angularai say is one turn. angularai agent works a directory, one tool a pass. "
            "The reply is the only thing either prints. "
            "--cortex PATH can be placed anywhere. The default cortex is ~/.angularai/default. "
            "--name NAME is the cortex NAME beside it, and can be placed anywhere too."
        ),
    )
    parser.add_argument("--version", action="version", version=f"angularai {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Create a cortex on disk.")
    add_commands(sub)

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

    run = sub.add_parser("run", help="Work a long task, checkpointed after every step.")
    run.add_argument("task", nargs="*")
    run.add_argument("--resume", metavar="ID", default=None, help="Continue a run.")
    run.add_argument("--list", action="store_true", help="List the runs.")
    run.add_argument("--root", type=Path, default=None)
    run.add_argument(
        "--max-steps", type=int, default=None, help="The run's step limit. Default 50."
    )
    run.add_argument(
        "--steps", type=int, default=None, help="Steps this call may take, then pause."
    )
    run.add_argument(
        "--budget",
        type=int,
        default=run_loop.DEFAULT_BUDGET,
        help="Tokens a step's workspace may take. The window is fixed.",
    )
    run.add_argument(
        "--allow-run",
        action="store_true",
        help="Let the run start allowlisted programs. Off by default.",
    )
    run.add_argument(
        "--json-acts", action="store_true", help="Use the JSON act protocol, not tool calls."
    )
    run.add_argument(
        "--files",
        choices=run_loop.FILES_VIEWS,
        default="list",
        help="How working memory shows the files touched so far.",
    )
    run.add_argument(
        "--finish-check",
        choices=("on", "off"),
        default="on",
        help="On: the first finish gets a check of the task and TODOs; a second ends.",
    )
    run.add_argument(
        "--markers",
        default=",".join(run_loop.DEFAULT_MARKERS),
        help="Comma-separated words that mark work left (default TODO,FIXME). A finish is "
        "refused while any line under the root holds one. Empty turns the count off.",
    )
    run.add_argument(
        "--summary",
        default=None,
        metavar="PATH",
        help="A file the run must end with: finish is refused until it is written last.",
    )
    run.add_argument(
        "--summary-of",
        default=None,
        metavar="GLOB",
        help="The files --summary must list, each with a specific description.",
    )
    run.add_argument(
        "--finish-refusals",
        type=int,
        default=5,
        help="Refused finishes in a row, with no change between, that stall the run.",
    )
    run.add_argument(
        "--search-hits",
        type=int,
        default=30,
        help="Lines a search returns at most (1-200); the window still clips the result.",
    )
    run.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Stop before a step that could take the run's spend over this many dollars.",
    )
    run.add_argument(
        "--price",
        default=None,
        metavar="IN,OUT,CACHED",
        help="Dollars per 1M tokens, for a model AngularAI has no price for.",
    )
    run.add_argument(
        "--finish-checks",
        type=int,
        default=2,
        help="At most this many finish checks a run; a finish after them ends it.",
    )
    run.add_argument(
        "--change-recall",
        choices=("on", "off"),
        default="on",
        help="On: the text of each edit and write is kept in the log for recall.",
    )

    ask_parser = sub.add_parser("ask", help="Ask another cortex one question.")
    ask_parser.add_argument("--to", required=True, metavar="NAME")
    ask_parser.add_argument("text", nargs="+")

    talk = sub.add_parser("converse", help="Let this cortex and another take turns.")
    talk.add_argument("--with", dest="peer", required=True, metavar="NAME")
    talk.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    talk.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    talk.add_argument("text", nargs="+")

    sub.add_parser("cortexes", help="List the named cortexes.")

    sub.add_parser("cortex", help="List what the cortex currently knows.")
    recall = sub.add_parser(
        "recall",
        help="Find traces. Lexical, and by meaning when ANGULARAI_EMBEDDINGS is set.",
    )
    recall.add_argument("query", nargs="+")
    recall.add_argument("--lexical", action="store_true", help="Match words only. Nothing is sent.")
    sub.add_parser("status", help="Show the budget and the last pass.")
    sub.add_parser("repair", help="Move unreadable log lines to passes.jsonl.corrupt.")

    budget = sub.add_parser("budget", help="Show or set the workspace budget.")
    budget.add_argument("tokens", nargs="?", type=int)

    lock = sub.add_parser("lock", help="Show or change value-lock phrases.")
    lock_sub = lock.add_subparsers(dest="lock_command")
    add = lock_sub.add_parser("add", help="Hold any reply that contains this phrase.")
    add.add_argument("phrase", nargs="+")
    remove = lock_sub.add_parser("remove", help="Remove a lock phrase.")
    remove.add_argument("phrase", nargs="+")
    return parser
