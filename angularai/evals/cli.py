"""angularai-eval: fetch, reduce, run, judge, report. Local models only.

Paid keys are removed from the environment before any command runs.
"""

import argparse
import json
import os
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import cast

from angularai.evals.client import PAID_KEYS, Chat, EvalError, require_local
from angularai.evals.dataset import DatasetError, fetch, load
from angularai.evals.reduce import read_subset, reduce, write_subset
from angularai.evals.runner import ARMS, Arm, judge, markdown, report, run
from angularai.provider import Policy
from angularai.semantic import EmbeddingsAPI
from angularai.substrate import substrate_for

LOCAL = "http://127.0.0.1:11434/v1"


def scrub(environ: MutableMapping[str, str]) -> None:
    """Paid keys leave the environment. Their values are never read."""

    for name in PAID_KEYS:
        environ.pop(name, None)


def local_embedder(model: str, base_url: str) -> EmbeddingsAPI:
    """An embedding model on the local server. No key is sent."""

    policy = Policy(connect_timeout=10.0, read_timeout=300.0, deadline=600.0, attempts=2)
    return EmbeddingsAPI(
        model=model,
        base_url=base_url,
        api_key="",
        dimensions=None,
        provider="local embeddings",
        policy=policy,
    )


def default_data() -> Path:
    return Path(os.environ.get("ANGULARAI_EVAL_DATA", "~/.cache/angularai-eval")).expanduser()


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="angularai-eval", description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)

    one = sub.add_parser("fetch", help="Download LongMemEval-S and check its hash.")
    one.add_argument("--data", type=Path, default=None)

    cut = sub.add_parser("reduce", help="Write the reduced, stratified subset.")
    cut.add_argument("--data", type=Path, default=None)
    cut.add_argument("--count", type=int, default=50)
    cut.add_argument("--seed", type=int, default=0)
    cut.add_argument("--haystack-tokens", type=int, default=12000)
    cut.add_argument("--out", type=Path, required=True)

    go = sub.add_parser("run", help="Answer the subset with one arm.")
    go.add_argument("--subset", type=Path, required=True)
    go.add_argument("--arm", choices=ARMS, required=True)
    go.add_argument("--model", required=True)
    go.add_argument("--base-url", default=LOCAL)
    go.add_argument("--budget", type=int, default=2000)
    go.add_argument("--limit", type=int, default=None)
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--allow-remote", action="store_true")
    go.add_argument("--embed-model", default=None, help="v2: a local embedding model.")
    go.add_argument("--context", type=int, default=0, help="v2: neighbor chunks per hit.")

    grade = sub.add_parser("judge", help="Grade predictions with the official prompts.")
    grade.add_argument("--subset", type=Path, required=True)
    grade.add_argument("--predictions", type=Path, required=True)
    grade.add_argument("--model", required=True)
    grade.add_argument("--base-url", default=LOCAL)
    grade.add_argument("--out", type=Path, required=True)
    grade.add_argument("--allow-remote", action="store_true")

    show = sub.add_parser("report", help="One JSON result over judged files.")
    show.add_argument("--subset", type=Path, required=True)
    show.add_argument("--judged", type=Path, nargs="+", required=True)
    show.add_argument("--out", type=Path, required=True)
    show.add_argument("--markdown", type=Path, default=None)
    return top


def _say(line: str) -> None:
    print(line, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    scrub(os.environ)
    args = parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (DatasetError, EvalError) as exc:
        print(f"angularai-eval: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    command = cast(str, args.command)
    if command == "fetch":
        _say(str(fetch(args.data or default_data())))
        return 0
    if command == "reduce":
        questions = load(fetch(args.data or default_data()))
        chosen, manifest = reduce(questions, args.count, args.seed, args.haystack_tokens)
        write_subset(args.out, chosen, manifest)
        _say(f"{manifest.count} of {manifest.eligible} eligible ({manifest.total} total)")
        _say(json.dumps(manifest.per_type, sort_keys=True))
        return 0
    if command == "run":
        questions, _ = read_subset(args.subset)
        if args.limit is not None:
            questions = questions[: args.limit]
        base = require_local(args.base_url, args.allow_remote)
        chat = Chat(model=args.model, base_url=base, allow_remote=args.allow_remote)
        arm = cast(Arm, args.arm)
        substrate = None
        embedder = None
        if arm in ("v1", "v2"):
            policy = Policy(connect_timeout=10.0, read_timeout=1800.0, deadline=1800.0, attempts=1)
            substrate = substrate_for("compatible", args.model, base, "", policy, json_mode=True)
        if arm == "v2" and args.embed_model:
            embedder = local_embedder(args.embed_model, base)
        made = run(
            arm,
            questions,
            args.out,
            chat,
            budget=args.budget,
            substrate=substrate,
            embedder=embedder,
            context=args.context,
            progress=_say,
        )
        _say(f"{made} answered")
        return 0
    if command == "judge":
        questions, _ = read_subset(args.subset)
        chat = Chat(model=args.model, base_url=args.base_url, allow_remote=args.allow_remote)
        _say(f"{judge(questions, args.predictions, args.out, chat, progress=_say)} judged")
        return 0
    questions, manifest = read_subset(args.subset)
    result = report(args.judged, manifest, questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = markdown(result)
    if args.markdown is not None:
        args.markdown.write_text(table, encoding="utf-8")
    _say(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
