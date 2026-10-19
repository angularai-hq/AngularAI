"""angularai-mcp: serve one AngularAI cortex over MCP. stdio by default.

Nothing but the protocol is written to stdout. Notices go to stderr.
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import uvicorn

from angularai.errors import AngularAIError
from angularai.names import cortex_home, resolve_root
from angularai.store import init_cortex
from angularai_mcp import __version__
from angularai_mcp.server import AngularAIServer, build_server
from angularai_mcp.web import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PATH,
    TOKEN_ENV,
    BearerGate,
    TokenProblem,
    is_loopback,
    token_from,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    environ: Mapping[str, str] = dict(os.environ)
    try:
        root = resolve_root(args.cortex, args.name, environ)
        if args.init and not (root / "cortex.json").is_file():
            if args.name is not None:
                cortex_home(environ).mkdir(parents=True, exist_ok=True, mode=0o700)
            init_cortex(root)
            _say(f"Cortex is at {root}")
        token = token_from(environ) if args.http else None
        if args.http:
            _check_bind(args.host, args.allow_remote, token)
        server = build_server(root, environ)
    except (AngularAIError, TokenProblem) as exc:
        _say(str(exc))
        return 2
    try:
        if args.http:
            serve_http(server, args.host, args.port, token)
        else:
            server.run("stdio")
    except KeyboardInterrupt:
        return 130
    return 0


def serve_http(server: AngularAIServer, host: str, port: int, token: str | None) -> None:
    app = server.streamable_http_app(host=host, streamable_http_path=PATH)
    served = app if token is None else BearerGate(app, token)
    shown = f"[{host}]" if ":" in host else host
    guard = "bearer token required" if token else "no auth"
    _say(f"AngularAI MCP at http://{shown}:{port}{PATH} ({guard})")
    config = uvicorn.Config(
        served, host=host, port=port, log_level="warning", access_log=False, lifespan="on"
    )
    uvicorn.Server(config).run()


def _check_bind(host: str, allow_remote: bool, token: str | None) -> None:
    if is_loopback(host):
        return
    if not allow_remote:
        raise AngularAIError(
            f"{host} is not a loopback address. Pass --allow-remote to listen there."
        )
    if token is None:
        raise AngularAIError(
            f"Listening beyond loopback needs {TOKEN_ENV}. HTTP has no other auth."
        )


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="angularai-mcp",
        description="Serve an AngularAI cortex as MCP tools: remember, recall, forget, status.",
        epilog=(
            "stdio unless --http. The cortex is the one angularai would use: --cortex PATH, "
            "else --name NAME, else ANGULARAI_CORTEX, else ~/.angularai/default."
        ),
    )
    parser.add_argument("--version", action="version", version=f"angularai-mcp {__version__}")
    parser.add_argument("--cortex", type=Path, default=None, metavar="PATH")
    parser.add_argument("--name", default=None, metavar="NAME", help="A named cortex.")
    parser.add_argument("--init", action="store_true", help="Create the cortex if it is missing.")
    parser.add_argument("--http", action="store_true", help="Serve streamable HTTP, not stdio.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"HTTP host. Default {DEFAULT_HOST}.")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"HTTP port. Default {DEFAULT_PORT}."
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=f"Allow a host beyond loopback. {TOKEN_ENV} is then required.",
    )
    return parser
