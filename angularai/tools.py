import os
import shlex
import shutil
import signal
import subprocess
import threading
from pathlib import Path

from angularai.models import Action
from angularai.secrets import redact

_READ_LINES = 160
_READ_CHARS = 1600
_LIST_MAX = 80
_SEARCH_MAX = 30
_WRITE_MAX = 80_000
_RESULT = 1600
_RUN_TIMEOUT = 20
_RUN_OUTPUT = 64 * 1024
_RUN_PATH = "/usr/bin:/bin"
_SKIP = {".git", ".venv", "node_modules", "__pycache__", "dist", ".mypy_cache", ".ruff_cache"}
# Programs run may start. None of them runs another program, opens a socket, or reads a
# list of file names from a file. Shells, interpreters, find, xargs, env, git, and make
# are absent on purpose.
ALLOWED = frozenset(
    {
        "basename",
        "cat",
        "cmp",
        "cp",
        "cut",
        "dirname",
        "echo",
        "false",
        "grep",
        "head",
        "ls",
        "mkdir",
        "mv",
        "pwd",
        "rm",
        "rmdir",
        "sort",
        "stat",
        "tail",
        "touch",
        "true",
        "uniq",
        "wc",
    }
)
# Long options that run a program, follow a link out, or read file names from a file.
# GNU accepts any unambiguous prefix, so a prefix is refused too.
_REFUSED_LONG = {
    "cp": ("--dereference",),
    "grep": ("--dereference-recursive",),
    "sort": ("--compress-program", "--files0-from"),
    "wc": ("--files0-from",),
}
# Short flags that follow a link out: cp -L and -H, grep -R.
_REFUSED_SHORT = {"cp": frozenset("LH"), "grep": frozenset("R")}


def execute(
    root: Path,
    action: Action,
    secrets: tuple[str, ...] = (),
    allow_run: bool = False,
) -> str:
    """Run one tool inside root. A refusal is a sentence, not an exception."""

    root = root.resolve()
    tool = action.tool
    if tool == "list":
        result = _list(root, action.path or ".")
    elif tool == "read":
        result = _read(root, action.path, action.start)
    elif tool == "search":
        result = _search(root, action.path or ".", action.query)
    elif tool == "edit":
        result = _edit(root, action.path, action.old, action.new)
    elif tool == "write":
        result = _write(root, action.path, action.text, action.overwrite)
    elif tool == "run":
        result = _run(root, action.command) if allow_run else _RUN_OFF
    else:
        result = f"Unknown tool {tool}."
    return _clip(redact(result, secrets))


def _list(root: Path, raw: str) -> str:
    path = _inside(root, raw)
    if path is None:
        return _leaves()
    if not path.exists():
        return "That path is not there."
    if not path.is_dir():
        return "That path is a file."
    names = sorted(
        (item for item in path.iterdir() if _stays(root, item)), key=lambda item: item.name
    )
    lines = [f"{item.name}/" if item.is_dir() else item.name for item in names[:_LIST_MAX]]
    if len(names) > _LIST_MAX:
        lines.append(f"... {len(names) - _LIST_MAX} more")
    return "\n".join(lines) if lines else "(empty)"


def _read(root: Path, raw: str, start: int) -> str:
    path = _inside(root, raw)
    if path is None:
        return _leaves()
    if not path.is_file():
        return "That path is not a file."
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "That file could not be read as text."
    if "\x00" in raw_text[:8000]:
        return "That file is not text."
    lines = raw_text.splitlines()
    begin = max(start, 1) - 1
    chosen = lines[begin : begin + _READ_LINES]
    body = "\n".join(f"{begin + index + 1}|{line}" for index, line in enumerate(chosen))
    return _clip(body, _READ_CHARS)


def _search(root: Path, raw: str, query: str) -> str:
    if not query.strip():
        return "Give a query."
    path = _inside(root, raw)
    if path is None:
        return _leaves()
    if not path.exists():
        return "That path is not there."
    hits: list[str] = []
    files = [path] if path.is_file() else _files(root, path)
    for file in files:
        if len(hits) >= _SEARCH_MAX:
            break
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if query not in line:
                continue
            relative = file.relative_to(root)
            hits.append(f"{relative}:{number}:{line.strip()[:160]}")
            if len(hits) >= _SEARCH_MAX:
                break
    return "\n".join(hits) if hits else "No matches."


def _edit(root: Path, raw: str, old: str, new: str) -> str:
    if not old:
        return "Give the text to replace."
    path = _inside(root, raw)
    if path is None:
        return _leaves()
    if not path.is_file():
        return "That path is not a file."
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "That file could not be read as text."
    count = text.count(old)
    if count == 0:
        return "That text is not in the file."
    if count > 1:
        return f"That text appears {count} times. Make it unique."
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"Edited {path.relative_to(root)}."


def _write(root: Path, raw: str, text: str, overwrite: bool) -> str:
    if len(text) > _WRITE_MAX:
        return "That text is too long to write."
    path = _inside(root, raw)
    if path is None:
        return _leaves()
    if path.is_dir():
        return "That path is a directory."
    if path.exists() and not overwrite:
        return "That file exists. Edit it, or set overwrite."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"Wrote {path.relative_to(root)}."


def _run(root: Path, command: str) -> str:
    if not command.strip():
        return "Give a command."
    try:
        argv = shlex.split(command)
    except ValueError:
        return "That command could not be read."
    if not argv or len(argv) > 32:
        return "That command could not be read."
    refusal = _refuse(root, argv)
    if refusal is not None:
        return refusal
    program = shutil.which(argv[0], path=_RUN_PATH)
    if program is None:
        return "That command was not found."
    if Path(program).resolve().is_relative_to(root):
        return "A program inside the directory is not run."
    try:
        process = subprocess.Popen(
            [program, *argv[1:]],
            cwd=root,
            env=_run_env(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return "That command could not be run."
    output = _Capture(process)
    try:
        code = process.wait(timeout=_RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        process.wait()
        output.join()
        return "The command timed out."
    _kill_group(process)
    output.join()
    head = f"exit {code}"
    if output.cut:
        head += f" (output cut at {_RUN_OUTPUT} bytes)"
    body = output.text().strip()
    return f"{head}\n{_clip(body, _READ_CHARS)}"


def _refuse(root: Path, argv: list[str]) -> str | None:
    name = argv[0]
    if "/" in name or name not in ALLOWED:
        allowed = ", ".join(sorted(ALLOWED))
        return f"Refused {name}. run starts one of: {allowed}."
    short = _REFUSED_SHORT.get(name, frozenset())
    refused = _REFUSED_LONG.get(name, ())
    for part in argv[1:]:
        if part.startswith("--") and len(part) > 2:
            option = part.split("=", 1)[0]
            if len(option) > 3 and any(full.startswith(option) for full in refused):
                return f"Refused {option}."
            value = part.split("=", 1)[1] if "=" in part else ""
            if value and not _arg_inside(root, value):
                return "That command leaves the directory."
        elif part.startswith("-") and len(part) > 1:
            if short & set(part[1:]):
                return f"Refused {part}."
            if "/" in part or ".." in part:
                return "That command leaves the directory."
        elif not _arg_inside(root, part):
            return "That command leaves the directory."
    return None


def _arg_inside(root: Path, part: str) -> bool:
    if ".." in Path(part).parts:
        return False
    candidate = Path(part)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        return target.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def _run_env(root: Path) -> dict[str, str]:
    return {
        "PATH": _RUN_PATH,
        "HOME": str(root),
        "LANG": os.environ.get("LANG", "C"),
    }


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class _Capture:
    """Read at most _RUN_OUTPUT bytes. Past that, the process group is killed."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._chunks: list[bytes] = []
        self._size = 0
        self.cut = False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        while True:
            chunk = os.read(stream.fileno(), 8192)
            if not chunk:
                break
            room = _RUN_OUTPUT - self._size
            if len(chunk) > room:
                self._chunks.append(chunk[:room])
                self._size = _RUN_OUTPUT
                self.cut = True
                _kill_group(self._process)
                break
            self._chunks.append(chunk)
            self._size += len(chunk)
        stream.close()

    def join(self) -> None:
        self._thread.join(timeout=5)

    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")


def _files(root: Path, path: Path) -> list[Path]:
    found: list[Path] = []
    for folder, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = sorted(item for item in dirs if item not in _SKIP)
        for name in sorted(names):
            item = Path(folder) / name
            if not _stays(root, item):
                continue
            if item.is_file():
                found.append(item)
            if len(found) >= 400:
                return found
    return found


def _stays(root: Path, item: Path) -> bool:
    """True when item, with every link followed, is still inside root."""

    try:
        return item.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def _inside(root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        target = candidate.resolve()
    else:
        target = (root / candidate).resolve()
    if not target.is_relative_to(root):
        return None
    return target


_RUN_OFF = "run is off. Start the agent with --allow-run to use it."


def _leaves() -> str:
    return "That path leaves the directory."


def _clip(text: str, limit: int = _RESULT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
