import shlex
import subprocess
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
_SKIP = {".git", ".venv", "node_modules", "__pycache__", "dist", ".mypy_cache", ".ruff_cache"}
_REFUSED = frozenset(
    {
        "sudo",
        "su",
        "ssh",
        "scp",
        "curl",
        "wget",
        "nc",
        "ncat",
        "bash",
        "sh",
        "zsh",
        "fish",
        "dash",
        "env",
    }
)


def execute(root: Path, action: Action, secrets: tuple[str, ...] = ()) -> str:
    """Run one tool inside root. A refusal is a sentence, not an exception."""

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
        result = _run(root, action.command)
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
    names = sorted(path.iterdir(), key=lambda item: item.name)
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
    files = [path] if path.is_file() else _files(path)
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
    program = Path(argv[0]).name
    if program in _REFUSED:
        return f"Refused {program}. Run the program directly, with no shell."
    if any(_escapes(part, root.resolve()) for part in argv):
        return "That command leaves the directory."
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "The command timed out."
    except FileNotFoundError:
        return "That command was not found."
    except OSError:
        return "That command could not be run."
    body = (completed.stdout + completed.stderr).strip()
    return f"exit {completed.returncode}\n{_clip(body, _READ_CHARS)}"


def _files(path: Path) -> list[Path]:
    found: list[Path] = []
    for item in sorted(path.rglob("*")):
        if any(part in _SKIP for part in item.parts):
            continue
        if item.is_file():
            found.append(item)
        if len(found) >= 400:
            break
    return found


def _inside(root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    base = root.resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        target = candidate.resolve()
    else:
        target = (base / candidate).resolve()
    if not target.is_relative_to(base):
        return None
    return target


def _escapes(part: str, root: Path) -> bool:
    if ".." in Path(part).parts:
        return True
    expanded = Path(part).expanduser()
    if not expanded.is_absolute():
        return False
    return not expanded.resolve().is_relative_to(root)


def _leaves() -> str:
    return "That path leaves the directory."


def _clip(text: str, limit: int = _RESULT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
