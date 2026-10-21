"""What is left before a run may finish: markers across the root, and a final summary.

Both checks are generic. The work left is every line that holds one of the markers
(TODO and FIXME by default) in any readable file under the root, counted in full, or
None when the root is too large to count. The summary check takes a file the task
must end with and the files it must cover (a glob): the summary must exist, be
written after every one of them, name each, and say something specific about each,
something its own text says that most of the others' do not.
"""

import ast
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from angularai import files, tools

COUNT_FILES = 5000
_FILE_BYTES = 1024 * 1024
_TOTAL_BYTES = 64 * 1024 * 1024
_UNITS = 2000
_SHOWN = 10
_UNIT_LINES = 3
_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "double": "2",
    "doubles": "2",
    "twice": "2",
    "triple": "3",
    "triples": "3",
    "thrice": "3",
    "quadruple": "4",
}


@dataclass(frozen=True)
class WorkLeft:
    """Marker lines under the root. count is None when it could not be counted."""

    count: int | None
    lines: tuple[str, ...] = ()
    files: tuple[str, ...] = ()


def work_left(root: Path, markers: Sequence[str], shown: int = 12) -> WorkLeft:
    """Every line under root holding a marker. None when there are no markers, too many
    files, too many bytes, or a file too large to read whole."""

    if not markers:
        return WorkLeft(None)
    found = tools.every_file(root, COUNT_FILES)
    if found is None:
        return WorkLeft(None)
    count = 0
    lines: list[str] = []
    holding: list[str] = []
    total = 0
    for file in found:
        try:
            text, cut = files.read_text(root, file.resolve(), _FILE_BYTES)
        except files.Refused:
            continue  # binary, or not a regular file: no text to finish
        except (OSError, RuntimeError, ValueError):
            return WorkLeft(None)
        if cut:
            return WorkLeft(None)
        total += len(text)
        if total > _TOTAL_BYTES:
            return WorkLeft(None)
        relative = file.relative_to(root).as_posix()
        here = 0
        for number, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in markers):
                here += 1
                if len(lines) < shown:
                    lines.append(f"{relative}:{number}:{line.strip()[:160]}")
        if here:
            count += here
            holding.append(relative)
    return WorkLeft(count, tuple(lines), tuple(holding))


def _tokens(text: str) -> set[str]:
    words = (_NUMBERS.get(word, word) for word in _WORD.findall(text.lower()))
    return {word for word in words if word.isdigit() or len(word) >= 3}


def unit_text(source: str, suffix: str) -> str:
    """What a file says it does: a Python docstring, else its first lines."""

    if suffix == ".py":
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            doc = ast.get_docstring(tree)
            for node in tree.body:
                if doc:
                    break
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
            return doc or ""
    kept = [line.strip() for line in source.splitlines() if line.strip()]
    return " ".join(kept[:_UNIT_LINES])


def summary_problems(root: Path, summary: str, units: str) -> list[str]:
    """Why the summary is not yet the run's last word. Empty when it is."""

    path = tools.inside(root, summary)
    if path is None:
        return [f"{summary} is outside the root."]
    if not path.is_file():
        return [f"{summary} is not written. Write it last."]
    try:
        text, cut = files.read_text(root, path, _FILE_BYTES)
    except (files.Refused, OSError, RuntimeError, ValueError):
        return [f"{summary} could not be read as text."]
    if cut:
        return [f"{summary} is too large."]
    if not units:
        return []
    covered = sorted(
        item for item in root.glob(units) if item.is_file() and item.resolve() != path.resolve()
    )
    if not covered or len(covered) > _UNITS:
        return [] if not covered else [f"{units} matches more than {_UNITS} files."]
    problems: list[str] = []
    written = path.stat().st_mtime
    newer = [item for item in covered if item.stat().st_mtime > written]
    if newer:
        names = ", ".join(item.relative_to(root).as_posix() for item in newer[:_SHOWN])
        problems.append(
            f"{summary} was written before {len(newer)} of the files it covers changed "
            f"({names}). Write or update it last."
        )
    lines = text.splitlines()
    sources: dict[Path, set[str]] = {}
    for item in covered:
        try:
            body, _ = files.read_text(root, item.resolve(), _FILE_BYTES)
        except (files.Refused, OSError, RuntimeError, ValueError):
            body = ""
        sources[item] = _tokens(unit_text(body, item.suffix)) - _tokens(item.stem)
    frequency: Counter[str] = Counter(token for words in sources.values() for token in words)
    common = {token for token, count in frequency.items() if count * 2 > len(covered)}
    missing: list[str] = []
    vague: list[str] = []
    said: dict[str, list[str]] = {}
    for item in covered:
        name = re.compile(rf"(?<![\w-]){re.escape(item.stem)}(?![\w-])")
        line = next((line for line in lines if name.search(line)), None)
        if line is None:
            missing.append(item.stem)
            continue
        description = name.sub(" ", line.replace(item.name, " "))
        words = _tokens(description) - _tokens(item.stem) - {"src", "py"}
        said.setdefault(" ".join(sorted(words)), []).append(item.stem)
        distinctive = sources[item] - common
        if not words or (distinctive and not words & distinctive):
            vague.append(item.stem)
    for stems in said.values():
        if len(stems) > 1 and len({frozenset(sources[_by(covered, s)]) for s in stems}) > 1:
            vague.extend(stem for stem in stems if stem not in vague)
    if missing:
        problems.append(
            f"{summary} does not list {len(missing)} of {len(covered)}: "
            + ", ".join(missing[:_SHOWN])
            + (" ..." if len(missing) > _SHOWN else "")
        )
    if vague:
        problems.append(
            f"{summary} is not specific about {len(vague)}: "
            + ", ".join(sorted(vague)[:_SHOWN])
            + (" ..." if len(vague) > _SHOWN else "")
            + ". Say what each one does, as its own text says."
        )
    return problems


def _by(covered: list[Path], stem: str) -> Path:
    return next(item for item in covered if item.stem == stem)
