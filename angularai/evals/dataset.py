"""LongMemEval-S: the pinned file, its hash, and typed questions.

Source: huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (MIT).
The revision and the SHA-256 are pinned. A file that does not match is refused.
"""

import hashlib
import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
FILENAME = "longmemeval_s_cleaned.json"
SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
URL = (
    f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{REVISION}/{FILENAME}"
)
TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)

Role = Literal["user", "assistant"]
# (url, destination) -> None. Writes the body of url to destination.
Download = Callable[[str, Path], None]


class DatasetError(Exception):
    """The file is missing, does not match its hash, or is not LongMemEval."""


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str


@dataclass(frozen=True)
class Session:
    id: str
    date: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    answer: str
    date: str
    sessions: tuple[Session, ...]
    evidence: tuple[str, ...]

    @property
    def abstention(self) -> bool:
        """An _abs question has no answer in the history. The right reply says so."""

        return self.id.endswith("_abs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)


def fetch(directory: Path, download: Download = _download) -> Path:
    """The pinned file in directory. Downloaded once; checked every time."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    if path.is_file() and sha256(path) == SHA256:
        return path
    partial = directory / (FILENAME + ".partial")
    download(URL, partial)
    if sha256(partial) != SHA256:
        partial.unlink()
        raise DatasetError(f"{FILENAME} does not match its pinned SHA-256. It was not kept.")
    partial.replace(path)
    return path


def load(path: Path, *, verify: bool = True) -> tuple[Question, ...]:
    if not path.is_file():
        raise DatasetError(f"No dataset at {path}. Run angularai-eval fetch.")
    if verify and sha256(path) != SHA256:
        raise DatasetError(f"{path} does not match the pinned SHA-256.")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{path} is not JSON.") from exc
    return parse(payload)


def parse(payload: object) -> tuple[Question, ...]:
    if not isinstance(payload, list):
        raise DatasetError("The dataset is not a list of questions.")
    return tuple(_question(item, index) for index, item in enumerate(payload))


def _question(item: object, index: int) -> Question:
    where = f"question {index}"
    if not isinstance(item, dict):
        raise DatasetError(f"{where} is not an object.")
    ids = _strings(item.get("haystack_session_ids"), f"{where} haystack_session_ids")
    dates = _strings(item.get("haystack_dates"), f"{where} haystack_dates")
    raw = item.get("haystack_sessions")
    if not isinstance(raw, list) or not (len(ids) == len(dates) == len(raw)):
        raise DatasetError(f"{where} has sessions, ids and dates of different lengths.")
    sessions = tuple(
        Session(id=sid, date=date, turns=_turns(turns, f"{where} session {sid}"))
        for sid, date, turns in zip(ids, dates, raw, strict=True)
    )
    answer = item.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, str | int | float):
        raise DatasetError(f"{where} has no answer.")
    kind = _text(item.get("question_type"), f"{where} question_type")
    if kind not in TYPES:
        raise DatasetError(f"{where} has an unknown question_type.")
    return Question(
        id=_text(item.get("question_id"), f"{where} question_id"),
        type=kind,
        question=_text(item.get("question"), f"{where} question"),
        answer=str(answer),
        date=_text(item.get("question_date"), f"{where} question_date"),
        sessions=sessions,
        evidence=_strings(item.get("answer_session_ids"), f"{where} answer_session_ids"),
    )


def _turns(value: object, where: str) -> tuple[Turn, ...]:
    if not isinstance(value, list):
        raise DatasetError(f"{where} is not a list of turns.")
    turns: list[Turn] = []
    for turn in value:
        if not isinstance(turn, dict):
            raise DatasetError(f"{where} has a turn that is not an object.")
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise DatasetError(f"{where} has a turn without a role or content.")
        turns.append(Turn(role="user" if role == "user" else "assistant", content=content))
    return tuple(turns)


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{where} is missing.")
    return value


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DatasetError(f"{where} is not a list of strings.")
    return tuple(str(item) for item in value)
