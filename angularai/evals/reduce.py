"""The reduced subset: a stratified sample, and haystacks cut to fit a local window.

For each sampled question: every evidence session is kept, then distractor sessions
from the same haystack are added in a seeded random order while the rendered history
stays within haystack_tokens. Kept sessions stay in their original order. A question
whose evidence alone does not fit is not eligible.

The sample is stratified by question_type with largest-remainder allocation. Within a
type, eligible questions are shuffled by the seed and the first are taken.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from angularai.budget import estimate_tokens
from angularai.evals.dataset import (
    REVISION,
    SHA256,
    TYPES,
    DatasetError,
    Question,
    Session,
    Turn,
)
from angularai.evals.render import session_text

FORMAT = 1


@dataclass(frozen=True)
class Manifest:
    format: int
    revision: str
    sha256: str
    seed: int
    count: int
    haystack_tokens: int
    eligible: int
    total: int
    per_type: dict[str, int]
    questions: dict[str, list[str]]


def history_tokens(sessions: Sequence[Session]) -> int:
    return sum(estimate_tokens(session_text(i + 1, s)) for i, s in enumerate(sessions))


def cut(question: Question, haystack_tokens: int, seed: int) -> Question | None:
    """question with a haystack of at most haystack_tokens, or None if evidence won't fit."""

    evidence = set(question.evidence)
    costs = [estimate_tokens(session_text(i + 1, s)) for i, s in enumerate(question.sessions)]
    keep = {i for i, s in enumerate(question.sessions) if s.id in evidence}
    used = sum(costs[i] for i in keep)
    if not keep or used > haystack_tokens:
        return None
    rest = [i for i in range(len(question.sessions)) if i not in keep]
    random.Random(f"{seed}:{question.id}").shuffle(rest)
    for index in rest:
        # Session numbers in the prompt shift as sessions drop; a two-digit header
        # change is at most one token, which the margin below covers.
        if used + costs[index] + 1 <= haystack_tokens:
            keep.add(index)
            used += costs[index] + 1
    sessions = tuple(question.sessions[i] for i in sorted(keep))
    while history_tokens(sessions) > haystack_tokens:
        drop = next(i for i, s in enumerate(sessions) if s.id not in evidence)
        sessions = sessions[:drop] + sessions[drop + 1 :]
    return Question(
        id=question.id,
        type=question.type,
        question=question.question,
        answer=question.answer,
        date=question.date,
        sessions=sessions,
        evidence=question.evidence,
    )


def allocate(available: dict[str, int], count: int) -> dict[str, int]:
    """count split over types in proportion to what is available. Largest remainder."""

    total = sum(available.values())
    if count > total:
        raise DatasetError(f"Only {total} questions are eligible; {count} were asked for.")
    if total == 0:
        return {kind: 0 for kind in available}
    exact = {kind: count * n / total for kind, n in available.items()}
    shares = {kind: int(value) for kind, value in exact.items()}
    order = sorted(available, key=lambda kind: (shares[kind] - exact[kind], kind))
    for kind in order[: count - sum(shares.values())]:
        shares[kind] += 1
    return shares


def reduce(
    questions: Sequence[Question], count: int, seed: int, haystack_tokens: int
) -> tuple[tuple[Question, ...], Manifest]:
    eligible: dict[str, list[Question]] = {kind: [] for kind in TYPES}
    for question in questions:
        reduced = cut(question, haystack_tokens, seed)
        if reduced is not None:
            eligible.setdefault(question.type, []).append(reduced)
    shares = allocate({kind: len(items) for kind, items in eligible.items()}, count)
    chosen: list[Question] = []
    for kind in sorted(eligible):
        pool = sorted(eligible[kind], key=lambda q: q.id)
        random.Random(f"{seed}:{kind}").shuffle(pool)
        chosen.extend(pool[: shares[kind]])
    chosen.sort(key=lambda q: q.id)
    manifest = Manifest(
        format=FORMAT,
        revision=REVISION,
        sha256=SHA256,
        seed=seed,
        count=len(chosen),
        haystack_tokens=haystack_tokens,
        eligible=sum(len(items) for items in eligible.values()),
        total=len(questions),
        per_type={kind: shares[kind] for kind in sorted(shares) if shares[kind]},
        questions={q.id: [s.id for s in q.sessions] for q in chosen},
    )
    return tuple(chosen), manifest


def write_subset(path: Path, questions: Sequence[Question], manifest: Manifest) -> None:
    payload = {"manifest": asdict(manifest), "questions": [_question_json(q) for q in questions]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def read_subset(path: Path) -> tuple[tuple[Question, ...], Manifest]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload["manifest"]
        manifest = Manifest(**raw)
        questions = tuple(_question_from(item) for item in payload["questions"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetError(f"{path} is not a subset written by angularai-eval reduce.") from exc
    if manifest.format != FORMAT:
        raise DatasetError(f"{path} has subset format {manifest.format}; {FORMAT} is read.")
    return questions, manifest


def _question_json(question: Question) -> dict[str, object]:
    return {
        "id": question.id,
        "type": question.type,
        "question": question.question,
        "answer": question.answer,
        "date": question.date,
        "evidence": list(question.evidence),
        "sessions": [
            {
                "id": s.id,
                "date": s.date,
                "turns": [{"role": t.role, "content": t.content} for t in s.turns],
            }
            for s in question.sessions
        ],
    }


def _question_from(item: dict[str, object]) -> Question:
    sessions = item["sessions"]
    evidence = item["evidence"]
    if not isinstance(sessions, list) or not isinstance(evidence, list):
        raise TypeError("subset question")
    return Question(
        id=str(item["id"]),
        type=str(item["type"]),
        question=str(item["question"]),
        answer=str(item["answer"]),
        date=str(item["date"]),
        evidence=tuple(str(e) for e in evidence),
        sessions=tuple(
            Session(
                id=str(s["id"]),
                date=str(s["date"]),
                turns=tuple(
                    Turn(role="user" if t["role"] == "user" else "assistant", content=t["content"])
                    for t in s["turns"]
                ),
            )
            for s in sessions
        ),
    )
