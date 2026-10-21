"""Run an arm over a subset, judge it, and report. Every step is resumable.

Predictions and verdicts are JSON lines, one per question, flushed as they are made.
A rerun skips every question already in the file.
"""

import json
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from angularai.budget import estimate_tokens
from angularai.errors import AngularAIError
from angularai.evals.client import Chat, EvalError
from angularai.evals.cortex import answer_v1
from angularai.evals.dataset import Question
from angularai.evals.reduce import Manifest
from angularai.evals.render import SYSTEM, full_prompt, oracle_prompt, truncated_prompt
from angularai.evals.scoring import (
    JUDGE_SYSTEM,
    Scored,
    contains,
    judge_prompt,
    summarize,
    usage,
    verdict,
)
from angularai.index import Unit
from angularai.semantic import Embedder
from angularai.substrate import Substrate

Arm = Literal["full", "truncated", "oracle", "v1", "v2"]
ARMS: tuple[Arm, ...] = ("full", "truncated", "oracle", "v1", "v2")
Record = dict[str, object]
Progress = Callable[[str], None]
MAX_ANSWER_TOKENS = 256


def done(path: Path) -> dict[str, Record]:
    """Records already written, by question id. A torn last line is ignored."""

    if not path.is_file():
        return {}
    found: dict[str, Record] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            found[item["id"]] = item
    return found


def append(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def answer(
    arm: Arm,
    question: Question,
    chat: Chat,
    *,
    budget: int,
    substrate: Substrate | None = None,
    embedder: Embedder | None = None,
    context: int = 0,
    unit: Unit = "sentence",
    word_rank: bool = False,
) -> Record:
    record: Record = {
        "id": question.id,
        "type": question.type,
        "arm": arm,
        "model": chat.model,
        "budget": budget if arm in ("truncated", "v1", "v2") else None,
        "prediction": "",
        "error": None,
        "prompt_tokens": None,
        "prompt_tokens_est": None,
        "completion_tokens": None,
        "seconds": None,
    }
    start = time.monotonic()
    try:
        if arm in ("v1", "v2"):
            if substrate is None:
                raise EvalError(f"The {arm} arm needs a substrate.")
            with tempfile.TemporaryDirectory(prefix="angularai-eval-") as scratch:
                result = answer_v1(
                    Path(scratch) / "cortex",
                    question,
                    substrate,
                    budget=budget,
                    model=chat.model,
                    base_url=chat.base_url,
                    v2=arm == "v2",
                    embedder=embedder,
                    context=context,
                    unit=unit,
                    word_rank=word_rank,
                )
            record.update(
                prediction=result.reply,
                prompt_tokens_est=result.used,
                episodes=result.episodes,
                episodes_shown=result.episodes_shown,
            )
            if arm == "v2":
                record.update(
                    chunks=result.chunks,
                    embed_model=None if embedder is None else embedder.model,
                    context=context,
                    unit=unit,
                    word_rank=word_rank,
                )
        else:
            prompt = _prompt(arm, question, budget)
            completion = chat.complete(SYSTEM, prompt, MAX_ANSWER_TOKENS)
            record.update(
                prediction=completion.text,
                prompt_tokens=completion.prompt_tokens,
                prompt_tokens_est=estimate_tokens(SYSTEM) + estimate_tokens(prompt),
                completion_tokens=completion.completion_tokens,
            )
    except (AngularAIError, EvalError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record["seconds"] = round(time.monotonic() - start, 2)
    return record


def _prompt(arm: Arm, question: Question, budget: int) -> str:
    if arm == "full":
        return full_prompt(question)
    if arm == "oracle":
        return oracle_prompt(question)
    return truncated_prompt(question, budget)


def run(
    arm: Arm,
    questions: Sequence[Question],
    out: Path,
    chat: Chat,
    *,
    budget: int,
    substrate: Substrate | None = None,
    embedder: Embedder | None = None,
    context: int = 0,
    unit: Unit = "sentence",
    word_rank: bool = False,
    progress: Progress | None = None,
) -> int:
    """Answer every question not yet in out. Returns how many were answered now."""

    seen = done(out)
    count = 0
    for question in questions:
        if question.id in seen:
            continue
        record = answer(
            arm,
            question,
            chat,
            budget=budget,
            substrate=substrate,
            embedder=embedder,
            context=context,
            unit=unit,
            word_rank=word_rank,
        )
        append(out, record)
        count += 1
        if progress is not None:
            progress(
                f"{arm} {question.id} {record['seconds']}s"
                + (f" error {record['error']}" if record["error"] else "")
            )
    return count


def judge(
    questions: Sequence[Question],
    predictions: Path,
    out: Path,
    chat: Chat,
    progress: Progress | None = None,
) -> int:
    answers = done(predictions)
    seen = done(out)
    count = 0
    for question in questions:
        record = answers.get(question.id)
        if record is None or question.id in seen:
            continue
        prediction = str(record.get("prediction") or "")
        if record.get("error") or not prediction:
            said = "no"
        else:
            said = chat.complete(JUDGE_SYSTEM, judge_prompt(question, prediction), 8).text
        append(
            out,
            {
                **record,
                "judge_model": chat.model,
                "judge_said": said[:40],
                "judge": verdict(said),
                "contains": contains(question, prediction),
                "abstention": question.abstention,
            },
        )
        count += 1
        if progress is not None:
            progress(f"judge {question.id} {verdict(said)}")
    return count


def report(
    judged: Iterable[Path], manifest: Manifest, questions: Sequence[Question]
) -> dict[str, object]:
    ids = {question.id for question in questions}
    arms: list[dict[str, object]] = []
    for path in judged:
        rows = [row for row in done(path).values() if row["id"] in ids]
        if not rows:
            continue
        first = rows[0]
        scored = [
            Scored(
                id=str(row["id"]),
                type=str(row["type"]),
                abstention=bool(row.get("abstention")),
                judge=_flag(row.get("judge")),
                contains=_flag(row.get("contains")),
            )
            for row in rows
        ]
        arms.append(
            {
                "file": path.name,
                "arm": first.get("arm"),
                "model": first.get("model"),
                "budget": first.get("budget"),
                "judge_model": first.get("judge_model"),
                "complete": len(rows) == len(ids),
                **summarize(scored),
                **usage(rows),
            }
        )
    manifest_json = asdict(manifest)
    manifest_json.pop("questions")
    return {"benchmark": "LongMemEval-S (reduced)", "subset": manifest_json, "arms": arms}


def markdown(result: Mapping[str, object]) -> str:
    rows = result.get("arms")
    lines = [
        "| arm | model | budget | n | judge % | contains % | est. tokens | server tokens | p50 s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in rows if isinstance(rows, list) else []:
        lines.append(
            f"| {arm.get('arm')} | {arm.get('model')} | {arm.get('budget') or '-'} "
            f"| {arm.get('n')} | {_cell(arm.get('judge'))} | {_cell(arm.get('contains'))} "
            f"| {_cell(arm.get('prompt_tokens_est_mean'))} "
            f"| {_cell(arm.get('prompt_tokens_mean'))} | {_cell(arm.get('seconds_p50'))} |"
        )
    return "\n".join(lines) + "\n"


def _flag(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _cell(value: object) -> str:
    return "-" if value is None else str(value)
