"""Scores. The judge prompts are LongMemEval's evaluate_qa.py, word for word.

judge: the official yes/no prompt, answered by a local model. The headline number.
contains: the normalized gold answer appears in the normalized prediction. A strict,
model-free check; it undercounts paraphrase and is not defined for preference rubrics.
"""

import re
import statistics
import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from angularai.evals.dataset import Question

_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
)
_TAIL = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_TEMPORAL = (
    "In addition, do not penalize off-by-one errors for the number of days. If the question "
    "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
)
_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer."
)
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
    "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)
JUDGE_SYSTEM = "You are a strict grader. Answer yes or no only."
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({char: " " for char in string.punctuation})


@dataclass(frozen=True)
class Scored:
    id: str
    type: str
    abstention: bool
    judge: bool | None
    contains: bool | None


def judge_prompt(question: Question, response: str) -> str:
    if question.abstention:
        return _ABSTENTION.format(question.question, question.answer, response)
    if question.type in ("single-session-user", "single-session-assistant", "multi-session"):
        return (_STANDARD + _TAIL).format(question.question, question.answer, response)
    if question.type == "temporal-reasoning":
        return (_STANDARD + _TEMPORAL + _TAIL).format(question.question, question.answer, response)
    if question.type == "knowledge-update":
        return (_UPDATE + _TAIL).format(question.question, question.answer, response)
    if question.type == "single-session-preference":
        return _PREFERENCE.format(question.question, question.answer, response)
    raise ValueError(f"No judge prompt for {question.type}.")


def verdict(text: str) -> bool:
    """yes counts. Anything else, including a refusal to grade, does not."""

    return "yes" in text.strip().lower()[:12]


def normalize(text: str) -> str:
    lowered = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def contains(question: Question, prediction: str) -> bool | None:
    if question.abstention or question.type == "single-session-preference":
        return None
    gold = normalize(question.answer)
    if not gold:
        return None
    return f" {gold} " in f" {normalize(prediction)} "


def summarize(rows: Iterable[Scored]) -> dict[str, object]:
    items = list(rows)
    by_type: dict[str, list[Scored]] = {}
    for row in items:
        by_type.setdefault(row.type, []).append(row)
    return {
        "n": len(items),
        "judge": _rate(row.judge for row in items),
        "contains": _rate(row.contains for row in items),
        "abstention": _rate(row.judge for row in items if row.abstention),
        "per_type": {
            kind: {
                "n": len(group),
                "judge": _rate(row.judge for row in group),
                "contains": _rate(row.contains for row in group),
            }
            for kind, group in sorted(by_type.items())
        },
    }


def usage(records: Iterable[Mapping[str, object]]) -> dict[str, float | None]:
    rows = list(records)
    return {
        "prompt_tokens_mean": _mean(row.get("prompt_tokens") for row in rows),
        "prompt_tokens_est_mean": _mean(row.get("prompt_tokens_est") for row in rows),
        "seconds_p50": _median(row.get("seconds") for row in rows),
        "errors": float(sum(1 for row in rows if row.get("error"))),
    }


def _rate(values: Iterable[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return round(100 * sum(known) / len(known), 1)


def _numbers(values: Iterable[object]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]


def _mean(values: Iterable[object]) -> float | None:
    numbers = _numbers(values)
    return round(statistics.fmean(numbers), 1) if numbers else None


def _median(values: Iterable[object]) -> float | None:
    numbers = _numbers(values)
    return round(statistics.median(numbers), 1) if numbers else None
