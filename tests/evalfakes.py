import json
from collections.abc import Sequence

from angularai.evals.dataset import Question, Session, Turn


def turn(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def raw_question(
    qid: str = "q1",
    kind: str = "single-session-user",
    sessions: int = 4,
    evidence: Sequence[int] = (1,),
    words: int = 40,
    answer: str | int = "blue",
) -> dict[str, object]:
    ids = [f"s{qid}-{i}" for i in range(sessions)]
    return {
        "question_id": qid,
        "question_type": kind,
        "question": "What color is my notebook?",
        "answer": answer,
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_session_ids": ids,
        "haystack_dates": [f"2023/05/{10 + i:02d} (Wed) 10:0{i % 10}" for i in range(sessions)],
        "haystack_sessions": [
            [
                turn("user", f"Session {i}. " + ("filler " * words)),
                turn("assistant", "Sure. " + ("reply " * words)),
            ]
            if i not in evidence
            else [
                turn("user", "My notebook is blue. " + ("filler " * words)),
                turn("assistant", "Noted, a blue notebook."),
            ]
            for i in range(sessions)
        ],
        "answer_session_ids": [ids[i] for i in evidence],
    }


def question(
    qid: str = "q1",
    kind: str = "single-session-user",
    answer: str = "blue",
    sessions: int = 3,
) -> Question:
    return Question(
        id=qid,
        type=kind,
        question="What color is my notebook?",
        answer=answer,
        date="2023/05/30 (Tue) 23:40",
        sessions=tuple(
            Session(
                id=f"s{i}",
                date=f"2023/05/1{i} (Wed) 10:00",
                turns=(
                    Turn("user", f"Turn {i}: my notebook is blue." if i == 0 else f"Turn {i}."),
                    Turn("assistant", f"Answer {i}."),
                ),
            )
            for i in range(sessions)
        ),
        evidence=("s0",),
    )


class FakePost:
    """A local chat server that answers from a list, and records every body."""

    def __init__(self, replies: Sequence[str] = ("blue",), status: int = 200) -> None:
        self.bodies: list[dict[str, object]] = []
        self.urls: list[str] = []
        self._replies = list(replies)
        self._status = status

    def __call__(self, url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
        self.urls.append(url)
        self.bodies.append(json.loads(body))
        text = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        payload = {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2},
        }
        return self._status, json.dumps(payload).encode()
