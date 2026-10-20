"""Prompts. The reading prompt follows LongMemEval's run_generation.py (history as text)."""

from angularai.budget import estimate_tokens
from angularai.evals.dataset import Question, Session, Turn

SYSTEM = "You are a helpful assistant. Answer the question from the chat history. Be brief."
_HEAD = (
    "I will give you several history chats between you and a user. "
    "Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n"
)


def ask(question: Question) -> str:
    return f"Current Date: {question.date}\nQuestion: {question.question}\nAnswer:"


def session_header(index: int, session: Session) -> str:
    return f"### Session {index}:\nSession Date: {session.date}\nSession Content:\n"


def turn_line(turn: Turn) -> str:
    return f"{turn.role}: {turn.content.strip()}\n"


def session_text(index: int, session: Session) -> str:
    return session_header(index, session) + "".join(turn_line(t) for t in session.turns) + "\n"


def history_prompt(question: Question, sessions: tuple[Session, ...]) -> str:
    body = "".join(session_text(index + 1, s) for index, s in enumerate(sessions))
    return _HEAD + body + "\n" + ask(question)


def full_prompt(question: Question) -> str:
    """Every session the question carries, oldest first."""

    return history_prompt(question, question.sessions)


def oracle_prompt(question: Question) -> str:
    """Only the sessions that hold the answer. An upper bound for reading, not for recall."""

    keep = set(question.evidence)
    return history_prompt(question, tuple(s for s in question.sessions if s.id in keep))


def truncated_prompt(question: Question, budget: int, system: str = SYSTEM) -> str:
    """The newest turns that fit budget, with the system prompt counted. Oldest are dropped.

    Each piece is estimated on its own. A sum of rounded-up estimates is never below
    the estimate of the whole, so the prompt never passes budget.
    """

    fixed = estimate_tokens(system) + estimate_tokens(_HEAD) + estimate_tokens("\n" + ask(question))
    room = budget - fixed
    kept: list[tuple[int, int]] = []
    headers: set[int] = set()
    for index in range(len(question.sessions) - 1, -1, -1):
        session = question.sessions[index]
        header = estimate_tokens(session_header(index + 1, session)) + 1
        stop = False
        for position in range(len(session.turns) - 1, -1, -1):
            cost = estimate_tokens(turn_line(session.turns[position]))
            if index not in headers:
                cost += header
            if cost > room:
                stop = True
                break
            room -= cost
            headers.add(index)
            kept.append((index, position))
        if stop:
            break
    parts: list[str] = []
    for index in sorted(headers):
        session = question.sessions[index]
        lines = "".join(
            turn_line(session.turns[position])
            for position in sorted(p for i, p in kept if i == index)
        )
        parts.append(session_header(index + 1, session) + lines + "\n")
    return _HEAD + "".join(parts) + "\n" + ask(question)
