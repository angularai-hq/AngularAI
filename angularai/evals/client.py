"""A plain chat call to a local OpenAI-compatible server (Ollama, llama.cpp server).

Free models only. A base URL off this machine is refused unless allow_remote is set,
and no API key is ever read or sent.
"""

import http.client
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# Environment names that hold paid keys. The harness removes them before it runs.
PAID_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANGULARAI_API_KEY",
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "ANGULARAI_EMBEDDINGS",
)

# (url, body, timeout seconds) -> (status, response body)
Post = Callable[[str, bytes, float], tuple[int, bytes]]


class EvalError(Exception):
    """A call or a file the harness cannot use."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float


def require_local(base_url: str, allow_remote: bool = False) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise EvalError(f"{base_url} is not an http URL.")
    if not allow_remote and parts.hostname not in LOCAL_HOSTS:
        raise EvalError(
            f"{parts.hostname} is not this machine. Evals run on local models only. "
            "A remote endpoint needs --allow-remote and the owner's approval."
        )
    return base_url.rstrip("/")


def _post(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    connection: http.client.HTTPConnection
    if parts.scheme == "https":
        connection = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("POST", parts.path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    except OSError as exc:
        raise EvalError(f"{host}:{port} did not answer: {type(exc).__name__}.") from exc
    finally:
        connection.close()


class Chat:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout: float = 1800.0,
        post: Post = _post,
        clock: Callable[[], float] = time.monotonic,
        allow_remote: bool = False,
    ) -> None:
        self.model = model
        self.base_url = require_local(base_url, allow_remote)
        self._timeout = timeout
        self._post = post
        self._clock = clock

    def complete(self, system: str, user: str, max_tokens: int = 256) -> Completion:
        payload = {
            "model": self.model,
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        start = self._clock()
        status, body = self._post(
            f"{self.base_url}/chat/completions", json.dumps(payload).encode(), self._timeout
        )
        seconds = self._clock() - start
        if status != 200:
            raise EvalError(f"The model server answered {status}.")
        return _completion(body, seconds)


def _completion(body: bytes, seconds: float) -> Completion:
    try:
        payload = json.loads(body)
        text = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise EvalError("The model server sent a reply that is not a chat completion.") from exc
    if not isinstance(text, str):
        raise EvalError("The model server sent a reply with no text.")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return Completion(
        text=text.strip(),
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
        seconds=seconds,
    )


def _count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
