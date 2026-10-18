"""Timeouts, retries, the pass deadline, the error taxonomy, and body parsing.

Every retry runs on a fake clock. conftest.py makes a real wait fail the test.
"""

import http.client
import json
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

import pytest

from angularai.errors import (
    AuthFailed,
    BadResponse,
    ConfigError,
    Forbidden,
    KeyRejected,
    NotFound,
    Overloaded,
    QuotaExceeded,
    RateLimited,
    RequestRejected,
    ServerError,
    SubstrateError,
    SubstrateUnavailable,
    SubstrateUnreadable,
    Timeout,
    Unreachable,
)
from angularai.http import Route, connection, http_json
from angularai.models import Config, Workspace
from angularai.pilot import pass_turn
from angularai.provider import (
    Clock,
    Policy,
    policy_from_env,
    retry_after,
    wait_before_retry,
)
from angularai.store import init_cortex, open_store
from angularai.substrate import (
    Answer,
    Anthropic,
    OpenAICompatible,
    Transport,
    build_substrate,
    error_for_status,
)

KEY = "sk-test-provider-secret-1234"
NOW = 1_800_000_000.0


class FakeClock:
    def __init__(self, random_value: float = 0.5) -> None:
        self.now = 100.0
        self.slept: list[float] = []
        self.random_value = random_value

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def clock(self) -> Clock:
        return Clock(
            monotonic=lambda: self.now,
            wall=lambda: NOW + self.now,
            sleep=self.sleep,
            random=lambda: self.random_value,
        )


def good(text: str = "Hi.", remember: list[object] | None = None) -> bytes:
    body = {"reply": text, "remember": remember or [], "supersede": []}
    return json.dumps({"choices": [{"message": {"content": json.dumps(body)}}]}).encode()


def openai_error(code: str | None, kind: str, message: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": kind, "code": code}}).encode()


Step = Answer | SubstrateError | Callable[[FakeClock], Answer]


class Script:
    """A transport that answers from a list and records every call."""

    def __init__(self, fake: FakeClock, *steps: Step) -> None:
        self.fake = fake
        self.steps: Iterator[Step] = iter(steps)
        self.timeouts: list[float] = []
        self.payloads: list[Mapping[str, object]] = []

    def __call__(
        self, url: str, headers: dict[str, str], payload: Mapping[str, object], timeout: float
    ) -> Answer:
        del url, headers
        self.timeouts.append(timeout)
        self.payloads.append(payload)
        step = next(self.steps)
        if isinstance(step, SubstrateError):
            raise step
        if callable(step):
            return step(self.fake)
        return step

    @property
    def calls(self) -> int:
        return len(self.timeouts)


def workspace() -> Workspace:
    return Workspace(
        system="SYS",
        user="USER",
        input_text="USER",
        trace_ids=(),
        estimated_tokens=2,
        budget=2000,
        traces_shown=0,
        traces_held=0,
        episodes_shown=0,
        episodes_held=0,
    )


def openai(
    transport: Transport,
    fake: FakeClock,
    policy: Policy | None = None,
    json_mode: bool = False,
    provider: str = "",
) -> OpenAICompatible:
    return OpenAICompatible(
        model="demo",
        base_url="https://api.example.test/v1",
        api_key=KEY,
        json_mode=json_mode,
        completion_field="max_tokens",
        transport=transport,
        policy=policy or Policy(),
        clock=fake.clock(),
        provider=provider,
    )


# Policy from the environment.


def test_the_policy_has_sane_defaults() -> None:
    policy = policy_from_env({})
    assert policy == Policy()
    assert (policy.connect_timeout, policy.read_timeout, policy.deadline) == (10, 90, 180)
    assert (policy.attempts, policy.max_wait) == (3, 30)


def test_the_policy_reads_the_environment() -> None:
    policy = policy_from_env(
        {
            "ANGULARAI_CONNECT_TIMEOUT": "3",
            "ANGULARAI_READ_TIMEOUT": "45.5",
            "ANGULARAI_DEADLINE": "60",
            "ANGULARAI_ATTEMPTS": "1",
            "ANGULARAI_MAX_WAIT": "0",
            "ANGULARAI_UNRELATED": "x",
        }
    )
    assert policy == Policy(
        connect_timeout=3, read_timeout=45.5, deadline=60, attempts=1, max_wait=0
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ANGULARAI_CONNECT_TIMEOUT", "0"),
        ("ANGULARAI_READ_TIMEOUT", "fast"),
        ("ANGULARAI_DEADLINE", "1e9"),
        ("ANGULARAI_DEADLINE", "nan"),
        ("ANGULARAI_ATTEMPTS", "2.5"),
        ("ANGULARAI_ATTEMPTS", "7"),
        ("ANGULARAI_MAX_WAIT", "-1"),
    ],
)
def test_a_bad_policy_value_names_the_variable(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        policy_from_env({name: value})


def test_build_substrate_names_the_provider_and_reads_the_policy() -> None:
    config = Config(
        substrate="openai",
        model="demo",
        base_url=None,
        workspace_budget=2000,
        max_input_tokens=1000,
        forbid=(),
        json_mode=False,
    )
    model = build_substrate(config, {"OPENAI_API_KEY": KEY, "ANGULARAI_ATTEMPTS": "2"})
    assert isinstance(model, OpenAICompatible)
    assert model._caller.policy.attempts == 2
    assert model._caller.provider == "OpenAI"
    with pytest.raises(ConfigError, match="ANGULARAI_DEADLINE"):
        build_substrate(config, {"OPENAI_API_KEY": KEY, "ANGULARAI_DEADLINE": "x"})


# Retry-After and the wait.


def test_retry_after_reads_every_form() -> None:
    later = format_datetime(datetime.fromtimestamp(NOW + 42, UTC), usegmt=True)
    earlier = format_datetime(datetime.fromtimestamp(NOW - 42, UTC), usegmt=True)
    assert retry_after({"Retry-After": "7"}, NOW) == 7
    assert retry_after({"retry-after": " 2.5 "}, NOW) == 2.5
    assert retry_after({"retry-after": "-3"}, NOW) == 0
    assert retry_after({"retry-after-ms": "1500"}, NOW) == 1.5
    assert retry_after({"retry-after-ms": "250", "retry-after": "9"}, NOW) == 0.25
    assert retry_after({"retry-after-ms": "soon", "retry-after": "9"}, NOW) == 9
    assert retry_after({"retry-after": later}, NOW) == pytest.approx(42)
    assert retry_after({"retry-after": earlier}, NOW) == 0
    assert retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00"}, NOW) is None
    assert retry_after({"retry-after": "soon"}, NOW) is None
    assert retry_after({"retry-after": "inf"}, NOW) is None
    assert retry_after({}, NOW) is None


def test_the_wait_is_full_jitter_capped_and_honors_the_provider() -> None:
    policy = Policy(max_wait=5)
    half = FakeClock(0.5).clock()
    top = FakeClock(0.999).clock()
    assert wait_before_retry(ServerError("x", 500), 1, policy, half) == 0.5
    assert wait_before_retry(ServerError("x", 500), 2, policy, half) == 1.0
    assert wait_before_retry(ServerError("x", 500), 3, policy, half) == 2.0
    assert wait_before_retry(ServerError("x", 500), 10, policy, top) == pytest.approx(4.995)
    assert wait_before_retry(Timeout("x"), 1, policy, FakeClock(0.0).clock()) == 0
    assert wait_before_retry(RateLimited("x", 4.0), 1, policy, half) == 4.0
    assert wait_before_retry(RateLimited("x", 6.0), 1, policy, half) is None
    assert wait_before_retry(RateLimited("x"), 1, policy, half) == 0.5
    for error in (
        AuthFailed("x"),
        Forbidden("x"),
        NotFound("x"),
        QuotaExceeded("x", code="insufficient_quota"),
        RequestRejected("x", 400),
        BadResponse("x"),
    ):
        assert wait_before_retry(error, 1, policy, half) is None


# What is retried, end to end.


@pytest.mark.parametrize(
    ("first", "kind"),
    [
        ((429, openai_error("rate_limit_exceeded", "requests", "slow"), {}), RateLimited),
        ((500, b"", {}), ServerError),
        ((502, b"<html><body>Bad gateway</body></html>", {}), ServerError),
        ((503, b"", {}), ServerError),
        ((529, b"", {}), Overloaded),
        (Timeout("The substrate timed out."), Timeout),
        (Unreachable("The substrate is unreachable."), Unreachable),
    ],
)
def test_a_retryable_failure_is_sent_again(first: Step, kind: type[SubstrateError]) -> None:
    fake = FakeClock()
    script = Script(fake, first, (200, good()))
    assert kind.retryable
    assert openai(script, fake).integrate(workspace()).reply == "Hi."
    assert script.calls == 2
    assert fake.slept == [0.5]
    assert script.payloads[0] == script.payloads[1]


def test_retry_after_seconds_and_dates_set_the_wait() -> None:
    fake = FakeClock()
    later = format_datetime(datetime.fromtimestamp(NOW + fake.now + 12, UTC), usegmt=True)
    script = Script(
        fake,
        (429, b"", {"retry-after": "3"}),
        (429, b"", {"retry-after": later}),
        (200, good()),
    )
    openai(script, fake).integrate(workspace())
    assert fake.slept[0] == 3
    assert fake.slept[1] == pytest.approx(9)  # the date was 12 s ahead; 3 s already passed


def test_retry_after_ms_sets_the_wait() -> None:
    fake = FakeClock()
    script = Script(fake, (429, b"", {"retry-after-ms": "750"}), (200, good()))
    openai(script, fake).integrate(workspace())
    assert fake.slept == [0.75]


def test_a_retry_after_longer_than_the_max_wait_ends_the_pass() -> None:
    fake = FakeClock()
    script = Script(fake, (429, b"", {"retry-after": "3600"}), (200, good()))
    with pytest.raises(RateLimited, match="Try again in 3600 seconds") as caught:
        openai(script, fake).integrate(workspace())
    assert caught.value.retry_after == 3600
    assert script.calls == 1
    assert fake.slept == []


def test_attempts_are_capped_and_the_waits_back_off() -> None:
    fake = FakeClock(random_value=1.0)
    script = Script(fake, (500, b"", {}), (503, b"", {}), (500, b"", {}), (200, good()))
    with pytest.raises(ServerError, match="unavailable"):
        openai(script, fake).integrate(workspace())
    assert script.calls == 3
    assert fake.slept == [1.0, 2.0]


def test_one_attempt_means_no_retry() -> None:
    fake = FakeClock()
    script = Script(fake, (529, b"", {}), (200, good()))
    with pytest.raises(Overloaded):
        openai(script, fake, Policy(attempts=1)).integrate(workspace())
    assert script.calls == 1


@pytest.mark.parametrize(
    ("answer", "kind", "words"),
    [
        (
            (429, openai_error("insufficient_quota", "insufficient_quota", "No credit."), {}),
            QuotaExceeded,
            "billing: insufficient_quota",
        ),
        ((402, b"", {}), QuotaExceeded, "billing: payment_required"),
        ((401, openai_error("invalid_api_key", "x", KEY), {}), AuthFailed, "rejected the API key"),
        ((403, openai_error(None, "permission_error", "No."), {}), Forbidden, "does not allow"),
        ((404, openai_error("model_not_found", "x", "No model."), {}), NotFound, "404"),
        ((400, openai_error(None, "invalid_request_error", "Bad."), {}), RequestRejected, "400"),
        ((422, b"", {}), RequestRejected, "returned 422"),
        ((200, b"<html>oops</html>", {}), BadResponse, "not JSON"),
    ],
)
def test_a_failure_that_waiting_cannot_fix_is_not_retried(
    answer: Answer, kind: type[SubstrateError], words: str
) -> None:
    fake = FakeClock()
    script = Script(fake, answer, (200, good()))
    with pytest.raises(kind, match=words) as caught:
        openai(script, fake).integrate(workspace())
    assert not caught.value.retryable
    assert script.calls == 1
    assert fake.slept == []
    assert KEY not in str(caught.value)


def test_typed_errors_keep_their_old_bases() -> None:
    assert issubclass(AuthFailed, KeyRejected) and issubclass(Forbidden, KeyRejected)
    unavailable: tuple[type[SubstrateError], ...] = (
        QuotaExceeded,
        RateLimited,
        NotFound,
        RequestRejected,
        Overloaded,
        ServerError,
        Timeout,
        Unreachable,
    )
    assert all(issubclass(kind, SubstrateUnavailable) for kind in unavailable)
    assert issubclass(BadResponse, SubstrateUnreadable)


# The deadline.


def test_each_attempt_reads_for_at_most_the_time_left() -> None:
    fake = FakeClock(random_value=0.0)

    def slow(fake: FakeClock) -> Answer:
        fake.now += 25
        return (503, b"", {})

    script = Script(fake, slow, (200, good()))
    openai(script, fake, Policy(read_timeout=20, deadline=40)).integrate(workspace())
    assert script.timeouts == [20, 15]


def test_the_deadline_ends_retries() -> None:
    fake = FakeClock(random_value=1.0)

    def slow(fake: FakeClock) -> Answer:
        fake.now += 9
        return (500, b"", {})

    script = Script(fake, slow, slow, slow)
    with pytest.raises(Timeout, match=r"did not answer within 10 seconds") as caught:
        openai(script, fake, Policy(deadline=10, attempts=3)).integrate(workspace())
    assert isinstance(caught.value.__cause__, ServerError)
    assert script.calls == 1  # a 1 s wait would reach the deadline; no second call
    assert fake.slept == []


def test_a_wait_that_would_pass_the_deadline_is_not_taken() -> None:
    fake = FakeClock()
    script = Script(fake, (429, b"", {"retry-after": "20"}), (200, good()))
    with pytest.raises(Timeout, match="within 15 seconds"):
        openai(script, fake, Policy(deadline=15)).integrate(workspace())
    assert fake.slept == []


def test_the_deadline_covers_the_whole_pass() -> None:
    fake = FakeClock()

    def rejected(fake: FakeClock) -> Answer:
        fake.now += 30
        return (400, openai_error(None, "invalid_request_error", "No json mode."), {})

    def too_late(fake: FakeClock) -> Answer:
        raise AssertionError("the fallback call ran past the deadline")

    script = Script(fake, rejected, too_late)
    with pytest.raises(Timeout, match="within 30 seconds"):
        openai(script, fake, Policy(deadline=30), json_mode=True).integrate(workspace())
    assert script.calls == 1


def test_the_json_mode_fallback_shares_the_deadline() -> None:
    fake = FakeClock()

    def rejected(fake: FakeClock) -> Answer:
        fake.now += 50
        return (400, openai_error(None, "invalid_request_error", "No json mode."), {})

    script = Script(fake, rejected, (200, good()))
    openai(script, fake, Policy(deadline=60, read_timeout=90), json_mode=True).integrate(
        workspace()
    )
    assert script.timeouts == [60, 10]
    assert "response_format" in script.payloads[0]
    assert "response_format" not in script.payloads[1]


# A retry writes once and counts once.


def test_a_retried_pass_is_written_once_and_counted_once(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    remember: list[object] = [{"text": "Teo likes tea.", "wavelengths": ["taste"]}]
    fake = FakeClock()
    once = pass_turn(store, openai(Script(fake, (200, good(remember=remember))), fake), "Hello.")

    other = tmp_path / "other"
    other.mkdir()
    init_cortex(other)
    second = open_store(other)
    fake = FakeClock()
    script = Script(
        fake,
        (429, b"", {"retry-after": "1"}),
        Unreachable("reset"),
        (200, good(remember=remember)),
    )
    retried = pass_turn(second, openai(script, fake), "Hello.")
    assert script.calls == 3
    assert retried.used == once.used and retried.budget == once.budget
    lines = (other / "passes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert len(second.view().traces) == len(store.view().traces) == 1


def test_a_pass_that_fails_after_retries_is_logged_once(tmp_path: Path) -> None:
    init_cortex(tmp_path)
    store = open_store(tmp_path)
    fake = FakeClock()
    script = Script(fake, (503, b"", {}), (503, b"", {}), (503, b"", {}))
    with pytest.raises(ServerError):
        pass_turn(store, openai(script, fake), "Hello.")
    assert len((tmp_path / "passes.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert store.view().traces == ()


# Bodies that are not what the provider documents.


@pytest.mark.parametrize(
    ("body", "words"),
    [
        (b"", "empty body"),
        (b"  \n", "empty body"),
        (b"<!DOCTYPE html><html><head><title>502</title></head><body>Gateway</body>", "not JSON"),
        (b'{"choices": [{"message": {"content": "hi', "not JSON"),
        (b"\xff\xfe", "not JSON"),
        (b"[1, 2]", "not JSON"),
        (b'"text"', "not JSON"),
        (b'{"error": {"message": "late failure"}}', "error and a success status"),
        (b'{"choices": "x"}', "no reply"),
        (b'{"choices": [1]}', "no reply"),
        (b'{"choices": [{"message": 3}]}', "no reply"),
        (b'{"choices": [{"message": {"content": null}}]}', "no reply"),
        (b'{"choices": [{"message": {"content": "  "}}]}', "no reply"),
        (b"{}", "no reply"),
    ],
)
def test_a_malformed_body_is_unreadable_and_never_a_crash(body: bytes, words: str) -> None:
    fake = FakeClock()
    with pytest.raises(SubstrateUnreadable, match=words):
        openai(Script(fake, (200, body)), fake).integrate(workspace())


@pytest.mark.parametrize(
    "body",
    [
        b'{"content": "x"}',
        b'{"content": [1, {"type": "image"}, {"type": "text", "text": 3}]}',
        b'{"type": "error", "error": {"type": "api_error", "message": "late"}}',
    ],
)
def test_a_malformed_anthropic_body_is_unreadable(body: bytes) -> None:
    fake = FakeClock()
    model = Anthropic(
        model="demo",
        base_url="https://api.example.test/v1",
        api_key=KEY,
        transport=Script(fake, (200, body)),
        clock=fake.clock(),
    )
    with pytest.raises(SubstrateUnreadable):
        model.integrate(workspace())


def test_a_bad_body_snippet_is_redacted_and_capped() -> None:
    fake = FakeClock()
    body = f"<html><script>var k = 1;</script><p>Upstream said {KEY} &amp; more</p>".encode()
    body += b"x" * 5000
    with pytest.raises(BadResponse) as caught:
        openai(Script(fake, (200, body)), fake).integrate(workspace())
    message = str(caught.value)
    assert KEY not in message
    assert "<" not in message and "var k" not in message
    assert "Upstream said [redacted] & more" in message
    assert len(message) < 300


def test_an_html_error_page_becomes_a_short_sentence() -> None:
    page = b"<html><head><style>p{}</style></head><body><h1>502 Bad Gateway</h1></body></html>"
    error = error_for_status(502, page, KEY)
    assert isinstance(error, ServerError) and error.status == 502
    assert str(error) == "The substrate is unavailable. 502 Bad Gateway"


def test_the_provider_name_is_in_the_message() -> None:
    error = error_for_status(401, b"", KEY, provider="OpenAI")
    assert str(error) == "The substrate (OpenAI) rejected the API key."
    fake = FakeClock()
    with pytest.raises(BadResponse, match=r"^The substrate \(compatible\) returned an empty"):
        openai(Script(fake, (200, b"")), fake, provider="compatible").integrate(workspace())


# The HTTP transport.


class FakeSocket:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def settimeout(self, value: float) -> None:
        self.timeout = value


class FakeResponse:
    def __init__(self, status: object, body: object, headers: list[tuple[str, str]]) -> None:
        self.status = status
        self._body = body
        self._headers = headers

    def getheaders(self) -> list[tuple[str, str]]:
        return self._headers

    def read(self) -> object:
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body


class FakeConnection(http.client.HTTPConnection):
    def __init__(
        self,
        response: FakeResponse | None = None,
        fail_at: str = "",
        error: BaseException | None = None,
    ) -> None:
        super().__init__("example.test")
        self.response = response
        self.fail_at = fail_at
        self.error = error
        self.sent: tuple[str, str, bytes, dict[str, str]] | None = None
        self.closed = False

    def _fail(self, where: str) -> None:
        if self.fail_at == where and self.error is not None:
            raise self.error

    def connect(self) -> None:
        self._fail("connect")
        self.sock = FakeSocket()

    def request(self, method: str, url: str, body: object = None, headers: object = None) -> None:  # type: ignore[override]
        self._fail("request")
        assert isinstance(body, bytes) and isinstance(headers, dict)
        self.sent = (method, url, body, headers)

    def getresponse(self) -> FakeResponse:  # type: ignore[override]
        self._fail("response")
        assert self.response is not None
        return self.response

    def close(self) -> None:
        self.closed = True


def through(conn: FakeConnection, extra: dict[str, str] | None = None) -> object:
    seen: dict[str, object] = {}

    def connect(url: SplitResult, connect_timeout: float) -> Route:
        seen["url"] = url.geturl()
        seen["connect_timeout"] = connect_timeout
        return conn, url.path or "/", extra or {}

    connect.seen = seen  # type: ignore[attr-defined]
    return connect


def test_http_sends_json_and_sets_both_timeouts() -> None:
    conn = FakeConnection(FakeResponse(200, b"{}", [("Content-Type", "application/json")]))
    connect = through(conn, {"Proxy-Authorization": "Basic x"})
    answer = http_json(
        "https://example.test/v1/chat",
        {"A": "1"},
        {"a": 1},
        12,
        connect_timeout=3,
        connect=connect,  # type: ignore[arg-type]
    )
    assert answer == (200, b"{}")
    assert connect.seen == {  # type: ignore[attr-defined]
        "url": "https://example.test/v1/chat",
        "connect_timeout": 3,
    }
    assert isinstance(conn.sock, FakeSocket) and conn.sock.timeout == 12
    assert conn.sent == (
        "POST",
        "/v1/chat",
        b'{"a": 1}',
        {"A": "1", "Proxy-Authorization": "Basic x"},
    )
    assert conn.closed


def test_http_returns_error_headers_and_never_follows_a_redirect() -> None:
    conn = FakeConnection(FakeResponse(429, b"{}", [("Retry-After", "7")]))
    assert http_json("http://example.test", {}, {}, connect=through(conn)) == (  # type: ignore[arg-type]
        429,
        b"{}",
        {"retry-after": "7"},
    )
    moved = FakeConnection(FakeResponse(302, b"", [("Location", "https://evil.test")]))
    status, _, headers = http_json("http://example.test", {}, {}, connect=through(moved))  # type: ignore[arg-type, misc]
    assert status == 302 and headers == {"location": "https://evil.test"}


def test_http_reads_a_bytearray_and_names_an_unreadable_status() -> None:
    conn = FakeConnection(FakeResponse(200, bytearray(b"{}"), []))
    assert http_json("http://example.test", {}, {}, connect=through(conn))[1] == b"{}"  # type: ignore[arg-type]
    odd = FakeConnection(FakeResponse("200", b"{}", []))
    with pytest.raises(SubstrateUnavailable, match="unreadable status"):
        http_json("http://example.test", {}, {}, connect=through(odd))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("where", "error", "kind", "words"),
    [
        ("connect", TimeoutError("connect"), Timeout, "timed out"),
        ("response", TimeoutError("read"), Timeout, "timed out"),
        ("read", TimeoutError("read"), Timeout, "timed out"),
        ("connect", ConnectionRefusedError(), Unreachable, "unreachable"),
        ("request", ConnectionResetError(), Unreachable, "unreachable"),
        ("response", http.client.RemoteDisconnected("gone"), Unreachable, "unreachable"),
        ("read", http.client.IncompleteRead(b"{", 10), BadResponse, "cut short"),
        ("connect", OSError("dns"), Unreachable, "unreachable"),
    ],
)
def test_http_maps_each_transport_failure(
    where: str, error: BaseException, kind: type[SubstrateError], words: str
) -> None:
    body: object = error if where == "read" else b"{}"
    conn = FakeConnection(FakeResponse(200, body, []), where, error)
    with pytest.raises(kind, match=words):
        http_json("http://example.test", {}, {}, connect=through(conn))  # type: ignore[arg-type]
    assert conn.closed


def test_http_refuses_a_url_that_is_not_http() -> None:
    with pytest.raises(Unreachable, match="not http"):
        http_json("file:///etc/passwd", {}, {})
    with pytest.raises(Unreachable, match="not http"):
        http_json("https://", {}, {})


def test_connection_goes_direct_without_a_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("angularai.http.getproxies", lambda: {})
    conn, target, extra = connection(urlsplit("https://api.example.test/v1/x?y=1"), 4)
    assert isinstance(conn, http.client.HTTPSConnection)
    assert (conn.host, conn.port, conn.timeout) == ("api.example.test", 443, 4)
    assert (target, extra) == ("/v1/x?y=1", {})
    plain, target, _ = connection(urlsplit("http://localhost:8080"), 4)
    assert not isinstance(plain, http.client.HTTPSConnection)
    assert (plain.host, plain.port, target) == ("localhost", 8080, "/")


def test_connection_tunnels_https_through_a_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "angularai.http.getproxies", lambda: {"https": "http://user:p%40ss@proxy.test:3128"}
    )
    monkeypatch.setattr("angularai.http.proxy_bypass", lambda host: False)
    conn, target, extra = connection(urlsplit("https://api.example.test/v1/x"), 4)
    assert isinstance(conn, http.client.HTTPSConnection)
    assert (conn.host, conn.port) == ("proxy.test", 3128)
    assert conn._tunnel_host == "api.example.test" and conn._tunnel_port == 443  # type: ignore[attr-defined]
    assert conn._tunnel_headers["Proxy-Authorization"] == "Basic dXNlcjpwQHNz"  # type: ignore[attr-defined]
    assert (target, extra) == ("/v1/x", {})


def test_connection_sends_plain_http_to_the_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("angularai.http.getproxies", lambda: {"http": "proxy.test:8000"})
    monkeypatch.setattr("angularai.http.proxy_bypass", lambda host: False)
    conn, target, extra = connection(urlsplit("http://api.example.test/v1"), 4)
    assert (conn.host, conn.port) == ("proxy.test", 8000)
    assert (target, extra) == ("http://api.example.test/v1", {})


def test_connection_skips_the_proxy_for_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("angularai.http.getproxies", lambda: {"https": "http://proxy.test:3128"})
    monkeypatch.setattr("angularai.http.proxy_bypass", lambda host: host == "api.example.test")
    conn, _, _ = connection(urlsplit("https://api.example.test/v1"), 4)
    assert conn.host == "api.example.test"
