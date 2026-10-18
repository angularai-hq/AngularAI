"""Properties of the provider layer: Retry-After, backoff, error bodies."""

from hypothesis import example, given
from hypothesis import strategies as st

from angularai.errors import (
    BadResponse,
    KeyRejected,
    Overloaded,
    RateLimited,
    ServerError,
    SubstrateError,
    SubstrateUnavailable,
    Timeout,
    Unreachable,
)
from angularai.provider import Clock, Policy, retry_after, wait_before_retry
from angularai.substrate import _object, error_for_status, provider_error

header = st.sampled_from(["Retry-After", "retry-after", "RETRY-AFTER", "retry-after-ms", "x"])


@given(st.dictionaries(header, st.text()), st.floats(allow_nan=True, allow_infinity=True))
@example({"retry-after": "Fri, 31 Dec 9999 23:59:59 GMT"}, 0.0)
@example({"retry-after": "Mon, 01 Jan 0001 00:00:00 GMT"}, 1e18)
@example({"retry-after": "1e400"}, 0.0)
@example({"retry-after-ms": "-0"}, 0.0)
@example({"retry-after": "Thu, 01 Jan 1970 00:00:00 +9999"}, 0.0)
def test_retry_after_never_raises_and_is_never_negative(
    headers: dict[str, str], now: float
) -> None:
    wait = retry_after(headers, now)
    assert wait is None or wait >= 0


retryable: list[SubstrateError] = [
    RateLimited("x"),
    Overloaded("x"),
    ServerError("x", 500),
    Timeout("x"),
    Unreachable("x"),
]
policies = st.builds(
    Policy,
    attempts=st.integers(1, 6),
    max_wait=st.floats(0, 120),
    base_wait=st.floats(0, 10),
)


@given(
    st.sampled_from(retryable) | st.builds(RateLimited, st.just("x"), st.floats(0, 1e9)),
    st.integers(-5, 10_000),
    policies,
    st.floats(0, 1, exclude_max=True),
)
# Found by Hypothesis: 2.0 ** 1024 overflowed before the cap applied.
@example(ServerError("x", 500), 1025, Policy(), 0.5)
def test_backoff_stays_within_zero_and_max_wait(
    error: SubstrateError, attempt: int, policy: Policy, random: float
) -> None:
    wait = wait_before_retry(error, attempt, policy, Clock(random=lambda: random))
    if wait is not None:
        assert 0 <= wait <= policy.max_wait


@given(st.binary(max_size=2000))
# Found while writing this property: deep nesting and long ints escaped json.loads.
@example(b"[" * 100_000)
@example(b'{"error": ' * 5000)
@example(b"1" * 5000)
@example(b'{"error": {"code": ' + b"9" * 5000 + b"}}")
def test_error_bodies_never_crash(body: bytes) -> None:
    labels, message = provider_error(body)
    assert isinstance(message, str)
    error = error_for_status(400, body, "sk-test-provider-secret", {})
    assert isinstance(error, SubstrateUnavailable | KeyRejected)
    assert len(str(error)) < 400
    try:
        _object(body)
    except BadResponse:
        pass
