"""Errors an operator can read. None of them include secrets."""


class AngularAIError(Exception):
    """Safe to print."""


class NoCortex(AngularAIError):
    """The path is not a cortex."""


class CortexExists(AngularAIError):
    """Refusing to overwrite a cortex."""


class ConfigError(AngularAIError):
    """The cortex file on disk, or an ANGULARAI_ setting, is not a valid config."""


class LogDamaged(AngularAIError):
    """A line in the cortex log cannot be read."""


class InputRejected(AngularAIError):
    """The input never entered the cortex."""


class InputTooLong(InputRejected):
    """The input is over a length cap. Nothing was written."""


class SecretRefused(InputRejected):
    """The input looks like a secret. Nothing was sent or written."""


class LockHeld(InputRejected):
    """The value lock holds the text: a locked phrase or a secret. Nothing was written."""


class TraceNotFound(AngularAIError):
    """No trace in the cortex has that id."""


class InputDoesNotFit(AngularAIError):
    """The fixed part of the workspace is already over budget."""


class SubstrateError(AngularAIError):
    """The model API failed. Traces from this pass were not written."""

    retryable = False


class KeyMissing(SubstrateError):
    """The environment has no API key."""


class KeyRejected(SubstrateError):
    """The API refused the key."""


class SubstrateUnavailable(SubstrateError):
    """The API could not complete the pass."""


class SubstrateUnreadable(SubstrateError):
    """The API answered, but not with a cortex reply."""


# The provider taxonomy. retryable says whether the same call may be sent again.
# Each class keeps the older base it replaces, so `except SubstrateUnavailable`
# and `except KeyRejected` still catch what they caught before.


class AuthFailed(KeyRejected):
    """401. The key is wrong or revoked."""

    retryable = False


class Forbidden(KeyRejected):
    """403. The key is valid and not allowed to do this."""

    retryable = False


class QuotaExceeded(SubstrateUnavailable):
    """Billing or quota. Waiting does not help."""

    retryable = False

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class RateLimited(SubstrateUnavailable):
    """A real rate limit. retry_after is in seconds when the provider said."""

    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NotFound(SubstrateUnavailable):
    """404. Usually the model name or the base URL."""

    retryable = False


class RequestRejected(SubstrateUnavailable):
    """Any other 4xx. The request itself is wrong; sending it again will not help."""

    retryable = False

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class Overloaded(SubstrateUnavailable):
    """529 or an overloaded error. The provider asks for time."""

    retryable = True


class ServerError(SubstrateUnavailable):
    """5xx."""

    retryable = True

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class Timeout(SubstrateUnavailable):
    """No answer within the connect or read timeout, or the pass deadline."""

    retryable = True


class Unreachable(SubstrateUnavailable):
    """The connection failed or was reset."""

    retryable = True


class BadResponse(SubstrateUnreadable):
    """2xx, but the body is not the JSON shape the provider documents."""

    retryable = False


class LockBypass(AngularAIError):
    """A reply tried to leave without an L4 verdict."""
