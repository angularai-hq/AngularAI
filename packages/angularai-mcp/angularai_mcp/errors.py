"""Typed tool errors. A client reads ``[code] sentence``, never a traceback.

Every message comes from an AngularAI error, which never holds a secret, and is
redacted again against the environment's secret values before it leaves.
"""

import logging
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError

from angularai.errors import (
    AngularAIError,
    ConfigError,
    InputRejected,
    InputTooLong,
    LockHeld,
    LogDamaged,
    NoCortex,
    SecretRefused,
    SubstrateError,
    TraceNotFound,
)
from angularai.secrets import redact

Code = Literal[
    "invalid_input",
    "too_long",
    "secret_refused",
    "locked",
    "not_found",
    "no_cortex",
    "config_error",
    "log_damaged",
    "unavailable",
    "storage_error",
    "internal",
]
CODES: tuple[Code, ...] = (
    "invalid_input",
    "too_long",
    "secret_refused",
    "locked",
    "not_found",
    "no_cortex",
    "config_error",
    "log_damaged",
    "unavailable",
    "storage_error",
    "internal",
)
_MAX_MESSAGE = 300
logger = logging.getLogger("angularai_mcp")


class CortexToolError(ToolError):
    """A failure the client can act on. str() is ``[code] message``."""

    def __init__(self, code: Code, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code: Code = code
        self.message = message


def code_for(exc: BaseException) -> Code:
    if isinstance(exc, SecretRefused):
        return "secret_refused"
    if isinstance(exc, LockHeld):
        return "locked"
    if isinstance(exc, InputTooLong):
        return "too_long"
    if isinstance(exc, TraceNotFound):
        return "not_found"
    if isinstance(exc, NoCortex):
        return "no_cortex"
    if isinstance(exc, ConfigError):
        return "config_error"
    if isinstance(exc, LogDamaged):
        return "log_damaged"
    if isinstance(exc, SubstrateError):
        return "unavailable"
    if isinstance(exc, InputRejected | AngularAIError):
        return "invalid_input"
    if isinstance(exc, OSError):
        return "storage_error"
    return "internal"


def tool_error(exc: BaseException, secrets: tuple[str, ...]) -> CortexToolError:
    """The typed error for exc. Only an AngularAI error keeps its words."""

    code = code_for(exc)
    if isinstance(exc, AngularAIError):
        message = " ".join(redact(str(exc), secrets).split())[:_MAX_MESSAGE]
    elif code == "storage_error":
        message = "The cortex could not be read or written."
    else:
        # The exception's text can hold anything. Only its type reaches the log.
        logger.warning("unexpected %s in a tool call", type(exc).__name__)
        message = "Unexpected failure. Its details stay on the server."
    return CortexToolError(code, message or code)
