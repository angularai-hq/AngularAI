"""Errors an operator can read. None of them include secrets."""


class ClaspError(Exception):
    """Safe to print."""


class NoCortex(ClaspError):
    """The path is not a cortex."""


class CortexExists(ClaspError):
    """Refusing to overwrite a cortex."""


class ConfigError(ClaspError):
    """The cortex file on disk is not a valid config."""


class LogDamaged(ClaspError):
    """A line in the cortex log cannot be read."""


class InputRejected(ClaspError):
    """The input never entered the cortex."""


class InputDoesNotFit(ClaspError):
    """The fixed part of the workspace is already over budget."""


class SubstrateError(ClaspError):
    """The model API failed. Traces from this pass were not written."""


class KeyMissing(SubstrateError):
    """The environment has no API key."""


class KeyRejected(SubstrateError):
    """The API refused the key."""


class SubstrateUnavailable(SubstrateError):
    """The API could not complete the pass."""


class SubstrateUnreadable(SubstrateError):
    """The API answered, but not with a cortex reply."""


class LockBypass(ClaspError):
    """A reply tried to leave without an L4 verdict."""
