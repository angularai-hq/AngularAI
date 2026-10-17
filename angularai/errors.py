"""Errors an operator can read. None of them include secrets."""


class AngularAIError(Exception):
    """Safe to print."""


class NoCortex(AngularAIError):
    """The path is not a cortex."""


class CortexExists(AngularAIError):
    """Refusing to overwrite a cortex."""


class ConfigError(AngularAIError):
    """The cortex file on disk is not a valid config."""


class LogDamaged(AngularAIError):
    """A line in the cortex log cannot be read."""


class InputRejected(AngularAIError):
    """The input never entered the cortex."""


class InputDoesNotFit(AngularAIError):
    """The fixed part of the workspace is already over budget."""


class SubstrateError(AngularAIError):
    """The model API failed. Traces from this pass were not written."""


class KeyMissing(SubstrateError):
    """The environment has no API key."""


class KeyRejected(SubstrateError):
    """The API refused the key."""


class SubstrateUnavailable(SubstrateError):
    """The API could not complete the pass."""


class SubstrateUnreadable(SubstrateError):
    """The API answered, but not with a cortex reply."""


class LockBypass(AngularAIError):
    """A reply tried to leave without an L4 verdict."""
