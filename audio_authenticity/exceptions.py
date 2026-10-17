"""Domain errors with safe, actionable messages for API callers."""


class AudioInputError(ValueError):
    """Audio file is missing, malformed, unsupported, or outside limits."""


class ModelUnavailableError(RuntimeError):
    """Selected inference adapter cannot be initialized in this environment."""


class InferenceError(RuntimeError):
    """A ready model failed while processing an input."""
