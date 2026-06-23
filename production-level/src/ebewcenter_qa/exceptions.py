"""Project-specific exceptions."""


class QAError(Exception):
    """Base exception for this package."""


class DataValidationError(QAError):
    """Raised when training or inference data is invalid."""
