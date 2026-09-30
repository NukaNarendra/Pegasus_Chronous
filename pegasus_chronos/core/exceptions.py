"""
PEGASUS-CHRONOS API Resilience Layer
Custom exceptions for the three-layer enterprise-grade resilience architecture
mirrorin the ARGUS proactive throttling + graceful fallback design.
"""
from __future__ import annotations


class PegasusAPIError(Exception):
    """Base exception for all PEGASUS API errors."""
    def __init__(self, message: str, status_code: int = 0, model: str = ""):
        self.message = message
        self.status_code = status_code
        self.model = model
        super().__init__(f"[{status_code}] {model}: {message}")


class ContextWindowExceededError(PegasusAPIError):
    """Raised when request payload exceeds model/tier context window (HTTP 413 or token limit errors).
    This is NOT retryable — the payload must be reduced or fallback data used."""
    def __init__(self, message: str, model: str = "", requested_tokens: int = 0, limit_tokens: int = 0):
        self.requested_tokens = requested_tokens
        self.limit_tokens = limit_tokens
        super().__init__(message=message, status_code=413, model=model)


class MaxRetriesExceededError(PegasusAPIError):
    """Raised when all retry attempts have been exhausted for rate-limited (429) or server (5xx) errors."""
    def __init__(self, message: str, model: str = "", attempts: int = 0):
        self.attempts = attempts
        super().__init__(message=message, status_code=429, model=model)


class ModelDeprecatedError(PegasusAPIError):
    """Raised when a model has been retired/EOL (HTTP 410 Gone)."""
    def __init__(self, message: str, model: str = ""):
        super().__init__(message=message, status_code=410, model=model)
