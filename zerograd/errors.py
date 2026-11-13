from __future__ import annotations

from typing import Any


class ZeroGradError(Exception):
    """Base class. `code` and `message` come from the API's error body."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.details = details

    def __str__(self) -> str:
        return self.message


class AuthError(ZeroGradError):
    """The API key is missing or invalid (401/403)."""


class NotFoundError(ZeroGradError):
    """The capability, run, pipeline or upload does not exist (404)."""


class ValidationError(ZeroGradError):
    """The data cannot be used as sent: a missing column, too many classes, a bad schedule (400/409/422)."""


class LimitError(ZeroGradError):
    """A size or usage limit was reached (413/429)."""


class RunFailedError(ZeroGradError):
    """A run finished as failed or canceled. `run` holds the final run."""

    def __init__(self, message: str, *, run: Any = None, code: str | None = None):
        super().__init__(message, code=code)
        self.run = run


class APIError(ZeroGradError):
    """Anything else the server answered with (5xx)."""


class UsageWarning(UserWarning):
    """Raised as a warning when the workspace is past 80% of its monthly rows."""


def from_response(status: int, body: Any) -> ZeroGradError:
    if isinstance(body, dict) and "message" in body:
        message, code, details = body["message"], body.get("error"), body.get("details")
    elif isinstance(body, dict) and "detail" in body:  # FastAPI validation errors
        message, code, details = str(body["detail"]), "invalid_request", None
    else:
        message, code, details = f"HTTP {status}: {str(body)[:300]}", None, None
    cls: type[ZeroGradError]
    if status in (401, 403):
        cls = AuthError
    elif status == 404:
        cls = NotFoundError
    elif status in (413, 429) and code != "rate_limited":
        cls = LimitError
    elif 400 <= status < 500:
        cls = ValidationError
    else:
        cls = APIError
    return cls(message, code=code, status=status, details=details)
