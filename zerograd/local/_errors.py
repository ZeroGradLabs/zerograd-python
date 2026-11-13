from __future__ import annotations

from typing import Any

from ..errors import ValidationError


def Invalid(message: str, *, code: str | None = None, **details: Any) -> ValidationError:  # noqa: N802
    """The data cannot be used as given. Same codes as the platform's 422 errors."""
    return ValidationError(message, code=code, details=details or None)
