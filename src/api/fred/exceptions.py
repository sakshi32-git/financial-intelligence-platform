"""
exceptions.py
=============
Domain exceptions for the FRED (Federal Reserve Economic Data) API client.

All exceptions inherit from :exc:`FREDError` so callers can catch the
entire hierarchy with a single clause when needed.

Hierarchy
---------
FREDError
├── FREDConnectionError    — network / transport failure
├── FREDRateLimitError     — HTTP 429 / too-many-requests
├── FREDNotFoundError      — series_id not found or empty response
├── FREDAuthError          — invalid or missing API key (HTTP 400 / 403)
├── FREDTimeoutError       — request exceeded allowed wall-clock time
└── FREDValidationError    — caller passed invalid arguments
"""

from __future__ import annotations


class FREDError(Exception):
    """Base exception for all FRED API client errors."""

    def __init__(self, message: str, series_id: str | None = None) -> None:
        super().__init__(message)
        self.series_id = series_id

    def __str__(self) -> str:
        prefix = f"[{self.series_id}] " if self.series_id else ""
        return f"{prefix}{super().__str__()}"


class FREDConnectionError(FREDError):
    """Raised when a network or transport failure prevents the request."""


class FREDRateLimitError(FREDError):
    """Raised when the FRED API signals too-many-requests (HTTP 429)."""


class FREDNotFoundError(FREDError):
    """Raised when the series_id is unknown or the API returns no data."""


class FREDAuthError(FREDError):
    """Raised when the API key is invalid, missing, or unauthorised."""


class FREDTimeoutError(FREDError):
    """Raised when the request wall-clock time exceeds the configured limit."""


class FREDValidationError(FREDError):
    """Raised when the caller supplies invalid arguments."""
