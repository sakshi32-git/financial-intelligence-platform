"""
exceptions.py
=============
Domain exceptions for the Yahoo Finance API client.

All exceptions inherit from :exc:`YahooFinanceError` so callers can catch
the entire hierarchy with a single clause when needed.

Hierarchy
---------
YahooFinanceError
├── YahooFinanceConnectionError   — network / transport failure
├── YahooFinanceRateLimitError    — HTTP 429 / too-many-requests
├── YahooFinanceNotFoundError     — unknown ticker or no data returned
├── YahooFinanceTimeoutError      — request exceeded allowed wall-clock time
└── YahooFinanceValidationError   — caller passed invalid arguments
"""

from __future__ import annotations


class YahooFinanceError(Exception):
    """Base exception for all Yahoo Finance client errors."""

    def __init__(self, message: str, ticker: str | None = None) -> None:
        super().__init__(message)
        self.ticker = ticker

    def __str__(self) -> str:
        prefix = f"[{self.ticker}] " if self.ticker else ""
        return f"{prefix}{super().__str__()}"


class YahooFinanceConnectionError(YahooFinanceError):
    """Raised when a network or transport failure prevents the request."""


class YahooFinanceRateLimitError(YahooFinanceError):
    """Raised when Yahoo Finance signals too-many-requests (HTTP 429)."""


class YahooFinanceNotFoundError(YahooFinanceError):
    """Raised when the ticker is unknown or the API returns no data."""


class YahooFinanceTimeoutError(YahooFinanceError):
    """Raised when the request wall-clock time exceeds the configured limit."""


class YahooFinanceValidationError(YahooFinanceError):
    """Raised when the caller supplies invalid arguments."""
