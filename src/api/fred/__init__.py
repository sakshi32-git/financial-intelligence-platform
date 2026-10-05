"""Federal Reserve (FRED) API client."""

from src.api.fred.client import FREDClient  # noqa: F401
from src.api.fred.exceptions import (  # noqa: F401
    FREDAuthError,
    FREDConnectionError,
    FREDError,
    FREDNotFoundError,
    FREDRateLimitError,
    FREDTimeoutError,
    FREDValidationError,
)

__all__ = [
    "FREDClient",
    "FREDError",
    "FREDAuthError",
    "FREDConnectionError",
    "FREDNotFoundError",
    "FREDRateLimitError",
    "FREDTimeoutError",
    "FREDValidationError",
]
