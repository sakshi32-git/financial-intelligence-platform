"""
client.py
=========
FRED (Federal Reserve Economic Data) REST API client.

Calls the official St. Louis Fed JSON API at
``https://api.stlouisfed.org/fred/`` using ``httpx`` with automatic
retries via ``tenacity``.

Public API
----------
``FREDClient``
    .get_series(series_id, start, end, frequency,
                aggregation_method, units)          → DataFrame
    .get_series_info(series_id)                     → DataFrame
    .search_series(search_text, limit, order_by,
                   sort_order)                      → DataFrame
    .get_release_series(release_id, limit)          → DataFrame

All public methods return a ``pandas.DataFrame``.
No database I/O. No ETL logic.

Usage
-----
    from src.api.fred.client import FREDClient

    client = FREDClient()

    # GDP observations (annual)
    gdp = client.get_series("GDP", start="2010-01-01", end="2024-01-01")

    # Series metadata
    info = client.get_series_info("UNRATE")

    # Search
    results = client.search_series("unemployment rate", limit=20)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Literal

import httpx
import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from src.api.fred.exceptions import (
    FREDAuthError,
    FREDConnectionError,
    FREDError,
    FREDNotFoundError,
    FREDRateLimitError,
    FREDTimeoutError,
    FREDValidationError,
)

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.stlouisfed.org/fred"

_VALID_FREQUENCIES: frozenset[str] = frozenset(
    {
        "d",  # Daily
        "w",  # Weekly
        "bw",  # Biweekly
        "m",  # Monthly
        "q",  # Quarterly
        "sa",  # Semiannual
        "a",  # Annual
    }
)

_VALID_AGGREGATIONS: frozenset[str] = frozenset({"avg", "sum", "eop"})

_VALID_UNITS: frozenset[str] = frozenset(
    {
        "lin",  # Levels (no transformation)
        "chg",  # Change
        "ch1",  # Change from 1 year ago
        "pch",  # Percent change
        "pc1",  # Percent change from 1 year ago
        "pca",  # Compounded annual rate of change
        "cch",  # Continuously compounded rate of change
        "cca",  # Continuously compounded annual rate of change
        "log",  # Natural log
    }
)

_VALID_SORT_ORDERS: frozenset[str] = frozenset({"asc", "desc"})

_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.TimeoutException,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _date_str(value: str | date | datetime | None) -> str | None:
    """Convert a date-like to the ``YYYY-MM-DD`` string FRED expects."""
    if value is None:
        return None
    if isinstance(value, str):
        # Validate by parsing then re-serialising.
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"Cannot convert {type(value)} to a date string.")


# ===========================================================================
# Client
# ===========================================================================


class FREDClient:
    """
    Typed, retry-enabled HTTP client for the FRED JSON REST API.

    Parameters
    ----------
    api_key:
        FRED API key.  Defaults to ``settings.api.fred_api_key``.
        Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html
    request_timeout:
        Per-request timeout in seconds.  Default: 15.
    max_retries:
        Maximum total attempts (1 initial + N-1 retries).  Default: 4.
    """

    def __init__(
        self,
        api_key: str | None = None,
        request_timeout: int = 15,
        max_retries: int = 4,
    ) -> None:
        resolved_key = api_key or settings.api.fred_api_key.get_secret_value()
        if not resolved_key or resolved_key.startswith("your-"):
            raise FREDValidationError(
                "A valid FRED API key is required. "
                "Set FRED_API_KEY in your .env file or pass api_key= explicitly."
            )
        if request_timeout < 1:
            raise FREDValidationError(
                f"request_timeout must be >= 1, got {request_timeout}."
            )
        if max_retries < 1:
            raise FREDValidationError(f"max_retries must be >= 1, got {max_retries}.")

        self._api_key = resolved_key
        self._timeout = request_timeout
        self._max_retries = max_retries

        # Shared httpx client — connection pool is reused across requests.
        self._http = httpx.Client(
            base_url=_BASE_URL,
            timeout=httpx.Timeout(self._timeout),
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )

        log.info(
            "FREDClient initialised",
            extra={"timeout": self._timeout, "max_retries": self._max_retries},
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()
        log.info("FREDClient connection pool closed.")

    def __enter__(self) -> "FREDClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_series(
        self,
        series_id: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
        units: str = "lin",
    ) -> pd.DataFrame:
        """
        Fetch time-series observations for a FRED series.

        Parameters
        ----------
        series_id:
            FRED series identifier, e.g. ``"GDP"``, ``"UNRATE"``, ``"CPIAUCSL"``.
        start:
            Inclusive observation start date.  ``None`` = earliest available.
        end:
            Inclusive observation end date.  ``None`` = latest available.
        frequency:
            Optional frequency aggregation: ``"d"``, ``"w"``, ``"m"``,
            ``"q"``, ``"sa"``, ``"a"``.  ``None`` = native series frequency.
        aggregation_method:
            How to aggregate when down-sampling: ``"avg"``, ``"sum"``,
            ``"eop"`` (end of period).  Required when ``frequency`` is set.
        units:
            Data transformation: ``"lin"`` (levels), ``"pch"`` (% change),
            ``"pc1"`` (% change from 1 year ago), etc.  Default: ``"lin"``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``date`` (``datetime64``), ``value`` (``float64``),
            ``series_id`` (``str``).
            Rows where ``value`` is ``"."`` (FRED missing sentinel) are
            replaced with ``NaN``.

        Raises
        ------
        FREDValidationError
            Invalid arguments.
        FREDAuthError
            Bad API key.
        FREDNotFoundError
            Series not found or no observations in range.
        FREDRateLimitError
            HTTP 429.
        FREDConnectionError / FREDTimeoutError
            Network failures.
        """
        series_id = self._validate_series_id(series_id)
        self._validate_units(units)
        if frequency is not None:
            self._validate_frequency(frequency)
        if aggregation_method is not None:
            self._validate_aggregation(aggregation_method)
        if frequency is None and aggregation_method is not None:
            raise FREDValidationError(
                "aggregation_method requires frequency to be set.",
                series_id=series_id,
            )

        params: dict[str, Any] = {
            "series_id": series_id,
            "units": units,
            "file_type": "json",
        }
        if start is not None:
            params["observation_start"] = _date_str(start)
        if end is not None:
            params["observation_end"] = _date_str(end)
        if frequency is not None:
            params["frequency"] = frequency
        if aggregation_method is not None:
            params["aggregation_method"] = aggregation_method

        log.info(
            "Fetching FRED series observations",
            extra={"series_id": series_id, "params": params},
        )

        data = self._get("/series/observations", params=params, series_id=series_id)
        observations: list[dict] = data.get("observations", [])

        if not observations:
            raise FREDNotFoundError(
                "No observations returned. The series may be empty for the "
                "requested date range.",
                series_id=series_id,
            )

        df = pd.DataFrame(observations)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["series_id"] = series_id
        df = df.sort_values("date").reset_index(drop=True)

        log.info(
            "FRED series observations fetched",
            extra={"series_id": series_id, "rows": len(df)},
        )
        return df

    def get_series_info(self, series_id: str) -> pd.DataFrame:
        """
        Return metadata for a FRED series as a single-row DataFrame.

        Columns include: ``id``, ``realtime_start``, ``realtime_end``,
        ``title``, ``observation_start``, ``observation_end``,
        ``frequency``, ``frequency_short``, ``units``, ``units_short``,
        ``seasonal_adjustment``, ``last_updated``, ``popularity``, ``notes``.

        Parameters
        ----------
        series_id:
            FRED series identifier.

        Returns
        -------
        pandas.DataFrame
            One row containing all series metadata fields.

        Raises
        ------
        FREDNotFoundError
            Series not found.
        """
        series_id = self._validate_series_id(series_id)
        log.info("Fetching FRED series info", extra={"series_id": series_id})

        data = self._get(
            "/series",
            params={"series_id": series_id, "file_type": "json"},
            series_id=series_id,
        )
        series_list: list[dict] = data.get("seriess", [])

        if not series_list:
            raise FREDNotFoundError(
                "Series not found in FRED catalogue.",
                series_id=series_id,
            )

        df = pd.DataFrame(series_list)
        log.info(
            "FRED series info fetched",
            extra={"series_id": series_id, "fields": len(df.columns)},
        )
        return df

    def search_series(
        self,
        search_text: str,
        limit: int = 100,
        order_by: str = "search_rank",
        sort_order: str = "asc",
    ) -> pd.DataFrame:
        """
        Full-text search across FRED series titles and descriptions.

        Parameters
        ----------
        search_text:
            Search query, e.g. ``"unemployment rate"`` or ``"CPI"``.
        limit:
            Maximum number of results (1–1000).  Default: 100.
        order_by:
            Field to sort by.  Common values: ``"search_rank"``,
            ``"series_id"``, ``"title"``, ``"popularity"``,
            ``"observation_start"``, ``"last_updated"``.
        sort_order:
            ``"asc"`` or ``"desc"``.

        Returns
        -------
        pandas.DataFrame
            Columns match FRED series metadata fields plus a ``search_rank``
            column when ordering by relevance.

        Raises
        ------
        FREDValidationError
            Empty search text, out-of-range limit, or invalid sort order.
        FREDNotFoundError
            No series matched the query.
        """
        if not search_text or not search_text.strip():
            raise FREDValidationError("search_text must be a non-empty string.")
        if not (1 <= limit <= 1000):
            raise FREDValidationError(f"limit must be between 1 and 1000, got {limit}.")
        if sort_order not in _VALID_SORT_ORDERS:
            raise FREDValidationError(
                f"sort_order={sort_order!r} is invalid. Choose 'asc' or 'desc'."
            )

        log.info(
            "Searching FRED series",
            extra={"search_text": search_text, "limit": limit},
        )

        data = self._get(
            "/series/search",
            params={
                "search_text": search_text.strip(),
                "limit": limit,
                "order_by": order_by,
                "sort_order": sort_order,
                "file_type": "json",
            },
        )
        series_list: list[dict] = data.get("seriess", [])

        if not series_list:
            raise FREDNotFoundError(
                f"No FRED series matched search_text={search_text!r}."
            )

        df = pd.DataFrame(series_list)
        log.info(
            "FRED series search complete",
            extra={"search_text": search_text, "results": len(df)},
        )
        return df

    def get_release_series(
        self,
        release_id: int,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Return all series belonging to a FRED release.

        Parameters
        ----------
        release_id:
            Numeric FRED release identifier (e.g. 53 = GDP, 10 = CPI).
        limit:
            Maximum number of series to return (1–1000).  Default: 1000.

        Returns
        -------
        pandas.DataFrame
            One row per series; columns match FRED series metadata fields.

        Raises
        ------
        FREDValidationError
            Invalid release_id or limit.
        FREDNotFoundError
            Release not found or contains no series.
        """
        if not isinstance(release_id, int) or release_id < 1:
            raise FREDValidationError(
                f"release_id must be a positive integer, got {release_id!r}."
            )
        if not (1 <= limit <= 1000):
            raise FREDValidationError(f"limit must be between 1 and 1000, got {limit}.")

        log.info(
            "Fetching FRED release series",
            extra={"release_id": release_id, "limit": limit},
        )

        data = self._get(
            "/release/series",
            params={
                "release_id": release_id,
                "limit": limit,
                "file_type": "json",
            },
        )
        series_list: list[dict] = data.get("seriess", [])

        if not series_list:
            raise FREDNotFoundError(f"No series found for release_id={release_id}.")

        df = pd.DataFrame(series_list)
        log.info(
            "FRED release series fetched",
            extra={"release_id": release_id, "results": len(df)},
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers — validation
    # ------------------------------------------------------------------

    def _validate_series_id(self, series_id: str) -> str:
        if not isinstance(series_id, str) or not series_id.strip():
            raise FREDValidationError(
                f"series_id must be a non-empty string, got {series_id!r}."
            )
        return series_id.strip().upper()

    def _validate_frequency(self, frequency: str) -> None:
        if frequency not in _VALID_FREQUENCIES:
            raise FREDValidationError(
                f"frequency={frequency!r} is not valid. "
                f"Choose from: {sorted(_VALID_FREQUENCIES)}."
            )

    def _validate_aggregation(self, method: str) -> None:
        if method not in _VALID_AGGREGATIONS:
            raise FREDValidationError(
                f"aggregation_method={method!r} is not valid. "
                f"Choose from: {sorted(_VALID_AGGREGATIONS)}."
            )

    def _validate_units(self, units: str) -> None:
        if units not in _VALID_UNITS:
            raise FREDValidationError(
                f"units={units!r} is not valid. "
                f"Choose from: {sorted(_VALID_UNITS)}."
            )

    # ------------------------------------------------------------------
    # Private helpers — HTTP
    # ------------------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        series_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a GET request with retry and map HTTP/network errors to
        domain exceptions.

        Parameters
        ----------
        endpoint:
            Path relative to ``_BASE_URL`` (e.g. ``"/series/observations"``).
        params:
            Query parameters dict.  The API key is injected automatically.
        series_id:
            Optional series identifier forwarded to domain exceptions for
            context.

        Returns
        -------
        dict
            Parsed JSON response body.
        """
        all_params = {**(params or {}), "api_key": self._api_key}

        @retry(
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(self._max_retries),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )
        def _execute() -> httpx.Response:
            return self._http.get(endpoint, params=all_params)

        try:
            response: httpx.Response = _execute()
        except httpx.TimeoutException as exc:
            raise FREDTimeoutError(
                f"Request to '{endpoint}' timed out after {self._timeout}s.",
                series_id=series_id,
            ) from exc
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            raise FREDConnectionError(
                f"Network failure on '{endpoint}': {exc}",
                series_id=series_id,
            ) from exc

        return self._handle_response(response, endpoint=endpoint, series_id=series_id)

    def _handle_response(
        self,
        response: httpx.Response,
        *,
        endpoint: str,
        series_id: str | None,
    ) -> dict[str, Any]:
        """Map HTTP status codes to domain exceptions; return parsed JSON."""
        status = response.status_code

        if status == 200:
            payload: dict[str, Any] = response.json()

            # FRED embeds error information inside 200 responses in some
            # cases (e.g. bad series_id → {"error_code": 400, ...}).
            if "error_code" in payload:
                self._raise_from_payload(payload, series_id=series_id)

            return payload

        if status in (400, 404):
            # Try to parse FRED's error message from the body.
            try:
                msg = response.json().get("error_message", response.text)
            except Exception:
                msg = response.text
            raise FREDNotFoundError(
                f"HTTP {status} from '{endpoint}': {msg}",
                series_id=series_id,
            )

        if status in (401, 403):
            raise FREDAuthError(
                f"HTTP {status}: API key rejected by FRED. "
                "Check FRED_API_KEY in your .env file.",
                series_id=series_id,
            )

        if status == 429:
            raise FREDRateLimitError(
                "HTTP 429: FRED rate limit exceeded. Reduce request frequency.",
                series_id=series_id,
            )

        # Unexpected status — surface raw message.
        raise FREDError(
            f"Unexpected HTTP {status} from '{endpoint}': {response.text[:200]}",
            series_id=series_id,
        )

    @staticmethod
    def _raise_from_payload(
        payload: dict[str, Any],
        *,
        series_id: str | None,
    ) -> None:
        """Raise a domain exception from a FRED error embedded in a 200 body."""
        code: int = payload.get("error_code", 0)
        msg: str = payload.get("error_message", "Unknown FRED error.")

        if code in (400, 404):
            raise FREDNotFoundError(msg, series_id=series_id)
        if code in (401, 403):
            raise FREDAuthError(msg, series_id=series_id)
        if code == 429:
            raise FREDRateLimitError(msg, series_id=series_id)
        raise FREDError(f"FRED error {code}: {msg}", series_id=series_id)
