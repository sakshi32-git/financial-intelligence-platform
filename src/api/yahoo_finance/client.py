"""
client.py
=========
Yahoo Finance API client for the Financial Intelligence Platform.

Wraps the ``yfinance`` library with:

* Structured logging on every call
* Automatic retries with exponential back-off (via ``tenacity``)
* Granular error classification into domain exceptions
* Strict argument validation before any network I/O
* Every public method returns a ``pandas.DataFrame``

Public API
----------
``YahooFinanceClient``
    .get_historical_prices(ticker, start, end, interval)  → DataFrame
    .get_company_info(ticker)                             → DataFrame
    .get_income_statement(ticker, quarterly)              → DataFrame
    .get_balance_sheet(ticker, quarterly)                 → DataFrame
    .get_cash_flow(ticker, quarterly)                     → DataFrame

Usage
-----
    from src.api.yahoo_finance.client import YahooFinanceClient

    client = YahooFinanceClient()
    df = client.get_historical_prices("AAPL", start="2023-01-01", end="2024-01-01")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

import pandas as pd
import yfinance as yf
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.api.yahoo_finance.exceptions import (
    YahooFinanceConnectionError,
    YahooFinanceError,
    YahooFinanceNotFoundError,
    YahooFinanceRateLimitError,
    YahooFinanceTimeoutError,
    YahooFinanceValidationError,
)

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Interval = Literal[
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
]

_VALID_INTERVALS: frozenset[str] = frozenset(Interval.__args__)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Internal retry helpers
# ---------------------------------------------------------------------------

_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

_RETRY_POLICY = dict(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=False,  # We map to domain exceptions in the wrapper.
)


# ===========================================================================
# Client
# ===========================================================================


class YahooFinanceClient:
    """
    Thin, opinionated wrapper around ``yfinance`` with retry and error handling.

    Parameters
    ----------
    request_timeout:
        Per-request timeout in seconds passed to yfinance.
        Default: 10.
    max_retries:
        Maximum number of attempts before raising.
        Default: 4 (1 initial + 3 retries).
    """

    def __init__(
        self,
        request_timeout: int = 10,
        max_retries: int = 4,
    ) -> None:
        if request_timeout < 1:
            raise YahooFinanceValidationError(
                f"request_timeout must be >= 1, got {request_timeout}"
            )
        if max_retries < 1:
            raise YahooFinanceValidationError(
                f"max_retries must be >= 1, got {max_retries}"
            )

        self._timeout = request_timeout
        self._max_retries = max_retries

        log.info(
            "YahooFinanceClient initialised",
            extra={"timeout": self._timeout, "max_retries": self._max_retries},
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_historical_prices(
        self,
        ticker: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: Interval = "1d",
        auto_adjust: bool = True,
        back_adjust: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch daily (or intraday) OHLCV history for a single ticker.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol, e.g. ``"AAPL"`` or ``"BTC-USD"``.
        start:
            Inclusive start date (``"YYYY-MM-DD"``, ``date``, or ``datetime``).
        end:
            Exclusive end date.
        interval:
            Data granularity — one of the ``Interval`` literals.
            Default: ``"1d"``.
        auto_adjust:
            Adjust OHLCV for splits and dividends.  Default: ``True``.
        back_adjust:
            Back-adjust prices to remove forward price jumps.  Default: ``False``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
            (plus ``Dividends``, ``Stock Splits`` when auto_adjust is True).
            Index: ``DatetimeIndex`` named ``"Date"``.

        Raises
        ------
        YahooFinanceValidationError
            Invalid ticker, interval, or date range.
        YahooFinanceNotFoundError
            Ticker exists but returned no data for the requested range.
        YahooFinanceConnectionError
            Network failure persisted beyond all retry attempts.
        YahooFinanceTimeoutError
            All attempts timed out.
        """
        ticker = self._validate_ticker(ticker)
        self._validate_interval(interval)
        self._validate_date_range(start, end, ticker)

        log.info(
            "Fetching historical prices",
            extra={
                "ticker": ticker,
                "start": str(start),
                "end": str(end),
                "interval": interval,
                "auto_adjust": auto_adjust,
            },
        )

        df = self._download_with_retry(
            ticker=ticker,
            start=str(start),
            end=str(end),
            interval=interval,
            auto_adjust=auto_adjust,
            back_adjust=back_adjust,
        )

        if df.empty:
            raise YahooFinanceNotFoundError(
                f"No price data returned for interval={interval!r} "
                f"between {start} and {end}.",
                ticker=ticker,
            )

        df.index.name = "Date"
        log.info(
            "Historical prices fetched",
            extra={"ticker": ticker, "rows": len(df), "columns": list(df.columns)},
        )
        return df

    def get_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Return a single-row DataFrame of company metadata and key statistics.

        Columns include (when available): ``longName``, ``sector``,
        ``industry``, ``country``, ``currency``, ``marketCap``,
        ``sharesOutstanding``, ``trailingPE``, ``forwardPE``,
        ``dividendYield``, ``beta``, ``52WeekHigh``, ``52WeekLow``.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.

        Returns
        -------
        pandas.DataFrame
            One row per ticker; columns are info keys from the Yahoo Finance API.

        Raises
        ------
        YahooFinanceNotFoundError
            Ticker not found or info payload is empty.
        YahooFinanceConnectionError / YahooFinanceTimeoutError
            Network failures.
        """
        ticker = self._validate_ticker(ticker)
        log.info("Fetching company info", extra={"ticker": ticker})

        yf_ticker = self._get_yf_ticker(ticker)
        try:
            raw: dict = yf_ticker.info  # .info is a property in yfinance >=0.2.x
        except Exception as exc:
            raise YahooFinanceError(
                f"Unexpected error while fetching 'info': {exc}",
                ticker=ticker,
            ) from exc

        if not raw or raw.get("quoteType") is None:
            raise YahooFinanceNotFoundError(
                "Ticker not found or returned an empty info payload.",
                ticker=ticker,
            )

        # Normalise to a tidy single-row DataFrame.
        df = pd.DataFrame([raw])
        df.insert(0, "ticker", ticker)

        log.info(
            "Company info fetched",
            extra={"ticker": ticker, "fields": len(df.columns)},
        )
        return df

    def get_income_statement(
        self, ticker: str, quarterly: bool = False
    ) -> pd.DataFrame:
        """
        Return the income statement for a ticker.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
        quarterly:
            If ``True`` return quarterly figures; otherwise annual.

        Returns
        -------
        pandas.DataFrame
            Index: financial line items.  Columns: reporting periods.

        Raises
        ------
        YahooFinanceNotFoundError
            No financials data available.
        """
        ticker = self._validate_ticker(ticker)
        label = "quarterly_income_stmt" if quarterly else "income_stmt"
        log.info(
            "Fetching income statement",
            extra={"ticker": ticker, "quarterly": quarterly},
        )
        return self._fetch_financials(ticker, label)

    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
        """
        Return the balance sheet for a ticker.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
        quarterly:
            If ``True`` return quarterly figures; otherwise annual.

        Returns
        -------
        pandas.DataFrame
            Index: balance sheet line items.  Columns: reporting periods.

        Raises
        ------
        YahooFinanceNotFoundError
            No balance sheet data available.
        """
        ticker = self._validate_ticker(ticker)
        label = "quarterly_balance_sheet" if quarterly else "balance_sheet"
        log.info(
            "Fetching balance sheet",
            extra={"ticker": ticker, "quarterly": quarterly},
        )
        return self._fetch_financials(ticker, label)

    def get_cash_flow(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:
        """
        Return the cash-flow statement for a ticker.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
        quarterly:
            If ``True`` return quarterly figures; otherwise annual.

        Returns
        -------
        pandas.DataFrame
            Index: cash-flow line items.  Columns: reporting periods.

        Raises
        ------
        YahooFinanceNotFoundError
            No cash-flow data available.
        """
        ticker = self._validate_ticker(ticker)
        label = "quarterly_cash_flow" if quarterly else "cash_flow"
        log.info(
            "Fetching cash flow",
            extra={"ticker": ticker, "quarterly": quarterly},
        )
        return self._fetch_financials(ticker, label)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_ticker(self, ticker: str) -> str:
        """Return the stripped, uppercased ticker or raise ValidationError."""
        if not isinstance(ticker, str) or not ticker.strip():
            raise YahooFinanceValidationError(
                f"ticker must be a non-empty string, got {ticker!r}."
            )
        return ticker.strip().upper()

    def _validate_interval(self, interval: str) -> None:
        if interval not in _VALID_INTERVALS:
            raise YahooFinanceValidationError(
                f"interval={interval!r} is not valid. "
                f"Choose from: {sorted(_VALID_INTERVALS)}."
            )

    def _validate_date_range(
        self,
        start: str | date | datetime,
        end: str | date | datetime,
        ticker: str,
    ) -> None:
        try:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
        except Exception as exc:
            raise YahooFinanceValidationError(
                f"Cannot parse date range start={start!r}, end={end!r}: {exc}",
                ticker=ticker,
            ) from exc

        if start_dt >= end_dt:
            raise YahooFinanceValidationError(
                f"start ({start}) must be strictly before end ({end}).",
                ticker=ticker,
            )

    def _get_yf_ticker(self, ticker: str) -> yf.Ticker:
        return yf.Ticker(ticker)

    @retry(**_RETRY_POLICY)  # type: ignore[arg-type]
    def _download_with_retry(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
        back_adjust: bool,
    ) -> pd.DataFrame:
        """
        Call ``yfinance.download`` inside the tenacity retry loop.

        Tenacity retries only on :data:`_RETRYABLE_EXCEPTIONS`; all other
        exceptions propagate immediately.
        """
        try:
            df: pd.DataFrame = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                back_adjust=back_adjust,
                progress=False,
                timeout=self._timeout,
                # Suppress the multi-level column header for a single ticker.
                group_by="ticker",
                threads=False,
            )
            # yfinance returns a multi-level column for multi-ticker downloads;
            # flatten if needed for the single-ticker case.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(1)
            return df
        except TimeoutError as exc:
            log.warning("Request timed out", extra={"ticker": ticker})
            raise TimeoutError(str(exc)) from exc
        except (ConnectionError, OSError) as exc:
            log.warning("Connection error", extra={"ticker": ticker, "error": str(exc)})
            raise

    def _safe_call(self, callable_: object, *, ticker: str, label: str) -> object:
        """
        Call a zero-argument callable, mapping exceptions to domain errors.

        Used for ``yf.Ticker`` property accessors (``info``, ``income_stmt``
        etc.) which may raise on network failure.
        """
        _retry_dec = retry(
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(self._max_retries),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )

        @_retry_dec
        def _call():
            return callable_()  # type: ignore[operator]

        try:
            return _call()
        except RetryError as exc:
            raise YahooFinanceConnectionError(
                f"All {self._max_retries} attempts failed for '{label}'.",
                ticker=ticker,
            ) from exc
        except TimeoutError as exc:
            raise YahooFinanceTimeoutError(
                f"Request for '{label}' timed out after {self._timeout}s.",
                ticker=ticker,
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise YahooFinanceConnectionError(
                f"Network error while fetching '{label}': {exc}",
                ticker=ticker,
            ) from exc
        except Exception as exc:
            raise YahooFinanceError(
                f"Unexpected error while fetching '{label}': {exc}",
                ticker=ticker,
            ) from exc

    def _fetch_financials(self, ticker: str, attribute: str) -> pd.DataFrame:
        """
        Retrieve a financial statement DataFrame from a ``yf.Ticker`` attribute.

        Parameters
        ----------
        ticker:
            Already validated/normalised ticker symbol.
        attribute:
            ``yf.Ticker`` attribute name (e.g. ``"income_stmt"``).
        """
        yf_ticker = self._get_yf_ticker(ticker)
        try:
            df = getattr(
                yf_ticker, attribute
            )  # these are properties in yfinance >=0.2.x
        except Exception as exc:
            raise YahooFinanceError(
                f"Unexpected error while fetching '{attribute}': {exc}",
                ticker=ticker,
            ) from exc

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise YahooFinanceNotFoundError(
                f"No data returned for '{attribute}'. "
                "This ticker may not have public financial filings.",
                ticker=ticker,
            )

        log.info(
            "Financials fetched",
            extra={
                "ticker": ticker,
                "attribute": attribute,
                "rows": len(df),
                "columns": len(df.columns),
            },
        )
        return df
