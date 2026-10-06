"""
yahoo_finance.py  —  Extractor
================================
Extracts raw data from Yahoo Finance via :class:`YahooFinanceClient` and
returns unmodified ``pandas.DataFrame`` objects ready for the transform stage.

No cleaning, no database I/O, no business logic.

Public API
----------
``YahooFinanceExtractor``
    .extract_prices(ticker, start, end, interval)   → DataFrame
    .extract_company_info(ticker)                   → DataFrame
    .extract_bulk_prices(tickers, start, end, ...)  → dict[str, DataFrame]
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

import pandas as pd

from src.api.yahoo_finance.client import YahooFinanceClient
from src.api.yahoo_finance.exceptions import (
    YahooFinanceError,
    YahooFinanceNotFoundError,
)

log = logging.getLogger(__name__)

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


class YahooFinanceExtractor:
    """
    Extraction layer for the Yahoo Finance ETL pipeline.

    Parameters
    ----------
    client:
        Optional pre-configured :class:`YahooFinanceClient`.  A new instance
        with default settings is created when not provided.
    skip_on_error:
        When ``True``, bulk operations skip failed tickers and log a warning
        instead of raising.  Default: ``True``.
    """

    def __init__(
        self,
        client: YahooFinanceClient | None = None,
        skip_on_error: bool = True,
    ) -> None:
        self._client = client or YahooFinanceClient()
        self._skip_on_error = skip_on_error

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract_prices(
        self,
        ticker: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: Interval = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Extract OHLCV price history for a single ticker.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol, e.g. ``"AAPL"``.
        start:
            Inclusive start date.
        end:
            Exclusive end date.
        interval:
            Price granularity.  Default: ``"1d"`` (daily).
        auto_adjust:
            Apply split/dividend adjustments.  Default: ``True``.

        Returns
        -------
        pandas.DataFrame
            Raw DataFrame from Yahoo Finance — no transformations applied.
            Columns: Open, High, Low, Close, Volume (+ Dividends, Stock Splits).
            Index: DatetimeIndex named ``"Date"``.

        Raises
        ------
        YahooFinanceError
            Any client-level error is propagated unchanged.
        """
        log.info(
            "Extracting prices",
            extra={"ticker": ticker, "start": str(start), "end": str(end)},
        )
        df = self._client.get_historical_prices(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )
        log.info(
            "Prices extracted",
            extra={"ticker": ticker, "rows": len(df)},
        )
        return df

    def extract_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Extract company metadata for a single ticker.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame from Yahoo Finance info payload.

        Raises
        ------
        YahooFinanceError
            Any client-level error is propagated unchanged.
        """
        log.info("Extracting company info", extra={"ticker": ticker})
        df = self._client.get_company_info(ticker)
        log.info(
            "Company info extracted",
            extra={"ticker": ticker, "fields": len(df.columns)},
        )
        return df

    def extract_bulk_prices(
        self,
        tickers: list[str],
        start: str | date | datetime,
        end: str | date | datetime,
        interval: Interval = "1d",
        auto_adjust: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Extract OHLCV price history for a list of tickers.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        start:
            Inclusive start date.
        end:
            Exclusive end date.
        interval:
            Price granularity.  Default: ``"1d"``.
        auto_adjust:
            Apply split/dividend adjustments.  Default: ``True``.

        Returns
        -------
        dict[str, DataFrame]
            Mapping of ``ticker → raw price DataFrame``.
            Tickers that failed are omitted when ``skip_on_error=True``
            or raise when ``skip_on_error=False``.
        """
        if not tickers:
            raise ValueError("tickers must be a non-empty list.")

        log.info(
            "Bulk price extraction started",
            extra={"count": len(tickers), "start": str(start), "end": str(end)},
        )

        results: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        for ticker in tickers:
            try:
                results[ticker] = self.extract_prices(
                    ticker=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                )
            except YahooFinanceNotFoundError:
                log.warning(
                    "Ticker not found — skipped",
                    extra={"ticker": ticker},
                )
                failed.append(ticker)
            except YahooFinanceError as exc:
                if self._skip_on_error:
                    log.warning(
                        "Extraction failed — skipped",
                        extra={"ticker": ticker, "error": str(exc)},
                    )
                    failed.append(ticker)
                else:
                    raise

        log.info(
            "Bulk price extraction complete",
            extra={
                "success": len(results),
                "failed": len(failed),
                "failed_tickers": failed,
            },
        )
        return results

    def extract_bulk_company_info(
        self,
        tickers: list[str],
    ) -> dict[str, pd.DataFrame]:
        """
        Extract company metadata for a list of tickers.

        Returns
        -------
        dict[str, DataFrame]
            Mapping of ``ticker → info DataFrame``.
        """
        if not tickers:
            raise ValueError("tickers must be a non-empty list.")

        log.info(
            "Bulk company info extraction started",
            extra={"count": len(tickers)},
        )

        results: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        for ticker in tickers:
            try:
                results[ticker] = self.extract_company_info(ticker)
            except YahooFinanceError as exc:
                if self._skip_on_error:
                    log.warning(
                        "Company info extraction failed — skipped",
                        extra={"ticker": ticker, "error": str(exc)},
                    )
                    failed.append(ticker)
                else:
                    raise

        log.info(
            "Bulk company info extraction complete",
            extra={"success": len(results), "failed": len(failed)},
        )
        return results
