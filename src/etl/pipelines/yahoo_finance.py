"""
yahoo_finance.py  —  Pipeline
================================
Orchestrates the full Yahoo Finance ETL cycle:

    Extract  →  Transform  →  Load  →  Report

Wires together:
* :class:`YahooFinanceExtractor`
* :class:`YahooFinanceTransformer`
* :class:`YahooFinanceLoader`

No business logic. No analytics.

Public API
----------
``YahooFinancePipeline``
    .run_company(ticker)                          → PipelineResult
    .run_prices(ticker, start, end, interval)     → PipelineResult
    .run_full(ticker, start, end, interval)       → PipelineResult
    .run_bulk(tickers, start, end, interval)      → list[PipelineResult]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from src.etl.extractors.yahoo_finance import YahooFinanceExtractor
from src.etl.loaders.yahoo_finance import LoadResult, YahooFinanceLoader
from src.etl.transformers.yahoo_finance import YahooFinanceTransformer

log = logging.getLogger(__name__)

Interval = Literal[
    "1m", "2m", "5m", "15m", "30m",
    "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """End-to-end result for a single ticker pipeline run."""

    ticker: str
    company_result: LoadResult | None = None
    price_result: LoadResult | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True only when every attempted stage completed without errors."""
        results = [r for r in (self.company_result, self.price_result) if r is not None]
        return bool(results) and all(r.success for r in results) and not self.errors

    @property
    def total_rows_loaded(self) -> int:
        total = 0
        if self.company_result:
            total += self.company_result.total
        if self.price_result:
            total += self.price_result.total
        return total


# ===========================================================================
# Pipeline
# ===========================================================================


class YahooFinancePipeline:
    """
    End-to-end ETL orchestrator for Yahoo Finance data.

    Parameters
    ----------
    extractor:
        Pre-configured extractor.  Creates a default instance when omitted.
    transformer:
        Pre-configured transformer.  Creates a default instance when omitted.
    loader:
        Pre-configured loader.  Creates a default instance when omitted.
    skip_on_error:
        When ``True``, bulk runs skip failed tickers and log a warning instead
        of raising.  Default: ``True``.
    """

    def __init__(
        self,
        extractor: YahooFinanceExtractor | None = None,
        transformer: YahooFinanceTransformer | None = None,
        loader: YahooFinanceLoader | None = None,
        skip_on_error: bool = True,
    ) -> None:
        self._extractor = extractor or YahooFinanceExtractor(
            skip_on_error=skip_on_error
        )
        self._transformer = transformer or YahooFinanceTransformer()
        self._loader = loader or YahooFinanceLoader()
        self._skip_on_error = skip_on_error

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run_company(self, ticker: str) -> PipelineResult:
        """
        Extract → Transform → Load company metadata for a single ticker.

        Steps
        -----
        1. Extract raw company info from Yahoo Finance API.
        2. Transform to ORM-compatible shape.
        3. Upsert into the ``companies`` table.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol, e.g. ``"AAPL"``.

        Returns
        -------
        PipelineResult
            Contains :attr:`company_result` with inserted/updated counts.
        """
        result = PipelineResult(ticker=ticker)
        log.info("Pipeline: run_company started", extra={"ticker": ticker})

        try:
            # Extract
            raw_df = self._extractor.extract_company_info(ticker)

            # Transform
            clean_df = self._transformer.transform_company_info(raw_df, ticker)

            if clean_df.empty:
                msg = f"Transformer returned empty DataFrame for company {ticker!r}."
                log.warning(msg)
                result.errors.append(msg)
                return result

            # Load
            result.company_result = self._loader.load_company(clean_df)

        except Exception as exc:
            msg = f"run_company failed for {ticker!r}: {exc}"
            log.exception(msg)
            result.errors.append(msg)
            if not self._skip_on_error:
                raise

        self._log_result(result, stage="run_company")
        return result

    def run_prices(
        self,
        ticker: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: Interval = "1d",
        auto_adjust: bool = True,
    ) -> PipelineResult:
        """
        Extract → Transform → Load daily OHLCV prices for a single ticker.

        Pre-requisite: the company record must already exist in the database
        (run :meth:`run_company` first, or call :meth:`run_full`).

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
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
        PipelineResult
            Contains :attr:`price_result` with inserted/updated counts.
        """
        result = PipelineResult(ticker=ticker)
        log.info(
            "Pipeline: run_prices started",
            extra={"ticker": ticker, "start": str(start), "end": str(end)},
        )

        try:
            # Extract
            raw_df = self._extractor.extract_prices(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )

            # Transform
            clean_df = self._transformer.transform_prices(raw_df, ticker)

            if clean_df.empty:
                msg = f"Transformer returned empty price DataFrame for {ticker!r}."
                log.warning(msg)
                result.errors.append(msg)
                return result

            # Load
            result.price_result = self._loader.load_prices(clean_df)

        except Exception as exc:
            msg = f"run_prices failed for {ticker!r}: {exc}"
            log.exception(msg)
            result.errors.append(msg)
            if not self._skip_on_error:
                raise

        self._log_result(result, stage="run_prices")
        return result

    def run_full(
        self,
        ticker: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: Interval = "1d",
        auto_adjust: bool = True,
    ) -> PipelineResult:
        """
        Run the complete ETL for a single ticker:
        company metadata first, then price history.

        This is the primary entry point for single-ticker ingestion.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
        start:
            Inclusive price history start date.
        end:
            Exclusive price history end date.
        interval:
            Price granularity.  Default: ``"1d"``.
        auto_adjust:
            Apply split/dividend adjustments.  Default: ``True``.

        Returns
        -------
        PipelineResult
            Combined result with both :attr:`company_result` and
            :attr:`price_result` populated.
        """
        log.info(
            "Pipeline: run_full started",
            extra={"ticker": ticker, "start": str(start), "end": str(end)},
        )

        # Stage 1 — company
        company_run = self.run_company(ticker)
        if not company_run.success and not self._skip_on_error:
            return company_run

        # Stage 2 — prices (proceeds even if company had non-fatal errors)
        price_run = self.run_prices(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )

        # Merge results
        result = PipelineResult(
            ticker=ticker,
            company_result=company_run.company_result,
            price_result=price_run.price_result,
            errors=company_run.errors + price_run.errors,
        )

        self._log_result(result, stage="run_full")
        return result

    def run_bulk(
        self,
        tickers: list[str],
        start: str | date | datetime,
        end: str | date | datetime,
        interval: Interval = "1d",
        auto_adjust: bool = True,
    ) -> list[PipelineResult]:
        """
        Run :meth:`run_full` for a list of tickers sequentially.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        start:
            Inclusive price history start date.
        end:
            Exclusive price history end date.
        interval:
            Price granularity.  Default: ``"1d"``.
        auto_adjust:
            Apply split/dividend adjustments.  Default: ``True``.

        Returns
        -------
        list[PipelineResult]
            One result per ticker, in input order.
        """
        if not tickers:
            raise ValueError("tickers must be a non-empty list.")

        log.info(
            "Pipeline: run_bulk started",
            extra={"count": len(tickers), "tickers": tickers},
        )

        results: list[PipelineResult] = []
        for ticker in tickers:
            res = self.run_full(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )
            results.append(res)

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        total_rows = sum(r.total_rows_loaded for r in results)

        log.info(
            "Pipeline: run_bulk complete",
            extra={
                "succeeded": succeeded,
                "failed": failed,
                "total_rows_loaded": total_rows,
            },
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_result(result: PipelineResult, stage: str) -> None:
        level = logging.INFO if result.success else logging.WARNING
        log.log(
            level,
            "Pipeline stage complete",
            extra={
                "stage": stage,
                "ticker": result.ticker,
                "success": result.success,
                "total_rows_loaded": result.total_rows_loaded,
                "errors": result.errors,
            },
        )
