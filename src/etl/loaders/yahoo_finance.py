"""
yahoo_finance.py  —  Loader
==============================
Persists cleaned Yahoo Finance DataFrames into PostgreSQL via SQLAlchemy.

Strategy:  upsert (INSERT … ON CONFLICT DO UPDATE) so the loader is safe
to run repeatedly without creating duplicates.  All operations are batched.

No API calls. No data transformations. Pure DataFrame → database writes.

Public API
----------
``YahooFinanceLoader``
    .load_company(df)                    → LoadResult
    .load_prices(df)                     → LoadResult
    .load_bulk_prices(price_map)         → dict[str, LoadResult]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.database.database import Company, SessionLocal, StockPrice

log = logging.getLogger(__name__)

# Rows written per database round-trip.
_DEFAULT_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LoadResult:
    """Summary of a single load operation."""

    ticker: str
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.inserted + self.updated

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ===========================================================================
# Loader
# ===========================================================================


class YahooFinanceLoader:
    """
    Load layer for the Yahoo Finance ETL pipeline.

    Parameters
    ----------
    session:
        Optional SQLAlchemy :class:`Session`.  When not provided a new session
        is created from :data:`SessionLocal` for each operation and closed
        on completion.
    batch_size:
        Number of rows per ``INSERT`` batch.  Default: 500.
    """

    def __init__(
        self,
        session: Session | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._external_session = session is not None
        self._session: Session = session or SessionLocal()
        self._batch_size = max(1, batch_size)

    def close(self) -> None:
        """Close the session if it was created internally."""
        if not self._external_session:
            self._session.close()

    def __enter__(self) -> "YahooFinanceLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load_company(self, df: pd.DataFrame) -> LoadResult:
        """
        Upsert a company record into the ``companies`` table.

        Expected columns (from the transformer):
        ``ticker``, ``name``, ``exchange``, ``currency``,
        ``sector``, ``industry``, ``country``,
        ``market_cap``, ``shares_outstanding``, ``isin``, ``cusip``.

        Conflict resolution: ``(ticker, exchange)`` unique constraint.
        On conflict: update mutable snapshot fields and ``updated_at``.

        Parameters
        ----------
        df:
            Single-row (or multi-row) DataFrame from the transformer.

        Returns
        -------
        LoadResult
            Counts of inserted / updated rows.
        """
        if df is None or df.empty:
            log.warning("load_company received an empty DataFrame.")
            ticker = "UNKNOWN"
            return LoadResult(ticker=ticker, skipped=1)

        ticker = str(df["ticker"].iloc[0]) if "ticker" in df.columns else "UNKNOWN"
        result = LoadResult(ticker=ticker)

        log.info("Loading company record", extra={"ticker": ticker})

        rows = self._df_to_records(df)
        for batch in self._batches(rows):
            try:
                dialect_name = (
                    self._session.bind.dialect.name
                    if self._session.bind
                    else "postgresql"
                )
                if dialect_name == "sqlite":
                    stmt = sqlite_insert(Company).values(batch)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["ticker", "exchange"],
                        set_={
                            "name": stmt.excluded.name,
                            "sector": stmt.excluded.sector,
                            "industry": stmt.excluded.industry,
                            "country": stmt.excluded.country,
                            "currency": stmt.excluded.currency,
                            "market_cap": stmt.excluded.market_cap,
                            "shares_outstanding": stmt.excluded.shares_outstanding,
                            "isin": stmt.excluded.isin,
                            "cusip": stmt.excluded.cusip,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    )
                else:
                    stmt = pg_insert(Company).values(batch)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_company_ticker_exchange",
                        set_={
                            "name": stmt.excluded.name,
                            "sector": stmt.excluded.sector,
                            "industry": stmt.excluded.industry,
                            "country": stmt.excluded.country,
                            "currency": stmt.excluded.currency,
                            "market_cap": stmt.excluded.market_cap,
                            "shares_outstanding": stmt.excluded.shares_outstanding,
                            "isin": stmt.excluded.isin,
                            "cusip": stmt.excluded.cusip,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    )
                res = self._session.execute(stmt)
                self._session.commit()

                # rowcount reflects total processed rows with psycopg2.
                result.inserted += res.rowcount
            except Exception as exc:
                self._session.rollback()
                msg = f"Batch load failed for company {ticker!r}: {exc}"
                log.exception(msg)
                result.errors.append(msg)

        log.info(
            "Company load complete",
            extra={
                "ticker": ticker,
                "inserted": result.inserted,
                "errors": len(result.errors),
            },
        )
        return result

    def load_prices(self, df: pd.DataFrame) -> LoadResult:
        """
        Upsert daily OHLCV price rows into the ``stock_prices`` table.

        Pre-requisite: the :class:`Company` record for ``ticker`` must already
        exist in the ``companies`` table (call :meth:`load_company` first).

        Conflict resolution: ``(company_id, price_date)`` unique constraint.
        On conflict: update all OHLCV columns and ``updated_at``.

        Expected columns (from the transformer):
        ``ticker``, ``price_date``, ``open_price``, ``high_price``,
        ``low_price``, ``close_price``, ``volume``, ``data_source``.

        Parameters
        ----------
        df:
            Transformed price DataFrame for a single ticker.

        Returns
        -------
        LoadResult
            Counts of inserted / updated rows.
        """
        if df is None or df.empty:
            log.warning("load_prices received an empty DataFrame.")
            return LoadResult(ticker="UNKNOWN", skipped=1)

        ticker = str(df["ticker"].iloc[0]) if "ticker" in df.columns else "UNKNOWN"
        result = LoadResult(ticker=ticker)

        # Resolve company_id — the company must exist first.
        company_id = self._get_company_id(ticker)
        if company_id is None:
            msg = (
                f"Company '{ticker}' not found in companies table. "
                "Run load_company before load_prices."
            )
            log.error(msg)
            result.errors.append(msg)
            return result

        log.info(
            "Loading price rows",
            extra={"ticker": ticker, "company_id": company_id, "rows": len(df)},
        )

        records = self._df_to_records(df)

        for batch in self._batches(records):
            # Inject company_id and remove ticker (not a DB column on StockPrice)
            enriched = []
            for row in batch:
                r = {k: v for k, v in row.items() if k != "ticker"}
                r["company_id"] = company_id
                # Nullable optional columns default to None when absent
                r.setdefault("vwap", None)
                r.setdefault("transactions", None)
                r.setdefault("adj_open", None)
                r.setdefault("adj_high", None)
                r.setdefault("adj_low", None)
                r.setdefault("adj_close", None)
                r.setdefault("adj_volume", None)
                enriched.append(r)

            try:
                dialect_name = (
                    self._session.bind.dialect.name
                    if self._session.bind
                    else "postgresql"
                )
                if dialect_name == "sqlite":
                    stmt = sqlite_insert(StockPrice).values(enriched)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["company_id", "price_date"],
                        set_={
                            "open_price": stmt.excluded.open_price,
                            "high_price": stmt.excluded.high_price,
                            "low_price": stmt.excluded.low_price,
                            "close_price": stmt.excluded.close_price,
                            "volume": stmt.excluded.volume,
                            "data_source": stmt.excluded.data_source,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    )
                else:
                    stmt = pg_insert(StockPrice).values(enriched)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_stockprice_company_date",
                        set_={
                            "open_price": stmt.excluded.open_price,
                            "high_price": stmt.excluded.high_price,
                            "low_price": stmt.excluded.low_price,
                            "close_price": stmt.excluded.close_price,
                            "volume": stmt.excluded.volume,
                            "data_source": stmt.excluded.data_source,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    )
                res = self._session.execute(stmt)
                self._session.commit()
                result.inserted += res.rowcount
            except Exception as exc:
                self._session.rollback()
                msg = f"Price batch failed for ticker {ticker!r}: {exc}"
                log.exception(msg)
                result.errors.append(msg)

        log.info(
            "Price load complete",
            extra={
                "ticker": ticker,
                "inserted": result.inserted,
                "errors": len(result.errors),
            },
        )
        return result

    def load_bulk_prices(
        self,
        price_map: dict[str, pd.DataFrame],
    ) -> dict[str, LoadResult]:
        """
        Upsert price DataFrames for multiple tickers.

        Parameters
        ----------
        price_map:
            ``{ticker: transformed_price_df}`` as produced by the pipeline.

        Returns
        -------
        dict[str, LoadResult]
            Per-ticker load results.
        """
        if not price_map:
            return {}

        log.info(
            "Bulk price load started",
            extra={"tickers": list(price_map.keys())},
        )

        results: dict[str, LoadResult] = {}
        for ticker, df in price_map.items():
            results[ticker] = self.load_prices(df)

        total_inserted = sum(r.inserted for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        log.info(
            "Bulk price load complete",
            extra={"total_inserted": total_inserted, "total_errors": total_errors},
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_company_id(self, ticker: str) -> int | None:
        """Look up the surrogate PK for a ticker (any exchange)."""
        result = self._session.execute(
            text("SELECT id FROM companies WHERE ticker = :ticker LIMIT 1"),
            {"ticker": ticker.upper()},
        ).fetchone()
        return result[0] if result else None

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert a DataFrame to a list of plain dicts, replacing NaN with None."""
        return [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]

    def _batches(
        self,
        records: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Split a list of records into batches of size ``_batch_size``."""
        return [
            records[i : i + self._batch_size]
            for i in range(0, len(records), self._batch_size)
        ]
