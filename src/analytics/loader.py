"""
loader.py
=========
Read price data from PostgreSQL into pandas DataFrames ready for the
analytics functions.

No API calls. No transformations beyond column casting.

Public API
----------
``AnalyticsLoader``
    .load_prices(ticker, start, end)       → DataFrame
    .load_prices_multi(tickers, start, end) → DataFrame
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

from src.database.database import SessionLocal

log = logging.getLogger(__name__)

# SQL that fetches the price series for one or more tickers.
_PRICE_QUERY = text(
    """
    SELECT
        c.ticker,
        sp.price_date,
        sp.open_price,
        sp.high_price,
        sp.low_price,
        sp.close_price,
        sp.volume,
        sp.adj_close
    FROM stock_prices sp
    JOIN companies c ON c.id = sp.company_id
    WHERE c.ticker = ANY(:tickers)
      AND (:start IS NULL OR sp.price_date >= :start::date)
      AND (:end   IS NULL OR sp.price_date <= :end::date)
    ORDER BY c.ticker, sp.price_date ASC
    """
)


class AnalyticsLoader:
    """
    Load price data from PostgreSQL for the analytics engine.

    Parameters
    ----------
    session:
        Optional SQLAlchemy session.  A new session is opened from
        :data:`SessionLocal` when not provided.
    """

    def __init__(self, session=None) -> None:
        self._external_session = session is not None
        self._session = session or SessionLocal()

    def close(self) -> None:
        if not self._external_session:
            self._session.close()

    def __enter__(self) -> "AnalyticsLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def load_prices(
        self,
        ticker: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load the price series for a single ticker.

        Parameters
        ----------
        ticker:
            Ticker symbol (case-insensitive).
        start:
            Inclusive start date.  ``None`` = earliest available.
        end:
            Inclusive end date.  ``None`` = latest available.

        Returns
        -------
        pandas.DataFrame
            Columns: ``ticker``, ``price_date``, ``open_price``,
            ``high_price``, ``low_price``, ``close_price``, ``volume``,
            ``adj_close``.
            Sorted by ``price_date`` ascending.
            Numeric columns cast to ``float64``.

        Raises
        ------
        ValueError
            Ticker not found or returned zero rows.
        """
        log.info(
            "Loading prices from DB",
            extra={"ticker": ticker, "start": str(start), "end": str(end)},
        )

        df = self._execute_price_query(
            tickers=[ticker.upper()],
            start=str(start) if start else None,
            end=str(end) if end else None,
        )

        if df.empty:
            raise ValueError(
                f"No price data found for ticker={ticker!r} "
                f"between start={start} and end={end}. "
                "Run the Yahoo Finance ETL pipeline first."
            )

        log.info(
            "Prices loaded",
            extra={"ticker": ticker, "rows": len(df)},
        )
        return df

    def load_prices_multi(
        self,
        tickers: list[str],
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load price series for multiple tickers in a single query.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        start:
            Inclusive start date.  ``None`` = earliest available.
        end:
            Inclusive end date.  ``None`` = latest available.

        Returns
        -------
        pandas.DataFrame
            Same schema as :meth:`load_prices` but containing rows for
            all tickers.  Sorted by ``(ticker, price_date)`` ascending.

        Raises
        ------
        ValueError
            No data found for any of the requested tickers.
        """
        if not tickers:
            raise ValueError("tickers must be a non-empty list.")

        upper_tickers = [t.upper() for t in tickers]
        log.info(
            "Loading multi-ticker prices from DB",
            extra={"tickers": upper_tickers, "start": str(start), "end": str(end)},
        )

        df = self._execute_price_query(
            tickers=upper_tickers,
            start=str(start) if start else None,
            end=str(end) if end else None,
        )

        if df.empty:
            raise ValueError(
                f"No price data found for tickers={upper_tickers}. "
                "Run the Yahoo Finance ETL pipeline first."
            )

        missing = set(upper_tickers) - set(df["ticker"].unique())
        if missing:
            log.warning(
                "Some tickers returned no data",
                extra={"missing": sorted(missing)},
            )

        log.info(
            "Multi-ticker prices loaded",
            extra={"rows": len(df), "tickers_found": df["ticker"].nunique()},
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_price_query(
        self,
        tickers: list[str],
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        rows = self._session.execute(
            _PRICE_QUERY,
            {"tickers": tickers, "start": start, "end": end},
        ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "ticker", "price_date", "open_price",
            "high_price", "low_price", "close_price",
            "volume", "adj_close",
        ])

        df["price_date"] = pd.to_datetime(df["price_date"])

        for col in ("open_price", "high_price", "low_price",
                    "close_price", "adj_close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        df = df.sort_values(["ticker", "price_date"]).reset_index(drop=True)
        return df

    def load_commodity_prices(
        self,
        symbol: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load the price series for a single commodity.

        Parameters
        ----------
        symbol:
            Commodity symbol (case-insensitive).
        start:
            Inclusive start date.  ``None`` = earliest available.
        end:
            Inclusive end date.  ``None`` = latest available.

        Returns
        -------
        pandas.DataFrame
            Sorted by ``price_date`` ascending.
            Numeric columns cast to ``float64``.
        """
        log.info(
            "Loading commodity prices from DB",
            extra={"symbol": symbol, "start": str(start), "end": str(end)},
        )
        
        query = text(
            """
            SELECT
                symbol as ticker,
                price_date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM commodities
            WHERE symbol = :symbol
              AND (:start IS NULL OR price_date >= :start::date)
              AND (:end   IS NULL OR price_date <= :end::date)
            ORDER BY price_date ASC
            """
        )
        
        rows = self._session.execute(
            query,
            {"symbol": symbol.upper(), "start": str(start) if start else None, "end": str(end) if end else None},
        ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "ticker", "price_date", "open_price",
            "high_price", "low_price", "close_price",
            "volume"
        ])

        df["price_date"] = pd.to_datetime(df["price_date"])
        for col in ("open_price", "high_price", "low_price", "close_price"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        return df
