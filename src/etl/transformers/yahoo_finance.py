"""
yahoo_finance.py  —  Transformer
==================================
Cleans and normalises raw Yahoo Finance DataFrames into structures that map
directly to the ``Company`` and ``StockPrice`` ORM models.

No API calls. No database I/O. Pure DataFrame → DataFrame transformations.

Public API
----------
``YahooFinanceTransformer``
    .transform_prices(raw_df, ticker)           → DataFrame
    .transform_company_info(raw_df, ticker)     → DataFrame
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

# Yahoo Finance OHLCV column names → StockPrice ORM field names
_PRICE_COLUMN_MAP: dict[str, str] = {
    "Open": "open_price",
    "High": "high_price",
    "Low": "low_price",
    "Close": "close_price",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}

# Yahoo Finance info keys → Company ORM field names
_INFO_COLUMN_MAP: dict[str, str] = {
    "longName": "name",
    "exchange": "exchange",
    "currency": "currency",
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "marketCap": "market_cap",
    "sharesOutstanding": "shares_outstanding",
    "isin": "isin",
    "cusip": "cusip",
}

# Required price columns — rows missing any are dropped.
_REQUIRED_PRICE_COLS: frozenset[str] = frozenset(
    {"open_price", "high_price", "low_price", "close_price", "volume"}
)

# Exchange normalisation: Yahoo suffix → canonical exchange name
_EXCHANGE_SUFFIX_MAP: dict[str, str] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "AMEX",
    "LSE": "LSE",
    "TSX": "TSX",
    "AMS": "AMS",
    "FRA": "XETRA",
    "TYO": "TSE",
    "HKG": "HKEX",
    "SHH": "SSE",
    "SHZ": "SZSE",
    "BSE": "BSE",
    "NSE": "NSE",
}


# ===========================================================================
# Transformer
# ===========================================================================


class YahooFinanceTransformer:
    """
    Stateless transformer for Yahoo Finance raw DataFrames.

    All methods are pure functions wrapped in a class for testability and
    future extensibility (e.g. pluggable cleaning strategies).
    """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def transform_prices(
        self,
        raw_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Clean and normalise a raw OHLCV DataFrame from Yahoo Finance.

        Steps applied (in order):
        1. Reset index → ``price_date`` column
        2. Rename columns to ORM field names
        3. Cast numeric types
        4. Drop rows with missing required OHLCV fields
        5. Drop rows where close ≤ 0 (bad data from provider)
        6. Enforce high ≥ low (swap if inverted)
        7. Clip negative volumes to zero
        8. Add ``ticker`` and ``data_source`` columns
        9. Deduplicate on ``(ticker, price_date)``

        Parameters
        ----------
        raw_df:
            Output of :meth:`YahooFinanceExtractor.extract_prices`.
        ticker:
            Ticker symbol — embedded as a column for downstream use.

        Returns
        -------
        pandas.DataFrame
            Columns: ``ticker``, ``price_date``, ``open_price``,
            ``high_price``, ``low_price``, ``close_price``, ``volume``,
            ``data_source``.
            Index: default RangeIndex.
        """
        if raw_df.empty:
            log.warning("Empty price DataFrame received", extra={"ticker": ticker})
            return pd.DataFrame()

        log.info(
            "Transforming prices",
            extra={"ticker": ticker, "input_rows": len(raw_df)},
        )

        df = raw_df.copy()

        # 1. Flatten index → price_date column
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "price_date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "price_date"})

        df["price_date"] = pd.to_datetime(df["price_date"]).dt.date

        # 2. Rename to ORM field names; keep only known columns
        df = df.rename(columns=_PRICE_COLUMN_MAP)
        keep_cols = ["price_date"] + [
            c for c in _PRICE_COLUMN_MAP.values() if c in df.columns
        ]
        df = df[keep_cols]

        # 3. Cast numeric columns
        numeric_cols = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "dividends",
            "stock_splits",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 4. Drop rows missing any required OHLCV field
        required_present = list(_REQUIRED_PRICE_COLS & set(df.columns))
        before = len(df)
        df = df.dropna(subset=required_present)
        dropped = before - len(df)
        if dropped:
            log.warning(
                "Dropped rows with missing OHLCV",
                extra={"ticker": ticker, "dropped": dropped},
            )

        # 5. Drop rows with non-positive close price
        before = len(df)
        df = df[df["close_price"] > 0]
        dropped = before - len(df)
        if dropped:
            log.warning(
                "Dropped rows with close_price <= 0",
                extra={"ticker": ticker, "dropped": dropped},
            )

        # 6. Enforce high >= low (provider sometimes inverts on ex-div days)
        mask = df["high_price"] < df["low_price"]
        if mask.any():
            df.loc[mask, ["high_price", "low_price"]] = df.loc[
                mask, ["low_price", "high_price"]
            ].values
            log.warning(
                "Swapped inverted high/low",
                extra={"ticker": ticker, "count": int(mask.sum())},
            )

        # 7. Clip negative volume to zero
        if "volume" in df.columns:
            neg_vol = (df["volume"] < 0).sum()
            if neg_vol:
                df["volume"] = df["volume"].clip(lower=0)
                log.warning(
                    "Clipped negative volume to 0",
                    extra={"ticker": ticker, "count": int(neg_vol)},
                )
            df["volume"] = df["volume"].astype("Int64")

        # 8. Add metadata columns
        df["ticker"] = ticker.upper().strip()
        df["data_source"] = "yahoo_finance"

        # 9. Deduplicate — keep last record per (ticker, price_date)
        before = len(df)
        df = df.drop_duplicates(subset=["ticker", "price_date"], keep="last")
        if len(df) < before:
            log.warning(
                "Removed duplicate (ticker, price_date) rows",
                extra={"ticker": ticker, "removed": before - len(df)},
            )

        df = df.reset_index(drop=True)

        log.info(
            "Price transformation complete",
            extra={"ticker": ticker, "output_rows": len(df)},
        )
        return df

    def transform_company_info(
        self,
        raw_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """
        Clean and normalise a raw company info DataFrame from Yahoo Finance.

        Steps applied (in order):
        1. Select and rename fields to ORM column names
        2. Add ``ticker`` column
        3. Normalise exchange code to canonical name
        4. Strip whitespace from all string columns
        5. Coerce numeric types (market_cap, shares_outstanding)
        6. Fill missing non-nullable fields with safe defaults

        Parameters
        ----------
        raw_df:
            Output of :meth:`YahooFinanceExtractor.extract_company_info`.
        ticker:
            Ticker symbol — used as fallback identifier.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame with columns matching :class:`Company` ORM
            fields: ``ticker``, ``name``, ``exchange``, ``currency``,
            ``sector``, ``industry``, ``country``, ``market_cap``,
            ``shares_outstanding``, ``isin``, ``cusip``.
        """
        if raw_df.empty:
            log.warning(
                "Empty company info DataFrame received", extra={"ticker": ticker}
            )
            return pd.DataFrame()

        log.info("Transforming company info", extra={"ticker": ticker})

        df = raw_df.copy()

        # 1. Select known columns only, rename to ORM names
        rename_map = {
            src: dst for src, dst in _INFO_COLUMN_MAP.items() if src in df.columns
        }
        df = df.rename(columns=rename_map)

        orm_cols = list(_INFO_COLUMN_MAP.values()) + ["ticker"]
        present = [c for c in orm_cols if c in df.columns]
        df = df[present]

        # 2. Ensure ticker column
        df["ticker"] = ticker.upper().strip()

        # 3. Normalise exchange
        if "exchange" in df.columns:
            df["exchange"] = df["exchange"].apply(self._normalise_exchange)
        else:
            df["exchange"] = "UNKNOWN"

        # 4. Strip strings
        str_cols = ["name", "exchange", "sector", "industry", "country", "currency"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: v.strip() if isinstance(v, str) else v
                )

        # 5. Coerce numerics
        for col in ("market_cap", "shares_outstanding"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 6. Fill required non-nullable fields with safe defaults
        if "name" not in df.columns or df["name"].isna().all():
            df["name"] = ticker.upper()
        if "currency" not in df.columns or df["currency"].isna().all():
            df["currency"] = "USD"
        if "exchange" not in df.columns or df["exchange"].isna().all():
            df["exchange"] = "UNKNOWN"

        df = df.reset_index(drop=True)

        log.info(
            "Company info transformation complete",
            extra={"ticker": ticker, "columns": list(df.columns)},
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_exchange(raw: Any) -> str:
        """Map a Yahoo Finance exchange code to a canonical name."""
        if not isinstance(raw, str) or not raw.strip():
            return "UNKNOWN"
        cleaned = raw.strip().upper()
        return _EXCHANGE_SUFFIX_MAP.get(cleaned, cleaned)
