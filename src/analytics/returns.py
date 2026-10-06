"""
returns.py
==========
Daily and monthly return calculations for financial price series.

All functions are pure: DataFrame in → DataFrame out.
No database I/O. No visualisation.

Public API
----------
compute_daily_returns(df, price_col, method)   → DataFrame
compute_monthly_returns(df, price_col, method) → DataFrame
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

ReturnMethod = Literal["simple", "log"]


def compute_daily_returns(
    df: pd.DataFrame,
    price_col: str = "close_price",
    method: ReturnMethod = "simple",
    ticker_col: str | None = "ticker",
) -> pd.DataFrame:
    """
    Compute daily returns from a price series.

    Parameters
    ----------
    df:
        DataFrame with at minimum a date index or ``price_date`` column and
        a ``price_col`` column.  May contain a ``ticker`` column for labelling.
    price_col:
        Column to compute returns from.  Default: ``"close_price"``.
    method:
        ``"simple"``  →  ``(P_t / P_{t-1}) - 1``
        ``"log"``     →  ``ln(P_t / P_{t-1})``
        Default: ``"simple"``.
    ticker_col:
        Name of the ticker column to preserve in output.  Pass ``None`` to
        skip.  Default: ``"ticker"``.

    Returns
    -------
    pandas.DataFrame
        Original columns retained plus:
        ``daily_return``     — arithmetic or log return (float64)
        ``return_method``    — literal ``"simple"`` or ``"log"``

        First row is ``NaN`` (no prior day to compare against).
        Rows where the price is NaN or zero are also ``NaN``.

    Raises
    ------
    ValueError
        ``price_col`` not found in DataFrame, or ``method`` is invalid.
    """
    _validate_price_col(df, price_col)
    _validate_method(method)

    log.info(
        "Computing daily returns",
        extra={"price_col": price_col, "method": method, "rows": len(df)},
    )

    result = df.copy()
    prices = pd.to_numeric(result[price_col], errors="coerce")

    if method == "simple":
        result["daily_return"] = prices.pct_change()
    else:
        result["daily_return"] = prices.apply(lambda x: x).pipe(
            lambda s: s.div(s.shift(1)).apply(_safe_log)
        )

    result["return_method"] = method

    log.info(
        "Daily returns computed",
        extra={
            "non_null": int(result["daily_return"].notna().sum()),
            "mean": (
                round(float(result["daily_return"].mean()), 6)
                if result["daily_return"].notna().any()
                else None
            ),
        },
    )
    return result


def compute_monthly_returns(
    df: pd.DataFrame,
    price_col: str = "close_price",
    method: ReturnMethod = "simple",
    date_col: str = "price_date",
    ticker_col: str | None = "ticker",
) -> pd.DataFrame:
    """
    Aggregate daily prices to month-end and compute monthly returns.

    Parameters
    ----------
    df:
        DataFrame with a ``date_col`` column and ``price_col`` column.
        Should be sorted by date ascending.
    price_col:
        Column to compute returns from.  Default: ``"close_price"``.
    method:
        ``"simple"`` or ``"log"``.  Default: ``"simple"``.
    date_col:
        Name of the date column.  Default: ``"price_date"``.
    ticker_col:
        Name of the ticker column carried into the output.  Default:
        ``"ticker"``.  Pass ``None`` to skip.

    Returns
    -------
    pandas.DataFrame
        Columns: ``period`` (Period[M]), ``month_end_price``,
        ``monthly_return``, ``return_method``.
        Optionally includes ``ticker`` when ``ticker_col`` is present.
        First row is always ``NaN`` (no prior month).

    Raises
    ------
    ValueError
        Required columns missing or ``method`` invalid.
    """
    _validate_price_col(df, price_col)
    _validate_method(method)
    if date_col not in df.columns:
        raise ValueError(f"date_col={date_col!r} not found in DataFrame.")

    log.info(
        "Computing monthly returns",
        extra={"price_col": price_col, "method": method, "input_rows": len(df)},
    )

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.set_index(date_col).sort_index()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")

    # Resample to month-end, taking the last available close price.
    agg: dict[str, object] = {price_col: "last"}
    if ticker_col and ticker_col in work.columns:
        agg[ticker_col] = "last"

    monthly = work.resample("ME").agg(agg).reset_index()
    monthly = monthly.rename(columns={date_col: "period", price_col: "month_end_price"})
    monthly["period"] = monthly["period"].dt.to_period("M")

    prices = monthly["month_end_price"]
    if method == "simple":
        monthly["monthly_return"] = prices.pct_change()
    else:
        monthly["monthly_return"] = prices.div(prices.shift(1)).apply(_safe_log)

    monthly["return_method"] = method

    log.info(
        "Monthly returns computed",
        extra={"months": len(monthly), "method": method},
    )
    return monthly


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_price_col(df: pd.DataFrame, price_col: str) -> None:
    if price_col not in df.columns:
        raise ValueError(
            f"price_col={price_col!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_method(method: str) -> None:
    if method not in ("simple", "log"):
        raise ValueError(f"method={method!r} is not valid. Choose 'simple' or 'log'.")


def _safe_log(ratio: float) -> float:
    """Natural log of ratio; returns NaN for non-positive or NaN input."""
    import math

    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
        return float("nan")
    if ratio <= 0:
        return float("nan")
    return math.log(ratio)
