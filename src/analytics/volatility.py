"""
volatility.py
=============
Historical volatility calculations for financial price series.

All functions are pure: DataFrame in → DataFrame out.
No database I/O. No visualisation.

Public API
----------
compute_volatility(df, return_col, windows, trading_days)  → DataFrame
compute_parkinson_volatility(df, windows, trading_days)    → DataFrame
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Standard annualisation factors.
_TRADING_DAYS_PER_YEAR: int = 252

# Default rolling windows (trading days).
_DEFAULT_WINDOWS: tuple[int, ...] = (10, 20, 30, 60)


def compute_volatility(
    df: pd.DataFrame,
    return_col: str = "daily_return",
    windows: list[int] | tuple[int, ...] = _DEFAULT_WINDOWS,
    trading_days: int = _TRADING_DAYS_PER_YEAR,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute rolling historical volatility (annualised standard deviation of
    log/simple returns) over multiple window sizes.

    Parameters
    ----------
    df:
        DataFrame with at least ``return_col`` (daily returns).
        Typically the output of :func:`compute_daily_returns` with
        ``method="log"``, but simple returns are also accepted.
    return_col:
        Column containing daily returns.  Default: ``"daily_return"``.
    windows:
        Rolling window sizes in trading days.
        Default: ``(10, 20, 30, 60)``.
    trading_days:
        Annualisation factor.  Default: 252 (US equity market).
    min_periods:
        Minimum observations for a non-NaN result.  Defaults to each
        window size.

    Returns
    -------
    pandas.DataFrame
        All original columns retained plus:

        * ``vol_{w}d``          — rolling annualised volatility for window ``w``
        * ``vol_{w}d_pct``      — same value expressed as a percentage

        Values are ``NaN`` for the leading ``w`` rows.

    Raises
    ------
    ValueError
        ``return_col`` not found, invalid windows, or ``trading_days`` ≤ 0.
    """
    if return_col not in df.columns:
        raise ValueError(
            f"return_col={return_col!r} not found in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    windows = list(windows)
    if not windows:
        raise ValueError("windows must contain at least one value.")
    for w in windows:
        if not isinstance(w, int) or w <= 1:
            raise ValueError(f"All window sizes must be integers > 1, got {w!r}.")
    if trading_days <= 0:
        raise ValueError(f"trading_days must be > 0, got {trading_days}.")

    log.info(
        "Computing historical volatility",
        extra={
            "return_col": return_col,
            "windows": windows,
            "trading_days": trading_days,
        },
    )

    result = df.copy()
    returns = pd.to_numeric(result[return_col], errors="coerce")
    annualise = math.sqrt(trading_days)

    for w in windows:
        mp = min_periods if min_periods is not None else w
        raw_vol = returns.rolling(window=w, min_periods=mp).std(ddof=1)
        col = f"vol_{w}d"
        result[col] = raw_vol * annualise
        result[f"{col}_pct"] = result[col] * 100

        log.debug(
            "Volatility window computed",
            extra={
                "window": w,
                "non_null": int(result[col].notna().sum()),
                "mean_annualised": (
                    round(float(result[col].mean()), 6)
                    if result[col].notna().any()
                    else None
                ),
            },
        )

    log.info(
        "Historical volatility complete",
        extra={"columns_added": [f"vol_{w}d" for w in windows]},
    )
    return result


def compute_parkinson_volatility(
    df: pd.DataFrame,
    windows: list[int] | tuple[int, ...] = _DEFAULT_WINDOWS,
    trading_days: int = _TRADING_DAYS_PER_YEAR,
    high_col: str = "high_price",
    low_col: str = "low_price",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute Parkinson (high-low range) volatility estimator.

    Parkinson's estimator uses the intraday high and low to estimate
    volatility more efficiently than close-to-close returns when the
    underlying price follows a diffusion process with no drift.

    Formula (per day)::

        parkinson_daily = sqrt( 1/(4 * ln2) * ln(H/L)^2 )

    Annualised::

        parkinson_annual = parkinson_daily * sqrt(trading_days)

    Parameters
    ----------
    df:
        DataFrame with ``high_price`` and ``low_price`` columns.
    windows:
        Rolling window sizes in trading days.
        Default: ``(10, 20, 30, 60)``.
    trading_days:
        Annualisation factor.  Default: 252.
    high_col:
        Name of the high-price column.  Default: ``"high_price"``.
    low_col:
        Name of the low-price column.  Default: ``"low_price"``.
    min_periods:
        Minimum observations for a non-NaN result.

    Returns
    -------
    pandas.DataFrame
        All original columns retained plus:

        * ``pk_daily``          — Parkinson daily volatility (point estimate)
        * ``pk_vol_{w}d``       — rolling annualised Parkinson volatility
        * ``pk_vol_{w}d_pct``   — same expressed as a percentage

    Raises
    ------
    ValueError
        Required columns missing or invalid window sizes.
    """
    for col in (high_col, low_col):
        if col not in df.columns:
            raise ValueError(
                f"{col!r} not found in DataFrame. " f"Available: {list(df.columns)}"
            )
    windows = list(windows)
    if not windows:
        raise ValueError("windows must contain at least one value.")
    for w in windows:
        if not isinstance(w, int) or w <= 1:
            raise ValueError(f"All window sizes must be integers > 1, got {w!r}.")

    log.info(
        "Computing Parkinson volatility",
        extra={"windows": windows, "trading_days": trading_days},
    )

    result = df.copy()
    high = pd.to_numeric(result[high_col], errors="coerce")
    low = pd.to_numeric(result[low_col], errors="coerce")

    _4ln2 = 4.0 * math.log(2)

    # Point-in-time daily Parkinson estimate.
    log_hl_sq = np.log(high / low) ** 2
    result["pk_daily"] = np.sqrt(log_hl_sq / _4ln2)

    annualise = math.sqrt(trading_days)

    for w in windows:
        mp = min_periods if min_periods is not None else w
        # Rolling mean of the squared log(H/L) term, then annualise.
        rolling_mean_sq = log_hl_sq.rolling(window=w, min_periods=mp).mean()
        col = f"pk_vol_{w}d"
        result[col] = np.sqrt(rolling_mean_sq / _4ln2) * annualise
        result[f"{col}_pct"] = result[col] * 100

        log.debug(
            "Parkinson window computed",
            extra={"window": w, "non_null": int(result[col].notna().sum())},
        )

    log.info(
        "Parkinson volatility complete",
        extra={"columns_added": [f"pk_vol_{w}d" for w in windows]},
    )
    return result
