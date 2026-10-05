"""
engineering.py
==============
Feature engineering for stock-return prediction.

Constructs a supervised learning dataset from OHLCV price data.
All functions are pure: DataFrame in → DataFrame out.
No model training. No database I/O.

Public API
----------
build_features(df, target_horizon, price_col, lags,
               sma_windows, vol_windows)   → DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10)
_DEFAULT_SMA_WINDOWS: tuple[int, ...] = (5, 10, 20)
_DEFAULT_VOL_WINDOWS: tuple[int, ...] = (5, 10, 20)


def build_features(
    df: pd.DataFrame,
    target_horizon: int = 1,
    price_col: str = "close_price",
    lags: tuple[int, ...] | list[int] = _DEFAULT_LAGS,
    sma_windows: tuple[int, ...] | list[int] = _DEFAULT_SMA_WINDOWS,
    vol_windows: tuple[int, ...] | list[int] = _DEFAULT_VOL_WINDOWS,
) -> pd.DataFrame:
    """
    Build a supervised feature matrix for stock-return prediction.

    Features engineered
    -------------------
    * **Lagged returns** — ``return_lag_{n}`` for each n in ``lags``
    * **SMA price ratios** — ``price_to_sma_{w}`` (close / SMA_w)
    * **Rolling volatility** — ``vol_{w}d`` (annualised, window w)
    * **Log volume change** — ``log_volume_chg`` (log(V_t / V_{t-1}))
    * **High-low range** — ``hl_range_pct`` ((H - L) / L × 100)
    * **Target** — ``target_return`` = forward simple return over
      ``target_horizon`` trading days (shifted backward so each row
      holds the label for the *current* feature set)

    Parameters
    ----------
    df:
        Price DataFrame with columns ``price_date``, ``close_price``,
        ``open_price``, ``high_price``, ``low_price``, ``volume``.
        Must be sorted by ``price_date`` ascending (single ticker).
    target_horizon:
        Number of trading days ahead to predict.  Default: 1 (next day).
    price_col:
        Column used for return and SMA calculations.
        Default: ``"close_price"``.
    lags:
        Lag periods for the return feature.  Default: ``(1, 2, 3, 5, 10)``.
    sma_windows:
        Window sizes for SMA ratio features.  Default: ``(5, 10, 20)``.
    vol_windows:
        Window sizes for rolling volatility.  Default: ``(5, 10, 20)``.

    Returns
    -------
    pandas.DataFrame
        Index: ``price_date`` (DatetimeIndex).
        Columns: all feature columns + ``target_return``.
        Rows with any NaN in features OR target are dropped.

    Raises
    ------
    ValueError
        Missing required columns, invalid horizon or lag values.
    """
    _validate_inputs(df, price_col, target_horizon, lags)

    log.info(
        "Building feature matrix",
        extra={
            "price_col": price_col,
            "target_horizon": target_horizon,
            "lags": list(lags),
            "input_rows": len(df),
        },
    )

    work = df.copy()
    work["price_date"] = pd.to_datetime(work["price_date"])
    work = work.set_index("price_date").sort_index()
    close = pd.to_numeric(work[price_col], errors="coerce")

    # ── Daily return (basis for lag features) ──────────────────────────
    daily_ret = close.pct_change()

    # ── Lagged returns ─────────────────────────────────────────────────
    for lag in lags:
        work[f"return_lag_{lag}"] = daily_ret.shift(lag)

    # ── SMA price ratios ───────────────────────────────────────────────
    for w in sma_windows:
        sma = close.rolling(window=w, min_periods=w).mean()
        work[f"price_to_sma_{w}"] = close / sma - 1.0  # Deviation from SMA

    # ── Rolling volatility (annualised) ───────────────────────────────
    _ann = np.sqrt(252)
    for w in vol_windows:
        work[f"vol_{w}d"] = (
            daily_ret.rolling(window=w, min_periods=w).std(ddof=1) * _ann
        )

    # ── Log volume change ─────────────────────────────────────────────
    if "volume" in work.columns:
        vol = pd.to_numeric(work["volume"], errors="coerce").replace(0, np.nan)
        work["log_volume_chg"] = np.log(vol / vol.shift(1))

    # ── High-low range ────────────────────────────────────────────────
    if "high_price" in work.columns and "low_price" in work.columns:
        high = pd.to_numeric(work["high_price"], errors="coerce")
        low = pd.to_numeric(work["low_price"], errors="coerce")
        work["hl_range_pct"] = (high - low) / low * 100.0

    # ── Target: forward return ────────────────────────────────────────
    work["target_return"] = close.pct_change(target_horizon).shift(-target_horizon)

    # ── Select only engineered + target columns ────────────────────────
    feature_cols = (
        [f"return_lag_{n}" for n in lags]
        + [f"price_to_sma_{w}" for w in sma_windows]
        + [f"vol_{w}d" for w in vol_windows]
        + (["log_volume_chg"] if "log_volume_chg" in work.columns else [])
        + (["hl_range_pct"] if "hl_range_pct" in work.columns else [])
    )

    result = work[feature_cols + ["target_return"]].copy()

    before = len(result)
    result = result.dropna()
    dropped = before - len(result)
    if dropped:
        log.debug(
            "Dropped NaN rows after feature construction",
            extra={"dropped": dropped, "remaining": len(result)},
        )

    log.info(
        "Feature matrix built",
        extra={
            "feature_count": len(feature_cols),
            "rows": len(result),
            "features": feature_cols,
        },
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    df: pd.DataFrame,
    price_col: str,
    target_horizon: int,
    lags: tuple | list,
) -> None:
    required = {"price_date", price_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    if not isinstance(target_horizon, int) or target_horizon < 1:
        raise ValueError(
            f"target_horizon must be a positive integer, got {target_horizon!r}."
        )
    for lag in lags:
        if not isinstance(lag, int) or lag < 1:
            raise ValueError(
                f"All lags must be positive integers, got {lag!r}."
            )
