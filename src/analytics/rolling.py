"""
rolling.py
==========
Rolling window metrics for financial price series.

All functions are pure: DataFrame in → DataFrame out.
No database I/O. No visualisation.

Public API
----------
compute_rolling_average(df, price_col, windows, min_periods)  → DataFrame
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Default window sizes (trading days) used when none are specified.
_DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 200)


def compute_rolling_average(
    df: pd.DataFrame,
    price_col: str = "close_price",
    windows: list[int] | tuple[int, ...] = _DEFAULT_WINDOWS,
    min_periods: int | None = None,
    center: bool = False,
) -> pd.DataFrame:
    """
    Compute simple moving averages (SMA) over multiple window sizes.

    Parameters
    ----------
    df:
        DataFrame sorted by date ascending with at least ``price_col``.
    price_col:
        Column to average.  Default: ``"close_price"``.
    windows:
        Sequence of integer window sizes in rows (trading days).
        Default: ``(5, 10, 20, 50, 200)``.
    min_periods:
        Minimum non-NaN observations required to produce a value.
        Defaults to each window size (i.e. full window required).
    center:
        When ``True`` the window is centred on the current row.
        Default: ``False`` (trailing window — standard for finance).

    Returns
    -------
    pandas.DataFrame
        All original columns retained plus one new column per window:

        * ``sma_{w}``  — simple moving average over ``w`` periods

        Leading rows (< window size) are ``NaN`` unless ``min_periods``
        is set to a smaller value.

    Raises
    ------
    ValueError
        ``price_col`` not found, no windows provided, or any window ≤ 0.
    """
    if price_col not in df.columns:
        raise ValueError(
            f"price_col={price_col!r} not found in DataFrame. "
            f"Available: {list(df.columns)}"
        )
    windows = list(windows)
    if not windows:
        raise ValueError("windows must contain at least one value.")
    for w in windows:
        if not isinstance(w, int) or w <= 0:
            raise ValueError(f"All window sizes must be positive integers, got {w!r}.")

    log.info(
        "Computing rolling averages",
        extra={"price_col": price_col, "windows": windows, "rows": len(df)},
    )

    result = df.copy()
    prices = pd.to_numeric(result[price_col], errors="coerce")

    for w in windows:
        mp = min_periods if min_periods is not None else w
        col_name = f"sma_{w}"
        result[col_name] = prices.rolling(
            window=w, min_periods=mp, center=center
        ).mean()
        log.debug(
            "SMA computed",
            extra={
                "window": w,
                "non_null": int(result[col_name].notna().sum()),
            },
        )

    log.info(
        "Rolling averages complete",
        extra={"columns_added": [f"sma_{w}" for w in windows]},
    )
    return result
