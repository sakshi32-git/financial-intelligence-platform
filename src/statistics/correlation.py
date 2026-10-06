"""
correlation.py
==============
Pearson correlation analysis for financial return series.

All functions are pure: DataFrame in → DataFrame out.
No database I/O. No visualisation.

Public API
----------
compute_correlation_matrix(returns_wide, min_periods)  → DataFrame
compute_pairwise_correlation(returns_wide, min_periods) → DataFrame
compute_rolling_correlation(returns_wide, ticker_a,
                            ticker_b, window, min_periods) → DataFrame
pivot_returns(returns_long, date_col, ticker_col,
              return_col)                               → DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


# ===========================================================================
# Public functions
# ===========================================================================


def pivot_returns(
    returns_long: pd.DataFrame,
    date_col: str = "price_date",
    ticker_col: str = "ticker",
    return_col: str = "daily_return",
) -> pd.DataFrame:
    """
    Pivot a long-format returns DataFrame to wide format.

    The wide format is required by all correlation functions in this module.

    Parameters
    ----------
    returns_long:
        Long DataFrame with columns ``date_col``, ``ticker_col``,
        ``return_col`` — typically the output of
        :func:`src.analytics.returns.compute_daily_returns` run over a
        multi-ticker price set.
    date_col:
        Name of the date column.  Default: ``"price_date"``.
    ticker_col:
        Name of the ticker column.  Default: ``"ticker"``.
    return_col:
        Name of the return column to pivot as values.
        Default: ``"daily_return"``.

    Returns
    -------
    pandas.DataFrame
        Index: dates (``DatetimeIndex``).
        Columns: one column per ticker.
        Values: daily returns (``float64``); missing trading days are ``NaN``.

    Raises
    ------
    ValueError
        Any required column is missing.
    """
    _require_cols(returns_long, [date_col, ticker_col, return_col])

    log.info(
        "Pivoting returns to wide format",
        extra={
            "date_col": date_col,
            "ticker_col": ticker_col,
            "return_col": return_col,
            "rows": len(returns_long),
        },
    )

    wide = returns_long.pivot_table(
        index=date_col,
        columns=ticker_col,
        values=return_col,
        aggfunc="last",
    ).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None  # Remove "ticker" label from column axis.

    log.info(
        "Pivot complete",
        extra={"shape": wide.shape, "tickers": list(wide.columns)},
    )
    return wide


def compute_correlation_matrix(
    returns_wide: pd.DataFrame,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix across all ticker pairs.

    Parameters
    ----------
    returns_wide:
        Wide-format DataFrame (dates × tickers) as produced by
        :func:`pivot_returns`.  Missing values (``NaN``) are handled
        pairwise.
    min_periods:
        Minimum number of overlapping non-NaN observations required to
        compute a valid correlation.  Pairs with fewer observations
        receive ``NaN``.  Default: 30.

    Returns
    -------
    pandas.DataFrame
        Symmetric ``N × N`` Pearson correlation matrix.
        Index and columns are ticker symbols.
        Diagonal is always ``1.0``.
        Off-diagonal entries are in ``[-1, 1]`` or ``NaN``.

    Raises
    ------
    ValueError
        DataFrame is empty or contains fewer than 2 columns.
    """
    _validate_wide(returns_wide)

    log.info(
        "Computing Pearson correlation matrix",
        extra={"tickers": list(returns_wide.columns), "min_periods": min_periods},
    )

    corr_matrix = returns_wide.corr(method="pearson", min_periods=min_periods)

    log.info(
        "Correlation matrix computed",
        extra={"shape": corr_matrix.shape},
    )
    return corr_matrix


def compute_pairwise_correlation(
    returns_wide: pd.DataFrame,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute Pearson correlation for every unique ticker pair with
    p-values and sample sizes in tidy (long) format.

    Parameters
    ----------
    returns_wide:
        Wide-format DataFrame (dates × tickers) as produced by
        :func:`pivot_returns`.
    min_periods:
        Minimum overlapping observations required.  Pairs below this
        threshold are marked ``NaN`` and flagged.  Default: 30.

    Returns
    -------
    pandas.DataFrame
        One row per unique ordered pair ``(ticker_a, ticker_b)`` where
        ``ticker_a < ticker_b`` (upper triangle only, no duplicates).

        Columns:

        * ``ticker_a``        — first ticker
        * ``ticker_b``        — second ticker
        * ``pearson_r``       — Pearson correlation coefficient
        * ``p_value``         — two-tailed p-value (H₀: ρ = 0)
        * ``n_obs``           — number of overlapping non-NaN observations
        * ``significant``     — ``True`` when p_value < 0.05
        * ``sufficient_data`` — ``True`` when n_obs ≥ min_periods

        Sorted by ``pearson_r`` descending.

    Raises
    ------
    ValueError
        DataFrame is empty or contains fewer than 2 columns.
    """
    _validate_wide(returns_wide)

    tickers = list(returns_wide.columns)
    log.info(
        "Computing pairwise Pearson correlations",
        extra={"tickers": tickers, "min_periods": min_periods},
    )

    rows: list[dict] = []

    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            s_a = returns_wide[a]
            s_b = returns_wide[b]

            # Align on common non-NaN rows.
            mask = s_a.notna() & s_b.notna()
            n_obs = int(mask.sum())

            if n_obs < 2:
                r, p = float("nan"), float("nan")
            else:
                r, p = stats.pearsonr(s_a[mask], s_b[mask])

            rows.append(
                {
                    "ticker_a": a,
                    "ticker_b": b,
                    "pearson_r": round(float(r), 6) if not np.isnan(r) else np.nan,
                    "p_value": round(float(p), 6) if not np.isnan(p) else np.nan,
                    "n_obs": n_obs,
                    "significant": (p < 0.05) if not np.isnan(p) else False,
                    "sufficient_data": n_obs >= min_periods,
                }
            )

    result = (
        pd.DataFrame(rows)
        .sort_values("pearson_r", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    log.info(
        "Pairwise correlations computed",
        extra={"pairs": len(result)},
    )
    return result


def compute_rolling_correlation(
    returns_wide: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute the rolling Pearson correlation between two tickers over time.

    Parameters
    ----------
    returns_wide:
        Wide-format DataFrame (dates × tickers) as produced by
        :func:`pivot_returns`.
    ticker_a:
        First ticker symbol — must be a column in ``returns_wide``.
    ticker_b:
        Second ticker symbol — must be a column in ``returns_wide``.
    window:
        Rolling window size in trading days.  Default: 60.
    min_periods:
        Minimum observations inside the window to produce a non-NaN
        value.  Defaults to ``window``.

    Returns
    -------
    pandas.DataFrame
        Columns:

        * ``date``            — ``DatetimeIndex`` dates
        * ``ticker_a``        — literal ticker symbol (constant column)
        * ``ticker_b``        — literal ticker symbol (constant column)
        * ``rolling_r``       — rolling Pearson correlation coefficient
        * ``window``          — the window size used (constant column)

        Leading ``window - 1`` rows are ``NaN``.

    Raises
    ------
    ValueError
        Either ticker is missing, tickers are identical, or window ≤ 1.
    """
    _validate_wide(returns_wide)

    for t in (ticker_a, ticker_b):
        if t not in returns_wide.columns:
            raise ValueError(
                f"ticker={t!r} not found in returns_wide. "
                f"Available: {list(returns_wide.columns)}"
            )
    if ticker_a == ticker_b:
        raise ValueError("ticker_a and ticker_b must be different tickers.")
    if not isinstance(window, int) or window <= 1:
        raise ValueError(f"window must be an integer > 1, got {window!r}.")

    mp = min_periods if min_periods is not None else window

    log.info(
        "Computing rolling Pearson correlation",
        extra={
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "window": window,
        },
    )

    rolling_r = (
        returns_wide[ticker_a]
        .rolling(window=window, min_periods=mp)
        .corr(returns_wide[ticker_b])
    )

    result = pd.DataFrame(
        {
            "date": returns_wide.index,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "rolling_r": rolling_r.values,
            "window": window,
        }
    ).reset_index(drop=True)

    log.info(
        "Rolling correlation computed",
        extra={
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "window": window,
            "non_null": int(result["rolling_r"].notna().sum()),
        },
    )
    return result


# ===========================================================================
# Internal helpers
# ===========================================================================


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. " f"Available: {list(df.columns)}"
        )


def _validate_wide(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("returns_wide DataFrame is empty.")
    if len(df.columns) < 2:
        raise ValueError(
            "returns_wide must contain at least 2 ticker columns, "
            f"got {list(df.columns)}."
        )
