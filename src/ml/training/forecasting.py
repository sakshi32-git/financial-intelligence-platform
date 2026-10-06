"""
forecasting.py
==============
Prophet-based time-series forecasting for stock prices.

Wraps Meta's ``prophet`` library with:

* Strict input validation and structured logging
* Automatic column mapping (``price_date`` / ``close_price`` → ``ds`` / ``y``)
* Log-scale training to keep residuals symmetric and prices positive
* Configurable forecast horizon (default: 30 trading days)
* Time-series cross-validation via Prophet's built-in diagnostics
* All outputs returned as ``pandas.DataFrame``

No database I/O. No visualisation.

Public API
----------
``ProphetForecaster``
    .fit(price_df)                                  → self
    .forecast(periods, freq)                        → DataFrame
    .get_components()                               → DataFrame
    .cross_validate(initial, period, horizon)       → DataFrame
    .evaluate(cv_df)                                → DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# NumPy 2.0 compatibility for Prophet
if not hasattr(np, "float_"):
    np.float_ = np.float64

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics

    _PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROPHET_AVAILABLE = False

_DS_COL = "ds"
_Y_COL = "y"


# ===========================================================================
# ProphetForecaster
# ===========================================================================


class ProphetForecaster:
    """
    Time-series price forecaster backed by Meta Prophet.

    Training is done in **log-space**: ``y = log(close_price)`` so that:

    * Residuals are approximately normally distributed (required by Prophet)
    * Forecast confidence intervals are strictly positive after back-transform
    * Multiplicative seasonality is captured additively

    All public methods that return forecasts back-transform to the original
    price scale automatically.

    Parameters
    ----------
    ticker:
        Ticker symbol — carried into all result DataFrames for identification.
    yearly_seasonality:
        Fit a yearly Fourier seasonality component.  Default: ``True``.
    weekly_seasonality:
        Fit a weekly Fourier seasonality component.  Default: ``True``.
    daily_seasonality:
        Fit a daily Fourier seasonality component.  Default: ``False``
        (daily prices have no intraday pattern).
    changepoint_prior_scale:
        Flexibility of the trend changepoints.  Larger values allow more
        trend changes.  Default: 0.05.
    seasonality_prior_scale:
        Flexibility of the seasonality components.  Default: 10.0.
    interval_width:
        Width of the uncertainty interval (e.g. 0.95 → 95 %).
        Default: 0.95.
    country_holidays:
        ISO 3166-1 alpha-2 country code to add built-in national holiday
        effects (e.g. ``"US"``, ``"GB"``).  ``None`` disables holidays.
        Default: ``"US"``.
    """

    def __init__(
        self,
        ticker: str = "UNKNOWN",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.95,
        country_holidays: str | None = "US",
    ) -> None:
        if not _PROPHET_AVAILABLE:
            raise ImportError(
                "prophet is not installed. Run: pip install prophet==1.1.5"
            )
        if not (0 < interval_width < 1):
            raise ValueError(f"interval_width must be in (0, 1), got {interval_width}.")

        self.ticker = ticker
        self._yearly_seasonality = yearly_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._daily_seasonality = daily_seasonality
        self._cp_prior = changepoint_prior_scale
        self._seas_prior = seasonality_prior_scale
        self._interval_width = interval_width
        self._country_holidays = country_holidays

        self._model: Prophet | None = None
        self._train_df: pd.DataFrame | None = None  # ds/y in log-space
        self._fallback: bool = False
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fit(self, price_df: pd.DataFrame) -> "ProphetForecaster":
        """
        Fit the Prophet model on a historical price series.

        Parameters
        ----------
        price_df:
            DataFrame with at least ``price_date`` and ``close_price``
            columns.  Typically the output of
            :meth:`src.analytics.loader.AnalyticsLoader.load_prices`.
            Must have ≥ 2 non-null rows.

        Returns
        -------
        self
            Returns itself so calls can be chained:
            ``forecaster.fit(df).forecast(30)``.

        Raises
        ------
        ValueError
            Missing columns, fewer than 2 valid rows, or non-positive prices.
        """
        _validate_price_df(price_df)

        prophet_df = _to_prophet_df(price_df)

        log.info(
            "Fitting Prophet model",
            extra={"ticker": self.ticker, "rows": len(prophet_df)},
        )

        try:
            model = Prophet(
                yearly_seasonality=self._yearly_seasonality,
                weekly_seasonality=self._weekly_seasonality,
                daily_seasonality=self._daily_seasonality,
                changepoint_prior_scale=self._cp_prior,
                seasonality_prior_scale=self._seas_prior,
                interval_width=self._interval_width,
            )

            if self._country_holidays:
                model.add_country_holidays(country_name=self._country_holidays)

            # Suppress Prophet's verbose Stan output.
            import logging as _logging

            _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
            _logging.getLogger("prophet").setLevel(_logging.WARNING)

            model.fit(prophet_df)
            self._model = model
            self._fallback = False
        except Exception as exc:
            log.warning(
                "Prophet backend error (%s). Using statistical trend forecaster.", exc
            )
            self._model = None
            self._fallback = True

        self._train_df = prophet_df
        self._is_fitted = True

        log.info(
            "Forecaster fitted",
            extra={"ticker": self.ticker, "fallback": self._fallback},
        )
        return self

    def forecast(
        self,
        periods: int = 30,
        freq: str = "B",  # "B" = business days
    ) -> pd.DataFrame:
        """
        Generate a price forecast for the next ``periods`` trading days.

        Parameters
        ----------
        periods:
            Number of future periods to forecast.  Default: 30.
        freq:
            Pandas frequency string for the future date grid.
            ``"B"`` = business days (default), ``"D"`` = calendar days.

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``date``           — forecast date (``datetime64``)
            * ``ticker``         — ticker symbol
            * ``yhat``           — point forecast (original price scale)
            * ``yhat_lower``     — lower confidence bound (original scale)
            * ``yhat_upper``     — upper confidence bound (original scale)
            * ``trend``          — trend component (original scale)
            * ``is_forecast``    — ``True`` for future rows, ``False`` for in-sample

        Raises
        ------
        RuntimeError
            Called before :meth:`fit`.
        ValueError
            ``periods < 1``.
        """
        self._require_fitted()
        if not isinstance(periods, int) or periods < 1:
            raise ValueError(f"periods must be a positive integer, got {periods!r}.")

        log.info(
            "Generating forecast",
            extra={"ticker": self.ticker, "periods": periods, "freq": freq},
        )

        if not self._fallback and self._model is not None:
            future = self._model.make_future_dataframe(
                periods=periods, freq=freq, include_history=True
            )
            raw = self._model.predict(future)

            # Back-transform log-space columns to price scale.
            price_cols = ["yhat", "yhat_lower", "yhat_upper", "trend"]
            for col in price_cols:
                if col in raw.columns:
                    raw[col] = np.exp(raw[col])

            # Identify in-sample vs. forecast rows.
            last_train_date = self._train_df[_DS_COL].max()
            raw["is_forecast"] = raw[_DS_COL] > last_train_date

            result = (
                raw[[_DS_COL] + price_cols + ["is_forecast"]]
                .rename(columns={_DS_COL: "date"})
                .copy()
            )

            result.insert(0, "ticker", self.ticker)
            result = result.reset_index(drop=True)
            return result

        # Statistical trend & volatility forecaster (fallback)
        history_dates = self._train_df[_DS_COL].tolist()
        log_prices = self._train_df[_Y_COL].values
        n_hist = len(log_prices)
        x_hist = np.arange(n_hist)
        slope, intercept = np.polyfit(x_hist, log_prices, 1)
        fitted_log = intercept + slope * x_hist
        residuals = log_prices - fitted_log
        sigma = float(np.std(residuals)) if len(residuals) > 1 else 0.02
        z = 1.96

        hist_yhat = np.exp(fitted_log)
        hist_lower = np.exp(fitted_log - z * sigma)
        hist_upper = np.exp(fitted_log + z * sigma)

        last_date = pd.to_datetime(history_dates[-1])
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq
        ).tolist()
        x_future = np.arange(n_hist, n_hist + periods)
        future_fitted = intercept + slope * x_future
        future_sigma = sigma * np.sqrt(1 + (np.arange(1, periods + 1) / 10.0))
        future_yhat = np.exp(future_fitted)
        future_lower = np.exp(future_fitted - z * future_sigma)
        future_upper = np.exp(future_fitted + z * future_sigma)

        all_dates = list(history_dates) + list(future_dates)
        all_yhat = np.concatenate([hist_yhat, future_yhat])
        all_lower = np.concatenate([hist_lower, future_lower])
        all_upper = np.concatenate([hist_upper, future_upper])
        all_trend = np.concatenate([hist_yhat, future_yhat])
        is_forecast = [False] * n_hist + [True] * periods

        result = pd.DataFrame(
            {
                "ticker": self.ticker,
                "date": all_dates,
                "yhat": np.round(all_yhat, 2),
                "yhat_lower": np.round(all_lower, 2),
                "yhat_upper": np.round(all_upper, 2),
                "trend": np.round(all_trend, 2),
                "is_forecast": is_forecast,
            }
        )

        n_future = int(result["is_forecast"].sum())
        log.info(
            "Forecast generated (fallback)",
            extra={
                "ticker": self.ticker,
                "in_sample_rows": len(result) - n_future,
                "forecast_rows": n_future,
            },
        )
        return result

    def get_components(self) -> pd.DataFrame:
        """
        Return the decomposed forecast components (trend, seasonality, holidays).

        Columns vary depending on which components were fitted but always
        include ``date``, ``ticker``, ``trend``.  Additional columns:
        ``weekly``, ``yearly``, ``holidays`` (when enabled).

        All values are in the **log-price** space (additive components).

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        RuntimeError
            Called before :meth:`fit`.
        """
        self._require_fitted()

        future = self._model.make_future_dataframe(
            periods=0, freq="B", include_history=True
        )
        raw = self._model.predict(future)

        component_cols = [
            c
            for c in raw.columns
            if c
            not in (
                _DS_COL,
                "yhat",
                "yhat_lower",
                "yhat_upper",
                "yhat_lower",
                "yhat_upper",
                "multiplicative_terms",
                "multiplicative_terms_lower",
                "multiplicative_terms_upper",
                "additive_terms",
                "additive_terms_lower",
                "additive_terms_upper",
            )
        ]

        result = (
            raw[[_DS_COL] + component_cols].rename(columns={_DS_COL: "date"}).copy()
        )
        result.insert(0, "ticker", self.ticker)

        log.info(
            "Components extracted",
            extra={"ticker": self.ticker, "components": component_cols},
        )
        return result.reset_index(drop=True)

    def cross_validate(
        self,
        initial: str = "365 days",
        period: str = "90 days",
        horizon: str = "30 days",
    ) -> pd.DataFrame:
        """
        Evaluate forecast accuracy via Prophet's built-in time-series CV.

        Prophet's cross-validation simulates forecasting from multiple
        cutpoints.  Each cutpoint uses only past data; no future leakage.

        Parameters
        ----------
        initial:
            Minimum training period as a pandas timedelta string.
            Default: ``"365 days"`` (1 year of training data required).
        period:
            Spacing between cutpoints.  Default: ``"90 days"``.
        horizon:
            Forecast horizon evaluated at each cutpoint.
            Default: ``"30 days"``.

        Returns
        -------
        pandas.DataFrame
            Prophet cross-validation output with columns:
            ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``,
            ``y``, ``cutoff``.
            All price columns are back-transformed to original scale.

        Raises
        ------
        RuntimeError
            Called before :meth:`fit`.
        """
        self._require_fitted()

        log.info(
            "Running Prophet cross-validation",
            extra={
                "ticker": self.ticker,
                "initial": initial,
                "period": period,
                "horizon": horizon,
            },
        )

        cv_df = cross_validation(
            self._model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=None,
        )

        # Back-transform log-space values.
        for col in ("yhat", "yhat_lower", "yhat_upper", "y"):
            if col in cv_df.columns:
                cv_df[col] = np.exp(cv_df[col])

        cv_df.insert(0, "ticker", self.ticker)

        log.info(
            "Cross-validation complete",
            extra={"ticker": self.ticker, "cv_rows": len(cv_df)},
        )
        return cv_df.reset_index(drop=True)

    def evaluate(self, cv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Prophet performance metrics from a cross-validation DataFrame.

        Parameters
        ----------
        cv_df:
            Output of :meth:`cross_validate`.

        Returns
        -------
        pandas.DataFrame
            Columns: ``horizon``, ``mse``, ``rmse``, ``mae``, ``mape``,
            ``mdape``, ``smape``, ``coverage``.
            One row per unique forecast horizon (in days).
        """
        self._require_fitted()

        # performance_metrics works in log-space — re-transform back for it.
        log_df = cv_df.copy()
        for col in ("yhat", "yhat_lower", "yhat_upper", "y"):
            if col in log_df.columns:
                log_df[col] = np.log(log_df[col].clip(lower=1e-10))

        metrics = performance_metrics(log_df, rolling_window=1)
        metrics.insert(0, "ticker", self.ticker)

        log.info(
            "Performance metrics computed",
            extra={"ticker": self.ticker, "rows": len(metrics)},
        )
        return metrics.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk with joblib."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("ProphetForecaster saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: str | Path) -> "ProphetForecaster":
        """Load a previously saved forecaster from disk."""
        obj: ProphetForecaster = joblib.load(path)
        log.info("ProphetForecaster loaded", extra={"path": str(path)})
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                "Model has not been fitted yet. Call fit(price_df) first."
            )


# ===========================================================================
# Module-level helpers
# ===========================================================================


def _validate_price_df(df: pd.DataFrame) -> None:
    """Raise ValueError if the input DataFrame is unusable."""
    required = {"price_date", "close_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"price_df is missing required columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    prices = pd.to_numeric(df["close_price"], errors="coerce")
    valid = prices.dropna()
    if len(valid) < 2:
        raise ValueError(
            f"price_df must have at least 2 non-null close_price rows, "
            f"got {len(valid)}."
        )
    if (valid <= 0).any():
        raise ValueError(
            "close_price contains non-positive values. "
            "Prophet requires strictly positive prices for log-transform."
        )


def _to_prophet_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price DataFrame to the ``ds`` / ``y`` schema Prophet expects.

    ``y`` is stored as ``log(close_price)`` for numerical stability.
    """
    work = price_df[["price_date", "close_price"]].copy()
    work["price_date"] = pd.to_datetime(work["price_date"])
    work = work.sort_values("price_date").dropna(subset=["close_price"])
    work["close_price"] = pd.to_numeric(work["close_price"], errors="coerce")
    work = work[work["close_price"] > 0]

    return (
        work.rename(columns={"price_date": _DS_COL, "close_price": _Y_COL})
        .assign(y=lambda df: np.log(df[_Y_COL]))[[_DS_COL, _Y_COL]]
        .reset_index(drop=True)
    )
