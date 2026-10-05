"""
regression.py
=============
Sklearn regression pipeline for predicting stock returns.

Implements train / evaluate / predict with time-series-safe splits.
No data leakage: future data never appears in training folds.
All outputs are pandas DataFrames.

Public API
----------
``ReturnRegressor``
    .fit(feature_df)                                   → TrainResult
    .predict(feature_df)                               → DataFrame
    .cross_validate(feature_df, n_splits)              → DataFrame
    .compare_models(feature_df)                        → DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation.metrics import (
    compute_directional_accuracy,
    compute_regression_metrics,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model identifiers
# ---------------------------------------------------------------------------

ModelName = Literal["linear", "ridge", "lasso", "elasticnet"]

_MODEL_REGISTRY: dict[str, object] = {
    "linear":     LinearRegression(),
    "ridge":      Ridge(alpha=1.0),
    "lasso":      Lasso(alpha=0.001, max_iter=10_000),
    "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10_000),
}

_TARGET_COL = "target_return"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Artefacts produced by a single :meth:`ReturnRegressor.fit` call."""

    model_name: str
    ticker: str
    pipeline: Pipeline
    feature_names: list[str]
    train_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    direction_metrics: pd.DataFrame
    predictions: pd.DataFrame          # Index-aligned test predictions
    coefficients: pd.DataFrame         # Feature importances / coefficients


# ===========================================================================
# Regressor
# ===========================================================================


class ReturnRegressor:
    """
    Sklearn regression pipeline for predicting forward stock returns.

    Wraps any sklearn linear regressor with:

    * ``StandardScaler`` preprocessing (features only, not the target)
    * Time-series-safe train/test split (no random shuffle)
    * Per-fold and aggregate evaluation metrics returned as DataFrames

    Parameters
    ----------
    model_name:
        One of ``"linear"``, ``"ridge"``, ``"lasso"``, ``"elasticnet"``.
        Default: ``"ridge"``.
    test_size:
        Fraction of rows held out as the test set (chronologically last).
        Must be in ``(0, 1)``.  Default: ``0.2``.
    ticker:
        Ticker symbol — carried into result objects for identification.
    """

    def __init__(
        self,
        model_name: ModelName = "ridge",
        test_size: float = 0.2,
        ticker: str = "UNKNOWN",
    ) -> None:
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"model_name={model_name!r} is not supported. "
                f"Choose from: {list(_MODEL_REGISTRY.keys())}."
            )
        if not (0 < test_size < 1):
            raise ValueError(
                f"test_size must be in (0, 1), got {test_size}."
            )

        self.model_name = model_name
        self.test_size = test_size
        self.ticker = ticker
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fit(self, feature_df: pd.DataFrame) -> TrainResult:
        """
        Train the regression pipeline on a feature matrix.

        The dataset is split chronologically: the last ``test_size``
        fraction of rows form the test set; no shuffling is applied.

        Parameters
        ----------
        feature_df:
            Output of :func:`src.ml.features.engineering.build_features`.
            Must contain ``target_return`` and at least one feature column.

        Returns
        -------
        TrainResult
            Contains the fitted pipeline, train/test metrics DataFrames,
            directional accuracy, index-aligned test predictions, and
            a coefficients DataFrame.

        Raises
        ------
        ValueError
            Target column missing, insufficient rows, or no features.
        """
        _validate_feature_df(feature_df)

        feature_names = [c for c in feature_df.columns if c != _TARGET_COL]
        X = feature_df[feature_names].values.astype(float)
        y = feature_df[_TARGET_COL].values.astype(float)

        split_idx = int(len(X) * (1 - self.test_size))
        if split_idx < 10:
            raise ValueError(
                f"Training set has only {split_idx} rows after split. "
                "Provide more data or reduce test_size."
            )

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        idx_test = feature_df.index[split_idx:]

        log.info(
            "Fitting regression model",
            extra={
                "model": self.model_name,
                "ticker": self.ticker,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "features": feature_names,
            },
        )

        # Build sklearn Pipeline: scaler → regressor
        estimator = _clone_estimator(self.model_name)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ])
        pipeline.fit(X_train, y_train)

        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred  = pipeline.predict(X_test)

        # Metrics
        train_metrics = compute_regression_metrics(
            y_train, y_train_pred, label=f"{self.model_name}_train"
        )
        test_metrics = compute_regression_metrics(
            y_test, y_test_pred, label=f"{self.model_name}_test"
        )
        direction_metrics = compute_directional_accuracy(
            y_test, y_test_pred, label=f"{self.model_name}_test"
        )

        # Test predictions DataFrame (index-aligned)
        predictions = pd.DataFrame(
            {
                "date":           idx_test,
                "ticker":         self.ticker,
                "y_true":         y_test,
                "y_pred":         y_test_pred,
                "residual":       y_test - y_test_pred,
                "model":          self.model_name,
            }
        ).reset_index(drop=True)

        # Coefficients / feature importances
        coef_df = self._extract_coefficients(pipeline, feature_names)

        self._pipeline = pipeline
        self._feature_names = feature_names

        log.info(
            "Model fitted",
            extra={
                "model": self.model_name,
                "ticker": self.ticker,
                "test_r2":   float(test_metrics["r2"].iloc[0]),
                "test_rmse": float(test_metrics["rmse"].iloc[0]),
            },
        )

        return TrainResult(
            model_name=self.model_name,
            ticker=self.ticker,
            pipeline=pipeline,
            feature_names=feature_names,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            direction_metrics=direction_metrics,
            predictions=predictions,
            coefficients=coef_df,
        )

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from a fitted pipeline.

        Parameters
        ----------
        feature_df:
            Feature matrix (same schema as used in :meth:`fit`).
            ``target_return`` column is optional — it will be included
            in the output only when present.

        Returns
        -------
        pandas.DataFrame
            Columns: ``date``, ``ticker``, ``y_pred``, and optionally
            ``y_true`` and ``residual`` when target is present.

        Raises
        ------
        RuntimeError
            Called before :meth:`fit`.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first."
            )

        missing = [f for f in self._feature_names if f not in feature_df.columns]
        if missing:
            raise ValueError(
                f"feature_df is missing columns used during training: {missing}."
            )

        X = feature_df[self._feature_names].values.astype(float)
        y_pred = self._pipeline.predict(X)

        result = pd.DataFrame(
            {
                "date":   feature_df.index,
                "ticker": self.ticker,
                "y_pred": y_pred,
                "model":  self.model_name,
            }
        )

        if _TARGET_COL in feature_df.columns:
            y_true = feature_df[_TARGET_COL].values.astype(float)
            result["y_true"]   = y_true
            result["residual"] = y_true - y_pred

        return result.reset_index(drop=True)

    def cross_validate(
        self,
        feature_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Time-series cross-validation returning per-fold metrics.

        Uses :class:`sklearn.model_selection.TimeSeriesSplit` — each fold's
        test set is strictly after its training set (no shuffling, no leakage).

        Parameters
        ----------
        feature_df:
            Output of :func:`build_features`.
        n_splits:
            Number of folds.  Default: 5.

        Returns
        -------
        pandas.DataFrame
            One row per fold with columns from
            :func:`compute_regression_metrics` plus ``fold``, ``model``,
            ``ticker``, ``train_rows``, ``test_rows``.
        """
        _validate_feature_df(feature_df)

        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

        feature_names = [c for c in feature_df.columns if c != _TARGET_COL]
        X = feature_df[feature_names].values.astype(float)
        y = feature_df[_TARGET_COL].values.astype(float)

        log.info(
            "Running time-series cross-validation",
            extra={
                "model": self.model_name,
                "ticker": self.ticker,
                "n_splits": n_splits,
                "rows": len(X),
            },
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results: list[pd.DataFrame] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", _clone_estimator(self.model_name)),
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            fold_metrics = compute_regression_metrics(
                y_test, y_pred, label=f"fold_{fold_idx}"
            )
            fold_metrics["fold"]       = fold_idx
            fold_metrics["model"]      = self.model_name
            fold_metrics["ticker"]     = self.ticker
            fold_metrics["train_rows"] = len(X_train)
            fold_metrics["test_rows"]  = len(X_test)

            fold_results.append(fold_metrics)

        cv_df = pd.concat(fold_results, ignore_index=True)

        log.info(
            "Cross-validation complete",
            extra={
                "model": self.model_name,
                "ticker": self.ticker,
                "mean_r2":   round(float(cv_df["r2"].mean()), 6),
                "mean_rmse": round(float(cv_df["rmse"].mean()), 6),
            },
        )
        return cv_df

    def compare_models(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Train and evaluate all supported models on the same data.

        Returns a single DataFrame sorted by test RMSE ascending so the
        best model appears first.

        Parameters
        ----------
        feature_df:
            Output of :func:`build_features`.

        Returns
        -------
        pandas.DataFrame
            Columns: all metrics from :func:`compute_regression_metrics`
            plus ``directional_accuracy``, ``model``, ``ticker``.
            One row per model.
        """
        _validate_feature_df(feature_df)

        log.info(
            "Comparing all regression models",
            extra={"ticker": self.ticker, "rows": len(feature_df)},
        )

        rows: list[pd.DataFrame] = []

        for name in _MODEL_REGISTRY:
            regressor = ReturnRegressor(
                model_name=name,        # type: ignore[arg-type]
                test_size=self.test_size,
                ticker=self.ticker,
            )
            result = regressor.fit(feature_df)

            combined = result.test_metrics.copy()
            combined["directional_accuracy"] = float(
                result.direction_metrics["directional_accuracy"].iloc[0]
            )
            combined["model"]  = name
            combined["ticker"] = self.ticker
            rows.append(combined)

        comparison = (
            pd.concat(rows, ignore_index=True)
            .sort_values("rmse", ascending=True)
            .reset_index(drop=True)
        )

        log.info(
            "Model comparison complete",
            extra={"best_model": comparison["model"].iloc[0]},
        )
        return comparison

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline to disk with joblib."""
        if self._pipeline is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self._pipeline, "feature_names": self._feature_names},
            path,
        )
        log.info("Model saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: str | Path, ticker: str = "UNKNOWN") -> "ReturnRegressor":
        """Load a previously saved pipeline from disk."""
        artefact = joblib.load(path)
        obj = cls.__new__(cls)
        obj._pipeline = artefact["pipeline"]
        obj._feature_names = artefact["feature_names"]
        obj.ticker = ticker
        obj.model_name = "loaded"
        obj.test_size = 0.2
        log.info("Model loaded", extra={"path": str(path)})
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_coefficients(
        pipeline: Pipeline,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Return a DataFrame of scaled feature coefficients."""
        regressor = pipeline.named_steps["regressor"]
        if not hasattr(regressor, "coef_"):
            return pd.DataFrame({"feature": feature_names, "coefficient": [None] * len(feature_names)})

        coef = regressor.coef_
        return (
            pd.DataFrame({"feature": feature_names, "coefficient": coef})
            .assign(abs_coefficient=lambda df: df["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _clone_estimator(model_name: str) -> object:
    """Return a fresh (unfitted) clone of the requested estimator."""
    from sklearn.base import clone
    return clone(_MODEL_REGISTRY[model_name])


def _validate_feature_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("feature_df is empty.")
    if _TARGET_COL not in df.columns:
        raise ValueError(
            f"feature_df must contain a '{_TARGET_COL}' column. "
            "Use build_features() to construct it."
        )
    feature_cols = [c for c in df.columns if c != _TARGET_COL]
    if not feature_cols:
        raise ValueError("feature_df contains no feature columns besides the target.")
