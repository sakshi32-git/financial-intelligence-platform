"""
metrics.py
==========
Regression evaluation metrics for stock-return prediction models.

All functions are pure: arrays/DataFrames in → DataFrame out.
No model training. No database I/O. No visualisation.

Public API
----------
compute_regression_metrics(y_true, y_pred, label)   → DataFrame
compute_directional_accuracy(y_true, y_pred)        → DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

log = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    label: str = "model",
) -> pd.DataFrame:
    """
    Compute standard regression metrics for a single model/fold.

    Metrics computed
    ----------------
    * ``r2``      — Coefficient of determination (R²)
    * ``mae``     — Mean absolute error
    * ``rmse``    — Root mean squared error
    * ``mse``     — Mean squared error
    * ``mape``    — Mean absolute percentage error (returns without zeros only)
    * ``corr``    — Pearson correlation between y_true and y_pred

    Parameters
    ----------
    y_true:
        Actual return values.
    y_pred:
        Predicted return values.
    label:
        Identifier for this result row (e.g. model name or fold number).

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with columns:
        ``label``, ``r2``, ``mae``, ``rmse``, ``mse``, ``mape``, ``corr``,
        ``n_samples``.

    Raises
    ------
    ValueError
        ``y_true`` and ``y_pred`` have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true (n={len(y_true)}) and y_pred (n={len(y_pred)}) "
            "must have the same length."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    # MAPE: skip zero actuals to avoid division-by-zero.
    nonzero_mask = y_true != 0
    mape = (
        float(mean_absolute_percentage_error(y_true[nonzero_mask], y_pred[nonzero_mask]))
        if nonzero_mask.any()
        else float("nan")
    )

    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    result = pd.DataFrame(
        [{
            "label":     label,
            "r2":        round(r2, 6),
            "mae":       round(mae, 6),
            "rmse":      round(rmse, 6),
            "mse":       round(mse, 6),
            "mape":      round(mape, 6) if not np.isnan(mape) else np.nan,
            "corr":      round(corr, 6),
            "n_samples": len(y_true),
        }]
    )

    log.info(
        "Regression metrics computed",
        extra={"label": label, "r2": r2, "rmse": rmse, "mae": mae},
    )
    return result


def compute_directional_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    label: str = "model",
) -> pd.DataFrame:
    """
    Compute directional accuracy: fraction of predictions where the
    predicted sign matches the actual sign.

    For return prediction, direction accuracy measures whether the model
    correctly predicts up vs. down moves regardless of magnitude.

    Parameters
    ----------
    y_true:
        Actual return values.
    y_pred:
        Predicted return values.
    label:
        Identifier for this result row.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with columns:
        ``label``, ``directional_accuracy``, ``correct_up``,
        ``correct_down``, ``total``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and same length.")

    correct = np.sign(y_true) == np.sign(y_pred)
    up_mask   = y_true > 0
    down_mask = y_true < 0

    directional_accuracy = float(correct.mean())
    correct_up   = float(correct[up_mask].mean())   if up_mask.any()   else float("nan")
    correct_down = float(correct[down_mask].mean())  if down_mask.any() else float("nan")

    result = pd.DataFrame(
        [{
            "label":                 label,
            "directional_accuracy":  round(directional_accuracy, 6),
            "correct_up":            round(correct_up, 6)   if not np.isnan(correct_up)   else np.nan,
            "correct_down":          round(correct_down, 6) if not np.isnan(correct_down) else np.nan,
            "total":                 len(y_true),
        }]
    )

    log.info(
        "Directional accuracy computed",
        extra={"label": label, "directional_accuracy": directional_accuracy},
    )
    return result
