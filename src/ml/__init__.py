"""Machine learning — feature engineering, regression training, Prophet forecasting, evaluation."""

from src.ml.evaluation.metrics import (  # noqa: F401
    compute_directional_accuracy,
    compute_regression_metrics,
)
from src.ml.features.engineering import build_features  # noqa: F401
from src.ml.training.forecasting import ProphetForecaster  # noqa: F401
from src.ml.training.regression import ReturnRegressor, TrainResult  # noqa: F401

__all__ = [
    "build_features",
    "ReturnRegressor",
    "TrainResult",
    "ProphetForecaster",
    "compute_regression_metrics",
    "compute_directional_accuracy",
]
