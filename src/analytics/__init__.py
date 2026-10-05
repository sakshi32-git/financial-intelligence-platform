"""Analytics engine — daily returns, monthly returns, rolling average, volatility."""

from src.analytics.loader import AnalyticsLoader  # noqa: F401
from src.analytics.returns import (  # noqa: F401
    compute_daily_returns,
    compute_monthly_returns,
)
from src.analytics.rolling import compute_rolling_average  # noqa: F401
from src.analytics.volatility import (  # noqa: F401
    compute_parkinson_volatility,
    compute_volatility,
)

__all__ = [
    # Data access
    "AnalyticsLoader",
    # Returns
    "compute_daily_returns",
    "compute_monthly_returns",
    # Rolling
    "compute_rolling_average",
    # Volatility
    "compute_volatility",
    "compute_parkinson_volatility",
]
