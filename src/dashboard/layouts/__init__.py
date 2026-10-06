"""Dash page layouts."""

from src.dashboard.layouts.overview import overview_layout
from src.dashboard.layouts.stock import stock_layout
from src.dashboard.layouts.commodity import commodity_layout
from src.dashboard.layouts.economic import economic_layout
from src.dashboard.layouts.correlation import correlation_layout

__all__ = [
    "overview_layout",
    "stock_layout",
    "commodity_layout",
    "economic_layout",
    "correlation_layout",
]
