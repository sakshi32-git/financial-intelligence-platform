"""Dash page layouts."""

from src.dashboard.layouts.stock import stock_layout
from src.dashboard.layouts.commodity import commodity_layout
from src.dashboard.layouts.economic import economic_layout
from src.dashboard.layouts.correlation import correlation_layout

__all__ = ["stock_layout", "commodity_layout", "economic_layout", "correlation_layout"]
