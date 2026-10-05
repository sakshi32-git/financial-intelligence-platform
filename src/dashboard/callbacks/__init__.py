"""Dash callbacks — interactivity and data binding."""

from src.dashboard.callbacks.stock import register_stock_callbacks
from src.dashboard.callbacks.commodity import register_commodity_callbacks
from src.dashboard.callbacks.economic import register_economic_callbacks
from src.dashboard.callbacks.correlation import register_correlation_callbacks
from src.dashboard.callbacks.eye_control import register_eye_control_callbacks

__all__ = [
    "register_stock_callbacks",
    "register_commodity_callbacks",
    "register_economic_callbacks",
    "register_correlation_callbacks",
    "register_eye_control_callbacks",
]
