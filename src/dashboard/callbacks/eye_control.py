"""
eye_control.py
==============
Callbacks for Eye Control Mode toggle, status updates, and calibration settings.
"""

from __future__ import annotations

import logging
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from src.eye_tracking.eye_tracker import eye_tracker

log = logging.getLogger(__name__)


def register_eye_control_callbacks(app: Dash) -> None:
    """Register all callbacks for Eye Control Mode."""

    @app.callback(
        [
            Output("eye-mode-status-badge", "children"),
            Output("eye-mode-status-badge", "color"),
            Output("eye-mode-alert", "children"),
            Output("eye-mode-alert", "is_open"),
            Output("eye-mode-alert", "color"),
        ],
        [
            Input("mouse-mode-radio", "value"),
            Input("eye-status-interval", "n_intervals"),
        ],
        prevent_initial_call=False,
    )
    def handle_eye_mode_toggle(mode: str, n_intervals: int):
        """Toggle Eye Control mode and update status indicators."""
        # Ensure eye_tracker thread state matches selected mode
        if mode == "EYE" and not eye_tracker.is_running:
            res = eye_tracker.start()
            log.info("Eye tracker start result: %s", res)
        elif mode == "NORMAL" and eye_tracker.is_running:
            res = eye_tracker.stop()
            log.info("Eye tracker stop result: %s", res)

        status = eye_tracker.get_status()
        is_active = status["active"]
        error = status["error"]
        msg = status["status_message"]

        if is_active:
            badge_text = "👁️ Eye Control Active"
            badge_color = "success"
            alert_text = msg
            alert_open = True
            alert_color = "info"
        elif error:
            badge_text = "⚠️ Camera / Permission Error"
            badge_color = "danger"
            alert_text = f"Eye Control Error: {error}"
            alert_open = True
            alert_color = "danger"
        else:
            badge_text = "🖱️ Normal Mouse Active"
            badge_color = "secondary"
            alert_text = "Normal Mouse Mode is active."
            alert_open = False
            alert_color = "secondary"

        return badge_text, badge_color, alert_text, alert_open, alert_color

    @app.callback(
        Output("eye-calibration-collapse", "is_open"),
        Input("eye-calibration-toggle-btn", "n_clicks"),
        State("eye-calibration-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_calibration_panel(n_clicks: int, is_open: bool):
        """Toggle the calibration settings panel."""
        return not is_open

    @app.callback(
        Output("eye-calibration-saved-alert", "is_open"),
        [
            Input("eye-smoothing-slider", "value"),
            Input("eye-blink-slider", "value"),
            Input("eye-cooldown-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_calibration_settings(smoothing: float, blink_thresh: float, cooldown: float):
        """Update live calibration parameters on the EyeTrackerController."""
        if smoothing:
            eye_tracker.smoothing_alpha = float(smoothing)
        if blink_thresh:
            eye_tracker.blink_threshold = float(blink_thresh)
        if cooldown:
            eye_tracker.click_cooldown = float(cooldown)
        log.info(
            "Updated Eye Control calibration: smoothing=%.2f, blink=%.2f, cooldown=%.2f",
            eye_tracker.smoothing_alpha,
            eye_tracker.blink_threshold,
            eye_tracker.click_cooldown,
        )
        return True
