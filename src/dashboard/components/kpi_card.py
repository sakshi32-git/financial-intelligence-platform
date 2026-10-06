"""
kpi_card.py
===========
Reusable Key Performance Indicator (KPI) card component.

Displays a primary metric, its title, an optional icon, and an optional
delta/change value with conditional colouring (green for positive, red for negative).

Exports
-------
kpi_card(title, value, delta, icon, delta_prefix) → dash component
"""

from __future__ import annotations

from typing import Any
import dash_bootstrap_components as dbc
from dash import html


def kpi_card(
    title: str,
    value: str | int | float,
    delta: float | None = None,
    icon: str | None = None,
    delta_prefix: str = "",
) -> html.Div:
    """
    Build a styled KPI card.

    Parameters
    ----------
    title : str
        The label for the metric (e.g., "Total Revenue", "Active Users").
    value : str | int | float
        The primary metric value to display.
    delta : float, optional
        The change value. If provided, renders a colored badge (green if > 0,
        red if < 0, neutral if == 0).
    icon : str, optional
        A Bootstrap Icon class (e.g., "bi-graph-up").
    delta_prefix : str, optional
        A string prepended to the delta (e.g., "+" or "-"). When None, computes
        automatically for numbers.

    Returns
    -------
    html.Div
        The KPI card component.
    """
    # ── Delta badge ────────────────────────────────────────────────────────
    delta_element = None
    if delta is not None:
        try:
            delta_val = float(delta)
            if delta_val > 0:
                color_class = "kpi-delta-positive"
                arrow = "bi-arrow-up-short"
                sign = "+"
            elif delta_val < 0:
                color_class = "kpi-delta-negative"
                arrow = "bi-arrow-down-short"
                sign = ""  # Negative sign is included in the number
            else:
                color_class = "kpi-delta-neutral"
                arrow = "bi-dash"
                sign = ""

            display_delta = f"{delta_prefix}{sign}{delta_val:g}%"

            delta_element = html.Div(
                className=f"kpi-delta {color_class}",
                children=[
                    html.I(className=f"bi {arrow} kpi-delta-icon"),
                    html.Span(display_delta, className="kpi-delta-text"),
                ],
            )
        except (ValueError, TypeError):
            # Fallback if delta is not a standard number
            delta_element = html.Div(
                className="kpi-delta kpi-delta-neutral",
                children=html.Span(str(delta), className="kpi-delta-text"),
            )

    # ── Icon ───────────────────────────────────────────────────────────────
    icon_element = None
    if icon:
        icon_element = html.Div(
            className="kpi-icon-wrap",
            children=html.I(className=f"bi {icon} kpi-icon"),
        )

    # ── Card layout ────────────────────────────────────────────────────────
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(
                className="kpi-header",
                children=[
                    html.Span(title, className="kpi-title"),
                    icon_element,
                ],
            ),
            html.Div(
                className="kpi-body",
                children=[
                    html.Span(str(value), className="kpi-value"),
                    delta_element,
                ],
            ),
        ],
    )
