"""
sidebar.py
==========
Responsive navigation sidebar for the Financial Intelligence Platform dashboard.

Renders a fixed left-panel with:
* Platform logo / branding header
* Navigation links for every page
* Active-link highlighting via ``dcc.Location``
* Collapse toggle for narrow screens

Exports
-------
sidebar()   → dash component tree (call once; mount in app layout)
PAGES       — ordered list of page metadata dicts used by the router
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# ---------------------------------------------------------------------------
# Page registry — single source of truth for routing + nav labels
# ---------------------------------------------------------------------------

PAGES: list[dict] = [
    {
        "href": "/",
        "label": "Overview",
        "icon": "bi bi-grid-1x2-fill",
    },
    {
        "href": "/stocks",
        "label": "Stocks",
        "icon": "bi bi-graph-up-arrow",
    },
    {
        "href": "/commodities",
        "label": "Commodities",
        "icon": "bi bi-fuel-pump-fill",
    },
    {
        "href": "/economic",
        "label": "Economic Indicators",
        "icon": "bi bi-bank2",
    },
    {
        "href": "/correlation",
        "label": "Correlation",
        "icon": "bi bi-diagram-3-fill",
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nav_link(page: dict) -> dbc.NavLink:
    """Return a single styled navigation link."""
    return dbc.NavLink(
        children=[
            html.I(className=f"{page['icon']} sidebar-nav-icon"),
            html.Span(page["label"], className="sidebar-nav-label"),
        ],
        href=page["href"],
        active="exact",
        className="sidebar-nav-item",
    )


def _logo_header() -> html.Div:
    """Return the branding block at the top of the sidebar."""
    return html.Div(
        className="sidebar-logo-block",
        children=[
            html.Div(
                className="sidebar-logo-icon-wrap",
                children=html.Span("FI", className="sidebar-logo-letters"),
            ),
            html.Div(
                className="sidebar-logo-text",
                children=[
                    html.Span("Financial", className="sidebar-logo-line1"),
                    html.Span("Intelligence", className="sidebar-logo-line2"),
                ],
            ),
        ],
    )


def _divider(label: str) -> html.Div:
    """Return a labelled section divider."""
    return html.Div(
        className="sidebar-section-divider",
        children=html.Span(label, className="sidebar-section-label"),
    )


def _footer() -> html.Div:
    """Return the sidebar footer with version info."""
    return html.Div(
        className="sidebar-footer",
        children=[
            html.Span("v1.0.0", className="sidebar-footer-version"),
            html.Span(
                "Financial Intelligence Platform", className="sidebar-footer-name"
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _eye_control_section() -> html.Div:
    """Return the Mouse Mode toggle and calibration controls block."""
    return html.Div(
        className="sidebar-eye-control-block px-3 py-2",
        children=[
            _divider("MOUSE MODE"),
            dcc.Interval(id="eye-status-interval", interval=1000),
            # Mode Selector Radio
            dbc.RadioItems(
                id="mouse-mode-radio",
                options=[
                    {"label": " 🖱️ Normal Mouse", "value": "NORMAL"},
                    {"label": " 👁️ Eye Control", "value": "EYE"},
                ],
                value="NORMAL",
                inline=False,
                className="mb-2 text-light font-weight-bold",
            ),
            # Status Badge
            html.Div(
                className="mb-2",
                children=dbc.Badge(
                    "🖱️ Normal Mouse Active",
                    id="eye-mode-status-badge",
                    color="secondary",
                    className="p-2 w-100 text-truncate",
                    style={"fontSize": "0.78rem"},
                ),
            ),
            # Status Alert
            dbc.Alert(
                id="eye-mode-alert",
                is_open=False,
                dismissable=False,
                className="p-2 mb-2 style-alert",
                style={"fontSize": "0.75rem", "lineHeight": "1.2"},
            ),
            # Calibration Collapsible Toggle Button
            dbc.Button(
                "⚙️ Calibration & Sensitivity",
                id="eye-calibration-toggle-btn",
                size="sm",
                color="outline-info",
                className="w-100 mb-2 py-1",
                style={"fontSize": "0.75rem"},
            ),
            # Calibration Panel Collapse
            dbc.Collapse(
                id="eye-calibration-collapse",
                is_open=False,
                children=dbc.Card(
                    className="p-2 bg-dark text-light border-secondary",
                    children=[
                        html.Label(
                            "Cursor Smoothing", className="small mb-0 text-muted"
                        ),
                        dcc.Slider(
                            id="eye-smoothing-slider",
                            min=0.05,
                            max=0.45,
                            step=0.05,
                            value=0.22,
                            marks={0.1: "Slow", 0.22: "Def", 0.4: "Fast"},
                            className="mb-2",
                        ),
                        html.Label(
                            "Blink Sensitivity", className="small mb-0 text-muted"
                        ),
                        dcc.Slider(
                            id="eye-blink-slider",
                            min=0.12,
                            max=0.28,
                            step=0.02,
                            value=0.20,
                            marks={0.15: "High", 0.20: "Def", 0.25: "Low"},
                            className="mb-2",
                        ),
                        html.Label(
                            "Click Cooldown (s)", className="small mb-0 text-muted"
                        ),
                        dcc.Slider(
                            id="eye-cooldown-slider",
                            min=0.4,
                            max=1.6,
                            step=0.2,
                            value=0.8,
                            marks={0.4: "0.4s", 0.8: "0.8s", 1.4: "1.4s"},
                            className="mb-1",
                        ),
                        dbc.Alert(
                            "Settings applied live!",
                            id="eye-calibration-saved-alert",
                            is_open=False,
                            duration=2000,
                            color="success",
                            className="p-1 mt-1 text-center small mb-0",
                        ),
                    ],
                ),
            ),
        ],
    )


def sidebar() -> html.Div:
    """
    Build and return the full sidebar component tree.

    Mount this component once in the root application layout.
    Active-link highlighting is handled automatically by
    ``dbc.NavLink(active="exact")``.

    Returns
    -------
    html.Div
        The complete sidebar ``html.Div`` with id ``"sidebar"``.
    """
    return html.Div(
        id="sidebar",
        className="sidebar",
        children=[
            # Hidden location tracker — required for active-link highlighting.
            dcc.Location(id="sidebar-location", refresh=False),
            # ── Branding ────────────────────────────────────────────────
            _logo_header(),
            html.Hr(className="sidebar-hr"),
            # ── Navigation ──────────────────────────────────────────────
            _divider("NAVIGATION"),
            dbc.Nav(
                id="sidebar-nav",
                vertical=True,
                pills=True,
                className="sidebar-nav",
                children=[_nav_link(page) for page in PAGES],
            ),
            html.Hr(className="sidebar-hr"),
            # ── Mouse & Eye Control Mode ────────────────────────────────
            _eye_control_section(),
            # ── Spacer pushes footer to bottom ──────────────────────────
            html.Div(className="sidebar-spacer"),
            html.Hr(className="sidebar-hr"),
            # ── Footer ──────────────────────────────────────────────────
            _footer(),
        ],
    )
