"""
app.py
======
Dash application entry-point for the Financial Intelligence Platform.

Mounts the sidebar.  Page layouts and callbacks are added incrementally
in subsequent prompts.

Run
---
    python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Thread limits ─────────────────────────────────────────────────────────────
# Must be set BEFORE numpy / scipy / prophet are imported on Windows to prevent
# OpenBLAS "Memory allocation failed" crashes when Dash's reloader forks the
# process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# When running directly as `python src/dashboard/app.py`, the project root
# is not in sys.path. We must add it so `config` and `src` can be imported.
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import dash
import dash_bootstrap_components as dbc
from dash import html

from config import settings
from logging_config import configure_logging
from src.dashboard.components.sidebar import PAGES, sidebar
from src.dashboard.layouts import stock_layout, commodity_layout, economic_layout, correlation_layout
from src.dashboard.callbacks import (
    register_stock_callbacks,
    register_commodity_callbacks,
    register_economic_callbacks,
    register_correlation_callbacks,
    register_eye_control_callbacks,
)

configure_logging()

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        # Bootstrap Icons (sidebar uses bi-* classes)
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
        # Google Fonts — Inter
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,  # Required while pages are added progressively.
    title="Financial Intelligence Platform",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description",
         "content": "Enterprise-grade financial analytics dashboard."},
    ],
)

server = app.server  # Exposed for WSGI deployment (Gunicorn).

# ---------------------------------------------------------------------------
# Callbacks Registration
# ---------------------------------------------------------------------------
register_stock_callbacks(app)
register_commodity_callbacks(app)
register_economic_callbacks(app)
register_correlation_callbacks(app)
register_eye_control_callbacks(app)

# ---------------------------------------------------------------------------
# Root layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    id="app-root",
    children=[
        sidebar(),
        # Page content is rendered here by the router (added next prompt).
        html.Div(
            id="page-content",
            className="page-content",
            children=html.P(
                "Pages are loaded progressively.",
                style={"color": "#6b7280", "marginTop": "40px"},
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@app.callback(
    dash.Output("page-content", "children"),
    dash.Input("sidebar-location", "pathname"),
)
def display_page(pathname: str):
    """Render the appropriate page layout based on the URL."""
    if pathname == "/stocks":
        return stock_layout()
    elif pathname == "/commodities":
        return commodity_layout()
    elif pathname == "/economic":
        return economic_layout()
    elif pathname == "/correlation":
        return correlation_layout()
    elif pathname == "/":
        return html.Div(
            className="text-center mt-5",
            children=[
                html.H2("Welcome to Financial Intelligence Platform"),
                html.P("Select a module from the sidebar to get started.", className="text-muted"),
            ]
        )
    else:
        return html.Div(
            className="text-center mt-5",
            children=[
                html.H2("404: Not Found", className="text-danger"),
                html.P(f"The pathname {pathname} was not recognized.", className="text-muted"),
            ]
        )

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        debug=settings.app.env == "development",
    )
