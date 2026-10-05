"""
stock.py
========
Layout for the Stocks page.

Displays a dropdown for ticker selection, a set of KPI cards for summary metrics,
and a main price chart.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.dashboard.components.kpi_card import kpi_card

def stock_layout() -> html.Div:
    """Return the layout for the Stock Analysis page."""
    return html.Div(
        className="page-container",
        children=[
            # ── Header & Controls ──────────────────────────────────────────
            dbc.Row(
                className="mb-4 align-items-center",
                children=[
                    dbc.Col(
                        html.H2("Stock Analysis", className="page-title m-0"),
                        width=12, md=6,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="stock-ticker-dropdown",
                            options=[
                                {"label": "Apple (AAPL)", "value": "AAPL"},
                                {"label": "Microsoft (MSFT)", "value": "MSFT"},
                                {"label": "Nvidia (NVDA)", "value": "NVDA"},
                                {"label": "Tesla (TSLA)", "value": "TSLA"},
                            ],
                            value="AAPL",
                            clearable=False,
                            className="dash-bootstrap",
                        ),
                        width=12, md=6,
                        className="text-end",
                    ),
                ],
            ),

            # ── KPI Cards ──────────────────────────────────────────────────
            dbc.Row(
                id="stock-kpi-row",
                className="mb-4 g-3",
                children=[
                    dbc.Col(kpi_card("Latest Close", "-", icon="bi-tag-fill"), width=12, sm=6, lg=3, id="stock-kpi-close"),
                    dbc.Col(kpi_card("Daily Return", "-", icon="bi-graph-up"), width=12, sm=6, lg=3, id="stock-kpi-return"),
                    dbc.Col(kpi_card("30d Volatility", "-", icon="bi-activity"), width=12, sm=6, lg=3, id="stock-kpi-volatility"),
                    dbc.Col(kpi_card("30d Trend (SMA)", "-", icon="bi-arrow-trend-up"), width=12, sm=6, lg=3, id="stock-kpi-sma"),
                ]
            ),

            # ── Charts ─────────────────────────────────────────────────────
            dbc.Row(
                children=[
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card",
                            children=[
                                dbc.CardHeader("Historical Price & Forecast"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="stock-price-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "400px"}
                                    )
                                )
                            ]
                        ),
                        width=12
                    )
                ]
            )
        ]
    )
