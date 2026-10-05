"""
economic.py
===========
Layout for the Economic Indicators page.

Displays a dropdown for indicator selection, KPI cards, and a price/rate chart.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.dashboard.components.kpi_card import kpi_card

def economic_layout() -> html.Div:
    """Return the layout for the Economic Indicators page."""
    return html.Div(
        className="page-container",
        children=[
            # ── Header & Controls ──────────────────────────────────────────
            dbc.Row(
                className="mb-4 align-items-center",
                children=[
                    dbc.Col(
                        html.H2("Economic Indicators", className="page-title m-0"),
                        width=12, md=6,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="economic-indicator-dropdown",
                            options=[
                                {"label": "Gross Domestic Product (GDP)", "value": "GDP"},
                                {"label": "Unemployment Rate (UNRATE)", "value": "UNRATE"},
                                {"label": "Consumer Price Index (CPI)", "value": "CPIAUCSL"},
                                {"label": "Federal Funds Rate (FEDFUNDS)", "value": "FEDFUNDS"},
                            ],
                            value="UNRATE",
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
                id="economic-kpi-row",
                className="mb-4 g-3",
                children=[
                    dbc.Col(kpi_card("Latest Value", "-", icon="bi-tag-fill"), width=12, sm=6, lg=3, id="economic-kpi-latest"),
                    dbc.Col(kpi_card("Period Change", "-", icon="bi-graph-up"), width=12, sm=6, lg=3, id="economic-kpi-change"),
                    dbc.Col(kpi_card("Year-over-Year", "-", icon="bi-calendar-check"), width=12, sm=6, lg=3, id="economic-kpi-yoy"),
                    dbc.Col(kpi_card("3-Year Trend", "-", icon="bi-arrow-trend-up"), width=12, sm=6, lg=3, id="economic-kpi-trend"),
                ]
            ),

            # ── Charts ─────────────────────────────────────────────────────
            dbc.Row(
                children=[
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card",
                            children=[
                                dbc.CardHeader("Historical Data"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="economic-chart",
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
