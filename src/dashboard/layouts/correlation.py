"""
correlation.py
==============
Layout for the Correlation Analysis page.

Displays a multi-select dropdown to choose assets and a heatmap showing
their price return correlations.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

def correlation_layout() -> html.Div:
    """Return the layout for the Correlation Analysis page."""
    return html.Div(
        className="page-container",
        children=[
            # ── Header & Controls ──────────────────────────────────────────
            dbc.Row(
                className="mb-4 align-items-center",
                children=[
                    dbc.Col(
                        html.H2("Asset Correlation", className="page-title m-0"),
                        width=12, md=6,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="correlation-ticker-dropdown",
                            options=[
                                {"label": "Apple (AAPL)", "value": "AAPL"},
                                {"label": "Microsoft (MSFT)", "value": "MSFT"},
                                {"label": "Nvidia (NVDA)", "value": "NVDA"},
                                {"label": "Tesla (TSLA)", "value": "TSLA"},
                                {"label": "Amazon (AMZN)", "value": "AMZN"},
                                {"label": "Alphabet (GOOGL)", "value": "GOOGL"},
                                {"label": "Meta (META)", "value": "META"},
                            ],
                            value=["AAPL", "MSFT", "NVDA", "TSLA"],
                            multi=True,
                            clearable=False,
                            className="dash-bootstrap",
                        ),
                        width=12, md=6,
                        className="text-end",
                    ),
                ],
            ),

            # ── Description ────────────────────────────────────────────────
            dbc.Row(
                className="mb-4",
                children=[
                    dbc.Col(
                        html.P(
                            "Analyse the linear relationship (Pearson correlation) between "
                            "daily price returns of selected assets over the past year. "
                            "Values near 1.0 indicate strong positive correlation, while values "
                            "near -1.0 indicate strong negative correlation.",
                            className="text-muted"
                        ),
                        width=12
                    )
                ]
            ),

            # ── Heatmap Chart ──────────────────────────────────────────────
            dbc.Row(
                children=[
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card",
                            children=[
                                dbc.CardHeader("Correlation Matrix (1 Year)"),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="correlation-heatmap",
                                        config={"displayModeBar": False},
                                        style={"height": "600px"}
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
