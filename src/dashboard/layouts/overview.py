"""
overview.py
===========
Executive Overview and Landing Page layout for the Financial Intelligence Platform.
Provides platform health status, featured analytics modules, and direct navigation.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.kpi_card import kpi_card


def overview_layout() -> html.Div:
    """Return the executive overview landing layout."""
    return html.Div(
        className="page-container",
        children=[
            # ── Header Banner ─────────────────────────────────────────────
            dbc.Row(
                className="mb-4 align-items-center",
                children=[
                    dbc.Col(
                        [
                            html.H2(
                                "Executive Overview",
                                className="page-title m-0",
                            ),
                            html.P(
                                "Multi-Asset Analytics, Machine Learning Forecasting & Accessible Interaction",
                                className="text-muted mt-1 mb-0",
                            ),
                        ],
                        width=12,
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Badge(
                                    "🟢 Live Platform",
                                    color="success",
                                    className="me-2 p-2",
                                ),
                                dbc.Badge(
                                    "⚡ ML Ready", color="primary", className="p-2"
                                ),
                            ],
                            className="text-md-end mt-2 mt-md-0",
                        ),
                        width=12,
                        md=4,
                    ),
                ],
            ),
            # ── Platform Metrics ──────────────────────────────────────────
            dbc.Row(
                className="mb-4 g-3",
                children=[
                    dbc.Col(
                        kpi_card(
                            title="Securities Tracked",
                            value="14 Assets",
                            delta="10 Stocks + 4 Commodities",
                            icon="bi-pie-chart-fill",
                        ),
                        width=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        kpi_card(
                            title="Database Records",
                            value="23,267 Rows",
                            delta="High Integrity Daily Bars",
                            icon="bi-database-fill-check",
                        ),
                        width=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        kpi_card(
                            title="ML Forecasting",
                            value="Prophet 30-Day",
                            delta="95% Confidence Intervals",
                            icon="bi-cpu-fill",
                        ),
                        width=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        kpi_card(
                            title="Interaction Modes",
                            value="Standard & Eye Tracking",
                            delta="Webcam Gaze & Blink",
                            icon="bi-eye-fill",
                        ),
                        width=12,
                        sm=6,
                        lg=3,
                    ),
                ],
            ),
            # ── Module Navigation Cards ───────────────────────────────────
            dbc.Row(
                className="mb-4 g-3",
                children=[
                    # Stocks Card
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card h-100",
                            children=[
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.I(
                                                className="bi bi-graph-up-arrow me-2 text-primary"
                                            ),
                                            html.Span("Stock Analytics & Forecasting"),
                                        ]
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Analyze historical prices, 30-day moving averages, daily returns, "
                                            "and historical volatility. Includes 30-day predictive forecasts powered by Meta Prophet.",
                                            className="text-secondary small mb-3",
                                        ),
                                        dbc.Button(
                                            "Explore Stocks →",
                                            href="/stocks",
                                            color="primary",
                                            size="sm",
                                            className="w-100 fw-bold",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        width=12,
                        md=6,
                        lg=3,
                    ),
                    # Commodities Card
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card h-100",
                            children=[
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.I(
                                                className="bi bi-fuel-pump-fill me-2 text-warning"
                                            ),
                                            html.Span("Commodity Markets"),
                                        ]
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Real-time and historical commodity tracking covering Crude Oil (CL), "
                                            "Gold (GC), Silver (SI), and Natural Gas (NG) with volatility analysis.",
                                            className="text-secondary small mb-3",
                                        ),
                                        dbc.Button(
                                            "Explore Commodities →",
                                            href="/commodities",
                                            color="warning",
                                            size="sm",
                                            className="w-100 fw-bold text-dark",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        width=12,
                        md=6,
                        lg=3,
                    ),
                    # Economic Indicators Card
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card h-100",
                            children=[
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.I(
                                                className="bi bi-bank2 me-2 text-success"
                                            ),
                                            html.Span("Macro Indicators (FRED)"),
                                        ]
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Direct integration with the St. Louis Federal Reserve (FRED) API. "
                                            "Track US GDP, CPI Inflation, Unemployment Rate, and Federal Funds Rate.",
                                            className="text-secondary small mb-3",
                                        ),
                                        dbc.Button(
                                            "Explore Economic Data →",
                                            href="/economic",
                                            color="success",
                                            size="sm",
                                            className="w-100 fw-bold",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        width=12,
                        md=6,
                        lg=3,
                    ),
                    # Correlation Matrix Card
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card h-100",
                            children=[
                                dbc.CardHeader(
                                    html.Div(
                                        [
                                            html.I(
                                                className="bi bi-diagram-3-fill me-2 text-info"
                                            ),
                                            html.Span("Cross-Asset Correlation"),
                                        ]
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Interactive Pearson correlation heatmap mapping inter-asset dependencies "
                                            "between technology equities, commodities, and macroeconomic drivers.",
                                            className="text-secondary small mb-3",
                                        ),
                                        dbc.Button(
                                            "Explore Correlation →",
                                            href="/correlation",
                                            color="info",
                                            size="sm",
                                            className="w-100 fw-bold text-white",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        width=12,
                        md=6,
                        lg=3,
                    ),
                ],
            ),
            # ── Architecture & Capabilities Summary ───────────────────────
            dbc.Row(
                className="g-3",
                children=[
                    dbc.Col(
                        dbc.Card(
                            className="dashboard-card",
                            children=[
                                dbc.CardHeader(
                                    "System Architecture & Pipeline Highlights"
                                ),
                                dbc.CardBody(
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6(
                                                        "🗄️ Storage & Data Layer",
                                                        className="fw-bold text-primary",
                                                    ),
                                                    html.P(
                                                        "SQLAlchemy ORM with PostgreSQL / SQLite support. Batch-loaded OHLCV "
                                                        "prices with automated migrations and upsert integrity.",
                                                        className="text-secondary small",
                                                    ),
                                                ],
                                                width=12,
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6(
                                                        "📈 Analytics & Machine Learning",
                                                        className="fw-bold text-primary",
                                                    ),
                                                    html.P(
                                                        "Vectorized daily & log returns, rolling Parkinson & historical volatility, "
                                                        "and Meta Prophet models with 95% forecast confidence bands.",
                                                        className="text-secondary small",
                                                    ),
                                                ],
                                                width=12,
                                                md=4,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.H6(
                                                        "👁️ Accessible Eye Tracking",
                                                        className="fw-bold text-primary",
                                                    ),
                                                    html.P(
                                                        "MediaPipe & OpenCV powered facial landmark mesh detecting pupil position and eye aspect ratio (EAR) "
                                                        "for hands-free dwell clicking and page control.",
                                                        className="text-secondary small",
                                                    ),
                                                ],
                                                width=12,
                                                md=4,
                                            ),
                                        ]
                                    )
                                ),
                            ],
                        ),
                        width=12,
                    )
                ],
            ),
        ],
    )
