"""
economic.py
===========
Callbacks for the Economic Indicators page.

Queries the FRED API directly.  Falls back to synthetic demo data when the
API key is not configured or the request fails.
"""

from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output

from src.dashboard.components.kpi_card import kpi_card
from src.dashboard.demo_data import generate_economic_series

log = logging.getLogger(__name__)


def _load_fred(series_id: str) -> pd.DataFrame:
    """Fetch from FRED API; fall back to demo data on any error."""
    try:
        from src.api.fred.client import FREDClient
        start = (pd.Timestamp.now() - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
        client = FREDClient()
        df = client.get_series(series_id, start=start)
        df = df.dropna(subset=["value"]).reset_index(drop=True)
        if not df.empty:
            return df
    except Exception as exc:
        log.warning("FRED API unavailable for %s, using demo data: %s", series_id, exc)
    return generate_economic_series(series_id)


def register_economic_callbacks(app: Dash) -> None:
    """Register all callbacks for the Economic Indicators page."""

    @app.callback(
        [
            Output("economic-kpi-latest", "children"),
            Output("economic-kpi-change", "children"),
            Output("economic-kpi-yoy", "children"),
            Output("economic-kpi-trend", "children"),
            Output("economic-chart", "figure"),
        ],
        Input("economic-indicator-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_economic_dashboard(series_id: str):
        """Update KPI cards and chart when an indicator is selected."""
        if not series_id:
            series_id = "UNRATE"
        log.info("Updating economic dashboard for series: %s", series_id)

        try:
            df = _load_fred(series_id)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            latest_row = df.iloc[-1]
            latest_val = float(latest_row["value"])
            latest_date = pd.Timestamp(latest_row["date"])
            prev_val = float(df.iloc[-2]["value"])
            period_chg = latest_val - prev_val

            is_rate = series_id in ("UNRATE", "FEDFUNDS")
            suffix = "%" if is_rate else ""

            # YoY change
            yoy_date = latest_date - pd.DateOffset(years=1)
            yoy_df = df[df["date"] <= yoy_date]
            yoy_chg = 0.0
            if not yoy_df.empty:
                yoy_val = float(yoy_df.iloc[-1]["value"])
                yoy_chg = (latest_val - yoy_val) if is_rate else (
                    (latest_val - yoy_val) / yoy_val * 100 if yoy_val else 0.0
                )

            # 3-Year trend
            trend_date = latest_date - pd.DateOffset(years=3)
            trend_df = df[df["date"] <= trend_date]
            trend_chg = 0.0
            if not trend_df.empty:
                trend_val = float(trend_df.iloc[-1]["value"])
                trend_chg = (latest_val - trend_val) if is_rate else (
                    (latest_val - trend_val) / trend_val * 100 if trend_val else 0.0
                )

            kpi_latest = kpi_card(
                f"Latest ({latest_date.strftime('%Y-%m')})",
                f"{latest_val:,.2f}{suffix}", icon="bi-tag-fill"
            )
            kpi_change = kpi_card(
                "Period Change", f"{period_chg:+.2f}{suffix}",
                delta=period_chg, icon="bi-graph-up"
            )
            kpi_yoy = kpi_card(
                "Year-over-Year",
                f"{yoy_chg:+.2f}{'%' if not is_rate else suffix}",
                delta=yoy_chg, icon="bi-calendar-check"
            )
            kpi_trend = kpi_card(
                "3-Year Trend",
                f"{trend_chg:+.2f}{'%' if not is_rate else suffix}",
                delta=trend_chg, icon="bi-arrow-trend-up"
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["value"],
                mode="lines", name=series_id,
                line=dict(color="#059669", width=2.5),
                fill="tozeroy", fillcolor="rgba(5,150,105,0.08)"
            ))
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#e2e8f0",
                    tickfont=dict(color="#0f172a", size=12),
                    title=dict(font=dict(color="#0f172a", size=13)),
                    linecolor="#cbd5e1"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#e2e8f0",
                    tickfont=dict(color="#0f172a", size=12),
                    title=dict(font=dict(color="#0f172a", size=13)),
                    linecolor="#cbd5e1"
                ),
                hovermode="x unified",
            )
            return kpi_latest, kpi_change, kpi_yoy, kpi_trend, fig

        except Exception as exc:
            log.error("Economic callback error: %s", exc, exc_info=True)
            empty_kpi = kpi_card("Error", "-", icon="bi-exclamation-triangle")
            empty_fig = go.Figure().update_layout(
                title=dict(text=str(exc), font=dict(color="#0f172a")),
                template="plotly_white",
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff"
            )
            return empty_kpi, empty_kpi, empty_kpi, empty_kpi, empty_fig
