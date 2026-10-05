"""
correlation.py
==============
Callbacks for the Correlation Analysis page.

Queries the database for multiple tickers.  Falls back to synthetic demo
data when the database is unavailable so the dashboard always renders.
"""

from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output

from src.dashboard.demo_data import generate_stock_prices_multi

log = logging.getLogger(__name__)

_DB_AVAILABLE = True


def _load_multi(tickers: list[str]) -> pd.DataFrame:
    """Load from DB; fall back to demo data on any error."""
    global _DB_AVAILABLE
    if _DB_AVAILABLE:
        try:
            from src.analytics.loader import AnalyticsLoader
            start = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            with AnalyticsLoader() as loader:
                df = loader.load_prices_multi(tickers=tickers, start=start)
            if not df.empty:
                return df
        except Exception as exc:
            log.warning("DB unavailable for correlation, using demo data: %s", exc)
            _DB_AVAILABLE = False
    return generate_stock_prices_multi(tickers)


def register_correlation_callbacks(app: Dash) -> None:
    """Register all callbacks for the Correlation Analysis page."""

    @app.callback(
        Output("correlation-heatmap", "figure"),
        Input("correlation-ticker-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_correlation_heatmap(tickers: list[str]):
        """Compute and render the Pearson correlation heatmap."""
        if not tickers or len(tickers) < 2:
            return go.Figure().update_layout(
                title=dict(text="Select at least 2 assets.", font=dict(color="#0f172a", size=14)),
                template="plotly_white",
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff"
            )
        log.info("Updating correlation heatmap for: %s", tickers)

        try:
            df = _load_multi(tickers)
            df["price_date"] = pd.to_datetime(df["price_date"])

            pivot = df.pivot(index="price_date", columns="ticker", values="close_price")
            returns = pivot.pct_change().dropna()
            corr = returns.corr()

            labels = corr.columns.tolist()
            z = corr.values

            fig = go.Figure(data=go.Heatmap(
                z=z, x=labels, y=labels,
                colorscale="RdBu", zmin=-1, zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(color="#0f172a", size=13, family="Inter, sans-serif"),
                hoverinfo="x+y+z",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Pearson<br>Correlation", font=dict(color="#0f172a", size=13)),
                    tickfont=dict(color="#0f172a", size=12)
                ),
            ))
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
                margin=dict(l=60, r=40, t=40, b=60),
                xaxis=dict(
                    tickangle=-45,
                    showgrid=False,
                    tickfont=dict(color="#0f172a", size=13),
                    title=dict(font=dict(color="#0f172a", size=13))
                ),
                yaxis=dict(
                    autorange="reversed",
                    showgrid=False,
                    tickfont=dict(color="#0f172a", size=13),
                    title=dict(font=dict(color="#0f172a", size=13))
                ),
                hovermode="closest",
            )
            return fig

        except Exception as exc:
            log.error("Correlation callback error: %s", exc, exc_info=True)
            return go.Figure().update_layout(
                title=dict(text=str(exc), font=dict(color="#0f172a")),
                template="plotly_white",
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff"
            )
