"""
stock.py
========
Callbacks for the Stocks Analysis page.

Queries the database via AnalyticsLoader.  Falls back to synthetic demo
data when the database is unavailable so the dashboard always renders.
"""

from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output

from src.analytics.returns import compute_daily_returns
from src.analytics.volatility import compute_volatility
from src.analytics.rolling import compute_rolling_average
from src.dashboard.components.kpi_card import kpi_card
from src.dashboard.demo_data import generate_stock_prices

log = logging.getLogger(__name__)

_DB_AVAILABLE = True


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetch 2 years of live daily prices from yfinance ending today.
    Returns a normalised DataFrame with price_date and close_price columns.
    """
    import yfinance as yf

    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=2)

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # include today
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")

    # Flatten multi-level columns (ticker-level) produced by newer yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.reset_index()
    raw.columns = [str(c).strip() for c in raw.columns]

    # Locate the date and close columns robustly
    date_col = next((c for c in raw.columns if c.lower() == "date"), raw.columns[0])
    close_col = next((c for c in raw.columns if c.lower() == "close"), None)
    open_col = next((c for c in raw.columns if c.lower() == "open"), None)
    high_col = next((c for c in raw.columns if c.lower() == "high"), None)
    low_col = next((c for c in raw.columns if c.lower() == "low"), None)
    volume_col = next((c for c in raw.columns if c.lower() == "volume"), None)

    if close_col is None:
        raise ValueError(f"No 'Close' column found. Columns: {raw.columns.tolist()}")

    df = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "price_date": pd.to_datetime(raw[date_col]),
            "open_price": (
                raw[open_col].round(2) if open_col else raw[close_col].round(2)
            ),
            "high_price": (
                raw[high_col].round(2) if high_col else raw[close_col].round(2)
            ),
            "low_price": raw[low_col].round(2) if low_col else raw[close_col].round(2),
            "close_price": raw[close_col].round(2),
            "adj_close": raw[close_col].round(2),
            "volume": raw[volume_col] if volume_col else 0,
        }
    )

    log.info(
        "yfinance: %s — %d rows, latest=%s",
        ticker,
        len(df),
        df["price_date"].max().strftime("%Y-%m-%d"),
    )
    return df


def _load_prices(ticker: str) -> pd.DataFrame:
    """Load live data: yfinance first (always up-to-date), DB if yfinance fails, GBM last resort."""
    # 1. Live yfinance — always try first so chart is always current
    try:
        return _fetch_yfinance(ticker)
    except Exception as exc:
        log.warning("yfinance fetch failed for %s: %s — trying DB", ticker, exc)

    # 2. DB fallback
    global _DB_AVAILABLE
    if _DB_AVAILABLE:
        try:
            from src.analytics.loader import AnalyticsLoader

            start = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
            with AnalyticsLoader() as loader:
                df = loader.load_prices(ticker=ticker, start=start)
            if not df.empty:
                return df
        except Exception as exc:
            log.warning("DB unavailable for stocks: %s", exc)
            _DB_AVAILABLE = False

    # 3. Synthetic GBM as final fallback
    log.warning("Using synthetic demo data for %s", ticker)
    return generate_stock_prices(ticker)


def register_stock_callbacks(app: Dash) -> None:
    """Register all callbacks for the Stock Analysis page."""

    @app.callback(
        [
            Output("stock-kpi-close", "children"),
            Output("stock-kpi-return", "children"),
            Output("stock-kpi-volatility", "children"),
            Output("stock-kpi-sma", "children"),
            Output("stock-price-chart", "figure"),
        ],
        Input("stock-ticker-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_stock_dashboard(ticker: str):
        """Update KPI cards and chart for the selected ticker."""
        if not ticker:
            ticker = "AAPL"
        log.info("Updating stock dashboard for ticker: %s", ticker)

        try:
            price_df = _load_prices(ticker)
            price_df = price_df.sort_values("price_date").reset_index(drop=True)
            price_df["price_date"] = pd.to_datetime(price_df["price_date"])

            latest_price = float(price_df.iloc[-1]["close_price"])
            prev_price = float(price_df.iloc[-2]["close_price"])
            day_change = ((latest_price - prev_price) / prev_price) * 100

            returns_df = compute_daily_returns(price_df)
            vol_df = compute_volatility(returns_df, windows=[30])
            latest_vol = float(vol_df.iloc[-1]["vol_30d"]) * 100

            sma_df = compute_rolling_average(price_df, windows=[30])
            latest_sma = float(sma_df.iloc[-1]["sma_30"])
            sma_delta = (
                ((latest_price - latest_sma) / latest_sma) * 100 if latest_sma else 0.0
            )

            # Forecast (optional — skip if prophet not installed)
            forecast_df = pd.DataFrame()
            try:
                from src.ml.training.forecasting import ProphetForecaster

                forecaster = ProphetForecaster(ticker=ticker)
                forecaster.fit(price_df)
                forecast_df = forecaster.forecast(periods=30)
            except Exception as exc:
                log.warning("Forecasting skipped: %s", exc)

            kpi_close = kpi_card(
                "Latest Close", f"${latest_price:.2f}", icon="bi-tag-fill"
            )
            kpi_ret = kpi_card(
                "Day Change", f"{day_change:.2f}%", delta=day_change, icon="bi-graph-up"
            )
            kpi_vol = kpi_card(
                "30d Volatility", f"{latest_vol:.2f}%", icon="bi-activity"
            )
            kpi_sma = kpi_card(
                "30d SMA",
                f"${latest_sma:.2f}",
                delta=sma_delta,
                icon="bi-arrow-trend-up",
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=price_df["price_date"],
                    y=price_df["close_price"],
                    mode="lines",
                    name="Historical Price",
                    line=dict(color="#2563eb", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.08)",
                )
            )

            if not forecast_df.empty:
                future = forecast_df[forecast_df["is_forecast"] == True]
                if not future.empty:
                    first_forecast_date = future["date"].iloc[0]
                    fig.add_vline(
                        x=first_forecast_date,
                        line_width=2,
                        line_dash="dash",
                        line_color="#475569",
                        annotation_text="Forecast Start",
                        annotation_position="top left",
                        annotation_font=dict(
                            color="#0f172a", size=11, family="Inter, sans-serif"
                        ),
                    )

                fig.add_trace(
                    go.Scatter(
                        x=future["date"],
                        y=future["yhat"],
                        mode="lines",
                        name="30d Forecast",
                        line=dict(color="#7c3aed", width=2.5, dash="dash"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([future["date"], future["date"][::-1]]),
                        y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
                        fill="toself",
                        fillcolor="rgba(124,58,237,0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Confidence Band",
                        showlegend=True,
                    )
                )

            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
                margin=dict(l=60, r=20, t=30, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#0f172a", size=12),
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#e2e8f0",
                    tickfont=dict(color="#0f172a", size=12),
                    title=dict(font=dict(color="#0f172a", size=13)),
                    linecolor="#cbd5e1",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#e2e8f0",
                    tickfont=dict(color="#0f172a", size=12),
                    title=dict(font=dict(color="#0f172a", size=13)),
                    linecolor="#cbd5e1",
                ),
                hovermode="x unified",
            )
            return kpi_close, kpi_ret, kpi_vol, kpi_sma, fig

        except Exception as exc:
            log.error("Stock callback error: %s", exc, exc_info=True)
            empty_kpi = kpi_card("Error", "-", icon="bi-exclamation-triangle")
            empty_fig = go.Figure().update_layout(
                title=dict(text=str(exc), font=dict(color="#0f172a")),
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                margin=dict(l=60, r=20, t=30, b=40),
            )
            return empty_kpi, empty_kpi, empty_kpi, empty_kpi, empty_fig
