"""
demo_data.py
============
Synthetic demo data generator for the Financial Intelligence Platform.

Used as a graceful fallback when the database is unavailable or empty,
so the dashboard always displays meaningful charts and KPIs.

All data is procedurally generated with realistic statistical properties
(geometric Brownian motion for prices).  No external API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Reproducible seed ─────────────────────────────────────────────────────────
_RNG = np.random.default_rng(42)


def _gbm_prices(
    start_price: float,
    days: int,
    mu: float = 0.0003,
    sigma: float = 0.015,
) -> np.ndarray:
    """Generate a synthetic price series via Geometric Brownian Motion."""
    dt = 1.0
    returns = _RNG.normal(mu * dt, sigma * np.sqrt(dt), days)
    prices = start_price * np.cumprod(1 + returns)
    return np.concatenate([[start_price], prices[:-1]])


def generate_stock_prices(
    ticker: str,
    days: int = 504,  # ~2 years of trading days
) -> pd.DataFrame:
    """Return a synthetic stock OHLCV DataFrame for ``ticker``."""
    _starts = {"AAPL": 150, "MSFT": 280, "NVDA": 400, "TSLA": 200,
               "AMZN": 130, "GOOGL": 120, "META": 300}
    start_price = _starts.get(ticker.upper(), 100.0)

    end_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end_date, periods=days)

    close = _gbm_prices(start_price, days)
    noise = _RNG.uniform(0.98, 1.02, days)
    high = close * _RNG.uniform(1.005, 1.025, days)
    low = close * _RNG.uniform(0.975, 0.995, days)
    open_ = close * noise
    volume = _RNG.integers(20_000_000, 80_000_000, days).astype(float)

    return pd.DataFrame({
        "ticker": ticker.upper(),
        "price_date": dates,
        "open_price": open_.round(2),
        "high_price": high.round(2),
        "low_price": low.round(2),
        "close_price": close.round(2),
        "volume": volume,
        "adj_close": close.round(2),
    })


def generate_commodity_prices(
    symbol: str,
    days: int = 504,
) -> pd.DataFrame:
    """Return a synthetic commodity OHLCV DataFrame for ``symbol``."""
    _starts = {"CL=F": 75.0, "GC=F": 1900.0, "SI=F": 24.0, "NG=F": 2.8}
    start_price = _starts.get(symbol.upper(), 100.0)

    end_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end_date, periods=days)

    close = _gbm_prices(start_price, days, mu=0.0001, sigma=0.018)
    high = close * _RNG.uniform(1.005, 1.03, days)
    low = close * _RNG.uniform(0.97, 0.995, days)
    open_ = close * _RNG.uniform(0.99, 1.01, days)
    volume = _RNG.integers(100_000, 500_000, days).astype(float)

    return pd.DataFrame({
        "ticker": symbol.upper(),
        "price_date": dates,
        "open_price": open_.round(2),
        "high_price": high.round(2),
        "low_price": low.round(2),
        "close_price": close.round(2),
        "volume": volume,
    })


def generate_stock_prices_multi(
    tickers: list[str],
    days: int = 252,  # 1 year
) -> pd.DataFrame:
    """Return synthetic OHLCV for multiple tickers concatenated."""
    frames = [generate_stock_prices(t, days) for t in tickers]
    return pd.concat(frames, ignore_index=True)


def generate_economic_series(series_id: str) -> pd.DataFrame:
    """Return a synthetic economic time series for ``series_id``."""
    _configs = {
        "GDP":      {"start": 20_000, "mu": 0.006,  "sigma": 0.008, "freq": "QS"},
        "UNRATE":   {"start": 4.5,    "mu": 0.0,    "sigma": 0.05,  "freq": "MS"},
        "CPIAUCSL": {"start": 280.0,  "mu": 0.003,  "sigma": 0.003, "freq": "MS"},
        "FEDFUNDS": {"start": 5.25,   "mu": -0.001, "sigma": 0.02,  "freq": "MS"},
    }
    cfg = _configs.get(series_id, {"start": 100.0, "mu": 0.002, "sigma": 0.01, "freq": "MS"})

    end_date = pd.Timestamp.now().normalize()
    dates = pd.date_range(end=end_date, periods=120, freq=cfg["freq"])
    n = len(dates)

    values = _gbm_prices(cfg["start"], n, mu=cfg["mu"], sigma=cfg["sigma"])

    return pd.DataFrame({
        "date": dates,
        "value": values.round(2),
        "series_id": series_id,
    })
