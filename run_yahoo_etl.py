"""
run_yahoo_etl.py
================
Loads stock and commodity price data from Yahoo Finance into the database.

Usage:
    .venv\Scripts\python run_yahoo_etl.py
"""

import logging
import sys
from datetime import date

from src.etl.pipelines.yahoo_finance import YahooFinancePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

# ── Stocks ────────────────────────────────────────────────────────────────────
STOCK_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "META",   # Meta
    "NVDA",   # NVIDIA
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "JNJ",    # Johnson & Johnson
]

# ── Commodities ───────────────────────────────────────────────────────────────
COMMODITY_TICKERS = [
    "CL=F",   # WTI Crude Oil
    "GC=F",   # Gold
    "SI=F",   # Silver
    "NG=F",   # Natural Gas
]

ALL_TICKERS = STOCK_TICKERS + COMMODITY_TICKERS

START_DATE = "2020-01-01"
END_DATE   = date.today().isoformat()

pipeline = YahooFinancePipeline(skip_on_error=True)

print(f"Running ETL for {len(ALL_TICKERS)} tickers from {START_DATE} to {END_DATE}...")
print("-" * 60)

results = pipeline.run_bulk(
    tickers=ALL_TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
)

# ── Summary ───────────────────────────────────────────────────────────────────
succeeded = [r for r in results if r.success]
failed    = [r for r in results if not r.success]

print("-" * 60)
print(f"Succeeded : {len(succeeded)}/{len(results)}")
print(f"Failed    : {len(failed)}/{len(results)}")
print(f"Rows loaded: {sum(r.total_rows_loaded for r in results)}")

if failed:
    print("\nFailed tickers:")
    for r in failed:
        print(f"  {r.ticker}: {r.errors}")
