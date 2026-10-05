"""ETL pipeline — extract, transform, load financial data."""

from src.etl.pipelines.yahoo_finance import (  # noqa: F401
    PipelineResult,
    YahooFinancePipeline,
)

__all__ = ["YahooFinancePipeline", "PipelineResult"]
