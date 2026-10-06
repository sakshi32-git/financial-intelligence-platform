"""
config.py
=========
Centralised settings for the Financial Intelligence Platform.

All values are read from environment variables (or a .env file in the
project root).  Import the singleton ``settings`` wherever configuration
is needed — never read ``os.environ`` directly in application code.

Usage
-----
    from config import settings

    db_url = settings.database.url
    log_level = settings.logging.level
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root — one level above this file
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ===========================================================================
# Sub-settings classes
# ===========================================================================


class AppSettings(BaseSettings):
    """General application metadata."""

    model_config = SettingsConfigDict(env_prefix="APP_")

    env: Literal["development", "staging", "production"] = "development"
    name: str = "financial_intelligence_platform"
    version: str = "1.0.0"
    secret_key: SecretStr = Field(default="change-me-to-a-long-random-string")


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and pool configuration."""

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        populate_by_name=True,
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "localhost"
    port: int = 5432
    db: str = "financial_platform"
    user: str = "fp_user"
    password: SecretStr = Field(default="change-me")
    pool_size: int = Field(default=10, ge=1, le=100, alias="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, ge=0, le=100, alias="DATABASE_MAX_OVERFLOW")
    database_url: str = Field(default="", alias="DATABASE_URL")

    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        """Synchronous SQLAlchemy connection URL."""
        if self.database_url:
            url_str = self.database_url
        elif self.host.startswith("postgresql://") or self.host.startswith("postgres://"):
            url_str = self.host
        else:
            pwd = self.password.get_secret_value()
            return (
                f"postgresql+psycopg2://{self.user}:{pwd}"
                f"@{self.host}:{self.port}/{self.db}?sslmode=require"
            )

        if url_str.startswith("postgresql://"):
            url_str = url_str.replace("postgresql://", "postgresql+psycopg2://", 1)
        elif url_str.startswith("postgres://"):
            url_str = url_str.replace("postgres://", "postgresql+psycopg2://", 1)
        return url_str

    @computed_field  # type: ignore[misc]
    @property
    def async_url(self) -> str:
        """Async SQLAlchemy connection URL (asyncpg driver)."""
        if self.database_url:
            url_str = self.database_url
        elif self.host.startswith("postgresql://") or self.host.startswith("postgres://"):
            url_str = self.host
        else:
            pwd = self.password.get_secret_value()
            return (
                f"postgresql+asyncpg://{self.user}:{pwd}"
                f"@{self.host}:{self.port}/{self.db}"
            )

        if url_str.startswith("postgresql://"):
            url_str = url_str.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url_str.startswith("postgres://"):
            url_str = url_str.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url_str.startswith("postgresql+psycopg2://"):
            url_str = url_str.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        return url_str


class APISettings(BaseSettings):
    """External financial data provider API keys."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    alpha_vantage_api_key: SecretStr = Field(
        default="your-alpha-vantage-key", alias="ALPHA_VANTAGE_API_KEY"
    )
    polygon_api_key: SecretStr = Field(
        default="your-polygon-key", alias="POLYGON_API_KEY"
    )
    yahoo_finance_api_key: SecretStr = Field(
        default="your-yahoo-finance-key", alias="YAHOO_FINANCE_API_KEY"
    )
    fred_api_key: SecretStr = Field(
        default="your-fred-key", alias="FRED_API_KEY"
    )


class ETLSettings(BaseSettings):
    """ETL pipeline behaviour."""

    model_config = SettingsConfigDict(env_prefix="ETL_")

    batch_size: int = Field(default=1000, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=5.0, ge=0)


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    dir: Path = Field(default=PROJECT_ROOT / "logs")


class MLSettings(BaseSettings):
    """Machine learning experiment tracking and model registry."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    model_registry_path: Path = Field(
        default=PROJECT_ROOT / "models", alias="MODEL_REGISTRY_PATH"
    )


class DashboardSettings(BaseSettings):
    """Plotly Dash server configuration."""

    model_config = SettingsConfigDict(env_prefix="DASHBOARD_")

    host: str = "0.0.0.0"
    port: int = Field(default=8050, ge=1, le=65535)
    debug: bool = False


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    password: SecretStr = Field(default="")

    @computed_field  # type: ignore[misc]
    @property
    def url(self) -> str:
        pwd = self.password.get_secret_value()
        auth = f":{pwd}@" if pwd else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


# ===========================================================================
# Root settings
# ===========================================================================


class Settings(BaseSettings):
    """
    Root settings object.

    Composes all sub-settings and loads values from the project-root
    ``.env`` file automatically.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    etl: ETLSettings = Field(default_factory=ETLSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    Using ``lru_cache`` ensures environment variables and the ``.env``
    file are parsed exactly once per process lifetime.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Module-level singleton — import this directly.
# ---------------------------------------------------------------------------
settings: Settings = get_settings()
