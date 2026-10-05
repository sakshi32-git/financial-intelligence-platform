"""
logging_config.py
=================
Structured logging setup for the Financial Intelligence Platform.

Supports two output formats controlled by the ``LOG_FORMAT`` environment
variable:

* ``json``  — machine-readable JSON lines (default; ideal for production /
              log aggregators such as Datadog, Loki, Splunk).
* ``text``  — human-readable coloured output (ideal for local development).

Usage
-----
    from logging_config import configure_logging

    configure_logging()          # call once at application entry-point
    import logging
    log = logging.getLogger(__name__)
    log.info("Platform started", extra={"phase": "startup"})

In any other module::

    import logging
    log = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

import structlog

from config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_log_dir(log_dir: Path) -> None:
    """Create the log directory if it does not already exist."""
    log_dir.mkdir(parents=True, exist_ok=True)


def _shared_processors() -> list[Any]:
    """Return processors applied to every log record regardless of format."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def _build_stdlib_logging_config(log_dir: Path, level: str) -> dict[str, Any]:
    """
    Build a ``logging.config.dictConfig``-compatible dictionary.

    Two handlers are always registered:
    * ``console`` — writes to *stderr*.
    * ``file``    — writes to ``<log_dir>/platform.log`` with daily rotation.
    """
    use_json: bool = settings.logging.format == "json"

    formatter_id = "json" if use_json else "text"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "text": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": formatter_id,
                "level": level,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_dir / "platform.log"),
                "mode": "a",
                "encoding": "utf-8",
                "formatter": formatter_id,
                "level": level,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
        "loggers": {
            # Silence overly verbose third-party libraries.
            "urllib3": {"level": "WARNING", "propagate": True},
            "httpx": {"level": "WARNING", "propagate": True},
            "sqlalchemy.engine": {"level": "WARNING", "propagate": True},
            "alembic": {"level": "INFO", "propagate": True},
            "mlflow": {"level": "WARNING", "propagate": True},
        },
    }


def _configure_structlog(use_json: bool) -> None:
    """Wire up structlog to delegate to the stdlib logging system."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """
    Initialise the logging stack.

    Must be called **once** at application start-up, before any other
    module emits log records.  Subsequent calls are safe but have no
    additional effect.
    """
    log_dir: Path = settings.logging.dir
    level: str = settings.logging.level
    use_json: bool = settings.logging.format == "json"

    _ensure_log_dir(log_dir)

    stdlib_config = _build_stdlib_logging_config(log_dir, level)
    logging.config.dictConfig(stdlib_config)

    _configure_structlog(use_json)

    # Emit a first record to confirm logging is active.
    log = logging.getLogger(__name__)
    log.info(
        "Logging initialised",
        extra={
            "level": level,
            "format": settings.logging.format,
            "log_dir": str(log_dir),
            "app_env": settings.app.env,
        },
    )
