"""
database.py
===========
SQLAlchemy 2.x models and connection management for the Financial
Intelligence Platform.

Exports
-------
Base          — declarative base shared by all models
engine        — configured Engine bound to settings.database.url
SessionLocal  — session factory (use as a context manager)
get_session   — dependency-injection helper that yields a Session
Company       — publicly listed company master data
StockPrice    — daily OHLCV price record per company
Commodity     — daily commodity price (energy, metals, agriculture)

Usage
-----
    from src.database.database import get_session, Company, StockPrice

    with get_session() as session:
        companies = session.query(Company).filter_by(is_active=True).all()
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Generator, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

import logging

log = logging.getLogger(__name__)


# ===========================================================================
# Base
# ===========================================================================


class Base(DeclarativeBase):
    """
    Declarative base for all ORM models.

    Provides two audit columns (``created_at``, ``updated_at``) that every
    table inherits automatically.
    """

    # Shared audit columns — automatically present on every subclass.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
        doc="UTC timestamp when the row was first inserted.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=text("NOW()"),
        doc="UTC timestamp of the last UPDATE to this row.",
    )


# ===========================================================================
# Engine & Session
# ===========================================================================


def _build_engine():
    """Construct and return a configured SQLAlchemy Engine."""
    db = settings.database
    url = db.url

    kwargs = {
        "pool_pre_ping": True,
        "echo": settings.app.env == "development",
        "future": True,
    }

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = db.pool_size
        kwargs["max_overflow"] = db.max_overflow
        kwargs["pool_recycle"] = 1800

    engine = create_engine(url, **kwargs)

    if not url.startswith("sqlite"):
        # Enforce UTC for every new PostgreSQL connection.
        @event.listens_for(engine, "connect")
        def _set_utc(dbapi_connection, _connection_record):
            try:
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET TIME ZONE 'UTC';")
            except Exception:
                pass

    log.info(
        "Database engine created",
        extra={
            "url": url,
        },
    )
    return engine


engine = _build_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context-manager that provides a transactional database session.

    Commits on clean exit; rolls back and re-raises on any exception.

    Usage::

        with get_session() as session:
            session.add(some_model_instance)
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        log.exception("Session rolled back due to unhandled exception.")
        raise
    finally:
        session.close()


# ===========================================================================
# Models
# ===========================================================================


class Company(Base):
    """
    Master record for a publicly listed company.

    One company can have many :class:`StockPrice` records.
    """

    __tablename__ = "companies"

    __table_args__ = (
        UniqueConstraint("ticker", "exchange", name="uq_company_ticker_exchange"),
        CheckConstraint("LENGTH(ticker) >= 1", name="ck_company_ticker_nonempty"),
        CheckConstraint(
            "market_cap IS NULL OR market_cap >= 0",
            name="ck_company_market_cap_nonneg",
        ),
        {"comment": "Master list of publicly traded companies."},
    )

    # ------------------------------------------------------------------
    # Primary key
    # ------------------------------------------------------------------
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        doc="Surrogate primary key.",
    )

    # ------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------
    ticker: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,
        doc="Exchange ticker symbol (e.g. 'AAPL', 'MSFT').",
    )
    exchange: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Exchange where the security is listed (e.g. 'NASDAQ', 'NYSE').",
    )
    isin: Mapped[Optional[str]] = mapped_column(
        String(12),
        nullable=True,
        unique=True,
        doc="ISO 6166 International Securities Identification Number.",
    )
    cusip: Mapped[Optional[str]] = mapped_column(
        String(9),
        nullable=True,
        unique=True,
        doc="9-character CUSIP identifier (US/Canada).",
    )

    # ------------------------------------------------------------------
    # Descriptive
    # ------------------------------------------------------------------
    name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        doc="Full legal company name.",
    )
    sector: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="GICS sector (e.g. 'Technology', 'Healthcare').",
    )
    industry: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="GICS industry sub-classification.",
    )
    country: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Country of incorporation (ISO 3166-1 alpha-2 preferred).",
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="USD",
        server_default="USD",
        doc="Reporting currency (ISO 4217, e.g. 'USD', 'EUR').",
    )

    # ------------------------------------------------------------------
    # Financials (point-in-time snapshot; not historical)
    # ------------------------------------------------------------------
    market_cap: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=20, scale=2),
        nullable=True,
        doc="Market capitalisation in reporting currency.",
    )
    shares_outstanding: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        doc="Total shares outstanding.",
    )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("TRUE"),
        doc="False when the company is delisted or acquired.",
    )
    ipo_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        doc="Date the company first listed on a public exchange.",
    )
    delisted_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        doc="Date the company was delisted, if applicable.",
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    stock_prices: Mapped[list[StockPrice]] = relationship(
        "StockPrice",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
        doc="All OHLCV price records for this company.",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Company id={self.id} ticker={self.ticker!r} exchange={self.exchange!r}>"


# ---------------------------------------------------------------------------


class StockPrice(Base):
    """
    Daily OHLCV (Open / High / Low / Close / Volume) price record.

    One record per company per trading day.  Adjusted prices account for
    dividends and stock splits.
    """

    __tablename__ = "stock_prices"

    __table_args__ = (
        UniqueConstraint(
            "company_id", "price_date", name="uq_stockprice_company_date"
        ),
        Index("ix_stockprice_price_date", "price_date"),
        Index("ix_stockprice_company_date", "company_id", "price_date"),
        CheckConstraint("open_price  > 0", name="ck_stockprice_open_pos"),
        CheckConstraint("high_price  > 0", name="ck_stockprice_high_pos"),
        CheckConstraint("low_price   > 0", name="ck_stockprice_low_pos"),
        CheckConstraint("close_price > 0", name="ck_stockprice_close_pos"),
        CheckConstraint("high_price >= low_price", name="ck_stockprice_high_gte_low"),
        CheckConstraint("volume >= 0", name="ck_stockprice_volume_nonneg"),
        {"comment": "Daily OHLCV price series per company."},
    )

    # ------------------------------------------------------------------
    # Primary key
    # ------------------------------------------------------------------
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        doc="Surrogate primary key.",
    )

    # ------------------------------------------------------------------
    # Foreign key
    # ------------------------------------------------------------------
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="FK → companies.id",
    )

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    price_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        doc="Trading date (market local date, usually yyyy-mm-dd).",
    )

    # ------------------------------------------------------------------
    # OHLCV — raw prices
    # ------------------------------------------------------------------
    open_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
        doc="Opening price for the session.",
    )
    high_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
        doc="Intraday high price for the session.",
    )
    low_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
        doc="Intraday low price for the session.",
    )
    close_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
        doc="Closing price for the session.",
    )
    volume: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        doc="Total shares traded during the session.",
    )

    # ------------------------------------------------------------------
    # Adjusted prices (split- and dividend-adjusted)
    # ------------------------------------------------------------------
    adj_open: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Split/dividend-adjusted open price.",
    )
    adj_high: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Split/dividend-adjusted high price.",
    )
    adj_low: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Split/dividend-adjusted low price.",
    )
    adj_close: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Split/dividend-adjusted close price.",
    )
    adj_volume: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        doc="Split-adjusted volume.",
    )

    # ------------------------------------------------------------------
    # Supplementary
    # ------------------------------------------------------------------
    vwap: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Volume-weighted average price for the session.",
    )
    transactions: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of trades executed during the session.",
    )
    data_source: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Provider that supplied this record (e.g. 'polygon', 'alpha_vantage').",
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    company: Mapped[Company] = relationship(
        "Company",
        back_populates="stock_prices",
        doc="Parent company for this price record.",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<StockPrice id={self.id} company_id={self.company_id} "
            f"date={self.price_date} close={self.close_price}>"
        )


# ---------------------------------------------------------------------------


class Commodity(Base):
    """
    Daily price record for a traded commodity.

    Covers energy (crude oil, natural gas), metals (gold, silver, copper),
    and agricultural products (wheat, corn, soybeans).
    """

    __tablename__ = "commodities"

    __table_args__ = (
        UniqueConstraint(
            "symbol", "price_date", name="uq_commodity_symbol_date"
        ),
        Index("ix_commodity_symbol", "symbol"),
        Index("ix_commodity_price_date", "price_date"),
        Index("ix_commodity_symbol_date", "symbol", "price_date"),
        CheckConstraint("close_price > 0", name="ck_commodity_close_pos"),
        CheckConstraint(
            "open_price  IS NULL OR open_price  > 0",
            name="ck_commodity_open_pos",
        ),
        CheckConstraint(
            "high_price  IS NULL OR high_price  > 0",
            name="ck_commodity_high_pos",
        ),
        CheckConstraint(
            "low_price   IS NULL OR low_price   > 0",
            name="ck_commodity_low_pos",
        ),
        CheckConstraint(
            "high_price IS NULL OR low_price IS NULL OR high_price >= low_price",
            name="ck_commodity_high_gte_low",
        ),
        {"comment": "Daily price series for traded commodities."},
    )

    # ------------------------------------------------------------------
    # Primary key
    # ------------------------------------------------------------------
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        doc="Surrogate primary key.",
    )

    # ------------------------------------------------------------------
    # Identifier
    # ------------------------------------------------------------------
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc=(
            "Commodity ticker or code "
            "(e.g. 'CL' crude oil, 'GC' gold, 'ZW' wheat, 'NG' natural gas)."
        ),
    )
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Human-readable commodity name (e.g. 'WTI Crude Oil').",
    )
    category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Commodity category: 'energy', 'metals', 'agriculture', 'livestock', 'softs'.",
    )
    unit: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Unit of measure for the price (e.g. 'USD/bbl', 'USD/troy oz', 'USc/bu').",
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="USD",
        server_default="USD",
        doc="ISO 4217 currency code for the quoted price.",
    )

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    price_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        doc="Date of the price observation.",
    )

    # ------------------------------------------------------------------
    # Price fields
    # ------------------------------------------------------------------
    open_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Opening price for the session.",
    )
    high_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Intraday high for the session.",
    )
    low_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Intraday low for the session.",
    )
    close_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=False,
        doc="Closing / settlement price — always required.",
    )
    settlement_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=6),
        nullable=True,
        doc="Official exchange settlement price (futures markets).",
    )

    # ------------------------------------------------------------------
    # Volume & open interest
    # ------------------------------------------------------------------
    volume: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        doc="Contracts or units traded during the session.",
    )
    open_interest: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        doc="Number of outstanding futures/options contracts (futures markets).",
    )

    # ------------------------------------------------------------------
    # Supplementary
    # ------------------------------------------------------------------
    contract_month: Mapped[Optional[str]] = mapped_column(
        String(8),
        nullable=True,
        doc="Active futures contract month in 'YYYY-MM' format, if applicable.",
    )
    data_source: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Provider that supplied this record (e.g. 'eia', 'quandl', 'fred').",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Commodity id={self.id} symbol={self.symbol!r} "
            f"date={self.price_date} close={self.close_price}>"
        )
