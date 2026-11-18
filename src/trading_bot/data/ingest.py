"""Pull prices (Alpaca), fundamentals (yfinance) and macro series (FRED) into SQLite.

The network calls are kept thin. Turning provider responses into rows happens in small pure
functions (``bars_to_rows``, ``fundamentals_row``, ``macro_rows``) so they can be tested offline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from trading_bot.data.db import connect
from trading_bot.data.universe import FRED_SERIES, WATCHLIST

log = logging.getLogger(__name__)


def bars_to_rows(symbol: str, bars: pd.DataFrame) -> list[tuple]:
    """Convert an Alpaca bars frame (plain or (symbol, timestamp) MultiIndex) to ohlcv rows."""
    if bars.empty:
        return []
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
    elif "timestamp" not in bars.columns:
        bars = bars.reset_index().rename(columns={"index": "timestamp"})
    if "vwap" not in bars.columns:
        bars = bars.assign(vwap=0.0)
    bars = bars.assign(vwap=bars["vwap"].fillna(0.0))
    rows = []
    for r in bars.itertuples(index=False):
        rows.append(
            (
                symbol,
                str(r.timestamp)[:10],
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                int(r.volume),
                float(r.vwap),
            )
        )
    return rows


def ingest_ohlcv(data_client, db_path: Path | str, symbols: Iterable[str] | None = None, years: int = 2) -> int:
    """Fetch ``years`` of daily bars per symbol from Alpaca and upsert them. Returns rows written."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    symbols = list(symbols or WATCHLIST)
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    log.info("Ingesting %dy of OHLCV for %d symbols", years, len(symbols))

    total = 0
    with connect(db_path) as conn:
        for symbol in symbols:
            try:
                request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
                rows = bars_to_rows(symbol, data_client.get_stock_bars(request).df)
            except Exception as exc:
                log.error("%s: bar download failed: %s", symbol, exc)
                continue
            conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
            total += len(rows)
            log.info("%s: %d bars", symbol, len(rows))
    log.info("Ingested %d OHLCV rows", total)
    return total


# 18 columns, matching fundamentals_row below.
_FUNDAMENTALS_INSERT = "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"


def fcf_yield(info: Mapping) -> float:
    """Free cash flow as a percent of market cap, 0 when either number is missing."""
    fcf = info.get("freeCashflow") or 0
    market_cap = info.get("marketCap") or 1
    return fcf / max(market_cap, 1) * 100


def fundamentals_row(symbol: str, info: Mapping, today: str) -> tuple:
    """Map a yfinance ``Ticker.info`` dict onto the fundamentals table columns."""
    return (
        symbol,
        today,
        info.get("trailingPE"),
        info.get("forwardPE"),
        info.get("pegRatio"),
        info.get("priceToBook"),
        info.get("priceToSalesTrailing12Months"),
        fcf_yield(info),
        info.get("returnOnEquity"),
        info.get("returnOnAssets"),
        info.get("debtToEquity"),
        info.get("currentRatio"),
        info.get("revenueGrowth"),
        info.get("earningsGrowth"),
        info.get("dividendYield"),
        info.get("marketCap"),
        info.get("recommendationMean"),  # 1 = strong buy ... 5 = strong sell
        info.get("targetMeanPrice"),
    )


def ingest_fundamentals(db_path: Path | str, symbols: Iterable[str] | None = None) -> None:
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance is not installed, skipping fundamentals")
        return

    symbols = list(symbols or WATCHLIST)
    today = datetime.now().strftime("%Y-%m-%d")
    log.info("Fetching fundamentals for %d symbols", len(symbols))
    with connect(db_path) as conn:
        for symbol in symbols:
            try:
                row = fundamentals_row(symbol, yf.Ticker(symbol).info, today)
            except Exception as exc:
                log.error("%s: fundamentals failed: %s", symbol, exc)
                continue
            conn.execute(_FUNDAMENTALS_INSERT, row)


def macro_rows(series_id: str, name: str, series: pd.Series) -> list[tuple]:
    """FRED observations to macro rows, skipping missing values."""
    return [(series_id, str(date)[:10], float(value), name) for date, value in series.items() if pd.notna(value)]


def ingest_macro(db_path: Path | str, fred_api_key: str | None, years: int = 2) -> None:
    if not fred_api_key:
        log.warning("FRED_API_KEY is not set, skipping macro data (free key at fred.stlouisfed.org)")
        return
    try:
        from fredapi import Fred
    except ImportError:
        log.warning("fredapi is not installed, skipping macro data")
        return

    fred = Fred(api_key=fred_api_key)
    start = datetime.now() - timedelta(days=365 * years)
    log.info("Fetching %d FRED series", len(FRED_SERIES))
    with connect(db_path) as conn:
        for series_id, name in FRED_SERIES.items():
            try:
                rows = macro_rows(series_id, name, fred.get_series(series_id, start))
            except Exception as exc:
                log.error("FRED %s failed: %s", series_id, exc)
                continue
            conn.executemany("INSERT OR REPLACE INTO macro VALUES (?,?,?,?)", rows)
