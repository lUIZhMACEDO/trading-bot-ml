"""SQLite storage: the 13-table schema and a connection helper.

Each call opens its own connection. The auto-trader runs in a background thread and FastAPI
serves sync routes from a thread pool, and a sqlite3 connection cannot be shared across threads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL, date DATE NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume INTEGER, vwap REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS technicals (
    symbol TEXT NOT NULL, date DATE NOT NULL,
    rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL,
    bb_upper REAL, bb_mid REAL, bb_lower REAL, bb_width REAL, bb_pctb REAL,
    ema_9 REAL, ema_21 REAL, sma_20 REAL, sma_50 REAL, sma_200 REAL,
    atr REAL, adx REAL,
    hurst_exponent REAL,
    momentum_5 REAL, momentum_10 REAL, momentum_21 REAL,
    vol_ratio REAL, vol_zscore REAL,
    return_1d REAL, return_5d REAL, return_21d REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT NOT NULL, updated_at DATE NOT NULL,
    pe_ratio REAL, forward_pe REAL, peg_ratio REAL,
    pb_ratio REAL, ps_ratio REAL,
    fcf_yield REAL, roe REAL, roa REAL,
    debt_to_equity REAL, current_ratio REAL,
    revenue_growth REAL, earnings_growth REAL,
    dividend_yield REAL, market_cap REAL,
    analyst_rating REAL, target_price REAL,
    PRIMARY KEY (symbol)
);

CREATE TABLE IF NOT EXISTS macro (
    series_id TEXT NOT NULL, date DATE NOT NULL,
    value REAL, name TEXT,
    PRIMARY KEY (series_id, date)
);

-- One row per strategy per symbol per scan. signal is BUY/SELL/HOLD, score is -1.0 to +1.0.
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, date DATE NOT NULL,
    strategy TEXT NOT NULL,
    signal TEXT NOT NULL,
    score REAL,
    reason TEXT,
    confidence REAL
);

-- Sum of the five strategy scores (-5.0 to +5.0) and its rating.
CREATE TABLE IF NOT EXISTS composite_scores (
    symbol TEXT NOT NULL, date DATE NOT NULL,
    composite_score REAL,
    rating TEXT,
    buy_signals INTEGER, sell_signals INTEGER, hold_signals INTEGER,
    ml_prediction REAL, ml_confidence REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL, side TEXT NOT NULL,
    qty REAL NOT NULL, price REAL,
    order_type TEXT, strategy TEXT,
    status TEXT, alpaca_id TEXT,
    tx_cost REAL, slippage REAL,
    composite_score REAL, reason TEXT
);

CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT, accuracy REAL,
    sharpe_pred REAL, features TEXT,
    walk_forward_folds INTEGER,
    train_size INTEGER, test_size INTEGER
);

CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT, symbol TEXT,
    start_date DATE, end_date DATE,
    total_return REAL, sharpe_ratio REAL,
    max_drawdown REAL, win_rate REAL,
    total_trades INTEGER, avg_trade_pnl REAL,
    tx_costs_total REAL
);

CREATE TABLE IF NOT EXISTS private_exposure (
    company TEXT NOT NULL, date DATE NOT NULL,
    exposure_score REAL,
    weighted_return REAL,
    proxy_details TEXT,
    PRIMARY KEY (company, date)
);

CREATE TABLE IF NOT EXISTS position_sizing (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    kelly_fraction REAL,
    vol_adjusted_size REAL,
    final_qty INTEGER,
    portfolio_pct REAL,
    rationale TEXT
);

-- regime is bull / bear / sideways / high_vol
CREATE TABLE IF NOT EXISTS market_regime (
    date DATE PRIMARY KEY,
    regime TEXT,
    vix_level REAL,
    yield_curve REAL,
    breadth_pct REAL,
    regime_confidence REAL
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    date DATE PRIMARY KEY,
    equity REAL, cash REAL,
    positions_value REAL,
    daily_pnl REAL, daily_return REAL,
    cumulative_return REAL,
    sharpe_rolling_30d REAL,
    max_drawdown REAL,
    num_positions INTEGER
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_tech_symbol ON technicals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
"""

TABLES = (
    "ohlcv", "technicals", "fundamentals", "macro", "signals",
    "composite_scores", "trades", "ml_models", "backtests",
    "private_exposure", "position_sizing", "market_regime", "portfolio_snapshots",
)  # fmt: skip


@contextmanager
def connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a WAL-mode connection, commit on success, and always close it."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str) -> None:
    """Create every table and index if they do not exist yet."""
    with connect(db_path) as conn:
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
        conn.executescript(SCHEMA)


def table_counts(db_path: Path | str) -> dict[str, int]:
    """Row count per table, 0 for a table that has not been created."""
    counts = {}
    with connect(db_path) as conn:
        for table in TABLES:
            try:
                # Table names come from the TABLES constant above, never from user input.
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            except sqlite3.OperationalError:
                counts[table] = 0
    return counts
