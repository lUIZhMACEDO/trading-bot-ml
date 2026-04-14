#!/usr/bin/env python3
"""
QUANTUM TRADER v2.0 — Alpaca Paper Trading Bot
================================================
Integrates techniques from Quant Research System v3.1.0:

  • 13-table normalized SQLite with WAL mode
  • FRED macro data (VIX, treasury yields, credit spreads)
  • Vectorized technicals (RSI, MACD, BB, Hurst exponent, momentum)
  • Walk-forward expanding-window ML with LightGBM + Ridge ensemble
  • 5 concurrent strategy signals → composite score → Strong Buy/Sell
  • Transaction cost modeling (0.05% per trade, next-open fills)
  • Private company synthetic exposure (Anthropic, SpaceX, etc.)
  • Fundamentals via yfinance (P/E, FCF yield, ROE)
  • Risk-adjusted position sizing (Kelly criterion + volatility scaling)

Setup:
  pip install alpaca-py python-dotenv pandas scikit-learn lightgbm yfinance fredapi ta
  cp .env.template .env   # paste your NEW Alpaca API keys
  python quantum_trader_v2.py

⚠️  NEVER hardcode API keys. Use .env file only.
"""

import os
import sys
import json
import time
import sqlite3
import threading
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ── Alpaca SDK ──
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── ML ──
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("QuantumTrader")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DB_PATH = Path("quantum_market.db")

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    "SPY", "QQQ", "AMD", "NFLX", "JPM", "V", "BA", "DIS",
    "CRM", "COST", "UNH", "HD", "PG", "INTC", "PYPL",
    "SQ", "COIN", "PLTR", "SOFI", "RIVN", "SNOW", "NET", "DDOG",
]

# Synthetic exposure for private companies via public proxy stocks
# Weights based on revenue exposure, partnership depth, and supply chain
PRIVATE_COMPANY_PROXIES = {
    "Anthropic": {"AMZN": 0.27, "GOOG": 0.10, "NVDA": 0.30, "MSFT": 0.08, "META": 0.05},
    "SpaceX":    {"BA": 0.15, "LMT": 0.12, "RTX": 0.10, "GOOG": 0.08, "TSLA": 0.20},
    "OpenAI":    {"MSFT": 0.40, "NVDA": 0.25, "GOOG": 0.05, "CRM": 0.05},
    "Stripe":    {"V": 0.20, "MA": 0.15, "SQ": 0.15, "PYPL": 0.10, "AMZN": 0.08},
    "xAI":       {"TSLA": 0.35, "NVDA": 0.25, "GOOG": 0.10, "AMD": 0.08},
}

# FRED macro series to track
FRED_SERIES = {
    "DGS2": "treasury_2y",
    "DGS10": "treasury_10y",
    "T10Y2Y": "yield_curve_spread",
    "VIXCLS": "vix",
    "BAMLH0A0HYM2": "hy_credit_spread",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds_rate",
    "DCOILWTICO": "crude_oil",
    "DTWEXBGS": "usd_index",
}

# Transaction cost model
TX_COST_BPS = 5  # 0.05% per trade (5 basis points)
SLIPPAGE_BPS = 2  # 0.02% slippage estimate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATABASE — 13-table normalized SQLite with WAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_db():
    """Create normalized SQLite database with WAL mode for concurrent safety."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    c = conn.cursor()

    # 1. Price data (OHLCV)
    c.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL, date DATE NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER, vwap REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    # 2. Computed technical indicators (vectorized, single-pass)
    c.execute("""
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
        )
    """)

    # 3. Fundamentals
    c.execute("""
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
        )
    """)

    # 4. Macro data (FRED)
    c.execute("""
        CREATE TABLE IF NOT EXISTS macro (
            series_id TEXT NOT NULL, date DATE NOT NULL,
            value REAL, name TEXT,
            PRIMARY KEY (series_id, date)
        )
    """)

    # 5. Strategy signals (5 concurrent strategies per ticker)
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, date DATE NOT NULL,
            strategy TEXT NOT NULL,
            signal TEXT NOT NULL,  -- BUY/SELL/HOLD
            score REAL,           -- -1.0 to +1.0
            reason TEXT,
            confidence REAL
        )
    """)

    # 6. Composite scores (aggregated from all strategies)
    c.execute("""
        CREATE TABLE IF NOT EXISTS composite_scores (
            symbol TEXT NOT NULL, date DATE NOT NULL,
            composite_score REAL,   -- -5.0 to +5.0
            rating TEXT,            -- Strong Buy / Buy / Neutral / Sell / Strong Sell
            buy_signals INTEGER, sell_signals INTEGER, hold_signals INTEGER,
            ml_prediction REAL, ml_confidence REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    # 7. Trade log
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL, side TEXT NOT NULL,
            qty REAL NOT NULL, price REAL,
            order_type TEXT, strategy TEXT,
            status TEXT, alpaca_id TEXT,
            tx_cost REAL, slippage REAL,
            composite_score REAL, reason TEXT
        )
    """)

    # 8. ML models metadata
    c.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT, accuracy REAL,
            sharpe_pred REAL, features TEXT,
            walk_forward_folds INTEGER,
            train_size INTEGER, test_size INTEGER
        )
    """)

    # 9. Backtest results
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT, symbol TEXT,
            start_date DATE, end_date DATE,
            total_return REAL, sharpe_ratio REAL,
            max_drawdown REAL, win_rate REAL,
            total_trades INTEGER, avg_trade_pnl REAL,
            tx_costs_total REAL
        )
    """)

    # 10. Private company proxy exposure
    c.execute("""
        CREATE TABLE IF NOT EXISTS private_exposure (
            company TEXT NOT NULL, date DATE NOT NULL,
            exposure_score REAL,
            weighted_return REAL,
            proxy_details TEXT,
            PRIMARY KEY (company, date)
        )
    """)

    # 11. Position sizing history
    c.execute("""
        CREATE TABLE IF NOT EXISTS position_sizing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            kelly_fraction REAL,
            vol_adjusted_size REAL,
            final_qty INTEGER,
            portfolio_pct REAL,
            rationale TEXT
        )
    """)

    # 12. Market regime
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_regime (
            date DATE PRIMARY KEY,
            regime TEXT,          -- bull / bear / sideways / high_vol
            vix_level REAL,
            yield_curve REAL,
            breadth_pct REAL,
            regime_confidence REAL
        )
    """)

    # 13. Daily portfolio snapshots
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            date DATE PRIMARY KEY,
            equity REAL, cash REAL,
            positions_value REAL,
            daily_pnl REAL, daily_return REAL,
            cumulative_return REAL,
            sharpe_rolling_30d REAL,
            max_drawdown REAL,
            num_positions INTEGER
        )
    """)

    # Indexes for fast queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tech_symbol ON technicals(symbol)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")

    conn.commit()
    conn.close()
    log.info(f"📦 Database initialized: {DB_PATH} (13 tables, WAL mode)")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALPACA CLIENT (unchanged from v1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENV_PATH = Path(".env")


def setup_api_keys():
    """Interactive first-time setup: prompts for API keys and saves to .env"""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    # Check if keys already exist and are valid (not placeholder)
    if api_key and secret_key and api_key != "your_new_key_here" and len(api_key) > 10:
        return api_key, secret_key

    # ── First-time setup ──
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │         FIRST-TIME SETUP — API Keys             │")
    print("  │                                                 │")
    print("  │  1. Go to https://app.alpaca.markets            │")
    print("  │  2. Switch to Paper Trading (top-left)          │")
    print("  │  3. Go to API Keys → Generate                  │")
    print("  │  4. Paste your Key and Secret below             │")
    print("  │                                                 │")
    print("  │  Keys are saved locally in .env (never shared)  │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    while True:
        api_key = input("  Alpaca API Key:    ").strip()
        if not api_key or len(api_key) < 10:
            print("  ⚠️  That doesn't look right. Key should start with PK... and be ~20 chars.")
            continue
        break

    while True:
        secret_key = input("  Alpaca Secret Key: ").strip()
        if not secret_key or len(secret_key) < 20:
            print("  ⚠️  Secret key should be ~40 characters long.")
            continue
        break

    # Optional: FRED key
    print()
    fred_key = input("  FRED API Key (optional, press Enter to skip): ").strip()

    # Optional: trading config
    print()
    print("  ── Trading Config (press Enter for defaults) ──")
    budget_input = input("  Max budget per auto-trade [$10000]: ").strip()
    budget = budget_input if budget_input else "10000"
    max_pos_input = input("  Max simultaneous positions [3]: ").strip()
    max_pos = max_pos_input if max_pos_input else "3"
    sl_input = input("  Stop loss % [2.0]: ").strip()
    sl = sl_input if sl_input else "2.0"
    tp_input = input("  Take profit % [5.0]: ").strip()
    tp = tp_input if tp_input else "5.0"

    # Write .env file
    env_content = f"""# Quantum Trader v2.0 — Auto-generated config
# Delete this file and restart to re-enter keys

ALPACA_API_KEY={api_key}
ALPACA_SECRET_KEY={secret_key}
ALPACA_BASE_URL=https://paper-api.alpaca.markets

FRED_API_KEY={fred_key if fred_key else 'your_fred_key_here'}

MAX_BUDGET={budget}
MAX_POSITIONS={max_pos}
STOP_LOSS_PCT={sl}
TAKE_PROFIT_PCT={tp}
"""
    ENV_PATH.write_text(env_content)
    print(f"\n  ✅ Keys saved to {ENV_PATH.resolve()}")
    print(f"  ⚠️  This file contains secrets — never share or commit it.\n")

    # Reload env
    load_dotenv(override=True)
    return api_key, secret_key


class AlpacaClient:
    def __init__(self):
        api_key, secret_key = setup_api_keys()
        self.api_key = api_key
        self.secret_key = secret_key

        # Test connection
        try:
            self.trading = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=True)
            self.data = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
            # Verify by fetching account
            acct = self.trading.get_account()
            log.info(f"✅ Connected to Alpaca Paper Trading (${float(acct.equity):,.2f})")
        except Exception as e:
            log.error(f"❌ Connection failed: {e}")
            log.error("   Your API keys may be invalid. Delete .env and restart to re-enter.")
            # Delete bad .env so next run prompts again
            if ENV_PATH.exists():
                remove = input("  Delete .env and try again? (y/n): ").strip().lower()
                if remove == "y":
                    ENV_PATH.unlink()
                    print("  .env deleted. Restart the bot to enter new keys.")
            sys.exit(1)

    def get_account(self):
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity), "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "pnl": float(acct.equity) - 100000,
        }

    def get_positions(self):
        positions = self.trading.get_all_positions()
        return [{
            "symbol": p.symbol, "qty": float(p.qty),
            "avg_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "pnl": float(p.unrealized_pl),
            "pnl_pct": float(p.unrealized_plpc) * 100,
            "market_value": float(p.market_value),
        } for p in positions]

    def buy(self, symbol, qty=None, notional=None, order_type="market", limit_price=None):
        return self._place_order(symbol, OrderSide.BUY, qty, notional, order_type, limit_price)

    def sell(self, symbol, qty=None, notional=None, order_type="market", limit_price=None):
        return self._place_order(symbol, OrderSide.SELL, qty, notional, order_type, limit_price)

    def _place_order(self, symbol, side, qty, notional, order_type, limit_price):
        try:
            if order_type == "limit" and limit_price:
                od = LimitOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY, limit_price=float(limit_price))
            elif notional:
                od = MarketOrderRequest(symbol=symbol, notional=float(notional), side=side, time_in_force=TimeInForce.DAY)
            else:
                od = MarketOrderRequest(symbol=symbol, qty=float(qty), side=side, time_in_force=TimeInForce.DAY)

            order = self.trading.submit_order(order_data=od)
            emoji = "🟢 BUY" if side == OrderSide.BUY else "🔴 SELL"
            log.info(f"{emoji} {qty or notional} {symbol} @ {order_type} → {order.status}")

            # Log with transaction cost modeling
            price = float(limit_price or 0)
            tx_cost = (price * float(qty or 0)) * TX_COST_BPS / 10000
            slippage = (price * float(qty or 0)) * SLIPPAGE_BPS / 10000

            conn = get_db()
            conn.execute(
                """INSERT INTO trades (symbol, side, qty, price, order_type, strategy, status,
                   alpaca_id, tx_cost, slippage) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (symbol, side.value, qty or 0, price, order_type, "manual",
                 str(order.status), str(order.id), tx_cost, slippage),
            )
            conn.commit()
            conn.close()
            return {"id": str(order.id), "symbol": symbol, "side": side.value,
                    "qty": qty, "status": str(order.status), "tx_cost": tx_cost}
        except Exception as e:
            log.error(f"Order failed: {e}")
            return {"error": str(e)}

    def close_position(self, symbol):
        try:
            self.trading.close_position(symbol)
            log.info(f"🔒 Closed position: {symbol}")
            return {"status": "closed", "symbol": symbol}
        except Exception as e:
            return {"error": str(e)}

    def close_all(self):
        self.trading.close_all_positions(cancel_orders=True)
        log.info("🔒 Closed ALL positions")

    def get_orders(self, status="all", limit=20):
        req = GetOrdersRequest(status=QueryOrderStatus.ALL if status == "all" else QueryOrderStatus.OPEN, limit=limit)
        return [{
            "id": str(o.id), "symbol": o.symbol, "side": str(o.side),
            "qty": float(o.qty) if o.qty else 0, "type": str(o.type),
            "status": str(o.status),
            "filled_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "created": str(o.created_at),
        } for o in self.trading.get_orders(filter=req)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA INGEST PIPELINE (from Quant Research System)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ingest_ohlcv(client: AlpacaClient, symbols=None, years=2):
    """Fetch multi-year OHLCV data via Alpaca and store in SQLite."""
    symbols = symbols or WATCHLIST
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    log.info(f"📥 Ingesting {years}yr OHLCV for {len(symbols)} symbols...")

    conn = get_db()
    total = 0
    for symbol in symbols:
        try:
            bars = client.data.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end,
            )).df
            if bars.empty:
                continue
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index()
            for _, r in bars.iterrows():
                d = str(r.get("timestamp", r.name))[:10]
                conn.execute(
                    "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)",
                    (symbol, d, float(r["open"]), float(r["high"]), float(r["low"]),
                     float(r["close"]), int(r["volume"]), float(r.get("vwap", 0))),
                )
                total += 1
            log.info(f"  ✅ {symbol}: {len(bars)} bars")
        except Exception as e:
            log.error(f"  ❌ {symbol}: {e}")
    conn.commit()
    conn.close()
    log.info(f"📦 Ingested {total} total OHLCV rows")
    return total


def ingest_fundamentals(symbols=None):
    """Fetch fundamentals via yfinance: P/E, FCF yield, ROE, analyst ratings."""
    if not HAS_YF:
        log.warning("yfinance not installed — skipping fundamentals")
        return
    symbols = symbols or WATCHLIST
    log.info(f"📊 Fetching fundamentals for {len(symbols)} symbols...")
    conn = get_db()
    today = datetime.now().strftime("%Y-%m-%d")

    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            conn.execute(
                """INSERT OR REPLACE INTO fundamentals VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol, today,
                    info.get("trailingPE"), info.get("forwardPE"), info.get("pegRatio"),
                    info.get("priceToBook"), info.get("priceToSalesTrailing12Months"),
                    # FCF Yield = FCF / Market Cap
                    (info.get("freeCashflow", 0) or 0) / max(info.get("marketCap", 1), 1) * 100,
                    info.get("returnOnEquity"), info.get("returnOnAssets"),
                    info.get("debtToEquity"), info.get("currentRatio"),
                    info.get("revenueGrowth"), info.get("earningsGrowth"),
                    info.get("dividendYield"), info.get("marketCap"),
                    info.get("recommendationMean"),  # 1=Strong Buy, 5=Strong Sell
                    info.get("targetMeanPrice"),
                ),
            )
        except Exception as e:
            log.error(f"  ❌ {symbol} fundamentals: {e}")
    conn.commit()
    conn.close()
    log.info("  ✅ Fundamentals stored")


def ingest_macro():
    """Fetch FRED macro series: VIX, yields, credit spreads, CPI, etc."""
    fred_key = os.getenv("FRED_API_KEY")
    if not HAS_FRED or not fred_key:
        log.warning("FRED not configured — skipping macro data. Set FRED_API_KEY in .env (free at fred.stlouisfed.org)")
        return
    fred = Fred(api_key=fred_key)
    conn = get_db()
    start = datetime.now() - timedelta(days=365 * 2)

    log.info(f"🏛️ Fetching {len(FRED_SERIES)} FRED macro series...")
    for series_id, name in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, start)
            for date, value in data.items():
                if pd.notna(value):
                    conn.execute(
                        "INSERT OR REPLACE INTO macro VALUES (?,?,?,?)",
                        (series_id, str(date)[:10], float(value), name),
                    )
        except Exception as e:
            log.error(f"  ❌ FRED {series_id}: {e}")
    conn.commit()
    conn.close()
    log.info("  ✅ Macro data stored")


def compute_private_exposure():
    """Compute synthetic exposure scores for private companies using proxy stocks."""
    conn = get_db()
    today = datetime.now().strftime("%Y-%m-%d")

    for company, proxies in PRIVATE_COMPANY_PROXIES.items():
        weighted_return = 0
        details = {}
        for proxy_symbol, weight in proxies.items():
            row = conn.execute(
                "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 2",
                (proxy_symbol,)
            ).fetchall()
            if len(row) >= 2:
                ret = (row[0][0] - row[1][0]) / row[1][0]
                weighted_return += ret * weight
                details[proxy_symbol] = {"weight": weight, "return": round(ret * 100, 2)}

        exposure_score = weighted_return * 100
        conn.execute(
            "INSERT OR REPLACE INTO private_exposure VALUES (?,?,?,?,?)",
            (company, today, round(exposure_score, 4), round(weighted_return, 6), json.dumps(details)),
        )
    conn.commit()
    conn.close()
    log.info("  ✅ Private company exposure computed")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VECTORIZED TECHNICAL INDICATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_hurst(series, max_lag=100):
    """Hurst exponent: <0.5 = mean reverting, 0.5 = random walk, >0.5 = trending."""
    if len(series) < max_lag:
        return 0.5
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    for lag in lags:
        pp = np.subtract(series[lag:].values, series[:-lag].values)
        tau.append(np.sqrt(np.std(pp)))
    try:
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return max(0, min(1, reg[0]))
    except Exception:
        return 0.5


def vectorized_technicals(df):
    """Single-pass vectorized computation of all technical indicators.
    Much faster than row-by-row — computes entire series at once."""
    if len(df) < 30:
        return df

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # ── RSI (Wilder's smoothing) ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── Moving Averages ──
    df["ema_9"] = close.ewm(span=9).mean()
    df["ema_21"] = close.ewm(span=21).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    # ── MACD ──
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ──
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = bb_range / df["bb_mid"]
    df["bb_pctb"] = (close - df["bb_lower"]) / bb_range.replace(0, 1e-10)

    # ── ATR (Average True Range) ──
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ── ADX (Average Directional Index, simplified) ──
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df["atr"].replace(0, 1e-10)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10) * 100
    df["adx"] = dx.rolling(14).mean()

    # ── Hurst Exponent (rolling 100-period) ──
    hurst_values = []
    for i in range(len(close)):
        if i < 100:
            hurst_values.append(0.5)
        else:
            hurst_values.append(compute_hurst(close.iloc[i-100:i]))
    df["hurst_exponent"] = hurst_values

    # ── Momentum ──
    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)
    df["momentum_21"] = close.pct_change(21)

    # ── Volume indicators ──
    vol_sma = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_sma.replace(0, 1e-10)
    vol_std = volume.rolling(20).std()
    df["vol_zscore"] = (volume - vol_sma) / vol_std.replace(0, 1e-10)

    # ── Returns ──
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_21d"] = close.pct_change(21)

    return df


def compute_and_store_technicals(symbols=None):
    """Compute technicals for all symbols and store in SQLite."""
    symbols = symbols or WATCHLIST
    conn = get_db()
    log.info(f"📐 Computing vectorized technicals for {len(symbols)} symbols...")

    for symbol in symbols:
        df = pd.read_sql_query(
            "SELECT * FROM ohlcv WHERE symbol=? ORDER BY date", conn, params=(symbol,)
        )
        if len(df) < 30:
            continue

        df = vectorized_technicals(df)
        tech_cols = [
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pctb",
            "ema_9", "ema_21", "sma_20", "sma_50", "sma_200",
            "atr", "adx", "hurst_exponent",
            "momentum_5", "momentum_10", "momentum_21",
            "vol_ratio", "vol_zscore",
            "return_1d", "return_5d", "return_21d",
        ]
        for _, row in df.dropna(subset=["rsi"]).iterrows():
            vals = [row.get(c) for c in tech_cols]
            conn.execute(
                f"INSERT OR REPLACE INTO technicals (symbol, date, {', '.join(tech_cols)}) VALUES (?,?,{','.join(['?']*len(tech_cols))})",
                (symbol, row["date"], *[float(v) if pd.notna(v) else None for v in vals]),
            )
    conn.commit()
    conn.close()
    log.info("  ✅ Technicals stored")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONCURRENT STRATEGY ENGINE (5 strategies per ticker)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_rsi_macd(df):
    """Strategy 1: RSI + MACD confluence (quick trades)."""
    if len(df) < 30:
        return "HOLD", 0, "Insufficient data"
    rsi = df["rsi"].iloc[-1]
    macd_h = df["macd_hist"].iloc[-1]
    vol_r = df["vol_ratio"].iloc[-1]

    if rsi < 30 and macd_h > 0:
        score = min(1.0, (30 - rsi) / 20 + 0.3)
        return "BUY", score, f"RSI={rsi:.0f} oversold + MACD bullish"
    elif rsi > 70 and macd_h < 0:
        score = min(1.0, (rsi - 70) / 20 + 0.3)
        return "SELL", -score, f"RSI={rsi:.0f} overbought + MACD bearish"
    elif rsi < 40 and macd_h > 0 and vol_r > 1.3:
        return "BUY", 0.4, f"RSI={rsi:.0f} + MACD bullish + high volume"
    return "HOLD", 0, f"RSI={rsi:.0f}, MACD_hist={macd_h:.3f}"


def strategy_bollinger_reversion(df):
    """Strategy 2: Bollinger Bands mean reversion."""
    if len(df) < 30:
        return "HOLD", 0, "Insufficient data"
    close = df["close"].iloc[-1]
    bb_pctb = df["bb_pctb"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    hurst = df["hurst_exponent"].iloc[-1]

    # Only trade mean reversion when Hurst < 0.5 (mean-reverting regime)
    if hurst >= 0.55:
        return "HOLD", 0, f"Hurst={hurst:.2f} — trending regime, skip reversion"

    if bb_pctb < 0.05 and rsi < 35:
        return "BUY", 0.8, f"Below lower BB (pctB={bb_pctb:.2f}), RSI={rsi:.0f}, Hurst={hurst:.2f}"
    elif bb_pctb > 0.95 and rsi > 65:
        return "SELL", -0.8, f"Above upper BB (pctB={bb_pctb:.2f}), RSI={rsi:.0f}"
    return "HOLD", 0, f"BB%B={bb_pctb:.2f}, Hurst={hurst:.2f}"


def strategy_golden_cross(df):
    """Strategy 3: Golden/Death Cross (SMA 50/200) — medium-term."""
    if len(df) < 210:
        return "HOLD", 0, "Need 200+ days of data"
    sma50 = df["sma_50"].iloc[-1]
    sma200 = df["sma_200"].iloc[-1]
    prev_sma50 = df["sma_50"].iloc[-2]
    prev_sma200 = df["sma_200"].iloc[-2]
    close = df["close"].iloc[-1]
    adx = df["adx"].iloc[-1]

    if prev_sma50 <= prev_sma200 and sma50 > sma200:
        return "BUY", 0.9, f"Golden Cross! SMA50({sma50:.0f}) > SMA200({sma200:.0f}), ADX={adx:.0f}"
    elif prev_sma50 >= prev_sma200 and sma50 < sma200:
        return "SELL", -0.9, f"Death Cross! SMA50({sma50:.0f}) < SMA200({sma200:.0f})"
    elif sma50 > sma200 and close > sma50:
        return "BUY", 0.3, f"Uptrend: price > SMA50 > SMA200"
    elif sma50 < sma200 and close < sma50:
        return "SELL", -0.3, f"Downtrend: price < SMA50 < SMA200"
    return "HOLD", 0, f"SMA50={sma50:.0f}, SMA200={sma200:.0f}"


def strategy_momentum_breakout(df):
    """Strategy 4: Multi-timeframe momentum breakout."""
    if len(df) < 30:
        return "HOLD", 0, "Insufficient data"
    m5 = df["momentum_5"].iloc[-1]
    m10 = df["momentum_10"].iloc[-1]
    m21 = df["momentum_21"].iloc[-1]
    vol_z = df["vol_zscore"].iloc[-1]
    adx = df["adx"].iloc[-1]

    # All three timeframes aligned + strong trend
    if m5 > 0.02 and m10 > 0.03 and m21 > 0.05 and adx > 25:
        score = min(1.0, (m21 * 10 + 0.3))
        return "BUY", score, f"Triple momentum aligned: 5d={m5:.1%}, 10d={m10:.1%}, 21d={m21:.1%}, ADX={adx:.0f}"
    elif m5 < -0.02 and m10 < -0.03 and m21 < -0.05 and adx > 25:
        score = min(1.0, abs(m21 * 10) + 0.3)
        return "SELL", -score, f"Triple momentum breakdown: 5d={m5:.1%}, 10d={m10:.1%}, 21d={m21:.1%}"

    # Volume breakout
    if vol_z > 2.0 and m5 > 0.03:
        return "BUY", 0.5, f"Volume breakout: VolZ={vol_z:.1f}, Mom5={m5:.1%}"
    return "HOLD", 0, f"Mom5={m5:.1%}, Mom21={m21:.1%}, ADX={adx:.0f}"


def strategy_hurst_regime(df):
    """Strategy 5: Hurst exponent regime-adaptive strategy."""
    if len(df) < 110:
        return "HOLD", 0, "Need 100+ days for Hurst"
    hurst = df["hurst_exponent"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    ema9 = df["ema_9"].iloc[-1]
    ema21 = df["ema_21"].iloc[-1]
    close = df["close"].iloc[-1]

    if hurst > 0.6:
        # Trending regime → trend follow
        if ema9 > ema21 and close > ema9:
            return "BUY", 0.6, f"Trending regime (H={hurst:.2f}): EMA9 > EMA21, ride trend"
        elif ema9 < ema21 and close < ema9:
            return "SELL", -0.6, f"Trending regime (H={hurst:.2f}): downtrend"
    elif hurst < 0.4:
        # Mean-reverting regime → fade extremes
        if rsi < 30:
            return "BUY", 0.7, f"Mean-reverting regime (H={hurst:.2f}): RSI={rsi:.0f} oversold"
        elif rsi > 70:
            return "SELL", -0.7, f"Mean-reverting regime (H={hurst:.2f}): RSI={rsi:.0f} overbought"

    return "HOLD", 0, f"Hurst={hurst:.2f} ({'trending' if hurst > 0.5 else 'mean-reverting'}), RSI={rsi:.0f}"


ALL_STRATEGIES = {
    "rsi_macd": ("RSI+MACD Confluence", strategy_rsi_macd),
    "bollinger": ("Bollinger Mean Reversion", strategy_bollinger_reversion),
    "golden_cross": ("Golden/Death Cross", strategy_golden_cross),
    "momentum": ("Momentum Breakout", strategy_momentum_breakout),
    "hurst": ("Hurst Regime Adaptive", strategy_hurst_regime),
}


def run_all_strategies(symbol):
    """Run all 5 strategies concurrently and compute composite score."""
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol=? ORDER BY date", conn, params=(symbol,))
    conn.close()
    if len(df) < 30:
        return None

    df = vectorized_technicals(df)
    today = datetime.now().strftime("%Y-%m-%d")

    results = []
    conn = get_db()
    for key, (name, func) in ALL_STRATEGIES.items():
        signal, score, reason = func(df)
        results.append({"strategy": key, "name": name, "signal": signal, "score": score, "reason": reason})
        conn.execute(
            "INSERT INTO signals (symbol, date, strategy, signal, score, reason) VALUES (?,?,?,?,?,?)",
            (symbol, today, key, signal, score, reason),
        )

    # Composite score: sum of all strategy scores (-5 to +5)
    composite = sum(r["score"] for r in results)
    buy_count = sum(1 for r in results if r["signal"] == "BUY")
    sell_count = sum(1 for r in results if r["signal"] == "SELL")
    hold_count = sum(1 for r in results if r["signal"] == "HOLD")

    # Rating mapping
    if composite >= 2.5:
        rating = "Strong Buy"
    elif composite >= 1.0:
        rating = "Buy"
    elif composite <= -2.5:
        rating = "Strong Sell"
    elif composite <= -1.0:
        rating = "Sell"
    else:
        rating = "Neutral"

    conn.execute(
        """INSERT OR REPLACE INTO composite_scores
           (symbol, date, composite_score, rating, buy_signals, sell_signals, hold_signals)
           VALUES (?,?,?,?,?,?,?)""",
        (symbol, today, composite, rating, buy_count, sell_count, hold_count),
    )
    conn.commit()
    conn.close()

    return {
        "symbol": symbol, "composite": composite, "rating": rating,
        "buy": buy_count, "sell": sell_count, "hold": hold_count,
        "strategies": results,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WALK-FORWARD ML ENGINE (LightGBM + Ridge Ensemble)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ML_FEATURES = [
    "rsi", "macd", "macd_hist", "bb_width", "bb_pctb",
    "atr", "adx", "hurst_exponent",
    "momentum_5", "momentum_10", "momentum_21",
    "vol_ratio", "vol_zscore",
    "return_1d", "return_5d",
]


def walk_forward_train(symbol, n_splits=5):
    """Walk-forward expanding-window cross-validation.
    More realistic than random train/test split — prevents lookahead bias."""
    conn = get_db()
    # Join OHLCV with technicals
    df = pd.read_sql_query("""
        SELECT o.*, t.rsi, t.macd, t.macd_hist, t.bb_width, t.bb_pctb,
               t.atr, t.adx, t.hurst_exponent,
               t.momentum_5, t.momentum_10, t.momentum_21,
               t.vol_ratio, t.vol_zscore, t.return_1d, t.return_5d
        FROM ohlcv o
        LEFT JOIN technicals t ON o.symbol = t.symbol AND o.date = t.date
        WHERE o.symbol = ?
        ORDER BY o.date
    """, conn, params=(symbol,))
    conn.close()

    if len(df) < 100:
        log.warning(f"Not enough data for ML: {symbol} ({len(df)} rows)")
        return None, None, None

    # Target: 21-day forward return (positive = 1, negative = 0)
    df["fwd_return_21d"] = df["close"].shift(-21) / df["close"] - 1
    df["target"] = (df["fwd_return_21d"] > 0).astype(int)
    df = df.dropna(subset=ML_FEATURES + ["target"])

    if len(df) < 60:
        return None, None, None

    available = [f for f in ML_FEATURES if f in df.columns and df[f].notna().sum() > len(df) * 0.5]
    X = df[available].fillna(0).values
    y = df["target"].values

    scaler = StandardScaler()

    # Walk-forward: expanding window
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    best_model = None
    best_score = 0

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Model 1: LightGBM (if available)
        if HAS_LGBM:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                num_leaves=31, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=42,
            )
            lgb_model.fit(X_train_s, y_train)
            lgb_pred = lgb_model.predict_proba(X_test_s)[:, 1]
        else:
            lgb_pred = np.full(len(X_test_s), 0.5)

        # Model 2: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=42
        )
        rf_model.fit(X_train_s, y_train)
        rf_pred = rf_model.predict_proba(X_test_s)[:, 1]

        # Model 3: Ridge regression (for continuous signal)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_s, y_train)
        ridge_pred = np.clip(ridge_model.predict(X_test_s), 0, 1)

        # Ensemble: weighted average (LightGBM 40%, RF 35%, Ridge 25%)
        if HAS_LGBM:
            ensemble_pred = 0.40 * lgb_pred + 0.35 * rf_pred + 0.25 * ridge_pred
        else:
            ensemble_pred = 0.60 * rf_pred + 0.40 * ridge_pred

        ensemble_class = (ensemble_pred > 0.5).astype(int)
        acc = accuracy_score(y_test, ensemble_class)
        fold_scores.append(acc)

        if acc > best_score:
            best_score = acc
            best_model = (lgb_model if HAS_LGBM else None, rf_model, ridge_model)

    avg_accuracy = np.mean(fold_scores)

    # Final fit on all data for live predictions
    X_all_s = scaler.fit_transform(X)
    if HAS_LGBM and best_model[0]:
        best_model[0].fit(X_all_s, y)
    best_model[1].fit(X_all_s, y)
    best_model[2].fit(X_all_s, y)

    # Log to DB
    conn = get_db()
    conn.execute(
        """INSERT INTO ml_models (symbol, model_type, accuracy, features,
           walk_forward_folds, train_size, test_size) VALUES (?,?,?,?,?,?,?)""",
        (symbol, "LightGBM+RF+Ridge Ensemble", avg_accuracy,
         json.dumps(available), n_splits, len(X), 0),
    )
    conn.commit()
    conn.close()

    log.info(f"🧠 {symbol}: Walk-forward accuracy = {avg_accuracy:.1%} (folds: {[f'{s:.1%}' for s in fold_scores]})")
    return best_model, scaler, available


def ml_predict(symbol, models_cache={}):
    """Get ML prediction for a symbol."""
    if symbol not in models_cache:
        result = walk_forward_train(symbol)
        if result[0] is None:
            return "HOLD", 0.5, "Insufficient data for ML"
        models_cache[symbol] = result

    model_tuple, scaler, features = models_cache[symbol]
    lgb_model, rf_model, ridge_model = model_tuple

    conn = get_db()
    df = pd.read_sql_query("""
        SELECT o.*, t.rsi, t.macd, t.macd_hist, t.bb_width, t.bb_pctb,
               t.atr, t.adx, t.hurst_exponent,
               t.momentum_5, t.momentum_10, t.momentum_21,
               t.vol_ratio, t.vol_zscore, t.return_1d, t.return_5d
        FROM ohlcv o
        LEFT JOIN technicals t ON o.symbol = t.symbol AND o.date = t.date
        WHERE o.symbol = ? ORDER BY o.date
    """, conn, params=(symbol,))
    conn.close()

    if df.empty:
        return "HOLD", 0.5, "No data"

    last = df[features].iloc[-1:].fillna(0).values
    last_s = scaler.transform(last)

    rf_prob = rf_model.predict_proba(last_s)[0][1]
    ridge_prob = np.clip(ridge_model.predict(last_s)[0], 0, 1)

    if lgb_model and HAS_LGBM:
        lgb_prob = lgb_model.predict_proba(last_s)[0][1]
        ensemble = 0.40 * lgb_prob + 0.35 * rf_prob + 0.25 * ridge_prob
    else:
        ensemble = 0.60 * rf_prob + 0.40 * ridge_prob

    if ensemble > 0.6:
        return "BUY", ensemble, f"ML ensemble confidence: {ensemble:.1%}"
    elif ensemble < 0.4:
        return "SELL", 1 - ensemble, f"ML ensemble bearish: {ensemble:.1%}"
    return "HOLD", 0.5, f"ML neutral: {ensemble:.1%}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RISK-ADJUSTED POSITION SIZING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def kelly_position_size(win_rate, avg_win, avg_loss, max_pct=0.10):
    """Kelly criterion for optimal bet sizing, capped at max_pct of portfolio."""
    if avg_loss == 0:
        return max_pct
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b
    # Use half-Kelly for safety
    half_kelly = max(0, kelly / 2)
    return min(half_kelly, max_pct)


def volatility_adjusted_qty(symbol, budget, equity):
    """Size positions inversely proportional to volatility."""
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 30",
        conn, params=(symbol,)
    )
    conn.close()
    if len(df) < 10:
        # Default to budget-based
        price = df["close"].iloc[0] if len(df) > 0 else 100
        return max(1, int(budget / price))

    returns = df["close"].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)  # Annualized vol
    price = df["close"].iloc[0]

    # Target 1% portfolio risk per position
    target_risk = 0.01 * equity
    # Dollar risk = price * vol * sqrt(holding period ~5 days)
    dollar_risk_per_share = price * vol * np.sqrt(5) / np.sqrt(252)
    if dollar_risk_per_share < 0.01:
        dollar_risk_per_share = 0.01

    vol_qty = int(target_risk / dollar_risk_per_share)
    budget_qty = int(budget / price)
    final = max(1, min(vol_qty, budget_qty))

    # Log sizing decision
    conn = get_db()
    conn.execute(
        "INSERT INTO position_sizing (symbol, vol_adjusted_size, final_qty, portfolio_pct, rationale) VALUES (?,?,?,?,?)",
        (symbol, vol_qty, final, round(final * price / equity * 100, 2),
         f"Vol={vol:.1%}, Price=${price:.2f}, Risk/share=${dollar_risk_per_share:.2f}"),
    )
    conn.commit()
    conn.close()
    return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MARKET REGIME DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_market_regime():
    """Detect current market regime using SPY + macro signals."""
    conn = get_db()

    # Get SPY data
    spy = pd.read_sql_query(
        "SELECT * FROM ohlcv WHERE symbol='SPY' ORDER BY date DESC LIMIT 60", conn
    ).sort_values("date")

    # Get VIX from macro table
    vix_row = conn.execute(
        "SELECT value FROM macro WHERE series_id='VIXCLS' ORDER BY date DESC LIMIT 1"
    ).fetchone()
    vix = vix_row[0] if vix_row else 20

    # Get yield curve
    yc_row = conn.execute(
        "SELECT value FROM macro WHERE series_id='T10Y2Y' ORDER BY date DESC LIMIT 1"
    ).fetchone()
    yield_curve = yc_row[0] if yc_row else 0.5
    conn.close()

    if len(spy) < 30:
        return {"regime": "unknown", "vix": vix, "yield_curve": yield_curve}

    spy = vectorized_technicals(spy)
    sma50 = spy["sma_50"].iloc[-1] if "sma_50" in spy.columns else spy["close"].iloc[-1]
    sma200 = spy["sma_200"].iloc[-1] if "sma_200" in spy.columns and pd.notna(spy["sma_200"].iloc[-1]) else spy["close"].iloc[-1]
    current = spy["close"].iloc[-1]

    # Breadth: % of watchlist above their SMA 20
    conn2 = get_db()
    above_sma = 0
    for sym in WATCHLIST:
        row = conn2.execute(
            "SELECT t.sma_20, o.close FROM technicals t JOIN ohlcv o ON t.symbol=o.symbol AND t.date=o.date WHERE t.symbol=? ORDER BY t.date DESC LIMIT 1",
            (sym,)
        ).fetchone()
        if row and row[0] and row[1] > row[0]:
            above_sma += 1
    conn2.close()
    breadth = above_sma / len(WATCHLIST) * 100

    # Classify regime
    if vix > 30:
        regime = "high_vol"
    elif current > sma50 > sma200 and breadth > 60:
        regime = "bull"
    elif current < sma50 < sma200 and breadth < 40:
        regime = "bear"
    else:
        regime = "sideways"

    confidence = 0.5
    if regime == "bull" and breadth > 75:
        confidence = 0.9
    elif regime == "bear" and breadth < 25:
        confidence = 0.9
    elif regime == "high_vol":
        confidence = 0.8

    # Store
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO market_regime VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d"), regime, vix, yield_curve, breadth, confidence),
    )
    conn.commit()
    conn.close()

    return {"regime": regime, "vix": vix, "yield_curve": yield_curve,
            "breadth": breadth, "confidence": confidence}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTESTER WITH TRANSACTION COSTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest_strategy(symbol, strategy_key, initial_capital=100000):
    """Backtest a strategy with realistic transaction costs and next-open fills."""
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol=? ORDER BY date", conn, params=(symbol,))
    conn.close()

    if len(df) < 60:
        return None

    df = vectorized_technicals(df)
    _, strategy_fn = ALL_STRATEGIES.get(strategy_key, (None, None))
    if not strategy_fn:
        return None

    cash = initial_capital
    shares = 0
    trades = []
    equity_curve = []
    peak_equity = initial_capital

    for i in range(30, len(df) - 1):
        window = df.iloc[:i+1].copy()
        signal, score, reason = strategy_fn(window)
        current_close = df.iloc[i]["close"]
        next_open = df.iloc[i+1]["open"]  # Next-open fill
        equity = cash + shares * current_close

        # Track drawdown
        peak_equity = max(peak_equity, equity)
        equity_curve.append({"date": df.iloc[i]["date"], "equity": equity})

        tx_cost_rate = (TX_COST_BPS + SLIPPAGE_BPS) / 10000

        if signal == "BUY" and shares == 0 and abs(score) > 0.3:
            qty = int(cash * 0.95 / next_open)  # 95% of cash
            if qty > 0:
                cost = qty * next_open * (1 + tx_cost_rate)
                cash -= cost
                shares = qty
                trades.append({"date": df.iloc[i+1]["date"], "side": "BUY",
                               "price": next_open, "qty": qty, "tx_cost": qty * next_open * tx_cost_rate})

        elif signal == "SELL" and shares > 0 and abs(score) > 0.3:
            proceeds = shares * next_open * (1 - tx_cost_rate)
            trades.append({"date": df.iloc[i+1]["date"], "side": "SELL",
                           "price": next_open, "qty": shares, "tx_cost": shares * next_open * tx_cost_rate})
            cash += proceeds
            shares = 0

    final_equity = cash + shares * df.iloc[-1]["close"]
    total_return = (final_equity / initial_capital - 1) * 100
    max_dd = 0
    if equity_curve:
        eq = [e["equity"] for e in equity_curve]
        peak = eq[0]
        for e in eq:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)

    # Win rate
    paired_trades = []
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            buy_p = trades[i]["price"]
            sell_p = trades[i+1]["price"]
            paired_trades.append(sell_p / buy_p - 1)
    win_rate = sum(1 for t in paired_trades if t > 0) / max(len(paired_trades), 1)
    avg_pnl = np.mean(paired_trades) if paired_trades else 0
    total_tx = sum(t["tx_cost"] for t in trades)

    # Sharpe (annualized from daily returns)
    if len(equity_curve) > 30:
        eq_series = pd.Series([e["equity"] for e in equity_curve])
        daily_rets = eq_series.pct_change().dropna()
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    else:
        sharpe = 0

    result = {
        "strategy": strategy_key, "symbol": symbol,
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "total_trades": len(trades),
        "avg_trade_pnl": round(avg_pnl * 100, 2),
        "tx_costs": round(total_tx, 2),
    }

    conn = get_db()
    conn.execute(
        """INSERT INTO backtests (strategy, symbol, start_date, end_date,
           total_return, sharpe_ratio, max_drawdown, win_rate,
           total_trades, avg_trade_pnl, tx_costs_total) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (strategy_key, symbol, str(df.iloc[30]["date"]), str(df.iloc[-1]["date"]),
         result["total_return"], sharpe, max_dd * 100, win_rate * 100,
         len(trades), avg_pnl * 100, total_tx),
    )
    conn.commit()
    conn.close()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENHANCED AUTO-TRADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AutoTrader:
    def __init__(self, client, strategy="composite", budget=10000,
                 max_positions=3, stop_loss_pct=2.0, take_profit_pct=5.0):
        self.client = client
        self.strategy = strategy
        self.budget = budget
        self.max_positions = max_positions
        self.stop_loss = stop_loss_pct / 100
        self.take_profit = take_profit_pct / 100
        self.running = False
        self.thread = None
        self.ml_cache = {}
        self.scan_interval = 60

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        log.info(f"▶ Auto-trader STARTED — Strategy: {self.strategy}, Budget: ${self.budget}")

    def stop(self):
        self.running = False
        log.info("⏹ Auto-trader STOPPED")

    def _run_loop(self):
        while self.running:
            try:
                self._scan_and_trade()
                self._snapshot_portfolio()
            except Exception as e:
                log.error(f"Auto-trade error: {e}")
            time.sleep(self.scan_interval)

    def _scan_and_trade(self):
        positions = self.client.get_positions()
        pos_symbols = [p["symbol"] for p in positions]

        # Stop-loss / take-profit
        for pos in positions:
            pnl_pct = pos["pnl_pct"] / 100
            if pnl_pct <= -self.stop_loss:
                log.info(f"🛑 STOP LOSS: {pos['symbol']} ({pos['pnl_pct']:.1f}%)")
                self.client.close_position(pos["symbol"])
            elif pnl_pct >= self.take_profit:
                log.info(f"🎯 TAKE PROFIT: {pos['symbol']} ({pos['pnl_pct']:.1f}%)")
                self.client.close_position(pos["symbol"])

        if len(pos_symbols) >= self.max_positions:
            return

        # Detect regime for strategy adjustment
        regime = detect_market_regime()

        # Score all watchlist symbols
        scored = []
        for symbol in WATCHLIST:
            if symbol in pos_symbols:
                continue

            if self.strategy == "composite":
                result = run_all_strategies(symbol)
                if result and abs(result["composite"]) >= 1.0:
                    scored.append((symbol, result["composite"], result["rating"]))

            elif self.strategy == "ml":
                signal, conf, reason = ml_predict(symbol, self.ml_cache)
                if signal == "BUY":
                    scored.append((symbol, conf, reason))

            else:
                # Single strategy mode
                conn = get_db()
                df = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol=? ORDER BY date", conn, params=(symbol,))
                conn.close()
                if len(df) < 30:
                    continue
                df = vectorized_technicals(df)
                _, func = ALL_STRATEGIES.get(self.strategy, (None, None))
                if func:
                    signal, score, reason = func(df)
                    if signal == "BUY" and score > 0.3:
                        scored.append((symbol, score, reason))

        # Sort by score descending, take top candidate
        scored.sort(key=lambda x: x[1], reverse=True)

        for symbol, score, reason in scored[:1]:
            # Risk-adjusted position sizing
            try:
                equity = self.client.get_account()["equity"]
            except Exception:
                equity = self.budget
            qty = volatility_adjusted_qty(symbol, self.budget / self.max_positions, equity)

            # Regime-adjust: reduce size in high-vol
            if regime.get("regime") == "high_vol":
                qty = max(1, qty // 2)
                log.info(f"  ⚠️ High-vol regime: halved position size")

            log.info(f"📊 {self.strategy} → BUY {qty} {symbol} (score={score:.2f}): {reason}")
            self.client.buy(symbol, qty=qty)

    def _snapshot_portfolio(self):
        """Take daily portfolio snapshot for performance tracking."""
        try:
            acct = self.client.get_account()
            positions = self.client.get_positions()
            conn = get_db()
            today = datetime.now().strftime("%Y-%m-%d")
            pos_value = sum(p["market_value"] for p in positions)

            # Get yesterday's snapshot for daily return
            prev = conn.execute(
                "SELECT equity FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
            ).fetchone()
            prev_equity = prev[0] if prev else 100000
            daily_return = (acct["equity"] / prev_equity - 1) if prev_equity > 0 else 0

            conn.execute(
                """INSERT OR REPLACE INTO portfolio_snapshots
                   (date, equity, cash, positions_value, daily_pnl, daily_return, num_positions)
                   VALUES (?,?,?,?,?,?,?)""",
                (today, acct["equity"], acct["cash"], pos_value,
                 acct["equity"] - prev_equity, daily_return, len(positions)),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL PIPELINE (like Quant Research System daily run)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_pipeline(client):
    """Run the complete daily pipeline (like ingest.py + main.py)."""
    log.info("=" * 60)
    log.info("  RUNNING FULL DAILY PIPELINE")
    log.info("=" * 60)

    # Step 1: Ingest data
    ingest_ohlcv(client, years=2)

    # Step 2: Fundamentals
    ingest_fundamentals()

    # Step 3: Macro data
    ingest_macro()

    # Step 4: Compute vectorized technicals
    compute_and_store_technicals()

    # Step 5: Private company exposure
    compute_private_exposure()

    # Step 6: Detect market regime
    regime = detect_market_regime()
    log.info(f"🌍 Market Regime: {regime['regime'].upper()} (VIX={regime['vix']:.1f}, Breadth={regime.get('breadth', 0):.0f}%)")

    # Step 7: Run all strategies for all symbols
    log.info(f"📊 Running 5 strategies × {len(WATCHLIST)} symbols...")
    results = []
    for symbol in WATCHLIST:
        r = run_all_strategies(symbol)
        if r:
            results.append(r)

    # Step 8: Summary
    results.sort(key=lambda x: x["composite"], reverse=True)
    log.info("\n  ── COMPOSITE SCORES ──")
    log.info(f"  {'Symbol':<8} {'Score':>7} {'Rating':<14} {'B':>3} {'S':>3} {'H':>3}")
    log.info(f"  {'─'*8} {'─'*7} {'─'*14} {'─'*3} {'─'*3} {'─'*3}")
    for r in results:
        log.info(f"  {r['symbol']:<8} {r['composite']:>+7.2f} {r['rating']:<14} {r['buy']:>3} {r['sell']:>3} {r['hold']:>3}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERACTIVE CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HELP_TEXT = """
╔═══════════════════════════════════════════════════════════╗
║           QUANTUM TRADER v2.0 — Commands                  ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  MANUAL TRADING                                           ║
║    buy AAPL 10           Market buy 10 shares              ║
║    buy AAPL 10 195.50    Limit buy at $195.50              ║
║    sell AAPL 10          Market sell 10 shares             ║
║    close AAPL            Close entire position             ║
║    close_all             Close ALL positions                ║
║                                                           ║
║  PORTFOLIO                                                ║
║    account               Account summary                   ║
║    positions             Open positions                    ║
║    orders                Recent orders                     ║
║                                                           ║
║  DATA PIPELINE (from Quant Research System)               ║
║    pipeline              Run FULL daily pipeline           ║
║    download              Ingest 2yr OHLCV data             ║
║    fundamentals          Fetch P/E, FCF, ROE, analyst      ║
║    macro                 Fetch FRED data (VIX, yields)     ║
║    technicals            Compute all indicators            ║
║    regime                Detect market regime              ║
║                                                           ║
║  ANALYSIS & ML                                            ║
║    analyze AAPL          Full multi-strategy analysis       ║
║    scan                  Score ALL watchlist symbols        ║
║    train AAPL            Walk-forward ML training           ║
║    predict AAPL          ML ensemble prediction             ║
║    backtest AAPL momentum  Backtest strategy w/ tx costs   ║
║    private               Show private company exposure     ║
║                                                           ║
║  AUTO-TRADING                                             ║
║    auto start composite  All 5 strategies → composite      ║
║    auto start ml         ML ensemble signals               ║
║    auto start momentum   Single strategy mode              ║
║    auto stop             Stop auto-trader                  ║
║    auto status           Show auto-trader state             ║
║                                                           ║
║  SYSTEM                                                   ║
║    watchlist             Show tracked symbols               ║
║    dbstats               Database statistics               ║
║    newkeys               Re-enter API keys                 ║
║    reset                 Delete .env and start fresh       ║
║    help                  Show this menu                    ║
║    quit                  Exit                              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


def main():
    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║                                                   ║")
    print("  ║    ⚛  QUANTUM TRADER v2.0                         ║")
    print("  ║    Alpaca Paper Trading • ML-Powered              ║")
    print("  ║                                                   ║")
    print("  ╚═══════════════════════════════════════════════════╝")

    init_db()
    client = AlpacaClient()
    auto_trader = None

    # Show account
    try:
        acct = client.get_account()
        print(f"\n   Portfolio: ${acct['equity']:,.2f}")
        print(f"   Cash:      ${acct['cash']:,.2f}")
        print(f"   P&L:       ${acct['pnl']:+,.2f}")
    except Exception as e:
        print(f"   ⚠️ Could not fetch account: {e}")

    # Check if this is first run (no data yet)
    conn = get_db()
    row_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    conn.close()

    if row_count == 0:
        print("\n  ┌─────────────────────────────────────────────┐")
        print("  │  First run detected — no market data yet.   │")
        print("  │  Want to download data and run the full     │")
        print("  │  pipeline? (takes ~2 min)                   │")
        print("  └─────────────────────────────────────────────┘")
        go = input("\n  Run pipeline now? (y/n): ").strip().lower()
        if go in ("y", "yes", ""):
            run_full_pipeline(client)
            print("\n  ✅ Pipeline complete! You're ready to trade.")
            print("  Try: scan, analyze AAPL, or auto start composite\n")

    print(f"\n   Type 'help' for commands\n")

    while True:
        try:
            cmd = input("quantum> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        # ── Manual Trading ──
        if action == "buy" and len(parts) >= 3:
            sym = parts[1].upper()
            qty = int(parts[2])
            lim = float(parts[3]) if len(parts) > 3 else None
            print(f"  → {json.dumps(client.buy(sym, qty=qty, order_type='limit' if lim else 'market', limit_price=lim), indent=2)}")

        elif action == "sell" and len(parts) >= 3:
            sym = parts[1].upper()
            qty = int(parts[2])
            lim = float(parts[3]) if len(parts) > 3 else None
            print(f"  → {json.dumps(client.sell(sym, qty=qty, order_type='limit' if lim else 'market', limit_price=lim), indent=2)}")

        elif action == "close" and len(parts) == 2:
            print(f"  → {json.dumps(client.close_position(parts[1].upper()), indent=2)}")

        elif action == "close_all":
            client.close_all()

        # ── Portfolio ──
        elif action == "account":
            a = client.get_account()
            print(f"  Equity:       ${a['equity']:>12,.2f}")
            print(f"  Cash:         ${a['cash']:>12,.2f}")
            print(f"  Buying Power: ${a['buying_power']:>12,.2f}")
            print(f"  P&L:          ${a['pnl']:>+12,.2f}")

        elif action == "positions":
            pos = client.get_positions()
            if not pos:
                print("  No open positions")
            else:
                print(f"  {'Symbol':<8} {'Qty':>6} {'Avg':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8}")
                print(f"  {'─'*52}")
                for p in pos:
                    s = "+" if p["pnl"] >= 0 else ""
                    print(f"  {p['symbol']:<8} {p['qty']:>6.0f} ${p['avg_price']:>9.2f} ${p['current_price']:>9.2f} {s}${p['pnl']:>8.2f} {s}{p['pnl_pct']:>6.1f}%")

        elif action == "orders":
            for o in client.get_orders(limit=10):
                print(f"  {o['created'][:19]} │ {o['side']:>4} {o['symbol']:<6} x{o['qty']:>4.0f} │ {o['type']:>6} │ {o['status']}")

        # ── Data Pipeline ──
        elif action == "pipeline":
            run_full_pipeline(client)

        elif action == "download":
            ingest_ohlcv(client, years=2)

        elif action == "fundamentals":
            ingest_fundamentals()

        elif action == "macro":
            ingest_macro()

        elif action == "technicals":
            compute_and_store_technicals()

        elif action == "regime":
            r = detect_market_regime()
            print(f"  Regime:      {r['regime'].upper()}")
            print(f"  VIX:         {r['vix']:.1f}")
            print(f"  Yield Curve: {r.get('yield_curve', 'N/A')}")
            print(f"  Breadth:     {r.get('breadth', 'N/A'):.0f}%")
            print(f"  Confidence:  {r.get('confidence', 0):.0%}")

        # ── Analysis ──
        elif action == "analyze" and len(parts) >= 2:
            sym = parts[1].upper()
            result = run_all_strategies(sym)
            if result:
                print(f"\n  ── {sym} Composite Analysis ──")
                print(f"  Score:  {result['composite']:+.2f}")
                print(f"  Rating: {result['rating']}")
                print(f"  Buy/Sell/Hold: {result['buy']}/{result['sell']}/{result['hold']}")
                print()
                for s in result["strategies"]:
                    emoji = "🟢" if s["signal"] == "BUY" else "🔴" if s["signal"] == "SELL" else "⚪"
                    print(f"  {emoji} {s['name']:<28} {s['signal']:>5} ({s['score']:+.2f})  {s['reason']}")
            else:
                print(f"  No data for {sym}. Run 'download' first.")

        elif action == "scan":
            results = []
            for sym in WATCHLIST:
                r = run_all_strategies(sym)
                if r:
                    results.append(r)
            results.sort(key=lambda x: x["composite"], reverse=True)
            print(f"\n  {'Symbol':<8} {'Score':>7} {'Rating':<14} {'Buy':>4} {'Sell':>5} {'Hold':>5}")
            print(f"  {'─'*48}")
            for r in results:
                print(f"  {r['symbol']:<8} {r['composite']:>+7.2f} {r['rating']:<14} {r['buy']:>4} {r['sell']:>5} {r['hold']:>5}")

        elif action == "train" and len(parts) >= 2:
            sym = parts[1].upper()
            model, scaler, feats = walk_forward_train(sym)
            if model:
                print(f"  ✅ Walk-forward ML trained for {sym} ({len(feats)} features)")

        elif action == "predict" and len(parts) >= 2:
            sym = parts[1].upper()
            signal, conf, reason = ml_predict(sym)
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
            print(f"  {emoji} {sym}: {signal} — {reason}")

        elif action == "backtest" and len(parts) >= 3:
            sym = parts[1].upper()
            strat = parts[2].lower()
            r = backtest_strategy(sym, strat)
            if r:
                print(f"\n  ── Backtest: {strat} on {sym} ──")
                print(f"  Return:     {r['total_return']:+.1f}%")
                print(f"  Sharpe:     {r['sharpe']:.2f}")
                print(f"  Max DD:     {r['max_drawdown']:.1f}%")
                print(f"  Win Rate:   {r['win_rate']:.0f}%")
                print(f"  Trades:     {r['total_trades']}")
                print(f"  Avg P&L:    {r['avg_trade_pnl']:+.2f}%")
                print(f"  TX Costs:   ${r['tx_costs']:.2f}")
            else:
                print(f"  Not enough data. Run 'download' first.")

        elif action == "private":
            conn = get_db()
            rows = conn.execute("SELECT * FROM private_exposure ORDER BY date DESC").fetchall()
            conn.close()
            if not rows:
                compute_private_exposure()
                conn = get_db()
                rows = conn.execute("SELECT * FROM private_exposure ORDER BY date DESC").fetchall()
                conn.close()
            for r in rows:
                print(f"  {r[0]:<12} Exposure: {r[2]:+.4f}  Return: {r[3]:+.6f}")
                details = json.loads(r[4]) if r[4] else {}
                for proxy, info in details.items():
                    print(f"    └─ {proxy}: weight={info['weight']:.0%}, return={info['return']:+.2f}%")

        # ── Auto Trading ──
        elif action == "auto":
            if len(parts) < 2:
                print("  Usage: auto start <strategy> | auto stop | auto status")
                continue
            sub = parts[1]
            if sub == "start":
                strat = parts[2] if len(parts) > 2 else "composite"
                strat_map = {"composite": "composite", "ml": "ml", "momentum": "rsi_macd",
                             "bollinger": "bollinger", "trend": "golden_cross", "hurst": "hurst"}
                strat_key = strat_map.get(strat, strat)
                budget = float(os.getenv("MAX_BUDGET", "10000"))
                max_pos = int(os.getenv("MAX_POSITIONS", "3"))
                sl = float(os.getenv("STOP_LOSS_PCT", "2.0"))
                tp = float(os.getenv("TAKE_PROFIT_PCT", "5.0"))
                auto_trader = AutoTrader(client, strat_key, budget, max_pos, sl, tp)
                auto_trader.start()
            elif sub == "stop":
                if auto_trader:
                    auto_trader.stop()
                    auto_trader = None
                else:
                    print("  Not running")
            elif sub == "status":
                if auto_trader and auto_trader.running:
                    print(f"  ▶ RUNNING — Strategy: {auto_trader.strategy}")
                    print(f"    Budget: ${auto_trader.budget:,.0f} | Max Pos: {auto_trader.max_positions}")
                    print(f"    SL: {auto_trader.stop_loss*100:.1f}% | TP: {auto_trader.take_profit*100:.1f}%")
                else:
                    print("  ⏹ Not running")

        elif action == "watchlist":
            print(f"  Tracking {len(WATCHLIST)} symbols: {', '.join(WATCHLIST)}")

        elif action == "dbstats":
            conn = get_db()
            tables = ["ohlcv", "technicals", "fundamentals", "macro", "signals",
                       "composite_scores", "trades", "ml_models", "backtests",
                       "private_exposure", "position_sizing", "market_regime", "portfolio_snapshots"]
            print(f"\n  ── Database: {DB_PATH} ──")
            for t in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    print(f"  {t:<22} {count:>8,} rows")
                except Exception:
                    print(f"  {t:<22} (not created)")
            conn.close()

        elif action == "reset":
            confirm = input("  Delete .env and re-enter API keys? (y/n): ").strip().lower()
            if confirm == "y":
                if ENV_PATH.exists():
                    ENV_PATH.unlink()
                print("  .env deleted. Restart the bot to enter new keys.")
                break

        elif action == "newkeys":
            print("  Re-entering API keys...")
            if ENV_PATH.exists():
                ENV_PATH.unlink()
            setup_api_keys()
            print("  ✅ New keys saved. Restart the bot to reconnect.")

        elif action == "help":
            print(HELP_TEXT)

        elif action in ("quit", "exit", "q"):
            if auto_trader and auto_trader.running:
                auto_trader.stop()
            print("Goodbye!")
            break

        else:
            print(f"  Unknown: {cmd}. Type 'help'.")


if __name__ == "__main__":
    main()
