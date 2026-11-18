"""Glue between the SQLite store and the pure analysis code.

Everything here reads from or writes to the database. The math lives in ``features``,
``strategies``, ``models``, ``backtest`` and ``risk``, which take and return plain DataFrames.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.engine import BacktestResult, run_backtest
from trading_bot.data import ingest
from trading_bot.data.db import connect
from trading_bot.data.loaders import load_ohlcv, load_ohlcv_with_technicals
from trading_bot.data.universe import PRIVATE_COMPANY_PROXIES, WATCHLIST
from trading_bot.features.technicals import TECHNICAL_COLUMNS, add_technicals, technical_rows
from trading_bot.models.ensemble import Ensemble, predict_latest, train_walk_forward
from trading_bot.risk.exposure import proxy_exposure
from trading_bot.risk.regime import classify_regime
from trading_bot.risk.sizing import SizeDecision, volatility_adjusted_qty
from trading_bot.strategies import STRATEGIES, score_all

log = logging.getLogger(__name__)

# Trained ensembles by symbol, shared by the CLI and dashboard for the life of the process.
MODEL_CACHE: dict[str, Ensemble] = {}

_TECH_INSERT = (
    f"INSERT OR REPLACE INTO technicals (symbol, date, {', '.join(TECHNICAL_COLUMNS)}) "  # noqa: S608
    f"VALUES (?,?,{','.join('?' * len(TECHNICAL_COLUMNS))})"
)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def compute_and_store_technicals(db_path: Path | str, symbols=None) -> None:
    symbols = list(symbols or WATCHLIST)
    log.info("Computing technicals for %d symbols", len(symbols))
    for symbol in symbols:
        df = load_ohlcv(db_path, symbol)
        if len(df) < 30:
            continue
        rows = technical_rows(symbol, add_technicals(df))
        with connect(db_path) as conn:
            conn.executemany(_TECH_INSERT, rows)


def analyze_symbol(db_path: Path | str, symbol: str) -> dict | None:
    """Run the five strategies on one symbol, store the signals and composite, and return them."""
    df = load_ohlcv(db_path, symbol)
    if len(df) < 30:
        return None
    result = score_all(add_technicals(df))
    today = _today()
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO signals (symbol, date, strategy, signal, score, reason) VALUES (?,?,?,?,?,?)",
            [(symbol, today, s["strategy"], s["signal"], s["score"], s["reason"]) for s in result.strategies],
        )
        conn.execute(
            """INSERT OR REPLACE INTO composite_scores
               (symbol, date, composite_score, rating, buy_signals, sell_signals, hold_signals)
               VALUES (?,?,?,?,?,?,?)""",
            (symbol, today, result.composite, result.rating, result.buy, result.sell, result.hold),
        )
    return {
        "symbol": symbol,
        "composite": result.composite,
        "rating": result.rating,
        "buy": result.buy,
        "sell": result.sell,
        "hold": result.hold,
        "strategies": result.strategies,
    }


def scan(db_path: Path | str, symbols=None) -> list[dict]:
    """Composite score for every symbol with data, best first."""
    results = [r for r in (analyze_symbol(db_path, s) for s in (symbols or WATCHLIST)) if r]
    return sorted(results, key=lambda r: r["composite"], reverse=True)


def strategy_signal(db_path: Path | str, symbol: str, strategy: str):
    """One strategy's ``(signal, score, reason)`` on the latest bar, or None without data."""
    _, fn = STRATEGIES[strategy]
    df = load_ohlcv(db_path, symbol)
    if len(df) < 30:
        return None
    return fn(add_technicals(df))


def train_symbol(db_path: Path | str, symbol: str, n_splits: int = 5) -> Ensemble | None:
    """Walk-forward train the ensemble for one symbol and record the run in ``ml_models``."""
    df = load_ohlcv_with_technicals(db_path, symbol)
    model = train_walk_forward(df, n_splits=n_splits)
    if model is None:
        log.warning("Not enough data for ML on %s (%d rows)", symbol, len(df))
        return None
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO ml_models (symbol, model_type, accuracy, features,
               walk_forward_folds, train_size, test_size) VALUES (?,?,?,?,?,?,?)""",
            (symbol, "LightGBM+RF+Ridge Ensemble", model.accuracy, json.dumps(model.features), n_splits,
             len(df), 0),
        )  # fmt: skip
    folds = ", ".join(f"{s:.1%}" for s in model.fold_scores)
    log.info("%s: walk-forward accuracy %.1f%% (folds: %s)", symbol, model.accuracy * 100, folds)
    return model


def predict_symbol(db_path: Path | str, symbol: str, cache: dict[str, Ensemble] | None = None):
    """``(signal, confidence, reason)`` from the ensemble, training it on first use."""
    cache = MODEL_CACHE if cache is None else cache
    if symbol not in cache:
        model = train_symbol(db_path, symbol)
        if model is None:
            return "HOLD", 0.5, "Insufficient data for ML"
        cache[symbol] = model
    return predict_latest(cache[symbol], load_ohlcv_with_technicals(db_path, symbol))


def backtest_symbol(db_path: Path | str, symbol: str, strategy: str, costs: CostModel | None = None) -> dict | None:
    if strategy not in STRATEGIES:
        return None
    df = load_ohlcv(db_path, symbol)
    if len(df) < 60:
        return None
    result: BacktestResult = run_backtest(
        add_technicals(df), STRATEGIES[strategy][1], strategy=strategy, symbol=symbol, costs=costs
    )
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO backtests (strategy, symbol, start_date, end_date,
               total_return, sharpe_ratio, max_drawdown, win_rate,
               total_trades, avg_trade_pnl, tx_costs_total) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (strategy, symbol, result.start_date, result.end_date, result.total_return, result.sharpe,
             result.max_drawdown, result.win_rate, result.total_trades, result.avg_trade_pnl, result.tx_costs),
        )  # fmt: skip
    return result.summary()


def size_position(db_path: Path | str, symbol: str, budget: float, equity: float) -> SizeDecision:
    """Volatility-targeted share count from the last 30 closes, logged to ``position_sizing``."""
    closes = load_ohlcv(db_path, symbol, limit=30)["close"].tolist()
    decision = volatility_adjusted_qty(closes, budget, equity)
    if decision.vol_qty is not None:
        with connect(db_path) as conn:
            conn.execute(
                """INSERT INTO position_sizing (symbol, vol_adjusted_size, final_qty, portfolio_pct, rationale)
                   VALUES (?,?,?,?,?)""",
                (symbol, decision.vol_qty, decision.qty, round(decision.qty * decision.price / equity * 100, 2),
                 decision.rationale),
            )  # fmt: skip
    return decision


def _latest_macro(conn, series_id: str, default: float) -> float:
    row = conn.execute("SELECT value FROM macro WHERE series_id=? ORDER BY date DESC LIMIT 1", (series_id,)).fetchone()
    return row[0] if row else default


def detect_market_regime(db_path: Path | str, symbols=None) -> dict:
    """Classify the market from SPY's last 60 bars, VIX, the yield curve and watchlist breadth."""
    symbols = list(symbols or WATCHLIST)
    spy = load_ohlcv(db_path, "SPY", limit=60)
    with connect(db_path) as conn:
        vix = _latest_macro(conn, "VIXCLS", 20)
        yield_curve = _latest_macro(conn, "T10Y2Y", 0.5)
        above = 0
        for symbol in symbols:
            row = conn.execute(
                """SELECT t.sma_20, o.close FROM technicals t
                   JOIN ohlcv o ON t.symbol = o.symbol AND t.date = o.date
                   WHERE t.symbol=? ORDER BY t.date DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
            if row and row[0] and row[1] > row[0]:
                above += 1
    if len(spy) < 30:
        return {"regime": "unknown", "vix": vix, "yield_curve": yield_curve}

    spy = add_technicals(spy)
    current = spy["close"].iloc[-1]
    sma50 = spy["sma_50"].iloc[-1] if spy["sma_50"].notna().iloc[-1] else current
    sma200 = spy["sma_200"].iloc[-1] if spy["sma_200"].notna().iloc[-1] else current
    breadth = above / len(symbols) * 100
    regime, confidence = classify_regime(current, sma50, sma200, vix, breadth)

    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO market_regime VALUES (?,?,?,?,?,?)",
            (_today(), regime, vix, yield_curve, breadth, confidence),
        )
    return {"regime": regime, "vix": vix, "yield_curve": yield_curve, "breadth": breadth, "confidence": confidence}


def compute_private_exposure(db_path: Path | str) -> list[dict]:
    today = _today()
    out = []
    with connect(db_path) as conn:
        for company, weights in PRIVATE_COMPANY_PROXIES.items():
            closes = {}
            for symbol in weights:
                rows = conn.execute(
                    "SELECT close FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT 2", (symbol,)
                ).fetchall()
                if len(rows) == 2:
                    closes[symbol] = (rows[0][0], rows[1][0])
            weighted, details = proxy_exposure(closes, weights)
            exposure = round(weighted * 100, 4)
            conn.execute(
                "INSERT OR REPLACE INTO private_exposure VALUES (?,?,?,?,?)",
                (company, today, exposure, round(weighted, 6), json.dumps(details)),
            )
            out.append({"company": company, "date": today, "exposure": exposure, "return": round(weighted, 6),
                        "details": details})  # fmt: skip
    return out


def snapshot_portfolio(db_path: Path | str, account: dict, positions: list[dict], starting_equity: float) -> None:
    """Upsert today's equity, cash and P&L into ``portfolio_snapshots``."""
    with connect(db_path) as conn:
        prev = conn.execute("SELECT equity FROM portfolio_snapshots ORDER BY date DESC LIMIT 1").fetchone()
        prev_equity = prev[0] if prev else starting_equity
        daily_return = account["equity"] / prev_equity - 1 if prev_equity > 0 else 0
        conn.execute(
            """INSERT OR REPLACE INTO portfolio_snapshots
               (date, equity, cash, positions_value, daily_pnl, daily_return, num_positions)
               VALUES (?,?,?,?,?,?,?)""",
            (_today(), account["equity"], account["cash"], sum(p["market_value"] for p in positions),
             account["equity"] - prev_equity, daily_return, len(positions)),
        )  # fmt: skip


def run_full_pipeline(broker, db_path: Path | str, fred_api_key: str | None = None) -> list[dict]:
    """The daily run: ingest, indicators, exposure, regime, then score the watchlist."""
    log.info("Running the full daily pipeline")
    ingest.ingest_ohlcv(broker.data, db_path, years=2)
    ingest.ingest_fundamentals(db_path)
    ingest.ingest_macro(db_path, fred_api_key)
    compute_and_store_technicals(db_path)
    compute_private_exposure(db_path)
    regime = detect_market_regime(db_path)
    log.info("Market regime: %s (VIX %.1f, breadth %.0f%%)", regime["regime"], regime["vix"], regime.get("breadth", 0))
    results = scan(db_path)
    for r in results:
        log.info(
            "%-6s %+6.2f %-12s B%d S%d H%d", r["symbol"], r["composite"], r["rating"], r["buy"], r["sell"], r["hold"]
        )
    return results
