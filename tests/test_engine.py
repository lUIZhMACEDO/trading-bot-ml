"""The database-backed functions on a temporary SQLite file."""

import json

import numpy as np
import pytest

from conftest import make_ohlcv, random_walk, store_ohlcv
from trading_bot import engine
from trading_bot.data.db import TABLES, connect, init_db, table_counts
from trading_bot.data.loaders import load_ohlcv


def test_init_db_creates_all_13_tables(db_path):
    counts = table_counts(db_path)
    assert list(counts) == list(TABLES) and len(TABLES) == 13
    assert set(counts.values()) == {0}
    init_db(db_path)  # running it twice is fine


def test_table_counts_on_a_missing_table(tmp_path):
    assert set(table_counts(tmp_path / "empty.db").values()) == {0}


def test_load_ohlcv_limit_keeps_the_latest_rows_in_order(db_path):
    df = make_ohlcv(np.arange(100.0, 140.0))
    store_ohlcv(db_path, "AAA", df)
    tail = load_ohlcv(db_path, "AAA", limit=5)
    assert tail["date"].tolist() == df["date"].iloc[-5:].tolist()


def test_analyze_and_scan_store_signals(db_path):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(260, seed=1)))
    store_ohlcv(db_path, "BBB", make_ohlcv(random_walk(260, seed=2)))
    result = engine.analyze_symbol(db_path, "AAA")
    assert result["symbol"] == "AAA" and len(result["strategies"]) == 5
    assert engine.analyze_symbol(db_path, "NONE") is None

    ranked = engine.scan(db_path, ["AAA", "BBB", "NONE"])
    assert {r["symbol"] for r in ranked} == {"AAA", "BBB"}
    composites = [r["composite"] for r in ranked]
    assert composites == sorted(composites, reverse=True)
    counts = table_counts(db_path)
    assert counts["signals"] == 15 and counts["composite_scores"] == 2


def test_technicals_are_stored(db_path):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(80)))
    engine.compute_and_store_technicals(db_path, ["AAA"])
    with connect(db_path) as conn:
        n, rsi_min, rsi_max = conn.execute("SELECT COUNT(*), MIN(rsi), MAX(rsi) FROM technicals").fetchone()
    assert n == 80 - 13 and 0 <= rsi_min <= rsi_max <= 100


def test_backtest_symbol_records_the_run(db_path):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(200, seed=3)))
    summary = engine.backtest_symbol(db_path, "AAA", "momentum")
    assert "trades" not in summary and summary["strategy"] == "momentum"
    assert engine.backtest_symbol(db_path, "AAA", "nope") is None
    with connect(db_path) as conn:
        assert conn.execute("SELECT strategy, symbol FROM backtests").fetchall() == [("momentum", "AAA")]


def test_train_and_predict_use_the_cache(db_path):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(300, seed=4, drift=0.0005)))
    engine.compute_and_store_technicals(db_path, ["AAA"])
    cache = {}
    signal, confidence, _ = engine.predict_symbol(db_path, "AAA", cache)
    assert signal in {"BUY", "SELL", "HOLD"} and "AAA" in cache
    with connect(db_path) as conn:
        assert table_counts(db_path)["ml_models"] == 1
        features = json.loads(conn.execute("SELECT features FROM ml_models").fetchone()[0])
    assert "rsi" in features
    engine.predict_symbol(db_path, "AAA", cache)
    assert table_counts(db_path)["ml_models"] == 1  # second call reuses the trained model
    assert engine.predict_symbol(db_path, "NONE", {})[0] == "HOLD"


def test_size_position_logs_the_decision(db_path):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(40, seed=5)))
    decision = engine.size_position(db_path, "AAA", budget=5_000, equity=100_000)
    assert decision.qty >= 1
    assert decision.qty * decision.price <= 5_000 + decision.price
    assert table_counts(db_path)["position_sizing"] == 1


def test_regime_without_data_is_unknown(db_path):
    assert engine.detect_market_regime(db_path, ["AAA"])["regime"] == "unknown"


def test_regime_with_spy_history(db_path):
    store_ohlcv(db_path, "SPY", make_ohlcv(np.linspace(400, 480, 120)))
    with connect(db_path) as conn:
        conn.execute("INSERT INTO macro VALUES ('VIXCLS', '2026-01-02', 35.0, 'vix')")
    result = engine.detect_market_regime(db_path, ["SPY"])
    assert result["regime"] == "high_vol" and result["vix"] == 35.0
    assert table_counts(db_path)["market_regime"] == 1


def test_private_exposure_weights_proxy_returns(db_path):
    store_ohlcv(db_path, "MSFT", make_ohlcv([100.0, 110.0]))
    store_ohlcv(db_path, "NVDA", make_ohlcv([100.0, 90.0]))
    rows = {r["company"]: r for r in engine.compute_private_exposure(db_path)}
    openai = rows["OpenAI"]
    # MSFT +10% at weight 0.40, NVDA -10% at 0.25; GOOG and CRM have no data and are skipped
    assert openai["return"] == pytest.approx(0.40 * 0.10 - 0.25 * 0.10)
    assert set(openai["details"]) == {"MSFT", "NVDA"}
    assert table_counts(db_path)["private_exposure"] == 5


def test_snapshot_portfolio_uses_the_previous_equity(db_path):
    engine.snapshot_portfolio(db_path, {"equity": 101_000, "cash": 50_000}, [{"market_value": 51_000}], 100_000)
    with connect(db_path) as conn:
        daily_pnl, daily_return, n = conn.execute(
            "SELECT daily_pnl, daily_return, num_positions FROM portfolio_snapshots"
        ).fetchone()
    assert daily_pnl == pytest.approx(1_000) and daily_return == pytest.approx(0.01) and n == 1
