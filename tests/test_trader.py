"""One auto-trader cycle at a time, with a fake broker and fake signals."""

import pytest

from conftest import FakeBroker, make_ohlcv, position, random_walk, store_ohlcv
from trading_bot import engine
from trading_bot.trader import AutoTrader


def trader(broker, db_path, **kw):
    kw.setdefault("watchlist", ["AAA", "BBB"])
    return AutoTrader(broker, db_path, **kw)


def fake_composites(monkeypatch, scores, regime="bull"):
    def analyze(db, symbol):
        return {"symbol": symbol, "composite": scores[symbol], "rating": "x"} if symbol in scores else None

    monkeypatch.setattr(engine, "analyze_symbol", analyze)
    monkeypatch.setattr(engine, "detect_market_regime", lambda *a: {"regime": regime})


def test_unknown_strategy_is_rejected(db_path):
    with pytest.raises(ValueError):
        AutoTrader(FakeBroker(), db_path, strategy="nonsense")


def test_stop_loss_and_take_profit(db_path):
    broker = FakeBroker([position("LOSS", -2.5), position("WIN", 6.0), position("OK", 1.0)])
    t = trader(broker, db_path, stop_loss_pct=2.0, take_profit_pct=5.0)
    assert t.apply_exits(broker.get_positions()) == ["LOSS", "WIN"]
    assert broker.closed == ["LOSS", "WIN"]


def test_no_new_buys_when_full(db_path, monkeypatch):
    broker = FakeBroker([position("AAA", 0.0), position("BBB", 0.0), position("CCC", 0.0)])
    monkeypatch.setattr(engine, "detect_market_regime", lambda *a: pytest.fail("should not scan"))
    assert trader(broker, db_path, max_positions=3).scan_and_trade() is None
    assert broker.orders == []


def test_buys_the_best_composite_with_its_strategy_tag(db_path, monkeypatch):
    for symbol in ("AAA", "BBB"):
        store_ohlcv(db_path, symbol, make_ohlcv(random_walk(40)))
    fake_composites(monkeypatch, {"AAA": 1.2, "BBB": 2.8})
    order = trader(FakeBroker(), db_path).scan_and_trade()
    assert order["symbol"] == "BBB" and order["strategy"] == "composite" and order["qty"] >= 1


def test_held_symbols_are_skipped(db_path, monkeypatch):
    fake_composites(monkeypatch, {"AAA": 1.2, "BBB": 2.8})
    order = trader(FakeBroker([position("BBB", 0.0)]), db_path).scan_and_trade()
    assert order["symbol"] == "AAA"


def test_negative_composites_are_never_bought(db_path, monkeypatch):
    fake_composites(monkeypatch, {"AAA": -3.0, "BBB": 0.5})
    broker = FakeBroker()
    assert trader(broker, db_path).scan_and_trade() is None
    assert broker.orders == []


def test_high_volatility_halves_the_size(db_path, monkeypatch):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(40)))
    fake_composites(monkeypatch, {"AAA": 2.0})
    normal = trader(FakeBroker(), db_path, watchlist=["AAA"]).scan_and_trade()["qty"]
    fake_composites(monkeypatch, {"AAA": 2.0}, regime="high_vol")
    halved = trader(FakeBroker(), db_path, watchlist=["AAA"]).scan_and_trade()["qty"]
    assert normal > 1 and halved == normal // 2


def test_single_strategy_mode(db_path, monkeypatch):
    store_ohlcv(db_path, "AAA", make_ohlcv(random_walk(40)))
    monkeypatch.setattr(engine, "detect_market_regime", lambda *a: {"regime": "bull"})
    monkeypatch.setattr(engine, "strategy_signal", lambda db, s, strat: ("BUY", 0.5, "test") if s == "AAA" else None)
    order = trader(FakeBroker(), db_path, strategy="momentum").scan_and_trade()
    assert order["symbol"] == "AAA" and order["strategy"] == "momentum"


def test_start_and_stop(db_path, monkeypatch):
    t = trader(FakeBroker(), db_path, scan_interval=3600)
    calls = []
    monkeypatch.setattr(t, "scan_and_trade", lambda: calls.append(1))
    monkeypatch.setattr(t, "snapshot_portfolio", lambda: None)
    t.start()
    assert t.running
    t.stop()
    t._thread.join(timeout=5)
    assert not t.running and calls == [1]
