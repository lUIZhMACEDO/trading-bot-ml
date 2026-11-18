"""PaperBroker against fake alpaca-py clients. Nothing here reaches Alpaca."""

from types import SimpleNamespace

import pytest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from trading_bot.backtest.costs import CostModel
from trading_bot.broker import alpaca as broker_mod
from trading_bot.broker.alpaca import BrokerError, PaperBroker, build_order_request
from trading_bot.data.db import connect


def test_build_order_request_variants():
    limit = build_order_request("AAPL", "buy", qty=5, order_type="limit", limit_price="195.5")
    assert isinstance(limit, LimitOrderRequest) and limit.limit_price == 195.5
    assert limit.side == OrderSide.BUY and limit.time_in_force == TimeInForce.DAY

    notional = build_order_request("AAPL", "sell", notional=1000)
    assert isinstance(notional, MarketOrderRequest) and notional.notional == 1000 and notional.side == OrderSide.SELL

    shares = build_order_request("AAPL", "buy", qty=3)
    assert isinstance(shares, MarketOrderRequest) and shares.qty == 3


class FakeTrading:
    def __init__(self, fail_account=False):
        self.fail_account = fail_account
        self.submitted = []

    def get_account(self):
        if self.fail_account:
            raise RuntimeError("unauthorized")
        return SimpleNamespace(equity="101000", cash="50000", buying_power="200000", portfolio_value="101000")

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return SimpleNamespace(id="order-1", status="accepted")

    def get_all_positions(self):
        return [
            SimpleNamespace(
                symbol="AAPL",
                qty="10",
                avg_entry_price="190",
                current_price="200",
                unrealized_pl="100",
                unrealized_plpc="0.0526",
                market_value="2000",
            )
        ]


@pytest.fixture
def fake_alpaca(monkeypatch):
    """Replace the alpaca-py client classes and record how they were built."""
    created = {}

    def install(trading):
        def trading_client(**kwargs):
            created.update(kwargs)
            return trading

        monkeypatch.setattr("alpaca.trading.client.TradingClient", trading_client)
        monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", lambda **kw: object())
        return created

    return install


def test_connect_is_always_paper(fake_alpaca, db_path):
    created = fake_alpaca(FakeTrading())
    broker = PaperBroker.connect("PK" + "x" * 18, "s" * 40, db_path)
    assert created["paper"] is True
    assert broker.get_account()["pnl"] == pytest.approx(1_000)


def test_connect_with_bad_keys_raises(fake_alpaca, db_path):
    fake_alpaca(FakeTrading(fail_account=True))
    with pytest.raises(BrokerError, match="unauthorized"):
        PaperBroker.connect("PK" + "x" * 18, "s" * 40, db_path)


def test_orders_are_logged_with_costs(db_path):
    broker = PaperBroker(FakeTrading(), None, db_path, CostModel(5, 2))
    out = broker.buy("AAPL", qty=10, order_type="limit", limit_price=200.0, strategy="momentum")
    assert out["status"] == "accepted" and out["tx_cost"] == pytest.approx(1.0)  # 5 bps of $2,000
    broker.sell("AAPL", qty=10)  # market order, no known price yet
    with connect(db_path) as conn:
        rows = conn.execute("SELECT side, qty, price, strategy, tx_cost, slippage FROM trades ORDER BY id").fetchall()
    assert rows[0] == ("buy", 10, 200.0, "momentum", pytest.approx(1.0), pytest.approx(0.4))
    assert rows[1] == ("sell", 10, 0.0, "manual", 0.0, 0.0)


def test_failed_order_returns_an_error(db_path):
    class Rejecting(FakeTrading):
        def submit_order(self, order_data):
            raise RuntimeError("insufficient buying power")

    out = PaperBroker(Rejecting(), None, db_path).buy("AAPL", qty=1)
    assert out == {"error": "insufficient buying power"}


def test_positions_are_plain_dicts(db_path):
    (p,) = PaperBroker(FakeTrading(), None, db_path).get_positions()
    assert p["symbol"] == "AAPL" and p["pnl_pct"] == pytest.approx(5.26)


def test_module_has_no_live_trading_switch():
    with open(broker_mod.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert "paper=True" in source and "paper=False" not in source
