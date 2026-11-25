"""The FastAPI app with a fake broker, through Starlette's in-process test client."""

import pytest
from fastapi.testclient import TestClient

from conftest import FakeBroker, make_ohlcv, random_walk, store_ohlcv
from trading_bot.config import Settings
from trading_bot.dashboard.app import create_app

KEYS = {"alpaca_api_key": "PK" + "x" * 18, "alpaca_secret_key": "s" * 40}
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def broker():
    return FakeBroker()


def make_client(tmp_path, broker, with_keys=False, **settings):
    db = tmp_path / "dash.db"
    cfg = Settings(db_path=db, **(KEYS if with_keys else {}), **settings)
    seen = []

    def connect_broker(api_key, secret_key, db_path, costs):
        seen.append((api_key, secret_key))
        if secret_key == "wrong" * 8:
            raise RuntimeError("Alpaca rejected the paper keys")
        return broker

    app = create_app(cfg, connect_broker=connect_broker)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    client.seen = seen
    return client


def test_index_serves_the_dashboard(tmp_path, broker):
    with make_client(tmp_path, broker) as client:
        html = client.get("/").text
    assert "<title>Trading Bot</title>" in html


def test_not_connected_until_keys_are_given(tmp_path, broker):
    with make_client(tmp_path, broker) as client:
        assert client.get("/api/status").json() == {"has_keys": False, "connected": False}
        assert client.get("/api/account").status_code == 401
        assert client.post("/api/trade", json={"symbol": "AAPL"}).status_code == 401


def test_auto_connects_with_keys_from_the_environment(tmp_path, broker):
    with make_client(tmp_path, broker, with_keys=True) as client:
        assert client.get("/api/status").json()["connected"] is True
        assert client.get("/api/account").json()["equity"] == 100_000


def test_connect_saves_keys_to_env(tmp_path, broker):
    with make_client(tmp_path, broker) as client:
        body = {"api_key": "PK" + "y" * 18, "secret_key": "z" * 40, "budget": "2500", "max_positions": "2"}
        out = client.post("/api/connect", json=body).json()
        assert out["status"] == "connected"
        env = (tmp_path / ".env").read_text()  # tests run with tmp_path as the working directory
    assert "ALPACA_API_KEY=PK" + "y" * 18 in env and "MAX_BUDGET=2500.0" in env


def test_connect_with_bad_keys_does_not_write_env(tmp_path, broker):
    with make_client(tmp_path, broker) as client:
        r = client.post("/api/connect", json={"api_key": "PK" + "y" * 18, "secret_key": "wrong" * 8})
    assert r.status_code == 400 and "rejected" in r.json()["error"]
    assert not (tmp_path / ".env").exists()


def test_empty_connect_reuses_environment_keys(tmp_path, broker):
    with make_client(tmp_path, broker, with_keys=True) as client:
        assert client.post("/api/connect", json={}).json()["status"] == "connected"
        assert client.seen[-1] == (KEYS["alpaca_api_key"], KEYS["alpaca_secret_key"])
    assert not (tmp_path / ".env").exists()


def test_trade_close_and_orders(tmp_path, broker):
    with make_client(tmp_path, broker, with_keys=True) as client:
        out = client.post("/api/trade", json={"symbol": "aapl", "qty": "5", "side": "sell", "limit_price": None})
        assert out.json()["symbol"] == "AAPL"
        assert broker.orders[-1]["side"] == "sell" and broker.orders[-1]["qty"] == 5
        assert client.post("/api/close/msft", headers=JSON).json()["symbol"] == "MSFT"
        assert client.post("/api/close_all", headers=JSON).json() == {"status": "all closed"}
        assert len(client.get("/api/orders").json()) == 1
    assert broker.closed == ["MSFT", "*"]


def test_cross_site_form_posts_are_blocked(tmp_path, broker):
    with make_client(tmp_path, broker, with_keys=True) as client:
        # what a hidden <form method=post> on another site would send
        r = client.post("/api/close_all", data={"x": "1"})
        assert r.status_code == 415
        r = client.post("/api/close_all", headers={**JSON, "Origin": "https://evil.example"})
        assert r.status_code == 403
        assert client.post("/api/close_all", headers={**JSON, "Origin": "http://127.0.0.1:8000"}).status_code == 200
    assert broker.closed == ["*"]


def test_unknown_host_header_is_rejected(tmp_path, broker):
    with make_client(tmp_path, broker) as client:
        assert client.get("/api/status", headers={"Host": "attacker.example"}).status_code == 400


def test_analysis_endpoints(tmp_path, broker):
    with make_client(tmp_path, broker) as client:
        store_ohlcv(tmp_path / "dash.db", "AAPL", make_ohlcv(random_walk(260, seed=9)))
        assert client.get("/api/analyze/aapl").json()["symbol"] == "AAPL"
        assert client.get("/api/analyze/none").status_code == 404
        assert [r["symbol"] for r in client.get("/api/scan").json()] == ["AAPL"]
        backtest = client.get("/api/backtest/AAPL/trend").json()
        assert backtest["strategy"] == "golden_cross" and "total_return" in backtest
        assert len(client.get("/api/ohlcv/AAPL?days=10").json()) == 10
        assert client.get("/api/dbstats").json()["ohlcv"] == 260
        assert "AAPL" in client.get("/api/watchlist").json()


def test_auto_trader_lifecycle(tmp_path, broker):
    with make_client(tmp_path, broker, with_keys=True, scan_interval_seconds=3600) as client:
        r = client.post("/api/auto/start", json={"strategy": "bogus"})
        assert r.status_code == 400
        started = client.post("/api/auto/start", json={"strategy": "trend", "budget": "5000"}).json()
        assert started == {"status": "started", "strategy": "golden_cross", "budget": 5000.0}
        status = client.get("/api/auto/status").json()
        assert status["running"] is True and status["stop_loss"] == pytest.approx(2.0)
        assert client.post("/api/auto/stop", headers=JSON).json() == {"status": "stopped"}
        assert client.get("/api/auto/status").json() == {"running": False}
