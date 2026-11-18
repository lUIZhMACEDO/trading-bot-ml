"""Shared fixtures. Nothing here touches the network: prices are synthetic or read from small CSVs."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.db import connect, init_db

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail any test that tries to reach a non-loopback host."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError(f"network access is not allowed in unit tests (tried {host})")
        return real_connect(self, address)

    def no_create_connection(address, *args, **kwargs):
        raise RuntimeError(f"network access is not allowed in unit tests (tried {address[0]})")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", no_create_connection)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Run each test in an empty directory with no trading-bot variables set."""
    for name in (
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "FRED_API_KEY", "DB_PATH", "MAX_BUDGET", "MAX_POSITIONS",
        "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "TX_COST_BPS", "SLIPPAGE_BPS", "DASHBOARD_HOST", "DASHBOARD_PORT",
    ):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def make_ohlcv(close, start: str = "2025-01-02", volume=1_000_000) -> pd.DataFrame:
    """Daily bars around a close path, with ``date`` as a column like the database returns."""
    close = np.asarray(close, dtype=float)
    volume = np.broadcast_to(np.asarray(volume, dtype=float), close.shape)
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close)).strftime("%Y-%m-%d"),
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        }
    )


def random_walk(n: int = 300, seed: int = 0, drift: float = 0.0, vol: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))


def store_ohlcv(db_path, symbol: str, df: pd.DataFrame) -> None:
    rows = [(symbol, r.date, r.open, r.high, r.low, r.close, int(r.volume), 0.0) for r in df.itertuples(index=False)]
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def sample_csv() -> Path:
    return EXAMPLES / "sample_prices.csv"


class FakeBroker:
    """Stands in for ``PaperBroker``: records orders, serves canned account and positions."""

    def __init__(self, positions=None, equity=100_000.0):
        self.positions = list(positions or [])
        self.equity = equity
        self.orders: list[dict] = []
        self.closed: list[str] = []
        self.data = None

    def get_account(self):
        return {"equity": self.equity, "cash": self.equity / 2, "buying_power": self.equity,
                "portfolio_value": self.equity, "pnl": self.equity - 100_000}  # fmt: skip

    def get_positions(self):
        return list(self.positions)

    def get_orders(self, status="all", limit=20):
        return self.orders[:limit]

    def buy(self, symbol, qty=None, order_type="market", limit_price=None, strategy="manual", **_):
        order = {"symbol": symbol, "side": "buy", "qty": qty, "order_type": order_type, "limit_price": limit_price,
                 "strategy": strategy, "status": "accepted"}  # fmt: skip
        self.orders.append(order)
        return order

    def sell(self, symbol, qty=None, order_type="market", limit_price=None, strategy="manual", **_):
        order = {"symbol": symbol, "side": "sell", "qty": qty, "order_type": order_type, "limit_price": limit_price,
                 "strategy": strategy, "status": "accepted"}  # fmt: skip
        self.orders.append(order)
        return order

    def close_position(self, symbol):
        self.closed.append(symbol)
        return {"status": "closed", "symbol": symbol}

    def close_all(self):
        self.closed.append("*")


def position(symbol, pnl_pct, market_value=1_000.0):
    return {"symbol": symbol, "qty": 10, "avg_price": 100, "current_price": 100 * (1 + pnl_pct / 100),
            "pnl": pnl_pct * 10, "pnl_pct": pnl_pct, "market_value": market_value}  # fmt: skip
