"""Thin wrapper around Alpaca's trading and market-data clients, locked to the paper endpoint.

There is no switch for live trading. ``PaperBroker`` always builds ``TradingClient(paper=True)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from trading_bot.backtest.costs import CostModel
from trading_bot.config import PAPER_STARTING_EQUITY
from trading_bot.data.db import connect

log = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """Raised when Alpaca rejects the keys at connect time."""


def build_order_request(symbol: str, side: str, qty=None, notional=None, order_type="market", limit_price=None):
    """Build the alpaca-py order request: a DAY limit order, a notional market order or a share market order."""
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
    if order_type == "limit" and limit_price:
        return LimitOrderRequest(
            symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY, limit_price=float(limit_price)
        )
    if notional:
        return MarketOrderRequest(
            symbol=symbol, notional=float(notional), side=order_side, time_in_force=TimeInForce.DAY
        )
    return MarketOrderRequest(symbol=symbol, qty=float(qty), side=order_side, time_in_force=TimeInForce.DAY)


class PaperBroker:
    """Account, positions and orders on an Alpaca paper account. Every order is logged to SQLite."""

    def __init__(self, trading, data, db_path: Path | str, costs: CostModel | None = None):
        self.trading = trading
        self.data = data
        self.db_path = db_path
        self.costs = costs or CostModel()

    @classmethod
    def connect(cls, api_key: str, secret_key: str, db_path: Path | str, costs: CostModel | None = None) -> PaperBroker:
        """Create the paper clients and check the keys by fetching the account."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        try:
            account = trading.get_account()
        except Exception as exc:
            raise BrokerError(f"Alpaca rejected the paper keys: {exc}") from exc
        log.info("Connected to Alpaca paper trading, equity $%s", f"{float(account.equity):,.2f}")
        return cls(trading, data, db_path, costs)

    def get_account(self) -> dict:
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "pnl": float(acct.equity) - PAPER_STARTING_EQUITY,
        }

    def get_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "pnl": float(p.unrealized_pl),
                "pnl_pct": float(p.unrealized_plpc) * 100,
                "market_value": float(p.market_value),
            }
            for p in self.trading.get_all_positions()
        ]

    def buy(self, symbol, qty=None, notional=None, order_type="market", limit_price=None, strategy="manual") -> dict:
        return self._place_order(symbol, "buy", qty, notional, order_type, limit_price, strategy)

    def sell(self, symbol, qty=None, notional=None, order_type="market", limit_price=None, strategy="manual") -> dict:
        return self._place_order(symbol, "sell", qty, notional, order_type, limit_price, strategy)

    def _place_order(self, symbol, side, qty, notional, order_type, limit_price, strategy) -> dict:
        try:
            order = self.trading.submit_order(
                order_data=build_order_request(symbol, side, qty, notional, order_type, limit_price)
            )
        except Exception as exc:
            log.error("Order failed: %s", exc)
            return {"error": str(exc)}
        log.info("%s %s %s (%s) -> %s", side.upper(), qty or notional, symbol, order_type, order.status)

        # Market orders have no known price until they fill, so their logged cost is 0.
        notional_value = float(limit_price or 0) * float(qty or 0)
        tx_cost = self.costs.commission(notional_value)
        slippage = self.costs.slippage(notional_value)
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trades (symbol, side, qty, price, order_type, strategy, status,
                   alpaca_id, tx_cost, slippage) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (symbol, side, qty or 0, float(limit_price or 0), order_type, strategy, str(order.status),
                 str(order.id), tx_cost, slippage),
            )  # fmt: skip
        return {"id": str(order.id), "symbol": symbol, "side": side, "qty": qty, "status": str(order.status),
                "tx_cost": tx_cost}  # fmt: skip

    def close_position(self, symbol: str) -> dict:
        try:
            self.trading.close_position(symbol)
        except Exception as exc:
            return {"error": str(exc)}
        log.info("Closed position %s", symbol)
        return {"status": "closed", "symbol": symbol}

    def close_all(self) -> None:
        self.trading.close_all_positions(cancel_orders=True)
        log.info("Closed all positions")

    def get_orders(self, status: str = "all", limit: int = 20) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        query = QueryOrderStatus.ALL if status == "all" else QueryOrderStatus.OPEN
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side),
                "qty": float(o.qty) if o.qty else 0,
                "type": str(o.type),
                "status": str(o.status),
                "filled_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "created": str(o.created_at),
            }
            for o in self.trading.get_orders(filter=GetOrdersRequest(status=query, limit=limit))
        ]
