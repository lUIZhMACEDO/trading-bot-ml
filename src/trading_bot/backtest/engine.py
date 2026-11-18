"""Long-only, one-position-at-a-time backtester.

On each bar i the strategy only sees rows 0..i. A signal on bar i fills at the open of bar i+1,
and every fill pays commission and slippage from the ``CostModel``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.metrics import max_drawdown, round_trip_returns, sharpe_ratio, win_rate
from trading_bot.strategies import StrategyFn

WARMUP_BARS = 30  # no signals before this many bars of history
MIN_SCORE = 0.3  # signals at or below this strength are ignored
CASH_FRACTION = 0.95  # share of cash put into each new position


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    total_return: float  # percent
    sharpe: float
    max_drawdown: float  # percent
    win_rate: float  # percent of round trips with a gain
    total_trades: int  # fills, so a round trip counts twice
    avg_trade_pnl: float  # percent, mean round-trip return from fill prices
    tx_costs: float  # dollars of commission plus slippage
    final_equity: float
    trades: list[dict]

    def summary(self) -> dict:
        """The headline numbers without the trade list (what the API and the database store)."""
        out = asdict(self)
        out.pop("trades")
        return out


def run_backtest(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    *,
    strategy: str = "",
    symbol: str = "",
    initial_capital: float = 100_000.0,
    costs: CostModel | None = None,
) -> BacktestResult | None:
    """Replay ``strategy_fn`` over ``df`` (which must already have its technical columns).

    Returns None when there are fewer than 60 bars.
    """
    if len(df) < 60:
        return None
    costs = costs or CostModel()
    df = df.reset_index(drop=True)

    cash, shares = float(initial_capital), 0
    trades: list[dict] = []
    equity_curve: list[float] = []

    for i in range(WARMUP_BARS, len(df) - 1):
        signal, score, _ = strategy_fn(df.iloc[: i + 1])
        close = float(df.at[i, "close"])
        next_open = float(df.at[i + 1, "open"])
        next_date = str(df.at[i + 1, "date"])
        equity_curve.append(cash + shares * close)

        if signal == "BUY" and shares == 0 and abs(score) > MIN_SCORE:
            qty = int(cash * CASH_FRACTION / next_open)
            if qty > 0:
                cash -= costs.buy_cash_out(qty, next_open)
                shares = qty
                trades.append(
                    {"date": next_date, "side": "BUY", "price": next_open, "qty": qty,
                     "tx_cost": costs.total(qty * next_open)}
                )  # fmt: skip
        elif signal == "SELL" and shares > 0 and abs(score) > MIN_SCORE:
            cash += costs.sell_cash_in(shares, next_open)
            trades.append(
                {"date": next_date, "side": "SELL", "price": next_open, "qty": shares,
                 "tx_cost": costs.total(shares * next_open)}
            )  # fmt: skip
            shares = 0

    final_equity = cash + shares * float(df["close"].iloc[-1])
    round_trips = round_trip_returns(trades)
    return BacktestResult(
        strategy=strategy,
        symbol=symbol,
        start_date=str(df.at[WARMUP_BARS, "date"]),
        end_date=str(df["date"].iloc[-1]),
        total_return=round((final_equity / initial_capital - 1) * 100, 2),
        sharpe=round(sharpe_ratio(equity_curve), 2),
        max_drawdown=round(max_drawdown(equity_curve) * 100, 2),
        win_rate=round(win_rate(round_trips) * 100, 1),
        total_trades=len(trades),
        avg_trade_pnl=round(float(np.mean(round_trips)) * 100 if round_trips else 0.0, 2),
        tx_costs=round(sum(t["tx_cost"] for t in trades), 2),
        final_equity=round(final_equity, 2),
        trades=trades,
    )
