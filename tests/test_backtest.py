"""Backtest cost and slippage math, fill timing and the metrics."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import max_drawdown, round_trip_returns, sharpe_ratio, win_rate


def test_cost_rate_is_commission_plus_slippage():
    costs = CostModel(commission_bps=5, slippage_bps=2)
    assert costs.rate == pytest.approx(0.0007)
    assert costs.commission(10_000) == pytest.approx(5.0)
    assert costs.slippage(10_000) == pytest.approx(2.0)
    assert costs.total(10_000) == pytest.approx(7.0)


def test_buy_and_sell_cash_flows():
    costs = CostModel(5, 2)
    assert costs.buy_cash_out(100, 50.0) == pytest.approx(5_003.5)
    assert costs.sell_cash_in(100, 50.0) == pytest.approx(4_996.5)
    free = CostModel(0, 0)
    assert free.buy_cash_out(100, 50.0) == free.sell_cash_in(100, 50.0) == 5_000


def scripted(signals: dict[int, tuple[str, float]]):
    """A strategy that returns the given signal on bar index i and HOLD everywhere else."""

    def fn(window: pd.DataFrame):
        signal, score = signals.get(len(window) - 1, ("HOLD", 0))
        return signal, score, "scripted"

    return fn


def price_frame(n=80):
    df = make_ohlcv(np.full(n, 100.0))
    df["open"] = 100.0 + np.arange(n)  # a distinct open on every bar so fills are traceable
    return df


def test_signals_fill_at_the_next_open_with_costs():
    df = price_frame()
    strategy = scripted({40: ("BUY", 0.8), 50: ("SELL", -0.8)})
    result = run_backtest(df, strategy, initial_capital=100_000, costs=CostModel(5, 2))

    buy, sell = result.trades
    assert (buy["side"], buy["price"], buy["date"]) == ("BUY", 141.0, df.at[41, "date"])
    assert (sell["side"], sell["price"], sell["date"]) == ("SELL", 151.0, df.at[51, "date"])

    qty = int(100_000 * 0.95 / 141.0)
    assert buy["qty"] == sell["qty"] == qty
    cash = 100_000 - qty * 141.0 * 1.0007 + qty * 151.0 * 0.9993
    assert result.final_equity == pytest.approx(round(cash, 2))
    assert result.total_return == pytest.approx(round((cash / 100_000 - 1) * 100, 2))
    assert result.tx_costs == pytest.approx(round(qty * 141.0 * 0.0007 + qty * 151.0 * 0.0007, 2))
    # win rate and average trade use fill prices, before costs
    assert result.win_rate == 100.0
    assert result.avg_trade_pnl == pytest.approx(round((151 / 141 - 1) * 100, 2))


def test_costs_reduce_the_result_by_exactly_the_cost_drag():
    df = price_frame()
    strategy = scripted({40: ("BUY", 0.8), 50: ("SELL", -0.8)})
    free = run_backtest(df, strategy, costs=CostModel(0, 0))
    paid = run_backtest(df, strategy, costs=CostModel(5, 2))
    qty = free.trades[0]["qty"]
    assert free.final_equity - paid.final_equity == pytest.approx(qty * (141 + 151) * 0.0007, abs=0.01)


def test_weak_signals_and_duplicate_entries_are_ignored():
    df = price_frame()
    strategy = scripted({35: ("BUY", 0.3), 40: ("BUY", 0.5), 45: ("BUY", 0.9), 50: ("SELL", -0.2)})
    result = run_backtest(df, strategy)
    assert [t["side"] for t in result.trades] == ["BUY"]  # 0.3 is not > 0.3, the second BUY is a no-op
    assert result.trades[0]["price"] == 141.0


def test_open_position_is_marked_at_the_last_close():
    df = price_frame()
    result = run_backtest(df, scripted({40: ("BUY", 0.8)}), costs=CostModel(0, 0))
    qty = result.trades[0]["qty"]
    expected = 100_000 - qty * 141.0 + qty * 100.0  # bought at the 141 open, closes are all 100
    assert result.final_equity == pytest.approx(expected)
    assert result.win_rate == 0.0 and result.avg_trade_pnl == 0.0  # no completed round trip


def test_no_signals_before_warmup_and_needs_60_bars():
    seen = []

    def spy(window):
        seen.append(len(window))
        return "HOLD", 0, ""

    run_backtest(price_frame(80), spy)
    assert min(seen) == 31  # first decision is on bar 30, with 31 rows of history
    assert max(seen) == 79  # the last bar has no next open to fill at
    assert run_backtest(price_frame(59), spy) is None


def test_strategy_never_sees_the_bar_it_fills_on():
    df = price_frame()
    checked = []

    def strategy(window):
        i = len(window) - 1
        assert window["date"].iloc[-1] == df.at[i, "date"]
        checked.append(i)
        return "HOLD", 0, ""

    run_backtest(df, strategy)
    assert checked


def test_max_drawdown():
    assert max_drawdown([100, 120, 90, 130, 65]) == pytest.approx(0.5)
    assert max_drawdown([100, 101, 102]) == 0.0
    assert max_drawdown([]) == 0.0


def test_sharpe_ratio():
    assert sharpe_ratio([100.0] * 40) == 0.0
    assert sharpe_ratio([100, 101, 102]) == 0.0  # too short
    equity = 100 * np.cumprod(np.r_[1.0, np.tile([1.01, 0.995], 30)])
    returns = pd.Series(equity).pct_change().dropna()
    assert sharpe_ratio(equity) == pytest.approx(returns.mean() / returns.std() * np.sqrt(252))


def test_round_trips_and_win_rate():
    trades = [{"price": 100}, {"price": 110}, {"price": 50}, {"price": 45}, {"price": 70}]
    rt = round_trip_returns(trades)
    assert rt == pytest.approx([0.10, -0.10])
    assert win_rate(rt) == 0.5
    assert win_rate([]) == 0.0
