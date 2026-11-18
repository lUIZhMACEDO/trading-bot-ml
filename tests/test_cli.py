"""The quick-start commands from the README, run against the committed examples/ file."""

import pytest

from conftest import EXAMPLES, FakeBroker
from trading_bot.cli import Shell, main
from trading_bot.config import Settings

SAMPLE = str(EXAMPLES / "sample_prices.csv")

EXPECTED_ANALYZE = """\
SAMPLE  composite +0.60  Neutral  (buy 1, sell 0, hold 4)
  rsi_macd      HOLD +0.00  RSI=75, MACD_hist=0.340
  bollinger     HOLD +0.00  Hurst=0.67, trending regime, skip reversion
  golden_cross  HOLD +0.00  SMA50=65, SMA200=77
  momentum      HOLD +0.00  Mom5=1.4%, Mom21=10.2%, ADX=59
  hurst         BUY  +0.60  Trending regime (H=0.67): EMA9 > EMA21, ride trend
"""

EXPECTED_BACKTEST = """\
backtest momentum on SAMPLE  2025-09-30 to 2026-10-28
  total return  +10.99%
  sharpe        0.79
  max drawdown  11.66%
  win rate      100.0%
  fills         3
  avg trade     +2.08%
  costs         $202.09
  2026-01-02  BUY   1210 @ 78.51  cost $66.50
  2026-07-06  SELL  1210 @ 80.14  cost $67.88
  2026-09-22  BUY   1466 @ 65.98  cost $67.71
"""


def test_analyze_matches_the_readme(capsys):
    assert main(["analyze", SAMPLE]) == 0
    assert capsys.readouterr().out == EXPECTED_ANALYZE


def test_backtest_matches_the_readme(capsys):
    assert main(["backtest", SAMPLE, "--strategy", "momentum", "--trades"]) == 0
    assert capsys.readouterr().out == EXPECTED_BACKTEST


def test_backtest_without_costs_does_better(capsys):
    main(["backtest", SAMPLE, "--strategy", "momentum", "--commission-bps", "0", "--slippage-bps", "0"])
    out = capsys.readouterr().out
    assert "costs         $0.00" in out
    total = float(out.split("total return")[1].split("%")[0])
    assert total > 10.99


def test_bad_input_returns_an_error_code(capsys, tmp_path):
    assert main(["backtest", SAMPLE, "--strategy", "nope"]) == 2
    assert "unknown strategy" in capsys.readouterr().err
    assert main(["analyze", str(tmp_path / "missing.csv")]) == 2


@pytest.fixture
def shell(db_path):
    return Shell(FakeBroker(), Settings(db_path=db_path, scan_interval_seconds=3600))


def test_shell_commands(shell, capsys):
    shell.dispatch("buy AAPL 10 195.5")
    assert shell.broker.orders[-1] == {"symbol": "AAPL", "side": "buy", "qty": 10, "order_type": "limit",
                                       "limit_price": 195.5, "strategy": "manual", "status": "accepted"}  # fmt: skip
    shell.dispatch("sell aapl 3")
    assert shell.broker.orders[-1]["order_type"] == "market"
    shell.dispatch("analyze ZZZ")
    shell.dispatch("dbstats")
    shell.dispatch("buy AAPL")  # missing quantity
    shell.dispatch("frobnicate")
    out = capsys.readouterr().out
    assert "No data for ZZZ" in out
    assert "ohlcv" in out
    assert "Bad arguments for buy" in out
    assert "Unknown command: frobnicate" in out


def test_shell_auto_start_maps_aliases(shell):
    shell.dispatch("auto start momentum")
    assert shell.auto_trader.strategy == "momentum"
    shell.dispatch("auto start trend")
    assert shell.auto_trader.strategy == "golden_cross"
    assert shell.dispatch("quit") is False
    assert not shell.auto_trader.running
