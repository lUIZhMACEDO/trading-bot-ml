"""Command line entry point.

``python -m trading_bot`` with no arguments opens the interactive shell against Alpaca paper
trading. Two subcommands work offline on a price CSV and need no keys:

    python -m trading_bot analyze examples/sample_prices.csv
    python -m trading_bot backtest examples/sample_prices.csv --strategy momentum
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path

from trading_bot import engine
from trading_bot.backtest.costs import CostModel
from trading_bot.backtest.engine import BacktestResult, run_backtest
from trading_bot.config import ENV_PATH, Settings, load_settings, write_env_file
from trading_bot.data.db import init_db, table_counts
from trading_bot.data.ingest import ingest_fundamentals, ingest_macro, ingest_ohlcv
from trading_bot.data.loaders import load_price_csv
from trading_bot.data.universe import WATCHLIST
from trading_bot.features.technicals import add_technicals
from trading_bot.strategies import STRATEGIES, resolve_strategy, score_all

log = logging.getLogger(__name__)

HELP_TEXT = """
Manual trading
  buy AAPL 10              market buy 10 shares
  buy AAPL 10 195.50       limit buy at $195.50
  sell AAPL 10             market sell 10 shares
  close AAPL               close the whole position
  close_all                close every position

Portfolio
  account                  account summary
  positions                open positions
  orders                   recent orders

Data pipeline
  pipeline                 run the full daily pipeline
  download                 ingest 2 years of daily bars
  fundamentals             fetch P/E, FCF yield, ROE, analyst ratings
  macro                    fetch FRED series (needs FRED_API_KEY)
  technicals               recompute all indicators
  regime                   detect the market regime

Analysis and ML
  analyze AAPL             five-strategy breakdown
  scan                     composite score for the whole watchlist
  train AAPL               walk-forward ML training
  predict AAPL             ML ensemble signal
  backtest AAPL momentum   backtest one strategy with costs
  private                  private-company proxy exposure

Auto-trading
  auto start composite     all five strategies, composite score
  auto start ml            ML ensemble signals
  auto start momentum      a single strategy (rsi_macd, bollinger, golden_cross, momentum, hurst)
  auto stop                stop the auto-trader
  auto status              show its settings

System
  watchlist                tracked symbols
  dbstats                  row counts per table
  newkeys                  re-enter API keys
  help                     this list
  quit                     exit
"""


# ---------------------------------------------------------------------------
# Output formatting (shared by the shell and the offline subcommands)
# ---------------------------------------------------------------------------


def format_analysis(symbol: str, result: dict) -> str:
    lines = [
        f"{symbol}  composite {result['composite']:+.2f}  {result['rating']}  "
        f"(buy {result['buy']}, sell {result['sell']}, hold {result['hold']})"
    ]
    for s in result["strategies"]:
        lines.append(f"  {s['strategy']:<13} {s['signal']:<4} {s['score']:+.2f}  {s['reason']}")
    return "\n".join(lines)


def format_backtest(result: dict | BacktestResult) -> str:
    r = result.summary() if isinstance(result, BacktestResult) else result
    header = f"backtest {r['strategy']} on {r['symbol']}"
    if r.get("start_date"):
        header += f"  {r['start_date']} to {r['end_date']}"
    return "\n".join(
        [
            header,
            f"  total return  {r['total_return']:+.2f}%",
            f"  sharpe        {r['sharpe']:.2f}",
            f"  max drawdown  {r['max_drawdown']:.2f}%",
            f"  win rate      {r['win_rate']:.1f}%",
            f"  fills         {r['total_trades']}",
            f"  avg trade     {r['avg_trade_pnl']:+.2f}%",
            f"  costs         ${r['tx_costs']:,.2f}",
        ]
    )


# ---------------------------------------------------------------------------
# Offline subcommands
# ---------------------------------------------------------------------------


def analyze_csv(path: Path | str, symbol: str) -> dict:
    df = add_technicals(load_price_csv(path))
    result = score_all(df)
    return {"symbol": symbol, "composite": result.composite, "rating": result.rating, "buy": result.buy,
            "sell": result.sell, "hold": result.hold, "strategies": result.strategies}  # fmt: skip


def backtest_csv(path: Path | str, symbol: str, strategy: str, capital: float, costs: CostModel) -> BacktestResult:
    key = resolve_strategy(strategy)
    if key not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, pick one of {', '.join(STRATEGIES)}")
    df = add_technicals(load_price_csv(path))
    result = run_backtest(df, STRATEGIES[key][1], strategy=key, symbol=symbol, initial_capital=capital, costs=costs)
    if result is None:
        raise ValueError(f"{path}: need at least 60 rows to backtest")
    return result


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------


def prompt_for_keys(env_path: Path = ENV_PATH) -> Settings:
    """First-run setup: ask for paper keys and trading defaults and save them to ``.env``."""
    print("\nNo Alpaca paper keys found. Create them at https://app.alpaca.markets")
    print("(switch to Paper Trading, then API Keys -> Generate). They are saved to .env locally.\n")
    while len(api_key := input("Alpaca API key id: ").strip()) < 10:
        print("That looks too short. Key ids start with PK and are about 20 characters.")
    while len(secret := getpass.getpass("Alpaca secret key (hidden): ").strip()) < 20:
        print("That looks too short. Secrets are about 40 characters.")
    fred = input("FRED API key (optional, Enter to skip): ").strip()

    print("\nTrading defaults (Enter keeps the default)")
    values = {
        "ALPACA_API_KEY": api_key,
        "ALPACA_SECRET_KEY": secret,
        "FRED_API_KEY": fred,
        "MAX_BUDGET": input("Max budget per auto-trade [10000]: ").strip() or "10000",
        "MAX_POSITIONS": input("Max simultaneous positions [3]: ").strip() or "3",
        "STOP_LOSS_PCT": input("Stop loss % [2.0]: ").strip() or "2.0",
        "TAKE_PROFIT_PCT": input("Take profit % [5.0]: ").strip() or "5.0",
    }
    write_env_file(values, env_path)
    print(f"Saved to {env_path.resolve()} (git-ignored, do not share it).\n")
    return load_settings(env_path, override=True)


class Shell:
    """The ``trading-bot>`` prompt. Each command maps to a ``do_<name>`` method."""

    def __init__(self, broker, settings: Settings):
        self.broker = broker
        self.settings = settings
        self.db = settings.db_path
        self.auto_trader = None
        self.costs = CostModel(settings.tx_cost_bps, settings.slippage_bps)

    def run(self) -> None:
        print("Type 'help' for commands.")
        while True:
            try:
                line = input("trading-bot> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if self.dispatch(line) is False:
                break
        self.do_quit([])

    def dispatch(self, line: str):
        name, *args = line.split()
        handler = getattr(self, f"do_{name.lower()}", None)
        if handler is None:
            print(f"Unknown command: {line}. Type 'help'.")
            return None
        try:
            return handler(args)
        except (IndexError, ValueError) as exc:
            print(f"Bad arguments for {name}: {exc}")
            return None

    # Manual trading
    def _order(self, side: str, args: list[str]) -> None:
        symbol, qty = args[0].upper(), int(args[1])
        limit = float(args[2]) if len(args) > 2 else None
        place = self.broker.buy if side == "buy" else self.broker.sell
        result = place(symbol, qty=qty, order_type="limit" if limit else "market", limit_price=limit)
        print(json.dumps(result, indent=2))

    def do_buy(self, args):
        self._order("buy", args)

    def do_sell(self, args):
        self._order("sell", args)

    def do_close(self, args):
        print(json.dumps(self.broker.close_position(args[0].upper()), indent=2))

    def do_close_all(self, args):
        self.broker.close_all()

    # Portfolio
    def do_account(self, args):
        a = self.broker.get_account()
        print(f"  equity        ${a['equity']:>12,.2f}")
        print(f"  cash          ${a['cash']:>12,.2f}")
        print(f"  buying power  ${a['buying_power']:>12,.2f}")
        print(f"  P&L           ${a['pnl']:>+12,.2f}")

    def do_positions(self, args):
        positions = self.broker.get_positions()
        if not positions:
            print("  No open positions")
            return
        print(f"  {'Symbol':<8} {'Qty':>6} {'Avg':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8}")
        for p in positions:
            print(f"  {p['symbol']:<8} {p['qty']:>6.0f} {p['avg_price']:>10.2f} {p['current_price']:>10.2f} "
                  f"{p['pnl']:>+10.2f} {p['pnl_pct']:>+7.1f}%")  # fmt: skip

    def do_orders(self, args):
        for o in self.broker.get_orders(limit=10):
            when = o["created"][:19]
            print(f"  {when}  {o['side']:>4} {o['symbol']:<6} x{o['qty']:>4.0f}  {o['type']:>6}  {o['status']}")

    # Data pipeline
    def do_pipeline(self, args):
        engine.run_full_pipeline(self.broker, self.db, self.settings.fred_api_key)

    def do_download(self, args):
        ingest_ohlcv(self.broker.data, self.db, years=2)

    def do_fundamentals(self, args):
        ingest_fundamentals(self.db)

    def do_macro(self, args):
        ingest_macro(self.db, self.settings.fred_api_key)

    def do_technicals(self, args):
        engine.compute_and_store_technicals(self.db)

    def do_regime(self, args):
        r = engine.detect_market_regime(self.db)
        print(f"  regime       {r['regime']}")
        print(f"  VIX          {r['vix']:.1f}")
        print(f"  yield curve  {r['yield_curve']}")
        if "breadth" in r:
            print(f"  breadth      {r['breadth']:.0f}%")
            print(f"  confidence   {r['confidence']:.0%}")

    # Analysis and ML
    def do_analyze(self, args):
        symbol = args[0].upper()
        result = engine.analyze_symbol(self.db, symbol)
        print(format_analysis(symbol, result) if result else f"  No data for {symbol}. Run 'download' first.")

    def do_scan(self, args):
        print(f"  {'Symbol':<8} {'Score':>7} {'Rating':<12} {'Buy':>4} {'Sell':>5} {'Hold':>5}")
        for r in engine.scan(self.db):
            counts = f"{r['buy']:>4} {r['sell']:>5} {r['hold']:>5}"
            print(f"  {r['symbol']:<8} {r['composite']:>+7.2f} {r['rating']:<12} {counts}")

    def do_train(self, args):
        symbol = args[0].upper()
        model = engine.train_symbol(self.db, symbol)
        if model is None:
            print(f"  Not enough data to train {symbol}.")
            return
        engine.MODEL_CACHE[symbol] = model
        folds = ", ".join(f"{s:.1%}" for s in model.fold_scores)
        print(f"  {symbol}: {len(model.features)} features, mean fold accuracy {model.accuracy:.1%} ({folds})")

    def do_predict(self, args):
        symbol = args[0].upper()
        signal, _, reason = engine.predict_symbol(self.db, symbol)
        print(f"  {symbol}: {signal}, {reason}")

    def do_backtest(self, args):
        symbol, strategy = args[0].upper(), resolve_strategy(args[1])
        result = engine.backtest_symbol(self.db, symbol, strategy, self.costs)
        print(format_backtest(result) if result else "  Unknown strategy or not enough data. Run 'download' first.")

    def do_private(self, args):
        for row in engine.compute_private_exposure(self.db):
            print(f"  {row['company']:<12} exposure {row['exposure']:+.4f}  return {row['return']:+.6f}")
            for proxy, info in row["details"].items():
                print(f"    {proxy}: weight {info['weight']:.0%}, return {info['return']:+.2f}%")

    # Auto-trading
    def do_auto(self, args):
        from trading_bot.trader import AutoTrader

        sub = args[0] if args else ""
        if sub == "start":
            strategy = resolve_strategy(args[1]) if len(args) > 1 else "composite"
            if self.auto_trader and self.auto_trader.running:
                self.auto_trader.stop()
            s = self.settings
            self.auto_trader = AutoTrader(
                self.broker, self.db, strategy, s.max_budget, s.max_positions, s.stop_loss_pct, s.take_profit_pct,
                s.scan_interval_seconds,
            )  # fmt: skip
            self.auto_trader.start()
        elif sub == "stop":
            if self.auto_trader and self.auto_trader.running:
                self.auto_trader.stop()
            else:
                print("  Not running")
        elif sub == "status":
            t = self.auto_trader
            if t and t.running:
                print(f"  running, strategy {t.strategy}, budget ${t.budget:,.0f}, max positions {t.max_positions}")
                print(f"  stop loss {t.stop_loss * 100:.1f}%, take profit {t.take_profit * 100:.1f}%")
            else:
                print("  Not running")
        else:
            print("  Usage: auto start <strategy> | auto stop | auto status")

    # System
    def do_watchlist(self, args):
        print(f"  Tracking {len(WATCHLIST)} symbols: {', '.join(WATCHLIST)}")

    def do_dbstats(self, args):
        print(f"  Database: {self.db}")
        for table, count in table_counts(self.db).items():
            print(f"  {table:<22} {count:>8,} rows")

    def do_newkeys(self, args):
        prompt_for_keys()
        print("  New keys saved. Restart to reconnect.")

    def do_help(self, args):
        print(HELP_TEXT)

    def do_quit(self, args):
        if self.auto_trader and self.auto_trader.running:
            self.auto_trader.stop()
        return False

    do_exit = do_q = do_quit


def run_shell() -> int:
    from trading_bot.broker.alpaca import BrokerError, PaperBroker

    settings = load_settings()
    if not settings.has_alpaca_keys:
        settings = prompt_for_keys()
    init_db(settings.db_path)
    try:
        broker = PaperBroker.connect(
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
            settings.db_path,
            CostModel(settings.tx_cost_bps, settings.slippage_bps),
        )
    except BrokerError as exc:
        print(f"{exc}\nCheck the keys in .env (or run with fresh ones after deleting it).")
        return 1

    shell = Shell(broker, settings)
    shell.do_account([])
    if table_counts(settings.db_path)["ohlcv"] == 0:
        answer = input("\nNo market data yet. Run the full pipeline now? It takes about two minutes. [Y/n] ")
        if answer.strip().lower() in ("", "y", "yes"):
            shell.do_pipeline([])
    shell.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading_bot", description="Alpaca paper-trading bot.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("shell", help="interactive shell against Alpaca paper trading (the default)")

    analyze = sub.add_parser("analyze", help="score the five strategies on the last bar of a price CSV")
    analyze.add_argument("csv", type=Path)
    analyze.add_argument("--symbol", default="SAMPLE")

    backtest = sub.add_parser("backtest", help="backtest one strategy on a price CSV")
    backtest.add_argument("csv", type=Path)
    backtest.add_argument("--strategy", default="momentum", help=", ".join(STRATEGIES))
    backtest.add_argument("--symbol", default="SAMPLE")
    backtest.add_argument("--capital", type=float, default=100_000.0)
    backtest.add_argument("--commission-bps", type=float, default=5.0)
    backtest.add_argument("--slippage-bps", type=float, default=2.0)
    backtest.add_argument("--trades", action="store_true", help="also print every fill")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command in (None, "shell"):
        return run_shell()
    try:
        if args.command == "analyze":
            print(format_analysis(args.symbol, analyze_csv(args.csv, args.symbol)))
        else:
            costs = CostModel(args.commission_bps, args.slippage_bps)
            result = backtest_csv(args.csv, args.symbol, args.strategy, args.capital, costs)
            print(format_backtest(result))
            if args.trades:
                for t in result.trades:
                    print(f"  {t['date']}  {t['side']:<4} {t['qty']:>5} @ {t['price']:.2f}  cost ${t['tx_cost']:.2f}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
