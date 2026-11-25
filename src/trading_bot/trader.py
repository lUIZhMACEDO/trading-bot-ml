"""Background auto-trader.

Every ``scan_interval`` seconds it closes positions that hit the stop-loss or take-profit, and if
there is room for another position it buys the single best-scoring symbol on the watchlist.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from trading_bot import engine
from trading_bot.config import PAPER_STARTING_EQUITY
from trading_bot.data.universe import WATCHLIST
from trading_bot.strategies import STRATEGIES

log = logging.getLogger(__name__)

COMPOSITE_MIN = 1.0  # composite needed to consider a symbol in composite mode (a "Buy" rating)
SINGLE_MIN = 0.3  # score needed to consider a symbol in single-strategy mode


class AutoTrader:
    def __init__(
        self,
        broker,
        db_path: Path | str,
        strategy: str = "composite",
        budget: float = 10_000,
        max_positions: int = 3,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 5.0,
        scan_interval: int = 60,
        watchlist: list[str] | None = None,
    ):
        if strategy not in ("composite", "ml") and strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.broker = broker
        self.db_path = db_path
        self.strategy = strategy
        self.budget = budget
        self.max_positions = max_positions
        self.stop_loss = stop_loss_pct / 100
        self.take_profit = take_profit_pct / 100
        self.scan_interval = scan_interval
        self.watchlist = watchlist or WATCHLIST
        self.ml_cache: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Auto-trader started: strategy=%s budget=$%s", self.strategy, f"{self.budget:,.0f}")

    def stop(self) -> None:
        self._stop.set()
        log.info("Auto-trader stopped")

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_and_trade()
                self.snapshot_portfolio()
            except Exception:
                log.exception("Auto-trade cycle failed")
            self._stop.wait(self.scan_interval)

    def apply_exits(self, positions: list[dict]) -> list[str]:
        """Close positions past the stop-loss or take-profit. Returns the closed symbols."""
        closed = []
        for pos in positions:
            pnl = pos["pnl_pct"] / 100
            if pnl <= -self.stop_loss:
                log.info("Stop loss on %s (%.1f%%)", pos["symbol"], pos["pnl_pct"])
            elif pnl >= self.take_profit:
                log.info("Take profit on %s (%.1f%%)", pos["symbol"], pos["pnl_pct"])
            else:
                continue
            self.broker.close_position(pos["symbol"])
            closed.append(pos["symbol"])
        return closed

    def candidates(self, held: set[str]) -> list[tuple[str, float, str]]:
        """``(symbol, score, reason)`` for every buyable symbol, best first."""
        scored = []
        for symbol in self.watchlist:
            if symbol in held:
                continue
            if self.strategy == "composite":
                result = engine.analyze_symbol(self.db_path, symbol)
                if result and result["composite"] >= COMPOSITE_MIN:
                    scored.append((symbol, result["composite"], result["rating"]))
            elif self.strategy == "ml":
                signal, confidence, reason = engine.predict_symbol(self.db_path, symbol, self.ml_cache)
                if signal == "BUY":
                    scored.append((symbol, confidence, reason))
            else:
                out = engine.strategy_signal(self.db_path, symbol, self.strategy)
                if out and out[0] == "BUY" and out[1] > SINGLE_MIN:
                    scored.append((symbol, out[1], out[2]))
        return sorted(scored, key=lambda s: s[1], reverse=True)

    def scan_and_trade(self) -> dict | None:
        """One cycle. Returns the order placed, if any."""
        positions = self.broker.get_positions()
        self.apply_exits(positions)
        # Count positions as they were at the start of the cycle, closes may still be pending.
        if len(positions) >= self.max_positions:
            return None

        regime = engine.detect_market_regime(self.db_path, self.watchlist)
        picks = self.candidates({p["symbol"] for p in positions})
        if not picks:
            return None

        symbol, score, reason = picks[0]
        try:
            equity = self.broker.get_account()["equity"]
        except Exception:
            equity = self.budget
        qty = engine.size_position(self.db_path, symbol, self.budget / self.max_positions, equity).qty
        if regime.get("regime") == "high_vol":
            qty = max(1, qty // 2)
            log.info("High-volatility regime, halving the position size")

        log.info("%s -> BUY %d %s (score %.2f): %s", self.strategy, qty, symbol, score, reason)
        return self.broker.buy(symbol, qty=qty, strategy=self.strategy)

    def snapshot_portfolio(self) -> None:
        try:
            engine.snapshot_portfolio(
                self.db_path, self.broker.get_account(), self.broker.get_positions(), PAPER_STARTING_EQUITY
            )
        except Exception:
            log.exception("Portfolio snapshot failed")
