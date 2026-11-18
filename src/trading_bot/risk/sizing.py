"""Position sizing: half-Kelly and volatility targeting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_PER_POSITION = 0.01  # target loss of 1% of equity on a typical move
HOLDING_DAYS = 5


def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float, max_pct: float = 0.10) -> float:
    """Half-Kelly fraction of the portfolio, floored at 0 and capped at ``max_pct``."""
    if avg_loss == 0:
        return max_pct
    b = avg_win / abs(avg_loss)
    kelly = (b * win_rate - (1 - win_rate)) / b
    return min(max(0.0, kelly / 2), max_pct)


@dataclass(frozen=True)
class SizeDecision:
    qty: int
    vol_qty: int | None
    price: float
    annual_vol: float | None
    risk_per_share: float | None

    @property
    def rationale(self) -> str:
        if self.annual_vol is None:
            return f"Not enough history, budget-based size at ${self.price:.2f}"
        return f"Vol={self.annual_vol:.1%}, Price=${self.price:.2f}, Risk/share=${self.risk_per_share:.2f}"


def volatility_adjusted_qty(closes: Sequence[float], budget: float, equity: float) -> SizeDecision:
    """Shares to buy so that a typical ``HOLDING_DAYS`` move costs about 1% of ``equity``.

    ``closes`` are recent closes in date order; the last one is the entry price. The result is
    also capped by ``budget`` and is always at least one share. With fewer than 10 closes the size
    is just ``budget / price`` (or a $100 price if there are no closes at all).
    """
    closes = [float(c) for c in closes]
    price = closes[-1] if closes else 100.0
    if len(closes) < 10:
        return SizeDecision(qty=max(1, int(budget / price)), vol_qty=None, price=price, annual_vol=None,
                            risk_per_share=None)  # fmt: skip

    returns = pd.Series(closes).pct_change().dropna()
    annual_vol = float(returns.std() * np.sqrt(252))
    risk_per_share = max(price * annual_vol * np.sqrt(HOLDING_DAYS) / np.sqrt(252), 0.01)
    vol_qty = int(RISK_PER_POSITION * equity / risk_per_share)
    budget_qty = int(budget / price)
    return SizeDecision(
        qty=max(1, min(vol_qty, budget_qty)),
        vol_qty=vol_qty,
        price=price,
        annual_vol=annual_vol,
        risk_per_share=float(risk_per_share),
    )
