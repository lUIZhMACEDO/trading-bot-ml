"""Transaction cost model: a flat commission plus flat slippage, both in basis points of notional."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 5.0
    slippage_bps: float = 2.0

    @property
    def rate(self) -> float:
        """Total cost as a fraction of notional. The defaults give 7 bps = 0.0007."""
        return (self.commission_bps + self.slippage_bps) / 10_000

    def commission(self, notional: float) -> float:
        return notional * self.commission_bps / 10_000

    def slippage(self, notional: float) -> float:
        return notional * self.slippage_bps / 10_000

    def total(self, notional: float) -> float:
        return notional * self.rate

    def buy_cash_out(self, qty: float, price: float) -> float:
        """Cash spent to buy ``qty`` at ``price`` including costs."""
        return qty * price * (1 + self.rate)

    def sell_cash_in(self, qty: float, price: float) -> float:
        """Cash received for selling ``qty`` at ``price`` after costs."""
        return qty * price * (1 - self.rate)
