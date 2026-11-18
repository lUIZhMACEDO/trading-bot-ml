"""Performance metrics for an equity curve and a list of fills."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough drop as a fraction (0.25 means -25%)."""
    worst, peak = 0.0, None
    for value in equity:
        peak = value if peak is None else max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def sharpe_ratio(equity: Sequence[float], periods_per_year: int = 252, min_points: int = 30) -> float:
    """Annualized Sharpe of daily equity returns (risk-free rate 0). 0 for short or flat curves."""
    if len(equity) <= min_points:
        return 0.0
    returns = pd.Series(equity, dtype=float).pct_change().dropna()
    std = returns.std()
    if not std > 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def round_trip_returns(trades: Sequence[dict]) -> list[float]:
    """Return of each buy -> sell pair, from fill prices. An open position at the end is skipped."""
    returns = []
    for i in range(0, len(trades) - 1, 2):
        returns.append(trades[i + 1]["price"] / trades[i]["price"] - 1)
    return returns


def win_rate(round_trips: Sequence[float]) -> float:
    return sum(r > 0 for r in round_trips) / max(len(round_trips), 1)
