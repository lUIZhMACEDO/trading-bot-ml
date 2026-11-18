"""Market regime from SPY's trend, VIX and breadth across the watchlist."""

from __future__ import annotations


def classify_regime(spy_close: float, sma50: float, sma200: float, vix: float, breadth_pct: float) -> tuple[str, float]:
    """Return ``(regime, confidence)``.

    VIX above 30 is ``high_vol`` regardless of trend. Otherwise price above SMA50 above SMA200
    with more than 60% of the watchlist above its 20-day average is ``bull``, the mirror image
    under 40% is ``bear``, and anything else is ``sideways``.
    """
    if vix > 30:
        return "high_vol", 0.8
    if spy_close > sma50 > sma200 and breadth_pct > 60:
        return "bull", 0.9 if breadth_pct > 75 else 0.5
    if spy_close < sma50 < sma200 and breadth_pct < 40:
        return "bear", 0.9 if breadth_pct < 25 else 0.5
    return "sideways", 0.5
