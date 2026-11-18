"""The five rule-based strategies and the composite score that adds them up.

Each strategy reads the last row (or last two rows) of a frame that already has the columns from
``features.technicals.add_technicals`` and returns ``(signal, score, reason)``: signal is BUY,
SELL or HOLD, and score runs from -1.0 (strong sell) to +1.0 (strong buy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

Signal = tuple[str, float, str]
StrategyFn = Callable[[pd.DataFrame], Signal]


def rsi_macd(df: pd.DataFrame) -> Signal:
    """Oversold or overbought RSI confirmed by the MACD histogram."""
    if len(df) < 30:
        return "HOLD", 0, "Insufficient data"
    rsi = df["rsi"].iloc[-1]
    macd_h = df["macd_hist"].iloc[-1]
    vol_r = df["vol_ratio"].iloc[-1]

    if rsi < 30 and macd_h > 0:
        score = min(1.0, (30 - rsi) / 20 + 0.3)
        return "BUY", score, f"RSI={rsi:.0f} oversold + MACD bullish"
    if rsi > 70 and macd_h < 0:
        score = min(1.0, (rsi - 70) / 20 + 0.3)
        return "SELL", -score, f"RSI={rsi:.0f} overbought + MACD bearish"
    if rsi < 40 and macd_h > 0 and vol_r > 1.3:
        return "BUY", 0.4, f"RSI={rsi:.0f} + MACD bullish + high volume"
    return "HOLD", 0, f"RSI={rsi:.0f}, MACD_hist={macd_h:.3f}"


def bollinger_reversion(df: pd.DataFrame) -> Signal:
    """Fade the Bollinger band extremes, but only when Hurst says the series is not trending."""
    if len(df) < 30:
        return "HOLD", 0, "Insufficient data"
    bb_pctb = df["bb_pctb"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    hurst = df["hurst_exponent"].iloc[-1]

    if hurst >= 0.55:
        return "HOLD", 0, f"Hurst={hurst:.2f}, trending regime, skip reversion"
    if bb_pctb < 0.05 and rsi < 35:
        return "BUY", 0.8, f"Below lower BB (pctB={bb_pctb:.2f}), RSI={rsi:.0f}, Hurst={hurst:.2f}"
    if bb_pctb > 0.95 and rsi > 65:
        return "SELL", -0.8, f"Above upper BB (pctB={bb_pctb:.2f}), RSI={rsi:.0f}"
    return "HOLD", 0, f"BB%B={bb_pctb:.2f}, Hurst={hurst:.2f}"


def golden_cross(df: pd.DataFrame) -> Signal:
    """SMA 50/200 crossovers, plus a weaker signal while the trend holds."""
    if len(df) < 210:
        return "HOLD", 0, "Need 200+ days of data"
    sma50, sma200 = df["sma_50"].iloc[-1], df["sma_200"].iloc[-1]
    prev_sma50, prev_sma200 = df["sma_50"].iloc[-2], df["sma_200"].iloc[-2]
    close = df["close"].iloc[-1]
    adx = df["adx"].iloc[-1]

    if prev_sma50 <= prev_sma200 and sma50 > sma200:
        return "BUY", 0.9, f"Golden cross: SMA50({sma50:.0f}) > SMA200({sma200:.0f}), ADX={adx:.0f}"
    if prev_sma50 >= prev_sma200 and sma50 < sma200:
        return "SELL", -0.9, f"Death cross: SMA50({sma50:.0f}) < SMA200({sma200:.0f})"
    if sma50 > sma200 and close > sma50:
        return "BUY", 0.3, "Uptrend: price > SMA50 > SMA200"
    if sma50 < sma200 and close < sma50:
        return "SELL", -0.3, "Downtrend: price < SMA50 < SMA200"
    return "HOLD", 0, f"SMA50={sma50:.0f}, SMA200={sma200:.0f}"


def momentum_breakout(df: pd.DataFrame) -> Signal:
    """5/10/21-day momentum all pointing the same way with ADX above 25, or a volume spike."""
    if len(df) < 30:
        return "HOLD", 0, "Insufficient data"
    m5 = df["momentum_5"].iloc[-1]
    m10 = df["momentum_10"].iloc[-1]
    m21 = df["momentum_21"].iloc[-1]
    vol_z = df["vol_zscore"].iloc[-1]
    adx = df["adx"].iloc[-1]

    if m5 > 0.02 and m10 > 0.03 and m21 > 0.05 and adx > 25:
        score = min(1.0, m21 * 10 + 0.3)
        return "BUY", score, f"Momentum aligned: 5d={m5:.1%}, 10d={m10:.1%}, 21d={m21:.1%}, ADX={adx:.0f}"
    if m5 < -0.02 and m10 < -0.03 and m21 < -0.05 and adx > 25:
        score = min(1.0, abs(m21 * 10) + 0.3)
        return "SELL", -score, f"Momentum breakdown: 5d={m5:.1%}, 10d={m10:.1%}, 21d={m21:.1%}"
    if vol_z > 2.0 and m5 > 0.03:
        return "BUY", 0.5, f"Volume breakout: VolZ={vol_z:.1f}, Mom5={m5:.1%}"
    return "HOLD", 0, f"Mom5={m5:.1%}, Mom21={m21:.1%}, ADX={adx:.0f}"


def hurst_regime(df: pd.DataFrame) -> Signal:
    """Trend-follow when Hurst is above 0.6, fade RSI extremes when it is below 0.4."""
    if len(df) < 110:
        return "HOLD", 0, "Need 100+ days for Hurst"
    hurst = df["hurst_exponent"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    ema9, ema21 = df["ema_9"].iloc[-1], df["ema_21"].iloc[-1]
    close = df["close"].iloc[-1]

    if hurst > 0.6:
        if ema9 > ema21 and close > ema9:
            return "BUY", 0.6, f"Trending regime (H={hurst:.2f}): EMA9 > EMA21, ride trend"
        if ema9 < ema21 and close < ema9:
            return "SELL", -0.6, f"Trending regime (H={hurst:.2f}): downtrend"
    elif hurst < 0.4:
        if rsi < 30:
            return "BUY", 0.7, f"Mean-reverting regime (H={hurst:.2f}): RSI={rsi:.0f} oversold"
        if rsi > 70:
            return "SELL", -0.7, f"Mean-reverting regime (H={hurst:.2f}): RSI={rsi:.0f} overbought"
    regime = "trending" if hurst > 0.5 else "mean-reverting"
    return "HOLD", 0, f"Hurst={hurst:.2f} ({regime}), RSI={rsi:.0f}"


# key -> (display name, function). The keys are what the CLI, dashboard and database use.
STRATEGIES: dict[str, tuple[str, StrategyFn]] = {
    "rsi_macd": ("RSI+MACD Confluence", rsi_macd),
    "bollinger": ("Bollinger Mean Reversion", bollinger_reversion),
    "golden_cross": ("Golden/Death Cross", golden_cross),
    "momentum": ("Momentum Breakout", momentum_breakout),
    "hurst": ("Hurst Regime Adaptive", hurst_regime),
}

# Short names accepted by "auto start <name>" on top of the keys above.
ALIASES = {"trend": "golden_cross"}


def resolve_strategy(name: str) -> str:
    """Map a user-typed strategy name to its key. Unknown names are returned lowercased."""
    name = name.lower()
    return ALIASES.get(name, name)


def rating_for(composite: float) -> str:
    if composite >= 2.5:
        return "Strong Buy"
    if composite >= 1.0:
        return "Buy"
    if composite <= -2.5:
        return "Strong Sell"
    if composite <= -1.0:
        return "Sell"
    return "Neutral"


@dataclass(frozen=True)
class CompositeResult:
    composite: float
    rating: str
    buy: int
    sell: int
    hold: int
    strategies: list[dict]


def score_all(df: pd.DataFrame) -> CompositeResult:
    """Run every strategy on ``df`` and sum the scores into a composite from -5 to +5."""
    results = []
    for key, (name, fn) in STRATEGIES.items():
        signal, score, reason = fn(df)
        results.append({"strategy": key, "name": name, "signal": signal, "score": float(score), "reason": reason})
    composite = float(sum(r["score"] for r in results))
    return CompositeResult(
        composite=composite,
        rating=rating_for(composite),
        buy=sum(r["signal"] == "BUY" for r in results),
        sell=sum(r["signal"] == "SELL" for r in results),
        hold=sum(r["signal"] == "HOLD" for r in results),
        strategies=results,
    )
