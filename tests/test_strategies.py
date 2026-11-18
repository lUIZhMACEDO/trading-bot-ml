"""Each strategy branch on hand-built indicator values."""

import pandas as pd
import pytest

from trading_bot import strategies as st

NEUTRAL = {
    "close": 100.0, "rsi": 50.0, "macd_hist": 0.0, "vol_ratio": 1.0, "bb_pctb": 0.5,
    "hurst_exponent": 0.5, "sma_50": 100.0, "sma_200": 100.0, "adx": 20.0,
    "momentum_5": 0.0, "momentum_10": 0.0, "momentum_21": 0.0, "vol_zscore": 0.0,
    "ema_9": 100.0, "ema_21": 100.0,
}  # fmt: skip


def frame(n: int = 220, prev: dict | None = None, **last) -> pd.DataFrame:
    """``n`` neutral rows; the last row gets ``last`` and the one before it gets ``prev``."""
    df = pd.DataFrame([NEUTRAL] * n)
    for key, value in (prev or {}).items():
        df.loc[n - 2, key] = value
    for key, value in last.items():
        df.loc[n - 1, key] = value
    return df


@pytest.mark.parametrize("fn", [st.rsi_macd, st.bollinger_reversion, st.momentum_breakout])
def test_short_history_holds(fn):
    assert fn(frame(29))[:2] == ("HOLD", 0)


def test_rsi_macd():
    signal, score, _ = st.rsi_macd(frame(rsi=20, macd_hist=0.5))
    assert signal == "BUY" and score == pytest.approx(0.8)
    signal, score, _ = st.rsi_macd(frame(rsi=85, macd_hist=-0.5))
    assert signal == "SELL" and score == pytest.approx(-1.0)  # capped at 1
    assert st.rsi_macd(frame(rsi=38, macd_hist=0.1, vol_ratio=1.5))[:2] == ("BUY", 0.4)
    assert st.rsi_macd(frame(rsi=38, macd_hist=0.1, vol_ratio=1.0))[0] == "HOLD"


def test_bollinger_skips_trending_series():
    assert st.bollinger_reversion(frame(bb_pctb=0.01, rsi=20, hurst_exponent=0.6))[0] == "HOLD"
    assert st.bollinger_reversion(frame(bb_pctb=0.01, rsi=20, hurst_exponent=0.4))[:2] == ("BUY", 0.8)
    assert st.bollinger_reversion(frame(bb_pctb=0.99, rsi=70, hurst_exponent=0.4))[:2] == ("SELL", -0.8)


def test_golden_cross():
    assert st.golden_cross(frame(200))[2] == "Need 200+ days of data"
    cross_up = frame(prev={"sma_50": 99, "sma_200": 100}, sma_50=101, sma_200=100)
    assert st.golden_cross(cross_up)[:2] == ("BUY", 0.9)
    cross_down = frame(prev={"sma_50": 101, "sma_200": 100}, sma_50=99, sma_200=100)
    assert st.golden_cross(cross_down)[:2] == ("SELL", -0.9)
    uptrend = frame(prev={"sma_50": 105, "sma_200": 100}, sma_50=105, sma_200=100, close=110)
    assert st.golden_cross(uptrend)[:2] == ("BUY", 0.3)


def test_momentum_breakout():
    aligned = frame(momentum_5=0.03, momentum_10=0.04, momentum_21=0.06, adx=30)
    signal, score, _ = st.momentum_breakout(aligned)
    assert signal == "BUY" and score == pytest.approx(0.9)
    weak_trend = frame(momentum_5=0.03, momentum_10=0.04, momentum_21=0.06, adx=20)
    assert st.momentum_breakout(weak_trend)[0] == "HOLD"
    breakdown = frame(momentum_5=-0.03, momentum_10=-0.04, momentum_21=-0.1, adx=30)
    assert st.momentum_breakout(breakdown)[:2] == ("SELL", -1.0)
    assert st.momentum_breakout(frame(vol_zscore=2.5, momentum_5=0.04))[:2] == ("BUY", 0.5)


def test_hurst_regime():
    assert st.hurst_regime(frame(100))[0] == "HOLD"
    trend = frame(hurst_exponent=0.7, ema_9=105, ema_21=100, close=106)
    assert st.hurst_regime(trend)[:2] == ("BUY", 0.6)
    reverting = frame(hurst_exponent=0.3, rsi=75)
    assert st.hurst_regime(reverting)[:2] == ("SELL", -0.7)


@pytest.mark.parametrize(
    ("composite", "rating"),
    [(3.0, "Strong Buy"), (2.5, "Strong Buy"), (1.0, "Buy"), (0.99, "Neutral"), (-1.0, "Sell"), (-2.5, "Strong Sell")],
)
def test_rating_thresholds(composite, rating):
    assert st.rating_for(composite) == rating


def test_score_all_adds_up_the_strategies():
    df = frame(rsi=20, macd_hist=0.5, bb_pctb=0.01, hurst_exponent=0.3)
    result = st.score_all(df)
    assert [s["strategy"] for s in result.strategies] == list(st.STRATEGIES)
    assert result.composite == pytest.approx(sum(s["score"] for s in result.strategies))
    assert result.buy + result.sell + result.hold == 5
    assert result.rating == st.rating_for(result.composite)


def test_strategy_aliases():
    assert st.resolve_strategy("Momentum") == "momentum"
    assert st.resolve_strategy("trend") == "golden_cross"
    assert st.resolve_strategy("composite") == "composite"
