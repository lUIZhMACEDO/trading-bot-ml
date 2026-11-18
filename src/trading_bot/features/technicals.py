"""Technical indicators computed over a whole price series at once.

Every column only uses data up to and including its own row (rolling windows, EWMs and
``pct_change`` all look backwards), so the backtester and the ML features never see the future.
``tests/test_technicals.py`` checks that by appending future bars and comparing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HURST_WINDOW = 100

TECHNICAL_COLUMNS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pctb",
    "ema_9", "ema_21", "sma_20", "sma_50", "sma_200",
    "atr", "adx", "hurst_exponent",
    "momentum_5", "momentum_10", "momentum_21",
    "vol_ratio", "vol_zscore",
    "return_1d", "return_5d", "return_21d",
]  # fmt: skip

_EPS = 1e-10


def compute_hurst(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst exponent: below 0.5 mean-reverting, about 0.5 random walk, above 0.5 trending.

    The standard deviation of lag-k differences grows like k**H, so H is twice the slope of
    log(sqrt(std)) against log(lag). The result is clipped to [0, 1], and 0.5 is returned when the
    series is shorter than ``max_lag``. On short windows the estimate is biased low: a 100-bar
    random walk averages around 0.4 rather than 0.5.
    """
    if len(series) < max_lag:
        return 0.5
    values = np.asarray(series, dtype=float)
    lags = list(range(2, min(max_lag, len(values) // 2)))
    tau = [np.sqrt(np.std(values[lag:] - values[:-lag])) for lag in lags]
    try:
        slope = 2.0 * np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except (ValueError, np.linalg.LinAlgError):
        return 0.5
    if not np.isfinite(slope):
        return 0.5
    return float(max(0.0, min(1.0, slope)))


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every column in ``TECHNICAL_COLUMNS`` added.

    ``df`` needs open/high/low/close/volume columns in date order. Frames shorter than 30 rows
    are returned unchanged, as there is not enough history for the indicators.
    """
    if len(df) < 30:
        return df
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # RSI with Wilder's smoothing
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, _EPS)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ema_9"] = close.ewm(span=9).mean()
    df["ema_21"] = close.ewm(span=21).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger bands, 20 days, 2 standard deviations
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = bb_range / df["bb_mid"]
    df["bb_pctb"] = (close - df["bb_lower"]) / bb_range.replace(0, _EPS)

    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # ADX, simplified: directional movement over the same 14-day ATR
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = df["atr"].replace(0, _EPS)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, _EPS) * 100
    df["adx"] = dx.rolling(14).mean()

    # Rolling Hurst over the previous 100 closes (the current bar is not included).
    df["hurst_exponent"] = [
        0.5 if i < HURST_WINDOW else compute_hurst(close.iloc[i - HURST_WINDOW : i]) for i in range(len(close))
    ]

    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)
    df["momentum_21"] = close.pct_change(21)

    vol_sma = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_sma.replace(0, _EPS)
    vol_std = volume.rolling(20).std()
    df["vol_zscore"] = (volume - vol_sma) / vol_std.replace(0, _EPS)

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_21d"] = close.pct_change(21)
    return df


def technical_rows(symbol: str, df: pd.DataFrame) -> list[tuple]:
    """Rows for the technicals table, skipping the warm-up bars where RSI is still undefined."""
    rows = []
    for _, row in df.dropna(subset=["rsi"]).iterrows():
        values = [row.get(c) for c in TECHNICAL_COLUMNS]
        rows.append((symbol, row["date"], *[float(v) if pd.notna(v) else None for v in values]))
    return rows
