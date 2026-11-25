import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv, random_walk
from trading_bot.features.technicals import TECHNICAL_COLUMNS, add_technicals, compute_hurst, technical_rows


def test_short_frames_are_returned_unchanged():
    df = make_ohlcv(np.linspace(100, 110, 29))
    out = add_technicals(df)
    assert out is df
    assert "rsi" not in out.columns


def test_adds_every_column_without_touching_the_input():
    df = make_ohlcv(random_walk(250))
    out = add_technicals(df)
    assert set(TECHNICAL_COLUMNS) <= set(out.columns)
    assert "rsi" not in df.columns


def test_moving_averages_match_pandas():
    df = make_ohlcv(random_walk(250, seed=3))
    out = add_technicals(df)
    pd.testing.assert_series_equal(out["sma_20"], df["close"].rolling(20).mean(), check_names=False)
    pd.testing.assert_series_equal(out["sma_200"], df["close"].rolling(200).mean(), check_names=False)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    np.testing.assert_allclose(out["macd"], ema12 - ema26)
    np.testing.assert_allclose(out["macd_hist"], out["macd"] - out["macd_signal"])


def test_rsi_extremes():
    up = add_technicals(make_ohlcv(np.linspace(100, 200, 60)))
    down = add_technicals(make_ohlcv(np.linspace(200, 100, 60)))
    assert up["rsi"].iloc[-1] > 99
    assert down["rsi"].iloc[-1] < 1
    assert up["rsi"].iloc[:13].isna().all()  # Wilder smoothing needs 14 bars


def test_rsi_stays_in_range():
    rsi = add_technicals(make_ohlcv(random_walk(300, seed=5)))["rsi"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_bollinger_band_geometry():
    out = add_technicals(make_ohlcv(random_walk(120, seed=8))).dropna(subset=["bb_mid"])
    std = (out["bb_upper"] - out["bb_mid"]) / 2
    np.testing.assert_allclose(out["bb_mid"] - out["bb_lower"], 2 * std)
    # %B is 0 at the lower band and 1 at the upper band
    np.testing.assert_allclose(
        out["bb_pctb"], (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]), rtol=1e-9
    )


def test_atr_of_a_constant_range():
    # high - low is always 2 and there are no gaps, so the true range is 2 on every bar
    close = np.full(60, 100.0)
    df = make_ohlcv(close)
    df["high"], df["low"] = 101.0, 99.0
    out = add_technicals(df)
    assert out["atr"].iloc[-1] == pytest.approx(2.0)


def test_momentum_and_returns():
    close = np.linspace(100, 160, 61)
    out = add_technicals(make_ohlcv(close))
    assert out["momentum_21"].iloc[-1] == pytest.approx(close[-1] / close[-22] - 1)
    assert out["return_1d"].iloc[-1] == pytest.approx(close[-1] / close[-2] - 1)


def test_volume_zscore_flags_a_spike():
    volume = np.full(60, 1_000_000.0)
    volume[::2] = 1_100_000.0
    volume[-1] = 5_000_000.0
    out = add_technicals(make_ohlcv(random_walk(60, seed=1), volume=volume))
    assert out["vol_zscore"].iloc[-1] > 3
    assert out["vol_ratio"].iloc[-1] > 3.5


def _mean_hurst(make, n=20, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([compute_hurst(pd.Series(make(rng))) for _ in range(n)]))


def test_hurst_orders_mean_reverting_random_and_persistent_series():
    noise = _mean_hurst(lambda rng: 100 + rng.normal(0, 1, 300))
    walk = _mean_hurst(lambda rng: 100 + np.cumsum(rng.normal(0, 1, 300)))
    persistent = _mean_hurst(
        lambda rng: 100 + np.cumsum(pd.Series(rng.normal(0, 1, 300)).ewm(alpha=0.1).mean().to_numpy())
    )
    assert noise < 0.1
    assert 0.3 < walk < 0.6  # about 0.5 in theory, a bit lower on short samples
    assert persistent > walk + 0.1


def test_hurst_needs_enough_history():
    assert compute_hurst(pd.Series(np.arange(50.0))) == 0.5
    assert 0.0 <= compute_hurst(pd.Series(random_walk(200))) <= 1.0


def test_indicators_never_use_future_bars():
    """Appending bars must not change any value already computed. This is what keeps the
    backtester and the ML features free of look-ahead."""
    full = make_ohlcv(random_walk(320, seed=11))
    head = add_technicals(full.iloc[:250].copy())
    whole = add_technicals(full)
    pd.testing.assert_frame_equal(
        head[TECHNICAL_COLUMNS].reset_index(drop=True), whole[TECHNICAL_COLUMNS].iloc[:250].reset_index(drop=True)
    )


def test_technical_rows_skip_warmup_and_store_nan_as_none():
    out = add_technicals(make_ohlcv(random_walk(60)))
    rows = technical_rows("TEST", out)
    assert len(rows) == out["rsi"].notna().sum()
    first = rows[0]
    assert first[0] == "TEST"
    assert len(first) == 2 + len(TECHNICAL_COLUMNS)
    sma_200 = first[2 + TECHNICAL_COLUMNS.index("sma_200")]
    assert sma_200 is None
