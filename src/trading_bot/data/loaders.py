"""Read price history from SQLite or from a CSV file into a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.data.db import connect

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# Columns joined from the technicals table for ML training and prediction.
_ML_JOIN_SQL = """
    SELECT o.*, t.rsi, t.macd, t.macd_hist, t.bb_width, t.bb_pctb,
           t.atr, t.adx, t.hurst_exponent,
           t.momentum_5, t.momentum_10, t.momentum_21,
           t.vol_ratio, t.vol_zscore, t.return_1d, t.return_5d
    FROM ohlcv o
    LEFT JOIN technicals t ON o.symbol = t.symbol AND o.date = t.date
    WHERE o.symbol = ?
    ORDER BY o.date
"""


def load_ohlcv(db_path: Path | str, symbol: str, limit: int | None = None) -> pd.DataFrame:
    """Daily bars for one symbol, oldest first. ``limit`` keeps only the most recent rows."""
    with connect(db_path) as conn:
        if limit is None:
            return pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol=? ORDER BY date", conn, params=(symbol,))
        df = pd.read_sql_query(
            "SELECT * FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT ?", conn, params=(symbol, limit)
        )
    return df.iloc[::-1].reset_index(drop=True)


def load_ohlcv_with_technicals(db_path: Path | str, symbol: str) -> pd.DataFrame:
    """Bars joined with the stored indicator columns the ML model uses."""
    with connect(db_path) as conn:
        return pd.read_sql_query(_ML_JOIN_SQL, conn, params=(symbol,))


def load_price_csv(path: Path | str) -> pd.DataFrame:
    """Load a daily OHLCV CSV with columns date, open, high, low, close, volume.

    Column names are matched case-insensitively, so a yfinance-style export also works.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    df = df[PRICE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError(f"{path}: duplicate dates")
    return df
