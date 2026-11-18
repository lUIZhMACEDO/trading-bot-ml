"""Turning provider responses into database rows, with no network calls."""

import pandas as pd
import pytest

from trading_bot.data.ingest import bars_to_rows, fcf_yield, fundamentals_row, macro_rows
from trading_bot.data.loaders import load_price_csv


def alpaca_bars():
    idx = pd.MultiIndex.from_tuples(
        [("AAPL", pd.Timestamp("2026-01-02 05:00", tz="UTC")), ("AAPL", pd.Timestamp("2026-01-05 05:00", tz="UTC"))],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5], "close": [1.2, 2.2],
         "volume": [100.0, 200.0], "trade_count": [5, 6], "vwap": [1.1, None]},
        index=idx,
    )  # fmt: skip


def test_bars_to_rows_from_alpaca_multiindex():
    rows = bars_to_rows("AAPL", alpaca_bars())
    assert rows == [
        ("AAPL", "2026-01-02", 1.0, 1.5, 0.5, 1.2, 100, 1.1),
        ("AAPL", "2026-01-05", 2.0, 2.5, 1.5, 2.2, 200, 0.0),
    ]


def test_bars_to_rows_empty():
    assert bars_to_rows("AAPL", pd.DataFrame()) == []


def test_fundamentals_row_has_one_value_per_column():
    row = fundamentals_row("AAPL", {"trailingPE": 30.0, "freeCashflow": 1e9, "marketCap": 1e11}, "2026-11-17")
    assert len(row) == 18  # the fundamentals table has 18 columns
    assert row[:3] == ("AAPL", "2026-11-17", 30.0)
    assert row[7] == pytest.approx(1.0)  # FCF yield in percent


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"freeCashflow": 5e8, "marketCap": 1e10}, 5.0),
        ({"marketCap": 1e10}, 0.0),
        ({"freeCashflow": None, "marketCap": None}, 0.0),
    ],
)
def test_fcf_yield_handles_missing_values(info, expected):
    assert fcf_yield(info) == pytest.approx(expected)


def test_macro_rows_skip_missing_values():
    series = pd.Series([4.1, float("nan"), 4.3], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))
    assert macro_rows("DGS10", "treasury_10y", series) == [
        ("DGS10", "2026-01-02", 4.1, "treasury_10y"),
        ("DGS10", "2026-01-06", 4.3, "treasury_10y"),
    ]


def test_load_price_csv_normalizes_columns(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n2026-01-05,2,3,1,2.5,10\n2026-01-02,1,2,0.5,1.5,10\n")
    df = load_price_csv(path)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].tolist() == ["2026-01-02", "2026-01-05"]


def test_load_price_csv_rejects_missing_columns_and_duplicates(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,close\n2026-01-02,100\n")
    with pytest.raises(ValueError, match="missing columns open, high, low, volume"):
        load_price_csv(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text("date,open,high,low,close,volume\n2026-01-02,1,1,1,1,1\n2026-01-02,1,1,1,1,1\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_price_csv(dup)
