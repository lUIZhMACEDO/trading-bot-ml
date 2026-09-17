# Quantum Trader

A paper-trading engine on Alpaca's API, with a browser dashboard and a terminal CLI. It scores a
watchlist with five rule-based strategies plus an ML ensemble, sizes positions by volatility, and
places paper orders. Everything runs locally against SQLite.

Paper trading only. No real money, and none of this is investment advice.

## How it works

**Machine learning** (`quantum_trader.py`)

The target is the direction of the 21-day forward return (up or down), not the return itself.
Training uses 5-fold `TimeSeriesSplit`, so each fold trains only on data before its test window and
the model never sees the future. Three models vote: LightGBM at 40%, random forest at 35%, Ridge at
25%. Features are the indicators below — RSI, MACD, Bollinger width, the Hurst exponent, ADX,
momentum, and a volume z-score.

The honest caveat: the headline metric is fold accuracy, and accuracy is a weak measure for 21-day
direction, because the base rate is already better than a coin flip. Treat the ensemble as a filter
on the rule-based signals rather than as a forecast.

**Strategies**

Five screens, each returning a signal per ticker, plus a composite that aggregates them:

| Strategy | What it looks for |
|---|---|
| RSI + MACD | Oversold RSI with a MACD histogram crossover |
| Bollinger reversion | Price at a band extreme, skipped when the Hurst exponent says trending |
| Golden/death cross | SMA 50/200 crossover, confirmed by ADX |
| Momentum breakout | 5/10/21-day alignment plus a volume check |
| Hurst regime | Switches between trend-following and mean-reversion by regime |

**Backtesting**

Trades fill at the next day's open, not the signal close. Costs are modeled at 5 bps per trade plus
2 bps of slippage. Results are stored per strategy so you can compare them against buy-and-hold.
I've deliberately left performance numbers out of this README — run it on your own watchlist and
dates and you'll get your own, which is the only version worth quoting.

**Risk and sizing**

Volatility-adjusted sizing targeting roughly 1% portfolio risk per position, Kelly-criterion bet
sizing, and a market-regime read (bull, bear, sideways, high-vol) from SPY, VIX, and breadth that
halves size when volatility spikes.

**Data**

2 years of OHLCV from Alpaca, fundamentals from yfinance, 10 FRED macro series, and indicators
computed in a single vectorized pass. Stored in a 13-table SQLite database with WAL mode.

## Running it

```bash
git clone https://github.com/lUIZhMACEDO/trading-bot-ml.git
cd trading-bot-ml
python setup_and_run.py     # installs dependencies, starts the server, opens the dashboard
```

On first run it asks for your [Alpaca paper trading](https://app.alpaca.markets) keys and writes
them to `.env` (see `.env.example`). Nothing is committed.

Dashboard only: `python quantum_web.py` — FastAPI on `localhost:8000`, 24 routes, with tabs for the
portfolio, manual trades, auto-trading, analysis, and the data pipeline.

Terminal: `python quantum_trader.py`

```
pipeline              run the full data ingest
scan                  score the watchlist
analyze AAPL          five-strategy breakdown for one ticker
train AAPL            walk-forward training
backtest AAPL momentum
buy AAPL 10           paper market order (add a price for a limit)
positions             current holdings
auto start composite  start the auto-trader
regime                current market regime
```

## Layout

```
quantum_trader.py   engine: data, indicators, strategies, ML, backtester, CLI
quantum_web.py      FastAPI server and the dashboard it serves
setup_and_run.py    one-command installer and launcher
index.html          static demo page
schema: created on first run in quantum_market.db
```

## Limitations

- Paper trading only, and the auto-trader has no kill switch beyond stopping the process
- Slippage is a flat constant, not a function of size or spread
- Hyperparameters are fixed, not searched inside each fold
- The watchlist is 30 large-cap US names; nothing here is tested on small caps or other asset classes

MIT licensed.
