# ⚛ Quantum Trader

**ML-powered paper trading bot with browser dashboard — built on Alpaca's API.**

Manual trades, auto-trading with 7 strategies, walk-forward ML, and a full web UI at `localhost:8000`.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Alpaca](https://img.shields.io/badge/Alpaca-Paper%20Trading-yellow?logo=alpaca&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/quantum-trader.git
cd quantum-trader
python setup_and_run.py
```

That's it. Installs dependencies, launches the server, opens your browser.  
First time: paste your [Alpaca paper trading](https://app.alpica.markets) API keys → click Connect.

---

## Features

### Manual Trading
Buy and sell any stock from the browser — market or limit orders, one click to close positions.

### Auto-Trading (7 Strategies)
Set a budget, pick a strategy, hit Start. The bot scans 30 stocks every 60 seconds and trades autonomously with stop-loss and take-profit.

| Strategy | Type | Win Rate | What It Does |
|----------|------|----------|-------------|
| **Composite** | Adaptive | 60-70% | Runs all 5 below → aggregates into a single score |
| **RSI + MACD** | Quick | 58-65% | Confluence of oversold RSI + MACD histogram crossover |
| **Bollinger Reversion** | Quick | 55-62% | Fades price extremes, skips when Hurst says trending |
| **Golden/Death Cross** | Medium | 45-55% | SMA 50/200 crossover confirmed by ADX |
| **Momentum Breakout** | Quick | 50-60% | Triple-timeframe alignment (5d/10d/21d) + volume |
| **Hurst Regime** | Medium | 55-65% | Auto-switches trend-follow vs mean-reversion |
| **ML Ensemble** | Medium | 60-70% | LightGBM + Random Forest + Ridge (walk-forward trained) |

### ML Engine
- Walk-forward expanding-window cross-validation (no lookahead bias)
- LightGBM + Random Forest + Ridge ensemble (40/35/25 weights)
- 15 features: RSI, MACD, Bollinger width, Hurst exponent, ADX, momentum, volume Z-score
- Predicts 21-day forward returns

### Data Pipeline
One command ingests everything:
- 2 years of OHLCV from Alpaca for 30 stocks
- Fundamentals from yfinance (P/E, FCF yield, ROE, analyst ratings)
- 10 FRED macro series (VIX, treasury yields, credit spreads, CPI)
- Vectorized technicals computed in a single pass (not row-by-row)
- Private company synthetic exposure (Anthropic, SpaceX, OpenAI, Stripe, xAI via public proxies)

### Risk Management
- Volatility-adjusted position sizing (targets 1% portfolio risk per position)
- Kelly criterion for optimal bet sizing
- Market regime detection (bull/bear/sideways/high-vol) using SPY + VIX + breadth
- Auto-halves position size in high-volatility regimes
- Transaction cost modeling (5bps + 2bps slippage) in backtests

### Database
13-table normalized SQLite with WAL mode:

```
ohlcv, technicals, fundamentals, macro, signals,
composite_scores, trades, ml_models, backtests,
private_exposure, position_sizing, market_regime, portfolio_snapshots
```

---

## Architecture

```
quantum-trader/
├── setup_and_run.py      # One-click installer + launcher
├── quantum_web.py        # FastAPI server + browser dashboard (28 API routes)
├── quantum_trader.py     # Trading engine (also works standalone as CLI)
├── requirements.txt
├── .env.example          # Config template
├── .gitignore
├── LICENSE
├── README.md
```

**How it works:**

```
Browser (localhost:8000)
    ↕ REST API (FastAPI)
quantum_web.py
    ↕ function calls
quantum_trader.py
    ↕ Alpaca SDK          ↕ yfinance/FRED
Alpaca Paper Trading    Market Data
    ↕
quantum_market.db (SQLite)
```

---

## Usage

### Browser Dashboard
```bash
python setup_and_run.py          # or: python quantum_web.py
```
Opens `http://localhost:8000` with tabs for Dashboard, Trade, Auto Trade, Analysis, and Data.

### Terminal CLI
```bash
python quantum_trader.py
```

Commands:
```
buy AAPL 10              # Market buy
buy AAPL 10 195.50       # Limit buy
sell AAPL 10             # Sell
close AAPL               # Close position
positions                # View holdings
pipeline                 # Run full data pipeline
scan                     # Score all 30 stocks
analyze AAPL             # 5-strategy deep dive
train AAPL               # Walk-forward ML training
predict AAPL             # ML prediction
backtest AAPL momentum   # Backtest with tx costs
auto start composite     # Start auto-trading
auto stop                # Stop auto-trading
regime                   # Detect market regime
private                  # Private company exposure
```

---

## Setup (Manual)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/quantum-trader.git
cd quantum-trader

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your Alpaca paper trading API keys

# 4. Run
python quantum_web.py    # Browser dashboard
# or
python quantum_trader.py  # Terminal CLI
```

### Get API Keys
1. Sign up at [app.alpaca.markets](https://app.alpica.markets)
2. Switch to **Paper Trading** (top-left dropdown)
3. Go to **API Keys** → **Generate**
4. Copy Key and Secret into `.env`

### Optional: FRED API Key
For macro data (VIX, yields, credit spreads):
1. Register at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Add key to `.env`

---

## Disclaimer

This software is for **paper trading and educational purposes only**. It does not constitute financial advice. All trading involves risk. Past performance and backtested results do not guarantee future returns. Use at your own risk.

---

## License

MIT — see [LICENSE](LICENSE)
