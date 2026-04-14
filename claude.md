# Quantum Trader — Project Goals & Status

## What This Project Is
An ML-powered paper trading bot built on Alpaca's API with a browser dashboard and terminal CLI.
Built for Al King III / AK3 Holdings.

## Current Status: ✅ Core Complete

### What's Done
- [x] **Trading Engine** (`quantum_trader.py`) — 1,893 lines
  - 13-table normalized SQLite with WAL mode
  - Alpaca paper trading integration (buy/sell/close/positions/orders)
  - API key onboarding (prompts on first run, saves to .env)
  - 2-year OHLCV data ingest from Alpaca
  - Fundamentals from yfinance (P/E, FCF yield, ROE, analyst ratings)
  - FRED macro data (VIX, yields, credit spreads, CPI — 10 series)
  - Vectorized technicals in single pass (RSI, MACD, Bollinger, Hurst, ADX, momentum, vol Z-score)
  - 5 concurrent strategies per ticker → composite score → Strong Buy/Sell rating
  - Walk-forward ML with LightGBM + Random Forest + Ridge ensemble
  - Backtester with transaction cost modeling (5bps + 2bps slippage)
  - Volatility-adjusted position sizing (Kelly criterion)
  - Market regime detection (bull/bear/sideways/high-vol)
  - Private company synthetic exposure (Anthropic, SpaceX, OpenAI, Stripe, xAI)
  - Auto-trader with start/stop, budget, stop-loss, take-profit
  - Full interactive CLI with 25+ commands

- [x] **Web Dashboard** (`quantum_web.py`) — 1,153 lines
  - FastAPI server with 28 API endpoints
  - Full HTML/CSS/JS dashboard served at localhost:8000
  - Tabs: Dashboard, Trade, Auto Trade, Analysis, Data & ML
  - Manual buy/sell with market + limit orders
  - Auto-trade controls with strategy selection
  - Portfolio tracking, positions, trade log
  - Auto-opens browser on launch

- [x] **Setup** (`setup_and_run.py`) — 114 lines
  - One-click installer: checks Python, installs all packages, launches web dashboard

- [x] **GitHub Pages Demo** (`index.html`)
  - Interactive demo with simulated data at luizhmacedo.github.io/trading-bot-ml
  - All 5 tabs working with fake live-updating prices

- [x] **GitHub Repo** — https://github.com/lUIZhMACEDO/trading-bot-ml

### 7 Trading Strategies
1. **Composite** — Runs all 5 below, aggregates into -5 to +5 score
2. **RSI + MACD Confluence** — Quick trades, 58-65% win rate
3. **Bollinger Mean Reversion** — Uses Hurst to skip trending regimes
4. **Golden/Death Cross** — SMA 50/200 with ADX confirmation
5. **Momentum Breakout** — Triple-timeframe (5d/10d/21d) + volume
6. **Hurst Regime Adaptive** — Auto-switches trend-follow vs mean-reversion
7. **ML Ensemble** — LightGBM 40% + Random Forest 35% + Ridge 25%

### Tech Stack
- Python 3.9+
- alpaca-py (trading + market data)
- pandas, numpy, scikit-learn, lightgbm
- yfinance (fundamentals), fredapi (macro)
- FastAPI + uvicorn (web server)
- SQLite with WAL mode (database)
- Pure HTML/CSS/JS (dashboard — no React build step)

## What's NOT Done Yet (Future Goals)
- [ ] Streamlit or richer charting (candlestick charts, equity curves)
- [ ] Email/SMS alerts when auto-trader executes
- [ ] WebSocket streaming for real-time price updates (currently polls)
- [ ] More ML models (XGBoost, LSTM for price prediction)
- [ ] Sentiment analysis (news/social media via FinBERT or Claude API)
- [ ] Options trading support
- [ ] Multi-account support
- [ ] Mobile-responsive dashboard improvements
- [ ] Deployment to cloud (AWS/GCP) for 24/7 auto-trading
- [ ] Integration with Quant Research System daily pipeline (4:05 PM ET cron)

## Architecture
```
Browser (localhost:8000)
    ↕ REST API (FastAPI, 28 routes)
quantum_web.py
    ↕ function calls
quantum_trader.py (trading engine)
    ↕ alpaca-py SDK         ↕ yfinance/FRED
Alpaca Paper Trading     Market Data Sources
    ↕
quantum_market.db (SQLite, 13 tables, WAL)
```

## Files
```
quantum-trader/
├── quantum_trader.py   # Trading engine + CLI (1,893 lines)
├── quantum_web.py      # FastAPI web dashboard (1,153 lines)
├── setup_and_run.py    # One-click launcher (114 lines)
├── index.html          # GitHub Pages demo
├── requirements.txt    # Dependencies
├── .env.example        # Config template
├── .gitignore          # Blocks .env, .db, __pycache__
├── LICENSE             # MIT
├── README.md           # Full docs
└── claude.md           # This file — project tracker
```

## Key Decisions
- **No LSTM** — We use Random Forest + LightGBM + Ridge ensemble. LSTM was NOT part of this project (Cowork incorrectly added it).
- **No sentiment analysis** — Not built yet. Listed as future goal.
- **SQLite over Postgres** — Simpler, no server needed, WAL mode handles concurrency.
- **FastAPI over Streamlit** — More control over the UI, single HTML file, no extra dependencies.
- **Walk-forward ML** — Prevents lookahead bias (expanding window TimeSeriesSplit).
- **Hurst exponent** — Key differentiator. Detects mean-reverting vs trending regimes so strategies adapt.
- **Transaction cost modeling** — Backtests include 5bps trading cost + 2bps slippage for realism.

## How To Run
```bash
cd quantum-trader
python setup_and_run.py    # Installs everything, opens browser dashboard
# OR
python quantum_trader.py   # Terminal CLI mode
```
