#!/usr/bin/env python3
"""
QUANTUM TRADER v2.0 — Web Dashboard
=====================================
Run:  python quantum_web.py
Open: http://localhost:8000

This wraps quantum_trader_v2.py with a FastAPI web server
and serves a full trading dashboard in your browser.
"""

import os
import sys
import json
import threading
import webbrowser
import time
import logging
from pathlib import Path
from datetime import datetime

# ── Check and install FastAPI/Uvicorn if needed ──
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("  Installing web server dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "fastapi", "uvicorn[standard]", "-q", "--disable-pip-version-check"])
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

from dotenv import load_dotenv

# ── Import the trading engine ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import quantum_trader as qt
except ImportError:
    try:
        import quantum_trader_v2 as qt
    except ImportError:
        print("  ❌ Can't find quantum_trader.py — put it in the same folder as this file.")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("QuantumWeb")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="Quantum Trader v2.0")
client = None          # AlpacaClient instance
auto_trader = None     # AutoTrader instance
ENV_PATH = Path(".env")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SETUP / AUTH ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/status")
def api_status():
    """Check if API keys are configured and connection is live."""
    global client
    load_dotenv(override=True)
    api_key = os.getenv("ALPACA_API_KEY", "")
    has_keys = api_key and api_key != "your_new_key_here" and len(api_key) > 10
    connected = client is not None
    return {"has_keys": has_keys, "connected": connected}


@app.post("/api/connect")
async def api_connect(request: Request):
    """Save API keys to .env and connect."""
    global client
    body = await request.json()
    api_key = body.get("api_key", "").strip()
    secret_key = body.get("secret_key", "").strip()
    fred_key = body.get("fred_key", "").strip()
    budget = body.get("budget", "10000")
    max_pos = body.get("max_positions", "3")
    stop_loss = body.get("stop_loss", "2.0")
    take_profit = body.get("take_profit", "5.0")

    if not api_key or not secret_key:
        return JSONResponse({"error": "API key and secret are required"}, 400)

    # Write .env
    env_content = f"""ALPACA_API_KEY={api_key}
ALPACA_SECRET_KEY={secret_key}
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FRED_API_KEY={fred_key or 'your_fred_key_here'}
MAX_BUDGET={budget}
MAX_POSITIONS={max_pos}
STOP_LOSS_PCT={stop_loss}
TAKE_PROFIT_PCT={take_profit}
"""
    ENV_PATH.write_text(env_content)
    load_dotenv(override=True)

    # Try connecting
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        tc = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        dc = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        acct = tc.get_account()

        # Build a lightweight client object
        client = type('Client', (), {})()
        client.api_key = api_key
        client.secret_key = secret_key
        client.trading = tc
        client.data = dc

        # Attach all AlpacaClient methods
        ac = qt.AlpacaClient.__new__(qt.AlpacaClient)
        ac.api_key = api_key
        ac.secret_key = secret_key
        ac.trading = tc
        ac.data = dc
        client = ac

        qt.init_db()
        log.info(f"✅ Web connected: ${float(acct.equity):,.2f}")
        return {"status": "connected", "equity": float(acct.equity), "cash": float(acct.cash)}
    except Exception as e:
        if ENV_PATH.exists():
            ENV_PATH.unlink()
        return JSONResponse({"error": str(e)}, 400)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACCOUNT / PORTFOLIO ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/account")
def api_account():
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    try:
        return client.get_account()
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/positions")
def api_positions():
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    try:
        return client.get_positions()
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/orders")
def api_orders():
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    try:
        return client.get_orders(limit=20)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRADING ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post("/api/trade")
async def api_trade(request: Request):
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    body = await request.json()
    symbol = body.get("symbol", "").upper()
    qty = int(body.get("qty", 1))
    side = body.get("side", "buy")
    order_type = body.get("order_type", "market")
    limit_price = body.get("limit_price")

    if side == "buy":
        result = client.buy(symbol, qty=qty, order_type=order_type, limit_price=limit_price)
    else:
        result = client.sell(symbol, qty=qty, order_type=order_type, limit_price=limit_price)
    return result

@app.post("/api/close/{symbol}")
def api_close(symbol: str):
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    return client.close_position(symbol.upper())

@app.post("/api/close_all")
def api_close_all():
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    client.close_all()
    return {"status": "all closed"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PIPELINE ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post("/api/pipeline")
def api_pipeline():
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    def run():
        qt.run_full_pipeline(client)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"status": "pipeline started in background"}

@app.post("/api/download")
def api_download():
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    def run():
        qt.ingest_ohlcv(client, years=2)
        qt.compute_and_store_technicals()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"status": "download started"}

@app.get("/api/dbstats")
def api_dbstats():
    conn = qt.get_db()
    stats = {}
    for table in ["ohlcv", "technicals", "fundamentals", "macro", "signals",
                   "composite_scores", "trades", "ml_models", "backtests",
                   "private_exposure", "position_sizing", "market_regime", "portfolio_snapshots"]:
        try:
            stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except:
            stats[table] = 0
    conn.close()
    return stats

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYSIS ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/api/analyze/{symbol}")
def api_analyze(symbol: str):
    result = qt.run_all_strategies(symbol.upper())
    if not result:
        return JSONResponse({"error": f"No data for {symbol}. Run pipeline first."}, 404)
    return result

@app.get("/api/scan")
def api_scan():
    results = []
    for sym in qt.WATCHLIST:
        r = qt.run_all_strategies(sym)
        if r:
            results.append(r)
    results.sort(key=lambda x: x["composite"], reverse=True)
    return results

@app.get("/api/regime")
def api_regime():
    return qt.detect_market_regime()

@app.get("/api/predict/{symbol}")
def api_predict(symbol: str):
    signal, conf, reason = qt.ml_predict(symbol.upper())
    return {"symbol": symbol.upper(), "signal": signal, "confidence": conf, "reason": reason}

@app.get("/api/backtest/{symbol}/{strategy}")
def api_backtest(symbol: str, strategy: str):
    result = qt.backtest_strategy(symbol.upper(), strategy)
    if not result:
        return JSONResponse({"error": "Not enough data"}, 404)
    return result

@app.get("/api/ohlcv/{symbol}")
def api_ohlcv(symbol: str, days: int = 90):
    conn = qt.get_db()
    rows = conn.execute(
        "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT ?",
        (symbol.upper(), days)
    ).fetchall()
    conn.close()
    return [{"date": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in reversed(rows)]

@app.get("/api/watchlist")
def api_watchlist():
    return qt.WATCHLIST

@app.get("/api/private")
def api_private():
    qt.compute_private_exposure()
    conn = qt.get_db()
    rows = conn.execute("SELECT * FROM private_exposure ORDER BY date DESC").fetchall()
    conn.close()
    return [{"company": r[0], "date": r[1], "exposure": r[2], "return": r[3],
             "details": json.loads(r[4]) if r[4] else {}} for r in rows]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-TRADE ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.post("/api/auto/start")
async def api_auto_start(request: Request):
    global auto_trader
    if not client: return JSONResponse({"error": "Not connected"}, 401)
    body = await request.json()
    strategy = body.get("strategy", "composite")
    budget = float(body.get("budget", os.getenv("MAX_BUDGET", "10000")))
    max_pos = int(body.get("max_positions", os.getenv("MAX_POSITIONS", "3")))
    stop_loss = float(body.get("stop_loss", os.getenv("STOP_LOSS_PCT", "2.0")))
    take_profit = float(body.get("take_profit", os.getenv("TAKE_PROFIT_PCT", "5.0")))

    if auto_trader and auto_trader.running:
        auto_trader.stop()

    auto_trader = qt.AutoTrader(client, strategy, budget, max_pos, stop_loss, take_profit)
    auto_trader.start()
    return {"status": "started", "strategy": strategy, "budget": budget}

@app.post("/api/auto/stop")
def api_auto_stop():
    global auto_trader
    if auto_trader and auto_trader.running:
        auto_trader.stop()
        auto_trader = None
        return {"status": "stopped"}
    return {"status": "not running"}

@app.get("/api/auto/status")
def api_auto_status():
    if auto_trader and auto_trader.running:
        return {
            "running": True, "strategy": auto_trader.strategy,
            "budget": auto_trader.budget, "max_positions": auto_trader.max_positions,
            "stop_loss": auto_trader.stop_loss * 100,
            "take_profit": auto_trader.take_profit * 100,
        }
    return {"running": False}

@app.get("/api/trades")
def api_trades():
    conn = qt.get_db()
    rows = conn.execute(
        "SELECT timestamp, symbol, side, qty, price, strategy, status FROM trades ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [{"time": r[0], "symbol": r[1], "side": r[2], "qty": r[3],
             "price": r[4], "strategy": r[5], "status": r[6]} for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SERVE THE DASHBOARD HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return DASHBOARD_HTML


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE DASHBOARD (single-file HTML + CSS + JS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Trader v2.0</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: #0a0e17; --bg2: #111827; --bg3: #0d1321;
  --border: #1e2a3a; --border2: #2d3b4f;
  --text: #c8d6e5; --text2: #94a3b8; --text3: #64748b; --text4: #475569;
  --white: #e2e8f0;
  --green: #4ade80; --green-bg: #052e16; --green-border: #166534;
  --red: #f87171; --red-bg: #450a0a;
  --blue: #60a5fa; --blue-bg: #172554;
  --purple: #c084fc; --purple-bg: #3b0764;
  --yellow: #fbbf24;
  --accent1: #00d4ff; --accent2: #7c3aed; --accent3: #f472b6;
}

body { font-family: 'JetBrains Mono', monospace; background: var(--bg); color: var(--text); font-size: 13px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Header ── */
.header { background: linear-gradient(135deg, #0d1321, #1a1f36); border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100; }
.logo { font-size: 18px; font-weight: 800; background: linear-gradient(135deg, var(--accent1), var(--accent2), var(--accent3));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1px; }
.tagline { font-size: 9px; color: var(--text3); letter-spacing: 2px; text-transform: uppercase; }
.header-status { display: flex; align-items: center; gap: 10px; }

/* ── Tabs ── */
.tabs { display: flex; gap: 2px; background: var(--bg2); border-radius: 8px; padding: 3px; }
.tab { padding: 8px 18px; border-radius: 6px; border: none; cursor: pointer; font-size: 11px;
  font-weight: 600; font-family: inherit; background: transparent; color: var(--text3);
  letter-spacing: 0.5px; transition: all 0.2s; }
.tab.active { background: linear-gradient(135deg, #1e3a5f, #2d1b69); color: #fff; }
.tab:hover { color: var(--text); }

/* ── Layout ── */
.main { padding: 16px 24px; max-width: 1400px; margin: 0 auto; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; }
.grid-main { display: grid; grid-template-columns: 1fr 340px; gap: 12px; }

/* ── Cards ── */
.card { background: linear-gradient(145deg, var(--bg2), var(--bg3)); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.card-header { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px;
  color: var(--text3); margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; }

/* ── Stats ── */
.stat { text-align: center; padding: 12px; }
.stat-value { font-size: 22px; font-weight: 800; line-height: 1.2; }
.stat-label { font-size: 9px; color: var(--text4); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

/* ── Forms ── */
input, select { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 12px; color: var(--white); font-size: 12px; font-family: inherit;
  width: 100%; outline: none; }
input:focus, select:focus { border-color: var(--blue); }
label { font-size: 9px; color: var(--text4); text-transform: uppercase; letter-spacing: 1px;
  display: block; margin-bottom: 4px; }
.form-group { margin-bottom: 10px; }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

/* ── Buttons ── */
.btn { border: none; border-radius: 8px; padding: 10px 20px; font-size: 12px; font-weight: 700;
  cursor: pointer; font-family: inherit; letter-spacing: 0.5px; transition: all 0.15s; }
.btn:hover { opacity: 0.9; transform: translateY(-1px); }
.btn:active { transform: translateY(0); }
.btn-buy { background: linear-gradient(135deg, #22c55e, #16a34a); color: #fff; }
.btn-sell { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; }
.btn-primary { background: linear-gradient(135deg, #3b82f6, #2563eb); color: #fff; }
.btn-secondary { background: var(--border); color: var(--text2); border: 1px solid var(--border2); }
.btn-danger { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; }
.btn-sm { padding: 4px 12px; font-size: 10px; }
.btn-full { width: 100%; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px;
  font-weight: 700; letter-spacing: 0.5px; }
.badge-green { background: var(--green-bg); color: var(--green); }
.badge-red { background: var(--red-bg); color: var(--red); }
.badge-blue { background: var(--blue-bg); color: var(--blue); }
.badge-purple { background: var(--purple-bg); color: var(--purple); }
.badge-gray { background: #1e293b; color: var(--text2); }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text4);
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
td { padding: 8px 10px; border-bottom: 1px solid var(--bg2); }
tr:hover td { background: var(--bg2); }

/* ── Helpers ── */
.pnl-pos { color: var(--green); font-weight: 700; }
.pnl-neg { color: var(--red); font-weight: 700; }
.text-center { text-align: center; }
.text-muted { color: var(--text4); }
.mt-2 { margin-top: 8px; }
.mb-3 { margin-bottom: 12px; }
.gap-2 { gap: 8px; }
.flex { display: flex; }
.items-center { align-items: center; }
.justify-between { justify-content: space-between; }
.hidden { display: none; }
.scroll-y { max-height: 400px; overflow-y: auto; }

/* ── Panels ── */
.info-box { background: var(--bg3); border-radius: 8px; padding: 10px; font-size: 11px; }

/* ── Connect Screen ── */
.connect-screen { display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
.connect-box { max-width: 480px; width: 100%; }
.connect-logo { font-size: 36px; text-align: center; margin-bottom: 8px; }

/* ── Pulse animation ── */
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.pulse { animation: pulse 2s infinite; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-green { background: var(--green); }
.dot-red { background: var(--red); }
.dot-gray { background: var(--text4); }

/* ── Strategy cards ── */
.strat-card { padding: 12px; margin-bottom: 8px; border-radius: 8px; cursor: pointer;
  border: 1px solid var(--border); background: var(--bg3); transition: all 0.2s; }
.strat-card.active { border-color: var(--blue); background: rgba(30, 58, 95, 0.15); }
.strat-card:hover { border-color: var(--border2); }

/* ── Loading ── */
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border);
  border-top-color: var(--blue); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Toast ── */
.toast { position: fixed; bottom: 20px; right: 20px; background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 20px; font-size: 12px; z-index: 999; transition: all 0.3s;
  opacity: 0; transform: translateY(20px); }
.toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<div id="app">
  <!-- Connect Screen -->
  <div id="connect-screen" class="connect-screen">
    <div class="card connect-box">
      <div class="connect-logo">⚛</div>
      <div class="text-center mb-3">
        <div class="logo" style="font-size:28px;">QUANTUM TRADER</div>
        <div class="tagline">ALPACA PAPER TRADING • ML-POWERED</div>
      </div>
      <div class="form-group">
        <label>API Key</label>
        <input id="inp-key" type="text" placeholder="PKXXXXXXXXXXXXXXXXXX">
      </div>
      <div class="form-group">
        <label>Secret Key</label>
        <input id="inp-secret" type="password" placeholder="••••••••••••••••••••••">
      </div>
      <div class="form-group">
        <label>FRED API Key (optional — for macro data)</label>
        <input id="inp-fred" type="text" placeholder="Leave blank to skip">
      </div>
      <div class="form-row mb-3">
        <div class="form-group"><label>Budget ($)</label><input id="inp-budget" value="10000" type="number"></div>
        <div class="form-group"><label>Max Positions</label><input id="inp-maxpos" value="3" type="number"></div>
      </div>
      <div class="form-row mb-3">
        <div class="form-group"><label>Stop Loss %</label><input id="inp-sl" value="2.0" type="number" step="0.5"></div>
        <div class="form-group"><label>Take Profit %</label><input id="inp-tp" value="5.0" type="number" step="0.5"></div>
      </div>
      <div id="connect-error" style="color:var(--red);font-size:11px;margin-bottom:8px;display:none;"></div>
      <button class="btn btn-buy btn-full" onclick="doConnect()">
        <span id="connect-text">Connect to Alpaca Paper Trading</span>
        <span id="connect-spinner" class="spinner hidden"></span>
      </button>
      <p style="font-size:10px;color:var(--text4);text-align:center;margin-top:14px;line-height:1.6;">
        Paper trading only. Get keys at <a href="https://app.alpaca.markets" target="_blank" style="color:var(--blue);">app.alpaca.markets</a>
        → Paper Trading → API Keys
      </p>
    </div>
  </div>

  <!-- Main App (hidden until connected) -->
  <div id="main-app" class="hidden">
    <header class="header">
      <div>
        <div class="logo">⚛ QUANTUM TRADER</div>
        <div class="tagline">Alpaca Paper Trading • ML Engine</div>
      </div>
      <div class="tabs">
        <button class="tab active" onclick="switchTab('dashboard',this)">Dashboard</button>
        <button class="tab" onclick="switchTab('trade',this)">Trade</button>
        <button class="tab" onclick="switchTab('auto',this)">Auto Trade</button>
        <button class="tab" onclick="switchTab('analysis',this)">Analysis</button>
        <button class="tab" onclick="switchTab('data',this)">Data & ML</button>
      </div>
      <div class="header-status">
        <span id="auto-dot" class="dot dot-gray"></span>
        <span style="font-size:11px;color:var(--text3);">Paper</span>
        <span class="badge badge-green">● Connected</span>
      </div>
    </header>

    <main class="main">
      <!-- ═══ DASHBOARD TAB ═══ -->
      <div id="tab-dashboard">
        <div class="grid-4" id="account-stats"></div>
        <div class="card">
          <div class="card-header">Open Positions <span id="pos-count" class="badge badge-gray">0</span></div>
          <div id="positions-table"><div class="text-center text-muted" style="padding:20px;">Loading...</div></div>
        </div>
        <div class="card">
          <div class="card-header">Recent Trades</div>
          <div id="trades-table" class="scroll-y"><div class="text-center text-muted" style="padding:20px;">No trades yet</div></div>
        </div>
      </div>

      <!-- ═══ TRADE TAB ═══ -->
      <div id="tab-trade" class="hidden">
        <div class="grid-main">
          <div>
            <div class="card">
              <div class="card-header">Market Data</div>
              <div id="market-prices" style="display:flex;flex-wrap:wrap;gap:8px;"></div>
            </div>
            <div class="card">
              <div class="card-header">Order History</div>
              <div id="orders-table" class="scroll-y"></div>
            </div>
          </div>
          <div>
            <div class="card">
              <div class="card-header">Place Order</div>
              <div class="form-group">
                <label>Symbol</label>
                <input id="trade-symbol" type="text" value="AAPL" style="text-transform:uppercase;">
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label>Side</label>
                  <select id="trade-side"><option value="buy">BUY</option><option value="sell">SELL</option></select>
                </div>
                <div class="form-group">
                  <label>Quantity</label>
                  <input id="trade-qty" type="number" value="10" min="1">
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label>Type</label>
                  <select id="trade-type" onchange="document.getElementById('limit-group').style.display=this.value==='limit'?'block':'none'">
                    <option value="market">Market</option><option value="limit">Limit</option>
                  </select>
                </div>
                <div class="form-group" id="limit-group" style="display:none;">
                  <label>Limit Price ($)</label>
                  <input id="trade-limit" type="number" step="0.01">
                </div>
              </div>
              <button id="trade-btn" class="btn btn-buy btn-full mt-2" onclick="placeTrade()">▲ BUY AAPL</button>
            </div>
            <div class="card">
              <div class="card-header">Quick Actions</div>
              <div id="quick-close" style="font-size:11px;color:var(--text4);">No open positions to close</div>
            </div>
          </div>
        </div>
      </div>

      <!-- ═══ AUTO TRADE TAB ═══ -->
      <div id="tab-auto" class="hidden">
        <div class="grid-2">
          <div>
            <div class="card">
              <div class="card-header">Strategy <span id="auto-badge" class="badge badge-gray">○ OFF</span></div>
              <div id="strategy-cards"></div>
            </div>
          </div>
          <div>
            <div class="card">
              <div class="card-header">Auto-Trade Configuration</div>
              <div class="form-row mb-3">
                <div class="form-group"><label>Budget ($)</label><input id="auto-budget" type="number" value="10000"></div>
                <div class="form-group"><label>Max Positions</label><input id="auto-maxpos" type="number" value="3" min="1" max="10"></div>
              </div>
              <div class="form-row mb-3">
                <div class="form-group"><label>Stop Loss (%)</label><input id="auto-sl" type="number" value="2.0" step="0.5"></div>
                <div class="form-group"><label>Take Profit (%)</label><input id="auto-tp" type="number" value="5.0" step="0.5"></div>
              </div>
              <button id="auto-toggle-btn" class="btn btn-buy btn-full" onclick="toggleAuto()">▶ START AUTO-TRADE</button>
              <div id="auto-running-info" class="hidden mt-2" style="padding:10px;background:var(--green-bg);border:1px solid var(--green-border);border-radius:8px;font-size:11px;color:var(--green);">
                <span class="dot dot-green pulse"></span> Auto-trading active
              </div>
            </div>
            <div class="card">
              <div class="card-header">Auto-Trade Log</div>
              <div id="auto-log" class="scroll-y"><div class="text-center text-muted" style="padding:20px;">Enable auto-trade to begin</div></div>
            </div>
          </div>
        </div>
      </div>

      <!-- ═══ ANALYSIS TAB ═══ -->
      <div id="tab-analysis" class="hidden">
        <div class="card">
          <div class="card-header">Analyze Symbol</div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;">
            <input id="analyze-symbol" type="text" value="AAPL" style="width:120px;text-transform:uppercase;">
            <button class="btn btn-primary" onclick="doAnalyze()">Analyze</button>
            <button class="btn btn-secondary" onclick="doScan()">Scan All 30 Stocks</button>
          </div>
        </div>
        <div id="analysis-result"></div>
        <div id="scan-result" class="hidden"></div>
      </div>

      <!-- ═══ DATA TAB ═══ -->
      <div id="tab-data" class="hidden">
        <div class="grid-3" id="db-stats-cards"></div>
        <div class="card">
          <div class="card-header">Data Pipeline</div>
          <p style="font-size:11px;color:var(--text2);line-height:1.7;margin-bottom:12px;">
            Downloads 2 years of OHLCV from Alpaca, fetches fundamentals from yfinance,
            macro data from FRED, computes vectorized technicals (RSI, MACD, Bollinger, Hurst, ADX),
            detects market regime, and runs all 5 strategies across 30 stocks.
          </p>
          <div style="display:flex;gap:8px;">
            <button class="btn btn-primary" onclick="runPipeline()">▶ Run Full Pipeline (~2 min)</button>
            <button class="btn btn-secondary" onclick="runDownload()">Download Data Only</button>
          </div>
          <div id="pipeline-status" class="mt-2 hidden" style="padding:10px;background:var(--blue-bg);border:1px solid #1e40af;border-radius:8px;font-size:11px;color:var(--blue);">
            <span class="spinner"></span> Pipeline running in background...
          </div>
        </div>
        <div class="card">
          <div class="card-header">Database Tables</div>
          <div id="db-table-stats"></div>
        </div>
      </div>
    </main>
  </div>

  <div id="toast" class="toast"></div>
</div>

<script>
// ━━━━ STATE ━━━━
let selectedStrategy = 'composite';
let autoRunning = false;
let refreshInterval = null;

const STRATEGIES = {
  composite: { name: 'Composite (All 5)', type: 'adaptive', winRate: '60-70%', risk: 'Medium',
    desc: 'Runs RSI+MACD, Bollinger, Golden Cross, Momentum, and Hurst — aggregates into composite score' },
  rsi_macd: { name: 'RSI + MACD Confluence', type: 'quick', winRate: '58-65%', risk: 'Medium',
    desc: 'Oversold/overbought RSI + MACD histogram crossover signals' },
  bollinger: { name: 'Bollinger Mean Reversion', type: 'quick', winRate: '55-62%', risk: 'Low-Med',
    desc: 'Fades price extremes at Bollinger Band edges, uses Hurst to confirm mean-reversion regime' },
  golden_cross: { name: 'Golden/Death Cross', type: 'medium', winRate: '45-55%', risk: 'Medium',
    desc: 'SMA 50/200 crossover for major trend changes, confirmed by ADX trend strength' },
  momentum: { name: 'Momentum Breakout', type: 'quick', winRate: '50-60%', risk: 'Med-High',
    desc: 'Triple-timeframe momentum alignment (5d/10d/21d) with volume confirmation' },
  hurst: { name: 'Hurst Regime Adaptive', type: 'medium', winRate: '55-65%', risk: 'Medium',
    desc: 'Switches between trend-following and mean-reversion based on Hurst exponent' },
  ml: { name: 'ML Ensemble (LightGBM+RF+Ridge)', type: 'medium', winRate: '60-70%', risk: 'Medium',
    desc: 'Walk-forward trained ensemble predicting 21-day returns with 15 features' },
};

// ━━━━ API HELPERS ━━━━
async function api(url, opts = {}) {
  try {
    const res = await fetch(url, { headers: {'Content-Type': 'application/json'}, ...opts });
    return await res.json();
  } catch (e) { console.error(e); return { error: e.message }; }
}

function toast(msg, duration = 3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), duration);
}

// ━━━━ CONNECT ━━━━
async function doConnect() {
  const btn = document.getElementById('connect-text');
  const spinner = document.getElementById('connect-spinner');
  const errEl = document.getElementById('connect-error');
  btn.textContent = 'Connecting...';
  spinner.classList.remove('hidden');
  errEl.style.display = 'none';

  const data = await api('/api/connect', { method: 'POST', body: JSON.stringify({
    api_key: document.getElementById('inp-key').value,
    secret_key: document.getElementById('inp-secret').value,
    fred_key: document.getElementById('inp-fred').value,
    budget: document.getElementById('inp-budget').value,
    max_positions: document.getElementById('inp-maxpos').value,
    stop_loss: document.getElementById('inp-sl').value,
    take_profit: document.getElementById('inp-tp').value,
  })});

  spinner.classList.add('hidden');
  if (data.error) {
    btn.textContent = 'Connect to Alpaca Paper Trading';
    errEl.textContent = '❌ ' + data.error;
    errEl.style.display = 'block';
    return;
  }

  document.getElementById('connect-screen').classList.add('hidden');
  document.getElementById('main-app').classList.remove('hidden');
  toast('✅ Connected — $' + parseFloat(data.equity).toLocaleString());
  startRefresh();
  renderStrategyCards();
}

// Check if already connected on load
async function checkStatus() {
  const s = await api('/api/status');
  if (s.connected) {
    document.getElementById('connect-screen').classList.add('hidden');
    document.getElementById('main-app').classList.remove('hidden');
    startRefresh();
    renderStrategyCards();
  } else if (s.has_keys) {
    // Auto-connect with saved keys
    const data = await api('/api/connect', { method: 'POST', body: JSON.stringify({}) });
    // Keys are in .env already, but endpoint needs them — let user re-enter
  }
}

// ━━━━ DATA REFRESH ━━━━
function startRefresh() {
  refreshData();
  refreshInterval = setInterval(refreshData, 5000);
}

async function refreshData() {
  // Account
  const acct = await api('/api/account');
  if (!acct.error) {
    const totalPnl = (acct.equity || 100000) - 100000;
    document.getElementById('account-stats').innerHTML = [
      { label: 'Portfolio Value', value: '$' + (acct.equity||0).toLocaleString(undefined,{minimumFractionDigits:2}), color: 'var(--white)' },
      { label: 'Cash Available', value: '$' + (acct.cash||0).toLocaleString(undefined,{minimumFractionDigits:2}), color: 'var(--blue)' },
      { label: 'Buying Power', value: '$' + (acct.buying_power||0).toLocaleString(undefined,{minimumFractionDigits:2}), color: 'var(--purple)' },
      { label: 'Total P&L', value: (totalPnl>=0?'+':'') + '$' + totalPnl.toFixed(2), color: totalPnl >= 0 ? 'var(--green)' : 'var(--red)' },
    ].map(s => `<div class="card"><div class="stat"><div class="stat-value" style="color:${s.color}">${s.value}</div><div class="stat-label">${s.label}</div></div></div>`).join('');
  }

  // Positions
  const pos = await api('/api/positions');
  if (Array.isArray(pos)) {
    document.getElementById('pos-count').textContent = pos.length;
    if (pos.length === 0) {
      document.getElementById('positions-table').innerHTML = '<div class="text-center text-muted" style="padding:20px;">No open positions</div>';
      document.getElementById('quick-close').innerHTML = '<span class="text-muted">No positions to close</span>';
    } else {
      document.getElementById('positions-table').innerHTML = `<table>
        <tr><th>Symbol</th><th>Qty</th><th>Avg Price</th><th>Current</th><th>P&L</th><th>P&L %</th><th></th></tr>
        ${pos.map(p => `<tr>
          <td style="font-weight:700;color:var(--white)">${p.symbol}</td>
          <td>${p.qty}</td><td>$${p.avg_price.toFixed(2)}</td><td>$${p.current_price.toFixed(2)}</td>
          <td class="${p.pnl>=0?'pnl-pos':'pnl-neg'}">${p.pnl>=0?'+':''}$${p.pnl.toFixed(2)}</td>
          <td class="${p.pnl_pct>=0?'pnl-pos':'pnl-neg'}">${p.pnl_pct>=0?'+':''}${p.pnl_pct.toFixed(1)}%</td>
          <td><button class="btn btn-danger btn-sm" onclick="closePos('${p.symbol}')">Close</button></td>
        </tr>`).join('')}</table>`;
      document.getElementById('quick-close').innerHTML = pos.map(p =>
        `<button class="btn btn-secondary btn-sm" style="margin:2px;" onclick="closePos('${p.symbol}')">Close ${p.symbol}</button>`
      ).join('') + `<button class="btn btn-danger btn-sm" style="margin:2px;" onclick="closeAll()">Close All</button>`;
    }
  }

  // Trades
  const trades = await api('/api/trades');
  if (Array.isArray(trades) && trades.length > 0) {
    document.getElementById('trades-table').innerHTML = trades.slice(0, 20).map(t => `
      <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--bg2);font-size:11px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="badge ${t.side==='buy'?'badge-green':'badge-red'}">${t.side.toUpperCase()}</span>
          <span style="font-weight:700;color:var(--white)">${t.symbol}</span>
        </div>
        <div style="text-align:right;">
          <div style="color:var(--text2)">${t.qty} @ $${t.price||'mkt'}</div>
          <div style="font-size:9px;color:var(--text4)">${t.strategy || 'manual'} • ${t.time||''}</div>
        </div>
      </div>
    `).join('');
    // Also update auto log
    const autoTrades = trades.filter(t => t.strategy && t.strategy !== 'manual');
    if (autoTrades.length > 0) {
      document.getElementById('auto-log').innerHTML = autoTrades.slice(0, 30).map(t => `
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--bg2);font-size:11px;">
          <div><span class="badge ${t.side==='buy'?'badge-green':'badge-red'}">${t.side.toUpperCase()}</span>
          <span style="font-weight:700;margin-left:8px;">${t.symbol}</span>
          <span style="color:var(--text3);margin-left:8px;">${t.qty} shares</span></div>
          <div style="color:var(--text4);font-size:9px;">${t.strategy}</div>
        </div>
      `).join('');
    }
  }

  // Auto status
  const autoSt = await api('/api/auto/status');
  autoRunning = autoSt.running;
  document.getElementById('auto-dot').className = 'dot ' + (autoRunning ? 'dot-green pulse' : 'dot-gray');
  document.getElementById('auto-badge').className = 'badge ' + (autoRunning ? 'badge-green' : 'badge-gray');
  document.getElementById('auto-badge').textContent = autoRunning ? '● LIVE' : '○ OFF';
  document.getElementById('auto-toggle-btn').className = 'btn btn-full ' + (autoRunning ? 'btn-danger' : 'btn-buy');
  document.getElementById('auto-toggle-btn').textContent = autoRunning ? '⏹ STOP AUTO-TRADE' : '▶ START AUTO-TRADE';
  document.getElementById('auto-running-info').classList.toggle('hidden', !autoRunning);

  // Update trade button
  const side = document.getElementById('trade-side')?.value || 'buy';
  const sym = document.getElementById('trade-symbol')?.value || 'AAPL';
  const tradeBtn = document.getElementById('trade-btn');
  if (tradeBtn) {
    tradeBtn.className = 'btn btn-full mt-2 ' + (side === 'buy' ? 'btn-buy' : 'btn-sell');
    tradeBtn.textContent = (side === 'buy' ? '▲ BUY ' : '▼ SELL ') + sym.toUpperCase();
  }
}

// ━━━━ TRADING ━━━━
async function placeTrade() {
  const data = {
    symbol: document.getElementById('trade-symbol').value,
    qty: document.getElementById('trade-qty').value,
    side: document.getElementById('trade-side').value,
    order_type: document.getElementById('trade-type').value,
    limit_price: document.getElementById('trade-limit')?.value || null,
  };
  const result = await api('/api/trade', { method: 'POST', body: JSON.stringify(data) });
  if (result.error) { toast('❌ ' + result.error); }
  else { toast(`✅ ${data.side.toUpperCase()} ${data.qty} ${data.symbol} — ${result.status}`); }
  refreshData();
}

async function closePos(symbol) {
  await api('/api/close/' + symbol, { method: 'POST' });
  toast('🔒 Closed ' + symbol);
  refreshData();
}

async function closeAll() {
  await api('/api/close_all', { method: 'POST' });
  toast('🔒 All positions closed');
  refreshData();
}

// ━━━━ AUTO TRADE ━━━━
function renderStrategyCards() {
  document.getElementById('strategy-cards').innerHTML = Object.entries(STRATEGIES).map(([key, s]) => `
    <div class="strat-card ${selectedStrategy===key?'active':''}" onclick="selectedStrategy='${key}';renderStrategyCards();">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-weight:700;color:${selectedStrategy===key?'var(--blue)':'var(--white)'};font-size:12px;">${s.name}</span>
        <span class="badge ${s.type==='quick'?'badge-blue':'badge-purple'}">${s.type}</span>
      </div>
      <div style="font-size:10px;color:var(--text3);margin-bottom:6px;line-height:1.5;">${s.desc}</div>
      <div style="display:flex;gap:12px;font-size:10px;">
        <span><span style="color:var(--text4)">Win:</span> <span style="color:var(--green);font-weight:700">${s.winRate}</span></span>
        <span><span style="color:var(--text4)">Risk:</span> <span style="color:var(--yellow)">${s.risk}</span></span>
      </div>
    </div>
  `).join('');
}

async function toggleAuto() {
  if (autoRunning) {
    await api('/api/auto/stop', { method: 'POST' });
    toast('⏹ Auto-trade stopped');
  } else {
    const data = {
      strategy: selectedStrategy,
      budget: document.getElementById('auto-budget').value,
      max_positions: document.getElementById('auto-maxpos').value,
      stop_loss: document.getElementById('auto-sl').value,
      take_profit: document.getElementById('auto-tp').value,
    };
    await api('/api/auto/start', { method: 'POST', body: JSON.stringify(data) });
    toast('▶ Auto-trade started: ' + STRATEGIES[selectedStrategy].name);
  }
  refreshData();
}

// ━━━━ ANALYSIS ━━━━
async function doAnalyze() {
  const sym = document.getElementById('analyze-symbol').value.toUpperCase();
  document.getElementById('analysis-result').innerHTML = '<div class="card"><div class="spinner"></div> Analyzing ' + sym + '...</div>';
  document.getElementById('scan-result').classList.add('hidden');

  const r = await api('/api/analyze/' + sym);
  if (r.error) {
    document.getElementById('analysis-result').innerHTML = `<div class="card" style="color:var(--red)">❌ ${r.error}</div>`;
    return;
  }

  const ratingColor = r.rating.includes('Buy') ? 'var(--green)' : r.rating.includes('Sell') ? 'var(--red)' : 'var(--text2)';
  document.getElementById('analysis-result').innerHTML = `
    <div class="grid-3">
      <div class="card"><div class="stat">
        <div class="stat-value" style="color:${r.composite>=0?'var(--green)':'var(--red)'}">${r.composite>=0?'+':''}${r.composite.toFixed(2)}</div>
        <div class="stat-label">Composite Score</div>
      </div></div>
      <div class="card"><div class="stat">
        <div class="stat-value" style="color:${ratingColor}">${r.rating}</div>
        <div class="stat-label">Rating</div>
      </div></div>
      <div class="card"><div class="stat">
        <div class="stat-value" style="color:var(--blue)">${r.buy}B / ${r.sell}S / ${r.hold}H</div>
        <div class="stat-label">Signal Split</div>
      </div></div>
    </div>
    <div class="card">
      <div class="card-header">${sym} — 5 Strategy Breakdown</div>
      <table><tr><th>Strategy</th><th>Signal</th><th>Score</th><th>Reason</th></tr>
      ${r.strategies.map(s => `<tr>
        <td style="font-weight:600">${s.name}</td>
        <td><span class="badge ${s.signal==='BUY'?'badge-green':s.signal==='SELL'?'badge-red':'badge-gray'}">${s.signal}</span></td>
        <td class="${s.score>=0?'pnl-pos':'pnl-neg'}">${s.score>=0?'+':''}${s.score.toFixed(2)}</td>
        <td style="font-size:11px;color:var(--text2)">${s.reason}</td>
      </tr>`).join('')}</table>
    </div>
  `;
}

async function doScan() {
  document.getElementById('scan-result').classList.remove('hidden');
  document.getElementById('scan-result').innerHTML = '<div class="card"><span class="spinner"></span> Scanning 30 stocks...</div>';

  const results = await api('/api/scan');
  if (!Array.isArray(results)) {
    document.getElementById('scan-result').innerHTML = '<div class="card" style="color:var(--red)">❌ No data. Run pipeline first.</div>';
    return;
  }

  document.getElementById('scan-result').innerHTML = `<div class="card">
    <div class="card-header">Scan Results — ${results.length} Stocks Scored</div>
    <table><tr><th>Symbol</th><th>Score</th><th>Rating</th><th>Buy</th><th>Sell</th><th>Hold</th><th></th></tr>
    ${results.map(r => {
      const rc = r.rating.includes('Buy')?'pnl-pos':r.rating.includes('Sell')?'pnl-neg':'text-muted';
      return `<tr>
        <td style="font-weight:700;color:var(--white)">${r.symbol}</td>
        <td class="${r.composite>=0?'pnl-pos':'pnl-neg'}">${r.composite>=0?'+':''}${r.composite.toFixed(2)}</td>
        <td class="${rc}">${r.rating}</td>
        <td>${r.buy}</td><td>${r.sell}</td><td>${r.hold}</td>
        <td><button class="btn btn-secondary btn-sm" onclick="document.getElementById('analyze-symbol').value='${r.symbol}';doAnalyze();">Detail</button></td>
      </tr>`;
    }).join('')}</table>
  </div>`;
}

// ━━━━ DATA PIPELINE ━━━━
async function runPipeline() {
  document.getElementById('pipeline-status').classList.remove('hidden');
  await api('/api/pipeline', { method: 'POST' });
  toast('▶ Pipeline started — running in background');
  // Poll DB stats
  const poll = setInterval(async () => {
    await loadDbStats();
  }, 5000);
  setTimeout(() => { clearInterval(poll); document.getElementById('pipeline-status').classList.add('hidden'); toast('✅ Pipeline complete'); }, 120000);
}

async function runDownload() {
  document.getElementById('pipeline-status').classList.remove('hidden');
  await api('/api/download', { method: 'POST' });
  toast('▶ Download started');
}

async function loadDbStats() {
  const stats = await api('/api/dbstats');
  if (stats.error) return;
  const total = Object.values(stats).reduce((a, b) => a + b, 0);
  document.getElementById('db-stats-cards').innerHTML = `
    <div class="card"><div class="stat"><div class="stat-value" style="color:var(--blue)">${Object.keys(stats).length}</div><div class="stat-label">Tables</div></div></div>
    <div class="card"><div class="stat"><div class="stat-value" style="color:var(--purple)">${total.toLocaleString()}</div><div class="stat-label">Total Rows</div></div></div>
    <div class="card"><div class="stat"><div class="stat-value" style="color:var(--yellow)">${(stats.ohlcv||0).toLocaleString()}</div><div class="stat-label">OHLCV Bars</div></div></div>
  `;
  document.getElementById('db-table-stats').innerHTML = `<table>
    <tr><th>Table</th><th>Rows</th></tr>
    ${Object.entries(stats).map(([t, c]) => `<tr><td style="font-weight:600">${t}</td><td>${c.toLocaleString()}</td></tr>`).join('')}
  </table>`;
}

// ━━━━ TAB SWITCHING ━━━━
function switchTab(id, el) {
  document.querySelectorAll('[id^="tab-"]').forEach(t => t.classList.add('hidden'));
  document.getElementById('tab-' + id).classList.remove('hidden');
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  if (id === 'data') loadDbStats();
  if (id === 'trade') loadOrders();
}

async function loadOrders() {
  const orders = await api('/api/orders');
  if (Array.isArray(orders) && orders.length > 0) {
    document.getElementById('orders-table').innerHTML = `<table>
      <tr><th>Time</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Type</th><th>Status</th></tr>
      ${orders.map(o => `<tr>
        <td style="font-size:10px;color:var(--text3)">${(o.created||'').slice(0,19)}</td>
        <td><span class="badge ${o.side.includes('buy')?'badge-green':'badge-red'}">${o.side}</span></td>
        <td style="font-weight:700">${o.symbol}</td>
        <td>${o.qty}</td><td>${o.type}</td>
        <td><span class="badge badge-gray">${o.status}</span></td>
      </tr>`).join('')}</table>`;
  }
}

// Update trade button on input changes
document.addEventListener('change', (e) => {
  if (['trade-side', 'trade-symbol'].includes(e.target?.id)) {
    const side = document.getElementById('trade-side').value;
    const sym = document.getElementById('trade-symbol').value.toUpperCase();
    const btn = document.getElementById('trade-btn');
    btn.className = 'btn btn-full mt-2 ' + (side === 'buy' ? 'btn-buy' : 'btn-sell');
    btn.textContent = (side === 'buy' ? '▲ BUY ' : '▼ SELL ') + sym;
  }
});

// ━━━━ INIT ━━━━
checkStatus();
</script>
</body>
</html>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-CONNECT ON STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.on_event("startup")
async def startup():
    global client
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if api_key and api_key != "your_new_key_here" and secret_key:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            ac = qt.AlpacaClient.__new__(qt.AlpacaClient)
            ac.api_key = api_key
            ac.secret_key = secret_key
            ac.trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
            ac.data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
            acct = ac.trading.get_account()
            client = ac
            qt.init_db()
            log.info(f"✅ Auto-connected: ${float(acct.equity):,.2f}")
        except Exception as e:
            log.warning(f"Auto-connect failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAUNCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║   ⚛  QUANTUM TRADER v2.0 — Web Dashboard     ║")
    print("  ╚═══════════════════════════════════════════════╝")
    print()
    print("  Opening http://localhost:8000 in your browser...")
    print("  Press Ctrl+C to stop the server.")
    print()

    # Open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
