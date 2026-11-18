"""FastAPI server for the browser dashboard.

The server can place paper orders and save API keys, so it is meant for your own machine only:
it binds to 127.0.0.1 by default, only answers requests addressed to a local host name, and
rejects state-changing requests that come from another site (see ``_guard_unsafe_requests``).
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import resources
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from trading_bot import engine
from trading_bot.backtest.costs import CostModel
from trading_bot.config import ENV_PATH, Settings, load_settings, write_env_file
from trading_bot.data.db import connect, init_db, table_counts
from trading_bot.data.ingest import ingest_ohlcv
from trading_bot.data.universe import WATCHLIST
from trading_bot.strategies import resolve_strategy

log = logging.getLogger(__name__)

LOCAL_HOSTS = ["127.0.0.1", "localhost"]
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class ConnectBody(BaseModel):
    api_key: str = ""
    secret_key: str = ""
    fred_key: str = ""
    budget: float | None = None
    max_positions: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class TradeBody(BaseModel):
    symbol: str
    qty: int = 1
    side: str = "buy"
    order_type: str = "market"
    limit_price: float | None = None


class AutoStartBody(BaseModel):
    strategy: str = "composite"
    budget: float | None = None
    max_positions: int | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class AppState:
    settings: Settings
    connect_broker: object  # callable(api_key, secret_key, db_path, costs) -> broker
    broker: object | None = None
    auto_trader: object | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def costs(self) -> CostModel:
        return CostModel(self.settings.tx_cost_bps, self.settings.slippage_bps)


def _default_connect(api_key, secret_key, db_path, costs):
    from trading_bot.broker.alpaca import PaperBroker

    return PaperBroker.connect(api_key, secret_key, db_path, costs)


def _dashboard_html() -> str:
    return resources.files("trading_bot.dashboard").joinpath("static/index.html").read_text(encoding="utf-8")


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(settings: Settings | None = None, connect_broker=None, auto_connect: bool = True) -> FastAPI:
    """Build the app. Tests pass their own ``settings`` and a fake ``connect_broker``."""
    state = AppState(settings=settings or load_settings(), connect_broker=connect_broker or _default_connect)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(state.settings.db_path)
        if auto_connect and state.settings.has_alpaca_keys:
            try:
                s = state.settings
                state.broker = state.connect_broker(s.alpaca_api_key, s.alpaca_secret_key, s.db_path, state.costs)
            except Exception as exc:
                log.warning("Auto-connect with the keys in the environment failed: %s", exc)
        yield
        if state.auto_trader is not None and state.auto_trader.running:
            state.auto_trader.stop()

    app = FastAPI(title="Trading bot dashboard", lifespan=lifespan)
    app.state.bot = state
    allowed_hosts = sorted({*LOCAL_HOSTS, state.settings.dashboard_host})
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def _guard_unsafe_requests(request: Request, call_next):
        """Block cross-site requests that try to change state.

        A page on another site can make the browser send a simple form POST here, but it cannot
        send ``Content-Type: application/json`` cross-origin without a CORS preflight, which this
        server never approves. The dashboard's own JavaScript always sends that header.
        """
        if request.method in UNSAFE_METHODS:
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type != "application/json":
                return _error("state-changing requests must send Content-Type: application/json", 415)
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return _error("cross-origin request blocked", 403)
        return await call_next(request)

    # Setup
    @app.get("/api/status")
    def api_status():
        return {"has_keys": state.settings.has_alpaca_keys, "connected": state.broker is not None}

    @app.post("/api/connect")
    def api_connect(body: ConnectBody):
        """Connect with the keys in the form, or with the keys already in the environment."""
        s = state.settings
        api_key = body.api_key.strip() or s.alpaca_api_key
        secret_key = body.secret_key.strip() or s.alpaca_secret_key
        if not api_key or not secret_key:
            return _error("API key and secret are required", 400)
        try:
            broker = state.connect_broker(api_key, secret_key, s.db_path, state.costs)
        except Exception as exc:
            return _error(str(exc), 400)

        if body.api_key.strip():
            # New keys typed into the form: save them so the next start connects on its own.
            values = {
                "ALPACA_API_KEY": api_key,
                "ALPACA_SECRET_KEY": secret_key,
                "FRED_API_KEY": body.fred_key.strip() or (s.fred_api_key or ""),
                "MAX_BUDGET": body.budget if body.budget is not None else s.max_budget,
                "MAX_POSITIONS": body.max_positions if body.max_positions is not None else s.max_positions,
                "STOP_LOSS_PCT": body.stop_loss if body.stop_loss is not None else s.stop_loss_pct,
                "TAKE_PROFIT_PCT": body.take_profit if body.take_profit is not None else s.take_profit_pct,
            }
            try:
                write_env_file(values, ENV_PATH)
            except ValueError as exc:
                return _error(str(exc), 400)
            state.settings = load_settings(ENV_PATH, override=True)

        state.broker = broker
        init_db(state.settings.db_path)
        account = broker.get_account()
        return {"status": "connected", "equity": account["equity"], "cash": account["cash"]}

    # Account and portfolio
    def _broker_call(fn_name: str, *args, **kwargs):
        broker = state.broker
        if broker is None:
            return _error("Not connected", 401)
        try:
            return getattr(broker, fn_name)(*args, **kwargs)
        except Exception as exc:
            return _error(str(exc), 500)

    @app.get("/api/account")
    def api_account():
        return _broker_call("get_account")

    @app.get("/api/positions")
    def api_positions():
        return _broker_call("get_positions")

    @app.get("/api/orders")
    def api_orders():
        return _broker_call("get_orders", limit=20)

    # Trading
    @app.post("/api/trade")
    def api_trade(body: TradeBody):
        method = "buy" if body.side == "buy" else "sell"
        return _broker_call(
            method, body.symbol.upper(), qty=body.qty, order_type=body.order_type, limit_price=body.limit_price
        )

    @app.post("/api/close/{symbol}")
    def api_close(symbol: str):
        return _broker_call("close_position", symbol.upper())

    @app.post("/api/close_all")
    def api_close_all():
        out = _broker_call("close_all")
        return out if isinstance(out, JSONResponse) else {"status": "all closed"}

    # Data pipeline
    def _in_background(target) -> None:
        threading.Thread(target=target, daemon=True).start()

    @app.post("/api/pipeline")
    def api_pipeline():
        if state.broker is None:
            return _error("Not connected", 401)
        s = state.settings
        _in_background(lambda: engine.run_full_pipeline(state.broker, s.db_path, s.fred_api_key))
        return {"status": "pipeline started in background"}

    @app.post("/api/download")
    def api_download():
        if state.broker is None:
            return _error("Not connected", 401)
        db = state.settings.db_path

        def run():
            ingest_ohlcv(state.broker.data, db, years=2)
            engine.compute_and_store_technicals(db)

        _in_background(run)
        return {"status": "download started"}

    @app.get("/api/dbstats")
    def api_dbstats():
        return table_counts(state.settings.db_path)

    # Analysis
    @app.get("/api/analyze/{symbol}")
    def api_analyze(symbol: str):
        result = engine.analyze_symbol(state.settings.db_path, symbol.upper())
        if not result:
            return _error(f"No data for {symbol.upper()}. Run the pipeline first.", 404)
        return result

    @app.get("/api/scan")
    def api_scan():
        return engine.scan(state.settings.db_path)

    @app.get("/api/regime")
    def api_regime():
        return engine.detect_market_regime(state.settings.db_path)

    @app.get("/api/predict/{symbol}")
    def api_predict(symbol: str):
        signal, confidence, reason = engine.predict_symbol(state.settings.db_path, symbol.upper())
        return {"symbol": symbol.upper(), "signal": signal, "confidence": confidence, "reason": reason}

    @app.get("/api/backtest/{symbol}/{strategy}")
    def api_backtest(symbol: str, strategy: str):
        result = engine.backtest_symbol(state.settings.db_path, symbol.upper(), resolve_strategy(strategy), state.costs)
        if not result:
            return _error("Unknown strategy or not enough data", 404)
        return result

    @app.get("/api/ohlcv/{symbol}")
    def api_ohlcv(symbol: str, days: int = 90):
        with connect(state.settings.db_path) as conn:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM ohlcv WHERE symbol=? ORDER BY date DESC LIMIT ?",
                (symbol.upper(), days),
            ).fetchall()
        keys = ("date", "open", "high", "low", "close", "volume")
        return [dict(zip(keys, row, strict=True)) for row in reversed(rows)]

    @app.get("/api/watchlist")
    def api_watchlist():
        return WATCHLIST

    @app.get("/api/private")
    def api_private():
        return engine.compute_private_exposure(state.settings.db_path)

    # Auto-trading
    @app.post("/api/auto/start")
    def api_auto_start(body: AutoStartBody):
        from trading_bot.trader import AutoTrader

        if state.broker is None:
            return _error("Not connected", 401)
        s = state.settings
        with state.lock:
            if state.auto_trader is not None and state.auto_trader.running:
                state.auto_trader.stop()
            try:
                trader = AutoTrader(
                    state.broker,
                    s.db_path,
                    resolve_strategy(body.strategy),
                    body.budget if body.budget is not None else s.max_budget,
                    body.max_positions if body.max_positions is not None else s.max_positions,
                    body.stop_loss if body.stop_loss is not None else s.stop_loss_pct,
                    body.take_profit if body.take_profit is not None else s.take_profit_pct,
                    s.scan_interval_seconds,
                )
            except ValueError as exc:
                return _error(str(exc), 400)
            trader.start()
            state.auto_trader = trader
        return {"status": "started", "strategy": trader.strategy, "budget": trader.budget}

    @app.post("/api/auto/stop")
    def api_auto_stop():
        with state.lock:
            if state.auto_trader is not None and state.auto_trader.running:
                state.auto_trader.stop()
                state.auto_trader = None
                return {"status": "stopped"}
        return {"status": "not running"}

    @app.get("/api/auto/status")
    def api_auto_status():
        t = state.auto_trader
        if t is not None and t.running:
            return {"running": True, "strategy": t.strategy, "budget": t.budget, "max_positions": t.max_positions,
                    "stop_loss": t.stop_loss * 100, "take_profit": t.take_profit * 100}  # fmt: skip
        return {"running": False}

    @app.get("/api/trades")
    def api_trades():
        with connect(state.settings.db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, symbol, side, qty, price, strategy, status FROM trades ORDER BY id DESC LIMIT 50"
            ).fetchall()
        keys = ("time", "symbol", "side", "qty", "price", "strategy", "status")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _dashboard_html()

    return app


def main(open_browser: bool = True) -> None:
    import webbrowser

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    url = f"http://{settings.dashboard_host}:{settings.dashboard_port}"
    print(f"Dashboard at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(settings), host=settings.dashboard_host, port=settings.dashboard_port, log_level="warning")
