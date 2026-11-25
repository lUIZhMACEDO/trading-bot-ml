"""Runtime settings.

Every key and tunable comes from an environment variable, optionally loaded from a local
``.env`` file. Nothing secret lives in the source tree; ``.env.example`` lists every variable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_PATH = Path(".env")

# Values that ship in .env.example or were written by older versions of the setup prompt. They count as "no key".
PLACEHOLDER_KEYS = {"", "your_api_key_here", "your_secret_key_here", "your_new_key_here", "your_fred_key_here"}

# Alpaca paper accounts start at $100k. Account P&L is reported relative to this.
PAPER_STARTING_EQUITY = 100_000.0


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return None if value in PLACEHOLDER_KEYS else value


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the environment."""

    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    fred_api_key: str | None = None

    db_path: Path = Path("trading_bot.db")

    # Auto-trader defaults (the dashboard and CLI can override them per run)
    max_budget: float = 10_000.0
    max_positions: int = 3
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 5.0
    scan_interval_seconds: int = 60

    # Cost model used by the backtester and the trade log
    tx_cost_bps: float = 5.0
    slippage_bps: float = 2.0

    # Dashboard server
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000

    @property
    def has_alpaca_keys(self) -> bool:
        """True when both keys are set to something that is not a placeholder.

        Alpaca key ids are 20 characters and secrets about 40, so anything much shorter is a
        typo or a leftover placeholder.
        """
        key, secret = self.alpaca_api_key or "", self.alpaca_secret_key or ""
        return len(key) > 10 and len(secret) > 10

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            alpaca_api_key=_clean(env.get("ALPACA_API_KEY")),
            alpaca_secret_key=_clean(env.get("ALPACA_SECRET_KEY")),
            fred_api_key=_clean(env.get("FRED_API_KEY")),
            db_path=Path(env.get("DB_PATH") or cls.db_path),
            max_budget=_as_float(env.get("MAX_BUDGET"), cls.max_budget),
            max_positions=_as_int(env.get("MAX_POSITIONS"), cls.max_positions),
            stop_loss_pct=_as_float(env.get("STOP_LOSS_PCT"), cls.stop_loss_pct),
            take_profit_pct=_as_float(env.get("TAKE_PROFIT_PCT"), cls.take_profit_pct),
            scan_interval_seconds=_as_int(env.get("SCAN_INTERVAL_SECONDS"), cls.scan_interval_seconds),
            tx_cost_bps=_as_float(env.get("TX_COST_BPS"), cls.tx_cost_bps),
            slippage_bps=_as_float(env.get("SLIPPAGE_BPS"), cls.slippage_bps),
            dashboard_host=env.get("DASHBOARD_HOST") or cls.dashboard_host,
            dashboard_port=_as_int(env.get("DASHBOARD_PORT"), cls.dashboard_port),
        )


def load_settings(env_path: Path = ENV_PATH, override: bool = False) -> Settings:
    """Load ``.env`` (if present) into the process environment, then read settings from it."""
    from dotenv import load_dotenv

    load_dotenv(env_path, override=override)
    return Settings.from_env()


def write_env_file(values: Mapping[str, object], path: Path = ENV_PATH) -> None:
    """Write ``KEY=value`` lines to a local ``.env`` file.

    Used by the first-run prompt and the dashboard's connect form. A value with a line break is
    rejected, since it could otherwise add extra variables to the file.
    """
    lines = []
    for key, value in values.items():
        text = "" if value is None else str(value).strip()
        if any(ch in text for ch in "\r\n\0"):
            raise ValueError(f"{key} contains a line break")
        lines.append(f"{key}={text}")
    path.write_text("# Local settings. This file holds secrets and is git-ignored.\n" + "\n".join(lines) + "\n")
