from pathlib import Path

import pytest

from trading_bot.config import Settings, load_settings, write_env_file


def test_defaults():
    s = Settings.from_env({})
    assert s.db_path == Path("trading_bot.db")
    assert (s.max_budget, s.max_positions, s.stop_loss_pct, s.take_profit_pct) == (10_000, 3, 2.0, 5.0)
    assert (s.tx_cost_bps, s.slippage_bps) == (5.0, 2.0)
    assert s.dashboard_host == "127.0.0.1"
    assert not s.has_alpaca_keys


def test_reads_environment_values():
    s = Settings.from_env(
        {"ALPACA_API_KEY": "PK1234567890ABCDEFGH", "ALPACA_SECRET_KEY": "s" * 40, "MAX_POSITIONS": "5",
         "TX_COST_BPS": "1.5", "DB_PATH": "other.db"}
    )  # fmt: skip
    assert s.has_alpaca_keys
    assert s.max_positions == 5 and s.tx_cost_bps == 1.5 and s.db_path == Path("other.db")


@pytest.mark.parametrize("key", ["your_api_key_here", "your_new_key_here", "", "   ", "short"])
def test_placeholder_or_short_keys_do_not_count(key):
    assert not Settings.from_env({"ALPACA_API_KEY": key, "ALPACA_SECRET_KEY": "s" * 40}).has_alpaca_keys


def test_bad_numbers_fall_back_to_defaults():
    s = Settings.from_env({"MAX_BUDGET": "lots", "MAX_POSITIONS": "2.5"})
    assert s.max_budget == 10_000 and s.max_positions == 3


def test_write_env_file_round_trip(tmp_path):
    env = tmp_path / ".env"
    write_env_file({"ALPACA_API_KEY": "PK1234567890ABCDEFGH", "ALPACA_SECRET_KEY": "x" * 40, "MAX_BUDGET": 2500}, env)
    s = load_settings(env, override=True)
    assert s.has_alpaca_keys and s.max_budget == 2500


def test_write_env_file_rejects_line_breaks(tmp_path):
    env = tmp_path / ".env"
    with pytest.raises(ValueError, match="line break"):
        write_env_file({"MAX_BUDGET": "100\nALPACA_API_KEY=attacker"}, env)
    assert not env.exists()
