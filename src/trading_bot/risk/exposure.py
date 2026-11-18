"""Synthetic exposure to private companies through weighted public proxy stocks."""

from __future__ import annotations


def proxy_exposure(last_two_closes: dict[str, tuple[float, float]], weights: dict[str, float]) -> tuple[float, dict]:
    """Weighted one-day return of a private company's public proxies.

    ``last_two_closes`` maps symbol to (latest close, previous close). Proxies without two closes
    are skipped. Returns the weighted return and a per-proxy breakdown in percent.
    """
    weighted, details = 0.0, {}
    for symbol, weight in weights.items():
        if symbol not in last_two_closes:
            continue
        latest, previous = last_two_closes[symbol]
        ret = (latest - previous) / previous
        weighted += ret * weight
        details[symbol] = {"weight": weight, "return": round(ret * 100, 2)}
    return weighted, details
