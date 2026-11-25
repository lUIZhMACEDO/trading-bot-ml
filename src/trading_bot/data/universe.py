"""What the bot tracks: the watchlist, the FRED series and the private-company proxy weights."""

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    "SPY", "QQQ", "AMD", "NFLX", "JPM", "V", "BA", "DIS",
    "CRM", "COST", "UNH", "HD", "PG", "INTC", "PYPL",
    "SQ", "COIN", "PLTR", "SOFI", "RIVN", "SNOW", "NET", "DDOG",
]  # fmt: skip

# Rough exposure to private companies through public names. The weights are my own guesses from
# revenue exposure, partnerships and supply chain; they are a heuristic, not a valuation.
PRIVATE_COMPANY_PROXIES = {
    "Anthropic": {"AMZN": 0.27, "GOOG": 0.10, "NVDA": 0.30, "MSFT": 0.08, "META": 0.05},
    "SpaceX": {"BA": 0.15, "LMT": 0.12, "RTX": 0.10, "GOOG": 0.08, "TSLA": 0.20},
    "OpenAI": {"MSFT": 0.40, "NVDA": 0.25, "GOOG": 0.05, "CRM": 0.05},
    "Stripe": {"V": 0.20, "MA": 0.15, "SQ": 0.15, "PYPL": 0.10, "AMZN": 0.08},
    "xAI": {"TSLA": 0.35, "NVDA": 0.25, "GOOG": 0.10, "AMD": 0.08},
}

FRED_SERIES = {
    "DGS2": "treasury_2y",
    "DGS10": "treasury_10y",
    "T10Y2Y": "yield_curve_spread",
    "VIXCLS": "vix",
    "BAMLH0A0HYM2": "hy_credit_spread",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds_rate",
    "DCOILWTICO": "crude_oil",
    "DTWEXBGS": "usd_index",
}
