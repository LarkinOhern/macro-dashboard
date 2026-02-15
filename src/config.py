"""Central configuration for all macroeconomic indicators."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "macro_indicators.db"
STATUS_PATH = DATA_DIR / "data_status.json"

# FRED API
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# How many years of history to load on initial setup
HISTORY_YEARS = 5

# ---------------------------------------------------------------------------
# FRED Indicators by category
# ---------------------------------------------------------------------------
FRED_INDICATORS = {
    "growth": [
        {
            "series_id": "A191RL1Q225SBEA",
            "name": "Real GDP Growth",
            "unit": "%",
            "freq": "quarterly",
            "description": "Real GDP percent change from preceding period (annualized)",
        },
        {
            "series_id": "UNRATE",
            "name": "Unemployment Rate",
            "unit": "%",
            "freq": "monthly",
            "description": "Civilian unemployment rate, seasonally adjusted",
        },
        {
            "series_id": "ICSA",
            "name": "Initial Jobless Claims",
            "unit": "thousands",
            "freq": "weekly",
            "description": "Initial claims for unemployment insurance",
        },
        {
            "series_id": "PAYEMS",
            "name": "Nonfarm Payrolls",
            "unit": "thousands",
            "freq": "monthly",
            "description": "All employees, total nonfarm, seasonally adjusted",
        },
        {
            "series_id": "MANEMP",
            "name": "Manufacturing Employment",
            "unit": "thousands",
            "freq": "monthly",
            "description": "All employees, manufacturing, seasonally adjusted",
        },
        {
            "series_id": "UMCSENT",
            "name": "Consumer Sentiment (UMich)",
            "unit": "index",
            "freq": "monthly",
            "description": "University of Michigan consumer sentiment index",
        },
    ],
    "inflation": [
        {
            "series_id": "CPIAUCSL",
            "name": "CPI (All Urban)",
            "unit": "index",
            "freq": "monthly",
            "description": "Consumer Price Index for all urban consumers, seasonally adjusted",
        },
        {
            "series_id": "PCEPILFE",
            "name": "Core PCE",
            "unit": "index",
            "freq": "monthly",
            "description": "Personal consumption expenditures excluding food and energy (chain-type price index)",
        },
        {
            "series_id": "T5YIE",
            "name": "5-Year Breakeven Inflation",
            "unit": "%",
            "freq": "daily",
            "description": "5-year breakeven inflation rate",
        },
        {
            "series_id": "T10YIE",
            "name": "10-Year Breakeven Inflation",
            "unit": "%",
            "freq": "daily",
            "description": "10-year breakeven inflation rate",
        },
    ],
    "credit": [
        {
            "series_id": "BAMLH0A0HYM2",
            "name": "HY Credit Spread (OAS)",
            "unit": "%",
            "freq": "daily",
            "description": "ICE BofA US High Yield Index Option-Adjusted Spread",
        },
        {
            "series_id": "T10Y2Y",
            "name": "2s10s Yield Curve Spread",
            "unit": "%",
            "freq": "daily",
            "description": "10-Year Treasury minus 2-Year Treasury constant maturity",
        },
        {
            "series_id": "DGS10",
            "name": "10-Year Treasury Yield",
            "unit": "%",
            "freq": "daily",
            "description": "Market yield on U.S. Treasury securities at 10-year constant maturity",
        },
        {
            "series_id": "DGS2",
            "name": "2-Year Treasury Yield",
            "unit": "%",
            "freq": "daily",
            "description": "Market yield on U.S. Treasury securities at 2-year constant maturity",
        },
        {
            "series_id": "GFDEGDQ188S",
            "name": "Federal Debt to GDP",
            "unit": "%",
            "freq": "quarterly",
            "description": "Federal debt held by the public as percent of GDP",
        },
    ],
    "monetary": [
        {
            "series_id": "FEDFUNDS",
            "name": "Effective Fed Funds Rate",
            "unit": "%",
            "freq": "monthly",
            "description": "Effective federal funds rate",
        },
        {
            "series_id": "M2SL",
            "name": "M2 Money Supply",
            "unit": "billions",
            "freq": "monthly",
            "description": "M2 money stock, seasonally adjusted",
        },
        {
            "series_id": "WALCL",
            "name": "Fed Balance Sheet (Total Assets)",
            "unit": "millions",
            "freq": "weekly",
            "description": "Federal Reserve total assets",
        },
    ],
}

# ---------------------------------------------------------------------------
# Market Indicators (via yfinance)
# ---------------------------------------------------------------------------
MARKET_INDICATORS = [
    {"ticker": "^GSPC", "name": "S&P 500", "category": "markets"},
    {"ticker": "^VIX", "name": "VIX Volatility", "category": "markets"},
    {"ticker": "DX-Y.NYB", "name": "US Dollar Index (DXY)", "category": "markets"},
    {"ticker": "GC=F", "name": "Gold Futures", "category": "markets"},
    {"ticker": "CL=F", "name": "WTI Crude Oil", "category": "markets"},
]

# ---------------------------------------------------------------------------
# Display thresholds for traffic-light coloring on the Economic Pulse page
# (green, yellow, red) — thresholds are (low_boundary, high_boundary)
# Values below low → green (or red for inverted), above high → red (or green)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "UNRATE": {"green": (0, 4.5), "yellow": (4.5, 6.0), "red": (6.0, 100)},
    "A191RL1Q225SBEA": {"green": (2, 100), "yellow": (0, 2), "red": (-100, 0)},
    "CPIAUCSL_YOY": {"green": (0, 2.5), "yellow": (2.5, 4.0), "red": (4.0, 100)},
    "FEDFUNDS": {"green": (0, 3), "yellow": (3, 5), "red": (5, 100)},
    "T10Y2Y": {"green": (0.5, 100), "yellow": (0, 0.5), "red": (-100, 0)},
    "BAMLH0A0HYM2": {"green": (0, 4), "yellow": (4, 6), "red": (6, 100)},
    "^VIX": {"green": (0, 20), "yellow": (20, 30), "red": (30, 200)},
}

# ---------------------------------------------------------------------------
# Regime classification helpers
# ---------------------------------------------------------------------------
REGIME_LABELS = {
    (True, False): "Goldilocks",      # growth rising, inflation falling
    (True, True): "Reflation",        # growth rising, inflation rising
    (False, True): "Stagflation",     # growth falling, inflation rising
    (False, False): "Deflation",      # growth falling, inflation falling
}

REGIME_COLORS = {
    "Goldilocks": "#2ecc71",
    "Reflation": "#f39c12",
    "Stagflation": "#e74c3c",
    "Deflation": "#3498db",
}

REGIME_DESCRIPTIONS = {
    "Goldilocks": "Growth rising, inflation falling — the ideal environment. Equities tend to perform well.",
    "Reflation": "Growth rising, inflation rising — early-cycle expansion. Commodities and value stocks benefit.",
    "Stagflation": "Growth falling, inflation rising — the worst quadrant. Cash and commodities may hedge.",
    "Deflation": "Growth falling, inflation falling — risk-off environment. Bonds and quality stocks outperform.",
}
