# Macro Economic Indicator Dashboard

A Ray Dalio-inspired macroeconomic indicator dashboard built with Streamlit. Tracks 25 indicators across 6 categories with automated daily data updates.

## Features

- **Economic Pulse** — At-a-glance regime classification (Goldilocks / Reflation / Stagflation / Deflation)
- **Growth & Employment** — GDP, unemployment, jobless claims, payrolls, consumer sentiment
- **Inflation Monitor** — CPI, Core PCE, breakeven inflation rates, expectations vs actual
- **Credit & Yield Curve** — 2s10s spread, HY credit spread, treasury yields, debt-to-GDP
- **Monetary Policy** — Fed Funds rate, M2 money supply, Fed balance sheet, real rates
- **Markets & Risk** — S&P 500, VIX, DXY, gold, oil with moving averages and correlation heatmap
- **Regime Tracker** — Dalio's 4-quadrant framework with historical regime timeline

## Setup

1. **Get a free FRED API key** at https://fred.stlouisfed.org/docs/api/api_key.html

2. **Create `.env` file:**
   ```
   FRED_API_KEY=your_key_here
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Load historical data (one-time, ~5 years):**
   ```bash
   cd src
   python create_database.py
   ```

5. **Launch the dashboard:**
   ```bash
   streamlit run src/app.py
   ```

## Daily Updates

- **Manual:** `cd src && python update_data.py`
- **Automated:** GitHub Actions runs daily at 9:30 AM ET (market open). Add `FRED_API_KEY` as a repository secret.

## Data Sources

| Source | Indicators | Auth |
|--------|-----------|------|
| FRED (Federal Reserve) | ~20 economic series | Free API key |
| Yahoo Finance | S&P 500, VIX, DXY, Gold, Oil | None |

## Tech Stack

- **Data:** Python, fredapi, yfinance, SQLite
- **Dashboard:** Streamlit, Plotly
- **Automation:** GitHub Actions
