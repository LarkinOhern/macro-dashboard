"""Macro Economic Indicator Dashboard — Streamlit App.

A Ray Dalio-inspired dashboard for tracking macroeconomic conditions.
Seven pages accessible via sidebar navigation.
"""

import json
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    DB_PATH,
    FRED_INDICATORS,
    MARKET_INDICATORS,
    REGIME_COLORS,
    REGIME_DESCRIPTIONS,
    REGIME_LABELS,
    STATUS_PATH,
    THRESHOLDS,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection():
    """Return a shared SQLite connection (cached across reruns)."""
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache_data(ttl=3600)
def query_fred(series_id: str, years: int = 5) -> pd.DataFrame:
    """Fetch a FRED series from the database."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, value FROM fred_indicators WHERE series_id = ? AND date >= ? ORDER BY date",
        conn,
        params=(series_id, cutoff),
        parse_dates=["date"],
    )
    return df


@st.cache_data(ttl=3600)
def query_market(ticker: str, years: int = 5) -> pd.DataFrame:
    """Fetch a market series from the database."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM market_indicators WHERE ticker = ? AND date >= ? ORDER BY date",
        conn,
        params=(ticker, cutoff),
        parse_dates=["date"],
    )
    return df


def latest_value(series_id: str, source: str = "fred") -> tuple:
    """Return (value, date, prev_value) for the latest and second-latest observation."""
    conn = get_connection()
    if source == "fred":
        rows = pd.read_sql_query(
            "SELECT date, value FROM fred_indicators WHERE series_id = ? ORDER BY date DESC LIMIT 2",
            conn,
            params=(series_id,),
        )
        if rows.empty:
            return (None, None, None)
        val = rows.iloc[0]["value"]
        dt = rows.iloc[0]["date"]
        prev = rows.iloc[1]["value"] if len(rows) > 1 else None
        return (val, dt, prev)
    else:
        rows = pd.read_sql_query(
            "SELECT date, close FROM market_indicators WHERE ticker = ? ORDER BY date DESC LIMIT 2",
            conn,
            params=(series_id,),
        )
        if rows.empty:
            return (None, None, None)
        val = rows.iloc[0]["close"]
        dt = rows.iloc[0]["date"]
        prev = rows.iloc[1]["close"] if len(rows) > 1 else None
        return (val, dt, prev)


def direction_arrow(current, previous) -> str:
    if current is None or previous is None:
        return "—"
    if current > previous * 1.001:
        return "▲"
    elif current < previous * 0.999:
        return "▼"
    return "▶"


def traffic_light_color(series_id: str, value: float) -> str:
    """Return hex color based on threshold config."""
    if series_id not in THRESHOLDS or value is None:
        return "#e0e0e0"
    t = THRESHOLDS[series_id]
    if t["green"][0] <= value <= t["green"][1]:
        return "#2ecc71"
    elif t["yellow"][0] <= value <= t["yellow"][1]:
        return "#f39c12"
    return "#e74c3c"


def compute_yoy(series_id: str) -> pd.DataFrame:
    """Compute year-over-year % change for a level series."""
    df = query_fred(series_id, years=6)
    if df.empty:
        return df
    df = df.set_index("date").sort_index()
    df["yoy"] = df["value"].pct_change(periods=12) * 100
    return df.dropna(subset=["yoy"]).reset_index()


def get_data_status() -> dict:
    """Read data_status.json."""
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {}


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime() -> tuple:
    """Determine the current Dalio regime quadrant.

    Returns (regime_name, growth_rising, inflation_rising, confidence).
    """
    # Growth signal: GDP trend + unemployment direction
    gdp = query_fred("A191RL1Q225SBEA", years=2)
    unemp = query_fred("UNRATE", years=2)

    growth_rising = True
    if not gdp.empty:
        recent_gdp = gdp.tail(2)
        if len(recent_gdp) == 2:
            growth_rising = recent_gdp.iloc[-1]["value"] >= recent_gdp.iloc[-2]["value"]

    if not unemp.empty:
        recent_unemp = unemp.tail(3)
        if len(recent_unemp) >= 2:
            unemp_falling = recent_unemp.iloc[-1]["value"] <= recent_unemp.iloc[-2]["value"]
            # Combine: growth rising if GDP up OR unemployment falling
            growth_rising = growth_rising or unemp_falling

    # Inflation signal: CPI YoY trend + Core PCE trend
    cpi = compute_yoy("CPIAUCSL")
    pce = compute_yoy("PCEPILFE")

    inflation_rising = True
    signals = []
    if not cpi.empty and len(cpi) >= 3:
        signals.append(cpi.iloc[-1]["yoy"] > cpi.iloc[-3]["yoy"])
    if not pce.empty and len(pce) >= 3:
        signals.append(pce.iloc[-1]["yoy"] > pce.iloc[-3]["yoy"])

    if signals:
        inflation_rising = sum(signals) > len(signals) / 2

    regime = REGIME_LABELS.get((growth_rising, inflation_rising), "Unknown")
    # Simple confidence: how many sub-signals agree
    confidence = 0.7  # baseline
    return regime, growth_rising, inflation_rising, confidence


def regime_history() -> pd.DataFrame:
    """Build a quarterly regime history from stored data."""
    gdp = query_fred("A191RL1Q225SBEA", years=5)
    cpi_yoy = compute_yoy("CPIAUCSL")

    if gdp.empty or cpi_yoy.empty:
        return pd.DataFrame()

    # Resample CPI to quarterly to align with GDP
    cpi_q = cpi_yoy.set_index("date").resample("QE").last().dropna(subset=["yoy"]).reset_index()

    # Merge on nearest quarter
    gdp = gdp.copy()
    gdp["quarter"] = gdp["date"].dt.to_period("Q")
    cpi_q["quarter"] = cpi_q["date"].dt.to_period("Q")

    merged = gdp.merge(cpi_q[["quarter", "yoy"]], on="quarter", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged = merged.rename(columns={"value": "gdp_growth", "yoy": "cpi_yoy"})

    # Direction changes
    merged["growth_rising"] = merged["gdp_growth"].diff().fillna(0) >= 0
    merged["inflation_rising"] = merged["cpi_yoy"].diff().fillna(0) >= 0
    merged["regime"] = merged.apply(
        lambda r: REGIME_LABELS.get((r["growth_rising"], r["inflation_rising"]), "Unknown"),
        axis=1,
    )
    merged["color"] = merged["regime"].map(REGIME_COLORS)
    return merged


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=30),
    font=dict(size=12),
    hovermode="x unified",
)


def line_chart(df: pd.DataFrame, y: str = "value", title: str = "", yaxis_title: str = "",
               color: str = "#4da6ff", height: int = 350, target_line: float = None) -> go.Figure:
    """Standard line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[y], mode="lines", line=dict(color=color, width=2), name=title,
    ))
    if target_line is not None:
        fig.add_hline(y=target_line, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Target: {target_line}%")
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=yaxis_title,
        height=height,
        showlegend=False,
    )
    return fig


def dual_line_chart(df1, df2, name1, name2, title, y="value", height=350) -> go.Figure:
    """Two lines on one chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1["date"], y=df1[y], mode="lines", name=name1,
                             line=dict(color="#4da6ff", width=2)))
    fig.add_trace(go.Scatter(x=df2["date"], y=df2[y], mode="lines", name=name2,
                             line=dict(color="#f39c12", width=2)))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(size=14)),
                      height=height, legend=dict(orientation="h", y=-0.15))
    return fig


def sparkline(df: pd.DataFrame, y: str = "value", color: str = "#4da6ff") -> go.Figure:
    """Tiny sparkline chart for metric cards."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[y], mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 Macro Dashboard")
    st.markdown("*Ray Dalio-inspired economic indicator tracker*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Economic Pulse",
            "Growth & Employment",
            "Inflation Monitor",
            "Credit & Yield Curve",
            "Monetary Policy",
            "Markets & Risk",
            "Regime Tracker",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Data freshness
    status = get_data_status()
    if status.get("last_update"):
        last = datetime.fromisoformat(status["last_update"])
        age = datetime.now() - last
        if age.days == 0:
            freshness = "Updated today"
            color = "🟢"
        elif age.days <= 1:
            freshness = "Updated yesterday"
            color = "🟡"
        else:
            freshness = f"Updated {age.days}d ago"
            color = "🔴"
        st.caption(f"{color} {freshness}")
        st.caption(f"Last: {last.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.caption("⚪ No data loaded yet")
        st.caption("Run `python src/create_database.py`")

    st.markdown("---")
    st.caption("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.")


# ---------------------------------------------------------------------------
# Check DB exists
# ---------------------------------------------------------------------------
if not DB_PATH.exists():
    st.error("Database not found. Run `python src/create_database.py` first to load historical data.")
    st.stop()


# ===========================================================================
# PAGE 1: Economic Pulse
# ===========================================================================
if page == "Economic Pulse":
    st.title("Economic Pulse")
    st.markdown("**Where are we now?** A snapshot of key macro indicators.")

    # --- Dalio Regime Quadrant ---
    regime, growth_up, inflation_up, confidence = classify_regime()
    regime_color = REGIME_COLORS.get(regime, "#888")

    col_regime, col_desc = st.columns([1, 2])
    with col_regime:
        st.markdown(
            f"""
            <div style="background:{regime_color}22; border:2px solid {regime_color};
                        border-radius:12px; padding:20px; text-align:center;">
                <h2 style="color:{regime_color}; margin:0;">{regime}</h2>
                <p style="margin:4px 0 0 0; font-size:0.9em;">
                    Growth {'↑' if growth_up else '↓'} &nbsp;|&nbsp;
                    Inflation {'↑' if inflation_up else '↓'}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_desc:
        st.markdown(f"**Current Regime:** {regime}")
        st.markdown(REGIME_DESCRIPTIONS.get(regime, ""))

    st.markdown("---")

    # --- Key Metrics Row ---
    metrics = [
        ("A191RL1Q225SBEA", "GDP Growth", "fred", "%"),
        ("UNRATE", "Unemployment", "fred", "%"),
        ("FEDFUNDS", "Fed Funds", "fred", "%"),
        ("T10Y2Y", "2s10s Spread", "fred", "%"),
    ]
    market_metrics = [
        ("^GSPC", "S&P 500", "yfinance", ""),
    ]

    cols = st.columns(len(metrics) + len(market_metrics))

    for i, (sid, label, source, unit) in enumerate(metrics + market_metrics):
        val, dt, prev = latest_value(sid, source)
        arrow = direction_arrow(val, prev)
        color = traffic_light_color(sid, val)

        with cols[i]:
            if val is not None:
                display_val = f"{val:,.2f}{unit}" if unit else f"{val:,.2f}"
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span style='color:#aaa; font-size:0.8em;'>{label}</span><br>"
                    f"<span style='color:{color}; font-size:1.5em; font-weight:bold;'>{display_val}</span>"
                    f"<span style='font-size:1.1em;'> {arrow}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Sparkline
                if source == "fred":
                    spark_df = query_fred(sid, years=1)
                    if not spark_df.empty:
                        st.plotly_chart(sparkline(spark_df, color=color), use_container_width=True)
                else:
                    spark_df = query_market(sid, years=1)
                    if not spark_df.empty:
                        spark_df = spark_df.rename(columns={"close": "value"})
                        st.plotly_chart(sparkline(spark_df, color=color), use_container_width=True)
            else:
                st.metric(label, "N/A")

    st.markdown("---")

    # Additional metrics row
    st.markdown("#### Additional Indicators")
    extras = [
        ("CPIAUCSL", "CPI", "fred"),
        ("BAMLH0A0HYM2", "HY Spread", "fred"),
        ("^VIX", "VIX", "yfinance"),
        ("DGS10", "10Y Yield", "fred"),
        ("UMCSENT", "Consumer Sent.", "fred"),
    ]
    ecols = st.columns(len(extras))
    for i, (sid, label, source) in enumerate(extras):
        val, dt, prev = latest_value(sid, source)
        with ecols[i]:
            if val is not None:
                delta = val - prev if prev else None
                st.metric(label, f"{val:.2f}", delta=f"{delta:+.2f}" if delta else None)
            else:
                st.metric(label, "N/A")


# ===========================================================================
# PAGE 2: Growth & Employment
# ===========================================================================
elif page == "Growth & Employment":
    st.title("Growth & Employment")

    col1, col2 = st.columns(2)

    with col1:
        gdp = query_fred("A191RL1Q225SBEA")
        if not gdp.empty:
            st.plotly_chart(
                line_chart(gdp, title="Real GDP Growth (Quarterly)", yaxis_title="%", color="#2ecc71"),
                use_container_width=True,
            )

        payrolls = query_fred("PAYEMS")
        if not payrolls.empty:
            payrolls["change"] = payrolls["value"].diff()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=payrolls["date"], y=payrolls["change"],
                marker_color=payrolls["change"].apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c"),
                name="Monthly Change",
            ))
            fig.update_layout(**CHART_LAYOUT, title="Nonfarm Payrolls (Monthly Change)", height=350,
                              yaxis_title="Thousands")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        unemp = query_fred("UNRATE")
        if not unemp.empty:
            st.plotly_chart(
                line_chart(unemp, title="Unemployment Rate", yaxis_title="%", color="#e74c3c"),
                use_container_width=True,
            )

        claims = query_fred("ICSA", years=2)
        if not claims.empty:
            st.plotly_chart(
                line_chart(claims, title="Initial Jobless Claims (Weekly)", yaxis_title="Thousands",
                           color="#f39c12"),
                use_container_width=True,
            )

    # Consumer sentiment — full width
    sent = query_fred("UMCSENT")
    if not sent.empty:
        st.plotly_chart(
            line_chart(sent, title="Consumer Sentiment (U. Michigan)", yaxis_title="Index",
                       color="#9b59b6"),
            use_container_width=True,
        )

    # Manufacturing employment
    mfg = query_fred("MANEMP")
    if not mfg.empty:
        st.plotly_chart(
            line_chart(mfg, title="Manufacturing Employment", yaxis_title="Thousands",
                       color="#3498db"),
            use_container_width=True,
        )


# ===========================================================================
# PAGE 3: Inflation Monitor
# ===========================================================================
elif page == "Inflation Monitor":
    st.title("Inflation Monitor")

    col1, col2 = st.columns(2)

    with col1:
        cpi_yoy = compute_yoy("CPIAUCSL")
        if not cpi_yoy.empty:
            st.plotly_chart(
                line_chart(cpi_yoy, y="yoy", title="CPI Year-over-Year %", yaxis_title="%",
                           color="#e74c3c", target_line=2.0),
                use_container_width=True,
            )

    with col2:
        pce_yoy = compute_yoy("PCEPILFE")
        if not pce_yoy.empty:
            st.plotly_chart(
                line_chart(pce_yoy, y="yoy", title="Core PCE Year-over-Year %", yaxis_title="%",
                           color="#f39c12", target_line=2.0),
                use_container_width=True,
            )

    st.markdown("#### Market Inflation Expectations (Breakevens)")
    col3, col4 = st.columns(2)
    with col3:
        t5 = query_fred("T5YIE")
        if not t5.empty:
            st.plotly_chart(
                line_chart(t5, title="5-Year Breakeven Inflation", yaxis_title="%", color="#3498db"),
                use_container_width=True,
            )
    with col4:
        t10 = query_fred("T10YIE")
        if not t10.empty:
            st.plotly_chart(
                line_chart(t10, title="10-Year Breakeven Inflation", yaxis_title="%", color="#2ecc71"),
                use_container_width=True,
            )

    # Expectations vs actual comparison
    st.markdown("#### Inflation Expectations vs. Actual")
    if not cpi_yoy.empty and not t5.empty:
        fig = dual_line_chart(
            cpi_yoy[["date", "yoy"]].rename(columns={"yoy": "value"}),
            t5, "CPI YoY (Actual)", "5Y Breakeven (Expected)",
            title="Actual vs. Expected Inflation",
        )
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4: Credit & Yield Curve
# ===========================================================================
elif page == "Credit & Yield Curve":
    st.title("Credit & Yield Curve")

    # Yield curve spread
    spread = query_fred("T10Y2Y")
    if not spread.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spread["date"], y=spread["value"], mode="lines",
            line=dict(color="#4da6ff", width=2), name="2s10s Spread",
            fill="tozeroy",
            fillcolor="rgba(77,166,255,0.1)",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                      annotation_text="Inversion (Recession Signal)")
        fig.update_layout(**CHART_LAYOUT, title="2s10s Yield Curve Spread",
                          yaxis_title="%", height=400)

        # Current state
        current_spread = spread.iloc[-1]["value"]
        if current_spread > 0.5:
            shape = "Normal"
        elif current_spread > 0:
            shape = "Flat"
        else:
            shape = "INVERTED"
        st.markdown(f"**Current Yield Curve Shape:** {shape} ({current_spread:+.2f}%)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        dgs10 = query_fred("DGS10")
        dgs2 = query_fred("DGS2")
        if not dgs10.empty and not dgs2.empty:
            st.plotly_chart(
                dual_line_chart(dgs10, dgs2, "10-Year", "2-Year",
                                title="Treasury Yields: 10Y vs 2Y"),
                use_container_width=True,
            )

    with col2:
        hy = query_fred("BAMLH0A0HYM2")
        if not hy.empty:
            fig = line_chart(hy, title="High Yield Credit Spread (OAS)", yaxis_title="%",
                             color="#e74c3c")
            fig.add_hline(y=5, line_dash="dot", line_color="#f39c12",
                          annotation_text="Stress threshold")
            st.plotly_chart(fig, use_container_width=True)

    # Federal debt to GDP
    debt = query_fred("GFDEGDQ188S")
    if not debt.empty:
        st.plotly_chart(
            line_chart(debt, title="Federal Debt to GDP", yaxis_title="%", color="#9b59b6"),
            use_container_width=True,
        )


# ===========================================================================
# PAGE 5: Monetary Policy
# ===========================================================================
elif page == "Monetary Policy":
    st.title("Monetary Policy & Liquidity")

    col1, col2 = st.columns(2)

    with col1:
        ffr = query_fred("FEDFUNDS")
        if not ffr.empty:
            st.plotly_chart(
                line_chart(ffr, title="Effective Fed Funds Rate", yaxis_title="%", color="#e74c3c"),
                use_container_width=True,
            )

        # Real interest rate = Fed Funds - CPI YoY
        cpi_yoy = compute_yoy("CPIAUCSL")
        if not ffr.empty and not cpi_yoy.empty:
            ffr_m = ffr.set_index("date").resample("ME").last().reset_index()
            cpi_m = cpi_yoy.set_index("date").resample("ME").last().reset_index()
            merged = ffr_m.merge(cpi_m[["date", "yoy"]], on="date", how="inner")
            merged["real_rate"] = merged["value"] - merged["yoy"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged["date"], y=merged["real_rate"], mode="lines",
                line=dict(color="#2ecc71", width=2), name="Real Rate",
                fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="#888")
            fig.update_layout(**CHART_LAYOUT, title="Real Interest Rate (Fed Funds − CPI YoY)",
                              yaxis_title="%", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        m2 = query_fred("M2SL")
        if not m2.empty:
            st.plotly_chart(
                line_chart(m2, title="M2 Money Supply", yaxis_title="$ Billions", color="#3498db"),
                use_container_width=True,
            )
            # M2 YoY growth
            m2_yoy = m2.set_index("date").sort_index()
            m2_yoy["yoy"] = m2_yoy["value"].pct_change(periods=12) * 100
            m2_yoy = m2_yoy.dropna(subset=["yoy"]).reset_index()
            if not m2_yoy.empty:
                st.plotly_chart(
                    line_chart(m2_yoy, y="yoy", title="M2 YoY Growth Rate", yaxis_title="%",
                               color="#9b59b6"),
                    use_container_width=True,
                )

    # Fed balance sheet — full width
    bs = query_fred("WALCL")
    if not bs.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bs["date"], y=bs["value"] / 1e6, mode="lines",
            line=dict(color="#f39c12", width=2), name="Total Assets",
            fill="tozeroy", fillcolor="rgba(243,156,18,0.1)",
        ))
        fig.update_layout(**CHART_LAYOUT, title="Fed Balance Sheet (Total Assets)",
                          yaxis_title="$ Trillions", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 6: Markets & Risk
# ===========================================================================
elif page == "Markets & Risk":
    st.title("Markets & Risk")

    # S&P 500 with moving averages
    sp = query_market("^GSPC")
    if not sp.empty:
        sp["MA50"] = sp["close"].rolling(50).mean()
        sp["MA200"] = sp["close"].rolling(200).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sp["date"], y=sp["close"], mode="lines",
                                 line=dict(color="#4da6ff", width=2), name="S&P 500"))
        fig.add_trace(go.Scatter(x=sp["date"], y=sp["MA50"], mode="lines",
                                 line=dict(color="#f39c12", width=1, dash="dot"), name="50-Day MA"))
        fig.add_trace(go.Scatter(x=sp["date"], y=sp["MA200"], mode="lines",
                                 line=dict(color="#e74c3c", width=1, dash="dot"), name="200-Day MA"))
        fig.update_layout(**CHART_LAYOUT, title="S&P 500 with Moving Averages",
                          yaxis_title="Price", height=450,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    # VIX with stress thresholds
    col1, col2 = st.columns(2)
    with col1:
        vix = query_market("^VIX")
        if not vix.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vix["date"], y=vix["close"], mode="lines",
                                     line=dict(color="#e74c3c", width=2), name="VIX"))
            for level, color_line, label in [(20, "#f39c12", "Caution"), (30, "#e74c3c", "Stress"),
                                              (40, "#8e44ad", "Panic")]:
                fig.add_hline(y=level, line_dash="dot", line_color=color_line,
                              annotation_text=label)
            fig.update_layout(**CHART_LAYOUT, title="VIX Volatility Index",
                              yaxis_title="Index", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        dxy = query_market("DX-Y.NYB")
        if not dxy.empty:
            fig = line_chart(
                dxy.rename(columns={"close": "value"}),
                title="US Dollar Index (DXY)", yaxis_title="Index", color="#2ecc71",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        gold = query_market("GC=F")
        if not gold.empty:
            st.plotly_chart(
                line_chart(gold.rename(columns={"close": "value"}),
                           title="Gold Futures", yaxis_title="$/oz", color="#f1c40f", height=350),
                use_container_width=True,
            )
    with col4:
        oil = query_market("CL=F")
        if not oil.empty:
            st.plotly_chart(
                line_chart(oil.rename(columns={"close": "value"}),
                           title="WTI Crude Oil", yaxis_title="$/bbl", color="#e67e22", height=350),
                use_container_width=True,
            )

    # Cross-asset correlation heatmap
    st.markdown("#### Cross-Asset Correlation (Rolling 30-Day)")
    tickers = ["^GSPC", "^VIX", "DX-Y.NYB", "GC=F", "CL=F"]
    names = ["S&P 500", "VIX", "DXY", "Gold", "Oil"]
    frames = {}
    for t, n in zip(tickers, names):
        df = query_market(t, years=1)
        if not df.empty:
            frames[n] = df.set_index("date")["close"].pct_change()

    if len(frames) >= 2:
        returns = pd.DataFrame(frames)
        corr = returns.rolling(30).corr().dropna()
        # Get latest correlation matrix
        if not corr.empty:
            latest_date = corr.index.get_level_values(0)[-1]
            corr_matrix = corr.loc[latest_date]

            fig = px.imshow(
                corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
            )
            fig.update_layout(**CHART_LAYOUT, title=f"30-Day Rolling Correlation (as of {latest_date.strftime('%Y-%m-%d')})",
                              height=400)
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 7: Regime Tracker
# ===========================================================================
elif page == "Regime Tracker":
    st.title("Economic Regime Tracker")
    st.markdown("*Based on Ray Dalio's framework: Growth × Inflation = Four Quadrants*")

    # Current regime
    regime, growth_up, inflation_up, confidence = classify_regime()
    regime_color = REGIME_COLORS.get(regime, "#888")

    # Quadrant diagram
    st.markdown("### Current Regime")
    cols = st.columns(3)

    with cols[0]:
        # 2x2 grid
        for gr_label, gr_val in [("Growth ↑", True), ("Growth ↓", False)]:
            c1, c2 = st.columns(2)
            for col_widget, (inf_label, inf_val) in zip([c1, c2], [("Inflation ↓", False), ("Inflation ↑", True)]):
                r = REGIME_LABELS.get((gr_val, inf_val), "?")
                rc = REGIME_COLORS.get(r, "#888")
                is_current = (r == regime)
                border = f"3px solid {rc}" if is_current else f"1px solid #333"
                opacity = "1.0" if is_current else "0.4"
                with col_widget:
                    st.markdown(
                        f"<div style='border:{border}; border-radius:8px; padding:10px; "
                        f"text-align:center; opacity:{opacity}; background:{rc}22;'>"
                        f"<strong style='color:{rc};'>{r}</strong><br>"
                        f"<span style='font-size:0.75em;'>{gr_label}, {inf_label}</span></div>",
                        unsafe_allow_html=True,
                    )

    with cols[1]:
        st.markdown(f"**Regime:** {regime}")
        st.markdown(f"**Growth:** {'Rising ↑' if growth_up else 'Falling ↓'}")
        st.markdown(f"**Inflation:** {'Rising ↑' if inflation_up else 'Falling ↓'}")
        st.markdown("---")
        st.markdown(REGIME_DESCRIPTIONS.get(regime, ""))

    with cols[2]:
        st.markdown("**Quadrant Reference:**")
        for r_name, r_desc in REGIME_DESCRIPTIONS.items():
            rc = REGIME_COLORS[r_name]
            st.markdown(f"<span style='color:{rc};'>● **{r_name}:**</span> {r_desc}",
                        unsafe_allow_html=True)

    st.markdown("---")

    # Regime history timeline
    st.markdown("### Regime History")
    hist = regime_history()
    if not hist.empty:
        fig = go.Figure()
        for r_name in REGIME_LABELS.values():
            mask = hist["regime"] == r_name
            if mask.any():
                fig.add_trace(go.Bar(
                    x=hist.loc[mask, "date"],
                    y=[1] * mask.sum(),
                    marker_color=REGIME_COLORS[r_name],
                    name=r_name,
                    hovertemplate="%{x|%Y-Q%q}<br>" + r_name,
                ))
        fig.update_layout(
            **CHART_LAYOUT,
            title="Regime History (Quarterly)",
            barmode="stack",
            height=200,
            yaxis=dict(visible=False),
            showlegend=True,
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot: GDP growth vs CPI YoY colored by regime
        st.markdown("### Growth vs. Inflation Scatter")
        fig = px.scatter(
            hist, x="gdp_growth", y="cpi_yoy", color="regime",
            color_discrete_map=REGIME_COLORS,
            labels={"gdp_growth": "GDP Growth (%)", "cpi_yoy": "CPI YoY (%)"},
            hover_data=["date"],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        # Mark the latest point
        latest = hist.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[latest["gdp_growth"]], y=[latest["cpi_yoy"]],
            mode="markers+text", text=["NOW"], textposition="top center",
            marker=dict(size=15, color=REGIME_COLORS.get(latest["regime"], "#fff"),
                        line=dict(width=2, color="white")),
            showlegend=False,
        ))
        fig.update_layout(**CHART_LAYOUT, title="Historical Regime Positions",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Not enough data to compute regime history. Load data with `create_database.py`.")

    # Asset performance by regime table
    st.markdown("### Typical Asset Performance by Regime")
    perf_data = {
        "Regime": ["Goldilocks", "Reflation", "Stagflation", "Deflation"],
        "Equities": ["Strong ↑", "Moderate ↑", "Weak ↓", "Moderate ↓"],
        "Bonds": ["Moderate ↑", "Weak ↓", "Weak ↓", "Strong ↑"],
        "Commodities": ["Neutral", "Strong ↑", "Moderate ↑", "Weak ↓"],
        "Gold": ["Neutral", "Moderate ↑", "Strong ↑", "Moderate ↑"],
        "Cash": ["Underperform", "Underperform", "Hold value", "Moderate"],
    }
    st.dataframe(pd.DataFrame(perf_data).set_index("Regime"), use_container_width=True)
