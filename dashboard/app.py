"""
Streamlit OMS Blotter & Portfolio Terminal:
Visualizes real-time and backtest blotter, positions, Greeks,
Gross vs. Net equity curve, and cumulative friction breakdown.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from config.settings import DATABASE_PATH
from oms.database import get_connection

st.set_page_config(
    page_title="Antigravity Quant OMS Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-metric-card {
        background-color: #1a1e24;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2e3640;
    }
    .metric-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
    .metric-delta-pos { font-size: 0.9rem; color: #3fb950; }
    .metric-delta-neg { font-size: 0.9rem; color: #f85149; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Multi-Asset & Options AI Quant Terminal")
st.caption("Self-Hosted Local OMS • C++ Microstructure Friction • Real-Time Greeks & Blotter")

# Sidebar
st.sidebar.header("🕹️ Controls & Navigation")
auto_refresh = st.sidebar.checkbox("Auto Refresh (Every 5s)", value=False)
if auto_refresh:
    time_delay = 5
    st.sidebar.info(f"Auto-refreshing every {time_delay}s")

# Load Data from oms.db
with get_connection(DATABASE_PATH) as conn:
    ledger_df = pd.read_sql("SELECT * FROM account_ledger ORDER BY ledger_id ASC", conn)
    pos_df = pd.read_sql("SELECT * FROM positions ORDER BY opened_at DESC", conn)
    exec_df = pd.read_sql("SELECT * FROM executions ORDER BY fill_timestamp DESC", conn)
    orders_df = pd.read_sql("SELECT * FROM orders ORDER BY created_at DESC", conn)
    signals_df = pd.read_sql("SELECT * FROM signals ORDER BY created_at DESC", conn)

# Top KPI Summary Cards
if not ledger_df.empty:
    latest = ledger_df.iloc[-1]
    nlv = float(latest["total_equity"])
    cash = float(latest["cash_balance"])
    realized_pnl = float(latest["realized_pnl"])
    unrealized_pnl = float(latest["unrealized_pnl"])
    cum_friction = float(latest["cumulative_friction_usd"])
else:
    nlv, cash, realized_pnl, unrealized_pnl, cum_friction = 100000.0, 100000.0, 0.0, 0.0, 0.0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Net Liquidation Value", f"${nlv:,.2f}", delta=f"${(realized_pnl + unrealized_pnl):+,.2f}")
with col2:
    st.metric("Available Cash", f"${cash:,.2f}")
with col3:
    st.metric("Realized PnL", f"${realized_pnl:+,.2f}")
with col4:
    st.metric("Unrealized PnL", f"${unrealized_pnl:+,.2f}")
with col5:
    st.metric("Cumulative Friction Paid", f"${cum_friction:,.2f}", help="Realistic C++ calculated slippage, spread-crossing & fees")

# Portfolio Greeks Cards
open_positions = pos_df[pos_df["status"] == "OPEN"] if not pos_df.empty else pd.DataFrame()
net_delta = open_positions["delta"].sum() if not open_positions.empty and "delta" in open_positions.columns else 0.0
net_gamma = open_positions["gamma"].sum() if not open_positions.empty and "gamma" in open_positions.columns else 0.0
net_vega = open_positions["vega"].sum() if not open_positions.empty and "vega" in open_positions.columns else 0.0
net_theta = open_positions["theta"].sum() if not open_positions.empty and "theta" in open_positions.columns else 0.0

st.write("---")
g_col1, g_col2, g_col3, g_col4 = st.columns(4)
with g_col1:
    st.metric("Portfolio Net Delta (Δ)", f"{net_delta:+,.2f}", help="Directional dollar exposure to underlying movements")
with g_col2:
    st.metric("Portfolio Net Gamma (Γ)", f"{net_gamma:+,.4f}", help="Curvature / rate of change of Delta")
with g_col3:
    st.metric("Portfolio Net Vega (V)", f"${net_vega:+,.2f}", help="P&L sensitivity per 1.0% change in Implied Volatility")
with g_col4:
    st.metric("Portfolio Net Theta (Θ)", f"${net_theta:+,.2f}/day", help="Calendar time decay decay harvested or paid per day")

st.write("---")

# Tabbed Interface
tab_equity, tab_positions, tab_blotter, tab_executions, tab_signals = st.tabs([
    "📈 Equity Curve & Friction",
    "💼 Open Positions",
    "📋 Order Blotter",
    "⚡ Executions & Fills",
    "🧠 AI Signals"
])

# 1. Equity Curve Tab
with tab_equity:
    if not ledger_df.empty and len(ledger_df) > 1:
        ledger_df["gross_equity"] = ledger_df["total_equity"] + ledger_df["cumulative_friction_usd"]
        chart_data = ledger_df[["timestamp_utc", "total_equity", "gross_equity", "cumulative_friction_usd"]].copy()

        st.subheader("Gross vs. Net Equity Curve (After C++ Slippage & Fees)")
        line_chart = alt.Chart(chart_data).transform_fold(
            ["total_equity", "gross_equity"],
            as_=["Metric", "Value"]
        ).mark_line(strokeWidth=2).encode(
            x=alt.X("timestamp_utc:T", title="Timeline (UTC)"),
            y=alt.Y("Value:Q", title="Equity (USD)", scale=alt.Scale(zero=False)),
            color=alt.Color("Metric:N", scale=alt.Scale(domain=["gross_equity", "total_equity"], range=["#2ca02c", "#1f77b4"]))
        ).properties(height=380)

        st.altair_chart(line_chart, use_container_width=True)

        st.subheader("Cumulative Execution Friction (Slippage + Exchange/Clearing Fees)")
        bar_chart = alt.Chart(chart_data).mark_area(color="#f85149", opacity=0.4).encode(
            x="timestamp_utc:T",
            y=alt.Y("cumulative_friction_usd:Q", title="Total Friction (USD)")
        ).properties(height=180)
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("No ledger snapshots recorded yet. Run a backtest or start paper trading to see performance.")

# 2. Open Positions Tab
with tab_positions:
    st.subheader(f"Active Positions ({len(open_positions)} Open)")
    if not open_positions.empty:
        display_pos = open_positions[[
            "position_id", "symbol", "underlying", "asset_class", "side", "strategy_type",
            "quantity", "avg_entry_price", "current_mark_price", "unrealized_pnl", "delta", "theta"
        ]].copy()
        display_pos["avg_entry_price"] = display_pos["avg_entry_price"].map("${:,.2f}".format)
        display_pos["current_mark_price"] = display_pos["current_mark_price"].map("${:,.2f}".format)
        display_pos["unrealized_pnl"] = display_pos["unrealized_pnl"].map("${:+,.2f}".format)
        display_pos["delta"] = display_pos["delta"].map("{:+,.2f}".format)
        display_pos["theta"] = display_pos["theta"].map("{:+,.2f}".format)
        st.dataframe(display_pos, use_container_width=True)
    else:
        st.info("No open positions. Portfolio is 100% in cash.")

# 3. Order Blotter Tab
with tab_blotter:
    st.subheader("Order Blotter")
    if not orders_df.empty:
        st.dataframe(orders_df[[
            "order_id", "created_at", "asset_class", "symbol", "side", "order_type",
            "quantity", "limit_price", "stop_price", "status", "reason"
        ]], use_container_width=True)
    else:
        st.info("No orders in ledger.")

# 4. Executions & Fills Tab
with tab_executions:
    st.subheader("Execution Fills & Cost Breakdown")
    if not exec_df.empty:
        disp_exec = exec_df[[
            "fill_id", "order_id", "fill_timestamp", "symbol", "side", "quantity",
            "mid_price_at_order", "fill_price", "spread_cost_usd", "slippage_cost_usd",
            "commission_fee_usd", "total_friction_usd"
        ]].copy()
        disp_exec["mid_price_at_order"] = disp_exec["mid_price_at_order"].map("${:,.2f}".format)
        disp_exec["fill_price"] = disp_exec["fill_price"].map("${:,.2f}".format)
        disp_exec["spread_cost_usd"] = disp_exec["spread_cost_usd"].map("${:,.2f}".format)
        disp_exec["slippage_cost_usd"] = disp_exec["slippage_cost_usd"].map("${:,.2f}".format)
        disp_exec["commission_fee_usd"] = disp_exec["commission_fee_usd"].map("${:,.2f}".format)
        disp_exec["total_friction_usd"] = disp_exec["total_friction_usd"].map("${:,.2f}".format)
        st.dataframe(disp_exec, use_container_width=True)
    else:
        st.info("No executions recorded.")

# 5. Signals Tab
with tab_signals:
    st.subheader("Generated AI Signals")
    if not signals_df.empty:
        st.dataframe(signals_df[[
            "signal_id", "created_at", "underlying", "direction", "strategy_type",
            "confidence_score", "iv_rank", "entry_limit_price", "stop_loss_price", "take_profit_price", "rationale"
        ]], use_container_width=True)
    else:
        st.info("No signals recorded.")
