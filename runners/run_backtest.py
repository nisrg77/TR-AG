"""
Historical Backtest Replay Runner:
Replays historical multi-asset data through the self-hosted OMS,
strictly enforcing zero lookahead bias and realistic C++ execution friction.
Compares Gross vs. Net PnL (after slippage & fees), Sharpe, Drawdown, and Win Rate.
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from config.settings import DATABASE_PATH, STARTING_EQUITY
from oms.database import reset_db, get_connection
from oms.engine import OrderManagementSystem
from data.market_fetcher import MarketDataFetcher
from agents.graph import MultiAgentTradingPipeline
from risk.dynamic_leverage import DynamicLeverageEngine
from risk.portfolio_guard import PortfolioRiskGuard

DEFAULT_BACKTEST_UNIVERSE = [
    "GLD", "SPY", "QQQ", "AAPL", "NVDA", "GOOGL", "BTC/USDT", "ETH/USDT", "SOL/USDT"
]

def run_backtest_simulation(
    symbols: List[str] = DEFAULT_BACKTEST_UNIVERSE,
    start_equity: float = STARTING_EQUITY,
    bars_count: int = 400
) -> Dict[str, Any]:
    print("=" * 85)
    print("      MULTI-ASSET & OPTIONS QUANT BACKTEST REPLAY (SELF-HOSTED OMS)")
    print("=" * 85)
    print(f"Initial Starting Capital: ${start_equity:,.2f}")
    print(f"Target Universe ({len(symbols)} Assets): {symbols}")
    print(f"Timeframe: 1-Hour Intraday Base Bars (aligned with 4-Hour Macro Trend Regime)")

    # 1. Reset local OMS database
    reset_db(DATABASE_PATH, initial_equity=start_equity)
    oms = OrderManagementSystem(DATABASE_PATH)
    fetcher = MarketDataFetcher()
    pipeline = MultiAgentTradingPipeline()
    dyn_lev = DynamicLeverageEngine()
    portfolio_guard = PortfolioRiskGuard(DATABASE_PATH)

    # 2. Load historical series for all symbols
    market_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ac = "CRYPTO" if "USDT" in sym else ("COMMODITY" if sym in ["GLD", "USO", "SLV"] else "US_EQUITY")
        df = fetcher.fetch_historical_bars(sym, asset_class=ac)
        if len(df) > bars_count:
            df = df.iloc[-bars_count:]
        market_data[sym] = df

    # Find common timestamp index
    ref_sym = symbols[0]
    time_index = market_data[ref_sym].index

    print(f"Replaying {len(time_index)} chronological market periods through OMS...")

    # Step through time chronologically
    for step_i in range(30, len(time_index)):
        current_time = time_index[step_i]
        time_iso = current_time.isoformat()

        # Build current quote snapshot
        current_quotes = {}
        for sym in symbols:
            df = market_data[sym]
            if step_i < len(df):
                row = df.iloc[step_i]
                current_quotes[sym] = float(row["close"])
                ac = "CRYPTO" if "USDT" in sym else ("COMMODITY" if sym in ["GLD", "USO", "SLV"] else "US_EQUITY")

                # Step OMS order matching for this symbol
                atr_val = float(row.get("high", row["close"]) - row.get("low", row["close"]))
                oms.process_market_snapshot(
                    symbol=sym,
                    current_price=float(row["close"]),
                    high_price=float(row.get("high", row["close"])),
                    low_price=float(row.get("low", row["close"])),
                    volume=float(row.get("volume", 100000)),
                    atr=max(0.1, atr_val),
                    timestamp_iso=time_iso
                )

                # Process option spreads underlyings
                oms.process_market_snapshot(
                    symbol=f"{sym}_BULL_CALL_SPREAD",
                    current_price=float(row["close"]),
                    timestamp_iso=time_iso
                )
                oms.process_market_snapshot(
                    symbol=f"{sym}_BEAR_PUT_SPREAD",
                    current_price=float(row["close"]),
                    timestamp_iso=time_iso
                )

        # Periodically evaluate signals (e.g., every 8 bars / ~1 day)
        if step_i % 8 == 0:
            acct = oms.portfolio_mgr.get_latest_account_state()
            with get_connection(DATABASE_PATH) as conn:
                open_pos_count = conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'").fetchone()[0]

            for sym in symbols:
                df = market_data[sym]
                historical_slice = df.iloc[:step_i+1] # Strictly zero lookahead!
                ac = "US_OPTION" if sym in ["SPY", "QQQ", "GLD", "NVDA"] else ("CRYPTO" if "USDT" in sym else "US_EQUITY")

                signal = pipeline.generate_signal(
                    symbol=sym,
                    asset_class=ac,
                    df_bars=historical_slice,
                    account_equity=acct["total_equity"],
                    current_open_positions=open_pos_count
                )

                if signal:
                    # Submit order to OMS
                    trade_symbol = signal["symbol"]
                    qty = signal.get("recommended_quantity", 1)
                    limit_px = signal.get("entry_limit_price")
                    legs = signal.get("contracts", [])

                    order_id = oms.submit_order(
                        symbol=trade_symbol,
                        asset_class=signal["asset_class"],
                        side="BUY",
                        order_type="LIMIT",
                        quantity=qty,
                        limit_price=limit_px,
                        stop_price=signal.get("stop_loss_price"),
                        legs=legs,
                        signal_id=signal["signal_id"],
                        timestamp_iso=time_iso
                    )

        # Mark-to-market daily
        if step_i % 6 == 0:
            oms.portfolio_mgr.update_mark_to_market(current_quotes, timestamp_iso=time_iso)
            oms.expire_options_sweep(current_quotes, current_time_iso=time_iso)

    # Final Mark-to-Market
    final_stats = oms.portfolio_mgr.update_mark_to_market(current_quotes, timestamp_iso=time_iso)

    # Query results from database
    with get_connection(DATABASE_PATH) as conn:
        ledger_df = pd.read_sql("SELECT * FROM account_ledger ORDER BY ledger_id ASC", conn)
        exec_df = pd.read_sql("SELECT * FROM executions", conn)
        pos_df = pd.read_sql("SELECT * FROM positions", conn)

    total_trades = len(exec_df)
    total_friction = ledger_df["cumulative_friction_usd"].iloc[-1] if not ledger_df.empty else 0.0
    final_equity = final_stats["total_equity"]
    net_pnl = final_equity - start_equity
    gross_pnl = net_pnl + total_friction

    # Performance metrics
    equity_curve = ledger_df["total_equity"].values
    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0.0])
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 6.5)) if len(returns) > 5 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    max_dd = float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    closed_pos = pos_df[pos_df["status"].isin(["CLOSED", "EXPIRED"])]
    winning_trades = len(closed_pos[closed_pos["realized_pnl"] > 0])
    win_rate = (winning_trades / len(closed_pos) * 100.0) if len(closed_pos) > 0 else 0.0

    print("\n" + "=" * 85)
    print("                    PER-ASSET BREAKDOWN & PnL ATTRIBUTION")
    print("=" * 85)
    print(f"{'Asset / Symbol':<18} | {'Asset Class':<12} | {'Fills':<6} | {'Realized PnL':<14} | {'Unrealized PnL':<14} | {'Status'}")
    print("-" * 85)
    for sym in symbols:
        underlying = sym.split("/")[0]
        sym_pos = pos_df[pos_df["underlying"].str.contains(underlying)] if not pos_df.empty else pd.DataFrame()
        sym_exec = exec_df[exec_df["symbol"].str.contains(underlying)] if not exec_df.empty else pd.DataFrame()
        
        realized = float(sym_pos["realized_pnl"].sum()) if not sym_pos.empty else 0.0
        unrealized = float(sym_pos["unrealized_pnl"].sum()) if not sym_pos.empty else 0.0
        fill_count = len(sym_exec)
        ac_name = "CRYPTO" if "USDT" in sym else ("COMMODITY" if sym in ["GLD", "USO", "SLV"] else "US_EQUITY")
        stat_str = "ACTIVE" if (not sym_pos.empty and (sym_pos["status"] == "OPEN").any()) else "FLAT"

        print(f"{sym:<18} | {ac_name:<12} | {fill_count:<6} | ${realized:>+12,.2f} | ${unrealized:>+12,.2f} | {stat_str}")

    print("=" * 85)

    print("\n" + "=" * 85)
    print("                    PORTFOLIO BACKTEST PERFORMANCE REPORT")
    print("=" * 85)
    print(f"  Starting Balance:             ${start_equity:,.2f}")
    print(f"  Ending Net Liquidation Value: ${final_equity:,.2f}")
    print(f"  Gross Realized PnL:           ${gross_pnl:+,.2f} ({gross_pnl/start_equity*100:+.2f}%)")
    print(f"  Total Slippage & Fees Paid:   ${total_friction:,.2f}")
    print(f"  Net Realized PnL:             ${net_pnl:+,.2f} ({net_pnl/start_equity*100:+.2f}%)")
    print(f"  Total Executed Orders:        {total_trades}")
    print(f"  Win Rate (Closed Trades):     {win_rate:.1f}%")
    print(f"  Max Account Drawdown:         {max_dd:.2f}%")
    print(f"  Annualized Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Portfolio Net Delta:          {final_stats['net_delta']:.2f}")
    print(f"  Portfolio Net Theta:          {final_stats['net_theta']:.2f}/day")
    print("=" * 85 + "\n")

    return {
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "total_friction": total_friction,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "sharpe": sharpe
    }

if __name__ == "__main__":
    run_backtest_simulation()
