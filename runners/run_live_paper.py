"""
Live Paper Trading Loop Runner:
Streams live/delayed market data into the self-hosted OMS in real time,
evaluates AI multi-agent signals, tracks open positions & Greeks,
and settles options daily at market close.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import DATABASE_PATH, STARTING_EQUITY
from oms.database import init_db, get_connection
from oms.engine import OrderManagementSystem
from data.market_fetcher import MarketDataFetcher
from agents.graph import MultiAgentTradingPipeline

def run_live_paper_loop(
    symbols: List[str] = ["SPY", "QQQ", "NVDA", "GLD", "BTC/USDT"],
    poll_interval_seconds: float = 5.0,
    max_cycles: Optional[int] = None
) -> None:
    print("=" * 80)
    print("      MULTI-ASSET & OPTIONS LIVE PAPER TRADING ENGINE (SELF-HOSTED OMS)")
    print("=" * 80)
    print(f"Monitoring Universe: {symbols}")
    print(f"OMS Database: {DATABASE_PATH}")
    print(f"Polling Interval: {poll_interval_seconds}s")

    init_db(DATABASE_PATH)
    oms = OrderManagementSystem(DATABASE_PATH)
    fetcher = MarketDataFetcher()
    pipeline = MultiAgentTradingPipeline()

    cycle = 0
    try:
        while True:
            cycle += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_quotes: Dict[str, float] = {}

            # 1. Fetch latest prices & update pending orders
            for sym in symbols:
                ac = "CRYPTO" if "USDT" in sym else ("COMMODITY" if sym in ["GLD", "USO", "SLV"] else "US_EQUITY")
                df = fetcher.fetch_historical_bars(sym, asset_class=ac, period="5d", interval="1h")
                if not df.empty:
                    last_bar = df.iloc[-1]
                    price = float(last_bar["close"])
                    high = float(last_bar.get("high", price))
                    low = float(last_bar.get("low", price))
                    vol = float(last_bar.get("volume", 50000))
                    current_quotes[sym] = price

                    # Match pending spot orders
                    oms.process_market_snapshot(
                        symbol=sym,
                        current_price=price,
                        high_price=high,
                        low_price=low,
                        volume=vol,
                        timestamp_iso=now_iso
                    )

                    # Match pending option spread orders
                    for spread_type in ["BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"]:
                        oms.process_market_snapshot(
                            symbol=f"{sym}_{spread_type}",
                            current_price=price,
                            high_price=high,
                            low_price=low,
                            volume=vol,
                            timestamp_iso=now_iso
                        )

            # 2. Mark-to-market portfolio & calculate Greeks
            if current_quotes:
                state = oms.portfolio_mgr.update_mark_to_market(current_quotes, timestamp_iso=now_iso)
                oms.expire_options_sweep(current_quotes, current_time_iso=now_iso)

            # 3. Periodically run multi-agent signal pipeline (every 4 cycles)
            if cycle % 4 == 1 and current_quotes:
                acct = oms.portfolio_mgr.get_latest_account_state()
                with get_connection(DATABASE_PATH) as conn:
                    open_pos_count = conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'").fetchone()[0]

                for sym in symbols:
                    ac = "US_OPTION" if sym in ["SPY", "QQQ", "GLD", "NVDA"] else ("CRYPTO" if "USDT" in sym else "US_EQUITY")
                    df = fetcher.fetch_historical_bars(sym, asset_class="CRYPTO" if "USDT" in sym else "US_EQUITY")
                    
                    sig = pipeline.generate_signal(
                        symbol=sym,
                        asset_class=ac,
                        df_bars=df,
                        account_equity=acct["total_equity"],
                        current_open_positions=open_pos_count
                    )

                    if sig:
                        print(f"[{now_time_str}] 🎯 Signal Generated: {sig['strategy_type']} on {sym} | Confidence: {sig['confidence_score']:.2f}")
                        order_id = oms.submit_order(
                            symbol=sig["symbol"],
                            asset_class=sig["asset_class"],
                            side="BUY",
                            order_type="LIMIT",
                            quantity=sig.get("recommended_quantity", 1),
                            limit_price=sig.get("entry_limit_price"),
                            stop_price=sig.get("stop_loss_price"),
                            legs=sig.get("contracts", []),
                            signal_id=sig["signal_id"],
                            timestamp_iso=now_iso
                        )
                        print(f"[{now_time_str}] 📨 Submitted Order: {order_id} ({sig['strategy_type']} {sig.get('recommended_quantity', 1)} units @ ${sig.get('entry_limit_price'):.2f})")

            # 4. Status heartbeat printout
            if current_quotes and cycle % 2 == 0:
                print(f"[{now_time_str}] NLV: ${state['total_equity']:,.2f} | Cash: ${state['cash_balance']:,.2f} | Unrealized PnL: ${state['unrealized_pnl']:+,.2f} | Net Delta: {state['net_delta']:.1f} | Net Theta: {state['net_theta']:.2f}/d")

            if max_cycles and cycle >= max_cycles:
                print("Completed requested execution cycles. Exiting.")
                break

            time.sleep(poll_interval_seconds)

    except KeyboardInterrupt:
        print("\nShutdown requested by user. OMS state safely preserved in SQLite.")

if __name__ == "__main__":
    run_live_paper_loop(poll_interval_seconds=3.0)
