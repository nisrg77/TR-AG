"""
Portfolio Ledger, Margin & Mark-to-Market Engine
Computes Net Liquidation Value (NLV), buying power, margin reservation, and Greeks.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from oms.database import get_connection
from config.settings import STARTING_EQUITY
from src.cpp.quant_bridge import aggregate_portfolio

class PortfolioManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_latest_account_state(self) -> Dict[str, Any]:
        """Fetches the most recent entry from account_ledger."""
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT cash_balance, reserved_margin, total_equity, realized_pnl, unrealized_pnl, cumulative_friction_usd, timestamp_utc
            FROM account_ledger
            ORDER BY ledger_id DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    "cash_balance": row["cash_balance"],
                    "reserved_margin": row["reserved_margin"],
                    "total_equity": row["total_equity"],
                    "realized_pnl": row["realized_pnl"],
                    "unrealized_pnl": row["unrealized_pnl"],
                    "cumulative_friction_usd": row["cumulative_friction_usd"],
                    "timestamp_utc": row["timestamp_utc"]
                }
            return {
                "cash_balance": 100000.0,
                "reserved_margin": 0.0,
                "total_equity": 100000.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "cumulative_friction_usd": 0.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }

    def update_mark_to_market(self, market_quotes: Dict[str, float], timestamp_iso: Optional[str] = None) -> Dict[str, Any]:
        """
        Updates unrealized PnL and Greeks across all open positions.
        market_quotes: dict of symbol/underlying -> current price
        """
        if not timestamp_iso:
            timestamp_iso = datetime.now(timezone.utc).isoformat()

        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions WHERE status = 'OPEN'")
            open_positions = cur.fetchall()

            total_unrealized_pnl = 0.0
            total_reserved_margin = 0.0
            portfolio_delta = 0.0
            portfolio_gamma = 0.0
            portfolio_vega = 0.0
            portfolio_theta = 0.0

            for pos in open_positions:
                pos_id = pos["position_id"]
                symbol = pos["symbol"]
                underlying = pos["underlying"]
                asset_class = pos["asset_class"]
                side = pos["side"]
                qty = pos["quantity"]
                entry_px = pos["avg_entry_price"]
                legs_json = pos["legs_json"]

                # Current mark price
                mark_px = market_quotes.get(symbol, market_quotes.get(underlying, entry_px))

                # Multiplier
                multiplier = 100.0 if asset_class == "US_OPTION" else 1.0

                if asset_class == "US_OPTION" and legs_json:
                    from oms.options_lifecycle import OptionsLifecycleEngine
                    opt_engine = OptionsLifecycleEngine()
                    contracts = json.loads(legs_json)
                    res = opt_engine.price_contract_or_spread(mark_px, contracts, current_time_iso=timestamp_iso)
                    current_spread_px = res["spread_price"]
                    unrealized_pnl = (current_spread_px - entry_px) * qty * multiplier
                    
                    pos_delta = res["net_delta"] * qty * multiplier
                    pos_gamma = res["net_gamma"] * qty * multiplier
                    pos_vega = res["net_vega"] * qty * multiplier
                    pos_theta = res["net_theta"] * qty * multiplier

                    # Margin for option: max loss = premium paid for debit spread / long
                    reserved_margin = max(0.0, entry_px * qty * multiplier)

                    cur.execute("""
                    UPDATE positions SET
                        current_mark_price = ?,
                        unrealized_pnl = ?,
                        delta = ?, gamma = ?, vega = ?, theta = ?
                    WHERE position_id = ?
                    """, (current_spread_px, unrealized_pnl, pos_delta, pos_gamma, pos_vega, pos_theta, pos_id))

                else:
                    # Equity, Crypto, Commodity
                    sign = 1.0 if side == "LONG" else -1.0
                    unrealized_pnl = sign * (mark_px - entry_px) * qty
                    pos_delta = sign * qty * mark_px
                    pos_gamma = 0.0
                    pos_vega = 0.0
                    pos_theta = 0.0
                    
                    # Margin (50% for equity, 100% for crypto/cash)
                    margin_pct = 0.50 if asset_class == "US_EQUITY" else 1.0
                    reserved_margin = mark_px * qty * margin_pct

                    cur.execute("""
                    UPDATE positions SET
                        current_mark_price = ?,
                        unrealized_pnl = ?,
                        delta = ?
                    WHERE position_id = ?
                    """, (mark_px, unrealized_pnl, pos_delta, pos_id))

                total_unrealized_pnl += unrealized_pnl
                total_reserved_margin += reserved_margin
                portfolio_delta += pos_delta
                portfolio_gamma += pos_gamma
                portfolio_vega += pos_vega
                portfolio_theta += pos_theta

            # Retrieve starting equity and cash
            state = self.get_latest_account_state()
            cash = state["cash_balance"]
            realized_pnl = state["realized_pnl"]
            friction = state["cumulative_friction_usd"]

            # Institutional Net Liquidation Value (NLV):
            # Total Equity = Initial Genesis Cash + Cumulative Realized PnL + Unrealized PnL - Cumulative Friction
            total_equity = STARTING_EQUITY + realized_pnl + total_unrealized_pnl - friction

            # Record ledger snapshot
            cur.execute("""
            INSERT INTO account_ledger (
                timestamp_utc, cash_balance, reserved_margin, total_equity,
                realized_pnl, unrealized_pnl, cumulative_friction_usd, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp_iso, cash, total_reserved_margin, total_equity, realized_pnl, total_unrealized_pnl, friction, "Mark-to-Market Snapshot"))

            conn.commit()

            return {
                "total_equity": round(total_equity, 2),
                "cash_balance": round(cash, 2),
                "reserved_margin": round(total_reserved_margin, 2),
                "buying_power": round(max(0.0, cash - total_reserved_margin), 2),
                "unrealized_pnl": round(total_unrealized_pnl, 2),
                "realized_pnl": round(realized_pnl, 2),
                "cumulative_friction_usd": round(friction, 2),
                "net_delta": round(portfolio_delta, 2),
                "net_gamma": round(portfolio_gamma, 4),
                "net_vega": round(portfolio_vega, 2),
                "net_theta": round(portfolio_theta, 2)
            }
