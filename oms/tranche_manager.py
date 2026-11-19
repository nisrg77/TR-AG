"""
Tranche Manager: Core (70%) + Runner (30%) Scaling & Dynamic Trailing Engine:
- Automatically decomposes filled orders into Core & Runner child tranches.
- Core: Takes hard profit at +1.5R, immediately ratcheting Runner SL to Entry + Fees (Auto-Breakeven).
- Runner: Trails with Chandelier ATR stop (Highest High - 2.0 * ATR) or DTE time decay limits.
"""

import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config.settings import DATABASE_PATH
from oms.database import get_connection

class TrancheManager:
    def __init__(self, db_path: Path = DATABASE_PATH, friction_per_unit_default: float = 0.05):
        self.db_path = db_path
        self.friction_per_unit_default = friction_per_unit_default

    def create_tranches_for_fill(
        self,
        parent_position_id: str,
        symbol: str,
        total_quantity: float,
        entry_price: float,
        stop_loss_price: float,
        atr: float = 1.0,
        r_multiple: float = 1.5,
        trailing_atr_mult: float = 2.0,
        pyramid_tier: int = 1,
        friction_per_unit: Optional[float] = None,
        conn: Optional[sqlite3.Connection] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Splits filled quantity into 70% Core and 30% Runner tranches.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 1e-4: risk_per_unit = entry_price * 0.02
        target_profit_price = entry_price + (risk_per_unit * r_multiple)

        core_qty = round(total_quantity * 0.70, 4)
        runner_qty = round(total_quantity - core_qty, 4)
        if total_quantity >= 1.0 and runner_qty < 1.0 and total_quantity % 1 == 0:
            core_qty = max(1.0, float(int(total_quantity * 0.70)))
            runner_qty = max(1.0, total_quantity - core_qty)

        core_id = f"TR-CORE-{uuid.uuid4().hex[:6].upper()}"
        runner_id = f"TR-RUN-{uuid.uuid4().hex[:6].upper()}"

        def _do_insert(cur_conn):
            cur = cur_conn.cursor()
            cur.execute("""
            INSERT INTO position_tranches (
                tranche_id, parent_position_id, symbol, pyramid_tier, role,
                quantity, entry_price, stop_loss, take_profit, trailing_atr_mult,
                highest_high, breakeven_armed, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'CORE', ?, ?, ?, ?, ?, ?, 0, 'OPEN', ?, ?)
            """, (
                core_id, parent_position_id, symbol, pyramid_tier, core_qty,
                entry_price, stop_loss_price, target_profit_price, trailing_atr_mult,
                entry_price, now_iso, now_iso
            ))

            cur.execute("""
            INSERT INTO position_tranches (
                tranche_id, parent_position_id, symbol, pyramid_tier, role,
                quantity, entry_price, stop_loss, take_profit, trailing_atr_mult,
                highest_high, breakeven_armed, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'RUNNER', ?, ?, ?, NULL, ?, ?, 0, 'OPEN', ?, ?)
            """, (
                runner_id, parent_position_id, symbol, pyramid_tier, runner_qty,
                entry_price, stop_loss_price, trailing_atr_mult,
                entry_price, now_iso, now_iso
            ))

        if conn is not None:
            _do_insert(conn)
        else:
            with get_connection(self.db_path) as local_c:
                _do_insert(local_c)
                local_c.commit()

        return {
            "tranche_id": core_id, "role": "CORE", "quantity": core_qty,
            "take_profit": target_profit_price, "stop_loss": stop_loss_price
        }, {
            "tranche_id": runner_id, "role": "RUNNER", "quantity": runner_qty,
            "stop_loss": stop_loss_price
        }

    def process_tranche_ticks(
        self,
        symbol: str,
        current_price: float,
        high_price: float,
        low_price: float,
        atr: float,
        timestamp_iso: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates open tranches on each bar/tick:
        1. Core Take-Profit (+1.5R) -> Auto-arms Runner Breakeven Stop
        2. Runner Chandelier Trailing Stop update (Highest High - 2.0 * ATR)
        3. Exits triggered tranches.
        """
        if not timestamp_iso:
            timestamp_iso = datetime.now(timezone.utc).isoformat()

        events = []

        def _do_process(c):
            cur = c.cursor()
            cur.execute("""
            SELECT * FROM position_tranches
            WHERE symbol = ? AND status = 'OPEN'
            """, (symbol,))
            tranches = cur.fetchall()

            for tr in tranches:
                t_id = tr["tranche_id"]
                parent_id = tr["parent_position_id"]
                role = tr["role"]
                qty = tr["quantity"]
                entry_px = tr["entry_price"]
                sl = tr["stop_loss"]
                tp = tr["take_profit"]
                h_high = max(tr["highest_high"] or entry_px, high_price)
                trail_mult = tr["trailing_atr_mult"] or 2.0
                be_armed = tr["breakeven_armed"]

                # 1. CORE TRANCHE EVALUATION
                if role == "CORE":
                    # Check Core Take Profit (+1.5R)
                    if tp is not None and high_price >= tp:
                        realized = (tp - entry_px) * qty
                        cur.execute("""
                        UPDATE position_tranches SET
                            status = 'CLOSED', realized_pnl = ?, updated_at = ?
                        WHERE tranche_id = ?
                        """, (realized, timestamp_iso, t_id))

                        # Auto-Breakeven Trigger on sibling RUNNER tranche!
                        cur.execute("""
                        SELECT tranche_id, entry_price FROM position_tranches
                        WHERE parent_position_id = ? AND role = 'RUNNER' AND status = 'OPEN'
                        """, (parent_id,))
                        runner_row = cur.fetchone()
                        if runner_row:
                            r_id = runner_row["tranche_id"]
                            be_stop = entry_px + self.friction_per_unit_default
                            cur.execute("""
                            UPDATE position_tranches SET
                                stop_loss = MAX(stop_loss, ?),
                                breakeven_armed = 1,
                                updated_at = ?
                            WHERE tranche_id = ?
                            """, (be_stop, timestamp_iso, r_id))

                        events.append({
                            "event": "CORE_TAKE_PROFIT_HIT",
                            "tranche_id": t_id,
                            "symbol": symbol,
                            "exit_price": tp,
                            "realized_pnl": realized,
                            "auto_breakeven_armed": True
                        })

                    # Check Core Stop Loss
                    elif sl is not None and low_price <= sl:
                        loss = (sl - entry_px) * qty
                        cur.execute("""
                        UPDATE position_tranches SET
                            status = 'CLOSED', realized_pnl = ?, updated_at = ?
                        WHERE tranche_id = ?
                        """, (loss, timestamp_iso, t_id))
                        events.append({
                            "event": "CORE_STOP_LOSS_HIT",
                            "tranche_id": t_id,
                            "symbol": symbol,
                            "exit_price": sl,
                            "realized_pnl": loss
                        })

                # 2. RUNNER TRANCHE EVALUATION
                elif role == "RUNNER":
                    # Update Chandelier Trailing Stop (Highest High - 2.0 * ATR)
                    new_trail_stop = h_high - (trail_mult * atr)
                    updated_stop = max(sl, new_trail_stop)

                    cur.execute("""
                    UPDATE position_tranches SET
                        highest_high = ?, stop_loss = ?, updated_at = ?
                    WHERE tranche_id = ?
                    """, (h_high, updated_stop, timestamp_iso, t_id))

                    # Check if Runner is stopped out by Chandelier stop
                    if low_price <= updated_stop:
                        runner_realized = (updated_stop - entry_px) * qty
                        cur.execute("""
                        UPDATE position_tranches SET
                            status = 'CLOSED', realized_pnl = ?, updated_at = ?
                        WHERE tranche_id = ?
                        """, (runner_realized, timestamp_iso, t_id))
                        events.append({
                            "event": "RUNNER_CHANDELIER_STOP_HIT",
                            "tranche_id": t_id,
                            "symbol": symbol,
                            "exit_price": updated_stop,
                            "realized_pnl": runner_realized
                        })

        if conn is not None:
            _do_process(conn)
        else:
            with get_connection(self.db_path) as local_conn:
                _do_process(local_conn)
                local_conn.commit()

        return events

        return events
