"""
Self-Hosted OMS Core Engine:
Handles order routing, realistic queue-aware limit fills, gap-aware stop fills,
state machine transitions, multi-leg options execution, and ledger accounting.
"""

import uuid
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from oms.database import get_connection, init_db
from oms.execution_costs import ExecutionCostEngine
from oms.options_lifecycle import OptionsLifecycleEngine
from oms.portfolio import PortfolioManager
from oms.tranche_manager import TrancheManager
from oms.book_simulator import OrderBookSimulator
from config.settings import STARTING_EQUITY

class OrderManagementSystem:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        init_db(self.db_path)
        self.cost_engine = ExecutionCostEngine()
        self.opt_engine = OptionsLifecycleEngine()
        self.portfolio_mgr = PortfolioManager(self.db_path)
        self.tranche_mgr = TrancheManager(self.db_path)
        self.book_sim = OrderBookSimulator()

    def submit_order(
        self,
        symbol: str,
        asset_class: str,
        side: str,              # BUY, SELL
        order_type: str,        # MARKET, LIMIT, STOP
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        legs: Optional[List[Dict[str, Any]]] = None,
        signal_id: Optional[str] = None,
        timestamp_iso: Optional[str] = None
    ) -> str:
        """
        Creates an order with PENDING status in the OMS.
        Returns order_id.
        """
        if not timestamp_iso:
            timestamp_iso = datetime.now(timezone.utc).isoformat()

        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        legs_json = json.dumps(legs) if legs else None

        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO orders (
                order_id, signal_id, timestamp_utc, asset_class, symbol,
                side, order_type, quantity, limit_price, stop_price,
                legs_json, status, reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id, signal_id, timestamp_iso, asset_class, symbol,
                side.upper(), order_type.upper(), quantity, limit_price, stop_price,
                legs_json, "PENDING", "Order submitted", timestamp_iso, timestamp_iso
            ))
            conn.commit()

        return order_id

    def process_market_snapshot(
        self,
        symbol: str,
        current_price: float,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        volume: float = 100000.0,
        open_interest: float = 1000.0,
        atr: float = 1.0,
        timestamp_iso: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes pending orders against the latest market bar/tick:
        1. Checks queue-aware limit fills.
        2. Checks gap-aware stop fills.
        3. Executes market orders immediately.
        4. Checks stop-loss / take-profit for existing open positions.
        """
        if not timestamp_iso:
            timestamp_iso = datetime.now(timezone.utc).isoformat()
        if high_price is None: high_price = current_price
        if low_price is None: low_price = current_price
        if bid_price is None: bid_price = current_price * 0.9995
        if ask_price is None: ask_price = current_price * 1.0005

        executed_fills = []

        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT * FROM orders
            WHERE symbol = ? AND status = 'PENDING'
            """, (symbol,))
            pending_orders = cur.fetchall()

            for order in pending_orders:
                order_id = order["order_id"]
                side = order["side"]
                order_type = order["order_type"]
                asset_class = order["asset_class"]
                qty = order["quantity"]
                limit_px = order["limit_price"]
                stop_px = order["stop_price"]
                legs_json = order["legs_json"]
                legs = json.loads(legs_json) if legs_json else []
                num_legs = len(legs) if legs else 1

                fill_triggered = False
                effective_mid = current_price

                # For multi-leg options, compute theoretical spread mid
                if asset_class == "US_OPTION" and legs:
                    res = self.opt_engine.price_contract_or_spread(current_price, legs, timestamp_iso)
                    effective_mid = max(0.01, res["spread_price"])
                    bid_price = effective_mid * 0.95
                    ask_price = effective_mid * 1.05

                # 1. MARKET ORDER
                if order_type == "MARKET":
                    fill_triggered = True

                # 2. LIMIT ORDER (Realistic Queue-Priority Check)
                elif order_type == "LIMIT" and limit_px is not None:
                    if side == "BUY":
                        # Limit buy fills only if Ask <= Limit, or Low penetrated strictly below Limit by 1 tick
                        tick = 0.01 if current_price < 500 else 0.05
                        if ask_price <= limit_px or low_price < (limit_px - tick):
                            fill_triggered = True
                    elif side == "SELL":
                        # Limit sell fills only if Bid >= Limit, or High penetrated strictly above Limit by 1 tick
                        tick = 0.01 if current_price < 500 else 0.05
                        if bid_price >= limit_px or high_price > (limit_px + tick):
                            fill_triggered = True

                # 3. STOP ORDER (Realistic Gap Handling)
                elif order_type == "STOP" and stop_px is not None:
                    if side == "SELL" and low_price <= stop_px:
                        # Price gapped or fell through stop price: fill at worse of current_price or stop_px
                        fill_triggered = True
                        effective_mid = min(current_price, stop_px)
                    elif side == "BUY" and high_price >= stop_px:
                        fill_triggered = True
                        effective_mid = max(current_price, stop_px)

                if fill_triggered:
                    # Calculate realistic fill price & costs
                    friction = self.cost_engine.calculate_fill_and_costs(
                        asset_class=asset_class,
                        side=side,
                        order_type=order_type,
                        mid_price=effective_mid,
                        bid_price=bid_price,
                        ask_price=ask_price,
                        atr=atr,
                        volume=volume,
                        open_interest=open_interest,
                        num_legs=num_legs,
                        quantity=qty
                    )

                    fill_id = f"FILL-{uuid.uuid4().hex[:8].upper()}"
                    fill_px = friction["fill_price"]
                    multiplier = 100.0 if asset_class == "US_OPTION" else 1.0
                    trade_notional = fill_px * qty * multiplier
                    total_friction = friction["total_friction_usd"]

                    # Log execution fill
                    cur.execute("""
                    INSERT INTO executions (
                        fill_id, order_id, fill_timestamp, symbol, asset_class,
                        side, quantity, mid_price_at_order, fill_price,
                        spread_cost_usd, slippage_cost_usd, commission_fee_usd,
                        total_friction_usd, legs_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fill_id, order_id, timestamp_iso, symbol, asset_class,
                        side, qty, friction["mid_price_at_order"], fill_px,
                        friction["spread_cost_usd"], friction["slippage_cost_usd"],
                        friction["commission_fee_usd"], total_friction, legs_json
                    ))

                    # Update order status to FILLED
                    cur.execute("""
                    UPDATE orders SET status = 'FILLED', updated_at = ? WHERE order_id = ?
                    """, (timestamp_iso, order_id))

                    # Update or create Position
                    self._update_position_on_fill(
                        conn=conn,
                        symbol=symbol,
                        asset_class=asset_class,
                        side=side,
                        quantity=qty,
                        fill_price=fill_px,
                        legs_json=legs_json,
                        timestamp_iso=timestamp_iso,
                        stop_loss=stop_px
                    )

                    # Automatically partition into Core (70%) + Runner (30%) child tranches
                    if side == "BUY":
                        cur.execute("SELECT position_id FROM positions WHERE symbol = ? AND status = 'OPEN'", (symbol,))
                        pos_row = cur.fetchone()
                        if pos_row:
                            self.tranche_mgr.create_tranches_for_fill(
                                parent_position_id=pos_row["position_id"],
                                symbol=symbol,
                                total_quantity=qty,
                                entry_price=fill_px,
                                stop_loss_price=stop_px if stop_px is not None else fill_px * 0.98,
                                atr=atr,
                                conn=conn
                            )

                    # Update Cash & Ledger
                    self._update_ledger_on_fill(
                        conn=conn,
                        side=side,
                        trade_notional=trade_notional,
                        total_friction=total_friction,
                        timestamp_iso=timestamp_iso,
                        note=f"Fill {fill_id} for {side} {qty} {symbol} @ {fill_px:.2f}"
                    )

                    executed_fills.append({
                        "fill_id": fill_id,
                        "order_id": order_id,
                        "symbol": symbol,
                        "side": side,
                        "quantity": qty,
                        "fill_price": fill_px,
                        "total_friction_usd": total_friction
                    })

            # 4. Check Open Positions for Stop-Loss & Take-Profit triggers
            self._check_position_stops_and_targets(
                conn=conn,
                symbol=symbol,
                current_price=current_price,
                high_price=high_price,
                low_price=low_price,
                atr=atr,
                timestamp_iso=timestamp_iso
            )

            # 5. Process Tranche Lifecycle (Core Take-Profit, Auto-Breakeven & Runner Trailing)
            self.tranche_mgr.process_tranche_ticks(
                symbol=symbol,
                current_price=current_price,
                high_price=high_price,
                low_price=low_price,
                atr=atr,
                timestamp_iso=timestamp_iso,
                conn=conn
            )

            conn.commit()

        return executed_fills

    def _update_position_on_fill(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: float,
        fill_price: float,
        legs_json: Optional[str],
        timestamp_iso: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        cur = conn.cursor()
        cur.execute("SELECT * FROM positions WHERE symbol = ? AND status = 'OPEN'", (symbol,))
        pos = cur.fetchone()

        if side == "BUY":
            if not pos:
                # Open new LONG position
                pos_id = f"POS-{uuid.uuid4().hex[:8].upper()}"
                cur.execute("""
                INSERT INTO positions (
                    position_id, symbol, underlying, asset_class, side, strategy_type,
                    quantity, avg_entry_price, current_mark_price, unrealized_pnl, realized_pnl,
                    stop_loss, take_profit, legs_json, status, opened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, ?, 'OPEN', ?)
                """, (
                    pos_id, symbol, symbol.split("_")[0], asset_class, "LONG",
                    "OPTION_SPREAD" if legs_json else "SPOT_LONG",
                    quantity, fill_price, fill_price, stop_loss, take_profit, legs_json, timestamp_iso
                ))
            else:
                # Add to existing position (dollar cost average entry)
                old_qty = pos["quantity"]
                old_entry = pos["avg_entry_price"]
                new_qty = old_qty + quantity
                new_entry = (old_entry * old_qty + fill_price * quantity) / new_qty
                cur.execute("""
                UPDATE positions SET
                    quantity = ?, avg_entry_price = ?, current_mark_price = ?
                WHERE position_id = ?
                """, (new_qty, new_entry, fill_price, pos["position_id"]))

        elif side == "SELL":
            if pos and pos["side"] == "LONG":
                # Close or scale out
                multiplier = 100.0 if asset_class == "US_OPTION" else 1.0
                close_qty = min(quantity, pos["quantity"])
                realized_pnl = (fill_price - pos["avg_entry_price"]) * close_qty * multiplier
                remaining_qty = pos["quantity"] - close_qty

                if remaining_qty <= 1e-5:
                    cur.execute("""
                    UPDATE positions SET
                        status = 'CLOSED', quantity = 0.0, realized_pnl = realized_pnl + ?,
                        unrealized_pnl = 0.0, closed_at = ?
                    WHERE position_id = ?
                    """, (realized_pnl, timestamp_iso, pos["position_id"]))
                else:
                    cur.execute("""
                    UPDATE positions SET
                        quantity = ?, realized_pnl = realized_pnl + ?
                    WHERE position_id = ?
                    """, (remaining_qty, realized_pnl, pos["position_id"]))

    def _update_ledger_on_fill(
        self,
        conn: sqlite3.Connection,
        side: str,
        trade_notional: float,
        total_friction: float,
        timestamp_iso: str,
        note: str
    ) -> None:
        cur = conn.cursor()
        cur.execute("""
        SELECT cash_balance, reserved_margin, total_equity, realized_pnl, unrealized_pnl, cumulative_friction_usd
        FROM account_ledger ORDER BY ledger_id DESC LIMIT 1
        """)
        row = cur.fetchone()
        cash = row["cash_balance"]
        realized_pnl = row["realized_pnl"]
        unrealized_pnl = row["unrealized_pnl"]
        friction = row["cumulative_friction_usd"] + total_friction

        if side == "BUY":
            cash -= (trade_notional + total_friction)
        elif side == "SELL":
            cash += (trade_notional - total_friction)

        cur.execute("SELECT COALESCE(SUM(realized_pnl), 0.0) FROM positions")
        realized_pnl = float(cur.fetchone()[0])
        cur.execute("SELECT COALESCE(SUM(unrealized_pnl), 0.0) FROM positions WHERE status = 'OPEN'")
        unrealized_pnl = float(cur.fetchone()[0])

        total_equity = STARTING_EQUITY + realized_pnl + unrealized_pnl - friction

        cur.execute("""
        INSERT INTO account_ledger (
            timestamp_utc, cash_balance, reserved_margin, total_equity,
            realized_pnl, unrealized_pnl, cumulative_friction_usd, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp_iso, cash, row["reserved_margin"], total_equity, realized_pnl, unrealized_pnl, friction, note))

    def _check_position_stops_and_targets(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        current_price: float,
        high_price: float,
        low_price: float,
        atr: float,
        timestamp_iso: str
    ) -> None:
        cur = conn.cursor()
        cur.execute("SELECT * FROM positions WHERE symbol = ? AND status = 'OPEN'", (symbol,))
        positions = cur.fetchall()

        for pos in positions:
            sl = pos["stop_loss"]
            tp = pos["take_profit"]
            qty = pos["quantity"]
            side = pos["side"]
            asset_class = pos["asset_class"]
            legs_json = pos["legs_json"]
            entry_px = pos["avg_entry_price"]

            eval_price = current_price
            if asset_class == "US_OPTION" and legs_json:
                legs = json.loads(legs_json)
                res = self.opt_engine.price_contract_or_spread(current_price, legs, timestamp_iso)
                eval_price = max(0.001, res["spread_price"])
                # Fallback stops for options debit spread if not set: 50% stop loss, 70% take profit
                if sl is None: sl = entry_px * 0.50
                if tp is None: tp = entry_px * 1.70
            else:
                if sl is None: sl = entry_px * 0.98
                if tp is None: tp = entry_px * 1.04

            exit_triggered = False
            exit_reason = ""
            exit_price = eval_price

            if side == "LONG":
                # Check Stop Loss (Gap-aware)
                if sl is not None and eval_price <= sl:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                    exit_price = eval_price
                # Check Take Profit
                elif tp is not None and eval_price >= tp:
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"
                    exit_price = eval_price

            if exit_triggered:
                # Trigger OMS sell order to close position
                order_id = f"ORD-EXIT-{uuid.uuid4().hex[:6].upper()}"
                cur.execute("""
                INSERT INTO orders (
                    order_id, timestamp_utc, asset_class, symbol, side, order_type,
                    quantity, limit_price, status, reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'SELL', 'MARKET', ?, ?, 'FILLED', ?, ?, ?)
                """, (order_id, timestamp_iso, pos["asset_class"], symbol, qty, exit_price, exit_reason, timestamp_iso, timestamp_iso))

                # Calculate friction & close position
                friction = self.cost_engine.calculate_fill_and_costs(
                    asset_class=pos["asset_class"],
                    side="SELL",
                    order_type="MARKET",
                    mid_price=exit_price,
                    atr=atr,
                    quantity=qty
                )
                fill_id = f"FILL-EXIT-{uuid.uuid4().hex[:6].upper()}"
                fill_px = friction["fill_price"]
                multiplier = 100.0 if pos["asset_class"] == "US_OPTION" else 1.0
                trade_notional = fill_px * qty * multiplier
                realized = (fill_px - pos["avg_entry_price"]) * qty * multiplier - friction["total_friction_usd"]

                cur.execute("""
                INSERT INTO executions (
                    fill_id, order_id, fill_timestamp, symbol, asset_class,
                    side, quantity, mid_price_at_order, fill_price,
                    spread_cost_usd, slippage_cost_usd, commission_fee_usd, total_friction_usd
                ) VALUES (?, ?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fill_id, order_id, timestamp_iso, symbol, pos["asset_class"],
                    qty, exit_price, fill_px, friction["spread_cost_usd"],
                    friction["slippage_cost_usd"], friction["commission_fee_usd"], friction["total_friction_usd"]
                ))

                cur.execute("""
                UPDATE positions SET
                    status = 'CLOSED', quantity = 0.0, current_mark_price = ?,
                    realized_pnl = realized_pnl + ?, unrealized_pnl = 0.0, closed_at = ?
                WHERE position_id = ?
                """, (fill_px, realized, timestamp_iso, pos["position_id"]))

                self._update_ledger_on_fill(
                    conn=conn,
                    side="SELL",
                    trade_notional=trade_notional,
                    total_friction=friction["total_friction_usd"],
                    timestamp_iso=timestamp_iso,
                    note=f"Position {pos['position_id']} closed via {exit_reason} @ {fill_px:.2f}"
                )

    def expire_options_sweep(self, underlying_quotes: Dict[str, float], current_time_iso: str) -> List[Dict[str, Any]]:
        """
        Daily expiration sweep (16:00 ET): settles expired options at intrinsic value.
        """
        expired_events = []
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions WHERE asset_class = 'US_OPTION' AND status = 'OPEN'")
            open_options = cur.fetchall()

            for pos in open_options:
                legs_json = pos["legs_json"]
                if not legs_json: continue
                contracts = json.loads(legs_json)
                underlying = pos["underlying"]
                spot = underlying_quotes.get(underlying, pos["avg_entry_price"])
                expiry_date = contracts[0].get("expiry", "")

                res = self.opt_engine.evaluate_expiration(
                    underlying_spot=spot,
                    contracts=contracts,
                    expiry_date_str=expiry_date,
                    current_time_iso=current_time_iso
                )

                if res["expired"]:
                    intrinsic_value = res["settlement_value"]
                    qty = pos["quantity"]
                    multiplier = 100.0
                    total_cash_settlement = intrinsic_value * qty * multiplier
                    entry_cost = pos["avg_entry_price"] * qty * multiplier
                    realized_pnl = total_cash_settlement - entry_cost

                    cur.execute("""
                    UPDATE positions SET
                        status = 'EXPIRED',
                        quantity = 0.0,
                        current_mark_price = ?,
                        realized_pnl = realized_pnl + ?,
                        unrealized_pnl = 0.0,
                        closed_at = ?
                    WHERE position_id = ?
                    """, (intrinsic_value, realized_pnl, current_time_iso, pos["position_id"]))

                    # Credit cash with intrinsic value
                    if total_cash_settlement > 0:
                        self._update_ledger_on_fill(
                            conn=conn,
                            side="SELL",
                            trade_notional=total_cash_settlement,
                            total_friction=0.0,
                            timestamp_iso=current_time_iso,
                            note=f"Options Expiration Cash Settlement {pos['symbol']} intrinsic: ${intrinsic_value:.2f}"
                        )

                    expired_events.append({
                        "position_id": pos["position_id"],
                        "symbol": pos["symbol"],
                        "intrinsic_value": intrinsic_value,
                        "realized_pnl": realized_pnl,
                        "is_itm": res["is_itm"]
                    })

            conn.commit()
        return expired_events
