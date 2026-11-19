"""
Safe 4-Tier Pyramiding Engine (Anti-Martingale Winner-Only Scaling):
Enforces the 5 Hard Pyramiding Rules:
1. Max 4 Active Tiers per symbol/direction.
2. Winner-Only Scaling: Tier N+1 blocked unless existing tiers profitable (Tier N >= +0.75R).
3. Minimum Price Spacing: Tier N+1 entry >= 0.75 * ATR further in trend direction.
4. Decaying Lot Sizing: Tier 1 (1.0x), Tier 2 (0.75x), Tier 3 (0.50x), Tier 4 (0.25x).
5. Synchronized Stop-Loss Ratcheting: Combined worst-case stop-out loss <= 3.0% NLV.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from config.settings import DATABASE_PATH
from oms.database import get_connection

TIER_MULTIPLIERS = {1: 1.00, 2: 0.75, 3: 0.50, 4: 0.25}
TIER_RISK_CAPS = {1: 0.012, 2: 0.009, 3: 0.006, 4: 0.003}

class PyramidManager:
    def __init__(self, db_path: Path = DATABASE_PATH, max_aggregate_pyramid_risk_pct: float = 0.030):
        self.db_path = db_path
        self.max_aggregate_pyramid_risk_pct = max_aggregate_pyramid_risk_pct

    def evaluate_pyramid_entry(
        self,
        symbol: str,
        side: str,
        current_price: float,
        atr: float,
        base_unit_qty: float,
        account_nlv: float
    ) -> Tuple[bool, str, int, float, float]:
        """
        Validates if adding another pyramid tier is allowed.
        Returns (is_approved, reason, next_tier_num, approved_qty, recommended_stop).
        """
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT tranche_id, pyramid_tier, quantity, entry_price, stop_loss
            FROM position_tranches
            WHERE symbol = ? AND status = 'OPEN'
            ORDER BY pyramid_tier ASC
            """, (symbol,))
            existing_tranches = cur.fetchall()

        # If no position open yet, this is Tier 1
        if not existing_tranches:
            tier_num = 1
            qty = max(1.0, base_unit_qty * TIER_MULTIPLIERS[1])
            stop = current_price - (2.0 * atr) if side == "BUY" else current_price + (2.0 * atr)
            return True, "Base Tier 1 Entry Approved", tier_num, qty, stop

        # Rule 1: Max 4 active tiers
        current_max_tier = max(t["pyramid_tier"] for t in existing_tranches)
        next_tier = current_max_tier + 1
        if next_tier > 4:
            return False, f"PYRAMID_MAX_TIERS: Already at max 4 active tiers for {symbol}", 0, 0.0, 0.0

        # Rule 2: Winner-Only Scaling (Anti-Martingale)
        # Check if ALL existing tiers are currently profitable
        for t in existing_tranches:
            entry = t["entry_price"]
            unrealized = (current_price - entry) if side == "BUY" else (entry - current_price)
            if unrealized < 0:
                return False, f"PYRAMID_TIER_UNPROFITABLE: Existing Tier {t['pyramid_tier']} is losing (${unrealized:.2f}). Averaging down strictly blocked.", 0, 0.0, 0.0

        # Check that the most recent tier has achieved at least +0.75R
        last_tranche = existing_tranches[-1]
        last_entry = last_tranche["entry_price"]
        last_stop = last_tranche["stop_loss"]
        r_unit = abs(last_entry - last_stop)
        if r_unit <= 1e-4: r_unit = atr * 1.5
        current_gain = (current_price - last_entry) if side == "BUY" else (last_entry - current_price)
        if current_gain < (0.75 * r_unit):
            return False, f"PYRAMID_GAIN_TOO_SMALL: Tier {last_tranche['pyramid_tier']} gain (${current_gain:.2f}) < 0.75R threshold (${0.75 * r_unit:.2f})", 0, 0.0, 0.0

        # Rule 3: Minimum Price Spacing (>= 0.75 * ATR)
        price_diff = abs(current_price - last_entry)
        min_spacing = 0.75 * atr
        if price_diff < min_spacing:
            return False, f"PYRAMID_SPACING_TOO_TIGHT: Price spacing (${price_diff:.2f}) < 0.75*ATR (${min_spacing:.2f})", 0, 0.0, 0.0

        # Rule 4: Decaying Lot Sizing
        mult = TIER_MULTIPLIERS.get(next_tier, 0.25)
        approved_qty = max(1.0, round(base_unit_qty * mult, 2))

        # Rule 5: Synchronized Stop-Loss Ratcheting
        # Place new tier stop 1.5*ATR below entry, and ratchet earlier stops
        recommended_stop = current_price - (1.5 * atr) if side == "BUY" else current_price + (1.5 * atr)

        # Check total combined risk across all tiers at new ratchet stop
        total_worst_case_loss = 0.0
        for t in existing_tranches:
            qty_t = t["quantity"]
            entry_t = t["entry_price"]
            loss_at_new_stop = max(0.0, (entry_t - recommended_stop) * qty_t if side == "BUY" else (recommended_stop - entry_t) * qty_t)
            total_worst_case_loss += loss_at_new_stop

        new_tier_loss = abs(current_price - recommended_stop) * approved_qty
        total_worst_case_loss += new_tier_loss
        total_risk_pct = total_worst_case_loss / account_nlv

        if total_risk_pct > self.max_aggregate_pyramid_risk_pct:
            # Downsize new tier to fit within 3.0% combined risk
            avail_risk = max(0.0, (self.max_aggregate_pyramid_risk_pct * account_nlv) - (total_worst_case_loss - new_tier_loss))
            approved_qty = max(1.0, round(avail_risk / abs(current_price - recommended_stop), 2))

        return True, f"Tier {next_tier} Approved with Decaying Size ({mult}x)", next_tier, approved_qty, recommended_stop

    def ratchet_stops_on_new_tier(self, symbol: str, side: str, ratcheted_stop: float) -> int:
        """
        Ratchets all existing tiers to the higher stop loss level.
        """
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            if side == "BUY":
                cur.execute("""
                UPDATE position_tranches
                SET stop_loss = MAX(stop_loss, ?)
                WHERE symbol = ? AND status = 'OPEN'
                """, (ratcheted_stop, symbol))
            else:
                cur.execute("""
                UPDATE position_tranches
                SET stop_loss = MIN(stop_loss, ?)
                WHERE symbol = ? AND status = 'OPEN'
                """, (ratcheted_stop, symbol))
            affected = cur.rowcount
            conn.commit()
            return affected
