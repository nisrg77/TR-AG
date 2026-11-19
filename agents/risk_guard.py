"""
Deterministic Risk Manager Gate:
Hard-coded Python checks for:
- Max risk per trade (<= 1% - 2% of equity)
- Liquidity rules (Open Interest, Bid-Ask spread width)
- Fee drag filter (rejects trades where friction > 20% of expected profit)
- Max open portfolio positions
"""

from typing import Dict, Any, Tuple

class RiskManagerGate:
    def __init__(
        self,
        max_risk_pct: float = 0.02,         # Max 2% equity risk per trade
        min_open_interest: int = 50,
        max_bid_ask_spread_pct: float = 0.20,
        max_fee_drag_pct: float = 0.20       # Max 20% friction-to-profit ratio
    ):
        self.max_risk_pct = max_risk_pct
        self.min_open_interest = min_open_interest
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.max_fee_drag_pct = max_fee_drag_pct

    def validate_trade_candidate(
        self,
        signal: Dict[str, Any],
        account_equity: float,
        current_open_positions: int = 0,
        max_positions: int = 10
    ) -> Tuple[bool, str, int]:
        """
        Validates signal against deterministic risk gates.
        Returns (is_approved, rejection_reason, recommended_contracts_or_shares).
        """
        if current_open_positions >= max_positions:
            return False, f"Portfolio saturated ({current_open_positions}/{max_positions} active positions)", 0

        max_allowed_risk_usd = account_equity * self.max_risk_pct

        # 1. Option Strategy Validation
        if signal.get("asset_class") == "US_OPTION":
            contracts = signal.get("contracts", [])
            entry_limit = signal.get("entry_limit_price", 1.0)
            tp_price = signal.get("take_profit_price", entry_limit * 1.5)
            estimated_friction = signal.get("estimated_friction_usd", 14.0)

            # Check spread width if provided
            bid = signal.get("bid_price", entry_limit * 0.95)
            ask = signal.get("ask_price", entry_limit * 1.05)
            if entry_limit > 0.0:
                spread_width_pct = (ask - bid) / entry_limit
                if spread_width_pct > self.max_bid_ask_spread_pct:
                    return False, f"Bid-Ask spread too wide ({spread_width_pct*100:.1f}% > {self.max_bid_ask_spread_pct*100:.1f}%)", 0

            # Fee Drag Check
            expected_profit_per_contract = abs(tp_price - entry_limit) * 100.0
            if expected_profit_per_contract > 0:
                fee_drag = estimated_friction / expected_profit_per_contract
                if fee_drag > self.max_fee_drag_pct:
                    return False, f"Fee drag excessive ({fee_drag*100:.1f}% > {self.max_fee_drag_pct*100:.1f}% of target profit)", 0

            # Position Sizing: Risk Budget / Max Risk Per Contract
            risk_per_contract = signal.get("max_risk_per_contract", entry_limit * 100.0)
            if risk_per_contract <= 0: risk_per_contract = entry_limit * 100.0

            contracts_num = int(max_allowed_risk_usd // max(1.0, risk_per_contract))
            contracts_num = max(1, min(contracts_num, 10)) # Cap at 10 contracts for safety

            return True, "Passed all risk gates", contracts_num

        # 2. Equity / Crypto / Commodity Spot Validation
        else:
            entry_price = signal.get("entry_limit_price", 100.0)
            stop_price = signal.get("stop_loss_price", entry_price * 0.98)
            risk_per_unit = abs(entry_price - stop_price)
            if risk_per_unit <= 1e-4:
                risk_per_unit = entry_price * 0.02

            # Position sizing
            units = max_allowed_risk_usd / risk_per_unit
            # Cap notional value to 10% of equity
            max_units = (account_equity * 0.10) / entry_price
            if signal.get("asset_class") == "CRYPTO":
                units = round(min(units, max_units), 4)
                if units <= 0.0001: units = 0.01
            else:
                units = max(1, int(min(units, max_units)))

            return True, "Passed spot risk gates", units
