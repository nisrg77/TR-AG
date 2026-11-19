"""
Options Lifecycle & Multi-Leg Settlement Engine
Handles:
1. Mark-to-market valuation & Greek aggregation (Delta, Gamma, Vega, Theta).
2. Automated expiration settlement (ITM cash settlement vs OTM worthless expiry).
3. Multi-leg spread synthesis (Bull Call Spread, Bear Put Spread, Iron Condor, etc.).
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from src.cpp.quant_bridge import black_scholes, aggregate_portfolio

class OptionsLifecycleEngine:
    def __init__(self, risk_free_rate: float = 0.045):
        self.r = risk_free_rate

    def price_contract_or_spread(
        self,
        underlying_spot: float,
        contracts: List[Dict[str, Any]],
        current_time_iso: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prices a single or multi-leg option position and returns total spread price & net Greeks.
        contracts: list of dicts:
          [{"action": "BUY"|"SELL", "type": "CALL"|"PUT", "strike": float, "expiry": "YYYY-MM-DD", "iv": float}]
        """
        net_price = 0.0
        net_delta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        net_theta = 0.0

        now = datetime.fromisoformat(current_time_iso.replace("Z", "+00:00")) if current_time_iso else datetime.now(timezone.utc)

        for leg in contracts:
            action = leg.get("action", "BUY").upper()
            is_call = leg.get("type", "CALL").upper() == "CALL"
            strike = float(leg.get("strike", underlying_spot))
            iv = float(leg.get("iv", 0.25))

            # Calculate TTE in years
            expiry_str = leg.get("expiry", "")
            try:
                exp_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00")) if "T" in expiry_str else datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                diff_days = (exp_dt - now).total_seconds() / 86400.0
                tte = max(1e-5, diff_days / 365.0)
            except Exception:
                tte = 30.0 / 365.0

            greeks = black_scholes(
                S=underlying_spot,
                K=strike,
                T=tte,
                r=self.r,
                sigma=iv,
                is_call=is_call
            )

            sign = 1.0 if action == "BUY" else -1.0
            net_price += sign * greeks["price"]
            net_delta += sign * greeks["delta"]
            net_gamma += sign * greeks["gamma"]
            net_vega  += sign * greeks["vega"]
            net_theta += sign * greeks["theta"]

        return {
            "spread_price": float(net_price),
            "net_delta": float(net_delta),
            "net_gamma": float(net_gamma),
            "net_vega": float(net_vega),
            "net_theta": float(net_theta),
        }

    def evaluate_expiration(
        self,
        underlying_spot: float,
        contracts: List[Dict[str, Any]],
        expiry_date_str: str,
        current_time_iso: str
    ) -> Dict[str, Any]:
        """
        Checks if the contract has expired.
        If expired, computes intrinsic cash settlement value and whether it is ITM/OTM.
        """
        now = datetime.fromisoformat(current_time_iso.replace("Z", "+00:00")) if current_time_iso else datetime.now(timezone.utc)
        try:
            exp_dt = datetime.fromisoformat(expiry_date_str.replace("Z", "+00:00")) if "T" in expiry_date_str else datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return {"expired": False, "settlement_value": 0.0, "is_itm": False}

        if now < exp_dt:
            return {"expired": False, "settlement_value": 0.0, "is_itm": False}

        # Expired: Compute intrinsic payoff
        total_intrinsic_payoff = 0.0
        is_itm = False

        for leg in contracts:
            action = leg.get("action", "BUY").upper()
            is_call = leg.get("type", "CALL").upper() == "CALL"
            strike = float(leg.get("strike", underlying_spot))
            
            if is_call:
                leg_intrinsic = max(0.0, underlying_spot - strike)
            else:
                leg_intrinsic = max(0.0, strike - underlying_spot)

            sign = 1.0 if action == "BUY" else -1.0
            total_intrinsic_payoff += sign * leg_intrinsic
            if leg_intrinsic > 0.01:
                is_itm = True

        return {
            "expired": True,
            "settlement_value": round(float(total_intrinsic_payoff), 4),
            "is_itm": is_itm
        }
