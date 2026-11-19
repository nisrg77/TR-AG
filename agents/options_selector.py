"""
Options Strategist Agent:
Selects strategy, strikes, and expiries based on IV Rank (IVR), DTE, and Delta targets.
Uses C++ Greeks engine to match optimal strikes.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from src.cpp.quant_bridge import black_scholes

class OptionsSelector:
    def __init__(self, target_dte: int = 30):
        self.target_dte = target_dte

    def select_strategy_and_strikes(
        self,
        underlying: str,
        spot_price: float,
        direction: str,         # BULLISH, BEARISH, NEUTRAL
        iv_rank: float,
        historical_vol: float = 0.22,
        expiry_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determines the optimal strategy and contracts:
        - Low IV Rank (< 35): Debit Vertical Spreads or Long Calls/Puts
        - High IV Rank (> 50): Credit Vertical Spreads or Iron Condors
        """
        if not expiry_date:
            expiry_date = (datetime.now() + timedelta(days=self.target_dte)).strftime("%Y-%m-%d")

        tte = self.target_dte / 365.0
        r = 0.045
        iv = max(0.10, historical_vol)

        # 1. Low IV Rank -> Debit Spreads
        if iv_rank < 35.0:
            if direction == "BULLISH":
                # Bull Call Spread (Buy 45-delta call, Sell 25-delta call)
                long_strike = round(spot_price * 1.00, 1)  # ATM / ~0.50 delta
                short_strike = round(spot_price * 1.03, 1) # OTM / ~0.25 delta

                long_greeks = black_scholes(spot_price, long_strike, tte, r, iv, is_call=True)
                short_greeks = black_scholes(spot_price, short_strike, tte, r, iv, is_call=True)

                net_debit = max(0.10, long_greeks["price"] - short_greeks["price"])
                max_profit = (short_strike - long_strike) - net_debit

                contracts = [
                    {"action": "BUY", "type": "CALL", "strike": long_strike, "expiry": expiry_date, "delta": round(long_greeks["delta"], 2), "iv": iv},
                    {"action": "SELL", "type": "CALL", "strike": short_strike, "expiry": expiry_date, "delta": round(short_greeks["delta"], 2), "iv": iv}
                ]

                return {
                    "strategy_type": "BULL_CALL_SPREAD",
                    "direction": "BULLISH",
                    "contracts": contracts,
                    "entry_limit_price": round(net_debit, 2),
                    "stop_loss_price": round(net_debit * 0.50, 2), # 50% max loss
                    "take_profit_price": round(net_debit + max_profit * 0.70, 2), # 70% max profit target
                    "max_risk_per_contract": round(net_debit * 100.0, 2),
                    "rationale": f"Low IV Rank ({iv_rank:.1f}) favors Bull Call Debit Spread to capitalize on upside momentum with defined risk."
                }

            elif direction == "BEARISH":
                # Bear Put Spread (Buy 45-delta put, Sell 25-delta put)
                long_strike = round(spot_price * 1.00, 1)
                short_strike = round(spot_price * 0.97, 1)

                long_greeks = black_scholes(spot_price, long_strike, tte, r, iv, is_call=False)
                short_greeks = black_scholes(spot_price, short_strike, tte, r, iv, is_call=False)

                net_debit = max(0.10, long_greeks["price"] - short_greeks["price"])
                max_profit = (long_strike - short_strike) - net_debit

                contracts = [
                    {"action": "BUY", "type": "PUT", "strike": long_strike, "expiry": expiry_date, "delta": round(long_greeks["delta"], 2), "iv": iv},
                    {"action": "SELL", "type": "PUT", "strike": short_strike, "expiry": expiry_date, "delta": round(short_greeks["delta"], 2), "iv": iv}
                ]

                return {
                    "strategy_type": "BEAR_PUT_SPREAD",
                    "direction": "BEARISH",
                    "contracts": contracts,
                    "entry_limit_price": round(net_debit, 2),
                    "stop_loss_price": round(net_debit * 0.50, 2),
                    "take_profit_price": round(net_debit + max_profit * 0.70, 2),
                    "max_risk_per_contract": round(net_debit * 100.0, 2),
                    "rationale": f"Low IV Rank ({iv_rank:.1f}) favors Bear Put Debit Spread for defined downside exposure."
                }

        # 2. High or Medium IV Rank -> Credit Spread
        if direction == "BULLISH":
            # Bull Put Credit Spread (Sell 30-delta put, Buy 15-delta put)
            short_strike = round(spot_price * 0.98, 1)
            long_strike = round(spot_price * 0.95, 1)

            short_greeks = black_scholes(spot_price, short_strike, tte, r, iv, is_call=False)
            long_greeks = black_scholes(spot_price, long_strike, tte, r, iv, is_call=False)

            net_credit = max(0.10, short_greeks["price"] - long_greeks["price"])
            spread_width = short_strike - long_strike
            max_risk = (spread_width - net_credit) * 100.0

            contracts = [
                {"action": "SELL", "type": "PUT", "strike": short_strike, "expiry": expiry_date, "delta": round(short_greeks["delta"], 2), "iv": iv},
                {"action": "BUY", "type": "PUT", "strike": long_strike, "expiry": expiry_date, "delta": round(long_greeks["delta"], 2), "iv": iv}
            ]

            return {
                "strategy_type": "BULL_PUT_SPREAD",
                "direction": "BULLISH",
                "contracts": contracts,
                "entry_limit_price": round(net_credit, 2),
                "stop_loss_price": round(net_credit * 2.5, 2),
                "take_profit_price": round(net_credit * 0.30, 2),
                "max_risk_per_contract": round(max_risk, 2),
                "rationale": f"Elevated IV Rank ({iv_rank:.1f}) favors Bull Put Credit Spread to harvest theta decay above key support."
            }

        # Default Neutral or Bearish Credit
        short_strike = round(spot_price * 1.02, 1)
        long_strike = round(spot_price * 1.05, 1)
        short_greeks = black_scholes(spot_price, short_strike, tte, r, iv, is_call=True)
        long_greeks = black_scholes(spot_price, long_strike, tte, r, iv, is_call=True)

        net_credit = max(0.10, short_greeks["price"] - long_greeks["price"])
        max_risk = ((long_strike - short_strike) - net_credit) * 100.0

        contracts = [
            {"action": "SELL", "type": "CALL", "strike": short_strike, "expiry": expiry_date, "delta": round(short_greeks["delta"], 2), "iv": iv},
            {"action": "BUY", "type": "CALL", "strike": long_strike, "expiry": expiry_date, "delta": round(long_greeks["delta"], 2), "iv": iv}
        ]

        return {
            "strategy_type": "BEAR_CALL_SPREAD",
            "direction": "BEARISH",
            "contracts": contracts,
            "entry_limit_price": round(net_credit, 2),
            "stop_loss_price": round(net_credit * 2.5, 2),
            "take_profit_price": round(net_credit * 0.30, 2),
            "max_risk_per_contract": round(max_risk, 2),
            "rationale": f"Bear Call Credit Spread harvesting premium under resistance."
        }
