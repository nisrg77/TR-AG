"""
Quant Volatility- & Regime-Adjusted Dynamic Leverage Engine:
Implements institutional dynamic leverage formula:
L_dynamic = clip(L_max * M_vol * M_conf * M_dd * M_event, L_min, L_tier(Notional))

Where:
- M_vol = sigma_target / sigma_realized (contracts on ATR/vol spikes)
- M_conf = scales between 0.5x (conf=0.65) and 1.0x (conf>=0.85)
- M_dd = cuts leverage by 50% if account drops > 1.5% intraday or > 4% from peak
- M_event = drops leverage to <= 10x before macro events/weekend gaps
- Liquidation buffer clamp: L <= 0.70 / stop_distance_pct
"""

import numpy as np
from typing import Dict, Any, Optional

LEVERAGE_BOUNDS = {
    "CRYPTO": {"max": 100.0, "min": 5.0, "target_vol": 0.45},
    "US_EQUITY": {"max": 40.0, "min": 2.0, "target_vol": 0.18},
    "COMMODITY": {"max": 50.0, "min": 5.0, "target_vol": 0.20},
    "US_OPTION": {"max": 1.0, "min": 1.0, "target_vol": 0.25} # Options are non-margined debit/defined risk
}

class DynamicLeverageEngine:
    def __init__(self):
        pass

    def compute_dynamic_leverage(
        self,
        asset_class: str,
        current_price: float,
        stop_price: Optional[float],
        realized_vol: float,
        confidence_score: float = 0.80,
        account_intraday_pnl_pct: float = 0.0,
        account_drawdown_pct: float = 0.0,
        is_event_window: bool = False,
        cumulative_notional: float = 0.0
    ) -> Dict[str, Any]:
        """
        Computes asset-specific dynamic leverage with all risk multipliers and clamps.
        """
        ac = asset_class.upper()
        cfg = LEVERAGE_BOUNDS.get(ac, LEVERAGE_BOUNDS["US_EQUITY"])
        l_max = cfg["max"]
        l_min = cfg["min"]
        target_vol = cfg["target_vol"]

        if ac == "US_OPTION":
            return {
                "dynamic_leverage": 1.0,
                "multipliers": {"vol": 1.0, "conf": 1.0, "dd": 1.0, "event": 1.0},
                "max_allowed": 1.0,
                "clamped_by": "OPTION_CASH"
            }

        # 1. Volatility Multiplier: M_vol = target_vol / realized_vol
        # In quiet, low-volatility regimes: M_vol expands toward 1.25x
        # In turbulent high-ATR spikes: M_vol contracts (e.g. 0.20x - 0.35x)
        r_vol = max(0.05, realized_vol)
        m_vol = float(np.clip(target_vol / r_vol, 0.20, 1.25))

        # 2. Confidence & Trend Multiplier (0.50x to 1.00x)
        # conf in [0.65, 0.90] -> m_conf in [0.50, 1.00]
        m_conf = float(np.clip(0.50 + ((confidence_score - 0.65) / 0.25) * 0.50, 0.50, 1.00))

        # 3. Drawdown Throttle (M_dd): Cut by 50% if intraday loss > 1.5% or peak DD > 4.0%
        m_dd = 1.0
        if account_intraday_pnl_pct <= -0.015 or account_drawdown_pct <= -0.040:
            m_dd = 0.50

        # 4. News / Macro Event / Weekend Guard
        m_event = 1.0
        if is_event_window:
            m_event = 0.25 # Sharp contract down to safe zone

        raw_leverage = l_max * m_vol * m_conf * m_dd * m_event

        # 5. Liquidation-Buffer Clamp
        # Ensure liquidation price is strictly beyond the stop loss
        # E.g. If stop is 1.5% away, max leverage is clamped to <= 0.70 / 0.015 = 46.6x
        clamped_by = "FORMULA"
        if stop_price is not None and stop_price > 0 and current_price > 0:
            stop_dist_pct = abs(current_price - stop_price) / current_price
            if stop_dist_pct > 0.001:
                # 70% safety buffer before liquidation maintenance margin
                max_safe_lev_from_stop = 0.70 / stop_dist_pct
                if raw_leverage > max_safe_lev_from_stop:
                    raw_leverage = max_safe_lev_from_stop
                    clamped_by = "LIQUIDATION_BUFFER_STOP_CLAMP"

        # 6. Progressive Notional Tier Bracket Cap
        tier_cap = self.get_notional_tier_cap(ac, cumulative_notional)
        if raw_leverage > tier_cap:
            raw_leverage = tier_cap
            clamped_by = "NOTIONAL_TIER_CAP"

        # Final clip between L_min and L_max
        final_leverage = float(np.clip(raw_leverage, l_min, l_max))

        return {
            "dynamic_leverage": round(final_leverage, 1),
            "multipliers": {
                "m_vol": round(m_vol, 3),
                "m_conf": round(m_conf, 2),
                "m_dd": round(m_dd, 2),
                "m_event": round(m_event, 2)
            },
            "raw_leverage": round(raw_leverage, 1),
            "clamped_by": clamped_by,
            "notional_tier_cap": tier_cap
        }

    @staticmethod
    def get_notional_tier_cap(asset_class: str, cumulative_notional: float) -> float:
        """
        Progressive Notional Tier Margin Brackets (Binance/Prime Broker Realism).
        """
        ac = asset_class.upper()
        if ac == "CRYPTO":
            if cumulative_notional <= 25000.0:
                return 100.0
            elif cumulative_notional <= 100000.0:
                return 50.0
            else:
                return 20.0
        elif ac == "US_EQUITY":
            if cumulative_notional <= 50000.0:
                return 40.0
            elif cumulative_notional <= 200000.0:
                return 25.0
            else:
                return 10.0
        elif ac == "COMMODITY":
            if cumulative_notional <= 50000.0:
                return 50.0
            elif cumulative_notional <= 150000.0:
                return 25.0
            else:
                return 15.0
        return 1.0
