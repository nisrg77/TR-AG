"""
Microstructure Features & Pre-Trade Routing Intelligence:
1. L2 Order Book Imbalance (OBI): top 5 levels in [-1, +1].
2. Smart Routing: Downgrades aggressive MARKET orders to passive LIMIT if OBI < -0.40 on BUY.
3. Liquidity Sweep (Stop-Hunt) Detector: Pierces 20-bar swing on >= 1.8x volume and closes inside range.
4. Options Liquidity Filter: Enforces OI >= 250, Vol >= 50, Spread <= 12%.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

class MicrostructureEngine:
    def __init__(self):
        pass

    def calculate_obi(
        self,
        bids: List[Tuple[float, float]], # List of (price, volume)
        asks: List[Tuple[float, float]],
        levels: int = 5
    ) -> float:
        """
        Computes Order Book Imbalance across top N levels:
        OBI = (Sum(V_bid) - Sum(V_ask)) / (Sum(V_bid) + Sum(V_ask)) in [-1, +1]
        """
        bid_vol = sum(vol for _, vol in bids[:levels])
        ask_vol = sum(vol for _, vol in asks[:levels])
        total_vol = bid_vol + ask_vol
        if total_vol <= 1e-6:
            return 0.0
        return float((bid_vol - ask_vol) / total_vol)

    def route_order_with_obi(
        self,
        intended_order_type: str,
        side: str,
        obi: float,
        best_bid: float,
        best_ask: float
    ) -> Tuple[str, Optional[float], str]:
        """
        Smart Order Routing:
        - If BUY MARKET and OBI < -0.40 (heavy ask wall overhead), downgrade to passive LIMIT at best bid/mid.
        - If SELL MARKET and OBI > +0.40 (heavy bid wall beneath), downgrade to passive LIMIT at best ask/mid.
        Returns (routed_order_type, limit_price, rationale).
        """
        mid = 0.5 * (best_bid + best_ask)

        if side.upper() == "BUY" and intended_order_type.upper() == "MARKET" and obi < -0.40:
            passive_px = round(best_bid + 0.01, 2)
            return "LIMIT", passive_px, f"OBI_DOWNGRADE: OBI={obi:.2f} indicates heavy ask resistance. Downgraded MARKET -> passive LIMIT @ ${passive_px:.2f}"

        if side.upper() == "SELL" and intended_order_type.upper() == "MARKET" and obi > 0.40:
            passive_px = round(best_ask - 0.01, 2)
            return "LIMIT", passive_px, f"OBI_DOWNGRADE: OBI={obi:.2f} indicates heavy bid support. Downgraded MARKET -> passive LIMIT @ ${passive_px:.2f}"

        return intended_order_type.upper(), None, "Order routed without modification"

    def detect_liquidity_sweep(
        self,
        df_bars: pd.DataFrame,
        lookback: int = 20,
        volume_threshold_mult: float = 1.8
    ) -> Dict[str, Any]:
        """
        Detects Institutional Liquidity Grab (Stop Hunt):
        Price pierces a 20-bar swing high/low on >= 1.8x volume,
        then closes back inside the prior range within 1-2 candles.
        """
        if len(df_bars) < lookback + 2:
            return {"sweep_detected": False, "sweep_type": "NONE"}

        highs = df_bars["high"].values
        lows = df_bars["low"].values
        closes = df_bars["close"].values
        volumes = df_bars["volume"].values

        prior_highs = highs[-(lookback + 2):-2]
        prior_lows = lows[-(lookback + 2):-2]
        swing_high = float(np.max(prior_highs))
        swing_low = float(np.min(prior_lows))

        avg_vol = float(np.mean(volumes[-(lookback + 2):-2]))
        current_vol = float(volumes[-1])
        vol_spike = (current_vol / max(1.0, avg_vol)) >= volume_threshold_mult

        curr_high = float(highs[-1])
        curr_low = float(lows[-1])
        curr_close = float(closes[-1])

        # Bearish Sweep of Highs (Bull Trap / Stop Run on Longs)
        if curr_high > swing_high and curr_close < swing_high and vol_spike:
            return {
                "sweep_detected": True,
                "sweep_type": "BEARISH_SWEEP_OF_HIGHS",
                "swing_level": swing_high,
                "vol_spike_ratio": round(current_vol / avg_vol, 2),
                "safe_stop_placement": round(curr_high + (curr_high - swing_high) * 0.5, 2)
            }

        # Bullish Sweep of Lows (Bear Trap / Stop Run on Shorts)
        if curr_low < swing_low and curr_close > swing_low and vol_spike:
            return {
                "sweep_detected": True,
                "sweep_type": "BULLISH_SWEEP_OF_LOWS",
                "swing_level": swing_low,
                "vol_spike_ratio": round(current_vol / avg_vol, 2),
                "safe_stop_placement": round(curr_low - (swing_low - curr_low) * 0.5, 2)
            }

        return {"sweep_detected": False, "sweep_type": "NONE"}

    def validate_options_liquidity(
        self,
        open_interest: int,
        daily_volume: int,
        bid_price: float,
        ask_price: float,
        min_oi: int = 250,
        min_vol: int = 50,
        max_spread_pct: float = 0.12
    ) -> Tuple[bool, str]:
        """
        Enforces Options Liquidity Gates:
        OI >= 250, Volume >= 50, Spread Width <= 12%.
        """
        if open_interest < min_oi:
            return False, f"OPTIONS_ILLIQUID_OI: Open Interest {open_interest} < {min_oi} threshold"
        if daily_volume < min_vol:
            return False, f"OPTIONS_ILLIQUID_VOL: Daily Volume {daily_volume} < {min_vol} threshold"
        mid = 0.5 * (bid_price + ask_price)
        if mid > 0:
            spread_pct = (ask_price - bid_price) / mid
            if spread_pct > max_spread_pct:
                return False, f"OPTIONS_WIDE_SPREAD: Spread width {spread_pct*100:.1f}% > {max_spread_pct*100:.1f}% limit"

        return True, "Passed options liquidity check"
