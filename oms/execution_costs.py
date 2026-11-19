"""
Execution Friction, Realistic Slippage & Cost Model
Integrates C++ kernel for microsecond pricing with realistic microstructure constraints:
- Equities: Dynamic ATR volatility slippage + SEC/FINRA fees
- Options: Liquidity-scaled bid/ask crossing (k in [0.35, 0.85]) + multi-leg spread penalty + $0.69 all-in/contract
- Crypto: Maker (0 bps spread) vs Taker (dynamic volume impact) + tiered bps
- Commodities: Minimum tick slippage (1.5 ticks) + clearing fees
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.cpp.quant_bridge import calculate_friction

class ExecutionCostEngine:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config" / "execution_costs.yaml"
        
        self.config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}

    def calculate_fill_and_costs(
        self,
        asset_class: str,
        side: str,              # BUY, SELL
        order_type: str,        # MARKET, LIMIT, STOP
        mid_price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        atr: float = 1.0,
        volume: float = 100000.0,
        open_interest: float = 1000.0,
        num_legs: int = 1,
        quantity: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculates realistic fill price and itemized transaction costs.
        """
        # Ensure sane bid/ask defaults if missing
        if mid_price <= 0.0:
            mid_price = 100.0
        
        if bid_price is None or bid_price <= 0.0:
            bid_price = mid_price * 0.9995
        if ask_price is None or ask_price <= 0.0:
            ask_price = mid_price * 1.0005

        # Call C++ accelerated microstructure kernel
        fill_price, slippage_usd, fee_usd = calculate_friction(
            asset_class=asset_class,
            side=side,
            order_type=order_type,
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            atr=atr,
            volume=volume,
            open_interest=open_interest,
            num_legs=num_legs,
            quantity=quantity
        )

        spread_cost_usd = abs(fill_price - mid_price) * quantity
        total_friction_usd = spread_cost_usd + slippage_usd + fee_usd

        return {
            "fill_price": round(float(fill_price), 4),
            "mid_price_at_order": round(float(mid_price), 4),
            "spread_cost_usd": round(float(spread_cost_usd), 4),
            "slippage_cost_usd": round(float(slippage_usd), 4),
            "commission_fee_usd": round(float(fee_usd), 4),
            "total_friction_usd": round(float(total_friction_usd), 4)
        }
