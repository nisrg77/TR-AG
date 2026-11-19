"""
L2 Order Book Simulator & Multi-Level Book Walking Engine:
- Models 5-level depth books for bids and asks.
- Enforces 10% Depth Participation Cap per level.
- Simulates realistic multi-level book walking across levels when order size exceeds top-of-book depth.
"""

from typing import Dict, Any, List, Tuple

class OrderBookSimulator:
    def __init__(self, depth_participation_cap_pct: float = 0.10):
        self.participation_cap = depth_participation_cap_pct

    def simulate_book_walk_fill(
        self,
        side: str,                  # BUY, SELL
        order_qty: float,
        book_levels: List[Tuple[float, float]], # List of (price, available_volume)
        mid_price: float
    ) -> Dict[str, Any]:
        """
        Simulates walking the book across multiple depth levels.
        book_levels: For BUY: list of asks sorted ascending by price [(100.05, 50), (100.10, 80), ...]
                     For SELL: list of bids sorted descending by price [(99.95, 50), (99.90, 80), ...]
        """
        remaining_qty = order_qty
        fills_per_level = []
        total_fill_cost = 0.0
        levels_walked = 0

        for level_idx, (lvl_price, lvl_vol) in enumerate(book_levels, start=1):
            levels_walked = level_idx
            # Available liquidity considering participation cap
            available_at_level = max(1.0, lvl_vol * self.participation_cap)

            filled_here = min(remaining_qty, available_at_level)
            fills_per_level.append({
                "level": level_idx,
                "price": lvl_price,
                "filled_qty": filled_here,
                "available_liquidity": lvl_vol
            })

            total_fill_cost += filled_here * lvl_price
            remaining_qty -= filled_here

            if remaining_qty <= 1e-5:
                break

        # If book ran out of depth, fill remainder at last level + penalty
        if remaining_qty > 1e-5:
            last_price = book_levels[-1][0]
            penalty = 0.001 * mid_price
            penalized_price = last_price + penalty if side.upper() == "BUY" else last_price - penalty
            fills_per_level.append({
                "level": levels_walked + 1,
                "price": penalized_price,
                "filled_qty": remaining_qty,
                "available_liquidity": 0.0
            })
            total_fill_cost += remaining_qty * penalized_price
            levels_walked += 1

        blended_fill_price = total_fill_cost / order_qty
        total_slippage_usd = abs(blended_fill_price - mid_price) * order_qty

        return {
            "order_qty": order_qty,
            "mid_price": mid_price,
            "blended_fill_price": round(blended_fill_price, 4),
            "total_slippage_usd": round(total_slippage_usd, 4),
            "levels_walked": levels_walked,
            "fills_per_level": fills_per_level
        }
