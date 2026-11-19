"""
Margin Engine, Liquidation Watcher & Financing Accrual:
1. Calculates exact liquidation prices based on maintenance margin requirements.
2. Intra-bar wick liquidation checker (liquidates position if bar low/high breaches liquidation price).
3. 8-Hour Crypto Funding Fee & Overnight Financing Swap Accrual (SOFR + 2.0% per night).
"""

from typing import Dict, Any, Optional, Tuple

class MarginLiquidationEngine:
    def __init__(
        self,
        crypto_maintenance_margin_pct: float = 0.005,  # 0.5% maintenance margin
        equity_maintenance_margin_pct: float = 0.020,  # 2.0% maintenance margin
        commodity_maintenance_margin_pct: float = 0.015,
        default_crypto_funding_rate_8h: float = 0.0001, # 0.01% per 8h
        annual_overnight_swap_rate: float = 0.075        # 7.5% per annum (SOFR + spread)
    ):
        self.crypto_mm = crypto_maintenance_margin_pct
        self.equity_mm = equity_maintenance_margin_pct
        self.commodity_mm = commodity_maintenance_margin_pct
        self.crypto_funding_8h = default_crypto_funding_rate_8h
        self.daily_swap_rate = annual_overnight_swap_rate / 365.0

    def calculate_liquidation_price(
        self,
        asset_class: str,
        side: str,              # BUY (LONG), SELL (SHORT)
        entry_price: float,
        leverage: float
    ) -> float:
        """
        Computes exact exchange-level liquidation price.
        """
        ac = asset_class.upper()
        if ac == "CRYPTO":
            mm = self.crypto_mm
        elif ac == "US_EQUITY":
            mm = self.equity_mm
        elif ac == "COMMODITY":
            mm = self.commodity_mm
        else:
            return 0.0 # Options cannot be margin-liquidated

        lev = max(1.0, leverage)
        initial_margin_req = 1.0 / lev

        if side.upper() in ["BUY", "LONG"]:
            # Price drops until remaining margin equals maintenance margin
            liq_price = entry_price * (1.0 - initial_margin_req + mm)
            return round(max(0.01, liq_price), 4)
        else:
            liq_price = entry_price * (1.0 + initial_margin_req - mm)
            return round(liq_price, 4)

    def check_wick_liquidation(
        self,
        side: str,
        liquidation_price: float,
        high_price: float,
        low_price: float
    ) -> Tuple[bool, float]:
        """
        Checks if the intra-bar candle wick breached the liquidation price.
        Returns (is_liquidated, fill_price).
        """
        if side.upper() in ["BUY", "LONG"]:
            if low_price <= liquidation_price:
                # Wiped out by intra-bar flush
                return True, liquidation_price
        else:
            if high_price >= liquidation_price:
                return True, liquidation_price

        return False, 0.0

    def calculate_funding_and_swap_accrual(
        self,
        asset_class: str,
        notional_value: float,
        hours_held: float
    ) -> float:
        """
        Computes 8-hour crypto funding fee or overnight financing swap fee.
        """
        ac = asset_class.upper()
        total_financing_fee = 0.0

        if ac == "CRYPTO":
            # 8-hour funding intervals
            intervals_8h = hours_held / 8.0
            total_financing_fee = notional_value * self.crypto_funding_8h * intervals_8h
        elif ac in ["US_EQUITY", "COMMODITY"]:
            # Overnight borrow interest on leveraged borrowed funds
            days_held = hours_held / 24.0
            total_financing_fee = notional_value * self.daily_swap_rate * days_held

        return round(float(total_financing_fee), 4)
