"""
C++ Quant Core ctypes Bridge:
Provides Python bindings to native C++17 Black-Scholes pricing, Greeks, Implied Volatility,
realistic execution friction, and portfolio risk aggregation.
"""

import ctypes
import os
import platform
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

class OptionGreeksStruct(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("vega", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("rho", ctypes.c_double),
    ]

_lib = None
_lib_loaded = False

def _load_native_library():
    global _lib, _lib_loaded
    if _lib_loaded:
        return _lib

    base_dir = Path(__file__).resolve().parent
    system = platform.system()
    lib_name = "quant_core.dll" if system == "Windows" else "quant_core.so"
    lib_path = base_dir / lib_name

    if lib_path.exists():
        try:
            _lib = ctypes.CDLL(str(lib_path))
            
            # 1. calculate_black_scholes
            _lib.calculate_black_scholes.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int
            ]
            _lib.calculate_black_scholes.restype = OptionGreeksStruct

            # 2. calculate_implied_volatility
            _lib.calculate_implied_volatility.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_double
            ]
            _lib.calculate_implied_volatility.restype = ctypes.c_double

            # 3. batch_calculate_greeks
            _lib.batch_calculate_greeks.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(OptionGreeksStruct)
            ]
            _lib.batch_calculate_greeks.restype = None

            # 4. calculate_execution_price_and_friction
            _lib.calculate_execution_price_and_friction.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_double,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
            ]
            _lib.calculate_execution_price_and_friction.restype = ctypes.c_double

            # 5. aggregate_portfolio_risk
            _lib.aggregate_portfolio_risk.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            _lib.aggregate_portfolio_risk.restype = None

            _lib_loaded = True
            return _lib
        except Exception as e:
            print(f"[C++ Bridge] Failed to load library: {e}")
            return None
    return None

_lib = _load_native_library()

def black_scholes(S: float, K: float, T: float, r: float = 0.045, sigma: float = 0.25, q: float = 0.0, is_call: bool = True) -> Dict[str, float]:
    """Calculate option price and all primary Greeks."""
    if _lib:
        res = _lib.calculate_black_scholes(S, K, T, r, sigma, q, 1 if is_call else 0)
        return {
            "price": res.price,
            "delta": res.delta,
            "gamma": res.gamma,
            "vega": res.vega,
            "theta": res.theta,
            "rho": res.rho,
        }
    
    # Python fallback using scipy
    from scipy.stats import norm
    if S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    if T <= 1e-6:
        val = max(0.0, S - K) if is_call else max(0.0, K - S)
        return {"price": val, "delta": 1.0 if (is_call and S > K) else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = (S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)) / 100.0
    theta = (-(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if is_call else -d2)) / 365.0
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2 if is_call else -d2)) / 100.0
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}

def implied_volatility(market_price: float, S: float, K: float, T: float, r: float = 0.045, q: float = 0.0, is_call: bool = True, initial_guess: float = 0.25) -> float:
    """Calculate implied volatility via C++ high-speed solver."""
    if _lib:
        return float(_lib.calculate_implied_volatility(market_price, S, K, T, r, q, 1 if is_call else 0, initial_guess))
    
    # Simple fallback
    sigma = initial_guess
    for _ in range(25):
        p = black_scholes(S, K, T, r, sigma, q, is_call)["price"]
        diff = p - market_price
        if abs(diff) < 1e-4:
            return sigma
        vega = black_scholes(S, K, T, r, sigma, q, is_call)["vega"] * 100.0
        if vega < 1e-5:
            break
        sigma -= diff / vega
        if sigma <= 0.01 or sigma > 5.0:
            return 0.25
    return sigma

def calculate_friction(
    asset_class: str,       # 'US_EQUITY', 'US_OPTION', 'CRYPTO', 'COMMODITY'
    side: str,              # 'BUY', 'SELL'
    order_type: str,        # 'MARKET', 'LIMIT', 'STOP'
    mid_price: float,
    bid_price: float,
    ask_price: float,
    atr: float = 1.0,
    volume: float = 100000.0,
    open_interest: float = 1000.0,
    num_legs: int = 1,
    quantity: float = 1.0
) -> Tuple[float, float, float]:
    """
    Returns (fill_price, slippage_usd, fee_usd) using native C++ microstructure cost engine.
    """
    ac_map = {"US_EQUITY": 0, "US_OPTION": 1, "CRYPTO": 2, "COMMODITY": 3}
    side_val = 1 if side.upper() == "BUY" else -1
    ot_map = {"MARKET": 0, "LIMIT": 1, "STOP": 2}
    
    ac_code = ac_map.get(asset_class.upper(), 0)
    ot_code = ot_map.get(order_type.upper(), 0)

    if _lib:
        slip_c = ctypes.c_double(0.0)
        fee_c = ctypes.c_double(0.0)
        fill_price = _lib.calculate_execution_price_and_friction(
            ac_code, side_val, ot_code,
            mid_price, bid_price, ask_price,
            atr, volume, open_interest,
            num_legs, quantity,
            ctypes.byref(slip_c), ctypes.byref(fee_c)
        )
        return float(fill_price), float(slip_c.value), float(fee_c.value)

    # Fallback
    slip = 0.01 * quantity
    fee = 1.0
    fill = ask_price if side_val == 1 else bid_price
    return fill, slip, fee

def aggregate_portfolio(
    spots: List[float],
    strikes: List[float],
    ttes: List[float],
    rates: List[float],
    ivs: List[float],
    is_calls: List[bool],
    quantities: List[float],
    entry_prices: List[float]
) -> Dict[str, float]:
    """Calculate portfolio net Greeks and unrealized PnL in C++ vector loop."""
    n = len(spots)
    if n == 0:
        return {"unrealized_pnl": 0.0, "net_delta": 0.0, "net_gamma": 0.0, "net_vega": 0.0, "net_theta": 0.0}

    if _lib:
        c_spots = (ctypes.c_double * n)(*spots)
        c_strikes = (ctypes.c_double * n)(*strikes)
        c_ttes = (ctypes.c_double * n)(*ttes)
        c_rates = (ctypes.c_double * n)(*rates)
        c_ivs = (ctypes.c_double * n)(*ivs)
        c_calls = (ctypes.c_int * n)(*[1 if c else 0 for c in is_calls])
        c_quantities = (ctypes.c_double * n)(*quantities)
        c_entries = (ctypes.c_double * n)(*entry_prices)

        out_pnl = ctypes.c_double(0.0)
        out_delta = ctypes.c_double(0.0)
        out_gamma = ctypes.c_double(0.0)
        out_vega = ctypes.c_double(0.0)
        out_theta = ctypes.c_double(0.0)

        _lib.aggregate_portfolio_risk(
            c_spots, c_strikes, c_ttes, c_rates, c_ivs, c_calls, c_quantities, c_entries, n,
            ctypes.byref(out_pnl), ctypes.byref(out_delta), ctypes.byref(out_gamma), ctypes.byref(out_vega), ctypes.byref(out_theta)
        )
        return {
            "unrealized_pnl": float(out_pnl.value),
            "net_delta": float(out_delta.value),
            "net_gamma": float(out_gamma.value),
            "net_vega": float(out_vega.value),
            "net_theta": float(out_theta.value),
        }

    # Python fallback
    pnl, delta, gamma, vega, theta = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        g = black_scholes(spots[i], strikes[i], ttes[i], rates[i], ivs[i], 0.0, is_calls[i])
        mult = 100.0 if strikes[i] > 0 else 1.0
        pnl += (g["price"] * mult - entry_prices[i] * mult) * quantities[i]
        delta += g["delta"] * mult * quantities[i]
        gamma += g["gamma"] * mult * quantities[i]
        vega += g["vega"] * mult * quantities[i]
        theta += g["theta"] * mult * quantities[i]
    return {
        "unrealized_pnl": pnl, "net_delta": delta, "net_gamma": gamma, "net_vega": vega, "net_theta": theta
    }
