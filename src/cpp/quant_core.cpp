#include "quant_core.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace {

constexpr double INV_SQRT_2PI = 0.398942280401432677939946059934; // 1 / sqrt(2 * pi)
constexpr double SQRT_2 = 1.414213562373095048801688724210;

inline double normal_pdf(double x) {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / SQRT_2);
}

} // anonymous namespace

extern "C" {

EXPORT_API OptionGreeks calculate_black_scholes(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    double q,
    int is_call
) {
    OptionGreeks greeks{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Boundary edge cases
    if (S <= 0.0 || K <= 0.0) {
        return greeks;
    }
    if (T <= 1e-6) {
        // Immediate expiration payoff
        if (is_call) {
            greeks.price = std::max(0.0, S - K);
            greeks.delta = (S > K) ? 1.0 : ((S == K) ? 0.5 : 0.0);
        } else {
            greeks.price = std::max(0.0, K - S);
            greeks.delta = (S < K) ? -1.0 : ((S == K) ? -0.5 : 0.0);
        }
        return greeks;
    }
    if (sigma <= 1e-6) {
        sigma = 1e-6;
    }

    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    double exp_qT = std::exp(-q * T);
    double exp_rT = std::exp(-r * T);
    double pdf_d1 = normal_pdf(d1);

    if (is_call) {
        double cdf_d1 = normal_cdf(d1);
        double cdf_d2 = normal_cdf(d2);
        greeks.price = S * exp_qT * cdf_d1 - K * exp_rT * cdf_d2;
        greeks.delta = exp_qT * cdf_d1;
        greeks.theta = (- (S * sigma * exp_qT * pdf_d1) / (2.0 * sqrtT)
                        - r * K * exp_rT * cdf_d2
                        + q * S * exp_qT * cdf_d1) / 365.0; // per calendar day
        greeks.rho = (K * T * exp_rT * cdf_d2) / 100.0;     // per 1% rate change
    } else {
        double cdf_minus_d1 = normal_cdf(-d1);
        double cdf_minus_d2 = normal_cdf(-d2);
        greeks.price = K * exp_rT * cdf_minus_d2 - S * exp_qT * cdf_minus_d1;
        greeks.delta = -exp_qT * cdf_minus_d1;
        greeks.theta = (- (S * sigma * exp_qT * pdf_d1) / (2.0 * sqrtT)
                        + r * K * exp_rT * cdf_minus_d2
                        - q * S * exp_qT * cdf_minus_d1) / 365.0;
        greeks.rho = (-K * T * exp_rT * cdf_minus_d2) / 100.0;
    }

    greeks.gamma = (exp_qT * pdf_d1) / (S * sigma * sqrtT);
    greeks.vega = (S * exp_qT * sqrtT * pdf_d1) / 100.0; // per 1% vol change

    return greeks;
}

EXPORT_API double calculate_implied_volatility(
    double market_price,
    double S,
    double K,
    double T,
    double r,
    double q,
    int is_call,
    double initial_guess
) {
    if (market_price <= 0.0 || S <= 0.0 || K <= 0.0 || T <= 1e-6) {
        return 0.0;
    }

    // Lower bound: intrinsic value
    double intrinsic = is_call ? std::max(0.0, S - K) : std::max(0.0, K - S);
    if (market_price < intrinsic) {
        return 0.0;
    }

    double sigma = (initial_guess > 0.01 && initial_guess < 5.0) ? initial_guess : 0.25;
    constexpr int MAX_ITER = 30;
    constexpr double TOLERANCE = 1e-5;

    // Newton-Raphson iteration
    for (int i = 0; i < MAX_ITER; ++i) {
        OptionGreeks greeks = calculate_black_scholes(S, K, T, r, sigma, q, is_call);
        double diff = greeks.price - market_price;
        if (std::abs(diff) < TOLERANCE) {
            return sigma;
        }

        double vega_raw = greeks.vega * 100.0; // convert back to standard derivative dV/dSigma
        if (vega_raw < 1e-6) {
            break; // vega too small, switch to bisection
        }

        double next_sigma = sigma - diff / vega_raw;
        if (next_sigma <= 0.001 || next_sigma > 10.0) {
            break; // out of reasonable bounds, switch to bisection
        }
        sigma = next_sigma;
    }

    // Fallback: Robust Bisection
    double low = 0.001;
    double high = 8.0;
    for (int i = 0; i < 40; ++i) {
        double mid = 0.5 * (low + high);
        OptionGreeks g = calculate_black_scholes(S, K, T, r, mid, q, is_call);
        double diff = g.price - market_price;
        if (std::abs(diff) < TOLERANCE) {
            return mid;
        }
        if (diff > 0.0) {
            high = mid;
        } else {
            low = mid;
        }
    }

    return 0.5 * (low + high);
}

EXPORT_API void batch_calculate_greeks(
    const double* S,
    const double* K,
    const double* T,
    const double* r,
    const double* sigma,
    const double* q,
    const int* is_call,
    int n,
    OptionGreeks* results_out
) {
    if (!results_out || n <= 0) return;
    for (int i = 0; i < n; ++i) {
        results_out[i] = calculate_black_scholes(
            S[i], K[i], T[i], r[i], sigma[i], q[i], is_call[i]
        );
    }
}

EXPORT_API double calculate_execution_price_and_friction(
    int asset_class,
    int side,           // 1 = BUY, -1 = SELL
    int order_type,     // 0 = MARKET, 1 = LIMIT, 2 = STOP
    double mid_price,
    double bid_price,
    double ask_price,
    double atr,
    double volume,
    double open_interest,
    int num_legs,
    double quantity,
    double* out_slippage_usd,
    double* out_fee_usd
) {
    if (mid_price <= 0.0) mid_price = (bid_price > 0.0 && ask_price > 0.0) ? 0.5 * (bid_price + ask_price) : 100.0;
    if (bid_price <= 0.0) bid_price = mid_price * 0.9995;
    if (ask_price <= 0.0) ask_price = mid_price * 1.0005;

    double fill_price = mid_price;
    double slippage_per_unit = 0.0;
    double fee_usd = 0.0;

    switch (asset_class) {
        case 0: { // US_EQUITY
            // Dynamic slippage: 1.5 bps base + volatility factor * (ATR / Price)
            double vol_mult = (mid_price > 0.0 && atr > 0.0) ? (0.05 * (atr / mid_price)) : 0.0002;
            double base_slip_bps = 0.00015; // 1.5 bps
            slippage_per_unit = mid_price * (base_slip_bps + vol_mult);

            if (order_type == 1) { // LIMIT: no adverse slippage on fill
                fill_price = (side == 1) ? bid_price : ask_price;
                slippage_per_unit = 0.0;
            } else { // MARKET / STOP
                fill_price = (side == 1) ? (ask_price + slippage_per_unit) : (bid_price - slippage_per_unit);
            }

            // Commission: $0.005/share (min $1.00, max 0.5% trade value)
            double base_comm = std::max(1.0, quantity * 0.005);
            base_comm = std::min(base_comm, 0.005 * quantity * mid_price);
            
            // Regulatory fees on sell side (SEC + FINRA TAF)
            double reg_fee = 0.0;
            if (side == -1) {
                reg_fee = 0.000166 * quantity + 0.0000278 * quantity * mid_price;
            }
            fee_usd = base_comm + reg_fee;
            break;
        }

        case 1: { // US_OPTION
            // Never fill at Mid!
            // Fill at Mid +/- k * (Ask - Bid) / 2.0
            // k in [0.35, 0.85] scaling inversely with liquidity (Open Interest & Volume)
            double total_liquidity = std::max(0.0, open_interest + volume);
            double k = 0.35 + 0.50 / (1.0 + std::log1p(total_liquidity / 100.0));
            k = std::clamp(k, 0.35, 0.85);

            double half_spread = 0.5 * std::max(0.01, ask_price - bid_price);
            double spread_cross = k * half_spread;

            // Multi-leg spread penalty (15% - 25% extra friction per additional leg)
            if (num_legs > 1) {
                double leg_penalty = (num_legs - 1) * 0.20 * half_spread;
                spread_cross += leg_penalty;
            }

            if (order_type == 1) { // Limit order resting
                fill_price = (side == 1) ? (mid_price - 0.25 * half_spread) : (mid_price + 0.25 * half_spread);
                slippage_per_unit = 0.0;
            } else {
                fill_price = (side == 1) ? (mid_price + spread_cross) : (mid_price - spread_cross);
                slippage_per_unit = spread_cross;
            }

            // Fee: $0.65 per contract per leg + $0.04 exchange clearing fee
            int legs = std::max(1, num_legs);
            fee_usd = (0.65 + 0.04) * legs * quantity;
            break;
        }

        case 2: { // CRYPTO
            if (order_type == 1) { // MAKER
                fill_price = (side == 1) ? bid_price : ask_price;
                slippage_per_unit = 0.0;
                fee_usd = (0.0004) * quantity * fill_price; // 4 bps maker
            } else { // TAKER
                double vol_impact = (volume > 0.0) ? (0.0001 * (quantity / volume)) : 0.0002;
                double taker_slip = mid_price * (0.0003 + vol_impact);
                fill_price = (side == 1) ? (ask_price + taker_slip) : (bid_price - taker_slip);
                slippage_per_unit = taker_slip;
                fee_usd = (0.0008) * quantity * fill_price; // 8 bps taker
            }
            break;
        }

        case 3: { // COMMODITY (Futures / ETFs)
            // 1.5 ticks base slippage
            double tick_size = (mid_price > 1000.0) ? 0.10 : 0.01; // e.g. Gold ($0.10) vs Crude ($0.01)
            slippage_per_unit = 1.5 * tick_size;
            if (order_type == 1) {
                fill_price = (side == 1) ? bid_price : ask_price;
                slippage_per_unit = 0.0;
            } else {
                fill_price = (side == 1) ? (ask_price + slippage_per_unit) : (bid_price - slippage_per_unit);
            }
            fee_usd = 2.50 * quantity; // All-in exchange + clearing fee per contract
            break;
        }

        default:
            fill_price = mid_price;
            slippage_per_unit = 0.0;
            fee_usd = 1.0;
            break;
    }

    if (out_slippage_usd) {
        *out_slippage_usd = slippage_per_unit * quantity;
    }
    if (out_fee_usd) {
        *out_fee_usd = fee_usd;
    }

    return fill_price;
}

EXPORT_API void aggregate_portfolio_risk(
    const double* spots,
    const double* strikes,
    const double* ttes,
    const double* rates,
    const double* ivs,
    const int* is_calls,
    const double* quantities,
    const double* entry_prices,
    int n,
    double* out_total_unrealized_pnl,
    double* out_net_delta,
    double* out_net_gamma,
    double* out_net_vega,
    double* out_net_theta
) {
    double total_pnl = 0.0;
    double net_delta = 0.0;
    double net_gamma = 0.0;
    double net_vega = 0.0;
    double net_theta = 0.0;

    for (int i = 0; i < n; ++i) {
        double S = spots[i];
        double K = strikes[i];
        double T = ttes[i];
        double r = rates[i];
        double iv = ivs[i];
        int is_call = is_calls[i];
        double qty = quantities[i];
        double entry = entry_prices[i];

        OptionGreeks g = calculate_black_scholes(S, K, T, r, iv, 0.0, is_call);
        
        // Multiplier: 100 for equity options
        double multiplier = (K > 0.0) ? 100.0 : 1.0;
        double current_val = g.price * multiplier;
        double entry_val = entry * multiplier;
        double pnl = (current_val - entry_val) * qty;

        total_pnl += pnl;
        net_delta += (g.delta * multiplier) * qty;
        net_gamma += (g.gamma * multiplier) * qty;
        net_vega  += (g.vega * multiplier) * qty;
        net_theta += (g.theta * multiplier) * qty;
    }

    if (out_total_unrealized_pnl) *out_total_unrealized_pnl = total_pnl;
    if (out_net_delta) *out_net_delta = net_delta;
    if (out_net_gamma) *out_net_gamma = net_gamma;
    if (out_net_vega)  *out_net_vega  = net_vega;
    if (out_net_theta) *out_net_theta = net_theta;
}

}
