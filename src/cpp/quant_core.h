#ifndef QUANT_CORE_H
#define QUANT_CORE_H

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

#include <cstdint>

extern "C" {

struct OptionGreeks {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// 1. Black-Scholes Pricing and Greeks (Single Contract)
// is_call: 1 for Call, 0 for Put
EXPORT_API OptionGreeks calculate_black_scholes(
    double S,       // Current Spot Price
    double K,       // Strike Price
    double T,       // Time to Expiry (in years, e.g. 30/365.0)
    double r,       // Risk-free interest rate (e.g. 0.045)
    double sigma,   // Implied Volatility (e.g. 0.25)
    double q,       // Dividend yield (e.g. 0.015)
    int is_call     // 1 for Call, 0 for Put
);

// 2. Implied Volatility Solver via Newton-Raphson / Bisection
EXPORT_API double calculate_implied_volatility(
    double market_price,
    double S,
    double K,
    double T,
    double r,
    double q,
    int is_call,
    double initial_guess
);

// 3. Batch Option Chain Pricing & Greeks
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
);

// 4. Execution Friction & Fill Price Calculation
EXPORT_API double calculate_execution_price_and_friction(
    int asset_class,        // 0: US_EQUITY, 1: US_OPTION, 2: CRYPTO, 3: COMMODITY
    int side,               // 1: BUY, -1: SELL
    int order_type,         // 0: MARKET, 1: LIMIT, 2: STOP
    double mid_price,
    double bid_price,
    double ask_price,
    double atr,
    double volume,
    double open_interest,
    int num_legs,           // e.g. 1 for single option, 2 for vertical, 4 for condor
    double quantity,
    double* out_slippage_usd,
    double* out_fee_usd
);

// 5. Portfolio Net Greeks & Mark-to-Market Vector Aggregator
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
);

}

#endif // QUANT_CORE_H
