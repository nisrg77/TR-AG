# Multi-Asset & Options AI Trading Signal Predictor
### Backtesting & Self-Hosted Order Management System (OMS) with C++ Accelerated Microstructure

A modular, institutional-grade quant trading engine covering **US Equities, Commodities, Crypto, and Options**. It generates structured trade signals using prebuilt forecasting/classification models and multi-agent synthesis, validates them via historical backtesting, and executes **live paper trading through a self-hosted local OMS**.

---

## 🚀 Key Architectural Highlights

1. **Native C++17 Greeks & Microstructure Engine (`src/cpp/quant_core.cpp`)**:
   - Vectorized Black-Scholes pricing & full Greeks ($\Delta, \Gamma, \mathcal{V}, \Theta, \rho$) in **1.69 microseconds/call**.
   - High-precision Newton-Raphson & bisection Implied Volatility (IV) solver.
   - Realistic microstructure friction: dynamic ATR volatility slippage, liquidity-scaled bid/ask crossing ($k \in [0.35, 0.85]$), multi-leg spread penalty (15%–25% extra cost per additional leg), maker/taker fees, and exchange regulatory fees.
2. **Unified Self-Hosted Local OMS (`oms/`)**:
   - Backed by SQLite (`data/oms.db`).
   - Order Router & State Machine: `PENDING` $\rightarrow$ `FILLED` / `CANCELLED` / `REJECTED`.
   - Realistic queue-aware limit fills and gap-aware stop-loss fills.
   - Multi-leg options lifecycle engine with mark-to-market and automatic 16:00 ET expiration cash settlement.
   - Net Liquidation Value (NLV) ledger accounting with gross vs. net equity tracking.
3. **Multi-Layer Institutional Risk Management (`risk/`)**:
   - Hierarchical Capital & Correlation Caps: Single Trade (1.5% NLV), Same-Symbol (3.0% NLV), Correlation Cluster Cap (5.0% NLV).
   - Portfolio Greeks Guardrails: Beta-Weighted Delta ($\pm 30\%$ NLV), Gamma Floor ($-0.05 \times \text{NLV}$), Vega Filter, Theta Cap.
   - Automated Circuit Breakers: Daily Loss Halt ($-3.0\%$), Peak Drawdown Throttle ($-8.0\% / -12.0\%$), Microstructure Volatility Spread Freeze.
   - 4-Tier Pyramiding Engine: Winner-only scaling (anti-martingale), $0.75 \times \text{ATR}$ spacing, decaying lot sizing ($1.0\times / 0.75\times / 0.50\times / 0.25\times$), and synchronized stop ratcheting.
4. **Leaving a "Runner" — Core (70%) + Runner (30%) Scaling (`oms/tranche_manager.py`)**:
   - Decomposes every filled order into Core (70%) and Runner (30%) child tranches.
   - Core takes profit at $+1.5R$, immediately ratcheting Runner SL to Entry + Fees (Auto-Breakeven).
   - Runner trails with Chandelier ATR stop ($\text{Highest High} - 2.0 \times \text{ATR}_{14}$) and options DTE time decay rules.
5. **Liquidity & Order Book Microstructure (`features/microstructure.py` & `oms/book_simulator.py`)**:
   - L2 Order Book Imbalance (OBI) across top 5 depth levels. Smart routing downgrades aggressive `MARKET` orders to passive `LIMIT` when $\text{OBI} < -0.40$.
   - Institutional Liquidity Sweep (Stop-Hunt) detector.
   - Multi-level order book walking with 10% depth participation cap.
6. **Interactive Streamlit Terminal (`dashboard/app.py`)**:
   - 7 tabbed views: Equity Curve, Open Positions, Core & Runner Tranches, Order Blotter, Executions, AI Signals, and Risk & Circuit Breaker Audit Logs.

---

## 📁 System Architecture & Directory Layout

```text
TR-AG/
├── config/
│   ├── universe.yaml          # Multi-asset universe & risk parameters
│   ├── execution_costs.yaml   # Realistic slippage bps, option spread weights & fees
│   ├── settings.py            # Global paths & starting capital
│   └── secrets.env.example    # API keys template
├── src/
│   └── cpp/
│       ├── quant_core.h       # C++ header for options & friction math
│       ├── quant_core.cpp     # C++17 native kernel (compiled with -O3)
│       └── quant_bridge.py    # ctypes Python bridge with numpy fallback
├── data/
│   ├── market_fetcher.py      # Read-only OHLCV & option chains (Equities, Crypto, Commodities)
│   └── feature_engine.py      # Technicals, Realized Volatility, ATR & IV Rank
├── models/
│   ├── chronos_forecaster.py  # Probabilistic return quantile forecaster
│   └── sentiment_scorer.py    # Financial sentiment scorer
├── agents/
│   ├── graph.py               # Multi-agent synthesis pipeline
│   ├── options_selector.py    # Strategy & strike/expiry selection logic
│   └── risk_guard.py          # Deterministic liquidity, risk & fee-drag gate
├── risk/                      # <--- Multi-Layer Institutional Risk & Pyramiding
│   ├── portfolio_guard.py     # Capital caps, Greeks guardrails & circuit breakers
│   └── pyramid_manager.py     # 4-tier anti-martingale pyramiding & stop ratcheting
├── features/
│   └── microstructure.py      # L2 Order Book Imbalance (OBI) & Liquidity Sweeps
├── oms/                       # <--- Self-Hosted Local OMS
│   ├── database.py            # SQLite schema (orders, executions, positions, tranches, ledger)
│   ├── execution_costs.py     # C++ accelerated realistic friction calculator
│   ├── engine.py              # Order router, queue-aware limit & gap-aware stop fills
│   ├── portfolio.py           # Margin, buying power & mark-to-market calculator
│   ├── tranche_manager.py     # Core (70%) + Runner (30%) & auto-breakeven
│   ├── book_simulator.py      # L2 order book walking & participation caps
│   └── options_lifecycle.py   # Multi-leg spread pricing & expiration settlement
├── runners/
│   ├── run_backtest.py        # Historical replay backtest runner
│   └── run_live_paper.py      # Real-time live paper trading loop
├── dashboard/
│   └── app.py                 # Streamlit blotter, Greeks & Gross vs. Net equity terminal
├── tests/
│   └── test_system_verification.py # Comprehensive 6-layer test suite
└── main.py                    # Unified CLI entrypoint
```

---

## ⚡ Quickstart & Execution Commands

### 1. Recompile Native C++ Kernel
```bash
python main.py --mode compile-cpp
# Or directly with g++:
g++ -O3 -shared -fPIC src/cpp/quant_core.cpp -o src/cpp/quant_core.dll # (Windows)
g++ -O3 -shared -fPIC src/cpp/quant_core.cpp -o src/cpp/quant_core.so  # (Linux)
```

### 2. Run Comprehensive 6-Layer Verification Test
```bash
python tests/test_system_verification.py
```

### 3. Run Historical Backtest Replay
```bash
python main.py --mode backtest --symbols SPY QQQ NVDA GLD BTC/USDT
```

### 4. Run Real-Time Live Paper Trading Loop
```bash
python main.py --mode live-paper --poll-interval 3.0
```

### 5. Launch Streamlit Quant Dashboard
```bash
python main.py --mode dashboard --port 8501
```
Open `http://localhost:8501` in your browser.
