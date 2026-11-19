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
3. **Multi-Agent Synthesis Layer (`agents/`)**:
   - **Quant / Technical Forecaster Agent**: Evaluates Amazon Chronos probabilistic return quantiles and trend regimes.
   - **Catalyst & Sentiment Agent**: Evaluates news and macro headlines (FinBERT / Lexicon).
   - **Options Strategist Agent**: Selects Bull Call / Bear Put debit spreads when IV Rank $< 35$, or credit spreads / condors when IV Rank $> 50$.
   - **Deterministic Risk Gate**: Hard-coded liquidity filters, fee-drag rejection ($< 20\%$ profit drag), and $2\%$ maximum portfolio risk per trade.
4. **Interactive Streamlit Terminal (`dashboard/app.py`)**:
   - Real-time blotter, open positions, live portfolio Greeks, Gross vs. Net equity curves, and signal audit logs.

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
├── oms/                       # <--- Self-Hosted Local OMS
│   ├── database.py            # SQLite schema (orders, executions, positions, ledger, signals)
│   ├── execution_costs.py     # C++ accelerated realistic friction calculator
│   ├── engine.py              # Order router, queue-aware limit & gap-aware stop fills
│   ├── portfolio.py           # Margin, buying power & mark-to-market calculator
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
