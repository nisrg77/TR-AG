"""
Main CLI Entrypoint:
Antigravity Multi-Asset & Options AI Quant Trading System.

Usage:
    python main.py --mode backtest
    python main.py --mode live-paper
    python main.py --mode dashboard
    python main.py --mode compile-cpp
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def compile_cpp():
    print("Compiling native C++ microstructure and options kernel with -O3...")
    cpp_src = ROOT_DIR / "src" / "cpp" / "quant_core.cpp"
    out_lib = ROOT_DIR / "src" / "cpp" / ("quant_core.dll" if sys.platform == "win32" else "quant_core.so")
    cmd = ["g++", "-O3", "-shared", "-fPIC", str(cpp_src), "-o", str(out_lib)]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print(f"✅ C++ kernel compiled successfully -> {out_lib}")
    else:
        print("❌ C++ compilation failed.")
    return res.returncode

def main():
    parser = argparse.ArgumentParser(description="Antigravity Multi-Asset & Options AI Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "live-paper", "dashboard", "compile-cpp"],
        default="backtest",
        help="System execution mode"
    )
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "NVDA", "GLD", "BTC/USDT"], help="Target universe")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Polling interval for live paper trading")
    args = parser.parse_args()

    if args.mode == "compile-cpp":
        sys.exit(compile_cpp())

    elif args.mode == "backtest":
        from runners.run_backtest import run_backtest_simulation
        run_backtest_simulation(symbols=args.symbols)

    elif args.mode == "live-paper":
        from runners.run_live_paper import run_live_paper_loop
        run_live_paper_loop(symbols=args.symbols, poll_interval_seconds=args.poll_interval)

    elif args.mode == "dashboard":
        dash_path = ROOT_DIR / "dashboard" / "app.py"
        print(f"Launching Streamlit Quant Dashboard on http://localhost:{args.port}...")
        subprocess.run(["streamlit", "run", str(dash_path), "--server.port", str(args.port), "--server.headless", "true"])

if __name__ == "__main__":
    main()
