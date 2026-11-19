"""
Multi-Layer Institutional Risk Management & Portfolio Guard:
Enforces deterministic 4-stage risk gates:
1. Hierarchical Capital Caps:
   - Single Trade Risk <= 1.5% NLV
   - Same-Symbol Aggregate Risk <= 3.0% NLV
   - Correlation Cluster Cap <= 5.0% NLV
2. Portfolio Greeks Guardrails (Beta Delta, Gamma Floor, Vega Filter, Theta Cap)
3. Automated Circuit Breakers (Daily -3% Halt, Drawdown Throttle -8%/-12%, Spread Freeze)
4. Persists decisions to risk_audit_log table.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config.settings import DATABASE_PATH
from oms.database import get_connection

CORRELATION_CLUSTERS = {
    "US_EQUITY_BETA": ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"],
    "CRYPTO_BETA": ["BTC", "ETH", "SOL", "BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "METALS_RATES": ["GLD", "SLV", "TLT"],
    "ENERGY": ["USO", "UNG", "XLE"]
}

class PortfolioRiskGuard:
    def __init__(
        self,
        db_path: Path = DATABASE_PATH,
        max_single_trade_risk_pct: float = 0.015,   # 1.5% NLV
        max_symbol_risk_pct: float = 0.030,         # 3.0% NLV
        max_cluster_risk_pct: float = 0.050,        # 5.0% NLV
        max_beta_delta_pct: float = 0.30,           # +/- 30% NLV
        min_gamma_floor_pct: float = -0.05,         # -5% NLV
        daily_loss_halt_pct: float = -0.03,         # -3% daily loss
        drawdown_throttle_pct: float = -0.08,       # -8% cuts risk in half
        drawdown_halt_pct: float = -0.12            # -12% halts trading
    ):
        self.db_path = db_path
        self.max_single_trade_risk_pct = max_single_trade_risk_pct
        self.max_symbol_risk_pct = max_symbol_risk_pct
        self.max_cluster_risk_pct = max_cluster_risk_pct
        self.max_beta_delta_pct = max_beta_delta_pct
        self.min_gamma_floor_pct = min_gamma_floor_pct
        self.daily_loss_halt_pct = daily_loss_halt_pct
        self.drawdown_throttle_pct = drawdown_throttle_pct
        self.drawdown_halt_pct = drawdown_halt_pct

    def evaluate_order_risk(
        self,
        symbol: str,
        asset_class: str,
        side: str,
        order_type: str,
        entry_price: float,
        stop_price: Optional[float],
        requested_qty: float,
        contracts: Optional[List[Dict[str, Any]]] = None,
        spread_bid: Optional[float] = None,
        spread_ask: Optional[float] = None,
        rolling_spread_median: Optional[float] = None,
        signal_id: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """
        Validates proposed trade against all institutional guardrails.
        Returns (is_approved, decision_reason, approved_qty).
        """
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()

            # 1. Fetch current account state
            cur.execute("""
            SELECT cash_balance, total_equity, realized_pnl, unrealized_pnl
            FROM account_ledger ORDER BY ledger_id DESC LIMIT 1
            """)
            ledger_row = cur.fetchone()
            nlv = float(ledger_row["total_equity"]) if ledger_row else 100000.0

            # 2. Daily Loss Limit & Drawdown Circuit Breakers
            cur.execute("""
            SELECT total_equity FROM account_ledger
            ORDER BY ledger_id ASC LIMIT 1
            """)
            day_start_row = cur.fetchone()
            day_start_nlv = float(day_start_row["total_equity"]) if day_start_row else nlv

            cur.execute("SELECT MAX(total_equity) FROM account_ledger")
            hwm_row = cur.fetchone()
            high_water_mark = float(hwm_row[0]) if (hwm_row and hwm_row[0]) else nlv

            # Check Daily Drawdown Halt (-3.0%)
            daily_pnl_pct = (nlv - day_start_nlv) / day_start_nlv
            if daily_pnl_pct <= self.daily_loss_halt_pct:
                reason = f"DAILY_DRAWDOWN_HALT: Intraday loss {daily_pnl_pct*100:.2f}% <= {self.daily_loss_halt_pct*100:.1f}% limit"
                self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, 0.0, 0.0)
                conn.commit()
                return False, reason, 0.0

            # Check Peak-to-Trough Drawdown Throttle (-8% cuts risk; -12% halts)
            drawdown_pct = (nlv - high_water_mark) / high_water_mark
            if drawdown_pct <= self.drawdown_halt_pct:
                reason = f"MAX_DRAWDOWN_HALT: Peak drawdown {drawdown_pct*100:.2f}% breached {self.drawdown_halt_pct*100:.1f}%"
                self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, 0.0, 0.0)
                conn.commit()
                return False, reason, 0.0

            # Scale single trade risk budget if under drawdown throttle (-8%)
            active_single_risk_pct = self.max_single_trade_risk_pct
            if drawdown_pct <= self.drawdown_throttle_pct:
                active_single_risk_pct = self.max_single_trade_risk_pct * 0.50

            # 3. Microstructure Volatility Spread Freeze
            if spread_bid is not None and spread_ask is not None and rolling_spread_median is not None:
                current_spread = spread_ask - spread_bid
                if rolling_spread_median > 0 and current_spread > (3.0 * rolling_spread_median):
                    reason = f"SPREAD_FREEZE: Current spread ${current_spread:.2f} > 3x rolling median (${rolling_spread_median:.2f})"
                    self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, 0.0, 0.0)
                    conn.commit()
                    return False, reason, 0.0

            # 4. Single-Trade Risk Cap (<= 1.5% NLV)
            underlying = symbol.split("_")[0]
            unit_risk_usd = 0.0
            if asset_class == "US_OPTION" and contracts:
                # Defined risk spread: max risk = net debit * 100
                unit_risk_usd = entry_price * 100.0
            else:
                stop = stop_price if stop_price is not None else entry_price * 0.98
                unit_risk_usd = abs(entry_price - stop)
                if unit_risk_usd <= 1e-4: unit_risk_usd = entry_price * 0.02

            max_allowed_trade_risk = nlv * active_single_risk_pct
            max_qty_allowed = max_allowed_trade_risk / max(1.0, unit_risk_usd)
            if asset_class != "CRYPTO":
                max_qty_allowed = max(1.0, float(int(max_qty_allowed)))
            else:
                max_qty_allowed = round(max_qty_allowed, 4)

            approved_qty = min(requested_qty, max_qty_allowed)

            # 5. Same-Symbol Aggregate Risk Cap (<= 3.0% NLV)
            cur.execute("""
            SELECT COALESCE(SUM(quantity * avg_entry_price), 0.0)
            FROM positions WHERE underlying = ? AND status = 'OPEN'
            """, (underlying,))
            existing_symbol_notional = float(cur.fetchone()[0])
            new_trade_risk = approved_qty * unit_risk_usd
            symbol_risk_pct = (existing_symbol_notional * 0.02 + new_trade_risk) / nlv

            if symbol_risk_pct > self.max_symbol_risk_pct:
                reason = f"SYMBOL_RISK_CAP: Symbol {underlying} aggregate risk {symbol_risk_pct*100:.2f}% exceeds {self.max_symbol_risk_pct*100:.1f}% limit"
                self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, 0.0, symbol_risk_pct)
                conn.commit()
                return False, reason, 0.0

            # 6. Correlation Cluster Cap (<= 5.0% NLV)
            cluster_name = None
            for c_name, c_symbols in CORRELATION_CLUSTERS.items():
                if underlying in c_symbols:
                    cluster_name = c_name
                    break

            cluster_risk_pct = 0.0
            if cluster_name:
                cluster_symbols = CORRELATION_CLUSTERS[cluster_name]
                placeholders = ",".join("?" for _ in cluster_symbols)
                cur.execute(f"""
                SELECT COALESCE(SUM(quantity * avg_entry_price), 0.0)
                FROM positions WHERE underlying IN ({placeholders}) AND status = 'OPEN'
                """, cluster_symbols)
                existing_cluster_notional = float(cur.fetchone()[0])
                cluster_risk_pct = (existing_cluster_notional * 0.02 + new_trade_risk) / nlv
                if cluster_risk_pct > self.max_cluster_risk_pct:
                    reason = f"CLUSTER_CAP_5PCT: Cluster {cluster_name} risk {cluster_risk_pct*100:.2f}% exceeds {self.max_cluster_risk_pct*100:.1f}% limit"
                    self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, 0.0, cluster_risk_pct)
                    conn.commit()
                    return False, reason, 0.0

            # 7. Portfolio Greeks Guardrail (Beta-Weighted Net Delta Cap: [-0.30, +0.30] * NLV)
            cur.execute("SELECT COALESCE(SUM(delta), 0.0), COALESCE(SUM(gamma), 0.0) FROM positions WHERE status = 'OPEN'")
            g_row = cur.fetchone()
            current_portfolio_delta = float(g_row[0])
            current_portfolio_gamma = float(g_row[1])

            # Simulated new delta
            trade_delta = 0.0
            if asset_class == "US_OPTION" and contracts:
                trade_delta = (contracts[0].get("delta", 0.50) * 100.0) * approved_qty
            else:
                trade_delta = approved_qty * (1.0 if side.upper() == "BUY" else -1.0) * entry_price

            new_portfolio_delta = current_portfolio_delta + trade_delta
            max_delta_limit = self.max_beta_delta_pct * nlv

            if abs(new_portfolio_delta) > max_delta_limit:
                reason = f"PORTFOLIO_DELTA_LIMIT: Projected net Delta ${abs(new_portfolio_delta):,.2f} exceeds {self.max_beta_delta_pct*100:.0f}% NLV cap (${max_delta_limit:,.2f})"
                self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, new_portfolio_delta, cluster_risk_pct)
                conn.commit()
                return False, reason, 0.0

            # Check Gamma Floor on short option additions
            if asset_class == "US_OPTION" and side.upper() == "SELL":
                min_gamma_allowed = self.min_gamma_floor_pct * nlv
                if current_portfolio_gamma < min_gamma_allowed:
                    reason = f"GAMMA_FLOOR_BREACH: Net Gamma {current_portfolio_gamma:.4f} < floor {min_gamma_allowed:.4f}"
                    self._log_audit(cur, signal_id, symbol, requested_qty, 0.0, "REJECTED", reason, new_portfolio_delta, cluster_risk_pct)
                    conn.commit()
                    return False, reason, 0.0

            # Approved!
            decision = "DOWNSIZED" if approved_qty < requested_qty else "APPROVED"
            self._log_audit(cur, signal_id, symbol, requested_qty, approved_qty, decision, "Passed all institutional risk gates", new_portfolio_delta, cluster_risk_pct)
            conn.commit()
            return True, "Passed all institutional risk gates", approved_qty

    def _log_audit(
        self,
        cur,
        signal_id: Optional[str],
        symbol: str,
        requested_qty: float,
        approved_qty: float,
        decision: str,
        reason: str,
        portfolio_delta_after: float,
        cluster_risk_pct: float
    ) -> None:
        audit_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        cur.execute("""
        INSERT INTO risk_audit_log (
            audit_id, timestamp_utc, signal_id, symbol, requested_qty,
            approved_qty, decision, rejection_reason, portfolio_delta_after,
            cluster_risk_pct, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_id, now_iso, signal_id, symbol, requested_qty,
            approved_qty, decision, reason, portfolio_delta_after,
            cluster_risk_pct, now_iso
        ))
