"""
Self-Hosted OMS Database Layer (SQLite / DuckDB compatible)
Stores signals, orders, fills/executions, positions, and account ledger.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from config.settings import DATABASE_PATH, STARTING_EQUITY

def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Path = DATABASE_PATH, initial_equity: float = STARTING_EQUITY) -> None:
    """Initializes tables and seeds initial ledger record if not present."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()

        # 1. Signals Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            underlying TEXT NOT NULL,
            direction TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            iv_rank REAL,
            entry_limit_price REAL,
            stop_loss_price REAL,
            take_profit_price REAL,
            max_risk_usd REAL,
            estimated_friction_usd REAL,
            contracts_json TEXT,
            rationale TEXT,
            status TEXT DEFAULT 'NEW',
            created_at TEXT NOT NULL
        );
        """)

        # 2. Orders Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            signal_id TEXT,
            timestamp_utc TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,             -- BUY, SELL
            order_type TEXT NOT NULL,       -- MARKET, LIMIT, STOP
            quantity REAL NOT NULL,
            limit_price REAL,
            stop_price REAL,
            legs_json TEXT,                 -- For multi-leg options
            status TEXT NOT NULL,           -- PENDING, FILLED, CANCELLED, REJECTED, EXPIRED
            reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
        );
        """)

        # 3. Executions (Fills) Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS executions (
            fill_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            fill_timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            mid_price_at_order REAL,
            fill_price REAL NOT NULL,
            spread_cost_usd REAL DEFAULT 0.0,
            slippage_cost_usd REAL DEFAULT 0.0,
            commission_fee_usd REAL DEFAULT 0.0,
            total_friction_usd REAL DEFAULT 0.0,
            legs_json TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        """)

        # 4. Positions Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            position_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            underlying TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            side TEXT NOT NULL,             -- LONG, SHORT
            strategy_type TEXT NOT NULL,
            quantity REAL NOT NULL,
            avg_entry_price REAL NOT NULL,
            current_mark_price REAL NOT NULL,
            unrealized_pnl REAL DEFAULT 0.0,
            realized_pnl REAL DEFAULT 0.0,
            stop_loss REAL,
            take_profit REAL,
            delta REAL DEFAULT 0.0,
            gamma REAL DEFAULT 0.0,
            vega REAL DEFAULT 0.0,
            theta REAL DEFAULT 0.0,
            legs_json TEXT,
            status TEXT NOT NULL,           -- OPEN, CLOSED, EXPIRED
            opened_at TEXT NOT NULL,
            closed_at TEXT
        );
        """)

        # 5. Account Ledger Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS account_ledger (
            ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            cash_balance REAL NOT NULL,
            reserved_margin REAL NOT NULL,
            total_equity REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            cumulative_friction_usd REAL NOT NULL,
            note TEXT
        );
        """)

        # Initialize ledger if empty
        cur.execute("SELECT COUNT(*) FROM account_ledger")
        if cur.fetchone()[0] == 0:
            now_iso = datetime.now(timezone.utc).isoformat()
            cur.execute("""
            INSERT INTO account_ledger (
                timestamp_utc, cash_balance, reserved_margin, total_equity,
                realized_pnl, unrealized_pnl, cumulative_friction_usd, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now_iso, initial_equity, 0.0, initial_equity, 0.0, 0.0, 0.0, "Initial Account Genesis"))

        conn.commit()

def reset_db(db_path: Path = DATABASE_PATH, initial_equity: float = STARTING_EQUITY) -> None:
    """Safely wipes and reinitializes database for clean backtest or fresh deployment."""
    if db_path.exists():
        db_path.unlink()
    init_db(db_path, initial_equity)
