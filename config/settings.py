"""
Global Configuration & Environment Settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "config" / "secrets.env")

# Database & Data Paths
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "oms.db"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Virtual Initial Capital
STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", "100000.0")) # $100,000 USD default

# Market Data & News API Keys (Optional Read-Only)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Server & Dashboard Settings
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
