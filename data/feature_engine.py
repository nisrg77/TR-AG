"""
Feature & Regime Engine:
Calculates Technicals, Realized Volatility (RV), ATR, IV Rank, and Regime Indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class FeatureEngine:
    def __init__(self):
        pass

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes raw OHLCV DataFrame and computes technical, volatility, and trend indicators.
        """
        df = df.copy()

        # Returns & Volatility
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["realized_vol_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252 * 6.5) # Annualized hourly

        # Moving Averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI (14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Average True Range (ATR 14)
        h_l = df["high"] - df["low"]
        h_pc = (df["high"] - df["close"].shift(1)).abs()
        l_pc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # Bollinger Bands
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = df["sma_20"] + (2 * std_20)
        df["bb_lower"] = df["sma_20"] - (2 * std_20)

        # Simulated IV Rank (normalized realized vol percentile)
        rolling_min_vol = df["realized_vol_20"].rolling(100, min_periods=20).min()
        rolling_max_vol = df["realized_vol_20"].rolling(100, min_periods=20).max()
        df["iv_rank"] = ((df["realized_vol_20"] - rolling_min_vol) / (rolling_max_vol - rolling_min_vol + 1e-6) * 100).clip(0, 100)

        # Regime Tag
        df["trend_regime"] = "NEUTRAL"
        bullish_mask = (df["close"] > df["sma_20"]) & (df["macd"] > df["macd_signal"]) & (df["rsi_14"] > 50)
        bearish_mask = (df["close"] < df["sma_20"]) & (df["macd"] < df["macd_signal"]) & (df["rsi_14"] < 50)
        df.loc[bullish_mask, "trend_regime"] = "BULLISH"
        df.loc[bearish_mask, "trend_regime"] = "BEARISH"

        df.bfill(inplace=True)
        return df
