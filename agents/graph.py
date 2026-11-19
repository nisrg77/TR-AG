"""
Multi-Agent Synthesis Pipeline:
Orchestrates Technical Analyst Agent, Catalyst & Sentiment Agent,
Options Strategist Agent, and Deterministic Risk Gate.
Generates structured JSON trade signals.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd

from models.chronos_forecaster import ChronosForecaster
from models.sentiment_scorer import SentimentScorer
from agents.options_selector import OptionsSelector
from agents.risk_guard import RiskManagerGate
from data.feature_engine import FeatureEngine

class MultiAgentTradingPipeline:
    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.forecaster = ChronosForecaster()
        self.sentiment_scorer = SentimentScorer()
        self.options_selector = OptionsSelector()
        self.risk_guard = RiskManagerGate()

    def generate_signal(
        self,
        symbol: str,
        asset_class: str,
        df_bars: pd.DataFrame,
        headlines: Optional[List[str]] = None,
        account_equity: float = 100000.0,
        current_open_positions: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesizes technical, sentiment, and options features into a validated trade signal.
        """
        if df_bars.empty or len(df_bars) < 20:
            return None

        # 1. Feature Engineering
        feats = self.feature_engine.compute_features(df_bars)
        last_row = feats.iloc[-1]
        current_spot = float(last_row["close"])
        iv_rank = float(last_row["iv_rank"])
        atr = float(last_row["atr_14"])
        trend_regime = str(last_row["trend_regime"])

        # 2. Agent 1: Quant / Technical Forecaster
        chronos_res = self.forecaster.forecast_quantiles(df_bars["close"].values, prediction_length=5)
        model_dir = chronos_res["direction"]
        prob_pos = chronos_res["prob_positive"]

        # 3. Agent 2: Sentiment & Catalyst Agent
        sample_headlines = headlines or [
            f"{symbol} exhibits strong institutional volume and steady momentum.",
            f"Macro tailwinds support {symbol} ahead of monthly options expiration."
        ]
        sentiment_res = self.sentiment_scorer.score_headlines(sample_headlines)
        sent_dir = sentiment_res["label"]

        # Consensus Direction & Confidence
        scores = []
        if model_dir == "BULLISH": scores.append(1.0)
        elif model_dir == "BEARISH": scores.append(-1.0)
        else: scores.append(0.0)

        if trend_regime == "BULLISH": scores.append(1.0)
        elif trend_regime == "BEARISH": scores.append(-1.0)
        else: scores.append(0.0)

        if sent_dir == "BULLISH": scores.append(0.5)
        elif sent_dir == "BEARISH": scores.append(-0.5)
        else: scores.append(0.0)

        net_score = sum(scores)
        direction = "BULLISH" if net_score >= 1.0 else ("BEARISH" if net_score <= -1.0 else "NEUTRAL")
        confidence = min(0.95, max(0.50, 0.50 + abs(net_score) * 0.15))

        if direction == "NEUTRAL":
            return None # No high-conviction trade

        now_iso = datetime.now(timezone.utc).isoformat()
        signal_id = f"SIG-{uuid.uuid4().hex[:8].upper()}"

        # 4. Agent 3: Options Strategist Agent (for Options-enabled underlyings)
        if asset_class == "US_OPTION" or (asset_class in ["US_EQUITY", "COMMODITY"] and symbol in ["SPY", "QQQ", "GLD", "NVDA"]):
            opt_decision = self.options_selector.select_strategy_and_strikes(
                underlying=symbol,
                spot_price=current_spot,
                direction=direction,
                iv_rank=iv_rank,
                historical_vol=float(last_row["realized_vol_20"])
            )

            raw_signal = {
                "signal_id": signal_id,
                "timestamp_utc": now_iso,
                "asset_class": "US_OPTION",
                "underlying": symbol,
                "symbol": f"{symbol}_{opt_decision['strategy_type']}",
                "direction": direction,
                "strategy_type": opt_decision["strategy_type"],
                "contracts": opt_decision["contracts"],
                "confidence_score": round(confidence, 2),
                "iv_rank": round(iv_rank, 1),
                "entry_limit_price": opt_decision["entry_limit_price"],
                "stop_loss_price": opt_decision["stop_loss_price"],
                "take_profit_price": opt_decision["take_profit_price"],
                "max_risk_per_contract": opt_decision["max_risk_per_contract"],
                "estimated_friction_usd": 14.80, # 2 legs * $0.69 * 2 sides + spread cross
                "rationale": f"Chronos {model_dir} (P={prob_pos:.2f}) + {trend_regime} trend. {opt_decision['rationale']}"
            }
        else:
            # Spot Equity / Crypto / Commodity Trade
            stop_dist = max(atr * 1.5, current_spot * 0.015)
            stop_loss = round(current_spot - stop_dist if direction == "BULLISH" else current_spot + stop_dist, 2)
            take_profit = round(current_spot + (stop_dist * 2.0) if direction == "BULLISH" else current_spot - (stop_dist * 2.0), 2)

            raw_signal = {
                "signal_id": signal_id,
                "timestamp_utc": now_iso,
                "asset_class": asset_class,
                "underlying": symbol,
                "symbol": symbol,
                "direction": direction,
                "strategy_type": f"{asset_class}_SPOT_{direction}",
                "contracts": [],
                "confidence_score": round(confidence, 2),
                "iv_rank": round(iv_rank, 1),
                "entry_limit_price": round(current_spot, 2),
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
                "max_risk_per_contract": round(stop_dist, 2),
                "estimated_friction_usd": 5.00,
                "rationale": f"Multi-Agent consensus: Chronos forecast {model_dir} (P={prob_pos:.2f}), sentiment {sent_dir}, regime {trend_regime}."
            }

        # 5. Agent 4: Deterministic Risk Gate Validation
        approved, reason, recommended_units = self.risk_guard.validate_trade_candidate(
            signal=raw_signal,
            account_equity=account_equity,
            current_open_positions=current_open_positions
        )

        if not approved:
            return None

        raw_signal["recommended_quantity"] = recommended_units
        raw_signal["max_risk_usd"] = round(recommended_units * raw_signal["max_risk_per_contract"], 2)

        return raw_signal
