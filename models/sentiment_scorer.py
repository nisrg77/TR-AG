"""
Financial NLP Sentiment Scorer:
Scores news headlines, macro events, and earnings announcements (FinBERT / Lexicon).
"""

from typing import Dict, Any, List

class SentimentScorer:
    def __init__(self):
        self._pipe = None

    def score_headlines(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Analyzes a list of news headlines and produces aggregate sentiment.
        """
        if not headlines:
            return {"sentiment_score": 0.0, "label": "NEUTRAL", "confidence": 0.50}

        bullish_keywords = ["surge", "jump", "record", "beat", "strong", "profit", "bullish", "rally", "breakout", "upgrade", "growth", "inflow"]
        bearish_keywords = ["drop", "fall", "crash", "miss", "weak", "loss", "bearish", "plunge", "downgrade", "probe", "outflow", "recession"]

        total_score = 0.0
        for h in headlines:
            h_lower = h.lower()
            bull_count = sum(1 for w in bullish_keywords if w in h_lower)
            bear_count = sum(1 for w in bearish_keywords if w in h_lower)
            if bull_count > bear_count:
                total_score += 1.0
            elif bear_count > bull_count:
                total_score -= 1.0

        avg_score = total_score / len(headlines) if headlines else 0.0
        avg_score = max(-1.0, min(1.0, avg_score))

        label = "BULLISH" if avg_score > 0.15 else ("BEARISH" if avg_score < -0.15 else "NEUTRAL")
        confidence = 0.50 + abs(avg_score) * 0.40

        return {
            "sentiment_score": round(avg_score, 2),
            "label": label,
            "confidence": round(confidence, 2),
            "headline_count": len(headlines)
        }
