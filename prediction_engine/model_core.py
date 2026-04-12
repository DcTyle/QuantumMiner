# ================================================================
# File: prediction_engine/model_core.py
# Purpose:
#   Core predictive model management layer.
#   Handles model loading, training, inference, and incremental updates.
# ================================================================

from __future__ import annotations
from typing import List, Dict, Tuple, Any
import random
import math
import statistics

from prediction_engine.data_processor import DataPreprocessor

# ================================================================
# ModelCore
# ================================================================

class ModelCore:
    """
    Core abstraction for predictive modeling.

    Responsibilities:
        - Load baseline or trained model parameters.
        - Train simple statistical or ML models on preprocessed data.
        - Generate predictions (price direction, magnitude, confidence).
        - Accept incremental live updates via update_with_tick().
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.preprocessor = DataPreprocessor()
        self._trained = False

    # ------------------------------------------------------------
    def load_models(self) -> None:
        """
        Initialize or load model parameters.
        For now, uses a lightweight statistical baseline.
        """
        self.models = {}
        self._trained = False
        print("[ModelCore] Model containers initialized.")

    # ------------------------------------------------------------
    def train(self, symbol: str, ohlcv: List[Dict[str, float]]) -> None:
        """
        Train a lightweight predictive model using simple
        statistical features for demonstration and bootstrapping.
        Future versions may integrate ML or NN backends.
        """
        if not ohlcv:
            return

        normalized = self.preprocessor.normalize(ohlcv)
        matrix = self.preprocessor.build_feature_matrix(normalized)
        train, test = self.preprocessor.split_train_test(matrix)

        # Simple baseline: compute mean of closing prices,
        # standard deviation, and slope for linear drift estimation.
        closes = [row[3] for row in matrix if len(row) >= 4]
        if len(closes) < 2:
            return

        mean_close = statistics.mean(closes)
        std_close = statistics.pstdev(closes)
        slope = (closes[-1] - closes[0]) / len(closes)

        self.models[symbol] = {
            "mean_close": mean_close,
            "std_close": std_close,
            "slope": slope,
            "trained_len": len(matrix),
        }
        self._trained = True
        print(f"[ModelCore] Trained model for {symbol}: mean={mean_close:.5f}, std={std_close:.5f}, slope={slope:.8f}")

    # ------------------------------------------------------------
    def predict(self, symbol: str, state_vector: List[float]) -> Dict[str, Any]:
        """
        Generate a simple price change prediction and confidence score.
        """
        if symbol not in self.models:
            # Untrained symbol -> fallback baseline
            return {"symbol": symbol, "predicted_change": 0.0, "confidence": 0.5}

        m = self.models[symbol]
        mean_close = m["mean_close"]
        std_close = max(m["std_close"], 1e-8)
        slope = m["slope"]

        # Last close (approx from vector)
        if not state_vector:
            return {"symbol": symbol, "predicted_change": 0.0, "confidence": 0.5}
        last_close = state_vector[3] if len(state_vector) >= 4 else state_vector[-1]

        delta = (last_close - mean_close) / std_close
        prediction = slope + delta * 0.01
        conf = 1 / (1 + abs(delta))
        conf = max(0.0, min(1.0, conf))

        return {
            "symbol": symbol,
            "predicted_change": prediction,
            "confidence": conf,
        }

    # ------------------------------------------------------------
    def bulk_predict(self, market_data: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Produce predictions for all symbols in one call.
        """
        results: List[Dict[str, Any]] = []
        for symbol, candles in market_data.items():
            if symbol not in self.models:
                self.train(symbol, candles)
            norm = self.preprocessor.normalize(candles)
            vec = self.preprocessor.generate_state_vector(norm)
            res = self.predict(symbol, vec)
            results.append(res)
        return results

    # ------------------------------------------------------------
    def update_with_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        Incrementally update model statistics using new live data.
        Expected keys: {"latest_block_rate": float, "timestamp": float}
        """
        rate = float(tick_data.get("latest_block_rate", 0.0))
        ts = float(tick_data.get("timestamp", 0.0))
        if rate <= 0.0:
            return

        # Apply simple exponential moving update to pseudo-metric model
        for symbol, model in self.models.items():
            old_mean = model["mean_close"]
            model["mean_close"] = 0.99 * old_mean + 0.01 * rate
            model["slope"] = 0.99 * model["slope"] + 0.01 * (rate - old_mean)
        self._trained = True

    # ------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return model status for diagnostics."""
        return {
            "model_count": len(self.models),
            "trained": self._trained,
            "symbols": list(self.models.keys()),
        }


# ---------------------------------------------------------------
# Example standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    from random import random
    # Generate mock candle data
    candles = [{"t": i, "o": 1 + random(), "h": 1.2 + random(),
                "l": 0.8 + random(), "c": 1 + random(), "v": 1000 + random() * 10}
               for i in range(100)]
    mc = ModelCore()
    mc.load_models()
    mc.train("BTC_USDT", candles)
    norm = mc.preprocessor.normalize(candles)
    vec = mc.preprocessor.generate_state_vector(norm)
    res = mc.predict("BTC_USDT", vec)
    print("Sample prediction:", res)
    print("Stats:", mc.stats())
