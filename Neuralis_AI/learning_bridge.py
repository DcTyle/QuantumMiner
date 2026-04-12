# ================================================================
# File: Neuralis_AI/learning_bridge.py
# Purpose:
#   Real-time learning connector between Prediction Engine signals
#   and CognitionLayer adaptive weights. Hydrates from memory vectors.
# ASCII-ONLY
# ================================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time

class LearningBridge:
    def __init__(self):
        self.cognition = None
        self.prediction_engine = None
        self.alpha = 0.01
        self.beta = 0.001
        self.last_hydrate_ts = 0.0

    # ------------------------------------------------------------
    def hydrate_from_memory(self, vector: Dict[str, Any]) -> None:
        """
        Accept a memory vector dict and set learning rates and knobs.
        Expected keys:
          learning: { alpha: float, beta: float }
        """
        try:
            learning = vector.get("learning", {})
            self.alpha = float(learning.get("alpha", self.alpha))
            self.beta = float(learning.get("beta", self.beta))
            self.last_hydrate_ts = time.time()
        except Exception:
            pass

    # ------------------------------------------------------------
    def attach(self, prediction_engine: Any, cognition_layer: Any) -> None:
        self.prediction_engine = prediction_engine
        self.cognition = cognition_layer

    # ------------------------------------------------------------
    def tick_update(self) -> None:
        """
        Pull latest fused signals from Prediction Engine and update
        CognitionLayer temporal cache and weights.
        """
        if not self.prediction_engine or not self.cognition:
            return
        try:
            signals: List[Dict[str, Any]] = self.prediction_engine.get_latest_signals()
            for s in signals:
                w = self.cognition.evaluate_signal(s)
            self.cognition.adapt_weights()
        except Exception:
            pass

    # ------------------------------------------------------------
    def backpropagate_outcome(self, outcomes: List[Dict[str, Any]]) -> None:
        """
        Accept realized outcomes to influence future weighting.
        Each outcome item keys:
          symbol, realized_pnl, was_trade, confidence_at_trade
        """
        if not self.cognition:
            return
        try:
            # Simple rule: nudge adaptive weights based on realized pnl sign
            for o in outcomes:
                symbol = str(o.get("symbol", "")).upper()
                pnl = float(o.get("realized_pnl", 0.0))
                conf = float(o.get("confidence_at_trade", 0.5))
                step = self.alpha * conf if pnl >= 0.0 else -self.alpha * (1.0 - conf)
                base = self.cognition.adaptive_weights.get(symbol, 0.5)
                self.cognition.adaptive_weights[symbol] = max(0.0, min(1.0, base + step))
            # Slow regularization
            for k in list(self.cognition.adaptive_weights.keys()):
                self.cognition.adaptive_weights[k] = (1.0 - self.beta) * self.cognition.adaptive_weights[k] + self.beta * 0.5
        except Exception:
            pass
