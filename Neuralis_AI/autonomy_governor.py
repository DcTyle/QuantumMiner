# ================================================================
# File: Neuralis_AI/autonomy_governor.py
# Purpose:
#   Safety governor for Neuralis AI learning and action.
#   Enforces rate limits and confidence thresholds.
#   Hydrates constraints from memory vectors.
# ASCII-ONLY
# ================================================================

from __future__ import annotations
from typing import Any, Dict
import time

class AutonomyGovernor:
    def __init__(self):
        self.enabled = True
        self.max_updates_per_min = 120
        self.min_conf_to_trade = 0.99
        self._update_budget = self.max_updates_per_min
        self._last_refill = time.time()

    # ------------------------------------------------------------
    def hydrate_from_memory(self, vector: Dict[str, Any]) -> None:
        """
        Expected keys:
          governor: { enabled: bool, max_updates_per_min: int, min_conf_to_trade: float }
        """
        try:
            g = vector.get("governor", {})
            self.enabled = bool(g.get("enabled", self.enabled))
            self.max_updates_per_min = int(g.get("max_updates_per_min", self.max_updates_per_min))
            self.min_conf_to_trade = float(g.get("min_conf_to_trade", self.min_conf_to_trade))
            self._update_budget = self.max_updates_per_min
            self._last_refill = time.time()
        except Exception:
            pass

    # ------------------------------------------------------------
    def allow_learning_update(self) -> bool:
        """
        Token bucket for learning updates per minute.
        """
        now = time.time()
        if now - self._last_refill >= 60.0:
            self._update_budget = self.max_updates_per_min
            self._last_refill = now
        if self._update_budget <= 0:
            return False
        self._update_budget -= 1
        return True

    # ------------------------------------------------------------
    def allow_trade(self, confidence: float) -> bool:
        if not self.enabled:
            return True
        return confidence >= self.min_conf_to_trade
