# ================================================================
# File: Neuralis_AI/cognition_layer.py
# Purpose:
#   Cognitive reasoning and adaptive weighting after rehydration.
#   Accepts neural field vectors from Rehydrator and builds
#   live reasoning cache for Neuralis AI.
# ================================================================

from __future__ import annotations
from typing import Dict, Any, List
import time
import math
import random

from VHW.vsd_manager import VSDManager
from Neuralis_AI.cognition_summary import build_summary

class CognitionLayer:
    def __init__(self):
        # Core caches (populated from rehydrated vectors)
        self.neural_field: Dict[str, float] = {}
        self.temporal_cache: List[Dict[str, Any]] = []
        self.adaptive_weights: Dict[str, float] = {}
        self.last_update_s: float = 0.0
        self._vsd: VSDManager | None = None

    # ------------------------------------------------------------
    def load_from_state(self, vector: Dict[str, Any]) -> None:
        """Simulated rehydration of neural field and weight data."""
        nf = vector.get("neural_field", {})
        aw = vector.get("adaptive_weights", {})
        self.neural_field.update({k: float(v) for k, v in nf.items()})
        self.adaptive_weights.update({k: float(v) for k, v in aw.items()})
        self.last_update_s = time.time()

    # ------------------------------------------------------------
    def evaluate_signal(self, signal: Dict[str, Any]) -> float:
        """Return dynamic cognitive weight based on confidence and context."""
        conf = float(signal.get("avg_confidence", 0.0))
        delta = float(signal.get("avg_predicted_change", 0.0))
        bias = self._context_bias(signal.get("symbol", ""))
        bias *= self._cognition_bias()
        weight = conf * (1.0 + delta * 0.5) * bias
        self._record(signal, weight)
        return max(0.0, min(1.0, weight))

    # ------------------------------------------------------------
    def _context_bias(self, symbol: str) -> float:
        """Bias calculation derived from neural field memory."""
        if not self.neural_field:
            return 1.0
        base = self.neural_field.get(symbol, 0.5)
        fluct = math.sin(time.time() * 0.001) * 0.05
        return max(0.5, min(1.5, base + fluct))

    # ------------------------------------------------------------
    def _record(self, signal: Dict[str, Any], weight: float) -> None:
        """Store evaluated signal in temporal cache."""
        entry = {
            "timestamp": time.time(),
            "symbol": signal.get("symbol", "UNKNOWN"),
            "confidence": signal.get("avg_confidence", 0.0),
            "predicted_change": signal.get("avg_predicted_change", 0.0),
            "weight": weight,
        }
        self.temporal_cache.append(entry)
        if len(self.temporal_cache) > 500:
            self.temporal_cache.pop(0)

    # ------------------------------------------------------------
    def adapt_weights(self) -> None:
        """Self-adjust adaptive weights based on temporal performance."""
        if not self.temporal_cache:
            return
        current = time.time()
        if current - self.last_update_s < 5.0:
            return
        avg_weight = sum(x["weight"] for x in self.temporal_cache) / len(self.temporal_cache)
        for k in list(self.adaptive_weights.keys()):
            self.adaptive_weights[k] = 0.99 * self.adaptive_weights[k] + 0.01 * avg_weight
        self.last_update_s = current

    # ------------------------------------------------------------
    def _get_vsd(self) -> VSDManager | None:
        if self._vsd is not None:
            return self._vsd
        try:
            self._vsd = VSDManager.global_instance()  # type: ignore[attr-defined]
        except Exception:
            self._vsd = None
        return self._vsd

    # ------------------------------------------------------------
    def _cognition_bias(self) -> float:
        """Bias factor derived from Neuralis cognition summary.

        This consults the VSD-resident cognition summary and boundary
        classifier result to gently nudge weights without creating any
        cross-subsystem imports.
        """
        vsd = self._get_vsd()
        if vsd is None:
            return 1.0
        try:
            summary = build_summary(vsd)
        except Exception:
            return 1.0

        # Boundary decision: if deny, down-bias; if allow, neutral.
        boundary = summary.get("boundary", {}) or {}
        decision = str(boundary.get("decision", "allow"))
        if decision == "deny":
            return 0.8

        # Optional: use number of recent entries as mild confidence signal.
        domains = summary.get("domains", {}) or {}
        total_recent = 0
        for d in domains.values():
            recent = d.get("recent", []) or []
            total_recent += len(recent)
        if total_recent > 0:
            # Very small upward bias when there is active cognition history.
            return 1.0 + min(0.05, total_recent * 0.001)

        return 1.0

    # ------------------------------------------------------------
    def summarize(self) -> Dict[str, Any]:
        """Return a lightweight snapshot of cognitive state."""
        avg_w = sum(self.adaptive_weights.values()) / len(self.adaptive_weights) if self.adaptive_weights else 0.0
        return {
            "entries": len(self.temporal_cache),
            "adaptive_avg": avg_w,
            "neural_field_size": len(self.neural_field),
        }
