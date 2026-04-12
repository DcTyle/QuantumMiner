# ============================================================================
# Path: Quantum Application/prediction_engine/ai_utility.py
# Version: v5 "Unified Imports"
# Description:
#   Read-only prediction engine adapter for Neuralis AI.
#   Provides a safe, analysis-only interface into prediction_engine so that
#   Neuralis can query predictions and scores without invoking any trading
#   side effects or order execution.
#   All code is ASCII-only.
# ============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bios.event_bus import BIOSUtilities  # Unified BIOS event utilities
from VHW.system_utils import *  # VHW unified system utilities

# We deliberately avoid importing any trade execution modules here.
# Only model / cluster / data processor components are used, and even those
# are guarded so that failures do not impact trading or miner behavior.
try:
    from prediction_engine.pridiction_engine import PredictionEngine  # type: ignore
except Exception:
    PredictionEngine = None  # type: ignore


class PredictionEngineAdapter:
    """
    Safe, read-only adapter for Neuralis.
    - Never sends trades.
    - Never mutates live trading state.
    - Uses prediction engine only in analysis mode.
    """
    def __init__(self) -> None:
        self._engine = None
        if PredictionEngine is not None:
            try:
                # We hint to the underlying engine (if it supports it)
                # that this is an analysis-only context.
                self._engine = PredictionEngine(mode="analysis_only")  # type: ignore
            except Exception:
                self._engine = None

    # ------------------------------------------------------------------
    def predict_context(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level context prediction entry for Neuralis.
        Returns a normalized result dict with 'confidence', 'label',
        and 'meta' fields so the AI layer can reason about behavior.
        """
        if self._engine is not None and hasattr(self._engine, "predict"):
            try:
                result = self._engine.predict(features, mode="analysis")  # type: ignore
                if isinstance(result, dict):
                    return result
            except Exception as exc:
                BIOSUtilities.publish("prediction.error", {"error": str(exc)})
        # Fallback neutral result
        return {
            "confidence": 0.0,
            "label": None,
            "meta": {"status": "unavailable"}
        }

    # ------------------------------------------------------------------
    def score_behavior(self, behavior_vector: List[float]) -> Dict[str, Any]:
        """
        Accepts a generic behavior vector (for example, Neuralis cognition
        features) and returns a scored assessment without issuing trades.
        """
        payload = {
            "behavior_vector": behavior_vector,
            "len": len(behavior_vector),
        }
        # Encode as ASCII for optional logging or VSD snapshot
        encoded = serialize_floatvec_with_tokens(behavior_vector)
        vsd_append_bounded("neuralis/behavior_vectors", {"payload": payload, "encoded": encoded})
        # We reuse predict_context semantics for scoring
        return self.predict_context(payload)

    # ------------------------------------------------------------------
    def safe_scenario_batch(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run a batch of hypothetical scenarios through the prediction engine
        in a way that is explicitly detached from live trading. This gives
        Neuralis a sandboxed backbone for analysis and planning.
        """
        results: List[Dict[str, Any]] = []
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            res = self.predict_context(scenario)
            results.append(res)
        return results




    # ------------------------------------------------------------------
    def explain_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a lightweight explanation object for a given feature dict.
        This is strictly for Neuralis reasoning and has no trading side effects.
        """
        base = self.predict_context(features)
        return {
            "prediction": base,
            "reasons": base.get("meta", {}).get("reasons", []),
            "raw_features_keys": list(features.keys()),
        }

    # ------------------------------------------------------------------
    def temporal_projection(self, features: Dict[str, Any], horizon_ticks: int = 10) -> Dict[str, Any]:
        """
        Simple wrapper to treat horizon as a feature for forward-looking analysis.
        The underlying engine decides what to do with horizon info.
        """
        enriched = dict(features)
        enriched["horizon_ticks"] = int(horizon_ticks)
        return self.predict_context(enriched)

    # ------------------------------------------------------------------
    def pattern_similarity(self, behavior_vector: List[float]) -> Dict[str, Any]:
        """
        Computes a basic similarity-esque summary for a behavior vector.
        This does not use any trading state and is safe for Neuralis introspection.
        """
        if not behavior_vector:
            return {"norm": 0.0, "length": 0, "status": "empty"}
        # Simple L2-like magnitude for introspection only
        s = sum(v * v for v in behavior_vector)
        norm = s ** 0.5
        return {"norm": float(norm), "length": len(behavior_vector), "status": "ok"}

    # ------------------------------------------------------------------
    def batch_explain(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch explanation for multiple hypothetical scenarios.
        Returns a list of explain_prediction() results.
        """
        out: List[Dict[str, Any]] = []
        for s in scenarios:
            if not isinstance(s, dict):
                continue
            out.append(self.explain_prediction(s))
        return out

__all__ = ["PredictionEngineAdapter"]
