# ================================================================
# File: Neuralis_AI/ai_console.py
# Purpose:
#   Minimal interactive console for Neuralis AI diagnostics.
#   Hydrates using memory vectors through AICore on demand.
# ASCII-ONLY
# ================================================================

from __future__ import annotations
from typing import Any, Dict, Optional
import json

class AIConsole:
    def __init__(self, ai_core: Any, telemetry_adapter: Any, learning_bridge: Any, autonomy_governor: Any):
        self.ai_core = ai_core
        self.telemetry = telemetry_adapter
        self.learning = learning_bridge
        self.governor = autonomy_governor

    # ------------------------------------------------------------
    def hydrate_from_memory(self, vector: Dict[str, Any]) -> None:
        """
        Fan-out hydration to subcomponents. Expected dict may contain:
          learning, governor, cognition (for CognitionLayer.load_from_state)
        """
        try:
            if hasattr(self.learning, "hydrate_from_memory"):
                self.learning.hydrate_from_memory(vector)
        except Exception:
            pass
        try:
            if hasattr(self.governor, "hydrate_from_memory"):
                self.governor.hydrate_from_memory(vector)
        except Exception:
            pass
        try:
            cogvec = vector.get("cognition", {})
            if hasattr(self.ai_core, "cognition") and hasattr(self.ai_core.cognition, "load_from_state"):
                self.ai_core.cognition.load_from_state(cogvec)
        except Exception:
            pass

    # ------------------------------------------------------------
    def dump_status(self) -> Dict[str, Any]:
        """
        Return a compact snapshot for display or logging.
        """
        out: Dict[str, Any] = {}
        try:
            if hasattr(self.ai_core, "cognition") and hasattr(self.ai_core.cognition, "summarize"):
                out["cognition"] = self.ai_core.cognition.summarize()
        except Exception:
            out["cognition"] = {}
        try:
            out["learning"] = {"alpha": getattr(self.learning, "alpha", None), "beta": getattr(self.learning, "beta", None)}
        except Exception:
            out["learning"] = {}
        try:
            out["governor"] = {
                "enabled": getattr(self.governor, "enabled", None),
                "max_updates_per_min": getattr(self.governor, "max_updates_per_min", None),
                "min_conf_to_trade": getattr(self.governor, "min_conf_to_trade", None),
            }
        except Exception:
            out["governor"] = {}
        # push to telemetry
        try:
            self.telemetry.push_cognition_summary(out.get("cognition", {}))
            self.telemetry.push_learning_knobs(out.get("learning", {}))
            self.telemetry.push_governor_status(out.get("governor", {}))
        except Exception:
            pass
        return out
