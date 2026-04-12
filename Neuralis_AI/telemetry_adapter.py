# ============================================================================
# Quantum Application / neuralis_ai
# ASCII-ONLY SOURCE FILE
# File: neuralis_ai/telemetry_adapter.py
# Version: v7 Unified Kernel
# ============================================================================
"""
Neuralis AI Telemetry Adapter

Purpose
-------
Provides a clean bridge between Neuralis AI internal state and the global VSD.
Writes compact summaries:
    - cognition summary
    - governor status
    - learning knob configuration

Dependencies
-----------
Uses the unified VSD + EventBus from:
    from VHW.vsd_manager import VSD, get_event_bus

No wildcard imports. No BIOSUtilities. No PredictionEngine imports.
Neuralis observes prediction output through VSD events.
"""

from __future__ import annotations
from typing import Dict, Any

from VHW.vsd_manager import VSD, get_event_bus


class TelemetryAdapter:
    """
    Neuralis AI's interface for writing high-level telemetry snapshots
    into the VSD for Control Center, BIOS diagnostics, and mining engine
    health tools.

    All writes are direct VSD operations:
        - vsd.store("telemetry/ai/...")

    All methods are ASCII-only and thread-safe via VSDManager internals.
    """

    def __init__(self, vsd: Any = None) -> None:
        self.vsd = vsd if vsd is not None else VSD
        self.bus = get_event_bus()

    # ------------------------------------------------------------------
    # WRITE TELEMETRY
    # ------------------------------------------------------------------
    def push_cognition_summary(self, summary: Dict[str, Any]) -> None:
        """
        Store summary of AI cognition state.
        """
        try:
            self.vsd.store("telemetry/ai/cognition_summary", dict(summary))
            self.bus.publish("neuralis.telemetry.update", {
                "kind": "cognition",
                "ts": time.time()
            })
        except Exception:
            pass

    def push_governor_status(self, status: Dict[str, Any]) -> None:
        """
        Store current Neuralis "self-governor" status.
        """
        try:
            self.vsd.store("telemetry/ai/governor_status", dict(status))
            self.bus.publish("neuralis.telemetry.update", {
                "kind": "governor",
                "ts": time.time()
            })
        except Exception:
            pass

    def push_learning_knobs(self, knobs: Dict[str, Any]) -> None:
        """
        Store Neuralis active learning knob configuration.
        """
        try:
            self.vsd.store("telemetry/ai/learning_knobs", dict(knobs))
            self.bus.publish("neuralis.telemetry.update", {
                "kind": "knobs",
                "ts": time.time()
            })
        except Exception:
            pass

    # ------------------------------------------------------------------
    # READ TELEMETRY (GUI uses this)
    # ------------------------------------------------------------------
    def read_all(self) -> Dict[str, Any]:
        """
        Return all Neuralis telemetry as a unified dictionary.
        """
        try:
            return {
                "cognition_summary": self.vsd.get(
                    "telemetry/ai/cognition_summary", {}
                ),
                "governor_status": self.vsd.get(
                    "telemetry/ai/governor_status", {}
                ),
                "learning_knobs": self.vsd.get(
                    "telemetry/ai/learning_knobs", {}
                ),
            }
        except Exception:
            return {}
