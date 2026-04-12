# =================================================================
# Quantum Application / control_center
# ASCII-ONLY SOURCE FILE
# File: telemetry_console.py
# Version: v4.8.3 (AI-Processor snapshots; unified with live console)
# -----------------------------------------------------------------
# Purpose
# -------
#   Main telemetry console for the Control Center GUI.
#   Displays aggregate hashrate, profitability, and system health
#   across all networks. Can be driven by telemetry_console_live.py
#   or queried on demand by Jarvis ADA AI subsystems.
#   Now also binds Prediction Engine signals into VSD for UI display.
# -----------------------------------------------------------------

from __future__ import annotations
import time
import os
import threading
from typing import Any, Dict, Callable, Optional

try:
    from core.AI_processor import Dehydrator
except Exception:
    Dehydrator = None  # type: ignore

# --- Prediction Engine feed binding ---
try:
    from prediction_engine.prediction_engine import PredictionEngine
except Exception:
    PredictionEngine = None  # type: ignore


class TelemetryConsole:
    """
    Read, display, and export live profitability and performance data.
    Provides a base layer for GUI widgets or headless CLI telemetry.
    """

    def __init__(self, vsd: Any, refresh_s: float = 10.0) -> None:
        self.vsd = vsd
        self.refresh_s = float(refresh_s)
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = False
        self.callbacks: list[Callable[[Dict[str, Any]], None]] = []
        self._last_snapshot_path: str = ""
        # Try to bind Prediction Engine feed automatically
        self._attach_prediction_feed()

    # --------------------------------------------------------------
    def register_callback(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        """Attach a callback to receive telemetry frames."""
        self.callbacks.append(fn)

    # --------------------------------------------------------------
    def _attach_prediction_feed(self) -> None:
        """Bind Prediction Engine output into VSD so GUI can read it."""
        try:
            if PredictionEngine is None:
                return
            engine = PredictionEngine.get_instance()
            # Write latest signals into VSD on every console refresh tick
            def _write_predictions(_: Dict[str, Any]) -> None:
                try:
                    self.vsd.store("telemetry/predictions/latest", engine.get_latest_signals())
                except Exception:
                    pass
            self.register_callback(_write_predictions)
        except Exception:
            pass

    # --------------------------------------------------------------
    def _fetch_telemetry(self) -> Dict[str, Any]:
        """Collect key telemetry data from VSD."""
        try:
            coins = ["ETC", "RVN", "FLUX", "ERG"]
            agg = self.vsd.get("telemetry/metrics/AGG/current", {}) or {}
            out: Dict[str, Any] = {"aggregate": agg, "networks": {}}
            for c in coins:
                net = self.vsd.get(f"telemetry/profit/{c}/current", {}) or {}
                out["networks"][c] = {
                    "profit_usd_day": float(net.get("usd_day", 0.0)),
                    "profit_usd_hour": float(net.get("usd_hour", 0.0)),
                    "profit_usd_min": float(net.get("usd_min", 0.0)),
                    "hashrate_hs": float(
                        self.vsd.get(f"telemetry/mine/{c}/accepted_hashrate_hs", 0.0)
                    ),
                }
            # Attach prediction signals for UI
            out["predictions"] = self.vsd.get("telemetry/predictions/latest", [])
            return out
        except Exception:
            return {}

    # --------------------------------------------------------------
    def _loop(self) -> None:
        """Periodic background refresh."""
        while not self._stop_flag:
            snap = self._fetch_telemetry()
            for cb in list(self.callbacks):
                try:
                    cb(snap)
                except Exception:
                    pass
            time.sleep(self.refresh_s)

    # --------------------------------------------------------------
    def start(self) -> None:
        """Start live telemetry polling."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="telemetry_console_thread"
        )
        self._thread.start()

    # --------------------------------------------------------------
    def stop(self, timeout: float = 2.0) -> None:
        """Stop background thread and emit AI-Processor snapshot."""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        # Pack full AI-Processor snapshot for diagnostics
        try:
            if Dehydrator is not None:
                stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                out_root = os.path.join("VHW", "VD", "state_vectors")
                os.makedirs(out_root, exist_ok=True)
                out_json = os.path.join(
                    out_root, f"telemetry_console_snapshot_{stamp}.json"
                )
                Dehydrator().pack(".", out_json)
                self._last_snapshot_path = out_json
                self.vsd.store("telemetry/snapshots/last_console_state", out_json)
        except Exception:
            pass
