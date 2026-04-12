# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# File: live.py
# Version: v4.8.7 Hybrid
#
# Purpose
# -------
#   Live telemetry bridge between Monitor (smoothed utilization metrics)
#   and upper-layer systems (MinerEngine, TelemetryConsole, Murphy Watchdog).
#
# v4.8.7 Hybrid Update Summary
# ----------------------------
#   - Added BIOS readiness gate via VSDManager.get("system/bios_boot_ok").
#   - Integrated core.event_bus for optional telemetry.perf publishes.
#   - Structured UTC logger 'VHW.Live'; safe fallback stubs for sandbox use.
#   - Maintained pulse / anti-pulse harmonic oscillator loop (cosine phase).
#   - ASCII only; no UTF-8 or Unicode characters.
#
# Public Surface
# --------------
# class LiveTelemetry:
#   .tick()     -> produce one telemetry frame (dict)
#   .read_raw() -> last raw Monitor output
#   .pulse()    -> harmonic pulse value for internal rhythm control
# ============================================================================

from __future__ import annotations
from typing import Dict, Any, Callable
import time
import math
import threading
import logging

# ---------------------------------------------------------------------------
# Layered imports (guarded fallbacks)
# ---------------------------------------------------------------------------
try:
    from config.manager import ConfigManager
except Exception:
    class ConfigManager:
        def get(self, k: str, default=None): return default

try:
    from core.event_bus import get_event_bus
except Exception:
    def get_event_bus():
        class _NoBus:
            def publish(self, *a, **kw): return None
            def subscribe(self, *a, **kw): return None
        return _NoBus()

try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def __init__(self): self._kv = {}
        def get(self, k: str, d=None): return self._kv.get(k, d)
        def store(self, k: str, v): self._kv[k] = v

from VHW.monitor import Monitor
from core.utils import append_telemetry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _utc_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)sZ %(name)s %(levelname)s %(message)s",
                                "%Y-%m-%dT%H:%M:%S")
        h.setFormatter(fmt)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg

LOG = _utc_logger("VHW.Live")

# ---------------------------------------------------------------------------
# Live Telemetry
# ---------------------------------------------------------------------------
class LiveTelemetry:
    """
    Consumes smoothed frames from Monitor and re-emits harmonized telemetry
    data for other modules. Maintains internal phase rhythm (pulse / anti-pulse).
    """

    def __init__(self, monitor_reader: Callable[[], Dict[str, Any]]) -> None:
        self._reader = monitor_reader
        self._lock = threading.RLock()
        self._last: Dict[str, Any] = {}
        self._pulse_state = 0.0
        self._phase_rate = 0.25
        self._last_pulse_ts = time.time()
        self._bus = get_event_bus()
        self._vsd = VSDManager()
        self._config = ConfigManager()
        self._bios_gate_logged = False

    # -----------------------------------------------------------------------
    # BIOS readiness check
    # -----------------------------------------------------------------------
    def _bios_ready(self) -> bool:
        try:
            return bool(self._vsd.get("system/bios_boot_ok", False))
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Core methods
    # -----------------------------------------------------------------------
    def tick(self) -> Dict[str, Any]:
        """
        Produce one telemetry snapshot by reading from the Monitor.
        Adds synthetic pulse / anti-pulse signal for synchronization.
        Publishes telemetry.perf to EventBus when BIOS ready.
        """
        with self._lock:
            if not self._bios_ready():
                if not self._bios_gate_logged:
                    LOG.warning("LiveTelemetry waiting for BIOS boot_ok...")
                    self._bios_gate_logged = True
                return dict(self._last or {})

            try:
                frame = self._reader() or {}
            except Exception:
                frame = {}

            global_util = float(frame.get("global_util", 0.0))
            gpu_util = float(frame.get("gpu_util", 0.0))
            gpu_mem_util = float(frame.get("gpu_mem_util", 0.0))
            mem_bw_util = float(frame.get("mem_bw_util", 0.0))
            cpu_util = float(frame.get("cpu_util", 0.0))
            util_cap = float(frame.get("util_cap", 0.75))
            headroom = float(frame.get("group_headroom", 0.10))
            target_util = float(frame.get("target_util", max(0.0, util_cap - headroom)))
            alert = bool(frame.get("alert", False))
            ts = str(frame.get("timestamp", _now_iso()))

            now_t = time.time()
            dt = now_t - self._last_pulse_ts
            self._last_pulse_ts = now_t
            self._pulse_state += self._phase_rate * dt * 2.0 * math.pi
            pulse = math.cos(self._pulse_state)
            anti_pulse = -pulse

            live_frame = {
                "timestamp": ts,
                "global_util": global_util,
                "gpu_util": gpu_util,
                "gpu_mem_util": gpu_mem_util,
                "mem_bw_util": mem_bw_util,
                "cpu_util": cpu_util,
                "util_cap": util_cap,
                "group_headroom": headroom,
                "target_util": target_util,
                "alert": alert,
                "pulse": pulse,
                "anti_pulse": anti_pulse
            }

            self._last = dict(live_frame)

            try:
                self._bus.publish("telemetry.perf", dict(live_frame))
            except Exception:
                pass

            append_telemetry("live", live_frame)
            return live_frame

    def read_raw(self) -> Dict[str, Any]:
        """Return last Monitor frame captured."""
        with self._lock:
            return dict(self._last)

    def pulse(self) -> float:
        """Return current harmonic pulse value."""
        with self._lock:
            return float(self._last.get("pulse", 0.0))

# End of file
