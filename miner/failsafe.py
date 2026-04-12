# ============================================================================
# VirtualMiner / MINER
# ASCII-ONLY SOURCE FILE
# File: failsafe.py
# Version: v5.1 Clean Imports
# ============================================================================

from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional
import time
import threading

# ----------------------------------------------------------------------------
# VSDManager guarded
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def get(self, *_a, **_k): return False
        def store(self, *_a, **_k): return None

# ----------------------------------------------------------------------------
# EventBus guarded
# ----------------------------------------------------------------------------
try:
    from bios.event_bus import get_event_bus
except Exception:
    class _Bus:
        def publish(self, *_a, **_k): pass
        def subscribe(self, *_a, **_k): pass
    def get_event_bus(): return _Bus()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _now(): return float(time.time())

def _safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def _bios_ready(vsd):
    try: return bool(vsd.get("system/bios_boot_ok", False))
    except Exception: return False

# ----------------------------------------------------------------------------
# FailsafeGovernor
# ----------------------------------------------------------------------------
class FailsafeGovernor:
    def __init__(
        self,
        list_lanes: Callable[[], List[Dict[str, Any]]] = lambda: [],
        disable_lane: Callable[[str], None] = lambda x: None,
        difficulty_of: Callable[[str], float] = lambda n: 1.0,
        cooldown_s: float = 5.0,
        bias: float = 0.0,
        vsd: Optional[VSDManager] = None,
    ):
        self._list = list_lanes
        self._disable = disable_lane
        self._diff = difficulty_of
        self._cooldown_s = float(max(1.0, cooldown_s))
        self._bias = float(bias)
        self._vsd = vsd or VSDManager()
        self._lock = threading.RLock()
        self._last_ts = 0.0
        self._snap = {}
        try:
            get_event_bus().subscribe("failsafe.prune", self._on_prune)
        except Exception:
            pass

    # ----------------------------------------------------------------------
    def _on_prune(self, payload: Dict[str, Any]):
        if not _bios_ready(self._vsd):
            return
        with self._lock:
            if _now() - self._last_ts < self._cooldown_s:
                return
            lane = str(payload.get("lane_id", "")).strip()
            if lane:
                try:
                    self._disable(lane)
                    self._last_ts = _now()
                    self._snap = {
                        "ts": self._last_ts,
                        "action": "disable_lane",
                        "lane_id": lane,
                        "source": "event",
                    }
                except Exception:
                    return

    # ----------------------------------------------------------------------
    def check_health(self, frame: Dict[str, Any]):
        """
        Governor v2 failsafe hooks:
          - Burst detection: if many valid shares arrive in the same cycle, record burst and request backpressure.
          - Queue depth guard: if submit queue is too deep, pause compute by emitting feedback.
        No density or lane-throttling logic is used.
        """
        if not _bios_ready(self._vsd):
            return
        with self._lock:
            if _now() - self._last_ts < self._cooldown_s:
                return

            # Read submitter telemetry
            try:
                base = "miner/submitter"
                queue_depth = int(self._vsd.get(base + "/queue_depth", 0))
                subs_this_sec = int(self._vsd.get(base + "/submissions_this_second", 0))
                allowed = float(self._vsd.get(base + "/allowed_rate_per_second", 2.0))
            except Exception:
                queue_depth, subs_this_sec, allowed = 0, 0, 2.0

            # Burst guard: submissions exceed allowed by 2x within the second
            is_burst = subs_this_sec > max(1.0, 2.0 * allowed)
            if is_burst:
                try:
                    self._vsd.store("miner/bursts/event", {
                        "ts": _now(),
                        "submissions_this_second": subs_this_sec,
                        "allowed_per_second": allowed,
                        "note": "burst_guard"
                    })
                    # Signal compute feedback to slow down
                    self._vsd.store("miner/compute_feedback/submission_rate_throttle", {
                        "ts": _now(),
                        "reason": "burst_guard",
                        "factor": 0.5
                    })
                    self._last_ts = _now()
                    self._snap = {"ts": self._last_ts, "action": "burst_guard"}
                except Exception:
                    pass
                return

            # Queue depth guard
            if queue_depth >= 2048:
                try:
                    self._vsd.store("miner/compute_feedback/submission_rate_throttle", {
                        "ts": _now(),
                        "reason": "queue_depth_guard",
                        "factor": 0.25
                    })
                    self._last_ts = _now()
                    self._snap = {"ts": self._last_ts, "action": "queue_depth_guard"}
                except Exception:
                    pass
                return

    # ----------------------------------------------------------------------
    def snapshot(self): return dict(self._snap)
