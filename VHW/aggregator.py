# ============================================================================
# VirtualMiner / VHW
# File: aggregator.py
# ASCII-ONLY SOURCE FILE
# Jarvis ADA v4.7 Hybrid compatible.
# ----------------------------------------------------------------------------
# Purpose
# -------
# Acts as the firmware telemetry multiplexer for all subsystems.
# This module:
#   - accepts arbitrary telemetry frames from ANY subsystem,
#   - normalizes, timestamps, and buffers them,
#   - blends low-level hardware telemetry sources (optional),
#   - emits unified frames for MinerEngine and TelemetryConsole.
#
# IMPORTANT
# ---------
# No module-specific logic is allowed here.
# Aggregator does NOT know what data means.
# Aggregator only transports and smooths telemetry frames.
#
# Public API:
#   agg = Aggregator(readers=[hw_reader1, hw_reader2])
#   agg.emit(src, tag, payload)     # from any subsystem
#   agg.read_unified()               # one-frame console snapshot
#   agg.pop_frames()                 # raw frames for higher layers
# ============================================================================

from __future__ import annotations
from typing import Dict, Any, List, Callable
import time
import threading

from core.utils import append_telemetry, log

LOG = log("VHW.Aggregator")

ASCII_MIN = 32
ASCII_MAX = 126

# ---------------------------------------------------------------------------
# Minimal helper functions
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _float_to_ascii(v: float) -> str:
    v = _clamp(v, -1.0, 1.0)
    code = int((v + 1.0) * 47) + 32
    code = _clamp(code, ASCII_MIN, ASCII_MAX)
    return chr(int(code))

def _ascii_from_vector(vec: List[float]) -> str:
    return "".join(_float_to_ascii(x) for x in vec)

# ---------------------------------------------------------------------------
# Unified Telemetry Wrapper (now inside Aggregator)
# ---------------------------------------------------------------------------
class TelemetryFrame:
    def __init__(self, src: str, tag: str, payload: Dict[str, Any]):
        self.ts = float(time.time())
        self.src = str(src)
        self.tag = str(tag)
        self.payload = dict(payload or {})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "src": self.src,
            "tag": self.tag,
            "payload": self.payload
        }

# ---------------------------------------------------------------------------
# Aggregator (Firmware Telemetry Multiplexer)
# ---------------------------------------------------------------------------
class Aggregator:
    def __init__(self, readers: List[Callable[[], Dict[str, Any]]]):
        """
        readers = low-level hardware telemetry sources
        These will be blended into the unified snapshot.
        """
        self.readers = list(readers)
        self._lock = threading.RLock()

        # This is where incoming subsystem frames are stored.
        self._frames: List[TelemetryFrame] = []

        # Aggregated hardware telemetry state
        self._state: Dict[str, float] = {
            "phase": 0.0
        }

    # ----------------------------------------------------------------------
    # Universal telemetry emitter (ALL subsystems use this)
    # ----------------------------------------------------------------------
    def emit(self, src: str, tag: str, payload: Dict[str, Any]) -> None:
        """Subsystems call this to push telemetry frames."""
        try:
            frame = TelemetryFrame(src, tag, payload)
            with self._lock:
                self._frames.append(frame)
        except Exception as exc:
            LOG.error("Aggregator.emit failed: %s", exc, exc_info=True)

    # ----------------------------------------------------------------------
    # Pop all raw telemetry frames for higher-level consumers
    # ----------------------------------------------------------------------
    def pop_frames(self) -> List[Dict[str, Any]]:
        """Returns and clears all pending telemetry frames."""
        with self._lock:
            out = [f.as_dict() for f in self._frames]
            self._frames.clear()
        return out

    # ----------------------------------------------------------------------
    # Internal hardware blending (unchanged, nonspecific)
    # ----------------------------------------------------------------------
    def _blend(self, vals: List[float]) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        weights = [0.5, 0.3, 0.2][:len(vals)]
        while len(weights) < len(vals):
            weights.append(0.0)
        return sum(vals[i] * weights[i] for i in range(len(vals)))

    # ----------------------------------------------------------------------
    # Unified hardware snapshot (for VHW telemetry only)
    # ----------------------------------------------------------------------
    def _read_hw(self) -> Dict[str, Any]:
        frames = []
        for r in self.readers:
            try:
                frames.append(r() or {})
            except Exception:
                frames.append({})

        def fget(key: str) -> float:
            return self._blend([float(fr.get(key, 0.0)) for fr in frames])

        # simple harmonic phase drift
        ph = (self._state.get("phase", 0.0) + 0.012) % 1.0
        self._state["phase"] = ph

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "global_util": fget("global_util"),
            "gpu_util": fget("gpu_util"),
            "gpu_mem_util": fget("gpu_mem_util"),
            "mem_bw_util": fget("mem_bw_util"),
            "cpu_util": fget("cpu_util"),
            "phase": ph
        }

    # ----------------------------------------------------------------------
    # Final unified telemetry read (for MinerEngine + Console)
    # ----------------------------------------------------------------------
    def read_unified(self) -> Dict[str, Any]:
        """
        Returns ONE snapshot:
          { hardware telemetry, pending frames (raw), timestamp }
        The console decides how to interpret/display the data.
        Aggregator MUST NOT interpret ANY semantic meaning.
        """
        with self._lock:
            hw = self._read_hw()
            frames = [f.as_dict() for f in self._frames]
            self._frames.clear()

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": hw,
            "frames": frames
        }
