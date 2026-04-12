# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# File: monitor.py (contextual rolling window)
# v4.8.8 Contextual + Phase E (BIOS-logging patch integration)
# ============================================================================

from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import time, threading, logging

# ---------------------------------------------------------------------------
# Layered imports with safe fallbacks
# ---------------------------------------------------------------------------
try:
    from core.utils import get, store, append_telemetry
except Exception:
    def get(*a, **k): return {}
    def store(*a, **k): return None
    def append_telemetry(*a, **k): return None

# Fallback implementation of RuleParams
class RuleParams:
    def __init__(self) -> None:
        self._ema_alpha = 0.20
        self._headroom = 0.12
        self._alert_bias = 0.0
    @property
    def ema_alpha(self) -> float: return self._ema_alpha
    @property
    def headroom(self) -> float: return self._headroom
    @property
    def alert_bias(self) -> float: return self._alert_bias

# ---------------------------------------------------------------------------
# Config Manager Fallback
# ---------------------------------------------------------------------------
class _ConfigManagerFallback:
    def __init__(self) -> None: self._cfg = {}
    def get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)

ConfigManager = _ConfigManagerFallback

try:
    from bios.event_bus import get_event_bus
except Exception:
    def get_event_bus():
        class _NoBus:
            def publish(self, *_a, **_k): return
        return _NoBus()

try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def get(self, *_a, **_k): return False

# ============================================================================
# BIOS Logging Convention
# ============================================================================
def _mk_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        handler.setFormatter(fmt)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg

# ============================================================================
# Constants & Helpers
# ============================================================================
DEFAULT_UTIL_CAP = 0.75
DEFAULT_REFRESH_S = 0.50

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _bios_ready(vsd: Optional[VSDManager]) -> bool:
    try:
        if vsd is None:
            return False
        return bool(vsd.get("system/bios_boot_ok", False))
    except Exception:
        return False

# ============================================================================
# EMA Filter
# ============================================================================
class _EMA:
    def __init__(self, alpha: float) -> None:
        self.a = float(alpha)
        self.y: Optional[float] = None

    def update(self, x: float) -> float:
        xv = float(x)
        if self.y is None:
            self.y = xv
        else:
            self.y = self.a * xv + (1.0 - self.a) * self.y
        return self.y

# ============================================================================
# Contextual Rolling Window
# ============================================================================
class ContextualWindow:
    """
    Maintains a rolling buffer for telemetry, extending only when alerts occur.
    """
    def __init__(self, max_len: int = 200):
        self._buf: list[Dict[str, Any]] = []
        self._max_len = max_len
        self._lock = threading.RLock()

    def append_if_relevant(self, frame: Dict[str, Any], alert_key: str = "alert") -> None:
        with self._lock:
            if frame.get(alert_key, False):
                self._buf.append(frame)
                if len(self._buf) > self._max_len:
                    self._buf = self._buf[-self._max_len:]

    def snapshot(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._buf)

# ============================================================================
# Hostile-Outreach Security Layer
# ============================================================================
HOSTILE_OUTREACH_POLICY_ACTIVE = True

def _sandbox_publish(bus, topic: str, payload: Dict[str, Any]) -> None:
    if HOSTILE_OUTREACH_POLICY_ACTIVE:
        allowed = ["telemetry.perf", "telemetry.internal"]
        if topic not in allowed:
            print(f"[SECURITY] Blocked publish to '{topic}'")
            return
    try:
        bus.publish(topic, payload)
    except Exception:
        pass

def _sandbox_read(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return fn()
    except Exception:
        return {}

# ============================================================================
# Monitor Class
# ============================================================================
class Monitor:
    def __init__(self, read_telemetry: Callable[[], Dict[str, Any]],
                 util_cap: float = DEFAULT_UTIL_CAP) -> None:

        self.cfg = ConfigManager()
        self.bus = get_event_bus()
        self.vsd = VSDManager()
        self.rule = RuleParams()
        self.read = read_telemetry

        self.util_cap_default = float(util_cap)

        # EMA smoothing parameters
        a = _clamp(float(getattr(self.rule, "ema_alpha", 0.20)), 0.05, 0.35)

        self._lock = threading.RLock()
        self.ema_global = _EMA(a)
        self.ema_gpu = _EMA(a)
        self.ema_gmem = _EMA(a)
        self.ema_bw = _EMA(a)
        self.ema_cpu = _EMA(a)

        # Bias control
        self.alert_bias = float(
            self.cfg.get("monitor.alert_bias", getattr(self.rule, "alert_bias", 0.0))
        )

        self.refresh_interval_s = float(
            self.cfg.get("monitor.refresh_interval_s", DEFAULT_REFRESH_S)
        )
        self._last_publish_ts = 0.0

        # BIOS-logging format
        self.log = _mk_logger("VHW.Monitor")

        # Contextual rolling window
        max_len = int(self.cfg.get("monitor.context_window_len", 200))
        self.context_window = ContextualWindow(max_len=max_len)

        self.log.info(
            "monitor.init util_cap_default=%.3f alpha=%.3f bias=%.3f refresh=%.2fs",
            self.util_cap_default, a, self.alert_bias, self.refresh_interval_s
        )

    # ----------------------------------------------------------------------
    # EMA Aggregation
    # ----------------------------------------------------------------------
    def _update_all_ema(self, raw: Dict[str, float]) -> Dict[str, float]:
        return {
            "global_util": self.ema_global.update(raw.get("global_util", 0.0)),
            "gpu_util": self.ema_gpu.update(raw.get("gpu_util", 0.0)),
            "gpu_mem_util": self.ema_gmem.update(raw.get("gpu_mem_util", 0.0)),
            "mem_bw_util": self.ema_bw.update(raw.get("mem_bw_util", 0.0)),
            "cpu_util": self.ema_cpu.update(raw.get("cpu_util", 0.0)),
        }

    # ----------------------------------------------------------------------
    # Main Frame
    # ----------------------------------------------------------------------
    def frame(self) -> Dict[str, Any]:
        with self._lock:
            if not _bios_ready(self.vsd):
                ts = _now_iso()
                return {
                    "timestamp": ts,
                    "global_util": 0.0,
                    "gpu_util": 0.0,
                    "gpu_mem_util": 0.0,
                    "mem_bw_util": 0.0,
                    "cpu_util": 0.0,
                    "util_cap": self.util_cap_default,
                    "group_headroom": float(getattr(self.rule, "headroom", 0.12)),
                    "target_util": max(0.0, self.util_cap_default - float(getattr(self.rule, "headroom", 0.12))),
                    "alert": False
                }

            # Read telemetry through secure wrapper
            src = _sandbox_read(self.read)

            raw = {k: float(src.get(k, 0.0)) for k in
                   ["global_util", "gpu_util", "gpu_mem_util", "mem_bw_util", "cpu_util"]}

            util_cap = float(src.get("util_cap", self.util_cap_default))

            smoothed = self._update_all_ema(raw)

            headroom = float(getattr(self.rule, "headroom", 0.12))
            target = max(0.0, util_cap - headroom)
            alert = smoothed["global_util"] > (target - self.alert_bias)

            ts = str(src.get("timestamp", _now_iso()))

            out = {
                "timestamp": ts,
                "global_util": smoothed["global_util"],
                "gpu_util": smoothed["gpu_util"],
                "gpu_mem_util": smoothed["gpu_mem_util"],
                "mem_bw_util": smoothed["mem_bw_util"],
                "cpu_util": smoothed["cpu_util"],
                "util_cap": util_cap,
                "group_headroom": headroom,
                "target_util": target,
                "alert": bool(alert)
            }

            hw = src.get("hardware")
            if isinstance(hw, dict):
                out["hardware"] = dict(hw)

            # Save only alerting frames
            self.context_window.append_if_relevant(out, alert_key="alert")

            # Rate-limited publish
            now = time.time()
            if now - self._last_publish_ts >= self.refresh_interval_s:
                _sandbox_publish(self.bus, "telemetry.perf", out)
                self._last_publish_ts = now

            return out

    # ----------------------------------------------------------------------
    # Accessor
    # ----------------------------------------------------------------------
    def get_context_window(self) -> list[Dict[str, Any]]:
        return self.context_window.snapshot()
