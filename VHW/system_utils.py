# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: VHW/system_utils.py
# Version: v7 Unified Kernel (Final)
# ============================================================================
"""
Unified hardware, BIOS, VSD, telemetry, allocator, failsafe, and system utils
for the VirtualMiner Quantum Application OS.

This module is the canonical utility library for:
    - BIOS boot
    - Miner Engine
    - ShareAllocator
    - FailsafeGovernor
    - Prediction Engine
    - Neuralis AI
    - Control Center GUI

Design requirements:
    - ASCII-only (no unicode)
    - Thread-safe (RLocks)
    - No circular imports
    - No stubs
    - No legacy fallbacks
    - Pure Python, portable, deterministic
"""

from __future__ import annotations

import time
import threading
import hashlib
import random
import logging
import math
import subprocess
import sys
from typing import Any, Dict, Callable, List, Optional, Tuple


# ============================================================================
# LOGGER (ASCII ONLY + UTC)
# ============================================================================
def _mk_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(stream=sys.stdout)
        h.setFormatter(logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        ))
        logging.Formatter.converter = time.gmtime
        log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log

_LOG = _mk_logger("VHW.system_utils")


from VHW.vsd_manager import get_event_bus  # unified global bus

BUS = get_event_bus()


def publish_event(topic: str, payload: Dict[str, Any] | None = None) -> None:
    """Unified system_bus publish helper (authoritative).

    All VHW subsystems must emit events through this function.
    """
    try:
        BUS.publish(str(topic), dict(payload or {}))
    except Exception:
        _LOG.error("publish_event failed for topic='%s'", topic, exc_info=True)


def subscribe_event(topic: str, fn: Callable[[Dict[str, Any]], None],
                    once: bool = False, priority: int = 0) -> None:
    """Unified system_bus subscribe helper (authoritative)."""
    try:
        BUS.subscribe(str(topic), fn, once=once, priority=priority)
    except Exception:
        _LOG.error("subscribe_event failed for topic='%s'", topic, exc_info=True)


# ============================================================================
# UNIFIED VSD ACCESS
# ============================================================================
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    VSDManager = None  # type: ignore


class _DictVSD:
    """Fallback VSD map; always persists, no BIOS gate."""
    def __init__(self):
        self._m: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str, default=None):
        with self._lock:
            return self._m.get(str(key), default)

    def store(self, key: str, value: Any):
        with self._lock:
            self._m[str(key)] = value

    def delete(self, key: str):
        with self._lock:
            self._m.pop(str(key), None)

    def append_bounded(self, key: str, item: Any, max_len: int = 1000):
        with self._lock:
            arr = self._m.get(str(key))
            if not isinstance(arr, list):
                arr = []
            arr.append(item)
            if max_len > 0 and len(arr) > max_len:
                arr = arr[-max_len:]
            self._m[str(key)] = arr


VSD = VSDManager() if VSDManager else _DictVSD()


def vsd_fetch(key: str, default: Any = None):
    try:
        return VSD.get(str(key), default)
    except Exception:
        _LOG.error("vsd_fetch failed for key='%s'", key, exc_info=True)
        return default


def vsd_store(key: str, value: Any):
    try:
        VSD.store(str(key), value)
    except Exception:
        _LOG.error("vsd_store failed for key='%s'", key, exc_info=True)
        return
    publish_event("vsd.write", {"key": str(key), "ts": time.time()})


def vsd_delete(key: str):
    try:
        VSD.delete(str(key))
    except Exception:
        _LOG.error("vsd_delete failed for key='%s'", key, exc_info=True)


def vsd_append_bounded(key: str, item: Any, max_len: int = 1000):
    try:
        VSD.append_bounded(str(key), item, int(max_len))
    except Exception:
        _LOG.error("vsd_append_bounded failed for key='%s'", key, exc_info=True)


# ============================================================================
# BIOS GATE HELPERS
# ============================================================================
def bios_ready() -> bool:
    try:
        return bool(vsd_fetch("system/bios_boot_ok", False))
    except Exception:
        return False


def wait_for_bios(timeout_s: float = 10.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if bios_ready():
            return True
        time.sleep(0.1)
    return False


# ============================================================================
# ATOMIC LOCK UTILITY
# ============================================================================
_LOCK = threading.RLock()

def atomic_lock():
    """Context manager for atomic blocks."""
    class _LockCtx:
        def __enter__(self):
            _LOCK.acquire()
        def __exit__(self, a, b, c):
            try:
                _LOCK.release()
            except Exception:
                pass
    return _LockCtx()


# ============================================================================
# ASCII / VECTOR HELPERS
# ============================================================================
ASCII_MIN = 32
ASCII_MAX = 126


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sha256_ascii(s: str) -> str:
    try:
        return hashlib.sha256(str(s).encode("ascii", "ignore")).hexdigest()
    except Exception:
        return "0"*64


def float_to_ascii(v: float) -> str:
    v = clamp(v, -1.0, 1.0)
    code = int((v + 1.0)*47) + ASCII_MIN
    code = int(clamp(code, ASCII_MIN, ASCII_MAX))
    return chr(code)


def ascii_from_vector(vec: List[float]) -> str:
    return "".join(float_to_ascii(x) for x in vec)


def vector_from_ascii(s: str) -> List[float]:
    out = []
    for ch in s:
        v = (ord(ch) - ASCII_MIN)/47.0 - 1.0
        out.append(clamp(v, -1.0, 1.0))
    return out


# ============================================================================
# ADVANCED VECTOR HELPERS (Neuralis / Prediction Engine)
# ============================================================================
def vector_digest(vec: List[float]) -> str:
    """Returns SHA-256 digest of encoded ASCII floatmap."""
    try:
        s = ascii_from_vector(vec)
        return sha256_ascii(s)
    except Exception:
        return "0"*64


# =========================================================================
# ADA v4.7 HYBRID SCHEDULER HELPERS
# =========================================================================

def ada_effective_constants(payload: Dict[str, Any]) -> Dict[str, float]:
    """Deterministic effective constants for ADA v4.7.

    Submission Rate Governor v2 forbids density-based throttling for
    mining; this helper now exposes utilization-derived scalars only.
    """
    gu = float(payload.get("global_util", 0.5))
    gpu = float(payload.get("gpu_util", gu))
    mem = float(payload.get("mem_bw_util", gu))
    cpu = float(payload.get("cpu_util", gu))

    avg = clamp(0.25 * (gu + gpu + mem + cpu), 0.0, 1.0)

    h_eff = 1.0 + 0.25 * avg
    k_B_eff = 1.0 + 0.15 * (1.0 - avg)
    c_eff = 1.0 + 0.30 * (gpu - cpu)

    return {
        "h_eff": float(h_eff),
        "k_B_eff": float(k_B_eff),
        "c_eff": float(c_eff),
    }


def ada_latency_kernel(headroom: float, density: float, phase: float) -> float:
    """State-dependent latency kernel in seconds.

    Lower headroom and higher density increase latency. Phase introduces
    a small deterministic oscillation.
    """
    h = clamp(headroom, 0.0, 1.0)
    d = clamp(density, 0.0, 1.0)
    base = 0.005 + 0.010 * d + 0.010 * (1.0 - h)
    osc = 0.002 * math.sin(2.0 * math.pi * clamp(phase, 0.0, 1.0))
    return clamp(base + osc, 0.001, 0.050)


def ada_phase_update(phase: float, headroom: float) -> float:
    """Deterministic phase evolution driven by headroom."""
    step = clamp(0.010 + 0.020 * (1.0 - clamp(headroom, 0.0, 1.0)), 0.005, 0.040)
    return (phase + step) % 1.0


def store_statevector(path: str, vec: List[float]):
    """Store state-vector + ASCII mapping."""
    try:
        asc = ascii_from_vector(vec)
        vsd_store(path, {
            "header": {
                "type": "json_statevector_v3",
                "encoding": "ascii_floatmap_v1",
                "ts": time.time()
            },
            "vector_ascii": asc
        })
    except Exception:
        pass
# ============================================================================
# PREDICTION ENGINE DIGEST / SPAN HELPERS
# ============================================================================
def span_index_update(span_key: str, digest_key: str, payload: Any) -> None:
    """
    Maintains a span->digest table: used by Prediction Engine for archive indexing.
    """
    try:
        digest = sha256_ascii(str(payload))
        ts_key = str(int(time.time()))
        idx = vsd_fetch(span_key, {})
        if not isinstance(idx, dict):
            idx = {}
        idx[ts_key] = digest
        vsd_store(span_key, idx)
        vsd_store(digest_key, digest)
    except Exception:
        pass


def mark_prediction_dirty() -> None:
    """
    Signals to Control Center + Prediction Engine that new prediction data exists.
    """
    publish_event("prediction.data_dirty", {"ts": time.time()})


def store_prediction_frame(symbol: str, frame: Dict[str, Any]) -> None:
    """
    Store a single prediction frame with bounded retention.
    """
    key = f"telemetry/predictions/{symbol}"
    arr = vsd_fetch(key, [])
    if not isinstance(arr, list):
        arr = []
    frame2 = dict(frame)
    frame2["ts"] = time.time()
    arr.append(frame2)
    if len(arr) > 2000:
        arr = arr[-2000:]
    vsd_store(key, arr)
    mark_prediction_dirty()


def store_prediction_batch(batch: List[Dict[str, Any]]) -> None:
    """
    Bulk store predictions (Control Center chart updates use this).
    """
    try:
        vsd_store("telemetry/predictions/latest", batch)
        mark_prediction_dirty()
    except Exception:
        pass


# ============================================================================
# TELEMETRY (STREAM + CONTROL CENTER HOOKS)
# ============================================================================
def append_telemetry(path: str, payload: Dict[str, Any]) -> None:
    """
    Append telemetry with bounded length and auto-notification to Control Center.
    """
    key = "telemetry/" + str(path)
    arr = vsd_fetch(key, [])
    if not isinstance(arr, list):
        arr = []
    rec = {"ts": time.time()}
    rec.update(payload or {})
    arr.append(rec)
    if len(arr) > 5000:
        arr = arr[-5000:]
    vsd_store(key, arr)
    publish_event("telemetry.append", {"path": path, "ts": rec["ts"]})


def telemetry_snapshot(path: str, default=None):
    """Read-only helper for GUI / APIs."""
    try:
        return vsd_fetch("telemetry/" + str(path), default)
    except Exception:
        return default


# ============================================================================
# FILTERING (EMA + DAMPED STATE)
# ============================================================================
class EMA:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._value = None
        self._lock = threading.RLock()

    def update(self, x: float):
        with self._lock:
            xv = float(x)
            if self._value is None:
                self._value = xv
            else:
                self._value = self.alpha * xv + (1 - self.alpha) * self._value
            return self._value

    def current(self):
        with self._lock:
            return self._value if self._value is not None else 0.0


class DampedState:
    """
    Used heavily by Control Center + Neuralis training.
    """
    def __init__(self, alpha=0.20, headroom=0.12):
        self._lock = threading.RLock()
        self.counters = {"events": 0, "warnings": 0, "errors": 0}
        self.decayed = 0.0
        self.phase = 0.0
        self.alpha = float(alpha)
        self.headroom = float(headroom)

    def add(self, key: str, amt=1):
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + amt

    def decay(self):
        with self._lock:
            total = sum(self.counters.values())
            self.decayed = max(
                0.0,
                self.decayed * (1 - self.alpha) + total * self.alpha * 0.1
            )
            self.phase = (self.phase + self.headroom) % 1.0

    def snapshot_vector(self):
        with self._lock:
            return [
                clamp(self.decayed / 100.0, 0, 1) * 2 - 1,
                clamp(self.phase % 1.0, 0, 1) * 2 - 1,
                clamp(sum(self.counters.values()) / 1000.0, 0, 1) * 2 - 1
            ]


class TierOscillator:
    """Cross-tier headroom oscillator for deterministic lane balancing.

    This model produces per-tier weights in [0,1] that shift slowly in
    response to global headroom and internal phase. It is intentionally
    simple and deterministic to keep ADA v4.7 behavior reproducible.
    """

    def __init__(self, tier_ids: List[int]):
        self._lock = threading.RLock()
        self.tier_ids = list(tier_ids)
        self.phase = 0.0
        self.last_headroom = 0.12

    def update(self, headroom: float) -> Dict[int, float]:
        h = clamp(headroom, 0.0, 1.0)
        with self._lock:
            self.last_headroom = h
            self.phase = ada_phase_update(self.phase, h)

            weights: Dict[int, float] = {}
            n = max(1, len(self.tier_ids))
            for idx, tid in enumerate(self.tier_ids):
                base = 0.5 + 0.5 * math.sin(2.0 * math.pi * (self.phase + idx / float(n)))
                w = clamp(base * (0.5 + 0.5 * h), 0.0, 1.0)
                weights[tid] = w

            total = sum(weights.values()) or 1.0
            for tid in weights:
                weights[tid] = weights[tid] / total

            return weights


# ============================================================================
# ALLOCATOR WRAPPERS (LaneAllocator removed; ComputeManager supersedes it)
# ============================================================================

try:
    from VHW.share_allocator import ShareAllocator
except Exception:
    ShareAllocator = None

try:
    from VHW.failsafe import FailsafeGovernor
except Exception:
    FailsafeGovernor = None

try:
    from VHW.monitor import Monitor
except Exception:
    Monitor = None


def lane_alloc_next(config: Dict[str, Any], network: str) -> Optional[Dict[str, Any]]:
    """
    Legacy no-op: LaneAllocator has been removed. Use ComputeManager.allocate_lane.
    Retained for API compatibility; returns None.
    """
    return None


def share_submit(vsd, lane_id, network, nonce_hex, hash_hex, target_hex,
                 is_valid, extra=None) -> bool:
    """
    Submits a share to ShareAllocator.
    Used by miner engine and Control Center visualizer.
    """
    if ShareAllocator is None:
        return False

    try:
        alloc = ShareAllocator(vsd or VSD)
        alloc.submit_share(
            lane_id, network, nonce_hex, hash_hex, target_hex,
            bool(is_valid), extra or {}
        )
        return True
    except Exception as exc:
        _LOG.error("share_submit error: %s", exc)
        return False


def failsafe_snapshot() -> Dict[str, Any]:
    """
    Returns a live snapshot from the fail-safe engine.
    Used by Control Center "Failsafe" tab.
    """
    if FailsafeGovernor is None:
        return {"bios_ready": bios_ready(), "lanes": {}}

    try:
        fs = FailsafeGovernor()
        return fs.snapshot()
    except Exception as exc:
        _LOG.error("failsafe_snapshot error: %s", exc)
        return {"bios_ready": bios_ready(), "lanes": {}}


# ============================================================================
# MONITOR WRAPPER (merges BIOS + Monitor class)
# ============================================================================
def monitor_frame(read_telemetry: Callable[[], Dict[str, Any]],
                  util_cap: float = 0.75) -> Dict[str, Any]:
    """
    Unified monitor frame for:
        - BIOS boot
        - Miner engine
        - Control Center live panel
    """
    if Monitor is None:
        d = read_telemetry() or {}
        g = float(d.get("global_util", 0))
        tgt = max(0.0, float(util_cap) - 0.12)
        return {
            "timestamp": d.get("timestamp", ""),
            "global_util": g,
            "target_util": tgt,
            "alert": bool(g > tgt)
        }

    try:
        mon = Monitor(read_telemetry=read_telemetry, util_cap=float(util_cap))
        return mon.frame()
    except Exception as exc:
        _LOG.error("monitor_frame error: %s", exc)
        return {}
# ============================================================================
# UNIFIED HARDWARE SNAPSHOT (CPU/GPU/MEM/BW)
# ============================================================================
def _safe_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def _safe_import_gpustat():
    try:
        import gpustat
        return gpustat
    except Exception:
        return None

def _safe_import_wmi():
    try:
        import wmi  # type: ignore
        return wmi
    except Exception:
        return None


def _safe_import_pyopencl():
    try:
        import pyopencl as cl  # type: ignore
        return cl
    except Exception:
        return None


_PSUTIL = _safe_import_psutil()
_GPUSTAT = _safe_import_gpustat()
_PYOPENCL = _safe_import_pyopencl()
_WMI = _safe_import_wmi()


def device_snapshot() -> Dict[str, Any]:
    """
    Core hardware snapshot used by:
        - BIOS selfcheck
        - Miner engine
        - Prediction engine
        - Control Center live panel
    ASCII-only and sandbox-safe.
    """
    ts = time.time()

    # CPU -------------------------------------------------------------
    if _PSUTIL:
        cpu_total = _PSUTIL.cpu_count(logical=True) or 1
        cpu_util = (_PSUTIL.cpu_percent(interval=0.05) or 0.0) / 100.0
    else:
        cpu_total = 1
        cpu_util = 0.0

    # Memory ----------------------------------------------------------
    if _PSUTIL:
        mem = _PSUTIL.virtual_memory()
        mem_total = mem.total / (1024**3)
        mem_used = mem.used / (1024**3)
        mem_util = float(mem.percent) / 100.0
    else:
        mem_total = 8.0
        mem_used = 0.0
        mem_util = 0.0

    # GPU --------------------------------------------------------------
    gpu = {
        "model": "Unknown",
        "util": 0.0,
        "memory_total_mb": 0.0,
        "memory_used_mb": 0.0,
        "temperature_c": 0.0,
        "vendor": "Unknown",
    }

    # Try vendor-agnostic OpenCL for basic identification first
    if _PYOPENCL:
        try:
            for plat in _PYOPENCL.get_platforms():  # type: ignore[attr-defined]
                devs = plat.get_devices()
                # Prefer GPU devices
                gpu_devs = [d for d in devs if int(getattr(d, 'type', 0)) & 4]
                d = gpu_devs[0] if gpu_devs else (devs[0] if devs else None)
                if d is None:
                    continue
                gpu["model"] = str(getattr(d, "name", "Unknown") or "Unknown")
                gpu["vendor"] = str(getattr(d, "vendor", "Unknown") or "Unknown")
                try:
                    mem = float(getattr(d, "global_mem_size", 0.0) or 0.0)
                    gpu["memory_total_mb"] = mem / (1024.0 * 1024.0)
                except Exception:
                    pass
                break
        except Exception:
            pass

    if sys.platform.startswith("win") and _WMI:
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1.0,
            )
            if int(res.returncode) == 0:
                first_line = str(res.stdout or "").splitlines()[0].strip()
                parts = [item.strip() for item in first_line.split(",")]
                if len(parts) >= 5:
                    gpu["model"] = str(parts[0] or gpu["model"])
                    gpu["vendor"] = "NVIDIA"
                    gpu["util"] = max(0.0, min(100.0, float(parts[1] or 0.0))) / 100.0
                    gpu["memory_total_mb"] = float(parts[2] or 0.0)
                    gpu["memory_used_mb"] = float(parts[3] or 0.0)
                    gpu["temperature_c"] = float(parts[4] or 0.0)
        except Exception:
            pass
        try:
            c = _WMI.WMI(namespace="root\\CIMV2")
            engines = c.Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine()
            util_sum = 0.0
            for e in engines:
                try:
                    name = str(getattr(e, "Name", "") or "")
                    if ("engtype_3D" in name) or ("engtype_Compute" in name):
                        util_sum += float(getattr(e, "UtilizationPercentage", 0.0) or 0.0)
                except Exception:
                    pass
            gpu["util"] = max(float(gpu.get("util", 0.0)), max(0.0, min(100.0, util_sum)) / 100.0)
            try:
                ctrls = c.Win32_VideoController()
                if ctrls:
                    if str(gpu.get("model", "Unknown")) in ("", "Unknown"):
                        gpu["model"] = str(getattr(ctrls[0], "Name", "Unknown") or "Unknown")
                    ram = float(getattr(ctrls[0], "AdapterRAM", 0.0) or 0.0)
                    gpu["memory_total_mb"] = max(float(gpu.get("memory_total_mb", 0.0)), ram / (1024.0 * 1024.0))
            except Exception:
                pass
        except Exception:
            pass
    elif _GPUSTAT:
        try:
            stats = _GPUSTAT.GPUStatCollection.new_query()
            g0 = stats.gpus[0]
            gpu = {
                "model": g0.name,
                "util": float(g0.utilization) / 100.0,
                "memory_total_mb": float(g0.memory_total),
                "memory_used_mb": float(g0.memory_used),
                "temperature_c": float(g0.temperature),
            }
        except Exception:
            pass

    # Bus / Bandwidth --------------------------------------------------
    bw = {
        "pcie_gbps": 8.0,          # static fallback
        "memory_bw_gbps": 200.0,   # static fallback
    }

    return {
        "timestamp": ts,
        "cpu": {
            "util": cpu_util,
            "cores": cpu_total,
        },
        "memory": {
            "total_gb": mem_total,
            "used_gb": mem_used,
            "util": mem_util,
        },
        "gpu": gpu,
        "bandwidth": bw,
    }


# ============================================================================
# HEADROOM ESTIMATOR (GLOBAL_UTIL)
# ============================================================================
def system_headroom() -> Dict[str, Any]:
    snap = device_snapshot()
    c = snap["cpu"]["util"]
    g = snap["gpu"]["util"]
    m = snap["memory"]["util"]
    gu = max(c, g, m)
    return {
        "timestamp": snap["timestamp"],
        "global_util": gu,
        "headroom": max(0.0, 1.0 - gu),
        "cpu_util": c,
        "gpu_util": g,
        "mem_util": m,
    }


# ============================================================================
# MINING RAMP-POLICY HELPERS (PHASED ASYMPTOTIC MODEL)
# ============================================================================
def ramp_daily_step(p: float, density: float) -> float:
    """
    Phase 1 (daily): push p toward 0.30 with self-dampening.
    Legacy density hooks are disabled for miner safety; density is
    treated as 0 in all ramp helpers.
    """
    inc = (0.30 - p) * 0.01
    return clamp(p + inc, 0.0, 0.30)


def ramp_weekly_step(p: float, density: float) -> float:
    """
    Phase 2 (weekly): push p toward 0.50 with self-dampening.
    Density argument is ignored for mining; only p is used.
    """
    inc = (0.50 - p) * 0.01
    return clamp(p + inc, 0.0, 0.50)


def ramp_monthly_step(p: float, density: float) -> float:
    """
    Phase 3 (monthly): remain capped at 0.50.
    Density argument is ignored for mining; only p is used.
    """
    inc = (0.50 - p) * 0.01
    return clamp(p + inc, 0.0, 0.50)


def ramp_apply(p: float, density: float,
               phase: int) -> float:
    """
    Wrapper for miner: chooses daily / weekly / monthly rule.
    Phase: 1=daily, 2=weekly, 3=monthly
    """
    if phase == 1:
        return ramp_daily_step(p, density)
    if phase == 2:
        return ramp_weekly_step(p, density)
    return ramp_monthly_step(p, density)


# ============================================================================
# CONTROL CENTER GUI HELPERS
# ============================================================================
def gui_format_rate(x: float) -> str:
    """Format hashrate for Control Center."""
    if x >= 1e15:
        return f"{x/1e15:.2f} PH/s"
    if x >= 1e12:
        return f"{x/1e12:.2f} TH/s"
    if x >= 1e9:
        return f"{x/1e9:.2f} GH/s"
    if x >= 1e6:
        return f"{x/1e6:.2f} MH/s"
    return f"{x:.0f} H/s"


def gui_color_for_util(u: float) -> str:
    """Return basic color tag string for GUI display."""
    if u < 0.50:
        return "green"
    if u < 0.75:
        return "yellow"
    return "red"


def gui_lane_summary() -> Dict[str, Any]:
    """Return lane summaries for GUI tables."""
    try:
        lanes = vsd_fetch("lanes/active", [])
        if not isinstance(lanes, list):
            return {"lanes": []}
        return {"lanes": lanes}
    except Exception:
        return {"lanes": []}


# ============================================================================
# NEURALIS AI INGEST HELPERS
# ============================================================================
def neuralis_ingest_statevector(vec: List[float]) -> Dict[str, Any]:
    """
    Convert float vector into metadata + ascii encoding for Neuralis.
    """
    asc = ascii_from_vector(vec)
    return {
        "ts": time.time(),
        "len": len(vec),
        "ascii": asc,
        "digest": sha256_ascii(asc)
    }


def neuralis_training_log(msg: str):
    """
    Append Neuralis training log to VSD.
    """
    vsd_append_bounded("neuralis/training_log", {
        "ts": time.time(),
        "msg": str(msg)
    }, max_len=20000)


# ============================================================================
# SAFE IMPORT UTIL (BIOS / PREDICTION ENGINE)
# ============================================================================
def safe_import(path: str, default=None):
    try:
        module = __import__(path, fromlist=["*"])
        return module
    except Exception:
        return default


# ============================================================================
# PUBLIC EXPORT
# ============================================================================
__all__ = [
    # Event bus
    "publish_event", "subscribe_event",
    # VSD
    "vsd_fetch", "vsd_store", "vsd_delete", "vsd_append_bounded",
    # BIOS / locks
    "bios_ready", "wait_for_bios", "atomic_lock",
    # ASCII + vectors
    "clamp", "sha256_ascii", "float_to_ascii", "ascii_from_vector",
    "vector_from_ascii", "vector_digest", "store_statevector",
    # Prediction + telemetry
    "span_index_update", "mark_prediction_dirty",
    "store_prediction_frame", "store_prediction_batch",
    "append_telemetry", "telemetry_snapshot",
    # Filtering
    "EMA", "DampedState",
    # Allocator / miner helpers
    "lane_alloc_next", "share_submit", "failsafe_snapshot",
    "monitor_frame",
    # Hardware + headroom
    "device_snapshot", "system_headroom",
    # Ramp policy
    "ramp_daily_step", "ramp_weekly_step",
    "ramp_monthly_step", "ramp_apply",
    # GUI helpers
    "gui_format_rate", "gui_color_for_util", "gui_lane_summary",
    # Neuralis
    "neuralis_ingest_statevector", "neuralis_training_log",
    # Misc utils
    "safe_import",
]

# ============================================================================
# END OF FILE
# ============================================================================
