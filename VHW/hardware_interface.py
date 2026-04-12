# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# NOTE: This file intentionally avoids any non-ASCII characters.
# Jarvis ADA v4.7 Strict mode. Unified rule-block tokens A..G.
# This module is part of the cleaned bundle generated for the user.
# ============================================================================

"""
File: hardware_interface.py
Purpose:
  Normalized hardware telemetry interface with pluggable adapters. Provides
  smoothed device utilization, conservative damping toward a utilization cap,
  and ascii_floatmap-friendly snapshots.

Adapters:
  Implement any subset of AdapterBase methods. The interface fills gaps with
  reasonable defaults and smoothing. A deterministic DummyAdapter is provided
  for lab or headless environments.

Public API:
  AdapterBase (abstract), DummyAdapter, HardwareInterface
  HardwareInterface.read() -> dict telemetry
  HardwareInterface.snapshot_statevector() -> dict
"""

from __future__ import annotations
import time
import math
import threading
from typing import Dict, Optional, Callable, List

# ---------------------------------------------------------------------------
# Layered imports
# ---------------------------------------------------------------------------
from VHW.system_utils import (
    ada_effective_constants,
    ada_latency_kernel,
    TierOscillator,
    device_snapshot,
    system_headroom,
    vector_from_ascii,
    store_statevector,
)

ASCII_MIN = 32
ASCII_MAX = 126

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def float_to_ascii(v: float) -> str:
    v = clamp(v, -1.0, 1.0)
    code = int((v + 1.0) * 47) + 32
    code = clamp(code, ASCII_MIN, ASCII_MAX)
    return chr(int(code))

def ascii_from_vector(vec: List[float]) -> str:
    return ''.join(float_to_ascii(x) for x in vec)

def sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class Gauge:
    def __init__(self, alpha: float = 0.20):
        self.alpha = float(alpha)
        self.value = 0.0
        self.last_ts = 0.0
    def update(self, new_val: float) -> float:
        now = time.time()
        if self.last_ts <= 0.0:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1.0 - self.alpha) * self.value
        self.last_ts = now
        return self.value

class AdapterBase:
    def read_gpu_util(self) -> Optional[float]: return None
    def read_gpu_mem_util(self) -> Optional[float]: return None
    def read_mem_bw_util(self) -> Optional[float]: return None
    def read_cpu_util(self) -> Optional[float]: return None
    def read_vram_mb(self) -> Optional[int]: return None
    def read_mem_bw_gbps(self) -> Optional[float]: return None
    def read_cpu_ghz(self) -> Optional[float]: return None

class DummyAdapter(AdapterBase):
    def __init__(self, seed: float = 0.13):
        self.t0 = time.time()
        self.seed = float(seed)
    def _osc(self, k: float, bias: float, amp: float) -> float:
        t = time.time() - self.t0
        return clamp(bias + amp * math.sin(k * t + self.seed), 0.0, 1.0)
    def read_gpu_util(self) -> Optional[float]: return self._osc(0.7, 0.64, 0.18)
    def read_gpu_mem_util(self) -> Optional[float]: return self._osc(0.5, 0.58, 0.22)
    def read_mem_bw_util(self) -> Optional[float]: return self._osc(0.6, 0.52, 0.25)
    def read_cpu_util(self) -> Optional[float]: return self._osc(0.9, 0.35, 0.15)
    def read_vram_mb(self) -> Optional[int]: return 8192
    def read_mem_bw_gbps(self) -> Optional[float]: return 448.0
    def read_cpu_ghz(self) -> Optional[float]: return 4.2

class HardwareInterface:
    """Authoritative hardware envelope for ADA v4.7.

    This class owns the normalized hardware vector and exposes it as
    a deterministic envelope that NonceMath and neural_object can
    consume via system_payload.
    """

    def __init__(
        self,
        adapter: Optional[AdapterBase] = None,
        util_cap: float = 0.75,
        group_headroom: float = 0.10,
        network_capacity_fn: Optional[Callable[[], Dict[str, float]]] = None,
        tier_ids: Optional[List[int]] = None,
    ):
        if adapter is None:
            raise ValueError("HardwareInterface requires an explicit adapter; no defaults permitted")
        self.adapter = adapter
        self.util_cap = float(util_cap)
        self.group_headroom = float(group_headroom)
        self.network_capacity_fn = network_capacity_fn or (lambda: {})
        self.gpu = Gauge(0.20)
        self.gpu_mem = Gauge(0.20)
        self.mem_bw = Gauge(0.20)
        self.cpu = Gauge(0.20)
        self.vram_mb = 8192
        self.mem_bw_gbps = 448.0
        self.cpu_ghz = 4.0
        self.phase = 0.0
        self._lock = threading.RLock()
        self._stop = False
        self._thread = None
        self._last_pub: Dict = {}
        self._tier_osc = TierOscillator(tier_ids or [0])

    def start(self, interval_s: float = 0.5):
        if self._thread: return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, args=(interval_s,), daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        if not self._thread: return
        self._stop = True
        self._thread.join(timeout=timeout)
        self._thread = None

    def _loop(self, interval_s: float):
        while not self._stop:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(max(0.05, float(interval_s)))

    def _pull(self, fn: Callable[[], Optional[float]], default):
        try:
            v = fn()
            return default if v is None else v
        except Exception:
            return default

    def _tick(self):
        with self._lock:
            gu = self._pull(self.adapter.read_gpu_util, 0.0)
            gm = self._pull(self.adapter.read_gpu_mem_util, 0.0)
            bw = self._pull(self.adapter.read_mem_bw_util, 0.0)
            cu = self._pull(self.adapter.read_cpu_util, 0.0)
            self.vram_mb = self._pull(self.adapter.read_vram_mb, self.vram_mb)
            self.mem_bw_gbps = self._pull(self.adapter.read_mem_bw_gbps, self.mem_bw_gbps)
            self.cpu_ghz = self._pull(self.adapter.read_cpu_ghz, self.cpu_ghz)
            gu = self.gpu.update(gu)
            gm = self.gpu_mem.update(gm)
            bw = self.mem_bw.update(bw)
            cu = self.cpu.update(cu)
            base = 0.45 * gu + 0.25 * gm + 0.20 * bw + 0.10 * cu
            if base > self.util_cap:
                base = self.util_cap + 0.4 * (base - self.util_cap)
            comp = clamp(base, 0.0, 1.0)
            # ADA scheduler hooks
            sh = system_headroom()
            self.phase = ada_phase_update(self.phase, float(sh.get("headroom", 0.12)))
            consts = ada_effective_constants({
                "global_util": comp,
                "gpu_util": gu,
                "mem_bw_util": bw,
                "cpu_util": cu,
            })
            density = consts["density"]
            latency = ada_latency_kernel(float(sh.get("headroom", 0.12)), density, self.phase)
            tier_weights = self._tier_osc.update(float(sh.get("headroom", 0.12)))
            net_caps = self._safe_network_caps()
            self._last_pub = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "global_util": comp,
                "gpu_util": gu,
                "gpu_mem_util": gm,
                "mem_bw_util": bw,
                "cpu_util": cu,
                "hardware": {
                    "vram_mb": int(self.vram_mb),
                    "mem_bw_gbps": float(self.mem_bw_gbps),
                    "cpu_ghz": float(self.cpu_ghz),
                },
                "network_capacity": net_caps,
                "util_cap": self.util_cap,
                "group_headroom": self.group_headroom,
                "phase": self.phase,
                "ada_constants": consts,
                "ada_latency_s": latency,
                "tier_weights": tier_weights,
            }

    def _safe_network_caps(self) -> Dict[str, float]:
        try:
            caps = self.network_capacity_fn() or {}
            out: Dict[str, float] = {}
            for k, v in caps.items():
                try:
                    out[k] = max(0.0, float(v))
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    def read(self) -> Dict:
        with self._lock:
            return dict(self._last_pub)

    def set_cap(self, new_cap: float):
        self.util_cap = clamp(float(new_cap), 0.10, 0.95)

    def inject_network_capacity(self, name: str, capacity_value: float):
        with self._lock:
            nc = dict(self._last_pub.get("network_capacity", {}))
            nc[name] = max(0.0, float(capacity_value))
            self._last_pub["network_capacity"] = nc

    def snapshot_statevector(self) -> Dict:
        d = self.read() or {}
        vec = [
            clamp(d.get("global_util", 0.0), 0.0, 1.0) * 2.0 - 1.0,
            clamp(d.get("gpu_util", 0.0), 0.0, 1.0) * 2.0 - 1.0,
            clamp(d.get("gpu_mem_util", 0.0), 0.0, 1.0) * 2.0 - 1.0,
            clamp(d.get("mem_bw_util", 0.0), 0.0, 1.0) * 2.0 - 1.0,
            clamp(d.get("cpu_util", 0.0), 0.0, 1.0) * 2.0 - 1.0,
            clamp(self.util_cap, 0.0, 1.0) * 2.0 - 1.0,
            clamp(self.group_headroom, 0.0, 1.0) * 2.0 - 1.0,
            (self.vram_mb / 16384.0) * 2.0 - 1.0,
            clamp(self.mem_bw_gbps / 1000.0, 0.0, 1.0) * 2.0 - 1.0,
            clamp(self.cpu_ghz / 8.0, 0.0, 1.0) * 2.0 - 1.0,
        ]
        tokenized = ascii_from_vector(vec)
        meta = {
            "type": "json_statevector_v3",
            "encoding": "ascii_floatmap_v1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hash": sha256(tokenized),
        }
        out = {"header": meta, "vector_ascii": tokenized}
        store_statevector("system/hardware/statevector", vec)
        return out
