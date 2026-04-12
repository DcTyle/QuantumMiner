# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: vqgpu.py
# Version: v4.8.7 Hybrid (BIOS-gated, buffered GPU command flow)
# Jarvis ADA v4.7 Hybrid Ready
# ----------------------------------------------------------------------------
# Purpose
# -------
# Virtual GPU execution module. Manages a buffered command queue and
# publishes compact telemetry snapshots at a configurable cadence.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import threading
import logging
import time
import math

# ----------------------------------------------------------------------------
# Global Logger (BIOS-style UTC logger for VHW)
# ----------------------------------------------------------------------------
_LOG = logging.getLogger("VHW.VQGPU")
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter(
        fmt="%(asctime)sZ [%(levelname)s] VHW.VQGPU: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    _h.setFormatter(_f)
    _LOG.addHandler(_h)
    logging.Formatter.converter = time.gmtime
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False

# ----------------------------------------------------------------------------
# Safe fallbacks
# ----------------------------------------------------------------------------

class _NoBus:
    def publish(self, *_a, **_k): return
    def subscribe(self, *_a, **_k): return

from VHW.system_utils import (
    ada_effective_constants,
    ada_latency_kernel,
    TierOscillator,
    system_headroom,
)


def _get_event_bus():
    try:
        from core.event_bus import get_event_bus
        return get_event_bus()
    except Exception:
        return _NoBus()

class _NoConfig:
    def get(self, key: str, default: Any = None) -> Any:
        return default

def _get_config_manager():
    try:
        from config.manager import ConfigManager
        return ConfigManager()
    except Exception:
        return _NoConfig()

class _MapVSD:
    def __init__(self): 
        self._m = {}
        self._lock = threading.RLock()

    def get(self, key, default=None):
        try:
            with self._lock:
                return self._m.get(str(key), default)
        except Exception:
            return default

    def store(self, key, value):
        try:
            with self._lock:
                self._m[str(key)] = value
        except Exception:
            pass

def _get_vsd_manager(env=None):
    if isinstance(env, dict) and env.get("vsd") is not None:
        return env["vsd"]
    try:
        from VHW.vsd_manager import VSDManager
        return VSDManager()
    except Exception:
        return _MapVSD()

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

_BIOS_KEY = "system/bios_boot_ok"

def _utc_now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _bios_ready(vsd: Any) -> bool:
    try:
        return bool(vsd.get(_BIOS_KEY, False))
    except Exception:
        return False

def _clampf(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _as_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

# ----------------------------------------------------------------------------
# VQGPU main class
# ----------------------------------------------------------------------------

class VQGPU:
    def __init__(self, env=None):
        self.vsd = _get_vsd_manager(env)
        self.cfg = _get_config_manager()
        self.bus = _get_event_bus()

        # Configurable parameters
        self.telemetry_interval_s = _as_float(self.cfg.get("vqgpu.telemetry_interval_s", 0.5), 0.5)
        self.batch_size = int(_as_float(self.cfg.get("vqgpu.batch_size", 32), 32.0))
        self.max_queue = int(_as_float(self.cfg.get("vqgpu.max_queue", 20000), 20000.0))
        self.topic_telemetry = str(self.cfg.get("vqgpu.topic_telemetry", "telemetry.vqgpu"))

        # Internal
        self._queue = []
        self._q_lock = threading.RLock()
        self._total_enqueued = 0
        self._total_processed = 0
        self._last_publish_ts = 0.0
        self._stop = False
        self._thr = None
        self._gpu_tick = None
        self._gpu_device_info = None
        self._gpu_actuate = None
        self._last_actuation = {}
        self._tier_osc = TierOscillator([0, 1, 2, 3])

        try:
            def _on_boot_complete(_evt=None):
                try:
                    self.start()
                except Exception as e:
                    _LOG.error("boot_complete handler failed: %s", e)
            self.bus.subscribe("boot.complete", _on_boot_complete)
        except Exception as e:
            _LOG.error("Failed to subscribe to boot.complete: %s", e)

        _LOG.info("init v4.8.7 Hybrid (BIOS-gated, buffered queue)")

    # ---------------------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------------------

    def set_hooks(self, gpu_tick=None, gpu_device_info=None, gpu_actuate=None):
        try:
            self._gpu_tick = gpu_tick
            self._gpu_device_info = gpu_device_info
            self._gpu_actuate = gpu_actuate
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Queue
    # ---------------------------------------------------------------------

    def enqueue(self, cmd):
        try:
            if not isinstance(cmd, dict):
                return
            with self._q_lock:
                if len(self._queue) >= self.max_queue:
                    drop = max(1, self.batch_size)
                    self._queue = self._queue[drop:]
                self._queue.append(dict(cmd))
                self._total_enqueued += 1
        except Exception as e:
            _LOG.error("enqueue failed: %s", e)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def start(self):
        try:
            if self._thr:
                return
            self._stop = False
            self._thr = threading.Thread(target=self._loop, daemon=True, name="vqgpu_worker")
            self._thr.start()
            _LOG.info("start() requested")
        except Exception as e:
            _LOG.error("start() failed: %s", e)

    def stop(self, timeout=2.0):
        try:
            if not self._thr:
                return
            self._stop = True
            self._thr.join(_clampf(timeout, 0.05, 10.0))
            self._thr = None
            _LOG.info("stop() completed")
        except Exception as e:
            _LOG.error("stop() failed: %s", e)

    # ---------------------------------------------------------------------
    # Processing
    # ---------------------------------------------------------------------

    def _drain_batch(self):
        try:
            with self._q_lock:
                if not self._queue:
                    return 0
                n = min(len(self._queue), max(1, self.batch_size))
                batch = list(self._queue[:n])
                del self._queue[:n]
                self._total_processed += n
            if callable(self._gpu_actuate):
                started = time.perf_counter()
                applied = False
                error_text = ""
                try:
                    res = self._gpu_actuate(list(batch)) or {}
                    applied = True
                    meta = dict(res) if isinstance(res, dict) else {}
                except Exception as exc:
                    meta = {}
                    error_text = str(exc)
                self._last_actuation = {
                    "applied": bool(applied),
                    "elapsed_s": float(max(0.0, time.perf_counter() - started)),
                    "batch_size": int(n),
                    "tag": str(meta.get("tag", "hook")),
                    "mode": str(meta.get("mode", "hook")),
                    "error": str(error_text),
                }
            else:
                self._last_actuation = {
                    "applied": False,
                    "elapsed_s": 0.0,
                    "batch_size": int(n),
                    "tag": "",
                    "mode": "passive",
                    "error": "",
                }
            return n
        except Exception as e:
            _LOG.error("drain_batch failed: %s", e)
            return 0

    # ---------------------------------------------------------------------
    # Telemetry
    # ---------------------------------------------------------------------

    def _merge_device_metrics(self, snap):
        try:
            if callable(self._gpu_tick):
                extra = self._gpu_tick() or {}
                if isinstance(extra, dict):
                    snap.update({str(k): extra[k] for k in extra})
        except Exception:
            pass
        return snap

    def _publish_telemetry(self, now_ts):
        try:
            with self._q_lock:
                q_len = len(self._queue)
                total_in = self._total_enqueued
                total_out = self._total_processed

            snap = {
                "ts": _utc_now_str(),
                "queue_len": q_len,
                "total_enqueued": total_in,
                "total_processed": total_out,
                "batch_size": self.batch_size,
                "interval_s": self.telemetry_interval_s,
            }
            snap = self._merge_device_metrics(snap)
            sh = system_headroom()
            consts = ada_effective_constants({
                "global_util": float(sh.get("global_util", 0.0)),
                "gpu_util": float(sh.get("gpu_util", 0.0)),
                "mem_bw_util": float(sh.get("mem_util", 0.0)),
                "cpu_util": float(sh.get("cpu_util", 0.0)),
            })
            weights = self._tier_osc.update(float(sh.get("headroom", 0.12)))
            snap["ada_constants"] = consts
            snap["tier_weights"] = weights

            try:
                self.bus.publish(self.topic_telemetry, snap)
            except Exception as e:
                _LOG.error("telemetry publish failed: %s", e)

            try:
                self.vsd.store("telemetry/vqgpu/current", snap)
            except Exception as e:
                _LOG.error("telemetry persist failed: %s", e)

            self._last_publish_ts = now_ts
        except Exception as e:
            _LOG.error("publish telemetry failed: %s", e)

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    def _loop(self):
        device_info_written = False

        while not self._stop:
            try:
                if not _bios_ready(self.vsd):
                    time.sleep(0.25)
                    continue

                if not device_info_written:
                    try:
                        if callable(self._gpu_device_info):
                            info = self._gpu_device_info() or {}
                            if info:
                                self.vsd.store("telemetry/vqgpu/device_info", {str(k): info[k] for k in info})
                    except Exception as e:
                        _LOG.error("device_info write failed: %s", e)
                    device_info_written = True

                sh = system_headroom()
                consts = ada_effective_constants({
                    "global_util": float(sh.get("global_util", 0.0)),
                    "gpu_util": float(sh.get("gpu_util", 0.0)),
                    "mem_bw_util": float(sh.get("mem_util", 0.0)),
                    "cpu_util": float(sh.get("cpu_util", 0.0)),
                })
                density = consts["density"]
                latency = ada_latency_kernel(float(sh.get("headroom", 0.12)), density, self._tier_osc.phase)

                processed = self._drain_batch()
                if processed == 0:
                    time.sleep(latency)

                now_ts = time.time()
                if now_ts - self._last_publish_ts >= _clampf(self.telemetry_interval_s, 0.05, 10.0):
                    self._publish_telemetry(now_ts)

            except Exception as e:
                _LOG.error("vqgpu loop error: %s", e)
                time.sleep(0.05)

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------

    def stats(self):
        try:
            with self._q_lock:
                return {
                    "ts": _utc_now_str(),
                    "running": bool(self._thr is not None),
                    "queue_len": len(self._queue),
                    "total_enqueued": self._total_enqueued,
                    "total_processed": self._total_processed,
                    "batch_size": self.batch_size,
                    "interval_s": self.telemetry_interval_s,
                    "topic": self.topic_telemetry,
                    "last_actuation_applied": bool(dict(self._last_actuation or {}).get("applied", False)),
                    "last_actuation_elapsed_s": float(dict(self._last_actuation or {}).get("elapsed_s", 0.0)),
                    "last_actuation_batch_size": int(dict(self._last_actuation or {}).get("batch_size", 0)),
                    "last_actuation_mode": str(dict(self._last_actuation or {}).get("mode", "")),
                    "last_actuation_tag": str(dict(self._last_actuation or {}).get("tag", "")),
                    "last_actuation_error": str(dict(self._last_actuation or {}).get("error", "")),
                }
        except Exception as e:
            return {"error": str(e)}

# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    vsd = _MapVSD()
    vsd.store(_BIOS_KEY, True)
    gpu = VQGPU(env={"vsd": vsd})

    def demo_tick(): return {"gpu_util": 0.42}
    def demo_info(): return {"name": "Virtual GPU", "driver": "vqgpu"}

    gpu.set_hooks(gpu_tick=demo_tick, gpu_device_info=demo_info)
    for i in range(50): gpu.enqueue({"op": "kernel", "id": i})
    gpu.start(); time.sleep(1); print(gpu.stats()); gpu.stop()
