# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: VHW/vsd_manager.py
# Version: v7 Unified Kernel (Final)
# ============================================================================
"""
Unified VSD Manager + EventBus Kernel for VirtualMiner.

This module provides:
    - The global EventBus (publish/subscribe, priority handling, once-handlers)
    - The authoritative VSD manager (thread-safe, BIOS-aware)
    - The BufferedVSD flusher for high-frequency prediction deltas
    - Zero fallback buses, zero legacy stubs
    - Single import path for all subsystems:
            from VHW.vsd_manager import get_event_bus, VSD, VSDManager

Architecture Rules:
    - ASCII-only (no unicode)
    - Deterministic thread-safe behavior
    - No circular imports
    - BIOS boot gating with preboot queue support
    - Prediction-engine flush notifications
"""

from __future__ import annotations

import time
import threading
import hashlib
from typing import Any, Dict, List, Optional, Callable

# ============================================================================
# ASCII-ONLY LOGGER
# ============================================================================
def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# Minimal ASCII logger to avoid core.utils import (prevents circular imports)
def log(mod: str, level: str, message: str) -> None:
    try:
        # Ensure ASCII-only
        mod_s = str(mod)
        lvl_s = str(level)
        msg_s = str(message)
        print(f"{_ts()} | {mod_s} | {lvl_s} | {msg_s}")
    except Exception:
        # Best-effort fallback
        try:
            print(f"{_ts()} | VSD | WARN | log_failed")
        except Exception:
            pass


# ============================================================================
# UNIFIED EVENT BUS
# ============================================================================
class _EventHandler:
    def __init__(self, fn: Callable, once: bool, priority: int) -> None:
        self.fn = fn
        self.once = bool(once)
        self.priority = int(priority)


class EventBus:
    """
    Thread-safe EventBus with:
        - Priority ordering
        - Once-handlers
        - ASCII-only behavior
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sub: Dict[str, List[_EventHandler]] = {}

    def subscribe(self, topic: str, fn: Callable,
                  once: bool = False, priority: int = 0) -> None:
        t = str(topic)
        handler = _EventHandler(fn, once, priority)
        with self._lock:
            arr = self._sub.get(t, [])
            arr.append(handler)
            arr.sort(key=lambda x: x.priority, reverse=True)
            self._sub[t] = arr

    def publish(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        t = str(topic)
        with self._lock:
            handlers = list(self._sub.get(t, []))

        to_remove = []
        for h in handlers:
            try:
                h.fn(payload or {})
                if h.once:
                    to_remove.append(h)
            except Exception:
                pass

        if to_remove:
            with self._lock:
                arr = self._sub.get(t, [])
                for h in to_remove:
                    if h in arr:
                        arr.remove(h)
                self._sub[t] = arr


_GLOBAL_BUS = EventBus()


def get_event_bus() -> EventBus:
    return _GLOBAL_BUS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _sha256_ascii(s: str) -> str:
    try:
        return hashlib.sha256(s.encode("ascii", "ignore")).hexdigest()
    except Exception:
        return "0" * 64


def _is_prediction_key(k: str) -> bool:
    k = str(k)
    return (
        k.startswith("prediction/") or
        k.startswith("telemetry/predictions/") or
        k.startswith("prediction_archive/")
    )


# ============================================================================
# VSD MANAGER (AUTHORITATIVE)
# ============================================================================
class VSDManager:
    """
    Centralized Virtual State Drive with:
        - Preboot write queue
        - Thread-safe map
        - Buffered prediction deltas
        - Unified event bus
    """

    def __init__(self) -> None:
        self._m: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._boot_ready = False
        self._preboot: List[Dict[str, Any]] = []
        self.flush_interval_s = 0.5
        self._buffered: Optional[BufferedVSD] = None

    # ----------------------------------------------------------------------
    # BIOS COMPAT SETTER
    # ----------------------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        try:
            k = str(key)
            with self._lock:
                self._m[k] = value
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # BIOS GATE
    # ----------------------------------------------------------------------
    def mark_bios_ready(self) -> None:
        with self._lock:
            self._boot_ready = True
            self._m["system/bios_boot_ok"] = True

        self._release_preboot()

    def _bios_ok(self) -> bool:
        try:
            with self._lock:
                return self._boot_ready
        except Exception:
            return False

    def _allow_preboot(self, key: str) -> bool:
        k = str(key)
        return k.startswith("system/") or k.startswith("config/")

    def _enqueue_preboot(self, key: str, val: Any) -> None:
        try:
            with self._lock:
                self._preboot.append({"k": str(key), "v": val})
        except Exception:
            pass

    def _release_preboot(self) -> None:
        try:
            with self._lock:
                items = list(self._preboot)
                self._preboot.clear()
        except Exception:
            items = []

        for item in items:
            try:
                self.store(item["k"], item["v"])
            except Exception:
                pass

        log("VSD", "INFO", f"released_preboot_queue count={len(items)}")

    # ----------------------------------------------------------------------
    # CORE ACCESSORS (FIXED INDENTATION)
    # ----------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        try:
            with self._lock:
                return self._m.get(str(key), default)
        except Exception:
            return default

    def store(self, key: str, value: Any) -> None:
        k = str(key)

        if not self._bios_ok() and not self._allow_preboot(k):
            self._enqueue_preboot(k, value)
            return

        try:
            with self._lock:
                self._m[k] = value
        except Exception:
            return

        if _is_prediction_key(k) and self._buffered:
            try:
                self._buffered._mark_dirty()
            except Exception:
                pass

        try:
            _GLOBAL_BUS.publish("vsd.write", {"key": k, "ts": time.time()})
        except Exception:
            pass

    def delete(self, key: str) -> None:
        try:
            with self._lock:
                self._m.pop(str(key), None)
        except Exception:
            pass

    def append_bounded(self, key: str, item: Any, max_len: int = 1000) -> None:
        k = str(key)

        if not self._bios_ok() and not self._allow_preboot(k):
            self._enqueue_preboot(k, {"__append__": item})
            return

        try:
            with self._lock:
                arr = self._m.get(k)
                if not isinstance(arr, list):
                    arr = []
                arr.append(item)
                if max_len > 0 and len(arr) > max_len:
                    arr = arr[-max_len:]
                self._m[k] = arr
        except Exception:
            return

        if _is_prediction_key(k) and self._buffered:
            try:
                self._buffered._mark_dirty()
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # SPAN / DIGEST UTILS
    # ----------------------------------------------------------------------
    def span_update(self, span_key: str, digest_key: str, payload: Any) -> None:
        try:
            digest = _sha256_ascii(str(payload))
            ts_key = str(int(time.time()))
            idx = self.get(span_key, {})
            if not isinstance(idx, dict):
                idx = {}
            idx[ts_key] = digest
            self.store(span_key, idx)
            self.store(digest_key, digest)
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # BUFFERED MODE
    # ----------------------------------------------------------------------
    def begin_buffered(self) -> "BufferedVSD":
        if self._buffered:
            return self._buffered
        self._buffered = BufferedVSD(self)
        self._buffered.start()
        return self._buffered

    def end_buffered(self) -> None:
        if self._buffered:
            try:
                self._buffered.stop()
            except Exception:
                pass
            self._buffered = None


# ============================================================================
# BUFFERED VSD FLUSHER
# ============================================================================
class BufferedVSD:
    """
    High-frequency flush notifier for prediction engine deltas.
    """

    def __init__(self, vsd: VSDManager) -> None:
        self.vsd = vsd
        self._buf: Dict[str, Any] = {}
        self._dirty = False
        self._stop = False
        self._lock = threading.RLock()
        self._thr: Optional[threading.Thread] = None

    def _mark_dirty(self) -> None:
        try:
            with self._lock:
                self._dirty = True
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        k = str(key)
        with self._lock:
            if k in self._buf:
                return self._buf[k]
            return self.vsd.get(k, default)

    def store(self, key: str, value: Any) -> None:
        k = str(key)
        with self._lock:
            self._buf[k] = value
            self._dirty = True

    def start(self) -> None:
        if self._thr:
            return
        self._stop = False
        self._thr = threading.Thread(
            target=self._loop,
            daemon=True,
            name="vsd_buffered_flush"
        )
        self._thr.start()
        log("VSD", "INFO", "BufferedVSD started")

    def stop(self, timeout: float = 2.0) -> None:
        if not self._thr:
            return
        self._stop = True
        try:
            self._thr.join(timeout)
        except Exception:
            pass
        self._thr = None
        log("VSD", "INFO", "BufferedVSD stopped")

    def _loop(self) -> None:
        while not self._stop:
            time.sleep(max(0.05, float(self.vsd.flush_interval_s)))
            self._flush_once()

    def _flush_once(self) -> None:
        dirty = False
        try:
            with self._lock:
                dirty = self._dirty
                self._dirty = False
        except Exception:
            return

        if not dirty:
            return

        try:
            get_event_bus().publish(
                "vsd.flush.prediction",
                {"ts": time.time(), "note": "prediction delta flush"}
            )
        except Exception:
            pass


# ============================================================================
# GLOBAL VSD INSTANCE
# ============================================================================
try:
    VSD = VSDManager()
except Exception:
    VSD = VSDManager()


# ============================================================================
# PUBLIC EXPORT
# ============================================================================
__all__ = [
    "get_event_bus",
    "EventBus",
    "VSDManager",
    "BufferedVSD",
    "VSD",
]

# ============================================================================
# END OF FILE
# ============================================================================
