# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# File: VHW/vhw_core.py
# Version: v4.8.7h ASCII Clean + Full Diagnostic Patch
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, Callable
import time
import threading
import logging

# ----------------------------------------------------------------------------
# BIOS-Aligned Logger (UTC, ASCII-only)
# ----------------------------------------------------------------------------
_LOG = logging.getLogger("VHW.Core")
if not _LOG.handlers:
    _LOG.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)sZ [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    ))
    logging.Formatter.converter = time.gmtime
    _LOG.addHandler(h)
    _LOG.propagate = False


# ----------------------------------------------------------------------------
# Fallback EventBus
# ----------------------------------------------------------------------------
class _FallbackBus:
    def publish(self, *_a, **_k):
        _LOG.warning("EventBus publish() called but fallback bus is active")
    def subscribe(self, *_a, **_k):
        _LOG.warning("EventBus subscribe() called but fallback bus is active")

def _get_bus_fallback():
    return _FallbackBus()


# ----------------------------------------------------------------------------
# EventBus guarded import
# ----------------------------------------------------------------------------
try:
    from bios.event_bus import get_event_bus
except Exception as e:
    _LOG.error("bios.event_bus import failed: %s", e)
    def get_event_bus():
        return _get_bus_fallback()


# ----------------------------------------------------------------------------
# Config Manager guarded import
# ----------------------------------------------------------------------------
try:
    from config.manager import ConfigManager
except Exception as e:
    _LOG.warning("ConfigManager import fallback engaged: %s", e)
    class ConfigManager:
        def __init__(self):
            self._cfg = {}
            self._lock = threading.RLock()
        def get(self, key, default=None):
            return self._cfg.get(str(key), default)
        def set(self, key, value):
            self._cfg[str(key)] = value


# ----------------------------------------------------------------------------
# VSD guarded import
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager, BufferedVSD
except Exception as e:
    _LOG.error("VSDManager import failed; using fallback: %s", e)

    class VSDManager:
        def __init__(self):
            self._m = {}
            self._lock = threading.RLock()
        def get(self, key, default=None):
            try:
                return self._m.get(str(key), default)
            except Exception as e:
                _LOG.error("VSD.get failed: %s", e)
                return default
        def store(self, key, value):
            try:
                self._m[str(key)] = value
            except Exception as e:
                _LOG.error("VSD.store failed: %s", e)

# Hardware subsystem guarded imports
# ----------------------------------------------------------------------------
def _device_import_failed(msg, e):
    _LOG.error("%s import failed; using fallback: %s", msg, e)


try:
    from VHW.vqpu import VQPU
except Exception as e:
    _device_import_failed("VQPU", e)
    class VQPU:
        def __init__(self, vsd): self.vsd = vsd; self._ok = False
        def start(self): self._ok = True
        def stop(self): self._ok = False
        def status(self): return {"ok": self._ok}


try:
    from VHW.vqgpu import VQGPU
except Exception as e:
    _device_import_failed("VQGPU", e)
    class VQGPU:
        def __init__(self, vsd): self.vsd = vsd; self._ok = False
        def start(self): self._ok = True
        def stop(self): self._ok = False
        def status(self): return {"ok": self._ok}


try:
    from VHW.vqram import VQRAM
except Exception as e:
    _device_import_failed("VQRAM", e)
    class VQRAM:
        def __init__(self, vsd): self.vsd = vsd; self._ok = False
        def start(self): self._ok = True
        def stop(self): self._ok = False
        def status(self): return {"ok": self._ok}


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------
def _utc_ts():
    return float(time.time())

def _utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _bios_ready(vsd):
    try:
        return vsd._bios_ok()
    except Exception as e:
        _LOG.error("bios_ready check failed: %s", e)
        return False

def _mk_buffered_vsd(vsd, cfg):
    base = vsd or VSDManager()
    try:
        return BufferedVSD(base)
    except Exception as e:
        _LOG.error("BufferedVSD init failed (fallback to default): %s", e)
        return BufferedVSD(base)


# ----------------------------------------------------------------------------
# State holder
# ----------------------------------------------------------------------------
class _CoreState:
    def __init__(self):
        self.started = None
        self.ready = False
        self.devices = {}
        self.flush_interval_s = 0.5

_STATE = _CoreState()


# ----------------------------------------------------------------------------
# VHW Core
# ----------------------------------------------------------------------------
class VHWCore:
    def __init__(self, vsd=None, cfg=None, bus=None):
        self.vsd = vsd or VSDManager()
        self.cfg = cfg or ConfigManager()
        self.bus = bus or get_event_bus()
        self._lock = threading.RLock()
        self._thr = None
        self._stop = False
        self._buffered = None

    def _init_devices(self):
        try:
            # Buffered VSD wrapper
            self._buffered = _mk_buffered_vsd(self.vsd, self.cfg)
            _STATE.flush_interval_s = float(self.cfg.get("vsd.flush_interval_s", 0.5))

            # Initialize devices
            vqpu = VQPU(self._buffered)
            vqgpu = VQGPU(self._buffered)
            vqram = VQRAM(self._buffered)

            with self._lock:
                _STATE.devices = {"vqpu": vqpu, "vqgpu": vqgpu, "vqram": vqram}

            # Start devices
            for name in ("vqram", "vqgpu", "vqpu"):
                try:
                    _STATE.devices[name].start()
                except Exception as e:
                    _LOG.error("Failed to start device %s: %s", name, e)

            _STATE.ready = True
            _STATE.started = _utc_iso()

            # Publish readiness
            try:
                self.bus.publish("vhw.core.ready", {
                    "ts": _utc_ts(),
                    "ts_iso": _utc_iso(),
                    "devices": list(_STATE.devices.keys())
                })
            except Exception as e:
                _LOG.error("Failed to publish vhw.core.ready: %s", e)

            _LOG.info("VHW devices ready.")

        except Exception as e:
            _LOG.error("Device init failed: %s", e, exc_info=True)

    def start(self):
        if self._thr:
            return
        self._stop = False
        self._thr = threading.Thread(target=self._init_devices, daemon=True, name="VHWCoreInit")
        self._thr.start()

    def stop(self, timeout=2.0):
        self._stop = True
        if self._thr:
            try:
                self._thr.join(timeout)
            except Exception as e:
                _LOG.error("Error stopping VHWCore init thread: %s", e)
            self._thr = None

        # Stop all devices
        with self._lock:
            for name, d in _STATE.devices.items():
                try:
                    d.stop()
                except Exception as e:
                    _LOG.error("Error stopping device %s: %s", name, e)

        # Stop the buffered VSD
        if self._buffered:
            try:
                self._buffered.stop()
            except Exception as e:
                _LOG.error("Error stopping BufferedVSD: %s", e)

        _LOG.info("VHWCore stopped.")

    def status(self):
        try:
            with self._lock:
                devs = {}
                for k, v in _STATE.devices.items():
                    try:
                        devs[k] = v.status()
                    except Exception as e:
                        _LOG.error("Device status failed for %s: %s", k, e)
                        devs[k] = {"ok": False}
                return {
                    "ready": _STATE.ready,
                    "started": _STATE.started,
                    "devices": devs,
                    "flush_interval_s": _STATE.flush_interval_s,
                    "ts": _utc_ts(),
                    "ts_iso": _utc_iso()
                }
        except Exception as e:
            _LOG.error("VHWCore.status failed: %s", e)
            return {"ready": False, "error": str(e)}


# ----------------------------------------------------------------------------
# Boot complete autowire
# ----------------------------------------------------------------------------
_BUS = get_event_bus()
_VHW_SINGLETON = None
_LOCK = threading.RLock()

def _on_boot_complete(event: Dict[str, Any] | None = None):
    global _VHW_SINGLETON
    try:
        with _LOCK:
            if _VHW_SINGLETON is None:
                _LOG.info("boot.complete received; starting VHWCore...")
                _VHW_SINGLETON = VHWCore()
                _VHW_SINGLETON.start()
    except Exception as e:
        _LOG.error("VHWCore autostart failed: %s", e)


try:
    _BUS.subscribe("boot.complete", _on_boot_complete)
except Exception as e:
    _LOG.error("Failed to subscribe to boot.complete: %s", e)


def global_core():
    global _VHW_SINGLETON
    with _LOCK:
        if _VHW_SINGLETON is None:
            _VHW_SINGLETON = VHWCore()
        return _VHW_SINGLETON
