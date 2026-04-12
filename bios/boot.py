# ============================================================================
# Quantum Application / bios
# ASCII-ONLY SOURCE FILE
# File: boot.py
# Version: v7 "SystemUtils Schema-Enforced Boot" (Patched for Shared VSD)
# ============================================================================
"""
Purpose
-------
Boot coordinator for BIOS.
Initializes subsystems concurrently, sets bios_boot_ok, persists
state to VSD, and announces 'boot.complete' over the BIOS EventBus.

System Utils Integration
------------------------
Stage _build_system_probe:
- Calls VHW.system_utils.device_snapshot() and system_headroom()
- Normalizes into the canonical schema
- Stores under:
    system/device_snapshot
    system/headroom
- Emits "bios.system_probe" with normalized payload
"""

from __future__ import annotations
import threading
import time
import logging
from typing import Any, Dict, Callable

# ----------------------------------------------------------------------------
# Structured Logging
# ----------------------------------------------------------------------------
_logger = logging.getLogger("bios.boot")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    _h.setFormatter(_fmt)
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)

logging.Formatter.converter = time.gmtime  # type: ignore[attr-defined]

# ----------------------------------------------------------------------------
# EventBus
# ----------------------------------------------------------------------------
try:
    from bios.event_bus import get_event_bus, event_map
except Exception:
    _logger.warning("bios.event_bus unavailable; using NoOp bus")

    class _NoBus:
        def publish(self, topic: str, data=None) -> None:
            return
        def subscribe(self, topic: str, handler=None,
                      once: bool = False, priority: int = 0,
                      name: str | None = None) -> None:
            return

    def get_event_bus():  # type: ignore[no-redef]
        return _NoBus()

    event_map: Dict[str, str] = {
        "BOOT_INIT": "bios.init",
        "BOOT_STAGE": "bios.stage",
        "BOOT_COMPLETE": "boot.complete",
    }

_bus = get_event_bus()

# ----------------------------------------------------------------------------
# VSD Manager (shared state)
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager  # type: ignore[import]
except Exception:
    class VSDManager:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._kv: Dict[str, Any] = {}
        def store(self, key: str, value: Any) -> None:
            self._kv[str(key)] = value
        def get(self, key: str, default: Any = None) -> Any:
            return self._kv.get(str(key), default)

# ---------------------------------------------------------------------------
# IMPORT SHARED VSD INSTANCE
# ---------------------------------------------------------------------------
# (FIRST PATCH APPLIED HERE)
try:
    from bios.main_runtime import _vsd as shared_vsd
except Exception:
    shared_vsd = VSDManager()

# ----------------------------------------------------------------------------
# System Utils (unified hardware API)
# ----------------------------------------------------------------------------
try:
    from VHW.system_utils import device_snapshot, system_headroom
except Exception:
    def device_snapshot() -> Dict[str, Any]:
        return {}
    def system_headroom() -> Dict[str, Any]:
        return {}

# Monitor is optional; we will gracefully fall back if it is missing
try:
    from VHW.monitor import Monitor as VHWMonitor  # type: ignore[import]
except Exception:
    VHWMonitor = None  # type: ignore[assignment]

# ----------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------
bios_boot_ok: bool = False

# ----------------------------------------------------------------------------
# Helper: BIOS Readiness Flag
# ----------------------------------------------------------------------------
def _bios_ready() -> bool:
    """Return True if BIOS boot flag exists in VSD."""
    try:
        return bool(shared_vsd.get("system/bios_boot_ok", False))
    except Exception:
        return False

# ----------------------------------------------------------------------------
# Simple Monitor Stub
# ----------------------------------------------------------------------------
class _MonitorStub:
    """
    Minimal monitor object used as a safe fallback.
    Provides frame() compatible with VHW.monitor.Monitor.frame().
    """
    def __init__(self) -> None:
        self._last: Dict[str, Any] = {}

    def frame(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "timestamp": now,
            "global_util": float(self._last.get("global_util", 0.0)),
            "target_util": float(self._last.get("target_util", 0.0)),
            "alert": bool(self._last.get("alert", False)),
        }

# ----------------------------------------------------------------------------
# BIOSBoot Class
# ----------------------------------------------------------------------------
class BIOSBoot:
    """
    Boot orchestrator for all major subsystems.
    Spawns concurrent stage threads and monitors progress.

    Design Rules
    ------------
    - Does NOT import or call other BIOS modules.
    - Does NOT depend on a config folder.
    - Communicates readiness only via:
        * EventBus (boot.complete)
        * VSD key "system/bios_boot_ok"
    """

    def __init__(self, vsd=None) -> None:
        if vsd is not None:
            global shared_vsd
            shared_vsd = vsd
        _logger.info("BIOS boot sequence starting")
        self._lock = threading.RLock()
        self._threads: Dict[str, threading.Thread] = {}
        self._stage_status: Dict[str, str] = {}
        self._stage_errors: Dict[str, str] = {}
        self._done_evt = threading.Event()

        # Exposed handles (some may be replaced by build stages)
        self.vsd: Any = shared_vsd
        self.monitor: Any = _MonitorStub()

        # Production: Initialize required miner attributes
        try:
            import json
            from VHW.compute_manager import ComputeManager
            from VHW.share_allocator import ShareAllocator
            from VHW.failsafe import FailsafeGovernor
            from miner.submitter import Submitter
            # ASCII-only enforcement
            def _ascii_check(val):
                if isinstance(val, str):
                    val.encode('ascii')
                elif isinstance(val, dict):
                    for k, v in val.items():
                        _ascii_check(k)
                        _ascii_check(v)
                elif isinstance(val, list):
                    for item in val:
                        _ascii_check(item)
            # Load runtime config using ConfigLoader
            try:
                from bios.config_loader import ConfigLoader
                cfg = ConfigLoader.load_config()
            except Exception:
                # Fallback: load from JSON directly
                with open("miner/miner_runtime_config.json", "r") as f:
                    cfg = json.load(f)
            if cfg is None:
                raise RuntimeError("Fatal: ConfigLoader.load_config() returned None")
            cfg["vsd"] = shared_vsd
            _ascii_check(cfg)
            object.__setattr__(self, 'runtime_cfg', dict(cfg))
            object.__setattr__(self, 'coin_cfg', dict(cfg.get("coins", {}) or {}))
            object.__setattr__(self, 'nonce_cfg', dict(cfg.get("miner_nonce", {}) or {}))
            object.__setattr__(self, 'compute_manager', ComputeManager(cfg))
            object.__setattr__(self, 'share_allocator', ShareAllocator(vsd=shared_vsd))
            object.__setattr__(self, 'submitter', Submitter(vsd=shared_vsd))
            # Ensure failsafe uses the shared VSD and bus for readiness and events
            object.__setattr__(self, 'failsafe', FailsafeGovernor(cfg, shared_vsd, _bus))
            # handler_miner no longer required; NonceMath integrated in ComputeManager
        except Exception as exc:
            _logger.critical(f"Production boot failed: could not initialize required attributes: {exc}")
            raise RuntimeError(f"Production boot failed: could not initialize required attributes: {exc}")

        # Stages, including system probe
        self._stages = [
            "_build_vsd",
            "_build_system_probe",
            "_build_monitor",
            "_build_live",
            "_build_pools",
            "_build_allocator",
            "_build_failsafe",
            "_build_share_allocator",
            "_build_engine",
        ]

        # Execute stages
        t0 = time.time()
        try:
            self._parallel_boot_tasks()
            self._done_evt.wait()
            dt = time.time() - t0
            _logger.info("BIOS boot sequence complete in %.3fs", dt)
        except Exception:
            _logger.exception("BIOS boot sequence failed unexpectedly")
            return

        # --------------------------------------------------------------------
        # Persist boot flag and announce boot.complete
        # (THIRD PATCH APPLIED HERE)
        # --------------------------------------------------------------------
        try:
            self.vsd.mark_bios_ready()

            global bios_boot_ok
            bios_boot_ok = True

            payload = {
                "ts": time.time(),
                "stages": self._stage_status.copy(),
                "errors": self._stage_errors.copy(),
                "duration_s": dt,
            }

            event = str(event_map.get("BOOT_COMPLETE", "boot.complete"))
            _bus.publish(event, payload)

            _logger.info("Published boot.complete event")
        except Exception as exc:
            _logger.warning("Failed to persist or publish boot.complete: %s", exc)

    # ------------------------------------------------------------------------
    # Parallel Tasks
    # ------------------------------------------------------------------------
    def _parallel_boot_tasks(self) -> None:
        with self._lock:
            for name in self._stages:
                self._stage_status[name] = "pending" if hasattr(self, name) else "skipped"

        for name in self._stages:
            if self._stage_status[name] != "pending":
                continue
            th = threading.Thread(target=self._run_stage, args=(name,), daemon=True)
            self._threads[name] = th
            th.start()

        monitor = threading.Thread(target=self._progress_loop, daemon=True)
        monitor.start()

    def _run_stage(self, name: str) -> None:
        try:
            with self._lock:
                self._stage_status[name] = "running"
            _logger.info("Boot stage starting: %s", name)
            builder: Callable[[], None] = getattr(self, name, lambda: _logger.info("Placeholder: %s", name))
            builder()
            with self._lock:
                self._stage_status[name] = "done"
            _logger.info("Boot stage completed: %s", name)
        except Exception as exc:
            with self._lock:
                self._stage_status[name] = "error"
                self._stage_errors[name] = str(exc)
            _logger.exception("Boot stage failed: %s", name)

    def _progress_loop(self) -> None:
        try:
            total = len(self._stages)
            while True:
                time.sleep(0.25)
                with self._lock:
                    done = sum(1 for s in self._stages if self._stage_status.get(s) in ("done", "skipped"))
                    running = [s for s in self._stages if self._stage_status.get(s) == "running"]
                    pending = [s for s in self._stages if self._stage_status.get(s) == "pending"]
                    errors = [s for s in self._stages if self._stage_status.get(s) == "error"]
                    all_threads_done = all(
                        (s not in self._threads) or (not self._threads[s].is_alive())
                        for s in self._threads
                    )
                _logger.info(
                    "Boot progress %d/%d | running=%s pending=%s errors=%s",
                    done, total, running, pending, errors,
                )
                if all_threads_done:
                    break
        finally:
            self._done_evt.set()

    # ------------------------------------------------------------------------
    # Stage: VSD
    # ------------------------------------------------------------------------
    def _build_vsd(self) -> None:
        try:
            # Do not create a new VSD. Already set in __init__ to shared_vsd.
            _logger.info("VSD subsystem initialized (shared)")
        except Exception as exc:
            _logger.warning("VSD subsystem init failed: %s", exc)

    # ------------------------------------------------------------------------
    # Stage: System Probe (schema-enforced)
    # ------------------------------------------------------------------------
    def _build_system_probe(self) -> None:
        """
        Run system_utils probes and normalize into canonical schema.
        Writes:
            system/device_snapshot
            system/headroom
        Emits:
            "bios.system_probe"
        """
        try:
            vsd = self.vsd

            dev = device_snapshot()
            hdr = system_headroom()

            dev_norm: Dict[str, Any] = {
                "timestamp": float(dev.get("timestamp", time.time())),
                "cpu": {
                    "count": int(dev.get("cpu", {}).get("count", 1)),
                    "threads": int(dev.get("cpu", {}).get("threads", 1)),
                    "util": float(dev.get("cpu", {}).get("util", 0.0)),
                },
                "gpu": {
                    "model": str(dev.get("gpu", {}).get("model", "Unknown")),
                    "memory_total_mb": float(dev.get("gpu", {}).get("memory_total_mb", 0.0)),
                    "memory_used_mb": float(dev.get("gpu", {}).get("memory_used_mb", 0.0)),
                    "util": float(dev.get("gpu", {}).get("util", 0.0)),
                    "temperature_c": float(dev.get("gpu", {}).get("temperature_c", 0.0)),
                },
                "memory": {
                    "total_gb": float(dev.get("memory", {}).get("total_gb", 0.0)),
                    "used_gb": float(dev.get("memory", {}).get("used_gb", 0.0)),
                    "util": float(dev.get("memory", {}).get("util", 0.0)),
                },
                "bandwidth": {
                    "pcie_gbps": float(dev.get("bandwidth", {}).get("pcie_gbps", 0.0)),
                    "memory_bw_gbps": float(dev.get("bandwidth", {}).get("memory_bw_gbps", 0.0)),
                },
            }

            hdr_norm: Dict[str, Any] = {
                "timestamp": float(hdr.get("timestamp", dev_norm["timestamp"])),
                "global_util": float(hdr.get("global_util", 0.0)),
                "cpu_util": float(hdr.get("cpu_util", dev_norm["cpu"]["util"])),
                "gpu_util": float(hdr.get("gpu_util", dev_norm["gpu"]["util"])),
                "mem_util": float(hdr.get("mem_util", dev_norm["memory"]["util"])),
                "headroom": float(hdr.get("headroom", 1.0)),
            }

            vsd.store("system/device_snapshot", dev_norm)
            vsd.store("system/headroom", hdr_norm)

            payload = {
                "ts": time.time(),
                "device_snapshot": dev_norm,
                "system_headroom": hdr_norm,
            }
            _bus.publish("bios.system_probe", payload)

            _logger.info("System probe completed and schema enforced")
        except Exception as exc:
            _logger.warning("System probe failed: %s", exc)

    # ------------------------------------------------------------------------
    # Remaining Stages
    # ------------------------------------------------------------------------
    def _build_monitor(self) -> None:
        try:
            if VHWMonitor is None:
                _logger.info("Monitor subsystem initialized (stub)")
                return
            vsd = self.vsd

            def _read() -> Dict[str, Any]:
                snap = vsd.get("telemetry/global", None)
                if not isinstance(snap, dict):
                    return {
                        "timestamp": time.time(),
                        "global_util": 0.0,
                        "target_util": 0.0,
                        "alert": False,
                    }
                return snap

            self.monitor = VHWMonitor(read_telemetry=_read, util_cap=0.75)
            _logger.info("Monitor subsystem initialized")
        except Exception as exc:
            _logger.warning("Monitor creation failed: %s", exc)

    def _build_live(self) -> None:
        _logger.info("Live control subsystem initialized")

    def _build_pools(self) -> None:
        _logger.info("Mining pools subsystem initialized")

    def _build_allocator(self) -> None:
        _logger.info("Allocator subsystem initialized")

    def _build_failsafe(self) -> None:
        _logger.info("Failsafe subsystem initialized")

    def _build_share_allocator(self) -> None:
        _logger.info("Share allocator subsystem initialized")

    def _build_engine(self) -> None:
        _logger.info("Compute engine subsystem initialized")

# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
def start(vsd=None) -> BIOSBoot:
    """External entry for BIOS boot sequence."""
    # BIOSBoot constructor performs full initialization and validation
    return BIOSBoot(vsd=vsd)

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
__all__ = [
    "start",
    "_bios_ready",
    "BIOSBoot",
    "bios_boot_ok",
]
