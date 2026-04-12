# ============================================================================
# Quantum Application / VHW
# ASCII-ONLY SOURCE FILE
# File: vqram.py
# Version: v4.8.7 Hybrid (final-integration, LOGGING-NORMALIZED)
# Jarvis ADA v4.7 Hybrid Ready
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import time
import threading
import logging

# ============================================================================
# BIOS-STANDARD LOGGER (MANDATED FORMAT)
# ============================================================================
def _mk_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.propagate = False
        lg.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg

LOG = _mk_logger("VHW.VQRAM")

# ============================================================================
# Safe fallbacks (ASCII-only stubs)
# ============================================================================
from config.manager import ConfigManager  # type: ignore
from bios.event_bus import get_event_bus  # type: ignore
from VHW.vsd_manager import VSDManager  # type: ignore
from VHW.system_utils import store_statevector

# ============================================================================
# Helpers
# ============================================================================
def _utc_ts_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _align_up(n: int, a: int) -> int:
    if a <= 0: return n
    r = n % a
    return n if r == 0 else (n + (a - r))

# ============================================================================
# VQRAM CLASS
# ============================================================================
class VQRAM:
    """
    Virtual GPU RAM allocator with telemetry + BIOS gating.
    """

    def __init__(self,
                 cfg: Optional[ConfigManager] = None,
                 bus_factory: Any = get_event_bus) -> None:

        self._cfg = cfg or ConfigManager()
        self._bus = bus_factory()
        self._vsd: Optional[VSDManager] = None

        # tunables
        self._pool_bytes: int = int(self._cfg.get("vqram.pool_bytes", 512 * 1024 * 1024))
        self._block_min: int = int(self._cfg.get("vqram.block_min", 4096))
        self._telemetry_interval_s: float = float(self._cfg.get("vqram.telemetry_interval_s", 1.0))

        # state
        self._lock = threading.RLock()
        self._running = False
        self._thr: Optional[threading.Thread] = None
        self._last_pub = 0.0
        self._alloc_count = 0
        self._free_count = 0
        self._allocated_bytes = 0

        # free list and allocation map
        self._free: List[Tuple[int, int]] = [(0, self._pool_bytes)]
        self._alloc_map: Dict[int, int] = {}

        # eventbus subscription
        try:
            self._bus.subscribe("boot.complete", self._on_boot_complete)
        except Exception as e:
            LOG.error("eventbus subscription failed: %s", e, exc_info=True)

        LOG.info("VQRAM initialized (v4.8.7 Hybrid)")

    # ============================================================================
    # BIOS readiness
    # ============================================================================
    def _bios_ready(self) -> bool:
        try:
            if self._vsd is None:
                return False
            return bool(self._vsd.get("system/bios_boot_ok", False))
        except Exception:
            return False

    # ============================================================================
    # Lifecycle
    # ============================================================================
    def start(self, vsd: Optional[VSDManager] = None) -> None:
        if self._running:
            return
        self._vsd = vsd or self._vsd or VSDManager()

        if not self._bios_ready():
            LOG.warning("VQRAM start skipped (BIOS not ready)")
            return

        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True, name="vqram_loop")
        self._thr.start()
        LOG.info("VQRAM started")

    def stop(self, timeout: float = 2.0) -> None:
        if not self._running:
            return
        self._running = False

        try:
            if self._thr:
                self._thr.join(timeout=timeout)
        except Exception as e:
            LOG.error("stop() failure: %s", e, exc_info=True)
        finally:
            self._thr = None
            LOG.info("VQRAM stopped")

    def _on_boot_complete(self, event: Dict[str, Any] | None = None) -> None:
        try:
            if not self._running:
                self.start(self._vsd or VSDManager())
        except Exception as e:
            LOG.error("boot.complete handler failed: %s", e, exc_info=True)

    # ============================================================================
    # Allocation
    # ============================================================================
    def allocate(self, size_bytes: int) -> Optional[Tuple[int, int]]:
        try:
            if size_bytes <= 0:
                return None
            size = _align_up(int(size_bytes), max(1, self._block_min))

            with self._lock:
                for idx, (off, sz) in enumerate(self._free):
                    if sz >= size:
                        alloc_off = off
                        remain = sz - size

                        self._alloc_map[alloc_off] = size
                        self._allocated_bytes += size
                        self._alloc_count += 1

                        if remain > 0:
                            self._free[idx] = (off + size, remain)
                        else:
                            del self._free[idx]

                        return (alloc_off, size)
            return None

        except Exception as e:
            LOG.error("allocation failed: %s", e, exc_info=True)
            return None

    def free(self, offset: int, size_bytes: int) -> None:
        try:
            if size_bytes <= 0:
                return
            size = int(size_bytes)

            with self._lock:
                known = self._alloc_map.pop(offset, None)
                if known is not None:
                    size = known

                self._allocated_bytes = max(0, self._allocated_bytes - size)
                self._free_count += 1

                self._free.append((offset, size))
                self._free.sort(key=lambda x: x[0])

                self._coalesce()

        except Exception as e:
            LOG.error("free() failed: %s", e, exc_info=True)

    def _coalesce(self) -> None:
        if not self._free:
            return
        merged: List[Tuple[int, int]] = []
        cur_off, cur_sz = self._free[0]

        for off, sz in self._free[1:]:
            if off == cur_off + cur_sz:
                cur_sz += sz
            else:
                merged.append((cur_off, cur_sz))
                cur_off, cur_sz = off, sz
        merged.append((cur_off, cur_sz))
        self._free = merged

    # ============================================================================
    # Telemetry
    # ============================================================================
    def _loop(self) -> None:
        while self._running:
            try:
                now = time.time()
                if now - self._last_pub >= self._telemetry_interval_s:
                    self._publish_telemetry()
                    self._last_pub = now
            except Exception as e:
                LOG.error("loop error: %s", e, exc_info=True)

            time.sleep(0.05)

    def _publish_telemetry(self) -> None:
        snap = self.stats()

        try:
            self._bus.publish("telemetry.vqram", snap)
        except Exception as e:
            LOG.error("telemetry publish failed: %s", e, exc_info=True)

        try:
            if self._vsd:
                self._vsd.store("telemetry/vqram/current", snap)
        except Exception as e:
            LOG.error("telemetry VSD store failed: %s", e, exc_info=True)

        try:
            vec = [
                float(snap.get("allocated_bytes", 0)) / float(max(1, snap.get("pool_bytes", 1))),
                float(snap.get("free_bytes", 0)) / float(max(1, snap.get("pool_bytes", 1))),
            ]
            store_statevector("system/vqram/statevector", [max(-1.0, min(1.0, v * 2.0 - 1.0)) for v in vec])
        except Exception:
            pass

    # ============================================================================
    # Stats
    # ============================================================================
    def stats(self) -> Dict[str, Any]:
        try:
            with self._lock:
                free_total = sum(sz for _o, sz in self._free)
                free_blocks = len(self._free)
                alloc_blocks = len(self._alloc_map)

            return {
                "ts": _utc_ts_str(),
                "running": bool(self._running),
                "pool_bytes": int(self._pool_bytes),
                "allocated_bytes": int(self._allocated_bytes),
                "free_bytes": int(free_total),
                "free_blocks": int(free_blocks),
                "alloc_blocks": int(alloc_blocks),
                "alloc_count": int(self._alloc_count),
                "free_count": int(self._free_count),
                "block_min": int(self._block_min),
            }

        except Exception as e:
            LOG.error("stats() failure: %s", e, exc_info=True)
            return {"error": str(e)}

# ============================================================================
# End of file
# ============================================================================
