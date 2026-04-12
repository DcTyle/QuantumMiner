# ============================================================================
# VirtualMiner / VHW
# File: share_allocator.py
# Version: v4.8.7 (bounded telemetry + digest reuse + EventBus)
# ASCII-ONLY SOURCE FILE
# ============================================================================
"""
Purpose
-------
Safe share allocation/telemetry bridge with:
- BIOS gate awareness via VSDManager pattern (handled by caller)
- Bounded share log (<= 1000 entries) to avoid unbounded growth
- Digest reuse for share validation (single SHA256 for stable records)
- EventBus publishes for telemetry bridge + VSD flush hints

Behavior preserved from prior versions:
- Lane-aware share submission and metric updates
- Existing prefix/target pre-filters (if provided by caller) are honored
"""

from __future__ import annotations
import time
import hashlib
import threading
import logging
from typing import Any, Dict, Optional

# ----------------------------------------------------------------------------
# BIOS logging standard (GLOBAL LOGGER)
# ----------------------------------------------------------------------------
def _mk_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        lg.propagate = False
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg

LOG = _mk_logger("VHW.ShareAllocator")

from VHW.vsd_manager import VSDManager  # type: ignore
from bios.event_bus import get_event_bus  # type: ignore

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _utc() -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception as e:
        LOG.error("_utc() time format failed: %s", str(e), exc_info=True)
        return "1970-01-01T00:00:00Z"

def _sha256_ascii(s: str) -> str:
    try:
        return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()
    except Exception as e:
        LOG.error("_sha256_ascii() failed: %s", str(e), exc_info=True)
        return ""

# ---------------------------------------------------------------------------
# ShareAllocator
# ---------------------------------------------------------------------------
class ShareAllocator:
    """
    Provides safe share submission and telemetry writing.
    Caller supplies network/lane logic; we append bounded telemetry and publish.
    """

    def __init__(self, vsd: Optional[VSDManager] = None) -> None:
        self.vsd = vsd or VSDManager()
        self._bus = get_event_bus()
        self._lock = threading.RLock()

        # Instance logger reference (C)
        self.log = LOG

        # Stats caches
        self._digest_cache_ok = True
        self.log.info("ShareAllocator initialized")

    # ---------------------------- Core API ---------------------------------
    def submit_share(self,
                     lane_id: str,
                     network: str,
                     nonce_hex: str,
                     hash_hex: str,
                     target_hex: str,
                     is_valid: bool,
                     extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a share and publish telemetry bridge events.
        """
        try:
            rec = {
                "ts": _utc(),
                "lane_id": str(lane_id),
                "network": str(network).upper(),
                "nonce": str(nonce_hex),
                "hash": str(hash_hex),
                "target": str(target_hex),
                "is_valid": bool(is_valid),
            }
        except Exception as e:
            self.log.error("Failed to build base share record: %s", str(e), exc_info=True)
            return

        try:
            if extra:
                for k, v in list(extra.items()):
                    rec[str(k)] = v
        except Exception as e:
            self.log.error("Failed copying extra fields: %s", str(e), exc_info=True)

        # Digest reuse (stable ordering)
        try:
            stable_s = (
                rec["ts"] + "|" + rec["lane_id"] + "|" + rec["network"] + "|" +
                rec["nonce"] + "|" + rec["hash"] + "|" + rec["target"] + "|" +
                ("1" if rec["is_valid"] else "0")
            )
            rec["digest"] = _sha256_ascii(stable_s)
        except Exception as e:
            self.log.error("digest generation failed: %s", str(e), exc_info=True)
            rec["digest"] = ""

        # Append to bounded log
        try:
            key_log = "telemetry/metrics/share_log"
            self.vsd.append_bounded(key_log, rec, max_len=1000)
        except Exception as e:
            self.log.error("append_bounded failed: %s", str(e), exc_info=True)

        # Update counters
        try:
            if is_valid:
                ok_key = "telemetry/metrics/valid_share_count"
                self.vsd.store(ok_key, int(self.vsd.get(ok_key, 0)) + 1)
            else:
                bad_key = "telemetry/metrics/invalid_share_count"
                self.vsd.store(bad_key, int(self.vsd.get(bad_key, 0)) + 1)
        except Exception as e:
            self.log.error("Failed updating counters: %s", str(e), exc_info=True)

        # Publish telemetry bridge event
        try:
            self._bus.publish("telemetry.bridge.share", {
                "ts": rec["ts"],
                "network": rec["network"],
                "lane_id": rec["lane_id"],
                "valid": rec["is_valid"],
                "digest": rec.get("digest", ""),
            })
        except Exception as e:
            self.log.error("EventBus publish (share) failed: %s", str(e), exc_info=True)

        # Flush hint
        try:
            self._bus.publish("vsd.flush.prediction", {
                "ts": _utc(),
                "note": "share log updated"
            })
        except Exception as e:
            self.log.error("EventBus publish (flush hint) failed: %s", str(e), exc_info=True)

    # ---------------------------- Diagnostics ------------------------------
    def stats(self) -> Dict[str, Any]:
        """
        Summarize current share telemetry (bounded window).
        """
        try:
            arr = self.vsd.get("telemetry/metrics/share_log", [])
            total = len(arr) if isinstance(arr, list) else 0
            valids = 0
            if isinstance(arr, list):
                for x in arr:
                    try:
                        if bool(x.get("is_valid")):
                            valids += 1
                    except Exception as e2:
                        self.log.error("stats valid-check failed: %s", str(e2), exc_info=True)
            skipped = int(self.vsd.get("telemetry/metrics/skipped_nonce_count", 0))
            return {
                "total": total,
                "valid": valids,
                "skipped": skipped,
                "valid_ratio": (float(valids) / float(total)) if total > 0 else 0.0,
            }
        except Exception as e:
            self.log.error("stats() failed: %s", str(e), exc_info=True)
            return {"total": 0, "valid": 0, "skipped": 0, "valid_ratio": 0.0}


# ---------------------------------------------------------------------------
# Inline self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    LOG.info("Starting share_allocator self-test")
    vsd = VSDManager()
    vsd.store("system/bios_boot_ok", True)

    alloc = ShareAllocator(vsd)
    alloc.submit_share("L1", "ETC", "00aa", "ff11", "0fff", True, {"latency_ms": 3.2})
    alloc.submit_share("L2", "ETC", "00bb", "ff22", "0fff", False, {"latency_ms": 4.1})
    print("stats:", alloc.stats())
# End of file
