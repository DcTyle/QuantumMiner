# =================================================================
# Quantum Application / control_center
# ASCII-ONLY SOURCE FILE
# File: telemetry_console_live.py
# Version: v4.8.3 (AI-Processor snapshots; writer removed)
# -----------------------------------------------------------------
# Purpose
# -------
#   Unified live telemetry refresher for Control Center GUI.
#   Periodically reads profit and hashrate data from VSD and
#   pushes snapshots to registered callback functions.
#   On stop(), emits a json_statevector_v2 snapshot using
#   core.AI_processor.Dehydrator.
# -----------------------------------------------------------------

from __future__ import annotations
import time
import threading
import os
from typing import Any, Callable, Dict

try:
    from core.AI_processor import Dehydrator
except Exception:
    Dehydrator = None  # type: ignore


class TelemetryConsoleLive:
    """Threaded live telemetry updater."""
    def __init__(self, vsd: Any, refresh_s: float = 15.0) -> None:
        self.vsd = vsd
        self.refresh_s = float(refresh_s)
        self._thread: threading.Thread | None = None
        self._stop_flag = False
        self.callbacks: list[Callable[[Dict[str, Any]], None]] = []
        self._last_snapshot_path: str = ""

    # --------------------------------------------------------------
    def register_callback(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        """Attach a callback to receive live telemetry dicts."""
        self.callbacks.append(fn)

    # --------------------------------------------------------------
    def _read_snapshot(self) -> Dict[str, Any]:
        """Fetch the latest telemetry data from VSD, including submitter metrics."""
        try:
            coins = ["ETC", "RVN", "FLUX", "ERG", "BTC", "LTC"]
            out: Dict[str, Any] = {}
            for c in coins:
                profit = self.vsd.get(f"telemetry/profit/{c}/current", {}) or {}
                metrics = self.vsd.get(f"telemetry/metrics/{c}/current", {}) or {}
                agg = self.vsd.get(f"telemetry/metrics/{c}/shares/aggregate", {}) or {}
                latency_avg = float(self.vsd.get(f"telemetry/metrics/{c}/latency/avg_ms", 0.0) or 0.0)
                lanes_idx = self.vsd.get(f"telemetry/metrics/{c}/shares/lanes/_index", []) or []
                lanes_out = {}
                for lid in lanes_idx:
                    ldat = self.vsd.get(f"telemetry/metrics/{c}/shares/lanes/{lid}", {}) or {}
                    lanes_out[str(lid)] = {
                        "submitted": int(ldat.get("submitted", 0)),
                        "accepted": int(ldat.get("accepted", 0)),
                        "found": int(ldat.get("found", 0)),
                        "submitted_hs": float(ldat.get("submitted_hs", 0.0)),
                        "accepted_hs": float(ldat.get("accepted_hs", 0.0)),
                        "acceptance_rate": float(ldat.get("acceptance_rate", 0.0)),
                        "entropy_score": float(ldat.get("entropy_score", 0.0)),
                        "last_ts": float(ldat.get("last_ts", 0.0)),
                    }
                out[c] = {
                    "profit_usd_day": float(profit.get("usd_day", 0.0)),
                    "profit_usd_hour": float(profit.get("usd_hour", 0.0)),
                    "profit_usd_min": float(profit.get("usd_min", 0.0)),
                    "hashrate_hs": float(self.vsd.get(f"telemetry/mine/{c}/accepted_hashrate_hs", 0.0)),
                    "hashes_submitted_hs": float(metrics.get("hashes_submitted_hs", 0.0)),
                    "hashes_found_hs": float(metrics.get("hashes_found_hs", 0.0)),
                    "accepted_hs": float(metrics.get("accepted_hs", 0.0)),
                    "acceptance_rate": float(metrics.get("acceptance_rate", 0.0)),
                    "submitted": int(agg.get("submitted", 0)),
                    "accepted": int(agg.get("accepted", 0)),
                    "found": int(agg.get("found", 0)),
                    "latency_avg_ms": latency_avg,
                    "lanes": lanes_out,
                }
            # Attach developer Nonce Evolution panel if present
            try:
                buf = self.vsd.get("debug/nonce_evolution", []) or []
            except Exception:
                buf = []
            out["panels"] = {"Nonce Evolution": buf}
            return out
        except Exception:
            return {}

    # --------------------------------------------------------------
    def _loop(self) -> None:
        """Internal update loop."""
        while not self._stop_flag:
            snap = self._read_snapshot()
            for cb in list(self.callbacks):
                try:
                    cb(snap)
                except Exception:
                    pass
            time.sleep(self.refresh_s)

    # --------------------------------------------------------------
    def start(self) -> None:
        """Start background update thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="telemetry_live_thread"
        )
        self._thread.start()

    # --------------------------------------------------------------
    def stop(self, timeout: float = 2.0) -> None:
        """Stop the live update thread and pack AI snapshot."""
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        # Optional AI-Processor state snapshot
        try:
            if Dehydrator is not None:
                stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                out_root = os.path.join("VHW", "VD", "state_vectors")
                os.makedirs(out_root, exist_ok=True)
                out_json = os.path.join(out_root, f"telemetry_live_snapshot_{stamp}.json")
                Dehydrator().pack(".", out_json)
                self._last_snapshot_path = out_json
                self.vsd.store("telemetry/snapshots/last_console_state", out_json)
        except Exception:
            pass

# -----------------------------------------------------------------
# Compatibility shim: LiveConsoleRunner expected by bios.main_runtime
# -----------------------------------------------------------------
class _NullVSD:
    def __init__(self) -> None:
        self._kv: Dict[str, Any] = {}
    def get(self, key: str, default: Any = None) -> Any:
        return self._kv.get(str(key), default)
    def store(self, key: str, value: Any) -> None:
        self._kv[str(key)] = value


class LiveConsoleRunner:
    """
    Backwards-compatible wrapper around TelemetryConsoleLive.
    Accepts env dict and internally uses env['vsd'] or a local VSD map.
    """
    def __init__(self, env: Dict[str, Any], refresh_s: float = 0.5) -> None:
        vsd = (env or {}).get("vsd") or _NullVSD()
        self._impl = TelemetryConsoleLive(vsd, refresh_s=max(0.1, float(refresh_s)))

    def start(self) -> None:
        self._impl.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._impl.stop(timeout=timeout)
