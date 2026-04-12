# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/panel_system.py
# Version: v1.0 System Panel
# ----------------------------------------------------------------------------
# Purpose
# -------
# Display live system status pulled from VSD and psutil when available:
#  - CPU load, memory usage
#  - GPU load (best-effort: from VSD keys)
#  - VHW lane count (best-effort from VSD lane map)
#  - Miner active coins and hashrates
#  - Prediction last asset + confidence
#  - Neuralis last output classification (best-effort key)
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit

# Optional psutil imports for CPU/Mem
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


class SystemPanel:
    def __init__(self, vsd: Any) -> None:
        self.vsd = vsd
        self._log: List[str] = []
        self._scroll: int = 0

    def _cpu_mem(self) -> Dict[str, float]:
        out = {"cpu": 0.0, "mem": 0.0}
        try:
            if psutil:
                out["cpu"] = float(psutil.cpu_percent(interval=0.0))
                out["mem"] = float(psutil.virtual_memory().percent)
        except Exception:
            pass
        return out

    def _gpu_load(self) -> float:
        try:
            g = self.vsd.get("system/device_snapshot", {}) or {}
            gpu = g.get("gpu", {}) or {}
            return float(gpu.get("util", 0.0))
        except Exception:
            return 0.0

    def _lane_count(self) -> int:
        # Try lane map first
        try:
            cmap = self.vsd.get("miner/lanes/coin_map", {}) or {}
            if isinstance(cmap, dict):
                return len(cmap.keys())
        except Exception:
            pass
        return 0

    def _miner_stats(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            idx = self.vsd.get("telemetry/metrics/index", []) or []
            nets = [str(x).upper() for x in idx] if isinstance(idx, list) else []
            for net in nets:
                cur = self.vsd.get(f"telemetry/metrics/{net}/current", {}) or {}
                rows.append({
                    "net": net,
                    "hs": float(cur.get("hashes_submitted_hs", 0.0)),
                })
        except Exception:
            pass
        return rows

    def _prediction_last(self) -> Dict[str, Any]:
        try:
            sigs = self.vsd.get("telemetry/predictions/latest", []) or []
            if isinstance(sigs, list) and sigs:
                s = sigs[0]
                return {
                    "symbol": str(s.get("symbol", "")),
                    "conf": float(s.get("avg_confidence", 0.0))
                }
        except Exception:
            pass
        return {"symbol": "", "conf": 0.0}

    def _neuralis_last(self) -> str:
        try:
            c = self.vsd.get("neuralis/last_class", "")
            return str(c or "")
        except Exception:
            return ""

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "System")
            h, w = win.getmaxyx()
            y = 1

            cm = self._cpu_mem()
            gpu = self._gpu_load()
            lanes = self._lane_count()
            pred = self._prediction_last()
            neu = self._neuralis_last()

            # Top blocks
            win.addstr(y, 1, fit("CPU: %.1f%%  MEM: %.1f%%  GPU: %.1f%%  LANES: %d" % (
                cm.get("cpu", 0.0), cm.get("mem", 0.0), gpu, lanes
            ), max(0, w - 2)))
            y += 2

            # Miner stats
            win.addstr(y, 1, "MINER", curses.A_BOLD)
            y += 1
            stats = self._miner_stats()
            for row in stats[: max(0, h - y - 6)]:
                if y >= h - 1:
                    break
                win.addstr(y, 1, fit("%s  %.2f h/s" % (row.get("net"), row.get("hs")), max(0, w - 2)))
                y += 1
            y += 1

            # Prediction and Neuralis summaries
            win.addstr(y, 1, "PREDICTION", curses.A_BOLD)
            y += 1
            win.addstr(y, 1, fit("Last: %s  conf=%.2f" % (pred.get("symbol"), pred.get("conf")), max(0, w - 2)))
            y += 2
            win.addstr(y, 1, "NEURALIS", curses.A_BOLD)
            y += 1
            win.addstr(y, 1, fit("Last class: %s" % (neu or ""), max(0, w - 2)))
            y += 2

            # Tail log
            if y + 2 < h - 1 and self._log:
                win.addstr(y, 1, "Events:", curses.A_BOLD)
                y += 1
                max_lines = (h - 1) - y
                start = max(0, len(self._log) - max_lines - self._scroll)
                end = max(0, len(self._log) - self._scroll)
                for line in self._log[start:end]:
                    if y >= h - 1:
                        break
                    win.addstr(y, 1, fit(line, max(0, w - 2)))
                    y += 1

            win.refresh()
        except Exception:
            pass

    def on_message(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            line = "%s %s" % (topic, str(payload)[:200])
            self._log.append(line)
            if len(self._log) > 500:
                self._log = self._log[-500:]
        except Exception:
            pass

    def handle_key(self, ch: int) -> None:
        try:
            if ch in (curses.KEY_PPAGE, ord('b')):
                self._scroll = min(490, self._scroll + 5)
            elif ch in (curses.KEY_NPAGE, ord('f')):
                self._scroll = max(0, self._scroll - 5)
        except Exception:
            pass

    def clear_log(self) -> None:
        self._log = []
        self._scroll = 0
