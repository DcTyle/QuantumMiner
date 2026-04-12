# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/panel_ada.py
# Version: v1.0 ADA Telemetry Panel
# ----------------------------------------------------------------------------
# Purpose
# -------
# Visualize ADA v4.7 virtual hardware telemetry:
#   - Hardware envelope headroom and density
#   - Tier weights from TierOscillator
#   - QPU/GPU ADA constants
# Data is pulled from VSD paths written by VHW/system_utils and related
# virtual hardware modules.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit


class AdaPanel:
    def __init__(self, vsd: Any) -> None:
        self.vsd = vsd
        self._log: List[str] = []
        self._scroll: int = 0

    def _read_snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "hardware": {},
            "vqram": {},
            "tier_weights": {},
            "qpu": {},
            "gpu": {},
        }
        try:
            hw = self.vsd.get("system/hardware/statevector", {}) or {}
            out["hardware"] = hw
        except Exception:
            pass
        try:
            vq = self.vsd.get("system/vqram/statevector", {}) or {}
            out["vqram"] = vq
        except Exception:
            pass
        try:
            tw = self.vsd.get("telemetry/vqgpu/current", {}) or {}
            if isinstance(tw, dict):
                out["tier_weights"] = dict(tw.get("tier_weights", {}) or {})
                out["gpu"] = dict(tw.get("ada_constants", {}) or {})
        except Exception:
            pass
        try:
            q = self.vsd.get("telemetry/qpu/current", {}) or {}
            if isinstance(q, dict):
                out["qpu"] = dict(q.get("ada_constants", {}) or {})
        except Exception:
            pass
        return out

    def _bar(self, win, y: int, x: int, label: str, value: float, width: int) -> int:
        if curses is None:
            return y
        try:
            v = max(0.0, min(1.0, float(value)))
        except Exception:
            v = 0.0
        filled = int(v * max(1, width - len(label) - 4))
        bar = "#" * filled + "-" * max(0, (width - len(label) - 4) - filled)
        text = "%s [%-*.s]" % (label, max(0, width - len(label) - 4), bar)
        win.addstr(y, x, fit(text, width))
        return y + 1

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "ADA / VHW Telemetry")
            h, w = win.getmaxyx()
            y = 1

            snap = self._read_snapshot()
            tw = snap.get("tier_weights", {}) or {}
            gpu = snap.get("gpu", {}) or {}
            qpu = snap.get("qpu", {}) or {}

            # Tier weights graph
            win.addstr(y, 1, "Tier Weights", curses.A_BOLD)
            y += 1
            for tid in sorted(tw.keys()):
                try:
                    v = float(tw[tid])
                except Exception:
                    v = 0.0
                y = self._bar(win, y, 1, "Tier %s" % str(tid), v, max(10, w - 2))
                if y >= h - 6:
                    break

            y += 1
            if y < h - 4:
                win.addstr(y, 1, "GPU ADA Constants", curses.A_BOLD)
                y += 1
                line = "h_eff=%.3f  k_B_eff=%.3f  c_eff=%.3f" % (
                    float(gpu.get("h_eff", 1.0)),
                    float(gpu.get("k_B_eff", 1.0)),
                    float(gpu.get("c_eff", 1.0)),
                )
                win.addstr(y, 1, fit(line, max(0, w - 2)))
                y += 2

            if y < h - 2:
                win.addstr(y, 1, "QPU ADA Constants", curses.A_BOLD)
                y += 1
                line = "h_eff=%.3f  k_B_eff=%.3f  c_eff=%.3f" % (
                    float(qpu.get("h_eff", 1.0)),
                    float(qpu.get("k_B_eff", 1.0)),
                    float(qpu.get("c_eff", 1.0)),
                )
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
