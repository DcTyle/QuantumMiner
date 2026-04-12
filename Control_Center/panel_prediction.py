# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/panel_prediction.py
# Version: v1.0 Prediction Panel
# ----------------------------------------------------------------------------
# Purpose
# -------
# Show last prediction report and a compact list of latest fused signals
# written by the PredictionEngine into VSD.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit


def _fmt_ts_age(ts_str: str) -> str:
    try:
        # Expect ISO-like Z timestamps Y-m-dTH:M:S
        # Fallback to seconds if not parseable
        # Use best-effort relative age
        t = time.time()
        return "0s" if not ts_str else ts_str
    except Exception:
        return ts_str


class PredictionPanel:
    def __init__(self, vsd: Any) -> None:
        self.vsd = vsd
        self._log: List[str] = []
        self._scroll: int = 0

    def _read_latest(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "signals": [],
            "orders": 0,
            "assets": 0,
            "ts": "",
        }
        try:
            rep = self.vsd.get("prediction/last_report", {}) or {}
            out["ts"] = str(rep.get("ts", ""))
            out["assets"] = int(rep.get("assets_considered", 0))
            out["orders"] = int(rep.get("orders", 0))
        except Exception:
            pass
        try:
            sigs = self.vsd.get("telemetry/predictions/latest", []) or []
            if isinstance(sigs, list):
                out["signals"] = sigs
        except Exception:
            pass
        return out

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "Prediction")
            h, w = win.getmaxyx()
            y = 1

            latest = self._read_latest()
            head = "TS: " + (latest.get("ts") or "")
            win.addstr(y, 1, fit(head, max(0, w - 2)))
            y += 1
            info = "Assets: %d  Orders: %d" % (
                int(latest.get("assets", 0)), int(latest.get("orders", 0))
            )
            win.addstr(y, 1, fit(info, max(0, w - 2)))
            y += 2

            # Table header
            cols = [("SYMBOL", 10), ("CONF", 6), ("NOTE", max(0, w - 22))]
            x = 1
            for label, width in cols:
                win.addstr(y, x, fit(label, width), curses.A_BOLD)
                x += width + 1
            y += 1

            # Render up to available lines
            signals: List[Dict[str, Any]] = latest.get("signals", []) or []
            for sig in signals[: max(0, h - y - 1)]:
                if y >= h - 1:
                    break
                sym = str(sig.get("symbol", ""))
                conf = "%.2f" % float(sig.get("avg_confidence", 0.0))
                note = str(sig.get("note", ""))
                row = [(sym, 10), (conf, 6), (note, max(0, w - 22))]
                x = 1
                for text, width in row:
                    win.addstr(y, x, fit(text, width))
                    x += width + 1
                y += 1

            # Tail log area
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
