# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/panel_settings.py
# Version: v1.0 Settings Panel
# ----------------------------------------------------------------------------
# Purpose
# -------
# Interactive settings for theme, timestamp precision, log verbosity, and
# per-panel auto-scroll toggles. Applies changes immediately via a callback
# into the Control Center app so layout/theme can react.
# ============================================================================

from __future__ import annotations
from typing import Any, Callable, Dict, List

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit


class SettingsPanel:
    def __init__(self, vsd: Any, on_change: Callable[[str, Any], None]) -> None:
        self.vsd = vsd
        self.on_change = on_change
        self.items: List[Dict[str, Any]] = [
            {"key": "theme", "label": "Theme", "value": "dark", "choices": ["dark", "light"]},
            {"key": "timestamp_precision", "label": "Timestamp Precision", "value": False},
            {"key": "verbosity", "label": "Log Verbosity", "value": "medium", "choices": ["low", "medium", "high"]},
            {"key": "autostart:miner", "label": "Start Miner on App Boot", "value": False},
            {"key": "autostart:prediction", "label": "Start Prediction Engine on App Boot", "value": False},
            {"key": "autoscroll:miner", "label": "Auto-scroll Miner", "value": True},
            {"key": "autoscroll:prediction", "label": "Auto-scroll Prediction", "value": True},
            {"key": "autoscroll:neuralis", "label": "Auto-scroll Neuralis", "value": True},
            {"key": "autoscroll:system", "label": "Auto-scroll System", "value": True},
        ]
        self.active: int = 0

    def _toggle_value(self, item: Dict[str, Any]) -> None:
        if "choices" in item and isinstance(item["choices"], list):
            choices = item["choices"]
            cur = item.get("value")
            try:
                idx = choices.index(cur)
            except Exception:
                idx = -1
            nxt = choices[(idx + 1) % len(choices)]
            item["value"] = nxt
        else:
            item["value"] = not bool(item.get("value", False))
        # call back
        try:
            self.on_change(str(item["key"]), item.get("value"))
        except Exception:
            pass

    def handle_key(self, ch: int) -> None:
        try:
            if ch in (curses.KEY_UP, ord('k')):
                self.active = (self.active - 1) % len(self.items)
            elif ch in (curses.KEY_DOWN, ord('j')):
                self.active = (self.active + 1) % len(self.items)
            elif ch in (curses.KEY_ENTER, 10, 13, ord(' ')):
                self._toggle_value(self.items[self.active])
        except Exception:
            pass

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "Settings")
            h, w = win.getmaxyx()
            y = 1
            for i, it in enumerate(self.items):
                if y >= h - 1:
                    break
                style = curses.A_NORMAL
                if i == self.active:
                    style |= curses.A_BOLD
                label = str(it.get("label"))
                value = it.get("value")
                if isinstance(value, bool):
                    vtxt = "ON" if value else "OFF"
                else:
                    vtxt = str(value)
                line = "%s: %s" % (label, vtxt)
                win.addstr(y, 1, fit(line, max(0, w - 2)), style)
                y += 1
            win.refresh()
        except Exception:
            pass
