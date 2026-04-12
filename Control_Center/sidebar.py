# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/sidebar.py
# Version: v1.0 Sidebar
# ----------------------------------------------------------------------------
# Purpose
# -------
# Left sidebar model and rendering for the Control Center TUI.
# Provides item selection and collapse/expand behavior.
# ============================================================================

from __future__ import annotations
from typing import List, Tuple
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit


class Sidebar:
    def __init__(self) -> None:
        # key, label, icon (ASCII-only)
        self.items: List[Tuple[str, str]] = [
            ("miner", "Miner"),
            ("prediction", "AutoTrader"),
            ("neuralis", "Neuralis AI"),
            ("ada", "VHW / ADA"),
            ("commands", "Commands"),
            ("system", "System Status"),
            ("settings", "Settings"),
        ]
        self.active: int = 0
        self.collapsed: bool = False
        self._last_action_ts: float = 0.0
        self._debounce_s: float = 0.2
        self._last_emit_key: str = ""

    def toggle(self) -> None:
        self.collapsed = not self.collapsed

    def width(self) -> int:
        return 5 if self.collapsed else 24

    def next(self) -> None:
        self.active = (self.active + 1) % len(self.items)

    def prev(self) -> None:
        self.active = (self.active - 1) % len(self.items)

    def current_key(self) -> str:
        return self.items[self.active][0]

    def click(self, bus=None) -> None:
        """Emit panel switch with debounce."""
        now = time.time()
        if (now - self._last_action_ts) < self._debounce_s:
            return
        self._last_action_ts = now
        key = self.current_key()
        if key == self._last_emit_key:
            return
        self._last_emit_key = key
        try:
            if bus is not None and hasattr(bus, "publish"):
                bus.publish("panel.switch", {"panel": key, "ts": now})
        except Exception:
            pass

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "Menu" if not self.collapsed else None)
            if self.collapsed:
                # Render icons-only (first 3 chars of label uppercase)
                h, w = win.getmaxyx()
                y = 1
                for idx, (key, label) in enumerate(self.items):
                    if y >= h - 1:
                        break
                    style = curses.color_pair(Theme.CP_SIDEBAR)
                    if idx == self.active:
                        style |= curses.A_BOLD
                    icon = (label.upper()[:3]).ljust(max(1, w - 2))
                    win.addstr(y, 1, icon[: max(0, w - 2)], style)
                    y += 1
                win.refresh()
                return
            h, w = win.getmaxyx()
            y = 1
            for idx, (_key, label) in enumerate(self.items):
                if y >= h - 1:
                    break
                style = curses.color_pair(Theme.CP_SIDEBAR)
                if idx == self.active:
                    style |= curses.A_BOLD
                line = "  " + fit(label, max(0, w - 4))
                win.addstr(y, 1, line, style)
                y += 1
            win.refresh()
        except Exception:
            pass
