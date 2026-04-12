# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/ui_layout.py
# Version: v1.0 TUI Layout
# ----------------------------------------------------------------------------
# Purpose
# -------
# Compose curses windows: status bar, sidebar, and main panel viewport.
# Manage tab switching and simple key bindings.
# ============================================================================

from __future__ import annotations
from typing import Dict, Any
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, pad_text, fit


class UiLayout:
    def __init__(self, stdscr, sidebar):
        self.stdscr = stdscr
        self.sidebar = sidebar
        self.status_win = None
        self.sidebar_win = None
        self.main_win = None
        self.active_tab = "miner"
        self.mic_enabled = False
        self._header = "Miner"
        self._notifications: list[str] = []
        self._auto_scroll: Dict[str, bool] = {
            "miner": True,
            "prediction": True,
            "neuralis": True,
            "ada": True,
            "system": True,
            "settings": True,
        }

    def resize(self) -> None:
        if curses is None:
            return
        h, w = self.stdscr.getmaxyx()
        # status bar (1 line)
        if self.status_win is None:
            self.status_win = curses.newwin(1, w, 0, 0)
        else:
            self.status_win.resize(1, w)
            self.status_win.mvwin(0, 0)

        # sidebar
        sw = self.sidebar.width()
        if self.sidebar_win is None:
            self.sidebar_win = curses.newwin(max(0, h - 1), sw, 1, 0)
        else:
            self.sidebar_win.resize(max(0, h - 1), sw)
            self.sidebar_win.mvwin(1, 0)

        # main panel occupies rest
        mw = max(1, w - sw)
        if self.main_win is None:
            self.main_win = curses.newwin(max(0, h - 1), mw, 1, sw)
        else:
            self.main_win.resize(max(0, h - 1), mw)
            self.main_win.mvwin(1, sw)

    def notify(self, msg: str) -> None:
        if not msg:
            return
        self._notifications.append(msg)
        if len(self._notifications) > 5:
            self._notifications = self._notifications[-5:]

    def render_status(self, note: str = "", input_buf: str = "") -> None:
        if curses is None:
            return
        if not self.status_win:
            return
        try:
            h, w = self.status_win.getmaxyx()
            self.status_win.erase()
            # Status content: app name + mic + time + note
            mic = "[voice:ON]" if self.mic_enabled else "[voice:OFF]"
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            # If mic is enabled, show input buffer inline
            cmd = (" cmd: " + input_buf) if (self.mic_enabled and input_buf) else ""
            left = " CC: " + self._header + "  " + mic + cmd
            notif = (" | ".join(self._notifications[-2:])) if self._notifications else ""
            right = ts
            center = ("  " + note if note else "") + ("  " + notif if notif else "")
            mid = max(0, w - len(left) - len(right))
            center = center[:mid]
            gap = max(0, mid - len(center))
            line = left + center + (" " * gap) + right
            self.status_win.addstr(0, 0, pad_text(line, w), curses.color_pair(Theme.CP_STATUS))
            self.status_win.refresh()
        except Exception:
            pass

    def render_sidebar(self) -> None:
        if curses is None or not self.sidebar_win:
            return
        self.sidebar.render(self.sidebar_win)

    def render_main(self, panel) -> None:
        if curses is None or not self.main_win:
            return
        panel.render(self.main_win)

    def show_panel(self, panel_name: str) -> None:
        """Switch active panel and update header and sidebar highlight."""
        name = str(panel_name or self.active_tab).lower()
        valid = {
            "miner": "Miner",
            "prediction": "AutoTrader",
            "neuralis": "Neuralis",
            "ada": "VHW / ADA",
            "system": "System",
            "settings": "Settings",
        }
        if name not in valid:
            return
        self.active_tab = name
        self._header = valid[name]
        # also notify to status line
        self.notify("Panel: " + self._header)
