# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/theme.py
# Version: v1.0 TUI Theme
# ----------------------------------------------------------------------------
# Purpose
# -------
# Curses color and drawing helpers for the ASCII-only Control Center.
# Provides a small palette and simple box/title utilities.
# ============================================================================

from __future__ import annotations
from typing import Tuple

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore


class Theme:
    """Simple color palette with light/dark modes.

    Mode toggling adjusts color pairs at runtime. All drawing uses
    these pairs so theme changes apply immediately on next render.
    """

    # Color pair ids (curses uses small ints)
    CP_DEFAULT = 1
    CP_ACCENT = 2
    CP_STATUS = 3
    CP_SIDEBAR = 4
    CP_WARNING = 5
    CP_OK = 6
    CP_ERROR = 7

    # Box drawing chars (ASCII only)
    H = "-"
    V = "|"
    TL = "+"
    TR = "+"
    BL = "+"
    BR = "+"

    MODE = "dark"

    @staticmethod
    def set_mode(mode: str) -> None:
        mode_l = str(mode or "dark").lower()
        if mode_l in ("light", "dark"):
            Theme.MODE = mode_l

    @staticmethod
    def init_colors() -> None:
        if curses is None:
            return
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()

        # Choose conservative mappings that work in most terminals
        try:
            if Theme.MODE == "light":
                curses.init_pair(Theme.CP_DEFAULT, curses.COLOR_BLACK, -1)
                curses.init_pair(Theme.CP_ACCENT, curses.COLOR_BLUE, -1)
                curses.init_pair(Theme.CP_STATUS, curses.COLOR_BLACK, curses.COLOR_WHITE)
                curses.init_pair(Theme.CP_SIDEBAR, curses.COLOR_BLUE, -1)
                curses.init_pair(Theme.CP_WARNING, curses.COLOR_YELLOW, -1)
                curses.init_pair(Theme.CP_OK, curses.COLOR_GREEN, -1)
                curses.init_pair(Theme.CP_ERROR, curses.COLOR_RED, -1)
            else:
                curses.init_pair(Theme.CP_DEFAULT, curses.COLOR_WHITE, -1)
                curses.init_pair(Theme.CP_ACCENT, curses.COLOR_CYAN, -1)
                curses.init_pair(Theme.CP_STATUS, curses.COLOR_BLACK, curses.COLOR_CYAN)
                curses.init_pair(Theme.CP_SIDEBAR, curses.COLOR_CYAN, -1)
                curses.init_pair(Theme.CP_WARNING, curses.COLOR_YELLOW, -1)
                curses.init_pair(Theme.CP_OK, curses.COLOR_GREEN, -1)
                curses.init_pair(Theme.CP_ERROR, curses.COLOR_RED, -1)
        except Exception:
            # If color init fails, proceed without colors
            pass


def draw_box(win, title: str | None = None) -> None:
    """Draw an ASCII box with optional title."""
    if curses is None:
        return
    try:
        h, w = win.getmaxyx()
        # Top/Bottom
        win.addstr(0, 0, Theme.TL + Theme.H * max(0, w - 2) + Theme.TR)
        win.addstr(max(0, h - 1), 0, Theme.BL + Theme.H * max(0, w - 2) + Theme.BR)
        # Sides
        for y in range(1, max(0, h - 1)):
            win.addstr(y, 0, Theme.V)
            win.addstr(y, max(0, w - 1), Theme.V)
        if title:
            t = " " + str(title) + " "
            t = t[: max(0, w - 4)]
            win.addstr(0, 2, t, curses.color_pair(Theme.CP_ACCENT))
    except Exception:
        pass


def pad_text(text: str, width: int) -> str:
    s = (text or "")[: max(0, width)]
    if len(s) < width:
        s = s + " " * (width - len(s))
    return s


def fit(value: str, w: int) -> str:
    s = str(value or "")
    if len(s) <= w:
        return s + (" " * (w - len(s)))
    if w <= 3:
        return s[:w]
    return s[: w - 3] + "..."
