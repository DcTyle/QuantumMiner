# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/panel_miner.py
# Version: v1.0 Miner Panel
# ----------------------------------------------------------------------------
# Purpose
# -------
# Display per-network mining telemetry: per-second rates and counters pulled
# from VSD paths written by the miner Submitter and related components.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List
import json
import os
import time

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit


def _read_runtime_networks() -> List[str]:
    """Attempt to read networks from miner_runtime_config.json."""
    try:
        root = os.path.dirname(os.path.dirname(__file__))
        cfg_path = os.path.join(root, "miner", "miner_runtime_config.json")
        if not os.path.exists(cfg_path):
            return []
        with open(cfg_path, "r", encoding="ascii", errors="ignore") as f:
            data = json.load(f)
        coins = data.get("coins", {}) or {}
        if isinstance(coins, dict) and "wallet" in coins and isinstance(coins["wallet"], dict):
            # Transposed structure
            nets = [str(k).upper() for k in coins["wallet"].keys()]
        else:
            # Old structure
            nets = [str(k).upper() for k in coins.keys()]
        nets.sort()
        return nets
    except Exception:
        return []


class MinerPanel:
    def __init__(self, vsd: Any) -> None:
        self.vsd = vsd
        self.networks: List[str] = []
        # Probe known index or fall back to runtime config
        self._discover_networks()
        self._last_refresh_s = 0.0
        self._log: List[str] = []
        self._scroll: int = 0  # 0 means bottom (auto-scroll)

    def _discover_networks(self) -> None:
        nets: List[str] = []
        try:
            idx = self.vsd.get("telemetry/metrics/index", [])
            if isinstance(idx, list):
                nets = [str(x).upper() for x in idx]
        except Exception:
            nets = []
        if not nets:
            nets = _read_runtime_networks()
        # Always include AGG if present
        try:
            agg = self.vsd.get("telemetry/metrics/AGG/current", None)
            if agg is not None and "AGG" not in nets:
                nets = ["AGG"] + nets
        except Exception:
            pass
        self.networks = nets

    def _read_row(self, net: str) -> Dict[str, Any]:
        cur_key = f"telemetry/metrics/{net}/current"
        agg_key = f"telemetry/metrics/{net}/shares/aggregate"
        out: Dict[str, Any] = {
            "net": net,
            "submitted_hs": 0.0,
            "found_hs": 0.0,
            "accepted_hs": 0.0,
            "submitted": 0,
            "accepted": 0,
            "found": 0,
            "age_s": 0.0,
        }
        now = time.time()
        try:
            cur = self.vsd.get(cur_key, {}) or {}
            out["submitted_hs"] = float(cur.get("hashes_submitted_hs", 0.0))
            out["found_hs"] = float(cur.get("hashes_found_hs", 0.0))
            out["accepted_hs"] = float(cur.get("accepted_hs", 0.0))
        except Exception:
            pass
        try:
            agg = self.vsd.get(agg_key, {}) or {}
            out["submitted"] = int(agg.get("submitted", 0))
            out["accepted"] = int(agg.get("accepted", 0))
            out["found"] = int(agg.get("found", 0))
            ts = float(agg.get("last_ts", 0.0) or 0.0)
            out["age_s"] = max(0.0, now - ts) if ts > 0 else 0.0
        except Exception:
            pass
        return out

    def _read_control_state(self) -> Dict[str, Any]:
        try:
            state = self.vsd.get("miner/control/state", {}) or {}
            return dict(state) if isinstance(state, dict) else {}
        except Exception:
            return {}

    def refresh_fast(self) -> None:
        # Periodically rediscover networks (slow path)
        if time.time() - self._last_refresh_s > 10.0:
            self._discover_networks()
            self._last_refresh_s = time.time()

    def on_message(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            line = "%s %s" % (topic, str(payload)[:200])
            self._log.append(line)
            if len(self._log) > 500:
                self._log = self._log[-500:]
            if self._scroll == 0:
                # stay at bottom when new messages arrive
                self._scroll = 0
        except Exception:
            pass

    def handle_key(self, ch: int) -> None:
        # Basic scrolling for the log tail (PgUp/PgDn)
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

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "Miner")
            h, w = win.getmaxyx()
            state = self._read_control_state()
            phase = str(state.get("phase", "running")).upper()
            paused = bool(state.get("paused", False))
            note = str(state.get("note", ""))
            status = "Status: %s" % phase
            if note:
                status += " | " + note
            win.addstr(1, 1, fit(status, max(0, w - 2)), curses.A_BOLD if paused else 0)
            # Header
            header = [
                ("NET", 6),
                ("SUB H/S", 12),
                ("FND H/S", 12),
                ("ACC H/S", 12),
                ("SUB", 8),
                ("ACC", 8),
                ("FND", 8),
                ("AGE", 8),
            ]
            x = 1
            y = 2
            for label, width in header:
                win.addstr(y, x, fit(label, width), curses.A_BOLD)
                x += width + 1
            y += 1
            # Rows
            for net in self.networks:
                if y >= h - 1:
                    break
                row = self._read_row(net)
                cols = [
                    (row["net"], 6),
                    ("%.2f" % row["submitted_hs"], 12),
                    ("%.2f" % row["found_hs"], 12),
                    ("%.2f" % row["accepted_hs"], 12),
                    (str(row["submitted"]), 8),
                    (str(row["accepted"]), 8),
                    (str(row["found"]), 8),
                    ("%.0f" % row["age_s"], 8),
                ]
                x = 1
                for text, width in cols:
                    win.addstr(y, x, fit(text, width))
                    x += width + 1
                y += 1

            # Render a small tail log at bottom if space allows
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
