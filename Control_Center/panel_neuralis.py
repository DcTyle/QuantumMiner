# ============================================================================
# Quantum Application / Control Center
# ASCII-ONLY SOURCE FILE
# File: Control_Center/panel_neuralis.py
# Version: v1.0 Neuralis Panel
# ----------------------------------------------------------------------------
# Purpose
# -------
# Display basic Neuralis/BIOS state: EventBus stats, last engine snapshot,
# Murphy watchdog health, and a small rolling alerts list.
# ============================================================================

from __future__ import annotations
from typing import Any, Dict, List
from Neuralis_AI.context_map import build_context_graph
from Neuralis_AI.cognition_summary import build_summary

try:
    import curses
except Exception:  # pragma: no cover - runtime-only
    curses = None  # type: ignore

from .theme import Theme, draw_box, fit


class NeuralisPanel:
    def __init__(self, vsd: Any, bus: Any) -> None:
        self.vsd = vsd
        self.bus = bus
        self._log: List[Any] = []  # accept str or (text, level)
        self._scroll: int = 0
        self._build_context_tab()

    def _read_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "bus": {},
            "snapshot": "",
            "murphy": {},
            "alerts": [],
        }
    # ------------------------------------------------------------
    def _build_context_tab(self) -> None:
        """Add a context graph + VSD-backed viewer tab.

        This tab does not create or write any files; it only reads
        from VSD and renders decoded content in-memory.
        """
        if not hasattr(self, "notebook"):
            return
        import tkinter as tk
        from tkinter import ttk as _ttk

        tab = _ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Context")

        left = _ttk.Frame(tab)
        right = _ttk.Frame(tab)
        left.pack(side="left", fill="y")
        right.pack(side="right", fill="both", expand=True)

        tree = _ttk.Treeview(left, columns=("coherence",), show="tree headings")
        tree.heading("#0", text="Node")
        tree.heading("coherence", text="Coherence")
        tree.pack(fill="both", expand=True)

        text = tk.Text(right, wrap="none")
        text.pack(fill="both", expand=True)

        self._context_tree = tree
        self._context_text = text

        self._populate_context_tree()
        tree.bind("<<TreeviewSelect>>", self._on_context_select)

    # ------------------------------------------------------------
    def _populate_context_tree(self) -> None:
        tree = getattr(self, "_context_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        graph = build_context_graph()
        for node in graph.get("nodes", []):
            nid = node.get("id", "")
            label = node.get("label", nid)
            coh = node.get("coherence", 0.0)
            tree.insert("", "end", iid=nid, text=label, values=(f"{coh:.2f}",))
        # Add some virtual documents as children of a synthetic root.
        docs = [
            ("capabilities", "Cognition Capabilities Packet"),
            ("summary", "Cognition Summary Packet"),
            ("history", "Domain History"),
            ("graph", "Context Graph Packet"),
        ]
        parent = tree.insert("", "end", iid="_docs", text="Neuralis Documents")
        for did, label in docs:
            tree.insert(parent, "end", iid=f"doc:{did}", text=label)

    # ------------------------------------------------------------
    def _on_context_select(self, event: Any) -> None:
        tree = getattr(self, "_context_tree", None)
        text = getattr(self, "_context_text", None)
        if tree is None or text is None:
            return
        sel = tree.selection()
        if not sel:
            return
        iid = sel[0]
        try:
            vsd = VSDManager.global_instance()  # type: ignore[attr-defined]
        except Exception:
            vsd = None
        content: Dict[str, Any] = {}
        if iid == "doc:capabilities":
            from Neuralis_AI.packet_cognition_capabilities import read_packet

            if vsd is not None:
                try:
                    content = read_packet(vsd) or {}
                except Exception:
                    content = {}
        elif iid == "doc:summary":
            if vsd is not None:
                try:
                    content = build_summary(vsd) or {}
                except Exception:
                    content = {}
        elif iid == "doc:history":
            from Neuralis_AI.cognition_history import get_history

            if vsd is not None:
                try:
                    dm = {}
                    from Neuralis_AI.packet_cognition_capabilities import read_packet

                    pkt = read_packet(vsd) or {}
                    meta = (pkt.get("meta", {}) or {}).get("domain_meta", {}) or {}
                    for domain in meta.keys():
                        dm[domain] = get_history(vsd, domain, limit=16)
                    content = dm
                except Exception:
                    content = {}
        elif iid == "doc:graph":
            if vsd is not None:
                from Neuralis_AI.context_map import store_context_graph

                try:
                    content = store_context_graph(vsd) or {}
                except Exception:
                    content = build_context_graph()
            else:
                content = build_context_graph()
        else:
            # For now nodes just show their metadata from the graph.
            graph = build_context_graph()
            for node in graph.get("nodes", []):
                if node.get("id") == iid:
                    content = node
                    break

        import json

        text.delete("1.0", "end")
        try:
            rendered = json.dumps(content, indent=2, sort_keys=True)
        except Exception:
            rendered = str(content)
        text.insert("1.0", rendered)
        # EventBus stats if available
        try:
            if hasattr(self.bus, "stats"):
                out["bus"] = self.bus.stats()
        except Exception:
            pass
        # Last miner engine snapshot path
        try:
            snap = self.vsd.get("snapshots/last_engine_state", "")
            out["snapshot"] = str(snap or "")
        except Exception:
            pass
        # Murphy health
        try:
            out["murphy"] = self.vsd.get("murphy/health", {}) or {}
        except Exception:
            pass
        # Alerts (last 10)
        try:
            alerts: List[Dict[str, Any]] = self.vsd.get("murphy/alerts", []) or []
            if isinstance(alerts, list):
                out["alerts"] = alerts[-10:]
        except Exception:
            pass
        return out

    def render(self, win) -> None:
        if curses is None:
            return
        try:
            win.erase()
            draw_box(win, "Neuralis")
            h, w = win.getmaxyx()
            y = 1
            st = self._read_state()
            # Bus stats summary line
            bus = st.get("bus", {}) or {}
            line = "Bus: topics=%s handlers=%s events=%s" % (
                bus.get("topics", "?"), bus.get("handlers", "?"), bus.get("events_published", "?")
            )
            win.addstr(y, 1, fit(line, max(0, w - 2)))
            y += 1

            # Snapshot path
            snap = st.get("snapshot") or ""
            win.addstr(y, 1, fit("Snapshot: " + snap, max(0, w - 2)))
            y += 2

            # Murphy health
            mur = st.get("murphy", {}) or {}
            mh = "Murphy: overload_ticks=%s last_frame_age_s=%.1f last_plan_age_s=%.1f" % (
                mur.get("overload_ticks", 0), float(mur.get("last_frame_age_s", 0.0)), float(mur.get("last_plan_age_s", 0.0))
            )
            win.addstr(y, 1, fit(mh, max(0, w - 2)))
            y += 2

            # Alerts table
            win.addstr(y, 1, "ALERTS", curses.A_BOLD)
            y += 1
            for rec in st.get("alerts", []) or []:
                if y >= h - 1:
                    break
                note = str(rec.get("note", ""))
                ts = str(rec.get("ts", ""))
                win.addstr(y, 1, fit(ts + " " + note, max(0, w - 2)))
                y += 1

            # Tail log
            if y + 2 < h - 1 and self._log:
                win.addstr(y, 1, "Events:", curses.A_BOLD)
                y += 1
                max_lines = (h - 1) - y
                start = max(0, len(self._log) - max_lines - self._scroll)
                end = max(0, len(self._log) - self._scroll)
                for rec in self._log[start:end]:
                    if y >= h - 1:
                        break
                    style = curses.A_NORMAL
                    text = None
                    if isinstance(rec, tuple) and len(rec) == 2:
                        text, lvl = rec[0], str(rec[1] or "").lower()
                        if lvl == "ok":
                            style |= curses.color_pair(Theme.CP_OK)
                        elif lvl in ("warn", "warning"):
                            style |= curses.color_pair(Theme.CP_WARNING)
                        elif lvl in ("err", "error"):
                            style |= curses.color_pair(Theme.CP_ERROR)
                    else:
                        text = str(rec)
                    win.addstr(y, 1, fit(str(text), max(0, w - 2)), style)
                    y += 1
            win.refresh()
        except Exception:
            pass

    def on_message(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            # Expect optional 'text' and 'level' for nicer display
            if isinstance(payload, dict) and ("text" in payload):
                text = str(payload.get("text", ""))
                lvl = str(payload.get("level", "")).lower()
                self._log.append((text, lvl))
            else:
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
