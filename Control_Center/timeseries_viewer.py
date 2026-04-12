from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from VHW.vsd_manager import VSDManager, get_event_bus
except Exception:
    VSDManager = object  # type: ignore

from prediction_engine.timeseries_writer import TimeSeriesWriter


class TimeSeriesViewer(QWidget):
    """Interactive time series viewer that expands on zoom.

    Data strategy:
      - Attempts to read from VSD keys:
          prediction/timeseries/{symbol}/1h/current
          prediction/timeseries/{symbol}/5m/current
          prediction/timeseries/{symbol}/1m/current
        Each expected to be a JSON-serializable list of bars with t,o,h,l,c,v
      - If not found, shows an empty graph.

    Zoom behavior:
      - >12h window -> show 1h bars
      - 1h..12h -> show 5m bars (expand 1h if 5m missing)
      - <1h -> show 1m bars (expand 5m when possible)
    """

    def __init__(self, vsd: Any, symbols: List[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.vsd = vsd
        # Local helper for range queries; safe to create lazily inside GUI.
        self._writer = TimeSeriesWriter(vsd)
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "ETCUSDT", "LTCUSDT", "RVNUSDT"]
        # Initialize from VSD-selected symbol if available
        try:
            sel = self.vsd.get("control_center/timeseries/selected_symbol", None)
            self.current_symbol = str(sel).upper() if sel else self.symbols[0]
        except Exception:
            self.current_symbol = self.symbols[0]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Grain + window state
        self._grains = ["YEAR", "MONTH", "WEEK", "DAY", "HOUR"]
        self._current_grain = "DAY"
        self._center_ts = int(time.time())

        # Controls
        row = QHBoxLayout()
        row.addWidget(QLabel("Symbol:"))
        self.cmb_symbol = QComboBox()
        self.cmb_symbol.addItems(self.symbols)
        self.cmb_symbol.currentTextChanged.connect(self._on_symbol_change)
        row.addWidget(self.cmb_symbol)

        self.lbl_grain = QLabel("")
        row.addWidget(self.lbl_grain)

        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self._reset_view)
        row.addStretch(1)
        row.addWidget(btn_reset)
        layout.addLayout(row)

        # Chart
        self.fig = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#151821")
        self.fig.patch.set_facecolor("#151821")
        self.ax.grid(color="#333333", linestyle=":", linewidth=0.5)
        self.ax.set_xlabel("t (epoch seconds)")
        self.ax.set_ylabel("price")

        self._line = None
        self.view_start = int(time.time()) - 24 * 3600
        self.view_end = int(time.time())

        # Connect scroll + hover
        self.cid_scroll = self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._hover_label = QLabel("")
        layout.addWidget(self._hover_label)

        self.setFocusPolicy(Qt.StrongFocus)

        # Subscribe to VSD changes so commands can switch symbols live
        try:
            get_event_bus().subscribe("vsd.write", self._on_vsd_write)
        except Exception:
            pass

        self._refresh_plot()

    # ------------------ Key events ------------------
    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key_Up:
            self._grain_up()
        elif key == Qt.Key_Down:
            self._grain_down()
        elif key == Qt.Key_Left:
            self._step_window(-1)
        elif key == Qt.Key_Right:
            self._step_window(1)
        else:
            super().keyPressEvent(event)

    # ------------------ UI Actions ------------------
    def _on_symbol_change(self, sym: str) -> None:
        self.current_symbol = sym
        try:
            self.vsd.store("control_center/timeseries/selected_symbol", sym)
        except Exception:
            pass
        self._refresh_plot()

    def _reset_view(self) -> None:
        self._center_ts = int(time.time())
        self._update_window_for_grain()
        self._refresh_plot()

    def _on_scroll(self, event) -> None:
        # scroll up = finer grain; scroll down = coarser grain
        if event.button == "up":
            self._grain_up()
        else:
            self._grain_down()

    def _on_motion(self, event) -> None:
        try:
            if event.xdata is None or not hasattr(self, "_last_bars"):
                return
            x = float(event.xdata)
            bars = self._last_bars
            if not bars:
                return
            # nearest bar by time
            nearest = min(bars, key=lambda b: abs(int(b.get("t", 0)) - x))
            ts = int(nearest.get("t", 0))
            price = float(nearest.get("c", nearest.get("o", 0.0)))
            vol = float(nearest.get("v", 0.0))
            self._hover_label.setText(f"t={ts} c={price:.4f} v={vol:.4f}")
        except Exception:
            pass

    def _on_vsd_write(self, payload: Dict[str, Any]) -> None:
        try:
            k = str((payload or {}).get("key", ""))
        except Exception:
            k = ""
        # React to symbol selection changes
        if k == "control_center/timeseries/selected_symbol":
            try:
                sym = str(self.vsd.get(k, self.current_symbol)).upper()
                if sym and sym in self.symbols and sym != self.current_symbol:
                    self.current_symbol = sym
                    # update combo without emitting change loop
                    idx = self.cmb_symbol.findText(sym)
                    if idx >= 0:
                        self.cmb_symbol.blockSignals(True)
                        self.cmb_symbol.setCurrentIndex(idx)
                        self.cmb_symbol.blockSignals(False)
                    self._refresh_plot()
            except Exception:
                pass
        # Refresh when timeseries data updates for current symbol
        elif k.startswith(f"prediction/timeseries/{self.current_symbol}/"):
            self._refresh_plot()

    # ------------------ Data loading ------------------
    def _load_json(self, key: str):
        try:
            val = self.vsd.get(key)
            return val
        except Exception:
            return None

    def _choose_level(self) -> str:
        # Map grain to preferred base resolution; actual data is 1m, so
        # this is mostly advisory for future use.
        if self._current_grain in ("YEAR", "MONTH", "WEEK"):
            return "1h"
        if self._current_grain == "DAY":
            return "1h"
        return "1m"

    def _load_bars(self) -> List[Dict[str, float]]:
        # Delegate to prediction_engine query_range via writer helper.
        try:
            return self._writer.query_range(self.current_symbol, int(self.view_start), int(self.view_end))
        except Exception:
            return []

    def _load_from_manifest(self, sym: str, lvl: str, manifest: Dict[str, Any]) -> List[Dict[str, float]]:
        # Expect manifest like { '1m': [ { 't0': 123, 'key': 'prediction/timeseries/...'}, ...], ... }
        spans = manifest.get(lvl)
        if not isinstance(spans, list):
            return []
        start, end = int(self.view_start), int(self.view_end)
        bars: List[Dict[str, float]] = []
        for entry in spans:
            try:
                t0 = int(entry.get('t0', 0)) if isinstance(entry, dict) else 0
                t1 = int(entry.get('t1', 0)) if isinstance(entry, dict) else 0
                # simple overlap check; if t1 missing, assume relevant
                if t1 and (t1 < start or t0 > end):
                    continue
                # resolve block(s)
                block_data = None
                if isinstance(entry, dict) and isinstance(entry.get('block'), dict):
                    block_data = entry['block']
                elif isinstance(entry, dict) and isinstance(entry.get('key'), str):
                    block_data = self._load_json(entry['key'])
                elif isinstance(entry, str):
                    block_data = self._load_json(entry)

                if not block_data:
                    continue

                # if this level is already bars (1m), extend directly
                if lvl == '1m' and isinstance(block_data, list):
                    for b in block_data:
                        t = int(b.get('t', 0))
                        if start <= t <= end:
                            bars.append(b)
                # if 5m level: blocks likely 5m objects; if list, take items
                elif lvl == '5m':
                    if isinstance(block_data, list):
                        # each item should be a 5m block; approximate using block base time
                        for blk in block_data:
                            t = int(blk.get('t0', 0))
                            if start <= t <= end:
                                bars.append({
                                    't': t,
                                    'c': float(blk.get('base', {}).get('c', 0)) / 1_000_000.0,
                                })
                    elif isinstance(block_data, dict):
                        t = int(block_data.get('t0', 0))
                        if start <= t <= end:
                            bars.append({
                                't': t,
                                'c': float(block_data.get('base', {}).get('c', 0)) / 1_000_000.0,
                            })
                # if 1h level: blocks likely 1h objects
                elif lvl == '1h':
                    if isinstance(block_data, list):
                        for blk in block_data:
                            t = int(blk.get('t0', 0))
                            if start <= t <= end:
                                bars.append({
                                    't': t,
                                    'c': float(blk.get('base', {}).get('c', 0)) / 1_000_000.0,
                                })
                    elif isinstance(block_data, dict):
                        t = int(block_data.get('t0', 0))
                        if start <= t <= end:
                            bars.append({
                                't': t,
                                'c': float(block_data.get('base', {}).get('c', 0)) / 1_000_000.0,
                            })
            except Exception:
                continue
        bars.sort(key=lambda b: b.get('t', 0))
        return bars

    # ------------------ Plotting ------------------
    def _refresh_plot(self) -> None:
        bars = self._load_bars()
        self._last_bars = bars
        self.ax.clear()
        self.ax.set_facecolor("#151821")
        self.ax.grid(color="#333333", linestyle=":", linewidth=0.5)
        self.ax.set_xlabel("t (epoch seconds)")
        self.ax.set_ylabel("price")

        # Decide if we can draw candles (need o,h,l,c)
        can_candle = bool(bars) and all(
            ('o' in b and 'h' in b and 'l' in b and 'c' in b) for b in bars
        )

        if can_candle:
            # Draw simple candles and volume overlay
            xs = [int(b.get('t', 0)) for b in bars]
            o = [float(b.get('o', 0.0)) for b in bars]
            h = [float(b.get('h', 0.0)) for b in bars]
            l = [float(b.get('l', 0.0)) for b in bars]
            c = [float(b.get('c', 0.0)) for b in bars]
            v = [float(b.get('v', 0.0)) for b in bars]

            w = max(5.0, (self.view_end - self.view_start) / 200.0)  # candle width in seconds
            for i, t in enumerate(xs):
                color = "#00ff84" if c[i] >= o[i] else "#ff5f5f"
                # wick
                self.ax.vlines(t, l[i], h[i], color=color, linewidth=0.8, zorder=2)
                # body
                y0, y1 = (o[i], c[i]) if c[i] >= o[i] else (c[i], o[i])
                self.ax.add_patch(
                    self._rect(t - w/2.0, y0, w, max(1e-9, y1 - y0), color=color, alpha=0.9)
                )
            # Volume on twin axis
            try:
                ax2 = self.ax.twinx()
                ax2.set_ylim(0, max(1.0, max(v) * 4.0))
                ax2.bar(xs, v, width=w, color="#3a4a5e", alpha=0.35, zorder=1)
                ax2.set_yticks([])
            except Exception:
                pass
        else:
            # Fallback to close line
            xs = [b.get("t", 0) for b in bars]
            ys = [b.get("c", 0.0) for b in bars]
            if xs:
                self.ax.plot(xs, ys, color="#00b3ff", linewidth=1.2)

        self.ax.set_xlim(self.view_start, self.view_end)
        self.canvas.draw_idle()

        # Update header label with basic window metrics
        try:
            if bars:
                closes = [float(b.get("c", b.get("o", 0.0))) for b in bars]
                vols = [float(b.get("v", 0.0)) for b in bars]
                span_txt = f"{self.view_start}..{self.view_end}"
                self.lbl_grain.setText(
                    f"{self.current_symbol} | {self._current_grain} | {span_txt} | "
                    f"min={min(closes):.4f} max={max(closes):.4f} vol={sum(vols):.4f}"
                )
            else:
                self.lbl_grain.setText(f"{self.current_symbol} | {self._current_grain} | no data")
        except Exception:
            self.lbl_grain.setText(f"{self.current_symbol} | {self._current_grain}")

    # ------------------ Grain/window helpers ------------------
    def _grain_index(self) -> int:
        try:
            return self._grains.index(self._current_grain)
        except ValueError:
            return 3  # default DAY

    def _grain_up(self) -> None:
        idx = self._grain_index()
        if idx < len(self._grains) - 1:
            self._current_grain = self._grains[idx + 1]
        self._update_window_for_grain()
        self._refresh_plot()

    def _grain_down(self) -> None:
        idx = self._grain_index()
        if idx > 0:
            self._current_grain = self._grains[idx - 1]
        self._update_window_for_grain()
        self._refresh_plot()

    def _step_window(self, direction: int) -> None:
        # direction: -1 = left, 1 = right
        unit = 3600
        if self._current_grain == "HOUR":
            unit = 3600
        elif self._current_grain == "DAY":
            unit = 24 * 3600
        elif self._current_grain == "WEEK":
            unit = 7 * 24 * 3600
        elif self._current_grain == "MONTH":
            unit = 30 * 24 * 3600
        elif self._current_grain == "YEAR":
            unit = 365 * 24 * 3600
        self._center_ts += direction * unit
        self._update_window_for_grain()
        self._refresh_plot()

    def _update_window_for_grain(self) -> None:
        mid = int(self._center_ts)
        if self._current_grain == "HOUR":
            span = 3600
        elif self._current_grain == "DAY":
            span = 24 * 3600
        elif self._current_grain == "WEEK":
            span = 7 * 24 * 3600
        elif self._current_grain == "MONTH":
            span = 30 * 24 * 3600
        elif self._current_grain == "YEAR":
            span = 365 * 24 * 3600
        else:
            span = self.view_end - self.view_start if self.view_end > self.view_start else 24 * 3600
        self.view_start = mid - span // 2
        self.view_end = mid + span // 2

    def _rect(self, x: float, y: float, w: float, h: float, color: str, alpha: float):
        from matplotlib.patches import Rectangle
        return Rectangle((x, y), w, h, facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.6)
