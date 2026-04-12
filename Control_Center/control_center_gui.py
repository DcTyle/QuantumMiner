# ============================================================================
# Quantum Application / control_center
# File: control_center_gui.py
# PySide6 GUI for "Neuralis Control Center"
# ============================================================================
# Requirements:
#   pip install PySide6 matplotlib
#
# Usage:
#   python control_center_gui.py
#
# Notes:
#   - Designed to resemble the Neuralis Control Center screenshot.
#   - Integrates with control_center.__init__ if available:
#       start_miner_button()
#       start_prediction_engine_button()
#       start_neuralis_button()
#   - All data is currently simulated; you can wire in real VSD / telemetry
#     by replacing the stub update methods.
# ============================================================================

from __future__ import annotations
import json
import sys
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

from PySide6.QtCore import Qt, QTimer, QSize, QPointF
from PySide6.QtGui import QPalette, QColor, QFont, QIcon, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QTextEdit,
    QFrame,
    QSizePolicy,
)

# Matplotlib embedding for live charts
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import os

from .timeseries_viewer import TimeSeriesViewer
from VHW.vsd_manager import VSD


RESEARCH_ROOT = Path(__file__).resolve().parent.parent / "ResearchConfinement"

APP_BG = "#08111d"
APP_BG_ALT = "#0d1726"
APP_PANEL = "#101b2d"
APP_PANEL_ALT = "#142137"
APP_BORDER = "#23324a"
APP_TEXT = "#e6eef7"
APP_MUTED = "#8ea3b7"
APP_ACCENT = "#38bdf8"
APP_ACCENT_STRONG = "#0f766e"


def load_json_with_comment(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if text.lstrip().startswith("//"):
        text = "\n".join(text.splitlines()[1:])
    try:
        loaded = json.loads(text)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def build_fallback_research_model() -> dict:
    return {
        "schematic": {
            "nodes": [
                {"id": "header", "label": "Header Pulse", "x": 0.10, "y": 0.55, "weight": 0.52},
                {"id": "lattice", "label": "Lattice Cal", "x": 0.22, "y": 0.34, "weight": 0.58},
                {"id": "encode", "label": "Psi Encode", "x": 0.36, "y": 0.36, "weight": 0.61},
                {"id": "clamp", "label": "Clamp Band", "x": 0.48, "y": 0.24, "weight": 0.50},
                {"id": "manifold", "label": "M_t Fold", "x": 0.62, "y": 0.56, "weight": 0.78},
                {"id": "project", "label": "R_eff Projection", "x": 0.76, "y": 0.32, "weight": 0.63},
                {"id": "crosstalk", "label": "Crosstalk Cluster", "x": 0.87, "y": 0.58, "weight": 0.72},
                {"id": "queue", "label": "Stratum Queue", "x": 0.96, "y": 0.38, "weight": 0.60},
            ],
            "edges": [
                {"source": "header", "target": "lattice"},
                {"source": "lattice", "target": "encode"},
                {"source": "encode", "target": "clamp"},
                {"source": "encode", "target": "manifold"},
                {"source": "clamp", "target": "manifold"},
                {"source": "manifold", "target": "project"},
                {"source": "project", "target": "crosstalk"},
                {"source": "project", "target": "queue"},
                {"source": "crosstalk", "target": "queue"},
            ],
        },
        "pseudocode": [
            "lattice_cal = calibrate_silicon_lattice(nist, tensor_gradients, packet_classes, photonic_basins)",
            "Psi_encode = encode_basis(F, A, I, V, lattice_cal, pulse_packet_dev, mixed_cross_terms)",
            "Psi_t = M_t(Psi_encode, lattice_cal.environment_tensor, temporal_fold, crosstalk=shared_phase_lock, depth<=5)",
            "R_eff = project_effective_vector(Psi_t) -> [X, Y, Z, T_eff]",
            "diag = evaluate_vector_consistency(R_eff, path_equiv, ordering_delta, basis_rotation)",
            "header_freq = pulse_packet_dev(block_header, quartet, photon_pulse)",
            "target_cap = clamp_band(a_code, amplitude_window, trap_ratio, viol_band)",
            "coherent_vectors = bias_nonce_trajectories(R_eff, target_cap, carriers=70)",
            "worker_queue = package_stratum_batches(coherent_vectors, workers=5..10)",
        ],
        "silicon_calibration": {
            "field_pressure": 0.584,
            "larger_field_exposure": 0.618,
            "dominant_basin": {
                "basin_id": "exciton_basin",
                "field": "coherence_field",
                "particle": "exciton",
            },
            "field_environment": {
                "charge_field": 0.612,
                "lattice_field": 0.553,
                "coherence_field": 0.641,
                "vacancy_field": 0.398,
            },
        },
        "yield_sequence": [24, 31, 37, 42, 39, 45, 48, 44],
        "coherence_sequence": [0.82, 0.86, 0.89, 0.91, 0.90, 0.93, 0.94, 0.92],
        "vector_magnitude_sequence": [0.41, 0.47, 0.54, 0.59, 0.57, 0.63, 0.66, 0.64],
        "temporal_projection_sequence": [0.52, 0.56, 0.61, 0.67, 0.65, 0.71, 0.74, 0.72],
        "effective_vector": {
            "x": 0.332,
            "y": 0.284,
            "z": 0.466,
            "t_eff": 0.721,
            "spatial_magnitude": 0.639,
            "coherence_bias": 0.873,
        },
        "manifold_diagnostics": {
            "path_equivalence_error": 0.041,
            "temporal_ordering_delta": 0.073,
            "basis_rotation_residual": 0.052,
        },
        "latest_summary": {
            "yield_count": 44,
            "coherence_peak": 0.92,
            "nesting_depth": 5,
            "worker_count": 7,
            "state_capacity": 1680700000,
            "field_pressure": 0.584,
            "larger_field_exposure": 0.618,
            "dominant_basin": "exciton_basin",
            "effective_vector": {
                "x": 0.332,
                "y": 0.284,
                "z": 0.466,
                "t_eff": 0.721,
                "spatial_magnitude": 0.639,
                "coherence_bias": 0.873,
            },
            "temporal_projection": 0.721,
            "path_equivalence_error": 0.041,
            "basis_rotation_residual": 0.052,
        },
        "pulse_batches": [
            {
                "pulse_id": idx,
                "yield_count": value,
                "coherence_peak": [0.82, 0.86, 0.89, 0.91, 0.90, 0.93, 0.94, 0.92][idx],
                "nesting_depth": 5,
                "worker_count": 7,
                "worker_batches": [],
                "nonces": [],
                "vector_magnitude": [0.41, 0.47, 0.54, 0.59, 0.57, 0.63, 0.66, 0.64][idx],
                "temporal_projection": [0.52, 0.56, 0.61, 0.67, 0.65, 0.71, 0.74, 0.72][idx],
                "field_pressure": 0.584,
                "larger_field_exposure": 0.618,
                "dominant_basin": "exciton_basin",
                "path_equivalence_error": 0.041 + idx * 0.001,
                "basis_rotation_residual": 0.052 + idx * 0.001,
                "effective_vector": {
                    "x": 0.332,
                    "y": 0.284,
                    "z": 0.466,
                    "t_eff": 0.721,
                    "spatial_magnitude": 0.639,
                    "coherence_bias": 0.873,
                },
            }
            for idx, value in enumerate([24, 31, 37, 42, 39, 45, 48, 44])
        ],
    }
# ---------------------------------------------------------------------------
# Optional integration with control_center "start buttons"
# ---------------------------------------------------------------------------
try:
    from . import (
        start_miner_button,
        start_prediction_engine_button,
        start_neuralis_button,
    )
except Exception:
    # If control_center is not present yet, define safe shims that just print.
    def start_miner_button() -> bool:
        print("[ControlCenterGUI] start_miner_button() not wired")
        return False

    def start_prediction_engine_button() -> bool:
        print("[ControlCenterGUI] start_prediction_engine_button() not wired")
        return False

    def start_neuralis_button() -> bool:
        print("[ControlCenterGUI] start_neuralis_button() not wired")
        return False


# ---------------------------------------------------------------------------
# Dark theme utilities
# ---------------------------------------------------------------------------
def apply_dark_palette(app: QApplication) -> None:
    """Apply the shared modern dark palette and widget stylesheet."""
    palette = QPalette()
    base = QColor(APP_BG)
    base_alt = QColor(APP_BG_ALT)
    text = QColor(APP_TEXT)
    mid_text = QColor(APP_MUTED)
    accent = QColor(APP_ACCENT)

    palette.setColor(QPalette.Window, base)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, QColor(APP_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(APP_PANEL_ALT))
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, QColor(APP_PANEL_ALT))
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.Highlight, accent)
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipBase, QColor(APP_PANEL_ALT))
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.PlaceholderText, mid_text)

    app.setPalette(palette)
    app.setStyle("Fusion")
    app.setStyleSheet(_modern_dark_stylesheet())


def _modern_dark_stylesheet() -> str:
    return """
QMainWindow {
    background: #08111d;
}
QWidget#appRoot {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #08111d,
        stop:0.45 #0b1524,
        stop:1 #101b2d);
}
QWidget {
    color: #e6eef7;
    background-color: transparent;
    font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
    font-size: 10pt;
}
QGroupBox {
    background-color: #101b2d;
    border: 1px solid #23324a;
    border-radius: 18px;
    margin-top: 16px;
    padding: 14px;
}
QGroupBox::title {
    color: #8ea3b7;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 4px;
    padding: 0 6px;
    background-color: #101b2d;
}
QFrame#metricCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #122033,
        stop:1 #0f1726);
    border: 1px solid #28405f;
    border-radius: 16px;
}
QLabel#metricCaption {
    color: #8ea3b7;
    font-size: 8pt;
    font-weight: 600;
}
QLabel#metricValue {
    color: #f7fbff;
    font-size: 17pt;
    font-weight: 700;
}
QTabWidget::pane {
    border: 1px solid #23324a;
    border-radius: 18px;
    background-color: #101b2d;
    top: -1px;
}
QTabBar::tab {
    background-color: #0e1726;
    color: #8ea3b7;
    border: 1px solid #23324a;
    border-bottom: none;
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
    padding: 10px 18px;
    margin-right: 6px;
}
QTabBar::tab:selected {
    background-color: #15243a;
    color: #f7fbff;
}
QTabBar::tab:hover:!selected {
    background-color: #132036;
    color: #cfe8fb;
}
QPushButton {
    background-color: #132036;
    color: #e6eef7;
    border: 1px solid #29415f;
    border-radius: 12px;
    padding: 8px 16px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #172740;
    border-color: #3f5f85;
}
QPushButton:pressed {
    background-color: #0f1a2b;
}
QPushButton[accent="true"] {
    background-color: #0f766e;
    border-color: #2dd4bf;
    color: #f7fffd;
}
QPushButton[accent="true"]:hover {
    background-color: #11857c;
    border-color: #67e8f9;
}
QPushButton[quiet="true"] {
    background-color: #0e1726;
    border-color: #23324a;
    color: #9eb3c8;
}
QLineEdit,
QTextEdit,
QListWidget,
QTableWidget,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    background-color: #0d1726;
    color: #e6eef7;
    border: 1px solid #23324a;
    border-radius: 12px;
    padding: 8px 10px;
    selection-background-color: #0f766e;
    selection-color: #f7fbff;
}
QLineEdit:focus,
QTextEdit:focus,
QListWidget:focus,
QTableWidget:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #38bdf8;
}
QHeaderView::section {
    background-color: #132036;
    color: #c7d6e4;
    border: none;
    border-right: 1px solid #23324a;
    border-bottom: 1px solid #23324a;
    padding: 8px 10px;
    font-weight: 600;
}
QTableCornerButton::section {
    background-color: #132036;
    border: 1px solid #23324a;
}
QMenuBar {
    background-color: #0b1524;
    color: #d8e4ef;
    border-bottom: 1px solid #23324a;
}
QMenuBar::item {
    padding: 6px 10px;
    margin: 3px;
    border-radius: 8px;
}
QMenuBar::item:selected {
    background-color: #15243a;
}
QMenu {
    background-color: #0f1726;
    color: #d8e4ef;
    border: 1px solid #23324a;
    padding: 6px;
}
QMenu::item {
    padding: 8px 18px;
    border-radius: 8px;
}
QMenu::item:selected {
    background-color: #15243a;
}
QStatusBar {
    background-color: #0b1524;
    color: #90a6bb;
    border-top: 1px solid #23324a;
}
QScrollBar:vertical {
    background: #0b1524;
    width: 12px;
    margin: 4px 0 4px 0;
}
QScrollBar::handle:vertical {
    background: #23324a;
    min-height: 24px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background: #315074;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}
QToolTip {
    background-color: #132036;
    color: #f7fbff;
    border: 1px solid #38bdf8;
    padding: 6px 8px;
}
"""


def make_header_label(text: str) -> QLabel:
    lbl = QLabel(text)
    font = QFont("Segoe UI Semibold")
    font.setPointSize(10)
    font.setBold(True)
    lbl.setFont(font)
    return lbl


def make_small_label(text: str) -> QLabel:
    lbl = QLabel(text)
    font = QFont("Segoe UI")
    font.setPointSize(8)
    lbl.setFont(font)
    return lbl


def style_button(button: QPushButton, accent: bool = False, quiet: bool = False) -> QPushButton:
    button.setCursor(Qt.PointingHandCursor)
    button.setProperty("accent", accent)
    button.setProperty("quiet", quiet)
    button.style().unpolish(button)
    button.style().polish(button)
    return button


def make_metric_card(label: str, value: str) -> Tuple[QFrame, QLabel]:
    frame = QFrame()
    frame.setObjectName("metricCard")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(3)

    caption = make_small_label(str(label).upper())
    caption.setObjectName("metricCaption")
    val = QLabel(value)
    val.setObjectName("metricValue")
    layout.addWidget(caption)
    layout.addWidget(val)
    return frame, val


def _style_chart_axes(fig: Figure, ax) -> None:
    ax.set_facecolor(APP_PANEL)
    fig.patch.set_facecolor(APP_PANEL)
    ax.grid(color=APP_BORDER, linestyle=":", linewidth=0.7)
    ax.tick_params(colors=APP_MUTED, labelsize=8)
    ax.xaxis.label.set_color(APP_MUTED)
    ax.yaxis.label.set_color(APP_MUTED)
    for spine in ax.spines.values():
        spine.set_color(APP_BORDER)


# ---------------------------------------------------------------------------
# Matplotlib chart widget
# ---------------------------------------------------------------------------
class LivePredictionChart(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Signal")
        self.ax.set_ylim(0, 40)
        _style_chart_axes(self.fig, self.ax)

        # Simulated time series for 6 assets
        self.t_vals: List[float] = [0.0]
        self.series = {
            "ETH": [12.0],
            "ETC": [10.0],
            "RVN": [8.0],
            "LTC": [6.0],
            "BTC": [5.0],
        }

        self.lines = {}
        colors = {
            "ETH": "#ffc300",
            "ETC": "#00ff84",
            "RVN": "#ff5f5f",
            "LTC": "#00b3ff",
            "BTC": "#ffffff",
        }

        for name, vals in self.series.items():
            line, = self.ax.plot(self.t_vals, vals, label=name, linewidth=1.5, color=colors[name])
            self.lines[name] = line

        self.ax.legend(loc="upper left", fontsize=7, frameon=False)
        self.canvas.draw()

    def tick(self) -> None:
        # simple dummy random-walk update
        t_next = self.t_vals[-1] + 0.2
        self.t_vals.append(t_next)
        for name, vals in self.series.items():
            last = vals[-1]
            delta = random.uniform(-0.6, 0.8)
            new = max(0.0, min(40.0, last + delta))
            vals.append(new)

            self.lines[name].set_data(self.t_vals, vals)

        # keep last 60 points
        if len(self.t_vals) > 60:
            self.t_vals = self.t_vals[-60:]
            for k in self.series:
                self.series[k] = self.series[k][-60:]
                self.lines[k].set_data(self.t_vals, self.series[k])

        self.ax.set_xlim(max(0.0, self.t_vals[0]), self.t_vals[-1])
        self.canvas.draw_idle()


class LiveNonceYieldChart(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Pulse")
        self.ax.set_ylabel("Yield")
        self.ax.set_ylim(0, 80)
        _style_chart_axes(self.fig, self.ax)

        self._yield_sequence: List[float] = [24.0]
        self._coherence_sequence: List[float] = [0.82]
        self._sequence_idx = -1
        self.t_vals: List[float] = [0.0]
        self.yield_vals: List[float] = [24.0]
        self.coherence_vals: List[float] = [0.82 * 70.0]

        self.yield_line, = self.ax.plot(self.t_vals, self.yield_vals, label="Yield", linewidth=1.8, color="#7ee787")
        self.coherence_line, = self.ax.plot(self.t_vals, self.coherence_vals, label="Coherence x70", linewidth=1.4, color="#58a6ff")
        self.ax.legend(loc="upper left", fontsize=7, frameon=False)
        self.canvas.draw()

    def load_sequences(self, yields: List[float], coherences: List[float]) -> None:
        if yields:
            self._yield_sequence = [float(value) for value in yields]
        if coherences:
            self._coherence_sequence = [float(value) for value in coherences]

    def tick(self) -> None:
        if not self._yield_sequence:
            return

        self._sequence_idx = (self._sequence_idx + 1) % len(self._yield_sequence)
        t_next = self.t_vals[-1] + 1.0
        y_next = float(self._yield_sequence[self._sequence_idx])
        c_next = float(
            self._coherence_sequence[self._sequence_idx % max(1, len(self._coherence_sequence))]
        ) * 70.0

        self.t_vals.append(t_next)
        self.yield_vals.append(y_next)
        self.coherence_vals.append(c_next)

        if len(self.t_vals) > 60:
            self.t_vals = self.t_vals[-60:]
            self.yield_vals = self.yield_vals[-60:]
            self.coherence_vals = self.coherence_vals[-60:]

        self.yield_line.set_data(self.t_vals, self.yield_vals)
        self.coherence_line.set_data(self.t_vals, self.coherence_vals)
        self.ax.set_xlim(max(0.0, self.t_vals[0]), self.t_vals[-1] + 0.5)
        self.canvas.draw_idle()


class ResearchPulseGraph(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._phase = 0.0
        self._schematic = build_fallback_research_model().get("schematic", {})
        self.setMinimumHeight(260)

    def set_schematic(self, schematic: dict) -> None:
        if schematic:
            self._schematic = schematic
        self.update()

    def tick(self) -> None:
        self._phase = (self._phase + 0.05) % 1.0
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(APP_PANEL))

        nodes = list((self._schematic or {}).get("nodes", []) or [])
        edges = list((self._schematic or {}).get("edges", []) or [])
        if not nodes:
            return

        width = float(self.width())
        height = float(self.height())
        margin_x = 52.0
        margin_y = 26.0
        draw_w = max(1.0, width - margin_x * 2.0)
        draw_h = max(1.0, height - margin_y * 2.0)

        points = {}
        for node in nodes:
            x = margin_x + float(node.get("x", 0.5)) * draw_w
            y = margin_y + float(node.get("y", 0.5)) * draw_h
            points[str(node.get("id", ""))] = QPointF(x, y)

        edge_pen = QPen(QColor(56, 189, 248, 150), 2.0)
        edge_pen.setStyle(Qt.DashLine)
        painter.setPen(edge_pen)
        for edge_idx, edge in enumerate(edges):
            src = points.get(str(edge.get("source", "")))
            dst = points.get(str(edge.get("target", "")))
            if src is None or dst is None:
                continue
            painter.drawLine(src, dst)
            t = (self._phase + edge_idx * 0.17) % 1.0
            excite_x = src.x() + (dst.x() - src.x()) * t
            excite_y = src.y() + (dst.y() - src.y()) * t
            painter.setBrush(QColor(34, 197, 94, 220))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(excite_x, excite_y), 4.0, 4.0)
            painter.setPen(edge_pen)

        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        for node in nodes:
            point = points.get(str(node.get("id", "")))
            if point is None:
                continue
            weight = max(0.0, min(1.0, float(node.get("weight", 0.5))))
            radius = 18.0 + weight * 18.0
            painter.setPen(QPen(QColor(56, 189, 248, 220), 2.0))
            painter.setBrush(QColor(15, 23, 38, 230))
            painter.drawEllipse(point, radius, radius)
            painter.setPen(QPen(QColor(45, 212, 191, int(120 + 100 * math.sin(self._phase * math.pi))), 1.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(point, radius + 8.0, radius + 8.0)

            tri = QPolygonF(
                [
                    QPointF(point.x() - radius * 0.22, point.y() - radius * 0.28),
                    QPointF(point.x() - radius * 0.22, point.y() + radius * 0.28),
                    QPointF(point.x() + radius * 0.34, point.y()),
                ]
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(230, 238, 247, 235))
            painter.drawPolygon(tri)

            painter.setPen(QColor(APP_TEXT))
            painter.drawText(
                int(point.x() - 64),
                int(point.y() + radius + 18.0),
                128,
                20,
                Qt.AlignCenter,
                str(node.get("label", node.get("id", "node"))),
            )


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class ControlCenterWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self.setWindowTitle("Quantum Miner")
        self.setMinimumSize(1180, 760)
        self.resize(1280, 720)

        font = self.font()
        font.setPointSize(9)
        self.setFont(font)

        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Top metric bar
        root_layout.addLayout(self._build_top_bar())

        # Middle: tabs + right operations column
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(6)
        root_layout.addLayout(mid_layout, stretch=1)

        # Left side: tabs content
        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_miner_tab(), "Miner")
        tabs.addTab(QWidget(), "Prediction Engine")
        tabs.addTab(QWidget(), "Murphy")
        tabs.addTab(QWidget(), "Watchdog")
        tabs.addTab(QWidget(), "VSD")
        self.research_tab_index = tabs.addTab(self._build_research_modeling_tab(), "Research & Modeling")
        self.tabs = tabs
        self._research_tick = 0
        mid_layout.addWidget(tabs, stretch=3)

        # Right side: operations panel
        mid_layout.addWidget(self._build_neuralis_panel(), stretch=1)

        # Bottom boot bar
        root_layout.addLayout(self._build_bottom_bar())

        # Timer for simulated updates
        self.chart_timer = QTimer(self)
        self.chart_timer.timeout.connect(self._tick_simulation)
        self.chart_timer.start(1000)

    # ------------------------------------------------------------------
    # Top status bar
    # ------------------------------------------------------------------
    def _build_top_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(12)

        for label, value in (
            ("Miner", "5843 MH/s"),
            ("Active Lanes", "3"),
            ("Network Share", "18.7 %"),
            ("VSD Snapshots", "325"),
            ("Session Throughput", "4.8 MH/s"),
        ):
            card, _ = make_metric_card(label, value)
            layout.addWidget(card)

        layout.addStretch(1)

        return layout

    # ------------------------------------------------------------------
    # Bottom boot bar
    # ------------------------------------------------------------------
    def _build_bottom_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(16)

        hint = make_small_label("Boot Sequence")
        layout.addWidget(hint)

        layout.addStretch(1)

        btn_miner = QPushButton("Start Miner")
        btn_pred = QPushButton("Start AutoTrader")
        btn_all = QPushButton("Start All")

        # Tooltip uses runtime submission rate telemetry; read-only, governor-safe.
        btn_miner.setToolTip(self._submission_rate_tooltip())

        btn_miner.clicked.connect(lambda: start_miner_button())
        btn_pred.clicked.connect(lambda: start_prediction_engine_button())
        btn_all.clicked.connect(self._start_all)

        for b in (btn_miner, btn_pred, btn_all):
            b.setFixedHeight(26)
        style_button(btn_miner)
        style_button(btn_pred)
        style_button(btn_all, accent=True)

        layout.addWidget(btn_miner)
        layout.addWidget(btn_pred)
        layout.addWidget(btn_all)

        return layout

    # ------------------------------------------------------------------
    # Miner tab layout
    # ------------------------------------------------------------------
    def _build_miner_tab(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(6)

        # Left column: Lane Allocation, Alert Feed, Historical charts
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        grid.addLayout(left_col, 0, 0)

        left_col.addWidget(self._build_lane_alloc_panel())
        left_col.addWidget(self._build_alert_feed_panel())
        left_col.addWidget(self._build_history_panel())

        # Center column: Live Predictions, VSD status
        center_col = QVBoxLayout()
        center_col.setSpacing(6)
        grid.addLayout(center_col, 0, 1)

        center_col.addWidget(self._build_live_predictions_panel())
        center_col.addWidget(self._build_vsd_panel())

        # Right side of tab currently unused (right column is Neuralis panel)
        # but we keep the grid 2 columns to match screenshot proportions.
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        return page

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def _build_lane_alloc_panel(self) -> QGroupBox:
        box = QGroupBox("Lane Allocation")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        self.lane_table = QTableWidget(6, 3)
        self.lane_table.setHorizontalHeaderLabels(["Lane", "Assigned Network", "Share %"])
        self.lane_table.verticalHeader().setVisible(False)
        self.lane_table.setShowGrid(True)
        self.lane_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.lane_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lane_table.setFixedHeight(180)
        self.lane_table.setAlternatingRowColors(True)
        self.lane_table.setFrameShape(QFrame.NoFrame)
        self.lane_table.horizontalHeader().setStretchLastSection(True)

        lanes = ["12.2", "12.5", "0.2", "4.2", "6.5", "7.3"]
        nets = ["ETH", "ETC", "RVN", "LTC", "BTC"]

        for row in range(6):
            self.lane_table.setItem(row, 0, QTableWidgetItem(lanes[row]))
            self.lane_table.setItem(row, 1, QTableWidgetItem(nets[row]))
            self.lane_table.setItem(row, 2, QTableWidgetItem("94 %"))

        self.lane_table.resizeColumnsToContents()
        layout.addWidget(self.lane_table)

        btn_row = QHBoxLayout()
        btn_pause = QPushButton("Pause")
        btn_test_sell = QPushButton("Run Test Sell")
        style_button(btn_pause, quiet=True)
        style_button(btn_test_sell)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_pause)
        btn_row.addWidget(btn_test_sell)
        layout.addLayout(btn_row)

        return box

    def _build_live_predictions_panel(self) -> QGroupBox:
        box = QGroupBox("Live Predictions")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        self.pred_chart = LivePredictionChart()
        layout.addWidget(self.pred_chart)

        btn_row = QHBoxLayout()
        btn_sim = QPushButton("Run Test Simulation")
        btn_export = QPushButton("Export Stats")
        style_button(btn_sim, accent=True)
        style_button(btn_export)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_sim)
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

        return box

    def _build_alert_feed_panel(self) -> QGroupBox:
        box = QGroupBox("Alert Feed")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        self.alert_list = QListWidget()
        self.alert_list.setFixedHeight(150)
        self.alert_list.setFrameShape(QFrame.NoFrame)

        # seed with a few example alerts
        self._push_alert("Alert: Lane 5 overload", level="HIGH")
        self._push_alert("Warning: Hash variance high", level="WARN")
        self._push_alert("Prediction confidence drift detected", level="INFO")

        layout.addWidget(self.alert_list)
        return box

    def _build_history_panel(self) -> QGroupBox:
        box = QGroupBox("Historical Charts")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        # Interactive time series viewer backed by VSD
        viewer = TimeSeriesViewer(VSD, symbols=["BTCUSDT", "ETHUSDT", "ETCUSDT", "LTCUSDT", "RVNUSDT"])
        layout.addWidget(viewer)

        return box

    def _build_vsd_panel(self) -> QGroupBox:
        box = QGroupBox("VSD Snapshots")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        self.vsd_list = QListWidget()
        self.vsd_list.setFixedHeight(150)
        self.vsd_list.setFrameShape(QFrame.NoFrame)

        # sample VSD records
        for i in range(3):
            item = QListWidgetItem(f"2024-04-0{i+1} 15:48:20 | snapshot_{i:04x}")
            self.vsd_list.addItem(item)

        layout.addWidget(self.vsd_list)

        btn_row = QHBoxLayout()
        btn_move = QPushButton("Move")
        btn_copy = QPushButton("Copy")
        btn_rename = QPushButton("Rename")
        btn_delete = QPushButton("Delete")
        for b in (btn_move, btn_copy, btn_rename, btn_delete):
            style_button(b, quiet=True)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        return box

    def _load_research_model(self) -> dict:
        fallback = build_fallback_research_model()
        summary = load_json_with_comment(RESEARCH_ROOT / "frequency_domain_run_summary.json")
        state = load_json_with_comment(RESEARCH_ROOT / "state.json")

        prototype = dict(summary.get("btc_miner_prototype", {}) or {})
        if not prototype:
            prototype = dict(state.get("btc_miner_prototype", {}) or {})
        if not prototype:
            prototype = dict(fallback)

        if "latest_summary" not in prototype:
            latest = dict(((state.get("runtime", {}) or {}).get("btc_miner_prototype", {})) or {})
            prototype["latest_summary"] = {
                "yield_count": int(latest.get("live_nonce_yield", 0)),
                "coherence_peak": float(latest.get("coherence_peak", 0.0)),
                "nesting_depth": int(latest.get("max_depth", 5)),
                "worker_count": int(latest.get("worker_count", 0)),
                "state_capacity": 1680700000,
                "effective_vector": dict(latest.get("effective_vector", {}) or {}),
                "path_equivalence_error": float(latest.get("path_equivalence_error", 0.0)),
                "basis_rotation_residual": float(latest.get("basis_rotation_residual", 0.0)),
            }
        latest_summary = dict(fallback.get("latest_summary", {}) or {})
        latest_summary.update(dict(prototype.get("latest_summary", {}) or {}))
        prototype["latest_summary"] = latest_summary

        for key in (
            "schematic",
            "pseudocode",
            "yield_sequence",
            "coherence_sequence",
            "vector_magnitude_sequence",
            "temporal_projection_sequence",
            "pulse_batches",
            "effective_vector",
            "manifold_diagnostics",
            "silicon_calibration",
        ):
            if key not in prototype:
                prototype[key] = fallback[key]

        if not prototype.get("effective_vector"):
            prototype["effective_vector"] = dict(prototype["latest_summary"].get("effective_vector", {}) or {})
        if not prototype.get("effective_vector"):
            prototype["effective_vector"] = dict(fallback.get("effective_vector", {}) or {})

        diagnostics = dict(fallback.get("manifold_diagnostics", {}) or {})
        diagnostics.update(dict(prototype.get("manifold_diagnostics", {}) or {}))
        if "path_equivalence_error" not in diagnostics:
            diagnostics["path_equivalence_error"] = float(prototype["latest_summary"].get("path_equivalence_error", 0.0))
        if "basis_rotation_residual" not in diagnostics:
            diagnostics["basis_rotation_residual"] = float(prototype["latest_summary"].get("basis_rotation_residual", 0.0))
        prototype["manifold_diagnostics"] = diagnostics
        calibration = dict(fallback.get("silicon_calibration", {}) or {})
        calibration.update(dict(prototype.get("silicon_calibration", {}) or {}))
        prototype["silicon_calibration"] = calibration
        return prototype

    def _apply_research_model(self, prototype: dict) -> None:
        latest = dict(prototype.get("latest_summary", {}) or {})
        pulses = list(prototype.get("pulse_batches", []) or [])
        latest_pulse = dict(pulses[-1] if pulses else {})
        yield_value = int(latest.get("yield_count", latest_pulse.get("yield_count", 0)))
        coherence_value = float(latest.get("coherence_peak", latest_pulse.get("coherence_peak", 0.0)))
        depth_value = int(latest.get("nesting_depth", latest_pulse.get("nesting_depth", 0)))
        worker_value = int(latest.get("worker_count", latest_pulse.get("worker_count", 0)))
        state_capacity = int(latest.get("state_capacity", latest_pulse.get("state_capacity", 0)))
        effective_vector = dict(prototype.get("effective_vector", {}) or {})
        if not effective_vector:
            effective_vector = dict(latest.get("effective_vector", {}) or {})
        if not effective_vector:
            effective_vector = dict(latest_pulse.get("effective_vector", {}) or {})
        calibration = dict(prototype.get("silicon_calibration", {}) or {})
        dominant_basin = dict(calibration.get("dominant_basin", {}) or {})
        diagnostics = dict(prototype.get("manifold_diagnostics", {}) or {})
        temporal_projection = float(
            latest.get(
                "temporal_projection",
                latest_pulse.get(
                    "temporal_projection",
                    effective_vector.get("t_eff", 0.0),
                ),
            )
        )
        path_equivalence_error = float(
            diagnostics.get(
                "path_equivalence_error",
                latest.get("path_equivalence_error", latest_pulse.get("path_equivalence_error", 0.0)),
            )
        )
        temporal_ordering_delta = float(
            diagnostics.get(
                "temporal_ordering_delta",
                latest_pulse.get("temporal_ordering_delta", 0.0),
            )
        )
        basis_rotation_residual = float(
            diagnostics.get(
                "basis_rotation_residual",
                latest.get("basis_rotation_residual", latest_pulse.get("basis_rotation_residual", 0.0)),
            )
        )
        field_pressure = float(
            latest.get(
                "field_pressure",
                latest_pulse.get("field_pressure", calibration.get("field_pressure", 0.0)),
            )
        )
        larger_field_exposure = float(
            latest.get(
                "larger_field_exposure",
                latest_pulse.get("larger_field_exposure", calibration.get("larger_field_exposure", 0.0)),
            )
        )

        self.research_chart.load_sequences(
            [float(v) for v in list(prototype.get("yield_sequence", []) or [])],
            [float(v) for v in list(prototype.get("coherence_sequence", []) or [])],
        )
        self.research_graph.set_schematic(dict(prototype.get("schematic", {}) or {}))

        self.research_yield_label.setText(f"{yield_value} / pulse")
        self.research_coherence_label.setText(f"{coherence_value:.3f} coherence")
        self.research_depth_label.setText(f"depth {depth_value}")
        self.research_workers_label.setText(f"{worker_value} workers")
        self.research_status.setText(
            f"state space {state_capacity:,} | |R_eff| {float(effective_vector.get('spatial_magnitude', 0.0)):.3f} | T_eff {temporal_projection:.3f} | field {field_pressure:.3f}"
        )
        self.research_vector_label.setText(
            "R_eff "
            f"[{float(effective_vector.get('x', 0.0)):.3f}, "
            f"{float(effective_vector.get('y', 0.0)):.3f}, "
            f"{float(effective_vector.get('z', 0.0)):.3f}]"
        )
        self.research_temporal_label.setText(
            f"T_eff {float(effective_vector.get('t_eff', temporal_projection)):.3f} | bias {float(effective_vector.get('coherence_bias', 0.0)):.3f}"
        )
        self.research_calibration_label.setText(
            f"lattice {field_pressure:.3f} | exposure {larger_field_exposure:.3f} | basin {str(dominant_basin.get('basin_id', latest.get('dominant_basin', 'unknown_basin')))}"
        )
        self.research_diag_label.setText(
            f"path {path_equivalence_error:.3f} | order {temporal_ordering_delta:.3f} | basis {basis_rotation_residual:.3f}"
        )

        pseudo_lines = [str(line) for line in list(prototype.get("pseudocode", []) or [])]
        self.research_pseudocode.setPlainText("\n".join(pseudo_lines))

        nonce_preview = ", ".join(list(latest_pulse.get("nonces", []) or [])[:12])
        self.research_nonce_preview.setText(nonce_preview or "Nonce preview unavailable")

        workers = list(latest_pulse.get("worker_batches", []) or [])
        self.research_worker_table.setRowCount(len(workers))
        for row, worker in enumerate(workers):
            self.research_worker_table.setItem(row, 0, QTableWidgetItem(str(worker.get("worker_id", ""))))
            self.research_worker_table.setItem(row, 1, QTableWidgetItem(str(int(worker.get("batch_size", 0)))))
            self.research_worker_table.setItem(row, 2, QTableWidgetItem(str(int(worker.get("queue_depth", 0)))))
            preview = ", ".join(list(worker.get("submit_preview", []) or [])[:4])
            self.research_worker_table.setItem(row, 3, QTableWidgetItem(preview))
        self.research_worker_table.resizeColumnsToContents()

    def _build_research_modeling_tab(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(6)

        schematic_box = QGroupBox("Pulse Graph")
        schematic_layout = QVBoxLayout(schematic_box)
        schematic_layout.setContentsMargins(6, 6, 6, 6)
        self.research_graph = ResearchPulseGraph()
        self.research_status = QLabel("state space 0")
        schematic_layout.addWidget(self.research_graph)
        schematic_layout.addWidget(self.research_status)
        vector_row = QHBoxLayout()
        self.research_vector_label = QLabel("R_eff [0.000, 0.000, 0.000]")
        self.research_temporal_label = QLabel("T_eff 0.000 | bias 0.000")
        self.research_calibration_label = QLabel("lattice 0.000 | exposure 0.000 | basin unknown")
        self.research_diag_label = QLabel("path 0.000 | order 0.000 | basis 0.000")
        for label in (
            self.research_vector_label,
            self.research_temporal_label,
            self.research_calibration_label,
            self.research_diag_label,
        ):
            vector_row.addWidget(label)
        vector_row.addStretch(1)
        schematic_layout.addLayout(vector_row)
        grid.addWidget(schematic_box, 0, 0)

        yield_box = QGroupBox("Live Nonce Yield")
        yield_layout = QVBoxLayout(yield_box)
        yield_layout.setContentsMargins(6, 6, 6, 6)
        self.research_chart = LiveNonceYieldChart()
        yield_layout.addWidget(self.research_chart)

        metrics_row = QHBoxLayout()
        self.research_yield_label = QLabel("0 / pulse")
        self.research_coherence_label = QLabel("0.000 coherence")
        self.research_depth_label = QLabel("depth 0")
        self.research_workers_label = QLabel("0 workers")
        for label in (
            self.research_yield_label,
            self.research_coherence_label,
            self.research_depth_label,
            self.research_workers_label,
        ):
            metrics_row.addWidget(label)
        metrics_row.addStretch(1)
        yield_layout.addLayout(metrics_row)
        grid.addWidget(yield_box, 0, 1)

        pseudo_box = QGroupBox("Pulse Injection Loop")
        pseudo_layout = QVBoxLayout(pseudo_box)
        pseudo_layout.setContentsMargins(6, 6, 6, 6)
        self.research_pseudocode = QTextEdit()
        self.research_pseudocode.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self.research_pseudocode.setFont(mono)
        pseudo_layout.addWidget(self.research_pseudocode)
        grid.addWidget(pseudo_box, 1, 0)

        queue_box = QGroupBox("Substrate Queue")
        queue_layout = QVBoxLayout(queue_box)
        queue_layout.setContentsMargins(6, 6, 6, 6)
        self.research_nonce_preview = QLabel("Nonce preview unavailable")
        self.research_nonce_preview.setWordWrap(True)
        queue_layout.addWidget(self.research_nonce_preview)

        self.research_worker_table = QTableWidget(0, 4)
        self.research_worker_table.setHorizontalHeaderLabels(["Worker", "Batch", "Queue", "Preview"])
        self.research_worker_table.verticalHeader().setVisible(False)
        self.research_worker_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.research_worker_table.setSelectionMode(QAbstractItemView.NoSelection)
        queue_layout.addWidget(self.research_worker_table)
        grid.addWidget(queue_box, 1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._apply_research_model(self._load_research_model())
        return page

    # ------------------------------------------------------------------
    # Operations right panel
    # ------------------------------------------------------------------
    def _build_neuralis_panel(self) -> QGroupBox:
        box = QGroupBox("Operations Console")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        # Runtime event history
        lbl_chat = make_header_label("Runtime Events")
        layout.addWidget(lbl_chat)

        self.chat_list = QListWidget()
        self.chat_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_list.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self.chat_list)

        # Seed with a few entries
        self._append_chat("Control surface initialized.")
        self._append_chat("Miner telemetry ready.")
        self._append_chat("Prediction runtime available.")
        self._append_chat("Backend controls online.")

        # Input row
        input_row = QHBoxLayout()
        self.command_edit = QLineEdit()
        self.command_edit.setPlaceholderText("Log an operator note")
        self.command_edit.setClearButtonEnabled(True)
        btn_send = QPushButton("Send")
        btn_send.setFixedWidth(60)
        style_button(btn_send, accent=True)
        btn_send.clicked.connect(self._send_command)
        input_row.addWidget(self.command_edit)
        input_row.addWidget(btn_send)
        layout.addLayout(input_row)

        return box

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _push_alert(self, text: str, level: str = "INFO") -> None:
        prefix = ""
        if level == "HIGH":
            prefix = "[HIGH] "
        elif level == "WARN":
            prefix = "[WARN] "
        item = QListWidgetItem(prefix + text)
        self.alert_list.addItem(item)
        self.alert_list.scrollToBottom()

    def _append_chat(self, text: str) -> None:
        item = QListWidgetItem(text)
        self.chat_list.addItem(item)
        self.chat_list.scrollToBottom()

    def _send_command(self) -> None:
        cmd = self.command_edit.text().strip()
        if not cmd:
            return
        self._append_chat("> " + cmd)
        self.command_edit.clear()
        # Hook direct control dispatch here if needed.

    def _submission_rate_tooltip(self) -> str:
        """Read submission-rate telemetry from VSD for display.

        This does not alter governor behavior; it mirrors
        miner/runtime/submission_rate or submitter telemetry
        for user-facing observability only.
        """
        try:
            sr = dict(VSD.get("miner/runtime/submission_rate_snapshot", {}))
            if not sr:
                sr = dict(VSD.get("miner/runtime/submission_rate", {}))
            allowed = float(sr.get("allowed_rate_per_second", 0.0))
            tick = float(sr.get("tick_duration", 0.0))
            max_tick = float(sr.get("max_submissions_per_tick", 0.0))
            parts = []
            if allowed > 0:
                parts.append("allowed %.1f / sec" % allowed)
            if tick > 0:
                parts.append("tick %.3f s" % tick)
            if max_tick > 0:
                parts.append("max %.1f / tick" % max_tick)
            if not parts:
                return "Submission rate telemetry unavailable"
            return "Submission governor: " + ", ".join(parts)
        except Exception:
            return "Submission rate telemetry unavailable"

    def _start_all(self) -> None:
        # Start prediction engine and Neuralis first, then miner.
        try:
            start_prediction_engine_button()
        except Exception:
            pass
        try:
            start_neuralis_button()
        except Exception:
            pass
        try:
            start_miner_button()
        except Exception:
            pass

    def _tick_simulation(self) -> None:
        # drive chart and fake lane numbers
        self.pred_chart.tick()
        self._apply_research_model(self._load_research_model())
        self.research_chart.tick()
        self.research_graph.tick()
        self._research_tick += 1
        spinner = ["|", "/", "-", "\\"][self._research_tick % 4]
        self.tabs.setTabText(
            self.research_tab_index,
            f"{spinner} Research & Modeling {self.research_yield_label.text()}",
        )

        # randomly update lane shares to simulate activity
        for row in range(self.lane_table.rowCount()):
            val_item = self.lane_table.item(row, 2)
            try:
                base = 60.0 + 40.0 * random.random()
                val_item.setText(f"{base:.1f} %")
            except Exception:
                pass

        # occasionally push alerts
        if random.random() < 0.25:
            self._push_alert("Auto-tuned lane allocation.", level="INFO")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)
    apply_dark_palette(app)

     # Set application icon (title bar, taskbar, alt-tab).
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "NeuralisIcon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    win = ControlCenterWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
