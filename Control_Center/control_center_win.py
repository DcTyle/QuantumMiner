from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import sys
import threading
import time

if os.name == "nt":
	os.environ.setdefault("QSG_RHI_BACKEND", "vulkan")
	os.environ.setdefault("QT_OPENGL", "desktop")
	os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
	os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
	QAbstractItemView,
	QApplication,
	QComboBox,
	QDialog,
	QDialogButtonBox,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QFrame,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QListWidget,
	QListWidgetItem,
	QMessageBox,
	QPushButton,
	QStackedWidget,
	QSpinBox,
	QSplitter,
	QTableWidget,
	QTableWidgetItem,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)

import Control_Center.control_center_gui as gui
from bios.main_runtime import pause_miner, run as bios_run, shutdown as bios_shutdown
from config.manager import ConfigManager
from prediction_engine.crypto_com_api import CryptoComAPI
from prediction_engine.trade_executor import TradeExecutor
from scripts.worker_session_probe import _collect_rows as _collect_worker_probe_rows


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "NeuralisIcon.png")
MINER_CONFIG_PATH = os.path.join(ROOT, "miner", "miner_runtime_config.json")


class UserSettingsDialog(QDialog):
	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.setObjectName("settingsDialog")
		self.setWindowTitle("User Settings")
		self.setModal(True)
		self.resize(560, 320)

		layout = QVBoxLayout(self)
		form = QFormLayout()
		layout.addLayout(form)

		self.api_key_edit = QLineEdit(str(ConfigManager.get("prediction.api_key", "") or ""))
		self.api_secret_edit = QLineEdit(str(ConfigManager.get("prediction.api_secret", "") or ""))
		self.api_secret_edit.setEchoMode(QLineEdit.Password)
		self.base_quote_edit = QLineEdit(str(ConfigManager.get("prediction.base_quote", "USDT") or "USDT"))

		self.max_assets_spin = QSpinBox()
		self.max_assets_spin.setRange(1, 100)
		self.max_assets_spin.setValue(int(ConfigManager.get("prediction.max_assets", 12) or 12))

		self.max_clusters_spin = QSpinBox()
		self.max_clusters_spin.setRange(1, 32)
		self.max_clusters_spin.setValue(int(ConfigManager.get("prediction.max_clusters", 4) or 4))

		self.lanes_per_cluster_spin = QSpinBox()
		self.lanes_per_cluster_spin.setRange(1, 64)
		self.lanes_per_cluster_spin.setValue(int(ConfigManager.get("prediction.lanes_per_cluster", 6) or 6))

		self.min_confidence_spin = QDoubleSpinBox()
		self.min_confidence_spin.setRange(0.0, 1.0)
		self.min_confidence_spin.setDecimals(4)
		self.min_confidence_spin.setSingleStep(0.01)
		self.min_confidence_spin.setValue(float(ConfigManager.get("prediction.min_confidence", 0.99) or 0.99))

		self.candle_timeframe_combo = QComboBox()
		self.candle_timeframe_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
		timeframe = str(ConfigManager.get("prediction.candle_timeframe", "1h") or "1h")
		idx = self.candle_timeframe_combo.findText(timeframe)
		self.candle_timeframe_combo.setCurrentIndex(max(0, idx))

		self.candle_limit_spin = QSpinBox()
		self.candle_limit_spin.setRange(50, 5000)
		self.candle_limit_spin.setValue(int(ConfigManager.get("prediction.candle_limit", 400) or 400))

		self.render_backend_edit = QLineEdit(str(ConfigManager.get("app.render_backend", "vulkan") or "vulkan"))
		self.render_backend_edit.setReadOnly(True)

		form.addRow("Crypto.com API Key", self.api_key_edit)
		form.addRow("Crypto.com API Secret", self.api_secret_edit)
		form.addRow("Prediction Base Quote", self.base_quote_edit)
		form.addRow("Prediction Max Assets", self.max_assets_spin)
		form.addRow("Prediction Max Clusters", self.max_clusters_spin)
		form.addRow("Lanes Per Cluster", self.lanes_per_cluster_spin)
		form.addRow("Min Confidence", self.min_confidence_spin)
		form.addRow("Candle Timeframe", self.candle_timeframe_combo)
		form.addRow("Candle Limit", self.candle_limit_spin)
		form.addRow("Render Backend", self.render_backend_edit)

		self.settings_path_label = QLabel(ConfigManager.path())
		self.settings_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
		form.addRow("Settings File", self.settings_path_label)

		buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=self)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		gui.style_button(buttons.button(QDialogButtonBox.Save), accent=True)
		gui.style_button(buttons.button(QDialogButtonBox.Cancel), quiet=True)
		layout.addWidget(buttons)

	def values(self) -> Dict[str, Any]:
		return {
			"prediction.api_key": self.api_key_edit.text().strip(),
			"prediction.api_secret": self.api_secret_edit.text().strip(),
			"prediction.base_quote": self.base_quote_edit.text().strip().upper() or "USDT",
			"prediction.max_assets": int(self.max_assets_spin.value()),
			"prediction.max_clusters": int(self.max_clusters_spin.value()),
			"prediction.lanes_per_cluster": int(self.lanes_per_cluster_spin.value()),
			"prediction.min_confidence": float(self.min_confidence_spin.value()),
			"prediction.candle_timeframe": self.candle_timeframe_combo.currentText(),
			"prediction.candle_limit": int(self.candle_limit_spin.value()),
			"app.render_backend": "vulkan",
		}


class Win64ControlCenterWindow(gui.ControlCenterWindow):
	def __init__(self, runtime_context: Dict[str, Any]) -> None:
		self.runtime_context = dict(runtime_context or {})
		self.vsd = self.runtime_context.get("vsd") or dict(self.runtime_context.get("env", {}) or {}).get("vsd")
		self.bus = self.runtime_context.get("bus")
		self._prediction_busy = False
		self.metric_labels: Dict[str, QLabel] = {}
		self._section_titles: List[str] = [
			"Operations Overview",
			"Miner Telemetry",
			"Prediction Runtime",
			"Research Lab",
			"Backend Controls",
		]
		gui.VSD = self.vsd
		super().__init__()
		self.setWindowTitle("Quantum Miner")
		if os.path.isfile(ICON_PATH):
			self.setWindowIcon(QIcon(ICON_PATH))
		self._install_operations_shell()
		self._install_menus()
		self.statusBar().showMessage("Win64 desktop launcher ready (render backend: %s)" % str(ConfigManager.get("app.render_backend", "vulkan") or "vulkan"))
		self._refresh_live_telemetry()

	def _build_top_bar(self) -> QHBoxLayout:
		layout = QHBoxLayout()
		layout.setSpacing(12)
		self.metric_labels = {}

		for key, label, value in (
			("miner_hashrate", "Miner Hashrate", "0.00 H/s"),
			("active_lanes", "Active Lanes", "0"),
			("network_share", "Observed Share", "0.00 %"),
			("worker_sessions", "Worker Sessions", "0 / 0"),
			("prediction_orders", "Prediction Orders", "0"),
		):
			card, val = gui.make_metric_card(label, value)
			self.metric_labels[key] = val
			layout.addWidget(card)
		layout.addStretch(1)
		return layout

	def _install_operations_shell(self) -> None:
		old = self.centralWidget()
		if old is not None:
			old.setParent(None)

		root = QWidget()
		root.setObjectName("appRoot")
		self.setCentralWidget(root)

		layout = QVBoxLayout(root)
		layout.setContentsMargins(12, 12, 12, 12)
		layout.setSpacing(12)
		layout.addLayout(self._build_top_bar())

		splitter = QSplitter(Qt.Horizontal)
		splitter.setChildrenCollapsible(False)
		layout.addWidget(splitter, stretch=1)

		left_shell = self._build_navigation_shell()
		splitter.addWidget(left_shell)

		self.section_stack = QStackedWidget()
		splitter.addWidget(self.section_stack)

		right_shell = self._build_right_dock()
		splitter.addWidget(right_shell)

		splitter.setStretchFactor(0, 0)
		splitter.setStretchFactor(1, 1)
		splitter.setStretchFactor(2, 0)
		splitter.setSizes([240, 860, 340])

		layout.addLayout(self._build_bottom_bar())

		self.section_stack.addWidget(self._build_operations_overview_page())
		self.section_stack.addWidget(self._build_miner_page())
		self.section_stack.addWidget(self._build_prediction_runtime_tab())
		self.section_stack.addWidget(self._build_research_modeling_tab())
		self.section_stack.addWidget(self._build_backend_controls_page())

		self.section_nav.setCurrentRow(0)

	def _build_navigation_shell(self) -> QGroupBox:
		box = QGroupBox("Control Rail")
		layout = QVBoxLayout(box)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(10)

		brand = QLabel("QUANTUM MINER")
		brand.setObjectName("metricValue")
		brand.setStyleSheet("font-size: 20pt;")
		strap = QLabel("Substrate Operations Console")
		strap.setObjectName("metricCaption")
		layout.addWidget(brand)
		layout.addWidget(strap)

		self.section_nav = QListWidget()
		self.section_nav.setFrameShape(QFrame.NoFrame)
		self.section_nav.setSpacing(6)
		for title in self._section_titles:
			self.section_nav.addItem(QListWidgetItem(title))
		self.section_nav.currentRowChanged.connect(self._switch_section)
		layout.addWidget(self.section_nav, stretch=1)

		quick_box = QGroupBox("Quick State")
		quick_layout = QVBoxLayout(quick_box)
		self.quick_state_list = QListWidget()
		self.quick_state_list.setFrameShape(QFrame.NoFrame)
		quick_layout.addWidget(self.quick_state_list)
		layout.addWidget(quick_box)

		return box

	def _build_right_dock(self) -> QWidget:
		dock = QWidget()
		layout = QVBoxLayout(dock)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(12)

		layout.addWidget(self._build_operator_dock())
		layout.addWidget(self._build_worker_status_panel(), stretch=1)
		return dock

	def _build_operator_dock(self) -> QGroupBox:
		box = QGroupBox("Command Console")
		layout = QVBoxLayout(box)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(10)

		self.command_summary = QLabel("Live miner telemetry, prediction controls, and backend access from one surface.")
		self.command_summary.setWordWrap(True)
		layout.addWidget(self.command_summary)

		actions_box = QGroupBox("Quick Actions")
		actions_layout = QVBoxLayout(actions_box)
		actions_layout.setContentsMargins(10, 10, 10, 10)
		actions_layout.setSpacing(8)

		row1 = QHBoxLayout()
		quick_resume = QPushButton("Resume Miner")
		quick_pause = QPushButton("Pause Miner")
		gui.style_button(quick_resume, accent=True)
		gui.style_button(quick_pause, quiet=True)
		quick_resume.clicked.connect(self._resume_miner)
		quick_pause.clicked.connect(self._pause_miner_from_ui)
		row1.addWidget(quick_resume)
		row1.addWidget(quick_pause)
		actions_layout.addLayout(row1)

		row2 = QHBoxLayout()
		quick_prediction = QPushButton("Run Prediction")
		quick_settings = QPushButton("Settings")
		gui.style_button(quick_prediction, accent=True)
		gui.style_button(quick_settings, quiet=True)
		quick_prediction.clicked.connect(self._run_prediction_cycle)
		quick_settings.clicked.connect(self._open_user_settings)
		row2.addWidget(quick_prediction)
		row2.addWidget(quick_settings)
		actions_layout.addLayout(row2)

		row3 = QHBoxLayout()
		quick_refresh = QPushButton("Refresh")
		quick_probe = QPushButton("Worker Probe")
		gui.style_button(quick_refresh, quiet=True)
		gui.style_button(quick_probe)
		quick_refresh.clicked.connect(self._refresh_live_telemetry)
		quick_probe.clicked.connect(self._show_worker_probe)
		row3.addWidget(quick_refresh)
		row3.addWidget(quick_probe)
		actions_layout.addLayout(row3)
		layout.addWidget(actions_box)

		snapshot_box = QGroupBox("Runtime Snapshot")
		snapshot_layout = QVBoxLayout(snapshot_box)
		snapshot_layout.setContentsMargins(10, 10, 10, 10)
		snapshot_layout.setSpacing(8)
		self.command_snapshot_list = QListWidget()
		self.command_snapshot_list.setFrameShape(QFrame.NoFrame)
		self.command_snapshot_list.setAlternatingRowColors(True)
		snapshot_layout.addWidget(self.command_snapshot_list)
		layout.addWidget(snapshot_box)

		events_box = QGroupBox("Event Feed")
		events_layout = QVBoxLayout(events_box)
		events_layout.setContentsMargins(10, 10, 10, 10)
		events_layout.setSpacing(8)
		self.command_event_list = QListWidget()
		self.command_event_list.setFrameShape(QFrame.NoFrame)
		events_layout.addWidget(self.command_event_list)
		layout.addWidget(events_box, stretch=1)
		self._log_console_event("Command console ready")
		return box

	def _build_worker_status_panel(self) -> QGroupBox:
		box = QGroupBox("Worker Sessions")
		layout = QVBoxLayout(box)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(8)

		self.worker_status_list = QListWidget()
		self.worker_status_list.setFrameShape(QFrame.NoFrame)
		self.worker_status_list.setAlternatingRowColors(True)
		layout.addWidget(self.worker_status_list)

		btn_row = QHBoxLayout()
		refresh_btn = QPushButton("Refresh")
		probe_btn = QPushButton("Probe")
		gui.style_button(refresh_btn, quiet=True)
		gui.style_button(probe_btn, accent=True)
		refresh_btn.clicked.connect(self._refresh_live_telemetry)
		probe_btn.clicked.connect(self._show_worker_probe)
		btn_row.addWidget(refresh_btn)
		btn_row.addWidget(probe_btn)
		layout.addLayout(btn_row)
		return box

	def _build_operations_overview_page(self) -> QWidget:
		page = QWidget()
		grid = QVBoxLayout(page)
		grid.setSpacing(12)

		top = QHBoxLayout()
		top.setSpacing(12)
		top.addWidget(self._build_overview_lane_panel(), stretch=3)
		top.addWidget(self._build_live_predictions_panel(), stretch=4)
		grid.addLayout(top, stretch=3)

		bottom = QHBoxLayout()
		bottom.setSpacing(12)
		bottom.addWidget(self._build_overview_alert_panel(), stretch=2)
		bottom.addWidget(self._build_vsd_panel(), stretch=2)
		grid.addLayout(bottom, stretch=2)
		return page

	def _build_overview_lane_panel(self) -> QGroupBox:
		box = QGroupBox("Lane Overview")
		layout = QVBoxLayout(box)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(8)
		self.ops_lane_list = QListWidget()
		self.ops_lane_list.setFrameShape(QFrame.NoFrame)
		self.ops_lane_list.setAlternatingRowColors(True)
		layout.addWidget(self.ops_lane_list)
		return box

	def _build_overview_alert_panel(self) -> QGroupBox:
		box = QGroupBox("System Alerts")
		layout = QVBoxLayout(box)
		layout.setContentsMargins(10, 10, 10, 10)
		layout.setSpacing(8)
		self.ops_alert_list = QListWidget()
		self.ops_alert_list.setFrameShape(QFrame.NoFrame)
		layout.addWidget(self.ops_alert_list)
		return box

	def _build_miner_page(self) -> QWidget:
		page = QWidget()
		grid = QHBoxLayout(page)
		grid.setSpacing(12)
		left = QVBoxLayout()
		left.setSpacing(12)
		left.addWidget(self._build_lane_alloc_panel(), stretch=2)
		left.addWidget(self._build_alert_feed_panel(), stretch=1)
		grid.addLayout(left, stretch=3)
		grid.addWidget(self._build_history_panel(), stretch=4)
		return page

	def _build_backend_controls_page(self) -> QWidget:
		page = QWidget()
		layout = QVBoxLayout(page)
		layout.setSpacing(12)

		files_box = QGroupBox("Backend Files")
		files_layout = QVBoxLayout(files_box)
		files_layout.setContentsMargins(10, 10, 10, 10)
		for label, path, accent in (
			("Open Miner Runtime Config", MINER_CONFIG_PATH, False),
			("Open User Settings", ConfigManager.path(), False),
			("Open Build Spec", os.path.join(ROOT, "quantum_application.spec"), False),
			("Open Worker Probe Script", os.path.join(ROOT, "scripts", "worker_session_probe.py"), True),
		):
			button = QPushButton(label)
			gui.style_button(button, accent=accent, quiet=not accent)
			button.clicked.connect(lambda _checked=False, p=path: self._open_path(p))
			files_layout.addWidget(button)
		layout.addWidget(files_box)

		status_box = QGroupBox("Runtime Notes")
		status_layout = QVBoxLayout(status_box)
		status_layout.setContentsMargins(10, 10, 10, 10)
		self.backend_notes = QTextEdit()
		self.backend_notes.setReadOnly(True)
		self.backend_notes.setFrameShape(QFrame.NoFrame)
		status_layout.addWidget(self.backend_notes)
		layout.addWidget(status_box, stretch=1)
		return page

	def _switch_section(self, index: int) -> None:
		if not hasattr(self, "section_stack"):
			return
		if index < 0 or index >= self.section_stack.count():
			return
		self.section_stack.setCurrentIndex(index)
		self._log_console_event("Switched section: %s" % self._section_titles[index])
		try:
			self.statusBar().showMessage("Section: %s" % self._section_titles[index], 2500)
		except Exception:
			pass

	def _log_console_event(self, text: str) -> None:
		if not hasattr(self, "command_event_list"):
			return
		self.command_event_list.addItem(QListWidgetItem(str(text)))
		while self.command_event_list.count() > 80:
			self.command_event_list.takeItem(0)
		self.command_event_list.scrollToBottom()

	def _push_alert(self, text: str, level: str = "INFO") -> None:
		try:
			super()._push_alert(text, level)
		except Exception:
			pass
		self._log_console_event("%s %s" % (level, str(text)))
		if not hasattr(self, "ops_alert_list"):
			return
		prefix = ""
		if level == "HIGH":
			prefix = "[HIGH] "
		elif level == "WARN":
			prefix = "[WARN] "
		item = QListWidgetItem(prefix + str(text))
		self.ops_alert_list.addItem(item)
		while self.ops_alert_list.count() > 60:
			self.ops_alert_list.takeItem(0)
		self.ops_alert_list.scrollToBottom()

	def _build_lane_alloc_panel(self) -> QGroupBox:
		box = QGroupBox("Lane Allocation")
		layout = QVBoxLayout(box)
		layout.setContentsMargins(6, 6, 6, 6)

		self.lane_table = QTableWidget(0, 3)
		self.lane_table.setHorizontalHeaderLabels(["Lane", "Assigned Network", "Telemetry"])
		self.lane_table.verticalHeader().setVisible(False)
		self.lane_table.setShowGrid(True)
		self.lane_table.setSelectionMode(QAbstractItemView.NoSelection)
		self.lane_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
		self.lane_table.setFixedHeight(220)
		self.lane_table.setAlternatingRowColors(True)
		self.lane_table.setFrameShape(QFrame.NoFrame)
		self.lane_table.horizontalHeader().setStretchLastSection(True)
		layout.addWidget(self.lane_table)

		btn_row = QHBoxLayout()
		self.btn_pause_miner = QPushButton("Pause Miner")
		self.btn_resume_miner = QPushButton("Resume Miner")
		self.btn_probe_workers = QPushButton("Worker Probe")
		gui.style_button(self.btn_pause_miner, quiet=True)
		gui.style_button(self.btn_resume_miner)
		gui.style_button(self.btn_probe_workers, accent=True)
		self.btn_pause_miner.clicked.connect(self._pause_miner_from_ui)
		self.btn_resume_miner.clicked.connect(self._resume_miner)
		self.btn_probe_workers.clicked.connect(self._show_worker_probe)
		btn_row.addStretch(1)
		btn_row.addWidget(self.btn_pause_miner)
		btn_row.addWidget(self.btn_resume_miner)
		btn_row.addWidget(self.btn_probe_workers)
		layout.addLayout(btn_row)
		return box

	def _build_bottom_bar(self) -> QHBoxLayout:
		layout = QHBoxLayout()
		layout.setSpacing(16)
		hint = gui.make_small_label("Desktop Controls")
		layout.addWidget(hint)
		layout.addStretch(1)

		self.btn_start_miner = QPushButton("Start Miner")
		self.btn_run_prediction = QPushButton("Run Prediction Cycle")
		self.btn_open_settings = QPushButton("User Settings")
		self.btn_start_all = QPushButton("Start All")

		self.btn_start_miner.clicked.connect(self._resume_miner)
		self.btn_run_prediction.clicked.connect(self._run_prediction_cycle)
		self.btn_open_settings.clicked.connect(self._open_user_settings)
		self.btn_start_all.clicked.connect(self._start_all)
		gui.style_button(self.btn_start_miner)
		gui.style_button(self.btn_run_prediction, accent=True)
		gui.style_button(self.btn_open_settings, quiet=True)
		gui.style_button(self.btn_start_all, accent=True)

		for button in (self.btn_start_miner, self.btn_run_prediction, self.btn_open_settings, self.btn_start_all):
			button.setFixedHeight(26)
			layout.addWidget(button)
		return layout

	def _replace_prediction_tab(self) -> None:
		try:
			self.tabs.removeTab(1)
		except Exception:
			pass
		self.tabs.insertTab(1, self._build_prediction_runtime_tab(), "Prediction Engine")

	def _build_prediction_runtime_tab(self) -> QWidget:
		page = QWidget()
		root = QVBoxLayout(page)
		root.setSpacing(6)

		summary_box = QGroupBox("Prediction Runtime")
		summary_layout = QVBoxLayout(summary_box)
		self.pred_boot_status = QLabel("API status: unknown")
		self.pred_config_status = QLabel("Base quote: USDT | confidence: 0.99 | timeframe: 1h/400")
		self.pred_last_report = QLabel("Last report: unavailable")
		for label in (self.pred_boot_status, self.pred_config_status, self.pred_last_report):
			summary_layout.addWidget(label)
		root.addWidget(summary_box)

		mid = QHBoxLayout()
		root.addLayout(mid, stretch=1)

		signals_box = QGroupBox("Latest Signals")
		signals_layout = QVBoxLayout(signals_box)
		self.pred_signal_list = QListWidget()
		self.pred_signal_list.setAlternatingRowColors(True)
		self.pred_signal_list.setFrameShape(QFrame.NoFrame)
		signals_layout.addWidget(self.pred_signal_list)
		mid.addWidget(signals_box, stretch=2)

		orders_box = QGroupBox("Recent Orders")
		orders_layout = QVBoxLayout(orders_box)
		self.pred_orders_log = QTextEdit()
		self.pred_orders_log.setReadOnly(True)
		self.pred_orders_log.setFrameShape(QFrame.NoFrame)
		orders_layout.addWidget(self.pred_orders_log)
		mid.addWidget(orders_box, stretch=2)

		btn_row = QHBoxLayout()
		btn_row.addStretch(1)
		btn_prediction = QPushButton("Run Prediction Cycle")
		btn_settings = QPushButton("Edit User Settings")
		gui.style_button(btn_prediction, accent=True)
		gui.style_button(btn_settings, quiet=True)
		btn_prediction.clicked.connect(self._run_prediction_cycle)
		btn_settings.clicked.connect(self._open_user_settings)
		btn_row.addWidget(btn_prediction)
		btn_row.addWidget(btn_settings)
		root.addLayout(btn_row)

		return page

	def _install_menus(self) -> None:
		menubar = self.menuBar()

		file_menu = menubar.addMenu("File")
		file_menu.addAction(self._make_action("Open Miner Config", lambda: self._open_path(MINER_CONFIG_PATH)))
		file_menu.addAction(self._make_action("Open User Settings File", lambda: self._open_path(ConfigManager.path())))
		file_menu.addSeparator()
		file_menu.addAction(self._make_action("Exit", self.close))

		edit_menu = menubar.addMenu("Edit")
		edit_menu.addAction(self._make_action("User Settings", self._open_user_settings))

		view_menu = menubar.addMenu("View")
		view_menu.addAction(self._make_action("Refresh Telemetry", self._refresh_live_telemetry))
		view_menu.addAction(self._make_action("Show Worker Probe", self._show_worker_probe))

		controls_menu = menubar.addMenu("Controls")
		controls_menu.addAction(self._make_action("Resume Miner", self._resume_miner))
		controls_menu.addAction(self._make_action("Pause Miner", self._pause_miner_from_ui))
		controls_menu.addAction(self._make_action("Run Prediction Cycle", self._run_prediction_cycle))
		controls_menu.addAction(self._make_action("Start All", self._start_all))

		backend_menu = menubar.addMenu("Backend")
		backend_menu.addAction(self._make_action("Open Build Spec", lambda: self._open_path(os.path.join(ROOT, "quantum_application.spec"))))
		backend_menu.addAction(self._make_action("Open Worker Probe Script", lambda: self._open_path(os.path.join(ROOT, "scripts", "worker_session_probe.py"))))

	def _make_action(self, text: str, callback) -> QAction:
		action = QAction(text, self)
		action.triggered.connect(callback)
		return action

	def _open_path(self, path: str) -> None:
		target = os.path.abspath(path)
		if not os.path.exists(target):
			QMessageBox.warning(self, "Missing File", target)
			return
		try:
			if os.name == "nt":
				os.startfile(target)  # type: ignore[attr-defined]
			else:
				QFileDialog.getOpenFileName(self, "Open File", target)
		except Exception as exc:
			QMessageBox.warning(self, "Open Failed", str(exc))

	def _resume_miner(self) -> None:
		engine = self.runtime_context.get("miner_engine")
		adapter = self.runtime_context.get("stratum_adapter")
		boot_obj = self.runtime_context.get("boot")
		submitter = getattr(boot_obj, "submitter", None) if boot_obj is not None else None
		for obj in (submitter, adapter, engine):
			if obj is not None and hasattr(obj, "resume"):
				try:
					obj.resume(note="GUI resume requested", source="control_center_win")
				except Exception:
					pass
		self._log_console_event("Miner resume requested")
		self.statusBar().showMessage("Miner resumed", 4000)

	def _pause_miner_from_ui(self) -> None:
		try:
			pause_miner(self.runtime_context, note="GUI pause requested", source="control_center_win")
			self._log_console_event("Miner pause requested")
			self.statusBar().showMessage("Miner paused", 4000)
		except Exception as exc:
			self._log_console_event("Pause failed: %s" % str(exc))
			QMessageBox.warning(self, "Pause Failed", str(exc))

	def _run_prediction_cycle(self) -> None:
		engine = self.runtime_context.get("prediction_engine")
		if engine is None:
			self._log_console_event("Prediction engine unavailable")
			self.statusBar().showMessage("Prediction engine unavailable", 4000)
			return
		if self._prediction_busy:
			self._log_console_event("Prediction cycle already running")
			self.statusBar().showMessage("Prediction cycle already running", 3000)
			return

		self._prediction_busy = True
		self._log_console_event("Prediction cycle started")
		self.statusBar().showMessage("Prediction cycle started...", 2000)

		def _work() -> None:
			text = "Prediction cycle completed"
			try:
				result = engine.run_once()
				text = "Prediction cycle completed: %s orders" % int((result or {}).get("orders", 0) if isinstance(result, dict) else 0)
			except Exception as exc:
				text = "Prediction cycle failed: %s" % str(exc)
			finally:
				self._prediction_busy = False
				QTimer.singleShot(0, lambda: self._finalize_prediction_cycle(text))

		threading.Thread(target=_work, daemon=True, name="prediction_cycle_ui").start()

	def _finalize_prediction_cycle(self, text: str) -> None:
		self._log_console_event(text)
		self.statusBar().showMessage(text, 6000)
		self._refresh_live_telemetry()

	def _start_all(self) -> None:
		self._resume_miner()
		self._run_prediction_cycle()

	def _open_user_settings(self) -> None:
		dialog = UserSettingsDialog(self)
		if dialog.exec() != QDialog.Accepted:
			self._log_console_event("User settings dismissed")
			return
		values = dialog.values()
		for key, value in values.items():
			ConfigManager.set(key, value)
		self._apply_prediction_settings(values)
		self._log_console_event("User settings saved")
		self.statusBar().showMessage("User settings saved", 4000)
		self._refresh_live_telemetry()

	def _apply_prediction_settings(self, values: Dict[str, Any]) -> None:
		os.environ["CRYPTOCOM_API_KEY"] = str(values.get("prediction.api_key", "") or "")
		os.environ["CRYPTOCOM_API_SECRET"] = str(values.get("prediction.api_secret", "") or "")
		engine = self.runtime_context.get("prediction_engine")
		if engine is None:
			return
		try:
			engine.base_quote = str(values.get("prediction.base_quote", engine.base_quote) or engine.base_quote).upper()
			engine.max_assets = int(values.get("prediction.max_assets", engine.max_assets) or engine.max_assets)
			engine.max_clusters = int(values.get("prediction.max_clusters", engine.max_clusters) or engine.max_clusters)
			engine.lanes_per_cluster = int(values.get("prediction.lanes_per_cluster", engine.lanes_per_cluster) or engine.lanes_per_cluster)
			engine.min_confidence = float(values.get("prediction.min_confidence", engine.min_confidence) or engine.min_confidence)
			engine.candle_timeframe = str(values.get("prediction.candle_timeframe", engine.candle_timeframe) or engine.candle_timeframe)
			engine.candle_limit = int(values.get("prediction.candle_limit", engine.candle_limit) or engine.candle_limit)
			engine.api = CryptoComAPI()
			if hasattr(engine, "trader"):
				engine.trader = TradeExecutor(min_confidence=engine.min_confidence)
			if hasattr(engine, "watchdog") and getattr(engine, "watchdog", None) is not None:
				engine.watchdog.trade_api = CryptoComAPI()
				engine.watchdog.trade_exec = TradeExecutor(min_confidence=engine.min_confidence)
			if self.vsd is not None:
				self.vsd.store("prediction/config/runtime", {
					"ts": time.time(),
					"base_quote": engine.base_quote,
					"max_assets": engine.max_assets,
					"max_clusters": engine.max_clusters,
					"lanes_per_cluster": engine.lanes_per_cluster,
					"min_confidence": engine.min_confidence,
					"candle_timeframe": engine.candle_timeframe,
					"candle_limit": engine.candle_limit,
					"api_configured": bool(values.get("prediction.api_key") and values.get("prediction.api_secret")),
				})
		except Exception:
			pass

	def _show_worker_probe(self) -> None:
		rows = _collect_worker_probe_rows()
		if not rows:
			self._log_console_event("Worker probe requested with no runtime rows")
			QMessageBox.information(self, "Worker Probe", "No worker-session runtime records found.")
			return
		self._log_console_event("Worker probe opened")
		text = json.dumps(rows, indent=2, sort_keys=True)
		box = QMessageBox(self)
		box.setWindowTitle("Worker Session Probe")
		box.setText("Active worker-session runtime state")
		box.setDetailedText(text)
		box.exec()

	def _tick_simulation(self) -> None:
		self._refresh_live_telemetry()
		self._apply_research_model(self._load_research_model())
		self.research_chart.tick()
		self.research_graph.tick()
		self._research_tick += 1
		if hasattr(self, "section_nav"):
			spinner = ["|", "/", "-", "\\"][self._research_tick % 4]
			item = self.section_nav.item(3)
			if item is not None:
				item.setText(f"{spinner} Research Lab {self.research_yield_label.text()}")

	def _refresh_live_telemetry(self) -> None:
		self._refresh_top_metrics()
		self._refresh_lane_table()
		self._refresh_prediction_runtime_tab()
		self._refresh_vsd_snapshot_list()
		self._refresh_worker_status_panel()
		self._refresh_quick_state()
		self._refresh_command_snapshot()
		self._refresh_backend_notes()

	def _refresh_top_metrics(self) -> None:
		idx = list(self.vsd.get("telemetry/metrics/index", []) or []) if self.vsd is not None else []
		total_hashrate = 0.0
		observed_share = 0.0
		observed_count = 0
		for net in idx:
			cur = dict(self.vsd.get("telemetry/metrics/%s/current" % str(net).upper(), {}) or {})
			total_hashrate += float(cur.get("hashes_submitted_hs", cur.get("hashes_found_hs", 0.0)) or 0.0)
			if float(cur.get("observed_target_fraction", 0.0) or 0.0) > 0.0:
				observed_share += float(cur.get("observed_target_fraction", 0.0) or 0.0)
				observed_count += 1
		worker_map = dict(self.vsd.get("miner/lanes/worker_map", {}) or {}) if self.vsd is not None else {}
		connected = 0
		for lane_id in worker_map:
			state = dict(self.vsd.get("miner/stratum/workers/%s/session" % lane_id, {}) or {})
			if bool(state.get("connected", False)):
				connected += 1
		last_report = dict(self.vsd.get("prediction/last_report", {}) or {}) if self.vsd is not None else {}
		self.metric_labels.get("miner_hashrate", QLabel()).setText("%.2f H/s" % total_hashrate)
		self.metric_labels.get("active_lanes", QLabel()).setText(str(len(worker_map)))
		avg_share = (observed_share / float(observed_count)) * 100.0 if observed_count > 0 else 0.0
		self.metric_labels.get("network_share", QLabel()).setText("%.4f %%" % avg_share)
		self.metric_labels.get("worker_sessions", QLabel()).setText("%d / %d" % (connected, len(worker_map)))
		self.metric_labels.get("prediction_orders", QLabel()).setText(str(int(last_report.get("orders", 0) or 0)))
		if hasattr(self, "command_summary"):
			self.command_summary.setText(
				"Hashrate %.2f H/s | share %.4f %% | worker sessions %d/%d | prediction orders %d"
				% (
					total_hashrate,
					avg_share,
					connected,
					len(worker_map),
					int(last_report.get("orders", 0) or 0),
				)
			)

	def _refresh_lane_table(self) -> None:
		worker_map = dict(self.vsd.get("miner/lanes/worker_map", {}) or {}) if self.vsd is not None else {}
		lane_ids = sorted(worker_map.keys())
		self.lane_table.setRowCount(len(lane_ids))
		if hasattr(self, "ops_lane_list"):
			self.ops_lane_list.clear()
		for row, lane_id in enumerate(lane_ids):
			worker = dict(worker_map.get(lane_id, {}) or {})
			coin = str(worker.get("coin", ""))
			runtime = dict(self.vsd.get("miner/runtime/submission_rate/workers/%s" % lane_id, {}) or {})
			lane_metrics = dict(self.vsd.get("telemetry/metrics/%s/shares/lanes/%s" % (coin, lane_id), {}) or {})
			session = dict(self.vsd.get("miner/stratum/workers/%s/session" % lane_id, {}) or {})
			allowed = float(runtime.get("allowed_rate_per_second", 0.0) or 0.0)
			assigned = float(runtime.get("assigned_share_difficulty", session.get("assigned_share_difficulty", 0.0)) or 0.0)
			submitted_hs = float(lane_metrics.get("submitted_hs", 0.0) or 0.0)
			accepted_hs = float(lane_metrics.get("accepted_hs", 0.0) or 0.0)
			connected = "connected" if bool(session.get("connected", False)) else "disconnected"
			telemetry = "%.2f/s allowed | %.2f/s sub | %.2f/s ok | diff %.3f | %s" % (
				allowed,
				submitted_hs,
				accepted_hs,
				assigned,
				connected,
			)
			self.lane_table.setItem(row, 0, QTableWidgetItem(str(lane_id)))
			self.lane_table.setItem(row, 1, QTableWidgetItem(coin))
			self.lane_table.setItem(row, 2, QTableWidgetItem(telemetry))
			if hasattr(self, "ops_lane_list"):
				self.ops_lane_list.addItem(QListWidgetItem("%s | %s | %.2f/s | diff %.3f" % (str(lane_id), coin, allowed, assigned)))
		self.lane_table.resizeColumnsToContents()
		if hasattr(self, "ops_lane_list") and not lane_ids:
			self.ops_lane_list.addItem(QListWidgetItem("No active lane-worker assignments yet"))

	def _refresh_worker_status_panel(self) -> None:
		if not hasattr(self, "worker_status_list"):
			return
		self.worker_status_list.clear()
		rows = _collect_worker_probe_rows()
		if not rows:
			self.worker_status_list.addItem(QListWidgetItem("No worker-session runtime records found"))
			return
		for row in rows[:16]:
			status = "connected" if bool(row.get("connected", False)) else "idle"
			text = "%s | %s | %s | %.3f/s | diff %.3f" % (
				str(row.get("lane_id", "")),
				str(row.get("coin", "")),
				status,
				float(row.get("allowed_rate_per_second", 0.0) or 0.0),
				float(row.get("assigned_share_difficulty", 0.0) or 0.0),
			)
			self.worker_status_list.addItem(QListWidgetItem(text))

	def _refresh_quick_state(self) -> None:
		if not hasattr(self, "quick_state_list"):
			return
		self.quick_state_list.clear()
		state = dict(self.vsd.get("miner/control/state", {}) or {}) if self.vsd is not None else {}
		boot = dict(self.vsd.get("prediction/boot", {}) or {}) if self.vsd is not None else {}
		self.quick_state_list.addItem(QListWidgetItem("Miner phase: %s" % str(state.get("phase", "unknown"))))
		self.quick_state_list.addItem(QListWidgetItem("Miner paused: %s" % ("yes" if bool(state.get("paused", False)) else "no")))
		self.quick_state_list.addItem(QListWidgetItem("Prediction API: %s" % ("connected" if bool(boot.get("api_ok", False)) else "unverified")))
		self.quick_state_list.addItem(QListWidgetItem("Settings: %s" % ConfigManager.path()))

	def _refresh_command_snapshot(self) -> None:
		if not hasattr(self, "command_snapshot_list"):
			return
		self.command_snapshot_list.clear()
		state = dict(self.vsd.get("miner/control/state", {}) or {}) if self.vsd is not None else {}
		boot = dict(self.vsd.get("prediction/boot", {}) or {}) if self.vsd is not None else {}
		report = dict(self.vsd.get("prediction/last_report", {}) or {}) if self.vsd is not None else {}
		self.command_snapshot_list.addItem(QListWidgetItem("Miner phase: %s" % str(state.get("phase", "unknown"))))
		self.command_snapshot_list.addItem(QListWidgetItem("Miner paused: %s" % ("yes" if bool(state.get("paused", False)) else "no")))
		self.command_snapshot_list.addItem(QListWidgetItem("Prediction API: %s" % ("connected" if bool(boot.get("api_ok", False)) else "unverified")))
		self.command_snapshot_list.addItem(QListWidgetItem("Last orders: %s" % str(report.get("orders", 0))))

	def _refresh_backend_notes(self) -> None:
		if not hasattr(self, "backend_notes") or self.vsd is None:
			return
		lines = []
		request_keys = sorted(self.vsd.get("miner/lanes/worker_map", {}).keys()) if isinstance(self.vsd.get("miner/lanes/worker_map", {}), dict) else []
		for lane_id in request_keys:
			worker = dict(self.vsd.get("miner/runtime/submission_rate/workers/%s" % lane_id, {}) or {})
			request = dict(self.vsd.get("miner/difficulty_request/%s/workers/%s" % (str(worker.get("coin", "")).upper(), lane_id), {}) or {})
			if not request:
				continue
			lines.append(
				"%s | req=%s | target diff %.3f | allowed %.3f/s"
				% (
					str(lane_id),
					str(bool(request.get("requested", False))),
					float(request.get("target_share_difficulty", 0.0) or 0.0),
					float(worker.get("allowed_rate_per_second", 0.0) or 0.0),
				)
			)
		self.backend_notes.setPlainText("\n".join(lines) if lines else "No backend difficulty-request notes recorded yet.")

	def _refresh_prediction_runtime_tab(self) -> None:
		if self.vsd is None:
			return
		boot = dict(self.vsd.get("prediction/boot", {}) or {})
		runtime_cfg = dict(self.vsd.get("prediction/config/runtime", {}) or {})
		last_report = dict(self.vsd.get("prediction/last_report", {}) or {})
		signals = list(self.vsd.get("telemetry/predictions/latest", []) or [])
		order_history = list(self.vsd.get("trade/orders/history", []) or [])

		api_ok = bool(boot.get("api_ok", False))
		self.pred_boot_status.setText("API status: %s" % ("connected" if api_ok else "unverified"))
		self.pred_config_status.setText(
			"Base quote: %s | confidence: %.4f | timeframe: %s/%s"
			% (
				str(runtime_cfg.get("base_quote", ConfigManager.get("prediction.base_quote", "USDT") or "USDT")),
				float(runtime_cfg.get("min_confidence", ConfigManager.get("prediction.min_confidence", 0.99) or 0.99)),
				str(runtime_cfg.get("candle_timeframe", ConfigManager.get("prediction.candle_timeframe", "1h") or "1h")),
				str(runtime_cfg.get("candle_limit", ConfigManager.get("prediction.candle_limit", 400) or 400)),
			)
		)
		self.pred_last_report.setText(
			"Last report: %s | assets %s | orders %s"
			% (
				str(last_report.get("ts", "unavailable")),
				str(last_report.get("assets_considered", 0)),
				str(last_report.get("orders", 0)),
			)
		)

		self.pred_signal_list.clear()
		for sig in signals[:20]:
			item = QListWidgetItem(
				"%s | conf %.4f | delta %.4f"
				% (
					str(sig.get("symbol", sig.get("asset", ""))),
					float(sig.get("avg_confidence", 0.0) or 0.0),
					float(sig.get("avg_predicted_change", 0.0) or 0.0),
				)
			)
			self.pred_signal_list.addItem(item)

		lines = []
		for order in order_history[-20:]:
			od = dict(order or {})
			lines.append(
				"%s | %s | ok=%s | conf=%.4f"
				% (
					str(od.get("symbol", "")),
					str(od.get("side", od.get("reason", ""))),
					str(bool(od.get("ok", False))),
					float(od.get("confidence", 0.0) or 0.0),
				)
			)
		self.pred_orders_log.setPlainText("\n".join(lines) if lines else "No prediction orders recorded yet.")

	def _refresh_vsd_snapshot_list(self) -> None:
		if not hasattr(self, "vsd_list") or self.vsd is None:
			return
		self.vsd_list.clear()
		worker_rows = _collect_worker_probe_rows()
		if not worker_rows:
			self.vsd_list.addItem(QListWidgetItem("No worker-session runtime records yet"))
			return
		for row in worker_rows[:12]:
			self.vsd_list.addItem(
				QListWidgetItem(
					"%s %s %.3f/s diff %.3f"
					% (
						str(row.get("lane_id", "")),
						str(row.get("username", "")),
						float(row.get("allowed_rate_per_second", 0.0) or 0.0),
						float(row.get("assigned_share_difficulty", 0.0) or 0.0),
					)
				)
			)

	def closeEvent(self, event: QCloseEvent) -> None:
		try:
			bios_shutdown(self.runtime_context)
		except Exception:
			pass
		super().closeEvent(event)


def main() -> None:
	ConfigManager.set("app.render_backend", str(ConfigManager.get("app.render_backend", "vulkan") or "vulkan"))
	app = QApplication(sys.argv)
	gui.apply_dark_palette(app)
	if os.path.isfile(ICON_PATH):
		app.setWindowIcon(QIcon(ICON_PATH))

	runtime_context = bios_run({
		"mode": "production",
		"debug": False,
		"ui": "control_center_win",
		"disable_live_console": True,
		"platform": "win64",
		"render_backend": "vulkan",
	})

	window = Win64ControlCenterWindow(runtime_context)
	window.show()
	rc = app.exec()
	sys.exit(rc)


if __name__ == "__main__":
	main()
