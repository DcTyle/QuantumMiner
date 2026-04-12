# ASCII-ONLY
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from Control_Center import control_center_gui as gui
from Control_Center.control_center_win import Win64ControlCenterWindow


class _DummyVSD:
    def __init__(self) -> None:
        self._m = {
            "telemetry/metrics/index": [],
            "miner/lanes/worker_map": {},
        }

    def get(self, key, default=None):
        return self._m.get(str(key), default)

    def store(self, key, value):
        self._m[str(key)] = value


class ControlCenterDarkGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        gui.apply_dark_palette(cls.app)

    def test_win64_console_shell_builds_operations_layout(self) -> None:
        window = Win64ControlCenterWindow({"vsd": _DummyVSD(), "bus": None})
        try:
            self.assertEqual(window.section_stack.count(), 5)
            self.assertEqual(window.section_nav.count(), 5)
            self.assertIn("miner_hashrate", window.metric_labels)
            self.assertTrue(hasattr(window, "command_event_list"))
            self.assertTrue(hasattr(window, "command_snapshot_list"))
            self.assertTrue(bool(self.app.styleSheet()))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()