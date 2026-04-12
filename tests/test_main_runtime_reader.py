# ASCII-ONLY
from __future__ import annotations

import types
import unittest


class _VSDStub:
    def __init__(self) -> None:
        self._data = {
            "telemetry/global": {
                "jobs_map": {"lane_0": {"job_id": "abc"}},
                "network_state": {"coin": "BTC"},
            }
        }

    def get(self, key, default=None):
        return self._data.get(key, default)


class MainRuntimeReaderTests(unittest.TestCase):
    def test_make_reader_preserves_live_jobs_map(self) -> None:
        import bios.main_runtime as main_runtime

        old_vsd = main_runtime._vsd
        main_runtime._vsd = _VSDStub()
        try:
            boot_obj = types.SimpleNamespace(
                monitor=types.SimpleNamespace(
                    frame=lambda: {
                        "timestamp": "2026-04-06T00:00:00Z",
                        "global_util": 0.44,
                        "gpu_util": 0.55,
                    }
                )
            )
            reader = main_runtime._make_reader(boot_obj)
            out = reader()

            self.assertEqual(float(out["global_util"]), 0.44)
            self.assertEqual(float(out["gpu_util"]), 0.55)
            self.assertIn("jobs_map", out)
            self.assertIn("lane_0", out["jobs_map"])
            self.assertIn("network_state", out)
        finally:
            main_runtime._vsd = old_vsd


if __name__ == "__main__":
    unittest.main()