from __future__ import annotations

from pathlib import Path
import sys
import unittest


THIS_DIR = Path(__file__).resolve().parent
HARNESS_DIR = THIS_DIR.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from validate_qbit_run45 import validate_run45  # noqa: E402


class TestValidateQBitRun45(unittest.TestCase):
    def test_run45_envelope_holds(self) -> None:
        summary = validate_run45()
        self.assertGreaterEqual(int(summary.get("frame_count", 0)), 8)
        self.assertGreaterEqual(
            float(summary.get("mean_inertial_mass_proxy", 0.0)),
            float(summary.get("mean_spin_momentum_score", 0.0)),
        )
        self.assertGreaterEqual(
            float(summary.get("mean_temporal_coupling_moment", 0.0)),
            float(summary.get("mean_phase_transport_term", 0.0)),
        )
        self.assertGreaterEqual(
            float(summary.get("mean_flux_transport_term", 0.0)),
            float(summary.get("mean_phase_transport_term", 0.0)),
        )


if __name__ == "__main__":
    unittest.main()