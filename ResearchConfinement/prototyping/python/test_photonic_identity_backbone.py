from __future__ import annotations

import unittest

from photonic_identity_backbone import (
    analyze_photonic_identity_backbone,
    compute_photonic_identity_record,
    load_default_inputs,
)


class TestPhotonicIdentityBackbone(unittest.TestCase):
    def test_same_inputs_produce_same_identity(self) -> None:
        bundle = load_default_inputs()
        frames = list(bundle.get("frames", []) or [])
        record_a = compute_photonic_identity_record(
            frame=dict(frames[0]),
            previous_frame=None,
            temporal_schema=dict(bundle.get("temporal_schema", {}) or {}),
            process_schema=dict(bundle.get("process_schema", {}) or {}),
            live_ledger=dict(bundle.get("live_ledger", {}) or {}),
            nist_reference=dict(bundle.get("nist_reference", {}) or {}),
        )
        record_b = compute_photonic_identity_record(
            frame=dict(frames[0]),
            previous_frame=None,
            temporal_schema=dict(bundle.get("temporal_schema", {}) or {}),
            process_schema=dict(bundle.get("process_schema", {}) or {}),
            live_ledger=dict(bundle.get("live_ledger", {}) or {}),
            nist_reference=dict(bundle.get("nist_reference", {}) or {}),
        )

        self.assertEqual(str(record_a["photonic_identity"]), str(record_b["photonic_identity"]))
        self.assertEqual(list(record_a["spectra_sig9"]), list(record_b["spectra_sig9"]))
        self.assertEqual(len(list(record_a["spectra_sig9"])), 9)

    def test_identity_changes_across_distinct_frames(self) -> None:
        bundle = load_default_inputs()
        frames = list(bundle.get("frames", []) or [])
        record_a = compute_photonic_identity_record(
            frame=dict(frames[0]),
            previous_frame=None,
            temporal_schema=dict(bundle.get("temporal_schema", {}) or {}),
            process_schema=dict(bundle.get("process_schema", {}) or {}),
            live_ledger=dict(bundle.get("live_ledger", {}) or {}),
            nist_reference=dict(bundle.get("nist_reference", {}) or {}),
        )
        record_b = compute_photonic_identity_record(
            frame=dict(frames[1]),
            previous_frame=dict(frames[0]),
            temporal_schema=dict(bundle.get("temporal_schema", {}) or {}),
            process_schema=dict(bundle.get("process_schema", {}) or {}),
            live_ledger=dict(bundle.get("live_ledger", {}) or {}),
            nist_reference=dict(bundle.get("nist_reference", {}) or {}),
        )

        self.assertNotEqual(str(record_a["photonic_identity"]), str(record_b["photonic_identity"]))
        self.assertGreaterEqual(float(dict(record_b.get("observer", {}) or {}).get("observer_damping", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("transport", {}) or {}).get("phase_correction_norm", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_temporal_accounting", {}) or {}).get("temporal_accuracy_score", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_anchor_vectors", {}) or {}).get("anchor_interference_norm", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_harmonic_noise", {}) or {}).get("harmonic_noise_reaction_norm", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_trajectory_state", {}) or {}).get("trajectory_conservation_9d", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_trajectory_state", {}) or {}).get("reverse_causal_flux_coherence", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_pulse_interference", {}) or {}).get("gpu_pulse_interference_norm", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(record_b.get("predictive_pulse_interference", {}) or {}).get("system_sensitivity_norm", 0.0)), 0.0)
        self.assertTrue(str(dict(record_b.get("predictive_photonic_identity", {}) or {}).get("phase_ring_utf8", "")).startswith("PRING|"))
        self.assertGreaterEqual(float(dict(record_b.get("predictive_phase_ring_trace", {}) or {}).get("zero_point_crossover", 0.0)), 0.0)

    def test_analysis_emits_packets_and_nodes(self) -> None:
        bundle = load_default_inputs()
        analysis = analyze_photonic_identity_backbone(
            frames=list(bundle.get("frames", []) or [])[:4],
            temporal_schema=dict(bundle.get("temporal_schema", {}) or {}),
            process_schema=dict(bundle.get("process_schema", {}) or {}),
            live_ledger=dict(bundle.get("live_ledger", {}) or {}),
            nist_reference=dict(bundle.get("nist_reference", {}) or {}),
        )

        summary = dict(analysis.get("summary", {}) or {})
        records = list(analysis.get("trace_records", []) or [])
        packets = list(analysis.get("api_packets", []) or [])
        nodes = list(analysis.get("disruption_nodes", []) or [])
        self.assertEqual(int(summary.get("input_frame_count", 0)), 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(len(packets), 4)
        self.assertTrue(nodes)
        self.assertGreater(int(summary.get("unique_photonic_identities", 0)), 0)
        self.assertGreaterEqual(float(summary.get("mean_temporal_coupling_moment", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_inertial_mass_proxy", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_temporal_accuracy", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_anchor_interference", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_harmonic_noise_reaction", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_trajectory_conservation", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_reverse_causal_flux_coherence", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_hidden_flux_correction", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_gpu_pulse_interference", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_system_sensitivity", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_pulse_backreaction", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_phase_ring_density", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_zero_point_crossover", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_identity_sweep_cluster", 0.0)), 0.0)
        self.assertGreaterEqual(float(summary.get("mean_predictive_crosstalk_cluster", 0.0)), 0.0)
        self.assertTrue(str(dict(packets[0].get("request", {}) or {}).get("photonic_identity", "")))
        self.assertIn("predictive_temporal_accounting", records[0])
        self.assertIn("predictive_anchor_vectors", records[0])
        self.assertIn("predictive_harmonic_noise", records[0])
        self.assertIn("predictive_trajectory_state", records[0])
        self.assertIn("predictive_pulse_interference", records[0])
        self.assertIn("predictive_phase_ring_trace", records[0])
        self.assertTrue(str(dict(records[0].get("predictive_photonic_identity", {}) or {}).get("phase_ring_utf8", "")).startswith("PRING|"))
        self.assertIn("predictive_calibration", summary)
        self.assertGreaterEqual(float(dict(summary.get("predictive_calibration", {}) or {}).get("mean_trajectory_conservation", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(summary.get("predictive_calibration", {}) or {}).get("mean_gpu_pulse_interference", 0.0)), 0.0)
        self.assertIn(str(dict(summary.get("predictive_calibration", {}) or {}).get("feedback_gate_state", "")), {"open", "gated", "unknown"})


if __name__ == "__main__":
    unittest.main()