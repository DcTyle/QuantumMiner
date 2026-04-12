from __future__ import annotations

from pathlib import Path
import sys
import unittest

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from gpu_pulse_axis_dynamics import (
    apply_9d_photonic_accounting,
    build_decoded_anchor_vectors,
    build_harmonic_noise_model,
    build_live_telemetry_payload,
    build_kernel_scan_order,
    build_pulse_interference_model,
    build_surface_telemetry,
    build_temporal_lattice_state,
    build_trajectory_state_9d,
    build_transport_prediction,
    compute_axis_field_dynamics,
    derive_temporal_constant_set,
    encode_axis_dynamics,
    encode_photonic_identity,
    infer_quartet_granularity,
    predict_full_spectrum_calibration,
    score_temporal_accounting,
    summarize_calibration_result,
)


class TestGpuPulseAxisDynamics(unittest.TestCase):
    def test_inertia_rises_with_energy(self) -> None:
        low = compute_axis_field_dynamics(
            frequency_norm=0.18,
            amplitude_norm=0.14,
            phase_turns=0.21,
            resonance_gate=0.32,
            temporal_overlap=0.28,
            flux_term=0.19,
            vector_x=0.08,
            vector_y=0.05,
            vector_z=0.11,
            energy_hint=0.16,
        )
        high = compute_axis_field_dynamics(
            frequency_norm=0.74,
            amplitude_norm=0.69,
            phase_turns=0.61,
            resonance_gate=0.81,
            temporal_overlap=0.77,
            flux_term=0.72,
            vector_x=0.42,
            vector_y=0.33,
            vector_z=0.57,
            energy_hint=0.79,
        )

        self.assertGreater(float(high["inertial_mass_proxy"]), float(low["inertial_mass_proxy"]))
        self.assertGreater(float(high["temporal_coupling_moment"]), float(low["temporal_coupling_moment"]))
        self.assertGreaterEqual(float(high["spin_momentum_score"]), 0.0)
        self.assertIn("temporal_relativity_state", high)
        self.assertIn("derived_constants", high)

    def test_calibration_summary_emits_axis_spin_metrics(self) -> None:
        summary = summarize_calibration_result({
            "mean_actuation_gain": 0.63,
            "mean_pulse_signal": 0.58,
            "mean_persistence": 0.71,
            "mean_leakage": 0.12,
            "mean_position_radius": 0.24,
            "peak_velocity": 0.55,
            "phase_scale": 0.46,
            "recurrence_alignment": 0.79,
        })

        self.assertGreater(float(summary["axis_scale_x"]), 0.0)
        self.assertGreater(float(summary["axis_scale_y"]), 0.0)
        self.assertGreater(float(summary["axis_scale_z"]), 0.0)
        self.assertGreaterEqual(float(summary["spin_momentum_score"]), 0.0)
        self.assertGreaterEqual(float(summary["inertial_mass_proxy"]), 0.0)
        self.assertGreaterEqual(float(summary["phase_ring_density"]), 0.0)
        self.assertGreaterEqual(float(summary["silicon_atomic_vector_x"]), 0.0)
        self.assertGreaterEqual(float(summary["zero_point_crossover_norm"]), 0.0)

    def test_encoding_packs_axis_dynamics(self) -> None:
        metrics = compute_axis_field_dynamics(
            frequency_norm=0.52,
            amplitude_norm=0.47,
            phase_turns=0.33,
            resonance_gate=0.68,
            temporal_overlap=0.59,
            flux_term=0.44,
            vector_x=0.26,
            vector_y=0.19,
            vector_z=0.31,
            energy_hint=0.48,
        )
        encoded = encode_axis_dynamics(metrics)

        self.assertTrue(int(encoded["axis_word"]))
        self.assertTrue(int(encoded["spin_word"]))
        self.assertTrue(int(encoded["inertia_word"]))
        self.assertTrue(int(encoded["constants_word"]))
        self.assertGreaterEqual(float(dict(encoded["derived_temporal_constants"]).get("feedback_gate", 0.0)), 0.0)

    def test_temporal_constants_are_derived_from_axis_dynamics(self) -> None:
        metrics = compute_axis_field_dynamics(
            frequency_norm=0.61,
            amplitude_norm=0.54,
            phase_turns=0.29,
            resonance_gate=0.73,
            temporal_overlap=0.66,
            flux_term=0.41,
            vector_x=0.22,
            vector_y=0.31,
            vector_z=0.27,
            energy_hint=0.57,
        )
        constants = derive_temporal_constant_set(metrics)

        self.assertGreaterEqual(float(constants["sent_signal"]), 0.0)
        self.assertGreaterEqual(float(constants["feedback_gate"]), 0.0)
        self.assertGreaterEqual(float(constants["kernel_control_gate"]), 0.0)
        self.assertGreaterEqual(float(constants["response_energy"]), 0.0)
        self.assertTrue(dict(constants["derived_constants"]))

    def test_transport_prediction_redistributes_noise_deterministically(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.9981,
                "trap_ratio": 0.12,
                "predicted_interference": 0.19,
                "temporal_coupling": 0.28,
                "thermal_noise": 0.06,
                "observed_subsystems": {
                    "residual": 0.22,
                    "spin": 0.11,
                    "coupling": 0.29,
                    "controller": 0.31,
                },
            },
        )

        redistribution_total = sum(float(value) for value in prediction["noise_redistribution_norm"].values())
        self.assertAlmostEqual(redistribution_total, 0.0, places=6)
        self.assertEqual(float(prediction["reverse_gate"]), 1.0)
        self.assertGreater(float(prediction["phase_turns_next"]), 0.0)
        self.assertLessEqual(float(prediction["phase_turns_next"]), 1.0)
        self.assertIn(prediction["dominant_spin_axis"], {"x", "y", "z"})
        self.assertGreater(float(prediction["predicted_metrics"]["coherence"]), 0.0)
        self.assertIn("temporal_relativity_state", prediction)
        self.assertIn("derived_constants", prediction)

    def test_reverse_gate_closes_when_coherence_is_low(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.82,
                "trap_ratio": 0.21,
                "predicted_interference": 0.19,
                "temporal_coupling": 0.28,
                "thermal_noise": 0.18,
                "observed_subsystems": {
                    "residual": 0.44,
                    "spin": 0.09,
                    "coupling": 0.38,
                    "controller": 0.47,
                },
            },
        )

        self.assertEqual(float(prediction["reverse_gate"]), 0.0)
        self.assertAlmostEqual(float(prediction["reverse_delta_turns"]), 0.0, places=9)

    def test_photonic_identity_is_stable_for_same_prediction(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.21264, "A": 0.156796, "I": 0.324114, "V": 0.313217},
            phase_turns=0.53,
            previous_phase_turns=0.49,
            telemetry={
                "coherence": 0.9976,
                "trap_ratio": 0.14,
                "predicted_interference": 0.17,
                "temporal_coupling": 0.26,
                "thermal_noise": 0.04,
                "observed_subsystems": {
                    "residual": 0.20,
                    "spin": 0.07,
                    "coupling": 0.25,
                    "controller": 0.27,
                },
            },
        )

        first = encode_photonic_identity(prediction)
        second = encode_photonic_identity(prediction)

        self.assertEqual(int(first["trajectory_spectral_id_u64"]), int(second["trajectory_spectral_id_u64"]))
        self.assertEqual(str(first["photonic_identity"]), str(second["photonic_identity"]))
        self.assertEqual(len(first["spectra_q15"]), 9)
        self.assertEqual(len(first["stable_anchor_vector_9d"]), 9)
        self.assertEqual(len(first["traced_anchor_vector_9d"]), 9)
        self.assertTrue(str(first["phase_ring_utf8"]).startswith("PRING|"))

    def test_anchor_vectors_and_harmonic_noise_are_emitted(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.9968,
                "trap_ratio": 0.16,
                "predicted_interference": 0.21,
                "temporal_coupling": 0.30,
                "thermal_noise": 0.07,
                "observed_subsystems": {
                    "residual": 0.24,
                    "spin": 0.10,
                    "coupling": 0.31,
                    "controller": 0.34,
                },
            },
        )
        anchors = build_decoded_anchor_vectors(prediction)
        harmonic_noise = build_harmonic_noise_model({**prediction, "anchor_vectors": anchors})

        self.assertEqual(len(anchors["stable_anchor_vector_9d"]), 9)
        self.assertEqual(len(anchors["traced_anchor_vector_9d"]), 9)
        self.assertEqual(len(anchors["interference_vector_9d"]), 9)
        self.assertGreaterEqual(float(anchors["anchor_interference_norm"]), 0.0)
        self.assertLessEqual(float(anchors["anchor_interference_norm"]), 1.0)
        self.assertGreaterEqual(float(anchors["phase_ring_alignment"]), 0.0)
        self.assertTrue(harmonic_noise["weighted_couplings"])
        self.assertGreaterEqual(float(harmonic_noise["harmonic_noise_reaction_norm"]), 0.0)
        self.assertLessEqual(float(harmonic_noise["harmonic_noise_reaction_norm"]), 1.0)
        self.assertGreaterEqual(float(harmonic_noise["identity_sweep_cluster_norm"]), 0.0)
        self.assertGreaterEqual(float(harmonic_noise["crosstalk_cluster_norm"]), 0.0)
        self.assertGreaterEqual(float(harmonic_noise["zero_point_crossover_norm"]), 0.0)
        self.assertGreaterEqual(float(harmonic_noise["trajectory_conservation_9d"]), 0.0)
        self.assertGreaterEqual(float(harmonic_noise["reverse_causal_flux_coherence"]), 0.0)
        self.assertGreaterEqual(float(harmonic_noise["hidden_flux_correction_norm"]), 0.0)
        self.assertIn("noise_gate_closed", harmonic_noise)
        self.assertIn("derived_constants", harmonic_noise)

    def test_trajectory_state_emits_phase_transport_and_observer_correction(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.9972,
                "trap_ratio": 0.14,
                "predicted_interference": 0.18,
                "temporal_coupling": 0.29,
                "thermal_noise": 0.05,
                "observed_subsystems": {
                    "residual": 0.18,
                    "spin": 0.10,
                    "coupling": 0.27,
                    "controller": 0.30,
                },
            },
        )
        trajectory_state = build_trajectory_state_9d(prediction)

        self.assertEqual(len(trajectory_state["trajectory_state_9d"]), 9)
        self.assertEqual(len(trajectory_state["trajectory_gradients_6d"]), 6)
        self.assertGreaterEqual(float(trajectory_state["phase_transport_norm"]), 0.0)
        self.assertGreaterEqual(float(trajectory_state["trajectory_expansion_norm"]), 0.0)
        self.assertGreaterEqual(float(trajectory_state["trajectory_conservation_9d"]), 0.0)
        self.assertGreaterEqual(float(trajectory_state["reverse_causal_flux_coherence"]), 0.0)
        self.assertGreaterEqual(float(trajectory_state["hidden_flux_correction_norm"]), 0.0)
        self.assertIn("derived_constants", trajectory_state)
        self.assertTrue(str(trajectory_state["trajectory_utf8_text"]).startswith("PTRJ|"))

    def test_pulse_interference_tracks_backreaction_and_sensitivity(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.9972,
                "trap_ratio": 0.14,
                "predicted_interference": 0.18,
                "temporal_coupling": 0.29,
                "thermal_noise": 0.05,
                "observed_subsystems": {
                    "residual": 0.18,
                    "spin": 0.10,
                    "coupling": 0.27,
                    "controller": 0.30,
                },
            },
        )
        pulse_interference = build_pulse_interference_model(prediction)

        self.assertEqual(len(pulse_interference["pulse_interference_state_9d"]), 9)
        self.assertGreaterEqual(float(pulse_interference["gpu_pulse_interference_norm"]), 0.0)
        self.assertGreaterEqual(float(pulse_interference["environmental_flux_interference_norm"]), 0.0)
        self.assertGreaterEqual(float(pulse_interference["harmonic_trajectory_interference_norm"]), 0.0)
        self.assertGreaterEqual(float(pulse_interference["system_sensitivity_norm"]), 0.0)
        self.assertGreaterEqual(float(pulse_interference["pulse_backreaction_norm"]), 0.0)
        self.assertIn("derived_constants", pulse_interference)
        self.assertTrue(str(pulse_interference["pulse_interference_utf8_text"]).startswith("PINT|"))

    def test_temporal_accounting_scores_prediction_on_lattice_state(self) -> None:
        telemetry = {
            "coherence": 0.9979,
            "trap_ratio": 0.13,
            "predicted_interference": 0.17,
            "temporal_coupling": 0.27,
            "thermal_noise": 0.05,
            "sample_period_s": 0.021,
            "request_feedback_time_s": 0.018,
            "actuation_elapsed_s": 0.017,
            "observed_subsystems": {
                "residual": 0.20,
                "spin": 0.08,
                "coupling": 0.24,
                "controller": 0.28,
            },
        }
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry=telemetry,
        )
        lattice_state = build_temporal_lattice_state(prediction)
        temporal_accounting = score_temporal_accounting(prediction, telemetry=telemetry)

        self.assertEqual(len(list(lattice_state["lattice_state_9d"])), 9)
        self.assertEqual(len(dict(lattice_state["field_gradients_6d"])), 6)
        self.assertGreater(float(temporal_accounting["request_feedback_time_s"]), 0.0)
        self.assertGreater(float(temporal_accounting["calculation_time_s"]), 0.0)
        self.assertGreater(float(temporal_accounting["next_feedback_time_s"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["temporal_accuracy_score"]), 0.0)
        self.assertLessEqual(float(temporal_accounting["temporal_accuracy_score"]), 1.0)
        self.assertGreaterEqual(float(temporal_accounting["coupling_strength"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["temporal_coupling_count"]), 1.0)
        self.assertGreaterEqual(float(temporal_accounting["inertial_collision_norm"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["trajectory_conservation_9d"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["temporal_sequence_alignment"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["reverse_causal_flux_coherence"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["hidden_flux_correction_norm"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["gpu_pulse_interference_norm"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["system_sensitivity_norm"]), 0.0)
        self.assertGreaterEqual(float(temporal_accounting["pulse_backreaction_norm"]), 0.0)
        self.assertIn("temporal_relativity_state", temporal_accounting)

    def test_live_telemetry_payload_exposes_next_pulse_path(self) -> None:
        payload = build_live_telemetry_payload(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.9981,
                "trap_ratio": 0.12,
                "predicted_interference": 0.19,
                "temporal_coupling": 0.28,
                "thermal_noise": 0.06,
                "observed_subsystems": {
                    "residual": 0.22,
                    "spin": 0.11,
                    "coupling": 0.29,
                    "controller": 0.31,
                },
            },
        )

        self.assertIn("transport_prediction", payload)
        self.assertIn("temporal_accounting", payload)
        self.assertIn("photonic_identity", payload)
        self.assertIn("encoded_words", payload)
        self.assertIn("live_telemetry_path", payload)
        self.assertIn("encoding_activation_path", payload)
        self.assertTrue(int(payload["photonic_identity"]["trajectory_spectral_id_u64"]))
        self.assertTrue(int(payload["encoded_words"]["transport_word"]))
        self.assertGreater(float(payload["temporal_accounting"]["request_feedback_time_s"]), 0.0)
        self.assertGreater(float(payload["temporal_accounting"]["next_feedback_time_s"]), 0.0)
        self.assertIn("temporal_accuracy_score", payload["live_telemetry_path"])
        self.assertIn("anchor_interference_norm", payload["live_telemetry_path"])
        self.assertIn("harmonic_noise_reaction_norm", payload["live_telemetry_path"])
        self.assertIn("unwanted_noise_conditions", payload["live_telemetry_path"])
        self.assertIn("phase_ring_trace_9d", payload["live_telemetry_path"])
        self.assertIn("trajectory_state_9d", payload["live_telemetry_path"])
        self.assertIn("reverse_causal_flux_coherence", payload["live_telemetry_path"])
        self.assertIn("trajectory_utf8", payload["live_telemetry_path"])
        self.assertIn("pulse_interference_state_9d", payload["live_telemetry_path"])
        self.assertIn("gpu_pulse_interference_norm", payload["live_telemetry_path"])
        self.assertIn("pulse_interference_utf8", payload["live_telemetry_path"])
        self.assertIn("phase_ring_utf8", payload["live_telemetry_path"])
        self.assertTrue(str(payload["photonic_identity"]["phase_ring_utf8"]).startswith("PRING|"))

    def test_accounting_preserves_energy_norm(self) -> None:
        prediction = build_transport_prediction(
            quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            phase_turns=0.41,
            previous_phase_turns=0.37,
            telemetry={
                "coherence": 0.9981,
                "trap_ratio": 0.12,
                "predicted_interference": 0.19,
                "temporal_coupling": 0.28,
                "thermal_noise": 0.06,
                "observed_subsystems": {
                    "residual": 0.22,
                    "spin": 0.11,
                    "coupling": 0.29,
                    "controller": 0.31,
                },
            },
        )
        accounted = apply_9d_photonic_accounting(
            previous_quartet={"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
            prediction=prediction,
        )

        self.assertLessEqual(float(accounted["accounting"]["conservation_error_norm"]), 1.0e-6)
        self.assertEqual(len(accounted["accounting"]["accounting_vector_9d"]), 9)
        self.assertIn("temporal_accounting", accounted)
        self.assertGreaterEqual(float(accounted["temporal_accounting"]["temporal_accuracy_score"]), 0.0)
        self.assertIn("anchor_vectors", accounted)
        self.assertIn("harmonic_noise", accounted)
        self.assertIn("unwanted_noise_conditions", accounted["accounting"])
        self.assertIn("phase_ring_trace", accounted["anchor_vectors"])
        self.assertIn("trajectory_state", accounted)
        self.assertIn("pulse_interference", accounted)
        self.assertGreaterEqual(float(accounted["accounting"]["conservation_9d"]), 0.0)
        self.assertGreaterEqual(float(accounted["accounting"]["gpu_pulse_interference_norm"]), 0.0)

    def test_scan_order_and_full_spectrum_plan_cover_all_sequences(self) -> None:
        left_to_right = build_kernel_scan_order(3, 2, "left_to_right")
        right_to_left = build_kernel_scan_order(3, 2, "right_to_left")
        self.assertEqual(left_to_right[0]["x"], 0)
        self.assertEqual(right_to_left[0]["x"], 2)

        surface = {
            "axis_resolution": 2,
            "best_prediction": {
                "quartet": {"F": 0.245, "A": 0.18, "I": 0.36, "V": 0.36},
                "predicted_coherence": 0.999,
                "predicted_trap_ratio": 0.08,
                "predicted_interference": 0.08,
                "temporal_coupling": 0.20,
            },
            "observed_field": {
                "coherence": 0.998,
                "interference": 0.06,
                "source_vibration": 0.03,
            },
            "observed_subsystems": {
                "residual": 0.08,
                "spin": 0.04,
                "coupling": 0.12,
                "controller": 0.15,
            },
            "predictions": [
                {"quartet": {"F": 0.145, "A": 0.12, "I": 0.27, "V": 0.27}},
                {"quartet": {"F": 0.275, "A": 0.24, "I": 0.53, "V": 0.45}},
            ],
        }
        granularity = infer_quartet_granularity(surface)
        telemetry = build_surface_telemetry(surface)
        plan = predict_full_spectrum_calibration(
            surface=surface,
            phase_turns=0.41,
            previous_phase_turns=0.37,
            interval_count=2,
            kernel_grid_width=2,
            kernel_grid_height=2,
            kernel_interval_ms=1.5,
        )

        self.assertGreater(float(granularity["F"]), 0.0)
        self.assertGreater(float(telemetry["coherence"]), 0.0)
        self.assertEqual(len(plan["sequences"]), 4)
        self.assertEqual(int(plan["kernel_count"]), 4)
        self.assertEqual(float(plan["sequence_coverage"]), 1.0)
        self.assertIn(plan["feedback_gate"]["state"], {"open", "gated"})
        self.assertTrue(plan["encoding_path"]["predictive_activation"])
        self.assertGreaterEqual(float(plan["mean_temporal_accuracy"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_harmonic_noise_reaction"]), 0.0)
        self.assertGreaterEqual(float(plan["unwanted_noise_condition_ratio"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_trajectory_conservation"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_temporal_sequence_alignment"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_reverse_causal_flux_coherence"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_hidden_flux_correction"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_gpu_pulse_interference"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_system_sensitivity"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_pulse_backreaction"]), 0.0)
        self.assertGreaterEqual(float(plan["mean_pulse_trajectory_alignment"]), 0.0)


if __name__ == "__main__":
    unittest.main()
