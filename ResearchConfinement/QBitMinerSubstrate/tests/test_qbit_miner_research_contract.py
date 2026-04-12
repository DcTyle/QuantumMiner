import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "spec" / "photonic_identity_trace_schema.json"
CASES_PATH = ROOT / "spec" / "substrate_research_cases.json"
EXPORT_SPEC_PATH = ROOT / "spec" / "calibration_export_format.json"
DEVICE_PROFIT_SPEC_PATH = ROOT / "spec" / "device_profit_validation_contract.json"
HARDWARE_PROFILE_SCHEMA_PATH = ROOT / "spec" / "hardware_profile_schema.json"
PROFIT_WINDOW_SCHEMA_PATH = ROOT / "spec" / "profit_window_schema.json"


class QBitMinerResearchContractTests(unittest.TestCase):
    def test_device_artifact_schemas_are_well_formed(self) -> None:
        hardware_schema = json.loads(HARDWARE_PROFILE_SCHEMA_PATH.read_text(encoding="ascii"))
        profit_schema = json.loads(PROFIT_WINDOW_SCHEMA_PATH.read_text(encoding="ascii"))

        self.assertEqual(hardware_schema["name"], "qbit_miner_device_hardware_profile")
        self.assertEqual(hardware_schema["version"], 1)
        self.assertIn("hardware_profile_id", hardware_schema["required_top_level_fields"])
        self.assertIn("power_draw_watts", hardware_schema["required_top_level_fields"])
        self.assertIn("electricity_cost_usd_per_kwh", hardware_schema["required_top_level_fields"])

        self.assertEqual(profit_schema["name"], "qbit_miner_profit_window")
        self.assertEqual(profit_schema["version"], 1)
        self.assertIn("window_start_utc", profit_schema["required_top_level_fields"])
        self.assertIn("net_profit_usd", profit_schema["required_top_level_fields"])
        self.assertIn("accepted_share_count", profit_schema["required_top_level_fields"])

    def test_device_profit_spec_is_well_formed(self) -> None:
        device_profit_spec = json.loads(DEVICE_PROFIT_SPEC_PATH.read_text(encoding="ascii"))

        self.assertEqual(device_profit_spec["name"], "qbit_miner_device_profit_validation_contract")
        self.assertEqual(device_profit_spec["version"], 1)
        self.assertEqual(device_profit_spec["repository_role"], "parallel_substrate_miner")
        self.assertTrue(device_profit_spec["device_required"])
        self.assertTrue(device_profit_spec["substrate_native_only"])
        self.assertEqual(device_profit_spec["primary_target_network"], "bitcoin")
        self.assertGreater(float(device_profit_spec["stretch_network_share_target_percent"]), 0.0)
        self.assertFalse(device_profit_spec["stretch_target_is_guaranteed"])

        completion_gate = dict(device_profit_spec["research_completion_gate"])
        self.assertTrue(completion_gate["requires_device_operation"])
        self.assertTrue(completion_gate["requires_positive_net_profit"])
        self.assertTrue(completion_gate["requires_power_cost_accounting"])
        self.assertTrue(completion_gate["requires_reproducible_profit_window"])

        self.assertIn("phase_vector_temporal_accounting", device_profit_spec["required_runtime_capabilities"])
        self.assertIn("substrate_native_microprocessing", device_profit_spec["required_runtime_capabilities"])
        self.assertIn("hardware_profile.json", device_profit_spec["required_device_artifacts"])
        self.assertIn("profit_window.json", device_profit_spec["required_device_artifacts"])
        self.assertIn("net_profit_usd", device_profit_spec["profit_window_required_fields"])
        self.assertIn("power_cost_usd", device_profit_spec["profit_window_required_fields"])
        self.assertTrue(dict(device_profit_spec["funding_model"])["self_funding_research"])

    def test_calibration_export_spec_is_well_formed(self) -> None:
        export_spec = json.loads(EXPORT_SPEC_PATH.read_text(encoding="ascii"))

        self.assertEqual(export_spec["name"], "quantum_miner_calibration_export_bundle")
        self.assertEqual(export_spec["version"], 1)
        self.assertIn("manifest.json", export_spec["bundle_root_files"])
        self.assertIn("traces.jsonl", export_spec["bundle_root_files"])
        self.assertIn("trace.json", export_spec["trace_directory_files"])
        self.assertIn("calibration_plan.json", export_spec["trace_directory_files"])
        self.assertEqual(export_spec["sweep_directory"], "sweeps")
        self.assertIn("export_format", export_spec["manifest_required_fields"])
        self.assertIn("trace_count", export_spec["manifest_required_fields"])
        self.assertIn("source_identity", export_spec["plan_required_fields"])
        self.assertIn("sweep_index", export_spec["sweep_required_fields"])

    def test_schema_and_cases_are_consistent(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="ascii"))
        cases = json.loads(CASES_PATH.read_text(encoding="ascii"))

        required = list(schema["required_top_level_fields"])
        array_lengths = dict(schema["array_lengths"])
        calibration_plan_fields = list(schema["calibration_plan_fields"])
        calibration_sweep_fields = list(schema["calibration_sweep_fields"])
        calibration_variable_order = list(schema["calibration_variable_order"])
        calibration_direction_order = list(schema["calibration_direction_order"])
        field_vector_fields = list(schema["field_vector_fields"])
        spin_fields = list(schema["spin_inertia_fields"])

        self.assertTrue(cases["cases"])
        for case in cases["cases"]:
            for field in required:
                self.assertIn(field, case)

            self.assertEqual(len(case["spectra_9d"]), array_lengths["spectra_9d"])
            self.assertEqual(len(case["trajectory_9d"]), array_lengths["trajectory_9d"])
            self.assertEqual(len(case["encoded_pulse"]), array_lengths["encoded_pulse"])
            self.assertEqual(len(case["rotational_velocity"]), array_lengths["rotational_velocity"])
            self.assertEqual(len(case["spin_inertia"]["axis_spin"]), array_lengths["axis_spin"])
            self.assertEqual(len(case["spin_inertia"]["axis_orientation"]), array_lengths["axis_orientation"])

            for field in field_vector_fields:
                self.assertIn(field, case["field_vector"])

            for field in spin_fields:
                self.assertIn(field, case["spin_inertia"])

            for field in calibration_plan_fields:
                self.assertIn(field, case["calibration_plan"])

            self.assertEqual(len(case["calibration_plan"]["sweeps"]), len(calibration_variable_order) * len(calibration_direction_order))
            for sweep in case["calibration_plan"]["sweeps"]:
                for field in calibration_sweep_fields:
                    self.assertIn(field, sweep)

            for variable_index, variable in enumerate(calibration_variable_order):
                start = variable_index * len(calibration_direction_order)
                block = case["calibration_plan"]["sweeps"][start:start + len(calibration_direction_order)]
                self.assertTrue(all(sweep["variable"] == variable for sweep in block))
                self.assertEqual([sweep["direction"] for sweep in block], calibration_direction_order)

            self.assertGreaterEqual(float(case["response_time_ms"]), float(case["request_time_ms"]))
            self.assertGreaterEqual(float(case["accounting_time_ms"]), 0.0)
            self.assertGreaterEqual(float(case["next_feedback_time_ms"]), 0.0)
            self.assertGreaterEqual(float(case["closed_loop_latency_ms"]), float(case["response_time_ms"]))
            self.assertGreaterEqual(float(case["encode_deadline_ms"]), 0.0)
            self.assertGreaterEqual(int(case["encodable_node_count"]), 0)
            self.assertGreaterEqual(float(case["coupling_strength"]), 0.0)
            self.assertGreaterEqual(float(case["coupling_collision_noise"]), 0.0)
            self.assertGreaterEqual(float(case["temporal_dynamics_noise"]), 0.0)
            self.assertGreaterEqual(float(case["reverse_causal_flux_coherence"]), 0.0)
            self.assertGreaterEqual(float(case["zero_point_overlap_score"]), 0.0)
            self.assertGreaterEqual(float(case["constant_phase_alignment"]), 0.0)
            self.assertGreaterEqual(float(case["trajectory_conservation_score"]), 0.0)
            self.assertGreaterEqual(float(case["expansion_factor"]), 1.0)
            self.assertGreaterEqual(float(case["substrate_inertia"]), 0.0)
            self.assertGreaterEqual(float(case["observer_factor"]), 0.0)
            self.assertLessEqual(float(case["observer_factor"]), 1.0)
            self.assertGreaterEqual(int(case["spin_inertia"]["temporal_coupling_count"]), 0)


if __name__ == "__main__":
    unittest.main()