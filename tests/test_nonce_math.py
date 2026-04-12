# ASCII-ONLY
import unittest
import copy

from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork, neural_objectSchema
from miner.nonce_math import NonceMath


class _LaneStub:
    def __init__(self, lane_id: str) -> None:
        self.lane_id = lane_id


class TestNonceMath(unittest.TestCase):
    def test_compute_derivative_nonce_basic(self):
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.Miner_DerivativeNonce,
            network=ComputeNetwork.BTC,
            raw_payload={"any": "value"},
            system_payload={},
            metadata={},
            derived_state={},
        )
        lane = _LaneStub("lane_test")
        out = NonceMath.compute(pkt, lane_state=lane, mode="derivative")
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)
        first = out[0]
        self.assertIn("nonce", first.system_payload)
        self.assertIsInstance(first.system_payload["nonce"], int)
        self.assertTrue(first.system_payload.get("valid", True))

    def test_phase_coherence_nonce_mode_exposes_target_gate_metrics(self):
        raw_job = {
            "job_id": "btc_test",
            "header_hex": (b"\x00" * 80).hex(),
            "target": "f" * 64,
            "ntime": "00000000",
            "extranonce2": "00000000",
        }
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload=copy.deepcopy(raw_job),
            system_payload={
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
            },
            metadata={},
            derived_state={},
        )
        lane = _LaneStub("lane_phase")
        out = NonceMath.compute(pkt, lane_state=lane, mode="phase_coherence", count=6)
        self.assertEqual(len(out), 6)
        self.assertTrue(all("coherence" in packet.system_payload for packet in out))
        snap = NonceMath.snapshot_lane_state(lane)
        self.assertGreater(snap.get("amplitude_cap", 0.0), 0.0)
        self.assertGreater(snap.get("target_interval", 0), 0)
        self.assertGreaterEqual(snap.get("coherence_peak", 0.0), 0.0)
        self.assertIn("atomic_vector_x", snap)
        self.assertIn("atomic_vector_y", snap)
        self.assertIn("atomic_vector_z", snap)
        self.assertGreaterEqual(float(snap.get("axis_scale_x", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("axis_scale_y", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("axis_scale_z", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("temporal_coupling_moment", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("inertial_mass_proxy", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("spin_momentum_score", 0.0)), 0.0)

    def test_phase_coherence_uses_substrate_trace_feedback(self):
        raw_job = {
            "job_id": "btc_trace_feedback",
            "header_hex": (b"\x00" * 80).hex(),
            "target": "f" * 64,
            "ntime": "00000000",
            "extranonce2": "00000000",
        }
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload=copy.deepcopy(raw_job),
            system_payload={
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
            },
            metadata={},
            derived_state={},
        )
        base_state = {
            "base_nonce": 0x12345678,
            "psi": 0.05,
            "flux": 0.02,
            "harmonic": 0.08,
            "phase": 0.17,
            "d1": 0,
            "entropy_score": 0.0,
            "coherence_peak": 0.0,
            "amplitude_cap": 0.0,
            "target_interval": 0,
            "candidate_count": 0,
            "valid_ratio": 0.0,
            "atomic_vector_x": 0.0,
            "atomic_vector_y": 0.0,
            "atomic_vector_z": 0.0,
        }
        lane_plain = _LaneStub("lane_trace_feedback")
        lane_plain._noncemath_state = dict(base_state)
        lane_trace = _LaneStub("lane_trace_feedback")
        lane_trace._noncemath_state = dict(base_state)

        plain = NonceMath.compute(
            pkt,
            lane_state=lane_plain,
            mode="phase_coherence",
            count=4,
            params={"sequence_scan": 64},
        )
        traced = NonceMath.compute(
            pkt,
            lane_state=lane_trace,
            mode="phase_coherence",
            count=4,
            params={
                "sequence_scan": 64,
                "substrate_feedback_weight": 0.55,
                "substrate_scan_boost": 96,
                "substrate_trace_state": {
                    "trace_support": 0.82,
                    "trace_resonance": 0.76,
                    "trace_alignment": 0.88,
                    "trace_memory": 0.71,
                    "trace_flux": 0.63,
                    "trace_temporal_persistence": 0.79,
                    "trace_temporal_overlap": 0.74,
                    "trace_voltage_frequency_flux": 0.69,
                    "trace_frequency_voltage_flux": 0.58,
                    "trace_phase_anchor_turns": 0.43,
                    "trace_axis_vector": [0.67, 0.59, 0.41, 0.72],
                    "trace_vector": [0.12, 0.18, 0.27, 0.43],
                    "trace_dof_vector": [0.61, 0.57, 0.49, 0.55, 0.58, 0.62, 0.53, 0.60, 0.52, 0.59],
                    "trace_phase_ring_strength": 0.67,
                    "trace_shared_vector_phase_lock": 0.64,
                    "trace_temporal_relativity_norm": 0.59,
                    "trace_zero_point_line_distance": 0.22,
                    "trace_field_interference_norm": 0.18,
                    "trace_resonant_interception_inertia": 0.44,
                    "process_resonance": 0.72,
                    "mining_resonance_gate": 0.81,
                    "collapse_readiness": 0.63,
                    "process_mode": "phase_transport",
                    "scheduler_mode": "transport",
                    "active_zone_name": "transport_gate",
                    "active_basin_name": "beta",
                },
            },
        )

        self.assertEqual(len(plain), 4)
        self.assertEqual(len(traced), 4)
        plain_nonces = [int(packet.system_payload["nonce"]) for packet in plain]
        traced_nonces = [int(packet.system_payload["nonce"]) for packet in traced]
        self.assertNotEqual(plain_nonces, traced_nonces)
        self.assertTrue(str(traced[0].system_payload.get("bucket_id", "")))
        self.assertTrue(str(traced[0].system_payload.get("worker_bucket", "")))
        self.assertGreaterEqual(float(traced[0].system_payload.get("bucket_priority", 0.0)), 0.0)
        self.assertLessEqual(float(traced[0].system_payload.get("bucket_priority", 0.0)), 1.0)
        self.assertGreater(float(traced[0].system_payload.get("mining_resonance_score", 0.0)), 0.0)
        self.assertGreater(float(traced[0].system_payload.get("temporal_relativity_norm", 0.0)), 0.0)
        self.assertTrue(str(traced[0].system_payload.get("process_mode", "")))

    def test_gpu_vectorized_mode_produces_ranked_candidates(self):
        raw_job = {
            "job_id": "btc_gpu_vector",
            "header_hex": (b"\x00" * 80).hex(),
            "target": "f" * 64,
            "ntime": "00000000",
            "extranonce2": "00000000",
        }
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload=copy.deepcopy(raw_job),
            system_payload={
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
            },
            metadata={},
            derived_state={},
        )
        lane = _LaneStub("lane_gpu_vector")
        out = NonceMath.compute(
            pkt,
            lane_state=lane,
            mode="gpu_vectorized",
            count=6,
            params={
                "sequence_scan": 64,
                "substrate_feedback_weight": 0.55,
                "substrate_scan_boost": 96,
                "substrate_trace_state": {
                    "trace_support": 0.82,
                    "trace_resonance": 0.76,
                    "trace_alignment": 0.88,
                    "trace_memory": 0.71,
                    "trace_flux": 0.63,
                    "trace_temporal_persistence": 0.79,
                    "trace_temporal_overlap": 0.74,
                    "trace_voltage_frequency_flux": 0.69,
                    "trace_frequency_voltage_flux": 0.58,
                    "trace_phase_anchor_turns": 0.43,
                    "trace_axis_vector": [0.67, 0.59, 0.41, 0.72],
                    "trace_vector": [0.12, 0.18, 0.27, 0.43],
                    "trace_dof_vector": [0.61, 0.57, 0.49, 0.55, 0.58, 0.62, 0.53, 0.60, 0.52, 0.59],
                    "trace_frequency_gradient_9d": [0.44, 0.41, 0.39, 0.42, 0.47, 0.45, 0.40, 0.43, 0.46],
                    "trace_gradient_spectral_id": "PID9TRACE",
                    "trace_gpu_pulse_phase_effect": 0.08,
                    "trace_phase_ring_closure": 0.63,
                    "trace_phase_ring_density": 0.58,
                    "trace_phase_ring_strength": 0.67,
                    "trace_zero_point_crossover": 0.54,
                    "trace_shared_vector_collapse": 0.51,
                    "trace_shared_vector_phase_lock": 0.64,
                    "trace_inertial_basin_strength": 0.62,
                    "trace_temporal_relativity_norm": 0.61,
                    "trace_zero_point_line_distance": 0.19,
                    "trace_field_interference_norm": 0.16,
                    "trace_resonant_interception_inertia": 0.47,
                    "process_resonance": 0.74,
                    "mining_resonance_gate": 0.83,
                    "collapse_readiness": 0.66,
                    "process_mode": "mining_resonance",
                    "scheduler_mode": "transport",
                    "active_zone_name": "transport_gate",
                    "active_basin_name": "beta",
                },
            },
        )

        self.assertEqual(len(out), 6)
        self.assertTrue(all("bucket_id" in packet.system_payload for packet in out))
        self.assertTrue(all("search_backend" in packet.system_payload for packet in out))
        self.assertTrue(all("temporal_coupling_moment" in packet.system_payload for packet in out))
        self.assertTrue(all("inertial_mass_proxy" in packet.system_payload for packet in out))
        self.assertTrue(all("spin_momentum_score" in packet.system_payload for packet in out))
        self.assertTrue(all("phase_ring_strength" in packet.system_payload for packet in out))
        self.assertTrue(all("shared_vector_phase_lock" in packet.system_payload for packet in out))
        self.assertTrue(all("zero_point_crossover_gate" in packet.system_payload for packet in out))
        self.assertTrue(all("mining_resonance_score" in packet.system_payload for packet in out))
        self.assertTrue(all("process_mode" in packet.system_payload for packet in out))
        snap = NonceMath.snapshot_lane_state(lane)
        self.assertGreaterEqual(int(snap.get("candidate_count", 0)), 6)
        self.assertGreaterEqual(float(snap.get("vector_path_score", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("mining_resonance_score", 0.0)), 0.0)
        self.assertTrue(str(snap.get("search_backend", "")))
        self.assertGreaterEqual(float(snap.get("vector_energy", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("relativistic_correlation", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("phase_ring_strength", 0.0)), 0.0)
        self.assertGreaterEqual(float(snap.get("shared_vector_phase_lock", 0.0)), 0.0)
        self.assertTrue(str(snap.get("process_mode", "")))
        self.assertTrue(isinstance(list(snap.get("frequency_gradient_9d", []) or []), list))

    def test_gpu_ranker_prefers_mining_resonance(self):
        from miner.gpu_vectorized_nonce_search import rank_candidates

        ranked = rank_candidates(
            candidate_pool=[
                {
                    "nonce": 1,
                    "coherence": 0.72,
                    "target_alignment": 0.70,
                    "trace_alignment": 0.68,
                    "motif_alignment": 0.67,
                    "phase_alignment_score": 0.71,
                    "sequence_persistence_score": 0.66,
                    "temporal_index_overlap": 0.65,
                    "temporal_coupling_moment": 0.62,
                    "inertial_mass_proxy": 0.48,
                    "spin_momentum_score": 0.52,
                    "relativistic_correlation": 0.44,
                    "temporal_relativity_norm": 0.22,
                    "zero_point_line_distance": 0.76,
                    "field_interference_norm": 0.73,
                    "resonant_interception_inertia": 0.28,
                    "process_resonance": 0.26,
                    "mining_resonance_gate": 0.24,
                    "collapse_readiness": 0.31,
                },
                {
                    "nonce": 2,
                    "coherence": 0.72,
                    "target_alignment": 0.70,
                    "trace_alignment": 0.68,
                    "motif_alignment": 0.67,
                    "phase_alignment_score": 0.71,
                    "sequence_persistence_score": 0.66,
                    "temporal_index_overlap": 0.65,
                    "temporal_coupling_moment": 0.62,
                    "inertial_mass_proxy": 0.48,
                    "spin_momentum_score": 0.52,
                    "relativistic_correlation": 0.44,
                    "temporal_relativity_norm": 0.86,
                    "zero_point_line_distance": 0.14,
                    "field_interference_norm": 0.12,
                    "resonant_interception_inertia": 0.62,
                    "process_resonance": 0.79,
                    "mining_resonance_gate": 0.84,
                    "collapse_readiness": 0.71,
                },
            ],
            batch_size=1,
            simulation_field_state={
                "field_alignment_score": 0.74,
                "kernel_control_gate": 0.71,
                "temporal_index_overlap": 0.65,
                "voltage_frequency_flux": 0.61,
                "frequency_voltage_flux": 0.58,
                "axis_scale_x": 0.63,
                "axis_scale_y": 0.59,
                "axis_scale_z": 0.57,
                "temporal_coupling_moment": 0.62,
                "inertial_mass_proxy": 0.48,
                "spin_momentum_score": 0.52,
                "temporal_relativity_norm": 0.78,
                "zero_point_line_distance": 0.18,
                "field_interference_norm": 0.17,
                "resonant_interception_inertia": 0.56,
                "process_resonance": 0.75,
                "mining_resonance_gate": 0.82,
                "collapse_readiness": 0.69,
            },
            target_profile={"difficulty_norm": 0.20},
            ranking_params={"mining_resonance_weight": 0.28},
        )

        selected = list(ranked.get("selected", []) or [])
        self.assertTrue(selected)
        self.assertEqual(int(selected[0].get("nonce", 0)), 2)
        self.assertGreater(float(selected[0].get("mining_resonance_score", 0.0)), 0.0)

    def test_gpu_vectorized_uses_configured_validation_window(self):
        raw_job = {
            "job_id": "btc_gpu_vector_window",
            "header_hex": (b"\x00" * 80).hex(),
            "target": "f" * 64,
            "share_target": "f" * 64,
            "active_target": "f" * 64,
            "ntime": "00000000",
            "extranonce2": "00000000",
        }
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload=copy.deepcopy(raw_job),
            system_payload={
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
            },
            metadata={},
            derived_state={},
        )
        lane = _LaneStub("lane_gpu_vector_window")
        out = NonceMath.compute(
            pkt,
            lane_state=lane,
            mode="gpu_vectorized",
            count=4,
            params={
                "sequence_scan": 64,
                "gpu_batch_size": 24,
                "substrate_feedback_weight": 0.55,
                "substrate_scan_boost": 96,
                "substrate_trace_state": {
                    "trace_support": 0.82,
                    "trace_resonance": 0.76,
                    "trace_alignment": 0.88,
                    "trace_memory": 0.71,
                    "trace_flux": 0.63,
                    "trace_temporal_persistence": 0.79,
                    "trace_temporal_overlap": 0.74,
                    "trace_voltage_frequency_flux": 0.69,
                    "trace_frequency_voltage_flux": 0.58,
                    "trace_phase_anchor_turns": 0.43,
                    "trace_axis_vector": [0.67, 0.59, 0.41, 0.72],
                    "trace_vector": [0.12, 0.18, 0.27, 0.43],
                    "trace_dof_vector": [0.61, 0.57, 0.49, 0.55, 0.58, 0.62, 0.53, 0.60, 0.52, 0.59],
                },
            },
        )

        snap = NonceMath.snapshot_lane_state(lane)
        self.assertGreater(int(snap.get("validated_candidate_count", 0)), 4)
        self.assertEqual(int(snap.get("valid_share_count", 0)), int(snap.get("validated_candidate_count", 0)))
        self.assertEqual(float(snap.get("valid_ratio", 0.0)), 1.0)
        self.assertGreater(len(out), 4)

    def test_gpu_vectorized_temporal_probe_expands_candidate_window(self):
        raw_job = {
            "job_id": "btc_gpu_vector_temporal_probe",
            "header_hex": (b"\x00" * 80).hex(),
            "target": "f" * 64,
            "share_target": "f" * 64,
            "active_target": "f" * 64,
            "ntime": "00000000",
            "extranonce2": "00000000",
        }
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload=copy.deepcopy(raw_job),
            system_payload={
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
            },
            metadata={},
            derived_state={},
        )
        base_state = {
            "base_nonce": 0x12345678,
            "psi": 0.05,
            "flux": 0.02,
            "harmonic": 0.08,
            "phase": 0.17,
            "d1": 0,
            "entropy_score": 0.0,
            "coherence_peak": 0.0,
            "amplitude_cap": 0.0,
            "target_interval": 0,
            "candidate_count": 0,
            "validated_candidate_count": 0,
            "valid_ratio": 0.0,
            "atomic_vector_x": 0.0,
            "atomic_vector_y": 0.0,
            "atomic_vector_z": 0.0,
        }
        lane_plain = _LaneStub("lane_gpu_temporal_probe")
        lane_plain._noncemath_state = dict(base_state)
        lane_temporal = _LaneStub("lane_gpu_temporal_probe")
        lane_temporal._noncemath_state = dict(base_state)
        base_params = {
            "sequence_scan": 64,
            "gpu_batch_size": 16,
            "substrate_feedback_weight": 0.55,
            "substrate_scan_boost": 96,
            "substrate_trace_state": {
                "trace_support": 0.82,
                "trace_resonance": 0.76,
                "trace_alignment": 0.88,
                "trace_memory": 0.71,
                "trace_flux": 0.63,
                "trace_temporal_persistence": 0.79,
                "trace_temporal_overlap": 0.74,
                "trace_voltage_frequency_flux": 0.69,
                "trace_frequency_voltage_flux": 0.58,
                "trace_phase_anchor_turns": 0.43,
                "trace_axis_vector": [0.67, 0.59, 0.41, 0.72],
                "trace_vector": [0.12, 0.18, 0.27, 0.43],
                "trace_dof_vector": [0.61, 0.57, 0.49, 0.55, 0.58, 0.62, 0.53, 0.60, 0.52, 0.59],
            },
        }

        _ = NonceMath.compute(
            pkt,
            lane_state=lane_plain,
            mode="gpu_vectorized",
            count=4,
            params=dict(base_params | {"temporal_probe_enabled": False}),
        )
        _ = NonceMath.compute(
            pkt,
            lane_state=lane_temporal,
            mode="gpu_vectorized",
            count=4,
            params=dict(base_params | {
                "temporal_probe_enabled": True,
                "temporal_probe_seed_budget": 8,
                "temporal_probe_variants_per_seed": 4,
            }),
        )

        snap_plain = NonceMath.snapshot_lane_state(lane_plain)
        snap_temporal = NonceMath.snapshot_lane_state(lane_temporal)
        self.assertGreaterEqual(int(snap_temporal.get("candidate_count", 0)), int(snap_plain.get("candidate_count", 0)))
        self.assertGreaterEqual(int(snap_temporal.get("validated_candidate_count", 0)), int(snap_plain.get("validated_candidate_count", 0)))

    def test_schema_verifier_prefers_share_target_over_block_target(self):
        packet_norm = {
            "raw_payload": {
                "target": "f" * 64,
                "share_target": "00000000ffff0000000000000000000000000000000000000000000000000000",
            }
        }
        pow_hex = "0000000100000000000000000000000000000000000000000000000000000000"
        verifier = neural_objectSchema[variable_format_enum.BTC_BlockTemplate]["verify_target"]

        self.assertFalse(verifier(packet_norm, pow_hex))


if __name__ == "__main__":
    unittest.main()
