# ASCII-ONLY
import unittest
import time


class _VSDStub:
    def __init__(self):
        self._kv = {"system/bios_boot_ok": True}

    def get(self, key, default=None):
        return self._kv.get(key, default)

    def store(self, key, value):
        self._kv[key] = value


class _SubmitterStub:
    def __init__(self):
        self.calls = []

    def submit_packet(self, lane_id, packet):
        self.calls.append((lane_id, packet))
        return True


class TestGpuPulseRuntime(unittest.TestCase):
    def test_bridge_builds_substrate_trace_from_runtime_state(self):
        from VHW.gpu_pulse_runtime import build_substrate_trace_runtime

        out = build_substrate_trace_runtime(
            lane_id="lane_trace",
            tick=3,
            system_payload={
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
            },
            nonce_snapshot={
                "entropy_score": 0.62,
                "psi": 0.18,
                "flux": 0.11,
                "harmonic": 0.24,
                "phase": 0.37,
                "coherence_peak": 0.74,
                "valid_ratio": 0.0,
                "atomic_vector_x": 0.09,
                "atomic_vector_y": 0.18,
                "atomic_vector_z": 0.27,
            },
            previous_trace_state={},
            sync_vram=False,
        )

        self.assertTrue(out.get("active"))
        self.assertIn("trace_state", out)
        self.assertGreater(float(out["trace_state"].get("trace_support", 0.0)), 0.0)
        self.assertGreater(float(out["trace_state"].get("trace_alignment", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(out.get("simulation_field_state", {}) or {}).get("axis_scale_x", 0.0)), 0.0)
        self.assertGreaterEqual(float(dict(out.get("simulation_field_state", {}) or {}).get("temporal_coupling_moment", 0.0)), 0.0)
        self.assertTrue(dict(dict(out.get("simulation_field_state", {}) or {}).get("simulation_temporal_constant_weights", {}) or {}))
        self.assertTrue(dict(dict(out.get("gpu_feedback", {}) or {}).get("feedback_temporal_constant_weights", {}) or {}))
        self.assertTrue(dict(dict(out.get("gpu_delta_feedback", {}) or {}).get("delta_temporal_constant_weights", {}) or {}))
        self.assertEqual(str(out["trace_vram"].get("reason", "")), "sync_disabled")

    def test_compute_manager_emits_substrate_trace_meta(self):
        from VHW.compute_manager import ComputeManager, ComputeWrapper, Subsystem, ComputeMode
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork

        vsd = _VSDStub()
        cm = ComputeManager({
            "tiers": [{"tier_id": 0, "vqram_mb": 256}],
            "vsd": vsd,
        })
        lane_id = cm.allocate_lane(0).lane_id
        submitter = _SubmitterStub()

        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload={
                "job_id": "btc_trace_job",
                "header_hex": (b"\x00" * 80).hex(),
                "target": "f" * 64,
                "ntime": "00000000",
                "extranonce2": "00000000",
            },
            system_payload={},
            metadata={},
            derived_state={},
        )

        snapshots = [
            {
                "timestamp": "cm_live_0000",
                "cpu": {"util": 0.31},
                "gpu": {"util": 0.68},
                "memory": {"util": 0.52},
            },
            {
                "timestamp": "cm_live_0001",
                "cpu": {"util": 0.34},
                "gpu": {"util": 0.71},
                "memory": {"util": 0.55},
            },
        ]
        snap_index = {"value": 0}

        def _snapshot_provider():
            idx = int(snap_index["value"])
            snap_index["value"] = (idx + 1) % len(snapshots)
            return dict(snapshots[idx])

        def _actuation_hook(meta):
            return {
                "mode": "compute_manager_test",
                "tag": "cm_live_probe",
                "load_hint": 0.72,
                "phase_injection_delta_turns": 0.07,
                "phase_injection_turns": 0.36,
                "phase_ring_strength": 0.66,
                "phase_ring_closure": 0.62,
                "phase_ring_density": 0.57,
                "shared_vector_phase_lock": 0.61,
                "zero_point_crossover_gate": 0.57,
                "frequency_gradient_9d": [0.23] * 9,
                "field_gradient_9d": [0.24] * 9,
                "gradient_spectral_id": "PID9CM",
            }

        wrapper = ComputeWrapper(
            subsys=Subsystem.MINER,
            mode=ComputeMode.BATCH,
            payload={
                "jobs_map": {lane_id: pkt},
                "global_util": 0.81,
                "gpu_util": 0.72,
                "mem_bw_util": 0.68,
                "cpu_util": 0.31,
                "phase_step": 0.09,
                "telemetry_mode": "live_startup",
                "telemetry_sample_period_s": 0.001,
                "capture_sleep": False,
                "_snapshot_provider": _snapshot_provider,
                "_actuation_hook": _actuation_hook,
            },
            params={
                "submitter": submitter,
                "mode": "phase_coherence",
                "count": 4,
            },
        )

        res = cm.dispatch(wrapper)
        self.assertIn(lane_id, res)
        meta = dict(res[lane_id].get("meta", {}) or {})
        self.assertIn("substrate_trace", meta)
        trace = dict(meta.get("substrate_trace", {}) or {})
        self.assertTrue(trace.get("active"))
        self.assertGreater(float(dict(trace.get("trace_state", {}) or {}).get("trace_support", 0.0)), 0.0)
        self.assertEqual(str(trace.get("path", "")), "live_photonic_cycle")
        self.assertTrue(bool(dict(trace.get("actuation_summary", {}) or {}).get("applied", False)))
        self.assertGreater(float(dict(trace.get("trace_state", {}) or {}).get("trace_phase_ring_strength", 0.0)), 0.0)

    def test_build_substrate_trace_runtime_live_cycle_uses_runtime_feedback(self):
        from VHW.gpu_pulse_runtime import build_substrate_trace_runtime
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork

        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload={
                "job_id": "btc_runtime_cycle",
                "header_hex": (b"\x11" * 80).hex(),
                "target": "f" * 64,
                "share_target": "f" * 64,
                "active_target": "f" * 64,
                "ntime": "00000000",
                "extranonce2": "00000000",
            },
            system_payload={},
            metadata={},
            derived_state={},
        )
        snapshots = [
            {"timestamp": "rt_live_0000", "cpu": {"util": 0.25}, "gpu": {"util": 0.64}, "memory": {"util": 0.49}},
            {"timestamp": "rt_live_0001", "cpu": {"util": 0.28}, "gpu": {"util": 0.67}, "memory": {"util": 0.51}},
        ]
        snap_index = {"value": 0}

        def _snapshot_provider():
            idx = int(snap_index["value"])
            snap_index["value"] = (idx + 1) % len(snapshots)
            return dict(snapshots[idx])

        def _actuation_hook(meta):
            return {
                "mode": "runtime_live_test",
                "tag": "runtime_cycle_probe",
                "load_hint": 0.74,
                "phase_injection_delta_turns": 0.08,
                "phase_injection_turns": 0.33,
                "phase_ring_strength": 0.69,
                "shared_vector_phase_lock": 0.63,
                "zero_point_crossover_gate": 0.58,
                "frequency_gradient_9d": [0.21] * 9,
                "field_gradient_9d": [0.22] * 9,
                "gradient_spectral_id": "PID9TEST",
            }

        out = build_substrate_trace_runtime(
            lane_id="lane_live",
            tick=7,
            system_payload={
                "global_util": 0.77,
                "gpu_util": 0.69,
                "mem_bw_util": 0.61,
                "cpu_util": 0.29,
                "phase_step": 0.07,
            },
            nonce_snapshot={
                "entropy_score": 0.58,
                "psi": 0.11,
                "flux": 0.09,
                "harmonic": 0.19,
                "phase": 0.27,
                "coherence_peak": 0.66,
                "valid_ratio": 0.0,
                "atomic_vector_x": 0.05,
                "atomic_vector_y": 0.12,
                "atomic_vector_z": 0.18,
            },
            previous_trace_state={},
            sync_vram=False,
            packet=pkt,
            runtime_payload={
                "telemetry_mode": "live_startup",
                "telemetry_sample_period_s": 0.001,
                "capture_sleep": False,
                "_snapshot_provider": _snapshot_provider,
                "_actuation_hook": _actuation_hook,
            },
            live_cycle=True,
        )

        self.assertTrue(out.get("active"))
        self.assertEqual(str(out.get("path", "")), "live_photonic_cycle")
        self.assertTrue(bool(dict(out.get("actuation_summary", {}) or {}).get("applied", False)))
        self.assertEqual(str(dict(out.get("frame", {}) or {}).get("source", "")), "live_startup")
        self.assertGreater(float(dict(out.get("trace_state", {}) or {}).get("trace_phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(dict(dict(out.get("result", {}) or {}).get("latency_calibration", {}) or {}).get("predicted_latency_s", 0.0)), 0.0)

    def test_telemetry_resonance_benchmark_matches_classical(self):
        from VHW.gpu_pulse_runtime import benchmark_telemetry_resonance

        out = benchmark_telemetry_resonance({
            "program_id": "telemetry_resonance_test",
            "frame_count": 96,
            "horizon_frames": 3,
            "pulse_cycles": 4,
            "repeat_count": 2,
            "warmup_count": 0,
            "noise_scale": 0.06,
        })

        self.assertTrue(out.get("ok"))
        self.assertTrue(out.get("results_match"))
        self.assertEqual(
            dict(out.get("substrate_result", {}) or {}).get("result"),
            dict(out.get("classical_result", {}) or {}).get("result"),
        )
        result = dict(dict(out.get("substrate_result", {}) or {}).get("result", {}) or {})
        self.assertTrue(list(result.get("dominant_nodes", []) or []))
        self.assertGreater(float(result.get("resonance_gate", 0.0)), 0.0)
        self.assertGreaterEqual(float(result.get("noise_gate", 0.0)), 0.0)
        latency = dict(result.get("latency_calibration", {}) or {})
        self.assertGreater(float(latency.get("predicted_latency_s", 0.0)), 0.0)
        self.assertGreater(float(latency.get("kernel_request_s", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("actuation_horizon_frames", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("actuation_compensation", 0.0)), 0.0)
        self.assertGreater(float(latency.get("transport_coherence", 0.0)), 0.0)
        self.assertGreater(float(latency.get("transport_damping_gate", 0.0)), 0.0)
        self.assertIn(str(latency.get("transport_input_mode", "")), ("telemetry_only", "actuation_adjusted_telemetry"))
        self.assertGreaterEqual(float(latency.get("reverse_transport_gate", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("axis_scale_x", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("axis_scale_y", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("axis_scale_z", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("temporal_coupling_moment", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("inertial_mass_proxy", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("spin_momentum_score", 0.0)), 0.0)
        self.assertTrue(str(latency.get("trajectory_spectral_id", "")).startswith("PID9"))
        self.assertTrue(str(latency.get("predicted_trajectory_spectral_id", "")).startswith("PID9"))
        self.assertGreaterEqual(float(latency.get("trajectory_conservation_alignment", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("trajectory_expansion_term", 0.0)), 0.0)
        self.assertEqual(len(list(latency.get("trajectory_9d", []) or [])), 9)
        self.assertEqual(len(list(latency.get("predicted_trajectory_9d", []) or [])), 9)
        hash_probe = dict(out.get("hash_probe", {}) or {})
        self.assertTrue(str(hash_probe.get("probe_hash_hex", "")))
        self.assertGreaterEqual(float(hash_probe.get("distance_score", 0.0)), 0.0)
        self.assertGreaterEqual(int(hash_probe.get("pow_zero_nibbles", 0)), 0)
        self.assertTrue(str(hash_probe.get("probe_source", "")))
        preview = list(out.get("forecast_preview", []) or [])
        self.assertTrue(preview)
        self.assertGreaterEqual(float(dict(preview[0]).get("predicted_latency_s", 0.0)), 0.0)
        self.assertIn("phase_transport_term", dict(preview[0]))

    def test_live_telemetry_resonance_benchmark_matches_classical(self):
        from VHW.gpu_pulse_runtime import benchmark_telemetry_resonance

        snapshots = [
            {
                "timestamp": "live_0000",
                "cpu": {"util": 0.31},
                "gpu": {"util": 0.68},
                "memory": {"util": 0.52},
            },
            {
                "timestamp": "live_0001",
                "cpu": {"util": 0.34},
                "gpu": {"util": 0.71},
                "memory": {"util": 0.55},
            },
            {
                "timestamp": "live_0002",
                "cpu": {"util": 0.29},
                "gpu": {"util": 0.64},
                "memory": {"util": 0.57},
            },
            {
                "timestamp": "live_0003",
                "cpu": {"util": 0.33},
                "gpu": {"util": 0.73},
                "memory": {"util": 0.58},
            },
        ]
        state = {"index": 0}
        actuation = {"count": 0}

        def _snapshot_provider():
            idx = int(state["index"])
            state["index"] = (idx + 1) % len(snapshots)
            return dict(snapshots[idx])

        def _actuation_hook(meta):
            actuation["count"] += 1
            return {
                "mode": "test_hook",
                "tag": "live_probe",
                "load_hint": 0.75,
            }

        out = benchmark_telemetry_resonance({
            "program_id": "telemetry_live_test",
            "telemetry_mode": "live_startup",
            "frame_count": 12,
            "horizon_frames": 2,
            "pulse_cycles": 3,
            "repeat_count": 1,
            "warmup_count": 0,
            "telemetry_sample_period_s": 0.001,
            "capture_sleep": False,
            "_snapshot_provider": _snapshot_provider,
            "_actuation_hook": _actuation_hook,
        })

        self.assertTrue(out.get("ok"))
        self.assertTrue(out.get("results_match"))
        self.assertEqual(str(out.get("telemetry_source", "")), "live_startup")
        actuation_summary = dict(out.get("actuation_summary", {}) or {})
        self.assertTrue(bool(actuation_summary.get("applied", False)))
        self.assertEqual(int(actuation_summary.get("call_count", 0)), 12)
        self.assertEqual(int(actuation_summary.get("applied_count", 0)), 12)
        self.assertEqual(str(actuation_summary.get("mode", "")), "test_hook")
        self.assertEqual(int(actuation["count"]), 12)
        preview = list(out.get("forecast_preview", []) or [])
        self.assertTrue(preview)
        self.assertEqual(str(dict(preview[0]).get("source", "")), "live_startup")
        result = dict(dict(out.get("substrate_result", {}) or {}).get("result", {}) or {})
        latency = dict(result.get("latency_calibration", {}) or {})
        self.assertGreater(float(latency.get("predicted_latency_s", 0.0)), 0.0)
        self.assertGreater(float(latency.get("throughput_hz", 0.0)), 0.0)
        self.assertEqual(str(latency.get("transport_input_mode", "")), "actuation_adjusted_telemetry")
        self.assertFalse(bool(latency.get("raw_telemetry_only", True)))
        self.assertGreater(float(latency.get("phase_transport_term", 0.0)), -1.0)
        self.assertGreater(float(latency.get("flux_transport_term", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("reverse_transport_gate", 0.0)), 0.0)
        self.assertGreater(float(latency.get("observer_damping", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("vector_energy", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("relativistic_correlation", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("spin_momentum_score", 0.0)), 0.0)
        self.assertTrue(str(latency.get("trajectory_spectral_id", "")).startswith("PID9"))
        self.assertGreaterEqual(float(latency.get("trajectory_conservation_alignment", 0.0)), 0.0)
        self.assertEqual(len(list(latency.get("trajectory_9d", []) or [])), 9)
        hash_probe = dict(out.get("hash_probe", {}) or {})
        self.assertTrue(str(hash_probe.get("probe_hash_hex", "")))
        self.assertGreaterEqual(float(hash_probe.get("distance_score", 0.0)), 0.0)
        self.assertTrue(str(hash_probe.get("probe_source", "")))
        comparison = dict(out.get("transport_mode_comparison", {}) or {})
        self.assertTrue(comparison)
        adjusted = dict(comparison.get("adjusted", {}) or {})
        raw_replay = dict(comparison.get("raw_replay", {}) or {})
        self.assertEqual(str(adjusted.get("transport_input_mode", "")), "actuation_adjusted_telemetry")
        self.assertEqual(str(raw_replay.get("transport_input_mode", "")), "telemetry_only")
        self.assertGreaterEqual(float(adjusted.get("hash_distance_score", 0.0)), 0.0)
        self.assertGreaterEqual(float(raw_replay.get("hash_distance_score", 0.0)), 0.0)

    def test_live_photonic_cycle_exposes_phase_ring_metrics(self):
        from VHW.gpu_pulse_runtime import run_live_photonic_substrate_cycle

        snapshot = {
            "timestamp": "cycle_0000",
            "cpu": {"util": 0.22},
            "gpu": {"util": 0.44},
            "memory": {"util": 0.38},
        }

        def _snapshot_provider():
            return dict(snapshot)

        def _actuation_hook(meta):
            return {
                "mode": "test_hook",
                "tag": "phase_ring_probe",
                "load_hint": 0.66,
                "dispatch_elapsed_ms": 4.25,
                "gpu_pulse_phase_effect": 0.08,
                "phase_injection_delta_turns": 0.08,
                "phase_injection_turns": 0.21,
                "phase_ring_closure": 0.77,
                "phase_ring_density": 0.69,
                "phase_ring_strength": 0.81,
                "zero_point_crossover_gate": 0.63,
                "shared_vector_collapse_gate": 0.59,
                "shared_vector_phase_lock": 0.72,
                "inertial_basin_strength": 0.67,
                "frequency_gradient_9d": [0.11, 0.12, 0.13, 0.21, 0.22, 0.23, 0.31, 0.32, 0.33],
            }

        out = run_live_photonic_substrate_cycle({
            "program_id": "live_photonic_cycle_test",
            "telemetry_sample_period_s": 0.001,
            "capture_sleep": False,
            "horizon_frames": 2,
            "pulse_cycles": 1,
            "_snapshot_provider": _snapshot_provider,
            "_actuation_hook": _actuation_hook,
        })

        self.assertTrue(out.get("ok"))
        frame = dict(out.get("frame", {}) or {})
        self.assertGreater(float(frame.get("phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(frame.get("shared_vector_collapse_gate", 0.0)), 0.0)
        self.assertEqual(len(list(frame.get("frequency_gradient_9d", []) or [])), 9)
        trace = dict(out.get("trace_state", {}) or {})
        self.assertGreater(float(trace.get("trace_phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(trace.get("trace_shared_vector_collapse", 0.0)), 0.0)
        self.assertGreater(float(trace.get("trace_temporal_relativity_norm", 0.0)), 0.0)
        latency = dict(out.get("latency_calibration", {}) or {})
        self.assertGreater(float(latency.get("phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(latency.get("zero_point_crossover_gate", 0.0)), 0.0)
        self.assertGreater(float(latency.get("temporal_relativity_norm", 0.0)), 0.0)
        self.assertGreaterEqual(float(latency.get("zero_point_line_distance", 0.0)), 0.0)
        self.assertGreater(float(latency.get("resonant_interception_inertia", 0.0)), 0.0)
        self.assertEqual(len(list(latency.get("frequency_gradient_9d", []) or [])), 9)
        memory_basin = dict(out.get("memory_basin_state", {}) or {})
        scheduler = dict(out.get("scheduler_state", {}) or {})
        process_state = dict(out.get("process_state", {}) or {})
        self.assertTrue(str(memory_basin.get("active_basin_name", "")))
        self.assertTrue(str(scheduler.get("scheduling_mode", "")))
        self.assertTrue(str(process_state.get("process_mode", "")))
        self.assertGreaterEqual(float(process_state.get("mining_resonance_gate", 0.0)), 0.0)
        forecast = list(out.get("forecast_preview", []) or [])
        self.assertTrue(forecast)
        self.assertGreater(float(dict(forecast[0]).get("phase_ring_strength", 0.0)), 0.0)
        self.assertTrue(str(dict(forecast[0]).get("process_mode", "")))

    def test_hash_probe_can_bind_real_btc_packet(self):
        from VHW.gpu_pulse_runtime import benchmark_telemetry_resonance
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork

        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload={
                "job_id": "btc_real_probe",
                "header_hex": (b"\x00" * 80).hex(),
                "target": "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "share_target": "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "active_target": "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "ntime": "00000000",
                "extranonce2": "00000000",
            },
            system_payload={},
            metadata={},
            derived_state={},
        )

        out = benchmark_telemetry_resonance({
            "program_id": "telemetry_real_hash_probe",
            "frame_count": 24,
            "horizon_frames": 2,
            "pulse_cycles": 2,
            "repeat_count": 1,
            "warmup_count": 0,
            "noise_scale": 0.04,
            "hash_probe_packet": pkt,
            "compare_raw_transport_modes": False,
        })

        hash_probe = dict(out.get("hash_probe", {}) or {})
        self.assertEqual(str(hash_probe.get("probe_source", "")), "real_job_packet")
        self.assertEqual(str(hash_probe.get("target_source", "")), "packet_share_target")
        self.assertTrue(str(hash_probe.get("probe_hash_hex", "")))
        self.assertTrue(str(hash_probe.get("selected_nonce_hex", "")))
        self.assertGreaterEqual(float(hash_probe.get("distance_score", 0.0)), 0.0)
        self.assertTrue(list(hash_probe.get("sample_preview", []) or []))

    def test_transport_hash_optimizer_ranks_variants(self):
        from VHW.gpu_pulse_runtime import optimize_telemetry_transport_hash
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork

        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload={
                "job_id": "btc_opt_probe",
                "header_hex": (b"\x00" * 80).hex(),
                "target": "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "share_target": "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "active_target": "0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "ntime": "00000000",
                "extranonce2": "00000000",
            },
            system_payload={},
            metadata={},
            derived_state={},
        )

        out = optimize_telemetry_transport_hash({
            "program_id": "telemetry_transport_optimizer",
            "frame_count": 16,
            "horizon_frames": 2,
            "pulse_cycles": 2,
            "repeat_count": 1,
            "warmup_count": 0,
            "noise_scale": 0.05,
            "hash_probe_packet": pkt,
        })

        self.assertTrue(out.get("ok"))
        best = dict(out.get("best", {}) or {})
        baseline = dict(out.get("baseline", {}) or {})
        ranked = list(out.get("ranked", []) or [])
        self.assertTrue(ranked)
        self.assertTrue(str(best.get("variant", "")))
        self.assertTrue(str(baseline.get("variant", "")))
        self.assertGreaterEqual(float(best.get("hash_distance_score", 0.0)), 0.0)
        self.assertGreaterEqual(int(best.get("hash_candidate_count", 0)), 0)

    def test_vqpu_telemetry_resonance_opcode_updates_stats(self):
        from VHW.vqpu import VQPU

        qpu = VQPU()
        out = qpu.execute("telemetry_resonance_benchmark", {
            "program_id": "telemetry_opcode_test",
            "frame_count": 72,
            "horizon_frames": 2,
            "pulse_cycles": 3,
            "repeat_count": 2,
            "warmup_count": 0,
            "noise_scale": 0.05,
        })

        self.assertTrue(out.get("ok"))
        self.assertTrue(out.get("results_match"))
        stats = qpu.stats()
        self.assertTrue(bool(stats.get("last_benchmark_results_match", False)))
        self.assertGreaterEqual(float(stats.get("microprocess_trace_gate", 0.0)), 0.0)
        self.assertTrue(str(stats.get("last_microprocess_tag", "")))
        self.assertGreater(float(stats.get("last_microprocess_predicted_latency_s", 0.0)), 0.0)
        self.assertGreater(float(stats.get("last_microprocess_kernel_request_s", 0.0)), 0.0)
        self.assertGreater(float(stats.get("last_microprocess_pulse_generation_s", 0.0)), 0.0)
        self.assertGreaterEqual(float(stats.get("last_microprocess_actuation_compensation", 0.0)), 0.0)
        self.assertGreater(float(stats.get("last_microprocess_flux_transport_term", 0.0)), 0.0)
        self.assertGreaterEqual(float(stats.get("last_microprocess_reverse_transport_gate", 0.0)), 0.0)
        self.assertGreater(float(stats.get("last_microprocess_observer_damping", 0.0)), 0.0)

    def test_gpu_initiator_publishes_photonic_cycle_state(self):
        from VHW.gpu_initiator import start_gpu_initiator, stop_gpu_initiator

        vsd = _VSDStub()

        def _snapshot_provider():
            return {
                "timestamp": "initiator_0000",
                "cpu": {"util": 0.20},
                "gpu": {"util": 0.40},
                "memory": {"util": 0.35},
            }

        def _actuation_hook(meta):
            return {
                "mode": "test_hook",
                "tag": "initiator_probe",
                "load_hint": 0.62,
                "dispatch_elapsed_ms": 3.75,
                "gpu_pulse_phase_effect": 0.06,
                "phase_injection_delta_turns": 0.06,
                "phase_injection_turns": 0.19,
                "phase_ring_closure": 0.74,
                "phase_ring_density": 0.66,
                "phase_ring_strength": 0.79,
                "zero_point_crossover_gate": 0.61,
                "shared_vector_collapse_gate": 0.57,
                "shared_vector_phase_lock": 0.71,
                "inertial_basin_strength": 0.65,
                "frequency_gradient_9d": [0.14, 0.15, 0.16, 0.24, 0.25, 0.26, 0.34, 0.35, 0.36],
            }

        stop_gpu_initiator()
        start_gpu_initiator(
            vsd=vsd,
            sustain_pct=0.05,
            actuation_profile={
                "program_id": "gpu_initiator_test",
                "telemetry_sample_period_s": 0.001,
                "loop_period_s": 0.05,
                "capture_sleep": False,
                "history_size": 4,
                "horizon_frames": 1,
                "pulse_cycles": 1,
                "force_gpu_available": True,
                "_snapshot_provider": _snapshot_provider,
                "_actuation_hook": _actuation_hook,
            },
        )
        time.sleep(0.15)
        stop_gpu_initiator()

        heartbeat = dict(vsd.get("vhw/gpu/heartbeat", {}) or {})
        photonic_latency = dict(vsd.get("vhw/gpu/photonic_latency", {}) or {})
        photonic_trace = dict(vsd.get("vhw/gpu/photonic_trace", {}) or {})
        photonic_basin = dict(vsd.get("vhw/gpu/photonic_memory_basin", {}) or {})
        photonic_scheduler = dict(vsd.get("vhw/gpu/photonic_scheduler", {}) or {})
        photonic_process = dict(vsd.get("vhw/gpu/photonic_process", {}) or {})
        self.assertEqual(str(heartbeat.get("mode", "")), "photonic_actuation")
        self.assertGreater(float(heartbeat.get("phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(heartbeat.get("temporal_relativity_norm", 0.0)), 0.0)
        self.assertGreater(float(photonic_latency.get("phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(photonic_latency.get("shared_vector_collapse_gate", 0.0)), 0.0)
        self.assertGreater(float(photonic_latency.get("temporal_relativity_norm", 0.0)), 0.0)
        self.assertGreater(float(photonic_trace.get("trace_phase_ring_strength", 0.0)), 0.0)
        self.assertGreater(float(photonic_trace.get("trace_shared_vector_collapse", 0.0)), 0.0)
        self.assertTrue(str(photonic_basin.get("active_basin_name", "")))
        self.assertTrue(str(photonic_scheduler.get("active_zone_name", "")))
        self.assertTrue(str(photonic_process.get("process_mode", "")))


if __name__ == "__main__":
    unittest.main()
