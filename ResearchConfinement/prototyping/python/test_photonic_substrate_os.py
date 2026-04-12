from __future__ import annotations

from pathlib import Path
import sys
import unittest

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from gpu_pulse_axis_dynamics import build_live_telemetry_payload
from photonic_substrate_os import (
    PhaseTransportObserver,
    PhotonicTextureMap9D,
    TemporalAccountingScheduler,
    build_default_static_base_tone,
    build_process_vector_9d,
)


def _sample_payload(
    quartet: dict | None = None,
    phase_turns: float = 0.41,
    previous_phase_turns: float = 0.37,
    telemetry: dict | None = None,
) -> dict:
    return build_live_telemetry_payload(
        quartet=quartet or {"F": 0.245, "A": 0.180, "I": 0.360, "V": 0.360},
        phase_turns=phase_turns,
        previous_phase_turns=previous_phase_turns,
        telemetry=telemetry or {
            "coherence": 0.9979,
            "trap_ratio": 0.13,
            "predicted_interference": 0.17,
            "temporal_coupling": 0.27,
            "thermal_noise": 0.05,
            "observed_subsystems": {
                "residual": 0.20,
                "spin": 0.08,
                "coupling": 0.24,
                "controller": 0.28,
            },
        },
    )


class TestPhotonicSubstrateOS(unittest.TestCase):
    def test_texture_map_round_trip_preserves_resume_state(self) -> None:
        payload = _sample_payload()
        texture = PhotonicTextureMap9D(texture_id="test_texture")
        texture.attach_base_tone(build_default_static_base_tone(payload))
        resume_state = texture.capture_resume_state(payload, label="boot_resume_state")
        texture.register_dynamic_trajectory(
            label="learned_memory_echo",
            vector_9d=build_process_vector_9d(payload),
            activation_quartet=dict(payload["encoding_activation_path"]["required_activation_pulse"]),
            resonance_threshold=0.93,
            metadata={"lane": "dynamic"},
        )

        snapshot = texture.to_dict()
        restored = PhotonicTextureMap9D.from_dict(snapshot)
        field_map = restored.compose_field_map()

        self.assertEqual(resume_state.carrier_id, restored.saved_resume_state.carrier_id)
        self.assertEqual(field_map["carrier_count"], 7)
        self.assertTrue(field_map["saved_resume_present"])
        self.assertEqual(len(field_map["carrier_field_vector_9d"]), 9)
        self.assertGreater(float(field_map["carrier_field_energy"]), 0.0)

    def test_phase_transport_observer_builds_deterministic_resync_plan(self) -> None:
        saved_payload = _sample_payload()
        live_payload_quartet = {"F": 0.205, "A": 0.165, "I": 0.315, "V": 0.300}
        texture = PhotonicTextureMap9D(texture_id="resync_texture")
        texture.attach_base_tone(build_default_static_base_tone(saved_payload))
        texture.capture_resume_state(saved_payload, label="saved_boot_state")

        observer = PhaseTransportObserver(similarity_threshold=0.97, max_resync_steps=5)
        plan_a = observer.build_resync_plan(
            texture_map=texture,
            quartet=live_payload_quartet,
            phase_turns=0.58,
            previous_phase_turns=0.53,
        )
        plan_b = observer.build_resync_plan(
            texture_map=texture,
            quartet=live_payload_quartet,
            phase_turns=0.58,
            previous_phase_turns=0.53,
        )

        self.assertTrue(plan_a["actuation_required"])
        self.assertEqual(plan_a["estimated_resync_steps"], plan_b["estimated_resync_steps"])
        self.assertEqual(plan_a["pulse_sequence"], plan_b["pulse_sequence"])
        self.assertLess(float(plan_a["similarity"]["overall_similarity"]), 0.97)
        self.assertEqual(len(plan_a["drift_vector_9d"]), 9)

    def test_scheduler_prefers_exact_dynamic_match(self) -> None:
        live_payload = _sample_payload()
        texture = PhotonicTextureMap9D(texture_id="schedule_texture")
        texture.attach_base_tone(build_default_static_base_tone(live_payload))
        exact_dynamic = texture.register_dynamic_trajectory(
            label="exact_live_process",
            vector_9d=build_process_vector_9d(live_payload),
            activation_quartet=dict(live_payload["encoding_activation_path"]["required_activation_pulse"]),
            resonance_threshold=0.95,
            metadata={"process_mode": "mining"},
        )

        scheduler = TemporalAccountingScheduler(similarity_threshold=0.90, minimum_temporal_accuracy=0.60)
        scheduled = scheduler.schedule(live_payload, texture)
        recommended = dict(scheduled["recommended_match"] or {})

        self.assertTrue(scheduled["scheduler_gate_open"])
        self.assertEqual(recommended["carrier_id"], exact_dynamic.carrier_id)
        self.assertEqual(recommended["carrier_source"], "dynamic")
        self.assertAlmostEqual(float(recommended["overall_similarity"]), 1.0, places=6)
        self.assertTrue(scheduled["actuation_events"])

    def test_scheduler_respects_temporal_accuracy_gate(self) -> None:
        live_payload = _sample_payload()
        live_payload["temporal_accounting"]["temporal_accuracy_score"] = 0.20
        texture = PhotonicTextureMap9D(texture_id="gated_texture")
        texture.attach_base_tone(build_default_static_base_tone(live_payload))
        texture.register_dynamic_trajectory(
            label="gated_live_process",
            vector_9d=build_process_vector_9d(live_payload),
            activation_quartet=dict(live_payload["encoding_activation_path"]["required_activation_pulse"]),
            resonance_threshold=0.90,
        )

        scheduler = TemporalAccountingScheduler(similarity_threshold=0.85, minimum_temporal_accuracy=0.65)
        scheduled = scheduler.schedule(live_payload, texture)

        self.assertFalse(scheduled["scheduler_gate_open"])
        self.assertFalse(scheduled["actuation_events"])
        self.assertEqual(dict(scheduled["recommended_match"] or {}).get("carrier_source"), "dynamic")


if __name__ == "__main__":
    unittest.main()