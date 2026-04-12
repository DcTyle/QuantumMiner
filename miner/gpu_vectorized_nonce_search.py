from __future__ import annotations

from typing import Any, Dict, List


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[float, ...]:
    return (
        float(candidate.get("mining_resonance_score", 0.0)),
        float(candidate.get("process_resonance", 0.0)),
        float(candidate.get("temporal_probe_score", 0.0)),
        float(candidate.get("gpu_feedback_delta_score", 0.0)),
        float(candidate.get("cuda_temporal_score", 0.0)),
        float(candidate.get("vector_path_score", candidate.get("coherence_peak", candidate.get("coherence", 0.0)))),
        float(candidate.get("temporal_relativity_norm", 0.0)),
        1.0 - float(candidate.get("zero_point_line_distance", 1.0)),
        1.0 - float(candidate.get("field_interference_norm", 1.0)),
        float(candidate.get("spin_momentum_score", 0.0)),
        float(candidate.get("temporal_coupling_moment", 0.0)),
        1.0 - float(candidate.get("inertial_mass_proxy", 0.0)),
        float(candidate.get("relativistic_correlation", 0.0)),
        float(candidate.get("trace_alignment", 0.0)),
        float(candidate.get("target_alignment", 0.0)),
        float(candidate.get("motif_alignment", 0.0)),
        float(candidate.get("sequence_persistence_score", 0.0)),
        float(candidate.get("temporal_index_overlap", 0.0)),
        float(candidate.get("phase_alignment_score", candidate.get("phase_length_pressure", 0.0))),
        1.0 - float(candidate.get("phase_confinement_cost", candidate.get("amplitude_ratio", 1.0))),
    )


def rank_candidates(
    candidate_pool: List[Dict[str, Any]],
    batch_size: int,
    simulation_field_state: Dict[str, Any],
    target_profile: Dict[str, Any],
    ranking_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    working = [dict(item or {}) for item in list(candidate_pool or [])]
    params = dict(ranking_params or {})
    telemetry: Dict[str, Any] = {
        "backend": "runtime_pulse_hash",
        "enabled": True,
        "device": "runtime_gpu_pulse",
        "reason": "ranked_candidates",
        "candidate_count": int(len(working)),
        "expanded_eval_count": int(len(working)),
        "expanded_keep_count": 0,
        "field_alignment_score": float(simulation_field_state.get("field_alignment_score", 0.0)),
        "kernel_control_gate": float(simulation_field_state.get("kernel_control_gate", 0.0)),
        "difficulty_norm": float(target_profile.get("difficulty_norm", 0.0)),
        "axis_scale_x": float(simulation_field_state.get("axis_scale_x", 0.0)),
        "axis_scale_y": float(simulation_field_state.get("axis_scale_y", 0.0)),
        "axis_scale_z": float(simulation_field_state.get("axis_scale_z", 0.0)),
        "temporal_coupling_moment": float(simulation_field_state.get("temporal_coupling_moment", 0.0)),
        "inertial_mass_proxy": float(simulation_field_state.get("inertial_mass_proxy", 0.0)),
        "spin_momentum_score": float(simulation_field_state.get("spin_momentum_score", 0.0)),
        "temporal_relativity_norm": float(simulation_field_state.get("temporal_relativity_norm", 0.0)),
        "process_resonance": float(simulation_field_state.get("process_resonance", 0.0)),
        "mining_resonance_gate": float(simulation_field_state.get("mining_resonance_gate", 0.0)),
    }

    if not working:
        return {
            "selected": [],
            "telemetry": telemetry,
            "candidate_count": 0,
        }

    field_alignment = _clamp01(simulation_field_state.get("field_alignment_score", 0.0))
    temporal_overlap = _clamp01(simulation_field_state.get("temporal_index_overlap", 0.0))
    voltage_flux = _clamp01(simulation_field_state.get("voltage_frequency_flux", 0.0))
    frequency_flux = _clamp01(simulation_field_state.get("frequency_voltage_flux", 0.0))
    difficulty_norm = _clamp01(target_profile.get("difficulty_norm", 0.0))
    field_axis_x = _clamp01(simulation_field_state.get("axis_scale_x", 0.0))
    field_axis_y = _clamp01(simulation_field_state.get("axis_scale_y", 0.0))
    field_axis_z = _clamp01(simulation_field_state.get("axis_scale_z", 0.0))
    field_temporal_coupling = _clamp01(simulation_field_state.get("temporal_coupling_moment", 0.0))
    field_inertia = _clamp01(simulation_field_state.get("inertial_mass_proxy", 0.0))
    field_spin = _clamp01(simulation_field_state.get("spin_momentum_score", 0.0))
    field_temporal_relativity = _clamp01(simulation_field_state.get("temporal_relativity_norm", 0.0))
    field_zero_point_line_distance = _clamp01(simulation_field_state.get("zero_point_line_distance", 0.0))
    field_interference = _clamp01(simulation_field_state.get("field_interference_norm", 0.0))
    field_interception_inertia = _clamp01(simulation_field_state.get("resonant_interception_inertia", 0.0))
    field_process_resonance = _clamp01(simulation_field_state.get("process_resonance", 0.0))
    field_mining_resonance_gate = _clamp01(simulation_field_state.get("mining_resonance_gate", 0.0))
    field_collapse_readiness = _clamp01(simulation_field_state.get("collapse_readiness", 0.0))
    mining_resonance_weight = _clamp01(params.get("mining_resonance_weight", 0.28))
    process_resonance_weight = _clamp01(params.get("process_resonance_weight", 0.18))
    temporal_relativity_weight = _clamp01(params.get("temporal_relativity_weight", 0.16))
    zero_point_line_weight = _clamp01(params.get("zero_point_line_weight", 0.14))
    field_interference_weight = _clamp01(params.get("field_interference_weight", 0.12))
    collapse_readiness_weight = _clamp01(params.get("collapse_readiness_weight", 0.12))
    axis_resonance = _clamp01(
        1.0 - (
            abs(field_axis_x - field_axis_y)
            + abs(field_axis_y - field_axis_z)
            + abs(field_axis_x - field_axis_z)
        ) / 3.0
    )

    for candidate in working:
        coherence = _clamp01(candidate.get("coherence_peak", candidate.get("coherence", 0.0)))
        trace_alignment = _clamp01(candidate.get("trace_alignment", 0.0))
        target_alignment = _clamp01(candidate.get("target_alignment", 0.0))
        motif_alignment = _clamp01(candidate.get("motif_alignment", 0.0))
        temporal_coupling = _clamp01(candidate.get("temporal_coupling_moment", field_temporal_coupling))
        inertial_mass = _clamp01(candidate.get("inertial_mass_proxy", field_inertia))
        spin_score = _clamp01(candidate.get("spin_momentum_score", field_spin))
        relativistic_correlation = _clamp01(candidate.get("relativistic_correlation", 0.0))
        temporal_relativity = _clamp01(candidate.get("temporal_relativity_norm", field_temporal_relativity))
        zero_point_line_distance = _clamp01(candidate.get("zero_point_line_distance", field_zero_point_line_distance))
        field_interference_norm = _clamp01(candidate.get("field_interference_norm", field_interference))
        resonant_interception_inertia = _clamp01(candidate.get("resonant_interception_inertia", field_interception_inertia))
        process_resonance = _clamp01(candidate.get("process_resonance", field_process_resonance))
        mining_resonance_gate = _clamp01(candidate.get("mining_resonance_gate", field_mining_resonance_gate))
        collapse_readiness = _clamp01(candidate.get("collapse_readiness", field_collapse_readiness))
        phase_alignment = _clamp01(
            candidate.get("phase_alignment_score", candidate.get("phase_length_pressure", coherence))
        )
        phase_cost = _clamp01(candidate.get("phase_confinement_cost", candidate.get("amplitude_ratio", 0.0)))
        sequence_persistence = _clamp01(candidate.get("sequence_persistence_score", 0.0))
        candidate["cuda_temporal_score"] = _clamp01(
            0.26 * coherence
            + 0.22 * temporal_overlap
            + 0.18 * sequence_persistence
            + 0.16 * phase_alignment
            + 0.18 * (1.0 - phase_cost)
        )
        candidate["gpu_feedback_delta_score"] = _clamp01(
            0.20 * field_alignment
            + 0.16 * trace_alignment
            + 0.16 * target_alignment
            + 0.10 * voltage_flux
            + 0.10 * frequency_flux
            + 0.10 * motif_alignment
            + 0.10 * temporal_coupling
            + 0.08 * spin_score
            + 0.06 * axis_resonance
            + 0.04 * relativistic_correlation
            + 0.06 * (1.0 - inertial_mass)
            + 0.08 * temporal_relativity
        )
        candidate["vector_path_score"] = _clamp01(
            0.22 * coherence
            + 0.14 * target_alignment
            + 0.12 * trace_alignment
            + 0.10 * motif_alignment
            + 0.08 * sequence_persistence
            + 0.06 * temporal_overlap
            + 0.10 * temporal_coupling
            + 0.08 * spin_score
            + 0.06 * axis_resonance
            + 0.04 * relativistic_correlation
            + 0.08 * _clamp01(candidate.get("temporal_probe_score", 0.0))
            + 0.08 * temporal_relativity
            + 0.06 * (1.0 - zero_point_line_distance)
            + 0.04 * (1.0 - field_interference_norm)
            + 0.10 * (1.0 - min(1.0, difficulty_norm + phase_cost * 0.5 + inertial_mass * 0.25))
        )
        candidate["mining_resonance_score"] = _clamp01(
            mining_resonance_weight * mining_resonance_gate
            + process_resonance_weight * process_resonance
            + temporal_relativity_weight * temporal_relativity
            + zero_point_line_weight * (1.0 - zero_point_line_distance)
            + field_interference_weight * (1.0 - field_interference_norm)
            + collapse_readiness_weight * collapse_readiness
            + 0.12 * candidate["vector_path_score"]
            + 0.10 * coherence
            + 0.08 * target_alignment
            + 0.08 * trace_alignment
            + 0.06 * (1.0 - inertial_mass)
            + 0.04 * resonant_interception_inertia
        )
    working.sort(key=_candidate_sort_key, reverse=True)
    seen = set()
    selected: List[Dict[str, Any]] = []
    max_keep = max(1, int(batch_size or 1))
    for candidate in working:
        try:
            nonce = int(candidate.get("nonce", 0)) & 0xFFFFFFFF
        except Exception:
            nonce = 0
        if nonce in seen:
            continue
        seen.add(nonce)
        selected.append(candidate)
        if len(selected) >= max_keep:
            break

    telemetry["candidate_count"] = int(len(working))
    telemetry["expanded_eval_count"] = int(len(working))
    telemetry["expanded_keep_count"] = int(len(selected))
    telemetry["top_vector_path_score"] = float(
        max([float(candidate.get("vector_path_score", 0.0)) for candidate in selected] or [0.0])
    )
    telemetry["top_mining_resonance_score"] = float(
        max([float(candidate.get("mining_resonance_score", 0.0)) for candidate in selected] or [0.0])
    )
    telemetry["top_temporal_coupling_moment"] = float(
        max([float(candidate.get("temporal_coupling_moment", 0.0)) for candidate in selected] or [0.0])
    )
    telemetry["top_spin_momentum_score"] = float(
        max([float(candidate.get("spin_momentum_score", 0.0)) for candidate in selected] or [0.0])
    )
    telemetry["top_inertial_mass_proxy"] = float(
        max([float(candidate.get("inertial_mass_proxy", 0.0)) for candidate in selected] or [0.0])
    )

    return {
        "selected": selected,
        "telemetry": telemetry,
        "candidate_count": int(len(working)),
    }
