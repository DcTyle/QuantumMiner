from __future__ import annotations

from typing import Any, Dict
import hashlib
import math


DEFAULT_NORMALIZED_WINDOW = {
    "frequency": {"min": 0.145, "max": 0.275},
    "amplitude": {"min": 0.12, "max": 0.24},
    "amperage": {"min": 0.27, "max": 0.53},
    "voltage": {"min": 0.27, "max": 0.45},
}

DEFAULT_PULSE_CODES = {
    "f_code": 0.245,
    "a_code": 0.18,
    "i_code": 0.33,
    "v_code": 0.33,
    "normalized_window": DEFAULT_NORMALIZED_WINDOW,
}

DEFAULT_DEVIATION_OPERATORS = {
    "score": {
        "center_value": 0.627545,
        "jacobian": {"F": -0.00955, "A": 0.0257, "I": -0.06335, "V": -0.0350333333333341},
        "hessian_diag": {"F": 0.03, "A": 0.005, "I": -0.145555555555547, "V": 0.0},
        "hessian_cross": {
            "FA": 0.0,
            "FI": -0.04,
            "FV": -0.000833333333449815,
            "AI": 0.000416666666678648,
            "AV": 0.0,
            "IV": 0.000277777777754926,
        },
    },
    "trap": {
        "center_value": 0.092102,
        "jacobian": {"F": -0.104115, "A": -0.0051975, "I": 0.161433333333333, "V": 0.0},
        "hessian_diag": {"F": 0.093, "A": 0.01475, "I": -0.296222222222229, "V": 0.0},
        "hessian_cross": {
            "FA": 0.00075,
            "FI": -0.0820833333333345,
            "FV": 0.0,
            "AI": 0.00195833333333182,
            "AV": 0.0,
            "IV": 0.0,
        },
    },
    "coherence": {
        "center_value": 0.995889,
        "jacobian": {"F": -0.00375, "A": -0.00005, "I": 0.00285, "V": 0.0001},
        "hessian_diag": {"F": 0.01, "A": 0.0, "I": 0.0322222222221619, "V": -0.00222222222228612},
        "hessian_cross": {
            "FA": 0.0,
            "FI": 0.00666666666658082,
            "FV": 0.0,
            "AI": 0.0,
            "AV": 0.0,
            "IV": -0.000277777777816605,
        },
    },
    "inertia": {
        "center_value": 0.000544171,
        "jacobian": {"F": 0.00017935, "A": 0.000743775, "I": 0.000206383333333334, "V": 0.0000591333333333338},
        "hessian_diag": {"F": -0.00491, "A": -0.0000025, "I": 0.0000833333333334408, "V": -0.00481111111111109},
        "hessian_cross": {
            "FA": 0.0002425,
            "FI": 0.00001,
            "FV": -0.002,
            "AI": 0.0002825,
            "AV": 0.0000720833333333389,
            "IV": -0.000124166666666664,
        },
    },
    "curvature": {
        "center_value": 17.456,
        "jacobian": {"F": -142.42, "A": -6.365, "I": -11.725, "V": -113.155},
        "hessian_diag": {"F": 8586.0, "A": 21.0, "I": 113.444444444445, "V": 7561.22222222222},
        "hessian_cross": {
            "FA": 169.25,
            "FI": -212.0,
            "FV": 3110.58333333333,
            "AI": -22.6666666666686,
            "AV": -87.2916666666675,
            "IV": -329.916666666668,
        },
    },
}

DEFAULT_COLLAPSE_GATES = {
    "gate_coherence": 0.9972,
    "gate_trap": 0.157,
    "gate_score": 0.66,
    "reverse_mutation_tolerance": 0.28,
}

SCAN_DIRECTIONS = (
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
)

QUARTET_TO_WINDOW_KEY = {
    "F": "frequency",
    "A": "amplitude",
    "I": "amperage",
    "V": "voltage",
}


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def clamp_signed(value: Any, limit: float = 1.0) -> float:
    bound = abs(float(limit))
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    return max(-bound, min(bound, numeric))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def vector_energy(values: list[float]) -> float:
    if not values:
        return 0.0
    total = sum(float(item) * float(item) for item in values)
    return clamp01(math.sqrt(max(total, 0.0)) / math.sqrt(float(len(values))))


def wrap_turns(value: Any) -> float:
    numeric = safe_float(value, 0.0)
    wrapped = numeric % 1.0
    if wrapped < 0.0:
        wrapped += 1.0
    return float(wrapped)


def signed_turn_delta(next_turns: Any, prev_turns: Any) -> float:
    delta = wrap_turns(next_turns) - wrap_turns(prev_turns)
    if delta > 0.5:
        delta -= 1.0
    elif delta < -0.5:
        delta += 1.0
    return float(delta)


def phase_zero_line_proximity(phase_turns: Any) -> float:
    phase = wrap_turns(phase_turns)
    distance = min(abs(phase), abs(phase - 0.5), abs(phase - 1.0))
    return clamp01(1.0 - (4.0 * distance))


def phase_peak_proximity(phase_turns: Any) -> float:
    phase = wrap_turns(phase_turns)
    distance = min(abs(phase - 0.25), abs(phase - 0.75))
    return clamp01(1.0 - (4.0 * distance))


def _normalize_weight_map(signals: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    positive_total = 0.0
    for name, value in signals.items():
        numeric = max(0.0, safe_float(value, 0.0))
        normalized[name] = float(numeric)
        positive_total += float(numeric)
    if positive_total <= 1.0e-9:
        count = max(len(normalized), 1)
        return {name: 1.0 / float(count) for name in normalized}
    return {name: float(value / positive_total) for name, value in normalized.items()}


def _temporal_mix(signals: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    values = {name: clamp01(value) for name, value in signals.items()}
    weights = _normalize_weight_map(values)
    mixed = sum(float(weights[name]) * float(values[name]) for name in values)
    return clamp01(mixed), weights


def _temporal_signed_mix(signals: Dict[str, Any], limit: float = 1.0) -> tuple[float, Dict[str, float]]:
    signed_values = {name: clamp_signed(value, limit=limit) for name, value in signals.items()}
    weights = _normalize_weight_map({name: abs(value) for name, value in signed_values.items()})
    mixed = sum(float(weights[name]) * float(signed_values[name]) for name in signed_values)
    return clamp_signed(mixed, limit=limit), weights


def build_temporal_relativity_state(
    phase_turns: Any,
    frequency_norm: Any,
    amplitude_norm: Any,
    resonance_gate: Any = 0.0,
    temporal_overlap: Any = 0.0,
    flux_norm: Any = 0.0,
    vector_x: Any = 0.0,
    vector_y: Any = 0.0,
    vector_z: Any = 0.0,
    speed_norm: Any = 0.0,
    coupling_strength: Any = 0.0,
    spin_momentum_score: Any = 0.0,
    orientation_shear_norm: Any = 0.0,
    observer_feedback_norm: Any = 0.0,
    phase_memory_norm: Any = 0.0,
    zero_point_crossover_norm: Any = 0.0,
) -> Dict[str, Any]:
    phase = wrap_turns(phase_turns)
    frequency = clamp01(frequency_norm)
    amplitude = clamp01(amplitude_norm)
    resonance = clamp01(resonance_gate)
    overlap = clamp01(temporal_overlap)
    flux = clamp01(abs(safe_float(flux_norm, 0.0)))
    speed = clamp01(speed_norm)
    coupling = clamp01(coupling_strength)
    spin = clamp01(spin_momentum_score)
    orientation_shear = clamp01(orientation_shear_norm)
    observer_feedback = clamp01(observer_feedback_norm)
    phase_memory = clamp01(phase_memory_norm)
    zero_point_crossover = clamp01(zero_point_crossover_norm)
    vector_x_signed = clamp_signed(vector_x)
    vector_y_signed = clamp_signed(vector_y)
    vector_z_signed = clamp_signed(vector_z)
    vector_energy_norm = clamp01(vector_energy([vector_x_signed, vector_y_signed, vector_z_signed]))
    zero_point_line = phase_zero_line_proximity(phase)
    phase_peak = phase_peak_proximity(phase)
    wavelength_time_norm = clamp01(1.0 - frequency)
    amplitude_excursion_norm = clamp01(amplitude * max(phase_peak, zero_point_line))
    field_time_norm, field_time_weights = _temporal_mix({
        "wavelength_time": wavelength_time_norm,
        "amplitude_excursion": amplitude_excursion_norm,
        "resonance": resonance,
        "overlap": overlap,
    })
    vector_zero_alignment = clamp01(
        1.0
        - (
            abs(abs(vector_x_signed) - zero_point_line)
            + abs(abs(vector_y_signed) - zero_point_line)
            + abs(abs(vector_z_signed) - zero_point_line)
        )
        / 3.0
    )
    path_speed_norm, path_speed_weights = _temporal_mix({
        "input_speed": speed,
        "vector_energy": vector_energy_norm,
        "field_time": field_time_norm,
        "flux": flux,
    })
    phase_alignment_probability, phase_alignment_weights = _temporal_mix({
        "zero_point_line": zero_point_line,
        "phase_peak": phase_peak,
        "overlap": overlap,
        "vector_zero_alignment": vector_zero_alignment,
        "resonance": resonance,
        "observer_feedback": observer_feedback,
    })
    entanglement_probability, entanglement_weights = _temporal_mix({
        "phase_alignment": phase_alignment_probability,
        "vector_alignment": vector_zero_alignment,
        "flux": flux,
        "spin": spin,
        "orientation": clamp01(1.0 - orientation_shear),
        "phase_memory": phase_memory,
    })
    cross_talk_force_norm, cross_talk_weights = _temporal_mix({
        "entanglement": entanglement_probability,
        "vector_energy": vector_energy_norm,
        "flux": flux,
        "zero_point": max(zero_point_crossover, zero_point_line),
        "phase_memory": phase_memory,
        "overlap": overlap,
    })
    intercept_inertia_norm, intercept_inertia_weights = _temporal_mix({
        "field_time": field_time_norm,
        "vector_energy": vector_energy_norm,
        "path_speed": path_speed_norm,
        "coupling": coupling,
        "spin": spin,
        "zero_point": max(zero_point_crossover, zero_point_line),
    })
    temporal_relativity_norm, temporal_relativity_weights = _temporal_mix({
        "field_time": field_time_norm,
        "phase_alignment": phase_alignment_probability,
        "entanglement": entanglement_probability,
        "path_speed": path_speed_norm,
        "intercept_inertia": intercept_inertia_norm,
        "zero_point": max(zero_point_crossover, zero_point_line),
    })
    return {
        "phase_turns": float(phase),
        "zero_point_line_proximity": float(zero_point_line),
        "phase_peak_proximity": float(phase_peak),
        "wavelength_time_norm": float(wavelength_time_norm),
        "amplitude_excursion_norm": float(amplitude_excursion_norm),
        "field_time_norm": float(field_time_norm),
        "vector_energy_norm": float(vector_energy_norm),
        "vector_zero_alignment": float(vector_zero_alignment),
        "path_speed_norm": float(path_speed_norm),
        "phase_alignment_probability": float(phase_alignment_probability),
        "entanglement_probability": float(entanglement_probability),
        "cross_talk_force_norm": float(cross_talk_force_norm),
        "intercept_inertia_norm": float(intercept_inertia_norm),
        "temporal_relativity_norm": float(temporal_relativity_norm),
        "derived_weights": {
            "field_time": field_time_weights,
            "path_speed": path_speed_weights,
            "phase_alignment": phase_alignment_weights,
            "entanglement": entanglement_weights,
            "cross_talk": cross_talk_weights,
            "intercept_inertia": intercept_inertia_weights,
            "temporal_relativity": temporal_relativity_weights,
        },
    }


def dominant_spin_axis(dynamics: Dict[str, Any]) -> tuple[str, float]:
    spin_axes = {
        "x": clamp_signed(dynamics.get("spin_axis_x", 0.0)),
        "y": clamp_signed(dynamics.get("spin_axis_y", 0.0)),
        "z": clamp_signed(dynamics.get("spin_axis_z", 0.0)),
    }
    axis_name = max(spin_axes, key=lambda key: abs(spin_axes[key]))
    return axis_name, float(spin_axes[axis_name])


def _default_schema() -> Dict[str, Any]:
    return {
        "pulse_codes": DEFAULT_PULSE_CODES,
        "deviation_operators": DEFAULT_DEVIATION_OPERATORS,
        "collapse_gates": DEFAULT_COLLAPSE_GATES,
    }


def _schema_section(schema: Dict[str, Any] | None, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(schema or {})
    section = payload.get(key, default)
    if isinstance(section, dict):
        return dict(section)
    return dict(default)


def _window_limits(window: Dict[str, Any], name: str) -> tuple[float, float]:
    entry = dict(window.get(name, {}) or {})
    lower = safe_float(entry.get("min", 0.0), 0.0)
    upper = safe_float(entry.get("max", 1.0), 1.0)
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def normalize_quartet(
    quartet: Dict[str, Any],
    normalized_window: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    window = dict(normalized_window or DEFAULT_NORMALIZED_WINDOW)
    out: Dict[str, float] = {}
    for quartet_key, window_name in QUARTET_TO_WINDOW_KEY.items():
        lower, upper = _window_limits(window, window_name)
        out[quartet_key] = clamp01((safe_float(quartet.get(quartet_key, 0.0), 0.0) - lower) / max(upper - lower, 1.0e-9))
    return out


def denormalize_quartet(
    quartet_norm: Dict[str, Any],
    normalized_window: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    window = dict(normalized_window or DEFAULT_NORMALIZED_WINDOW)
    out: Dict[str, float] = {}
    for quartet_key, window_name in QUARTET_TO_WINDOW_KEY.items():
        lower, upper = _window_limits(window, window_name)
        span = upper - lower
        out[quartet_key] = float(lower + clamp01(quartet_norm.get(quartet_key, 0.0)) * span)
    return out


def compute_axis_field_dynamics(
    frequency_norm: Any,
    amplitude_norm: Any,
    phase_turns: Any,
    resonance_gate: Any,
    temporal_overlap: Any,
    flux_term: Any,
    vector_x: Any = 0.0,
    vector_y: Any = 0.0,
    vector_z: Any = 0.0,
    energy_hint: Any = 0.0,
) -> Dict[str, float]:
    frequency = clamp01(frequency_norm)
    amplitude = clamp01(amplitude_norm)
    phase = clamp01(phase_turns)
    resonance = clamp01(resonance_gate)
    overlap = clamp01(temporal_overlap)
    flux = clamp01(flux_term)
    energy_gate = clamp01(energy_hint)
    x = clamp_signed(vector_x)
    y = clamp_signed(vector_y)
    z = clamp_signed(vector_z)

    joint_gate = clamp01(math.sqrt(max(frequency * amplitude, 0.0)))
    temporal_input_state = build_temporal_relativity_state(
        phase_turns=phase,
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        resonance_gate=resonance,
        temporal_overlap=overlap,
        flux_norm=flux,
        vector_x=x,
        vector_y=y,
        vector_z=z,
        speed_norm=energy_gate,
    )
    axis_scale_x, axis_scale_x_weights = _temporal_mix({
        "field_time": temporal_input_state["field_time_norm"],
        "frequency": frequency,
        "resonance": resonance,
        "vector_alignment": temporal_input_state["vector_zero_alignment"],
        "zero_point": temporal_input_state["zero_point_line_proximity"],
    })
    axis_scale_y, axis_scale_y_weights = _temporal_mix({
        "field_time": temporal_input_state["field_time_norm"],
        "amplitude": amplitude,
        "overlap": overlap,
        "phase_alignment": temporal_input_state["phase_alignment_probability"],
        "vector_alignment": temporal_input_state["vector_zero_alignment"],
    })
    axis_scale_z, axis_scale_z_weights = _temporal_mix({
        "field_time": temporal_input_state["field_time_norm"],
        "joint_gate": joint_gate,
        "flux": flux,
        "entanglement": temporal_input_state["entanglement_probability"],
        "vector_energy": temporal_input_state["vector_energy_norm"],
    })
    axis_resonance = clamp01(
        1.0
        - (
            abs(axis_scale_x - axis_scale_y)
            + abs(axis_scale_y - axis_scale_z)
            + abs(axis_scale_x - axis_scale_z)
        )
        / 3.0
    )

    scaled_x = x * (0.5 + 0.5 * axis_scale_x)
    scaled_y = y * (0.5 + 0.5 * axis_scale_y)
    scaled_z = z * (0.5 + 0.5 * axis_scale_z)
    field_energy = clamp01(vector_energy([scaled_x, scaled_y, scaled_z]) + 0.20 * energy_gate)
    vector_product_x = scaled_y * scaled_z
    vector_product_y = scaled_x * scaled_z
    vector_product_z = scaled_x * scaled_y
    vector_product_norm = clamp01(vector_energy([vector_product_x, vector_product_y, vector_product_z]))
    speed_measure, speed_measure_weights = _temporal_mix({
        "field_energy": field_energy,
        "frequency": frequency,
        "amplitude": amplitude,
        "field_time": temporal_input_state["field_time_norm"],
        "entanglement": temporal_input_state["entanglement_probability"],
    })
    gamma = 1.0 / math.sqrt(max(0.08, 1.0 - (0.92 * speed_measure * speed_measure)))
    relativistic_correlation = clamp01((gamma - 1.0) / 2.5)
    phase_coherence, phase_coherence_weights = _temporal_mix({
        "resonance": resonance,
        "axis_resonance": axis_resonance,
        "overlap": overlap,
        "phase_alignment": temporal_input_state["phase_alignment_probability"],
        "zero_point": temporal_input_state["zero_point_line_proximity"],
    })
    temporal_coupling_moment, temporal_coupling_weights = _temporal_mix({
        "resonance": resonance,
        "axis_resonance": axis_resonance,
        "overlap": overlap,
        "joint_gate": joint_gate,
        "flux": flux,
        "field_time": temporal_input_state["field_time_norm"],
    })
    relative_temporal_coupling, relative_temporal_coupling_weights = _temporal_mix({
        "temporal_coupling": temporal_coupling_moment,
        "phase_coherence": phase_coherence,
        "vector_product": vector_product_norm,
        "joint_gate": joint_gate,
        "entanglement": temporal_input_state["entanglement_probability"],
    })
    temporal_coupling_count = 1.0 + (8.0 * relative_temporal_coupling)

    spin_axis_x = clamp_signed((scaled_y * axis_scale_z) - (scaled_z * axis_scale_y))
    spin_axis_y = clamp_signed((scaled_z * axis_scale_x) - (scaled_x * axis_scale_z))
    spin_axis_z = clamp_signed((scaled_x * axis_scale_y) - (scaled_y * axis_scale_x))
    spin_momentum_score = clamp01(vector_energy([spin_axis_x, spin_axis_y, spin_axis_z]))
    coupling_strength, coupling_strength_weights = _temporal_mix({
        "phase_coherence": phase_coherence,
        "temporal_coupling": relative_temporal_coupling,
        "vector_product": vector_product_norm,
        "spin_momentum": spin_momentum_score,
        "axis_resonance": axis_resonance,
        "temporal_relativity": temporal_input_state["temporal_relativity_norm"],
    })
    inertial_force_norm, inertial_force_weights = _temporal_mix({
        "field_energy": field_energy,
        "vector_product": vector_product_norm,
        "coupling": coupling_strength,
        "coupling_count": clamp01(temporal_coupling_count / 9.0),
        "relativistic": relativistic_correlation,
        "spin_momentum": spin_momentum_score,
        "intercept_inertia": temporal_input_state["intercept_inertia_norm"],
    })
    rotation_scale, rotation_scale_weights = _temporal_mix({
        "coupling": coupling_strength,
        "coupling_count": clamp01(temporal_coupling_count / 9.0),
        "inertial_force": inertial_force_norm,
        "phase_alignment": temporal_input_state["phase_alignment_probability"],
        "path_speed": temporal_input_state["path_speed_norm"],
    })
    spin_rotation_velocity_x = clamp_signed(spin_axis_x * rotation_scale)
    spin_rotation_velocity_y = clamp_signed(spin_axis_y * rotation_scale)
    spin_rotation_velocity_z = clamp_signed(spin_axis_z * rotation_scale)
    orientation_magnitude = math.sqrt(
        max(
            (spin_rotation_velocity_x * spin_rotation_velocity_x)
            + (spin_rotation_velocity_y * spin_rotation_velocity_y)
            + (spin_rotation_velocity_z * spin_rotation_velocity_z),
            0.0,
        )
    )
    if orientation_magnitude <= 1.0e-9:
        spin_orientation_x = 0.0
        spin_orientation_y = 0.0
        spin_orientation_z = 0.0
    else:
        spin_orientation_x = clamp_signed(spin_rotation_velocity_x / orientation_magnitude)
        spin_orientation_y = clamp_signed(spin_rotation_velocity_y / orientation_magnitude)
        spin_orientation_z = clamp_signed(spin_rotation_velocity_z / orientation_magnitude)
    spin_orientation_norm = clamp01(orientation_magnitude)
    orientation_shear_norm = clamp01(
        (
            abs(spin_orientation_x - spin_orientation_y)
            + abs(spin_orientation_y - spin_orientation_z)
            + abs(spin_orientation_x - spin_orientation_z)
        )
        / 3.0
    )
    pre_zero_temporal_state = build_temporal_relativity_state(
        phase_turns=phase,
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        resonance_gate=resonance,
        temporal_overlap=overlap,
        flux_norm=flux,
        vector_x=spin_orientation_x,
        vector_y=spin_orientation_y,
        vector_z=spin_orientation_z,
        speed_norm=speed_measure,
        coupling_strength=coupling_strength,
        spin_momentum_score=spin_momentum_score,
        orientation_shear_norm=orientation_shear_norm,
    )
    light_speed_proxy_x, light_speed_proxy_x_weights = _temporal_mix({
        "path_speed": speed_measure,
        "axis_scale": axis_scale_x,
        "phase_coherence": phase_coherence,
        "coupling": coupling_strength,
        "spin_orientation": abs(spin_orientation_x),
        "temporal_relativity": pre_zero_temporal_state["temporal_relativity_norm"],
    })
    light_speed_proxy_y, light_speed_proxy_y_weights = _temporal_mix({
        "path_speed": speed_measure,
        "axis_scale": axis_scale_y,
        "temporal_coupling": relative_temporal_coupling,
        "coupling": coupling_strength,
        "spin_orientation": abs(spin_orientation_y),
        "temporal_relativity": pre_zero_temporal_state["temporal_relativity_norm"],
    })
    light_speed_proxy_z, light_speed_proxy_z_weights = _temporal_mix({
        "path_speed": speed_measure,
        "axis_scale": axis_scale_z,
        "temporal_coupling": temporal_coupling_moment,
        "inertial_force": inertial_force_norm,
        "spin_orientation": abs(spin_orientation_z),
        "temporal_relativity": pre_zero_temporal_state["temporal_relativity_norm"],
    })
    temporal_coupling_count_norm = clamp01(temporal_coupling_count / 9.0)
    phase_ring_factor_x, phase_ring_factor_x_weights = _temporal_mix({
        "light_speed": light_speed_proxy_x,
        "coupling_count": temporal_coupling_count_norm,
        "rotation": abs(spin_rotation_velocity_x),
        "phase_alignment": pre_zero_temporal_state["phase_alignment_probability"],
        "zero_point": pre_zero_temporal_state["zero_point_line_proximity"],
    })
    phase_ring_factor_y, phase_ring_factor_y_weights = _temporal_mix({
        "light_speed": light_speed_proxy_y,
        "coupling_count": temporal_coupling_count_norm,
        "rotation": abs(spin_rotation_velocity_y),
        "phase_alignment": pre_zero_temporal_state["phase_alignment_probability"],
        "zero_point": pre_zero_temporal_state["zero_point_line_proximity"],
    })
    phase_ring_factor_z, phase_ring_factor_z_weights = _temporal_mix({
        "light_speed": light_speed_proxy_z,
        "coupling_count": temporal_coupling_count_norm,
        "rotation": abs(spin_rotation_velocity_z),
        "entanglement": pre_zero_temporal_state["entanglement_probability"],
        "zero_point": pre_zero_temporal_state["zero_point_line_proximity"],
    })
    phase_ring_x = clamp01(abs(phase - axis_scale_x) * phase_ring_factor_x)
    phase_ring_y = clamp01(abs(phase - axis_scale_y) * phase_ring_factor_y)
    phase_ring_z = clamp01(abs(phase - axis_scale_z) * phase_ring_factor_z)
    phase_ring_density = clamp01((phase_ring_x + phase_ring_y + phase_ring_z) / 3.0)
    phase_ring_balance = clamp01(
        1.0
        - (
            abs(phase_ring_x - phase_ring_y)
            + abs(phase_ring_y - phase_ring_z)
            + abs(phase_ring_x - phase_ring_z)
        )
        / 3.0
    )
    phase_ring_stability, phase_ring_stability_weights = _temporal_mix({
        "phase_ring_balance": phase_ring_balance,
        "orientation_alignment": clamp01(1.0 - orientation_shear_norm),
        "phase_alignment": pre_zero_temporal_state["phase_alignment_probability"],
        "entanglement": pre_zero_temporal_state["entanglement_probability"],
    })
    zero_point_crossover_norm, zero_point_crossover_weights = _temporal_mix({
        "zero_point": pre_zero_temporal_state["zero_point_line_proximity"],
        "phase_coherence": phase_coherence,
        "coupling_count": temporal_coupling_count_norm,
        "phase_ring_stability": phase_ring_stability,
        "phase_peak": pre_zero_temporal_state["phase_peak_proximity"],
    })
    temporal_relativity_state = build_temporal_relativity_state(
        phase_turns=phase,
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        resonance_gate=resonance,
        temporal_overlap=overlap,
        flux_norm=flux,
        vector_x=spin_orientation_x,
        vector_y=spin_orientation_y,
        vector_z=spin_orientation_z,
        speed_norm=speed_measure,
        coupling_strength=coupling_strength,
        spin_momentum_score=spin_momentum_score,
        orientation_shear_norm=orientation_shear_norm,
        zero_point_crossover_norm=zero_point_crossover_norm,
    )
    silicon_atomic_vector_x, silicon_atomic_vector_x_weights = _temporal_mix({
        "light_speed": light_speed_proxy_x,
        "phase_ring": phase_ring_x,
        "coupling": coupling_strength,
        "zero_point": zero_point_crossover_norm,
        "spin_orientation": abs(spin_orientation_x),
        "temporal_relativity": temporal_relativity_state["temporal_relativity_norm"],
    })
    silicon_atomic_vector_y, silicon_atomic_vector_y_weights = _temporal_mix({
        "light_speed": light_speed_proxy_y,
        "phase_ring": phase_ring_y,
        "coupling": coupling_strength,
        "zero_point": zero_point_crossover_norm,
        "spin_orientation": abs(spin_orientation_y),
        "temporal_relativity": temporal_relativity_state["temporal_relativity_norm"],
    })
    silicon_atomic_vector_z, silicon_atomic_vector_z_weights = _temporal_mix({
        "light_speed": light_speed_proxy_z,
        "phase_ring": phase_ring_z,
        "inertial_force": inertial_force_norm,
        "zero_point": zero_point_crossover_norm,
        "spin_orientation": abs(spin_orientation_z),
        "temporal_relativity": temporal_relativity_state["temporal_relativity_norm"],
    })
    phase_field_vector_x, phase_field_vector_x_weights = _temporal_signed_mix({
        "spin_orientation": spin_orientation_x * temporal_relativity_state["phase_alignment_probability"],
        "light_speed_shift": light_speed_proxy_x - 0.5,
        "zero_point_shift": zero_point_crossover_norm - temporal_relativity_state["zero_point_line_proximity"],
        "vector_alignment": x * temporal_relativity_state["vector_zero_alignment"],
    })
    phase_field_vector_y, phase_field_vector_y_weights = _temporal_signed_mix({
        "spin_orientation": spin_orientation_y * temporal_relativity_state["phase_alignment_probability"],
        "light_speed_shift": light_speed_proxy_y - 0.5,
        "zero_point_shift": zero_point_crossover_norm - temporal_relativity_state["zero_point_line_proximity"],
        "vector_alignment": y * temporal_relativity_state["vector_zero_alignment"],
    })
    phase_field_vector_z, phase_field_vector_z_weights = _temporal_signed_mix({
        "spin_orientation": spin_orientation_z * temporal_relativity_state["phase_alignment_probability"],
        "light_speed_shift": light_speed_proxy_z - 0.5,
        "zero_point_shift": zero_point_crossover_norm - temporal_relativity_state["zero_point_line_proximity"],
        "vector_alignment": z * temporal_relativity_state["vector_zero_alignment"],
    })
    temporal_nonlocal_coupling_norm, temporal_nonlocal_coupling_weights = _temporal_mix({
        "phase_coherence": phase_coherence,
        "zero_point": zero_point_crossover_norm,
        "coupling_count": temporal_coupling_count_norm,
        "phase_ring_stability": phase_ring_stability,
        "entanglement": temporal_relativity_state["entanglement_probability"],
    })
    identity_sweep_cluster_norm, identity_sweep_cluster_weights = _temporal_mix({
        "zero_point": zero_point_crossover_norm,
        "coupling_count": temporal_coupling_count_norm,
        "phase_ring_density": phase_ring_density,
        "coupling": coupling_strength,
        "phase_alignment": temporal_relativity_state["phase_alignment_probability"],
    })
    phase_field_energy = clamp01(vector_energy([phase_field_vector_x, phase_field_vector_y, phase_field_vector_z]))
    crosstalk_cluster_norm, crosstalk_cluster_weights = _temporal_mix({
        "orientation_shear": orientation_shear_norm,
        "axis_detune": clamp01(1.0 - axis_resonance),
        "phase_ring_density": phase_ring_density,
        "temporal_coupling": relative_temporal_coupling,
        "phase_field_energy": phase_field_energy,
        "cross_talk_force": temporal_relativity_state["cross_talk_force_norm"],
    })
    inertial_mass_proxy, inertial_mass_proxy_weights = _temporal_mix({
        "field_energy": field_energy,
        "relativistic": relativistic_correlation,
        "spin": spin_momentum_score,
        "temporal_coupling": temporal_coupling_moment,
        "coupling": coupling_strength,
        "inertial_force": inertial_force_norm,
        "nonlocal": temporal_nonlocal_coupling_norm,
        "zero_point": zero_point_crossover_norm,
        "intercept_inertia": temporal_relativity_state["intercept_inertia_norm"],
    })
    return {
        "axis_scale_x": float(axis_scale_x),
        "axis_scale_y": float(axis_scale_y),
        "axis_scale_z": float(axis_scale_z),
        "axis_resonance": float(axis_resonance),
        "phase_turns": float(phase),
        "vector_energy": float(field_energy),
        "vector_product_norm": float(vector_product_norm),
        "speed_measure": float(speed_measure),
        "relativistic_correlation": float(relativistic_correlation),
        "phase_coherence": float(phase_coherence),
        "temporal_coupling_moment": float(temporal_coupling_moment),
        "relative_temporal_coupling": float(relative_temporal_coupling),
        "temporal_coupling_count": float(temporal_coupling_count),
        "spin_axis_x": float(spin_axis_x),
        "spin_axis_y": float(spin_axis_y),
        "spin_axis_z": float(spin_axis_z),
        "spin_momentum_score": float(spin_momentum_score),
        "coupling_strength": float(coupling_strength),
        "inertial_force_norm": float(inertial_force_norm),
        "spin_rotation_velocity_x": float(spin_rotation_velocity_x),
        "spin_rotation_velocity_y": float(spin_rotation_velocity_y),
        "spin_rotation_velocity_z": float(spin_rotation_velocity_z),
        "spin_orientation_x": float(spin_orientation_x),
        "spin_orientation_y": float(spin_orientation_y),
        "spin_orientation_z": float(spin_orientation_z),
        "spin_orientation_norm": float(spin_orientation_norm),
        "orientation_shear_norm": float(orientation_shear_norm),
        "light_speed_proxy_x": float(light_speed_proxy_x),
        "light_speed_proxy_y": float(light_speed_proxy_y),
        "light_speed_proxy_z": float(light_speed_proxy_z),
        "phase_ring_x": float(phase_ring_x),
        "phase_ring_y": float(phase_ring_y),
        "phase_ring_z": float(phase_ring_z),
        "phase_ring_density": float(phase_ring_density),
        "phase_ring_stability": float(phase_ring_stability),
        "zero_point_crossover_norm": float(zero_point_crossover_norm),
        "silicon_atomic_vector_x": float(silicon_atomic_vector_x),
        "silicon_atomic_vector_y": float(silicon_atomic_vector_y),
        "silicon_atomic_vector_z": float(silicon_atomic_vector_z),
        "phase_field_vector_x": float(phase_field_vector_x),
        "phase_field_vector_y": float(phase_field_vector_y),
        "phase_field_vector_z": float(phase_field_vector_z),
        "temporal_nonlocal_coupling_norm": float(temporal_nonlocal_coupling_norm),
        "identity_sweep_cluster_norm": float(identity_sweep_cluster_norm),
        "crosstalk_cluster_norm": float(crosstalk_cluster_norm),
        "inertial_mass_proxy": float(inertial_mass_proxy),
        "temporal_relativity_state": temporal_relativity_state,
        "derived_constants": {
            "axis_scale_x": axis_scale_x_weights,
            "axis_scale_y": axis_scale_y_weights,
            "axis_scale_z": axis_scale_z_weights,
            "speed_measure": speed_measure_weights,
            "phase_coherence": phase_coherence_weights,
            "temporal_coupling_moment": temporal_coupling_weights,
            "relative_temporal_coupling": relative_temporal_coupling_weights,
            "coupling_strength": coupling_strength_weights,
            "inertial_force": inertial_force_weights,
            "rotation_scale": rotation_scale_weights,
            "light_speed_proxy_x": light_speed_proxy_x_weights,
            "light_speed_proxy_y": light_speed_proxy_y_weights,
            "light_speed_proxy_z": light_speed_proxy_z_weights,
            "phase_ring_x": phase_ring_factor_x_weights,
            "phase_ring_y": phase_ring_factor_y_weights,
            "phase_ring_z": phase_ring_factor_z_weights,
            "phase_ring_stability": phase_ring_stability_weights,
            "zero_point_crossover": zero_point_crossover_weights,
            "silicon_atomic_vector_x": silicon_atomic_vector_x_weights,
            "silicon_atomic_vector_y": silicon_atomic_vector_y_weights,
            "silicon_atomic_vector_z": silicon_atomic_vector_z_weights,
            "phase_field_vector_x": phase_field_vector_x_weights,
            "phase_field_vector_y": phase_field_vector_y_weights,
            "phase_field_vector_z": phase_field_vector_z_weights,
            "temporal_nonlocal_coupling": temporal_nonlocal_coupling_weights,
            "identity_sweep_cluster": identity_sweep_cluster_weights,
            "crosstalk_cluster": crosstalk_cluster_weights,
            "inertial_mass_proxy": inertial_mass_proxy_weights,
        },
    }


def compute_live_axis_dynamics(
    quartet: Dict[str, Any],
    phase_turns: Any,
    telemetry: Dict[str, Any] | None = None,
    normalized_window: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    telemetry_state = dict(telemetry or {})
    quartet_norm = normalize_quartet(quartet, normalized_window)
    coherence = clamp01(telemetry_state.get("coherence", telemetry_state.get("predicted_coherence", 0.0)))
    trap_ratio = clamp01(telemetry_state.get("trap_ratio", telemetry_state.get("predicted_trap_ratio", 0.0)))
    temporal_overlap = clamp01(
        telemetry_state.get(
            "temporal_overlap",
            telemetry_state.get("temporal_coupling", telemetry_state.get("prediction_lattice_temporal_coupling_norm", 0.0)),
        )
    )
    flux_factor = clamp01(
        telemetry_state.get(
            "flux",
            telemetry_state.get("predicted_interference", telemetry_state.get("interference", telemetry_state.get("lattice_interference", 0.0))),
        )
    )
    thermal_noise = clamp01(telemetry_state.get("thermal_noise", telemetry_state.get("source_vibration", 0.0)))
    controller_norm = clamp01(telemetry_state.get("controller", telemetry_state.get("subsystem_feedback", 0.0)))
    resonance_gate = clamp01(0.48 * coherence + 0.32 * (1.0 - trap_ratio) + 0.20 * controller_norm)
    vector_x = clamp_signed(quartet_norm["F"] - quartet_norm["V"])
    vector_y = clamp_signed(quartet_norm["A"] - thermal_noise)
    vector_z = clamp_signed(quartet_norm["I"] + flux_factor - quartet_norm["V"])
    energy_hint = clamp01(0.34 * quartet_norm["V"] + 0.33 * quartet_norm["I"] + 0.18 * controller_norm + 0.15 * abs(vector_z))
    return compute_axis_field_dynamics(
        frequency_norm=quartet_norm["F"],
        amplitude_norm=quartet_norm["A"],
        phase_turns=phase_turns,
        resonance_gate=resonance_gate,
        temporal_overlap=temporal_overlap,
        flux_term=flux_factor,
        vector_x=vector_x,
        vector_y=vector_y,
        vector_z=vector_z,
        energy_hint=energy_hint,
    )


def summarize_calibration_result(calibration_result: Dict[str, Any]) -> Dict[str, float]:
    result = dict(calibration_result or {})
    return compute_axis_field_dynamics(
        frequency_norm=result.get("frequency_norm", result.get("calibratedFrequency", result.get("mean_actuation_gain", 0.0))),
        amplitude_norm=result.get("amplitude_norm", result.get("calibratedAmplitude", result.get("mean_pulse_signal", 0.0))),
        phase_turns=result.get("phase_turns", result.get("phase_scale", 0.0)),
        resonance_gate=0.5 * clamp01(result.get("mean_actuation_gain", 0.0)) + 0.5 * clamp01(result.get("recurrence_alignment", 0.0)),
        temporal_overlap=result.get("mean_persistence", 0.0),
        flux_term=clamp01(result.get("mean_leakage", 0.0)) + 0.5 * clamp01(abs(result.get("mean_pulse_signal", 0.0))),
        vector_x=clamp01(result.get("mean_actuation_gain", 0.0)) - clamp01(result.get("mean_leakage", 0.0)),
        vector_y=clamp01(result.get("mean_persistence", 0.0)) - clamp01(result.get("mean_position_radius", 0.0)),
        vector_z=result.get("mean_pulse_signal", 0.0),
        energy_hint=0.5 * clamp01(result.get("mean_pulse_signal", 0.0)) + 0.5 * clamp01(result.get("peak_velocity", 0.0)),
    )


def evaluate_quartet_deviation(metric: Dict[str, Any], deltas: Dict[str, float]) -> float:
    center = safe_float(metric.get("center_value", 0.0), 0.0)
    total = center
    jacobian = dict(metric.get("jacobian", {}) or {})
    hessian_diag = dict(metric.get("hessian_diag", {}) or {})
    hessian_cross = dict(metric.get("hessian_cross", {}) or {})
    for axis_name, delta_value in deltas.items():
        total += safe_float(jacobian.get(axis_name, 0.0), 0.0) * float(delta_value)
    for axis_name, delta_value in deltas.items():
        total += 0.5 * safe_float(hessian_diag.get(axis_name, 0.0), 0.0) * float(delta_value) * float(delta_value)
    for pair_name, coeff in hessian_cross.items():
        if len(str(pair_name)) != 2:
            continue
        lhs = str(pair_name)[0]
        rhs = str(pair_name)[1]
        total += safe_float(coeff, 0.0) * float(deltas.get(lhs, 0.0)) * float(deltas.get(rhs, 0.0))
    return float(total)


def predict_deviation_metrics(
    quartet: Dict[str, Any],
    schema: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    schema_payload = dict(_default_schema())
    schema_payload.update(dict(schema or {}))
    pulse_codes = _schema_section(schema_payload, "pulse_codes", DEFAULT_PULSE_CODES)
    deviation_operators = _schema_section(schema_payload, "deviation_operators", DEFAULT_DEVIATION_OPERATORS)
    deltas = {
        "F": safe_float(quartet.get("F", 0.0), 0.0) - safe_float(pulse_codes.get("f_code", DEFAULT_PULSE_CODES["f_code"]), DEFAULT_PULSE_CODES["f_code"]),
        "A": safe_float(quartet.get("A", 0.0), 0.0) - safe_float(pulse_codes.get("a_code", DEFAULT_PULSE_CODES["a_code"]), DEFAULT_PULSE_CODES["a_code"]),
        "I": safe_float(quartet.get("I", 0.0), 0.0) - safe_float(pulse_codes.get("i_code", DEFAULT_PULSE_CODES["i_code"]), DEFAULT_PULSE_CODES["i_code"]),
        "V": safe_float(quartet.get("V", 0.0), 0.0) - safe_float(pulse_codes.get("v_code", DEFAULT_PULSE_CODES["v_code"]), DEFAULT_PULSE_CODES["v_code"]),
    }
    out: Dict[str, float] = {}
    for metric_name, metric_payload in deviation_operators.items():
        out[str(metric_name)] = float(evaluate_quartet_deviation(dict(metric_payload or {}), deltas))
    return out


def choose_surface_quartet(surface: Dict[str, Any]) -> Dict[str, float]:
    payload = dict(surface or {})
    best_prediction = dict(payload.get("best_prediction", {}) or {})
    candidates = [
        dict(best_prediction.get("adapted_quartet", {}) or {}),
        dict(best_prediction.get("quartet", {}) or {}),
        dict(best_prediction.get("next_pulse_quartet", {}) or {}),
        dict(payload.get("observed_gpu", {}) or {}),
    ]
    for candidate in candidates:
        if all(key in candidate for key in ("F", "A", "I", "V")):
            return {
                "F": float(candidate["F"]),
                "A": float(candidate["A"]),
                "I": float(candidate["I"]),
                "V": float(candidate["V"]),
            }
    return {
        "F": float(DEFAULT_PULSE_CODES["f_code"]),
        "A": float(DEFAULT_PULSE_CODES["a_code"]),
        "I": float(DEFAULT_PULSE_CODES["i_code"]),
        "V": float(DEFAULT_PULSE_CODES["v_code"]),
    }


def build_surface_telemetry(surface: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(surface or {})
    best_prediction = dict(payload.get("best_prediction", {}) or {})
    observed_field = dict(payload.get("observed_field", {}) or {})
    observed_subsystems = dict(payload.get("observed_subsystems", {}) or {})
    lattice_probe = dict(payload.get("prediction_lattice_probe", {}) or {})
    return {
        "coherence": float(best_prediction.get("predicted_coherence", observed_field.get("coherence", 0.0))),
        "trap_ratio": float(best_prediction.get("predicted_trap_ratio", 0.0)),
        "predicted_interference": float(best_prediction.get("predicted_interference", observed_field.get("interference", 0.0))),
        "temporal_coupling": float(best_prediction.get("temporal_coupling", lattice_probe.get("temporal_coupling_norm", 0.0))),
        "thermal_noise": float(observed_field.get("source_vibration", 0.0)),
        "observed_subsystems": {
            "residual": float(observed_subsystems.get("residual", 0.0)),
            "spin": float(observed_subsystems.get("spin", 0.0)),
            "coupling": float(observed_subsystems.get("coupling", 0.0)),
            "controller": float(observed_subsystems.get("controller", 0.0)),
        },
    }


def infer_quartet_granularity(
    surface: Dict[str, Any],
    normalized_window: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    payload = dict(surface or {})
    window = dict(normalized_window or DEFAULT_NORMALIZED_WINDOW)
    predictions = list(payload.get("predictions", []) or [])
    axis_resolution = max(int(payload.get("axis_resolution", 6)), 2)
    granularity: Dict[str, float] = {}
    for quartet_key, window_name in QUARTET_TO_WINDOW_KEY.items():
        values = []
        for entry in predictions:
            quartet = dict(entry.get("quartet", {}) or {})
            if quartet_key in quartet:
                values.append(round(float(quartet[quartet_key]), 12))
        unique_values = sorted(set(values))
        steps = [
            unique_values[index + 1] - unique_values[index]
            for index in range(len(unique_values) - 1)
            if (unique_values[index + 1] - unique_values[index]) > 1.0e-9
        ]
        if steps:
            granularity[quartet_key] = float(min(steps))
            continue
        lower, upper = _window_limits(window, window_name)
        granularity[quartet_key] = float((upper - lower) / float(axis_resolution - 1))
    return granularity


def build_temporal_lattice_state(prediction: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(prediction or {})
    axis_dynamics = dict(payload.get("axis_dynamics", {}) or {})
    axis_scale_x = clamp01(axis_dynamics.get("silicon_atomic_vector_x", axis_dynamics.get("light_speed_proxy_x", axis_dynamics.get("axis_scale_x", 0.0))))
    axis_scale_y = clamp01(axis_dynamics.get("silicon_atomic_vector_y", axis_dynamics.get("light_speed_proxy_y", axis_dynamics.get("axis_scale_y", 0.0))))
    axis_scale_z = clamp01(axis_dynamics.get("silicon_atomic_vector_z", axis_dynamics.get("light_speed_proxy_z", axis_dynamics.get("axis_scale_z", 0.0))))
    phase_next = wrap_turns(payload.get("phase_turns_next", payload.get("phase_turns", 0.0)))
    observer_feedback = clamp01(payload.get("observer_feedback_norm", 0.0))
    predicted_interference = clamp01(payload.get("predicted_interference_norm", 0.0))
    temporal_overlap = clamp01(payload.get("temporal_overlap_norm", 0.0))
    phase_ring_x = clamp01(axis_dynamics.get("phase_ring_x", phase_next))
    phase_ring_y = clamp01(axis_dynamics.get("phase_ring_y", observer_feedback))
    phase_ring_z = clamp01(axis_dynamics.get("phase_ring_z", predicted_interference))
    identity_cluster = clamp01(axis_dynamics.get("identity_sweep_cluster_norm", observer_feedback))
    crosstalk_cluster = clamp01(axis_dynamics.get("crosstalk_cluster_norm", temporal_overlap))
    field_gradients_6d = {
        "g_xx": clamp01(abs(axis_scale_x - phase_ring_x)),
        "g_xy": clamp01(abs(axis_scale_x - axis_scale_y)),
        "g_xz": clamp01(abs(axis_scale_x - axis_scale_z)),
        "g_yy": clamp01(abs(axis_scale_y - phase_ring_y)),
        "g_yz": clamp01(abs(identity_cluster - crosstalk_cluster)),
        "g_zz": clamp01(abs(axis_scale_z - phase_ring_z)),
    }
    lattice_state_9d = [
        float(axis_scale_x),
        float(axis_scale_y),
        float(axis_scale_z),
        float(field_gradients_6d["g_xx"]),
        float(field_gradients_6d["g_xy"]),
        float(field_gradients_6d["g_xz"]),
        float(field_gradients_6d["g_yy"]),
        float(field_gradients_6d["g_yz"]),
        float(field_gradients_6d["g_zz"]),
    ]
    spatial_energy_3d = clamp01((axis_scale_x + axis_scale_y + axis_scale_z) / 3.0)
    phase_ring_energy_3d = clamp01((phase_ring_x + phase_ring_y + phase_ring_z) / 3.0)
    gradient_energy = clamp01(sum(float(value) for value in field_gradients_6d.values()) / 6.0)
    gradient_alignment = clamp01(
        1.0
        - (
            abs(field_gradients_6d["g_xx"] - field_gradients_6d["g_yy"])
            + abs(field_gradients_6d["g_yy"] - field_gradients_6d["g_zz"])
            + abs(field_gradients_6d["g_xy"] - field_gradients_6d["g_yz"])
            + abs(field_gradients_6d["g_xz"] - field_gradients_6d["g_xy"])
        )
        / 4.0
    )
    conservation_9d = clamp01(
        1.0
        - (
            abs(spatial_energy_3d - phase_ring_energy_3d)
            + abs(spatial_energy_3d - gradient_energy)
            + abs(phase_ring_energy_3d - gradient_energy)
        )
        / 3.0
    )
    trajectory_stability = _temporal_mix({
        "gradient_alignment": gradient_alignment,
        "phase_ring_stability": clamp01(axis_dynamics.get("phase_ring_stability", 0.0)),
        "conservation": conservation_9d,
        "orientation_alignment": clamp01(1.0 - axis_dynamics.get("orientation_shear_norm", 0.0)),
        "temporal_relativity": clamp01(dict(axis_dynamics.get("temporal_relativity_state", {}) or {}).get("temporal_relativity_norm", 0.0)),
    })[0]
    return {
        "spatial_axes_3d": {
            "x": float(axis_scale_x),
            "y": float(axis_scale_y),
            "z": float(axis_scale_z),
        },
        "phase_ring_basis": {
            "x": float(phase_ring_x),
            "y": float(phase_ring_y),
            "z": float(phase_ring_z),
        },
        "cluster_state": {
            "identity_sweep": float(identity_cluster),
            "crosstalk": float(crosstalk_cluster),
        },
        "field_gradients_6d": field_gradients_6d,
        "lattice_state_9d": lattice_state_9d,
        "spatial_energy_3d": float(spatial_energy_3d),
        "phase_ring_energy_3d": float(phase_ring_energy_3d),
        "gradient_energy": float(gradient_energy),
        "gradient_alignment": float(gradient_alignment),
        "conservation_9d": float(conservation_9d),
        "trajectory_stability": float(trajectory_stability),
    }


def build_trajectory_state_9d(prediction: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(prediction or {})
    axis_dynamics = dict(payload.get("axis_dynamics", {}) or {})
    lattice_state = dict(payload.get("lattice_state", {}) or build_temporal_lattice_state(payload))
    photonic_x = clamp01(axis_dynamics.get("silicon_atomic_vector_x", axis_dynamics.get("axis_scale_x", 0.0)))
    photonic_y = clamp01(axis_dynamics.get("silicon_atomic_vector_y", axis_dynamics.get("axis_scale_y", 0.0)))
    photonic_z = clamp01(axis_dynamics.get("silicon_atomic_vector_z", axis_dynamics.get("axis_scale_z", 0.0)))
    rotation_x_signed = clamp_signed(axis_dynamics.get("spin_rotation_velocity_x", 0.0))
    rotation_y_signed = clamp_signed(axis_dynamics.get("spin_rotation_velocity_y", 0.0))
    rotation_z_signed = clamp_signed(axis_dynamics.get("spin_rotation_velocity_z", 0.0))
    rotation_x = clamp01(0.5 + 0.5 * rotation_x_signed)
    rotation_y = clamp01(0.5 + 0.5 * rotation_y_signed)
    rotation_z = clamp01(0.5 + 0.5 * rotation_z_signed)
    phase_delta_norm = clamp01(abs(safe_float(payload.get("phase_delta_turns", 0.0), 0.0)) * 4.0)
    reverse_delta_norm = clamp01(abs(safe_float(payload.get("reverse_delta_turns", 0.0), 0.0)) * 4.0)
    transport_drive_norm = clamp01(abs(safe_float(payload.get("transport_drive_norm", 0.0), 0.0)))
    flux_transport_norm = clamp01(abs(safe_float(payload.get("flux_transport_norm", 0.0), 0.0)))
    temporal_overlap = clamp01(payload.get("temporal_overlap_norm", 0.0))
    observer_feedback = clamp01(payload.get("observer_feedback_norm", 0.0))
    zero_point_crossover = clamp01(axis_dynamics.get("zero_point_crossover_norm", 0.0))
    phase_ring_stability = clamp01(axis_dynamics.get("phase_ring_stability", 0.0))
    source_temporal_state = dict(axis_dynamics.get("temporal_relativity_state", {}) or build_temporal_relativity_state(
        phase_turns=payload.get("phase_turns_next", payload.get("phase_turns", 0.0)),
        frequency_norm=photonic_x,
        amplitude_norm=photonic_y,
        resonance_gate=clamp01(axis_dynamics.get("axis_resonance", 0.0)),
        temporal_overlap=temporal_overlap,
        flux_norm=flux_transport_norm,
        vector_x=rotation_x_signed,
        vector_y=rotation_y_signed,
        vector_z=rotation_z_signed,
        speed_norm=clamp01(axis_dynamics.get("speed_measure", 0.0)),
        coupling_strength=clamp01(axis_dynamics.get("coupling_strength", 0.0)),
        spin_momentum_score=clamp01(axis_dynamics.get("spin_momentum_score", 0.0)),
        orientation_shear_norm=clamp01(axis_dynamics.get("orientation_shear_norm", 0.0)),
        observer_feedback_norm=observer_feedback,
        phase_memory_norm=reverse_delta_norm,
        zero_point_crossover_norm=zero_point_crossover,
    ))
    phase_transport_factor_x, phase_transport_factor_x_weights = _temporal_mix({
        "phase_delta": phase_delta_norm,
        "transport_drive": transport_drive_norm,
        "light_speed": clamp01(axis_dynamics.get("light_speed_proxy_x", 0.0)),
        "temporal_overlap": temporal_overlap,
        "phase_alignment": source_temporal_state.get("phase_alignment_probability", 0.0),
    })
    phase_transport_factor_y, phase_transport_factor_y_weights = _temporal_mix({
        "phase_delta": phase_delta_norm,
        "observer_feedback": observer_feedback,
        "light_speed": clamp01(axis_dynamics.get("light_speed_proxy_y", 0.0)),
        "temporal_overlap": temporal_overlap,
        "phase_alignment": source_temporal_state.get("phase_alignment_probability", 0.0),
    })
    phase_transport_factor_z, phase_transport_factor_z_weights = _temporal_mix({
        "phase_delta": phase_delta_norm,
        "flux_transport": flux_transport_norm,
        "light_speed": clamp01(axis_dynamics.get("light_speed_proxy_z", 0.0)),
        "temporal_overlap": temporal_overlap,
        "entanglement": source_temporal_state.get("entanglement_probability", 0.0),
    })
    phase_transport_x = clamp01(abs(safe_float(axis_dynamics.get("phase_field_vector_x", 0.0), 0.0)) * phase_transport_factor_x)
    phase_transport_y = clamp01(abs(safe_float(axis_dynamics.get("phase_field_vector_y", 0.0), 0.0)) * phase_transport_factor_y)
    phase_transport_z = clamp01(abs(safe_float(axis_dynamics.get("phase_field_vector_z", 0.0), 0.0)) * phase_transport_factor_z)
    expansion_x, expansion_x_weights = _temporal_mix({
        "phase_gap": abs(phase_transport_x - photonic_x),
        "zero_point": zero_point_crossover,
        "rotation": abs(rotation_x_signed),
        "temporal_overlap": temporal_overlap,
        "intercept_inertia": source_temporal_state.get("intercept_inertia_norm", 0.0),
    })
    expansion_y, expansion_y_weights = _temporal_mix({
        "phase_gap": abs(phase_transport_y - photonic_y),
        "zero_point": zero_point_crossover,
        "rotation": abs(rotation_y_signed),
        "observer_feedback": observer_feedback,
        "intercept_inertia": source_temporal_state.get("intercept_inertia_norm", 0.0),
    })
    expansion_z, expansion_z_weights = _temporal_mix({
        "phase_gap": abs(phase_transport_z - photonic_z),
        "zero_point": zero_point_crossover,
        "rotation": abs(rotation_z_signed),
        "flux_transport": flux_transport_norm,
        "intercept_inertia": source_temporal_state.get("intercept_inertia_norm", 0.0),
    })
    trajectory_gradients_6d = {
        "g_vx": clamp01(abs(photonic_x - rotation_x)),
        "g_vy": clamp01(abs(photonic_y - rotation_y)),
        "g_vz": clamp01(abs(photonic_z - rotation_z)),
        "g_px": clamp01(abs(photonic_x - phase_transport_x)),
        "g_py": clamp01(abs(photonic_y - phase_transport_y)),
        "g_pz": clamp01(abs(photonic_z - phase_transport_z)),
    }
    trajectory_state_9d = [
        float(photonic_x),
        float(photonic_y),
        float(photonic_z),
        float(trajectory_gradients_6d["g_vx"]),
        float(trajectory_gradients_6d["g_vy"]),
        float(trajectory_gradients_6d["g_vz"]),
        float(trajectory_gradients_6d["g_px"]),
        float(trajectory_gradients_6d["g_py"]),
        float(trajectory_gradients_6d["g_pz"]),
    ]
    photonic_energy = clamp01((photonic_x + photonic_y + photonic_z) / 3.0)
    rotation_energy = clamp01((abs(rotation_x_signed) + abs(rotation_y_signed) + abs(rotation_z_signed)) / 3.0)
    phase_transport_norm = clamp01((phase_transport_x + phase_transport_y + phase_transport_z) / 3.0)
    trajectory_expansion_norm = clamp01((expansion_x + expansion_y + expansion_z) / 3.0)
    trajectory_gradient_energy = clamp01(sum(float(value) for value in trajectory_gradients_6d.values()) / 6.0)
    trajectory_conservation_9d = clamp01(
        1.0
        - (
            abs(photonic_energy - rotation_energy)
            + abs(photonic_energy - phase_transport_norm)
            + abs(trajectory_gradient_energy - clamp01(lattice_state.get("gradient_energy", 0.0)))
        )
        / 3.0
    )
    reverse_causal_flux_coherence, reverse_causal_flux_coherence_weights = _temporal_mix({
        "reverse_delta": reverse_delta_norm,
        "observer_feedback": observer_feedback,
        "flux_alignment": clamp01(1.0 - abs(flux_transport_norm - phase_transport_norm)),
        "trajectory_conservation": trajectory_conservation_9d,
        "phase_ring_stability": phase_ring_stability,
        "entanglement": source_temporal_state.get("entanglement_probability", 0.0),
    })
    temporal_sequence_alignment, temporal_sequence_alignment_weights = _temporal_mix({
        "phase_continuity": clamp01(1.0 - phase_delta_norm),
        "temporal_overlap": temporal_overlap,
        "trajectory_conservation": trajectory_conservation_9d,
        "trajectory_stability": clamp01(lattice_state.get("trajectory_stability", 0.0)),
        "phase_ring_stability": clamp01(axis_dynamics.get("phase_ring_stability", 0.0)),
        "temporal_relativity": source_temporal_state.get("temporal_relativity_norm", 0.0),
    })
    hidden_flux_correction_norm, hidden_flux_correction_weights = _temporal_mix({
        "reverse_causal_flux": reverse_causal_flux_coherence,
        "observer_feedback": observer_feedback,
        "transport_drive": transport_drive_norm,
        "temporal_sequence_alignment": temporal_sequence_alignment,
        "trajectory_conservation": trajectory_conservation_9d,
        "phase_alignment": source_temporal_state.get("phase_alignment_probability", 0.0),
    })
    temporal_occlusion_norm, temporal_occlusion_weights = _temporal_mix({
        "sequence_detune": clamp01(1.0 - temporal_sequence_alignment),
        "trajectory_expansion": trajectory_expansion_norm,
        "reverse_flux_detune": clamp01(1.0 - reverse_causal_flux_coherence),
        "zero_point": zero_point_crossover,
        "cross_talk": source_temporal_state.get("cross_talk_force_norm", 0.0),
    })
    trajectory_noise_reference, trajectory_noise_reference_weights = _temporal_mix({
        "trajectory_detune": clamp01(1.0 - trajectory_conservation_9d),
        "trajectory_expansion": trajectory_expansion_norm,
        "sequence_detune": clamp01(1.0 - temporal_sequence_alignment),
        "zero_point": zero_point_crossover,
        "temporal_occlusion": temporal_occlusion_norm,
        "hidden_flux_margin": clamp01(1.0 - hidden_flux_correction_norm),
    })
    trajectory_q15 = [int(round(clamp01(value) * 32767.0)) for value in trajectory_state_9d]
    trajectory_utf8_text = "PTRJ|" + "|".join(str(value) for value in trajectory_q15)
    trajectory_utf8_hex = trajectory_utf8_text.encode("utf-8").hex()
    return {
        "trajectory_state_9d": trajectory_state_9d,
        "trajectory_gradients_6d": trajectory_gradients_6d,
        "photonic_vector_3d": {"x": float(photonic_x), "y": float(photonic_y), "z": float(photonic_z)},
        "rotation_velocity_3d": {"x": float(rotation_x_signed), "y": float(rotation_y_signed), "z": float(rotation_z_signed)},
        "phase_transport_vector_3d": {"x": float(phase_transport_x), "y": float(phase_transport_y), "z": float(phase_transport_z)},
        "expansion_vector_3d": {"x": float(expansion_x), "y": float(expansion_y), "z": float(expansion_z)},
        "phase_transport_norm": float(phase_transport_norm),
        "trajectory_expansion_norm": float(trajectory_expansion_norm),
        "trajectory_conservation_9d": float(trajectory_conservation_9d),
        "reverse_causal_flux_coherence": float(reverse_causal_flux_coherence),
        "temporal_sequence_alignment": float(temporal_sequence_alignment),
        "hidden_flux_correction_norm": float(hidden_flux_correction_norm),
        "temporal_occlusion_norm": float(temporal_occlusion_norm),
        "trajectory_noise_reference": float(trajectory_noise_reference),
        "derived_constants": {
            "phase_transport_x": phase_transport_factor_x_weights,
            "phase_transport_y": phase_transport_factor_y_weights,
            "phase_transport_z": phase_transport_factor_z_weights,
            "expansion_x": expansion_x_weights,
            "expansion_y": expansion_y_weights,
            "expansion_z": expansion_z_weights,
            "reverse_causal_flux_coherence": reverse_causal_flux_coherence_weights,
            "temporal_sequence_alignment": temporal_sequence_alignment_weights,
            "hidden_flux_correction": hidden_flux_correction_weights,
            "temporal_occlusion": temporal_occlusion_weights,
            "trajectory_noise_reference": trajectory_noise_reference_weights,
        },
        "trajectory_utf8_text": trajectory_utf8_text,
        "trajectory_utf8_hex": trajectory_utf8_hex,
    }


def build_pulse_interference_model(prediction: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(prediction or {})
    axis_dynamics = dict(payload.get("axis_dynamics", {}) or {})
    lattice_state = dict(payload.get("lattice_state", {}) or build_temporal_lattice_state(payload))
    trajectory_state = dict(payload.get("trajectory_state", {}) or build_trajectory_state_9d({**payload, "lattice_state": lattice_state}))
    observed_quartet_norm = dict(payload.get("observed_quartet_norm", {}) or {})
    if not observed_quartet_norm:
        observed_quartet_norm = normalize_quartet(dict(payload.get("observed_quartet", {}) or {}), DEFAULT_NORMALIZED_WINDOW)
    next_quartet_norm = dict(payload.get("next_quartet_norm", {}) or {})
    if not next_quartet_norm:
        next_quartet_norm = normalize_quartet(dict(payload.get("next_pulse_quartet", {}) or {}), DEFAULT_NORMALIZED_WINDOW)

    photonic_vector = dict(trajectory_state.get("photonic_vector_3d", {}) or {})
    phase_transport_vector = dict(trajectory_state.get("phase_transport_vector_3d", {}) or {})
    expansion_vector = dict(trajectory_state.get("expansion_vector_3d", {}) or {})
    rotation_velocity = dict(trajectory_state.get("rotation_velocity_3d", {}) or {})

    pulse_vector_x, pulse_vector_x_weights = _temporal_signed_mix({
        "frequency_delta": safe_float(next_quartet_norm.get("F", 0.0), 0.0) - safe_float(observed_quartet_norm.get("F", 0.0), 0.0),
        "voltage_delta": safe_float(next_quartet_norm.get("V", 0.0), 0.0) - safe_float(observed_quartet_norm.get("V", 0.0), 0.0),
        "expansion_bias": clamp_signed(expansion_vector.get("x", 0.0) - photonic_vector.get("x", 0.0)),
    })
    pulse_vector_y, pulse_vector_y_weights = _temporal_signed_mix({
        "amplitude_delta": safe_float(next_quartet_norm.get("A", 0.0), 0.0) - safe_float(observed_quartet_norm.get("A", 0.0), 0.0),
        "frequency_delta": safe_float(next_quartet_norm.get("F", 0.0), 0.0) - safe_float(observed_quartet_norm.get("F", 0.0), 0.0),
        "expansion_bias": clamp_signed(expansion_vector.get("y", 0.0) - photonic_vector.get("y", 0.0)),
    })
    pulse_vector_z, pulse_vector_z_weights = _temporal_signed_mix({
        "amperage_delta": safe_float(next_quartet_norm.get("I", 0.0), 0.0) - safe_float(observed_quartet_norm.get("I", 0.0), 0.0),
        "drive_delta": safe_float(next_quartet_norm.get("V", 0.0), 0.0) - safe_float(observed_quartet_norm.get("A", 0.0), 0.0),
        "expansion_bias": clamp_signed(expansion_vector.get("z", 0.0) - photonic_vector.get("z", 0.0)),
    })
    pulse_vector_energy_norm = clamp01(vector_energy([pulse_vector_x, pulse_vector_y, pulse_vector_z]))

    observed_wave_energy = clamp01(math.sqrt(max(safe_float(observed_quartet_norm.get("F", 0.0), 0.0) * safe_float(observed_quartet_norm.get("A", 0.0), 0.0), 0.0)))
    observed_drive_energy = clamp01(math.sqrt(max(safe_float(observed_quartet_norm.get("I", 0.0), 0.0) * safe_float(observed_quartet_norm.get("V", 0.0), 0.0), 0.0)))
    next_wave_energy = clamp01(math.sqrt(max(safe_float(next_quartet_norm.get("F", 0.0), 0.0) * safe_float(next_quartet_norm.get("A", 0.0), 0.0), 0.0)))
    next_drive_energy = clamp01(math.sqrt(max(safe_float(next_quartet_norm.get("I", 0.0), 0.0) * safe_float(next_quartet_norm.get("V", 0.0), 0.0), 0.0)))
    pulse_source_temporal_state = build_temporal_relativity_state(
        phase_turns=payload.get("phase_turns_next", payload.get("phase_turns", 0.0)),
        frequency_norm=clamp01(next_quartet_norm.get("F", observed_quartet_norm.get("F", 0.0))),
        amplitude_norm=clamp01(next_quartet_norm.get("A", observed_quartet_norm.get("A", 0.0))),
        resonance_gate=clamp01(axis_dynamics.get("axis_resonance", 0.0)),
        temporal_overlap=clamp01(payload.get("temporal_overlap_norm", 0.0)),
        flux_norm=clamp01(abs(safe_float(payload.get("flux_transport_norm", 0.0), 0.0))),
        vector_x=pulse_vector_x,
        vector_y=pulse_vector_y,
        vector_z=pulse_vector_z,
        speed_norm=clamp01(axis_dynamics.get("speed_measure", 0.0)),
        coupling_strength=clamp01(axis_dynamics.get("coupling_strength", 0.0)),
        spin_momentum_score=clamp01(axis_dynamics.get("spin_momentum_score", 0.0)),
        orientation_shear_norm=clamp01(axis_dynamics.get("orientation_shear_norm", 0.0)),
        observer_feedback_norm=clamp01(payload.get("observer_feedback_norm", 0.0)),
        zero_point_crossover_norm=clamp01(axis_dynamics.get("zero_point_crossover_norm", 0.0)),
    )
    pulse_creation_energy_norm, pulse_creation_energy_weights = _temporal_mix({
        "observed_wave_energy": observed_wave_energy,
        "observed_drive_energy": observed_drive_energy,
        "next_wave_energy": next_wave_energy,
        "next_drive_energy": next_drive_energy,
        "pulse_vector_energy": pulse_vector_energy_norm,
        "field_time": pulse_source_temporal_state.get("field_time_norm", 0.0),
    })

    pulse_trajectory_alignment = clamp01(
        1.0
        - (
            abs(abs(pulse_vector_x) - clamp01(photonic_vector.get("x", 0.0)))
            + abs(abs(pulse_vector_y) - clamp01(photonic_vector.get("y", 0.0)))
            + abs(abs(pulse_vector_z) - clamp01(photonic_vector.get("z", 0.0)))
        )
        / 3.0
    )
    pulse_phase_alignment = clamp01(
        1.0
        - (
            abs(abs(pulse_vector_x) - clamp01(phase_transport_vector.get("x", 0.0)))
            + abs(abs(pulse_vector_y) - clamp01(phase_transport_vector.get("y", 0.0)))
            + abs(abs(pulse_vector_z) - clamp01(phase_transport_vector.get("z", 0.0)))
        )
        / 3.0
    )
    pulse_rotation_alignment = clamp01(
        1.0
        - (
            abs(abs(pulse_vector_x) - clamp01(abs(safe_float(rotation_velocity.get("x", 0.0), 0.0))))
            + abs(abs(pulse_vector_y) - clamp01(abs(safe_float(rotation_velocity.get("y", 0.0), 0.0))))
            + abs(abs(pulse_vector_z) - clamp01(abs(safe_float(rotation_velocity.get("z", 0.0), 0.0))))
        )
        / 3.0
    )

    system_sensitivity_norm, system_sensitivity_weights = _temporal_mix({
        "trajectory_detune": clamp01(1.0 - clamp01(trajectory_state.get("trajectory_conservation_9d", 0.0))),
        "sequence_detune": clamp01(1.0 - clamp01(trajectory_state.get("temporal_sequence_alignment", 0.0))),
        "trajectory_expansion": clamp01(trajectory_state.get("trajectory_expansion_norm", 0.0)),
        "zero_point": clamp01(axis_dynamics.get("zero_point_crossover_norm", 0.0)),
        "reverse_flux_detune": clamp01(1.0 - clamp01(trajectory_state.get("reverse_causal_flux_coherence", 0.0))),
        "stability_detune": clamp01(1.0 - clamp01(lattice_state.get("trajectory_stability", 0.0))),
        "intercept_inertia": pulse_source_temporal_state.get("intercept_inertia_norm", 0.0),
    })
    gpu_pulse_interference_norm, gpu_pulse_interference_weights = _temporal_mix({
        "pulse_creation": pulse_creation_energy_norm,
        "pulse_vector_energy": pulse_vector_energy_norm,
        "trajectory_detune": clamp01(1.0 - pulse_trajectory_alignment),
        "phase_detune": clamp01(1.0 - pulse_phase_alignment),
        "system_sensitivity": system_sensitivity_norm,
        "trajectory_noise": clamp01(trajectory_state.get("trajectory_noise_reference", 0.0)),
        "entanglement": pulse_source_temporal_state.get("entanglement_probability", 0.0),
    })
    environmental_flux_interference_norm, environmental_flux_interference_weights = _temporal_mix({
        "environmental_noise": clamp01(payload.get("environmental_noise_susceptibility", payload.get("thermal_noise_norm", 0.0))),
        "temporal_occlusion": clamp01(trajectory_state.get("temporal_occlusion_norm", 0.0)),
        "flux_transport": clamp01(abs(safe_float(payload.get("flux_transport_norm", 0.0), 0.0))),
        "system_sensitivity": system_sensitivity_norm,
        "hidden_flux_margin": clamp01(1.0 - clamp01(trajectory_state.get("hidden_flux_correction_norm", 0.0))),
        "cross_talk": pulse_source_temporal_state.get("cross_talk_force_norm", 0.0),
    })
    harmonic_trajectory_interference_norm, harmonic_trajectory_interference_weights = _temporal_mix({
        "phase_ring_density": clamp01(axis_dynamics.get("phase_ring_density", 0.0)),
        "identity_sweep_cluster": clamp01(axis_dynamics.get("identity_sweep_cluster_norm", 0.0)),
        "crosstalk_cluster": clamp01(axis_dynamics.get("crosstalk_cluster_norm", 0.0)),
        "trajectory_noise": clamp01(trajectory_state.get("trajectory_noise_reference", 0.0)),
        "rotation_detune": clamp01(1.0 - pulse_rotation_alignment),
        "gpu_pulse_interference": gpu_pulse_interference_norm,
    })
    pulse_backreaction_norm, pulse_backreaction_weights = _temporal_mix({
        "gpu_pulse_interference": gpu_pulse_interference_norm,
        "system_sensitivity": system_sensitivity_norm,
        "environmental_flux_interference": environmental_flux_interference_norm,
        "harmonic_trajectory_interference": harmonic_trajectory_interference_norm,
        "trajectory_expansion": clamp01(trajectory_state.get("trajectory_expansion_norm", 0.0)),
        "intercept_inertia": pulse_source_temporal_state.get("intercept_inertia_norm", 0.0),
    })

    pulse_gradients_6d = {
        "g_tx": clamp01(abs(abs(pulse_vector_x) - clamp01(photonic_vector.get("x", 0.0)))),
        "g_ty": clamp01(abs(abs(pulse_vector_y) - clamp01(photonic_vector.get("y", 0.0)))),
        "g_tz": clamp01(abs(abs(pulse_vector_z) - clamp01(photonic_vector.get("z", 0.0)))),
        "g_px": clamp01(abs(abs(pulse_vector_x) - clamp01(phase_transport_vector.get("x", 0.0)))),
        "g_py": clamp01(abs(abs(pulse_vector_y) - clamp01(phase_transport_vector.get("y", 0.0)))),
        "g_pz": clamp01(abs(abs(pulse_vector_z) - clamp01(phase_transport_vector.get("z", 0.0)))),
    }
    pulse_interference_state_9d = [
        float(abs(pulse_vector_x)),
        float(abs(pulse_vector_y)),
        float(abs(pulse_vector_z)),
        float(pulse_gradients_6d["g_tx"]),
        float(pulse_gradients_6d["g_ty"]),
        float(pulse_gradients_6d["g_tz"]),
        float(pulse_gradients_6d["g_px"]),
        float(pulse_gradients_6d["g_py"]),
        float(pulse_gradients_6d["g_pz"]),
    ]
    pulse_interference_q15 = [int(round(clamp01(value) * 32767.0)) for value in pulse_interference_state_9d]
    pulse_interference_utf8_text = "PINT|" + "|".join(str(value) for value in pulse_interference_q15)
    pulse_interference_utf8_hex = pulse_interference_utf8_text.encode("utf-8").hex()
    return {
        "pulse_vector_3d": {"x": float(pulse_vector_x), "y": float(pulse_vector_y), "z": float(pulse_vector_z)},
        "pulse_interference_state_9d": pulse_interference_state_9d,
        "pulse_interference_q15": pulse_interference_q15,
        "pulse_creation_energy_norm": float(pulse_creation_energy_norm),
        "pulse_vector_energy_norm": float(pulse_vector_energy_norm),
        "pulse_trajectory_alignment": float(pulse_trajectory_alignment),
        "pulse_phase_alignment": float(pulse_phase_alignment),
        "pulse_rotation_alignment": float(pulse_rotation_alignment),
        "gpu_pulse_interference_norm": float(gpu_pulse_interference_norm),
        "environmental_flux_interference_norm": float(environmental_flux_interference_norm),
        "harmonic_trajectory_interference_norm": float(harmonic_trajectory_interference_norm),
        "system_sensitivity_norm": float(system_sensitivity_norm),
        "pulse_backreaction_norm": float(pulse_backreaction_norm),
        "derived_constants": {
            "pulse_vector_x": pulse_vector_x_weights,
            "pulse_vector_y": pulse_vector_y_weights,
            "pulse_vector_z": pulse_vector_z_weights,
            "pulse_creation_energy": pulse_creation_energy_weights,
            "system_sensitivity": system_sensitivity_weights,
            "gpu_pulse_interference": gpu_pulse_interference_weights,
            "environmental_flux_interference": environmental_flux_interference_weights,
            "harmonic_trajectory_interference": harmonic_trajectory_interference_weights,
            "pulse_backreaction": pulse_backreaction_weights,
        },
        "pulse_interference_utf8_text": pulse_interference_utf8_text,
        "pulse_interference_utf8_hex": pulse_interference_utf8_hex,
    }


def build_phase_ring_trace(prediction: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(prediction or {})
    axis_dynamics = dict(payload.get("axis_dynamics", {}) or {})
    lattice_state = dict(payload.get("lattice_state", {}) or build_temporal_lattice_state(payload))
    atomic_x = clamp01(axis_dynamics.get("silicon_atomic_vector_x", axis_dynamics.get("light_speed_proxy_x", axis_dynamics.get("axis_scale_x", 0.0))))
    atomic_y = clamp01(axis_dynamics.get("silicon_atomic_vector_y", axis_dynamics.get("light_speed_proxy_y", axis_dynamics.get("axis_scale_y", 0.0))))
    atomic_z = clamp01(axis_dynamics.get("silicon_atomic_vector_z", axis_dynamics.get("light_speed_proxy_z", axis_dynamics.get("axis_scale_z", 0.0))))
    phase_ring_x = clamp01(axis_dynamics.get("phase_ring_x", 0.0))
    phase_ring_y = clamp01(axis_dynamics.get("phase_ring_y", 0.0))
    phase_ring_z = clamp01(axis_dynamics.get("phase_ring_z", 0.0))
    identity_cluster = clamp01(axis_dynamics.get("identity_sweep_cluster_norm", 0.0))
    crosstalk_cluster = clamp01(axis_dynamics.get("crosstalk_cluster_norm", 0.0))
    zero_point_crossover = clamp01(axis_dynamics.get("zero_point_crossover_norm", 0.0))
    phase_ring_density = clamp01(axis_dynamics.get("phase_ring_density", 0.0))
    phase_ring_stability = clamp01(axis_dynamics.get("phase_ring_stability", 0.0))
    ring_coherence = clamp01(1.0 - axis_dynamics.get("orientation_shear_norm", 0.0))
    phase_ring_trace_9d = [
        float(atomic_x),
        float(atomic_y),
        float(atomic_z),
        float(phase_ring_x),
        float(phase_ring_y),
        float(phase_ring_z),
        float(identity_cluster),
        float(crosstalk_cluster),
        float(zero_point_crossover),
    ]
    phase_ring_q15 = [int(round(clamp01(value) * 32767.0)) for value in phase_ring_trace_9d]
    utf8_trace_text = "PRING|" + "|".join(str(value) for value in phase_ring_q15)
    utf8_trace_hex = utf8_trace_text.encode("utf-8").hex()
    return {
        "phase_ring_trace_9d": phase_ring_trace_9d,
        "phase_ring_q15": phase_ring_q15,
        "phase_ring_density": float(phase_ring_density),
        "phase_ring_stability": float(phase_ring_stability),
        "ring_coherence": float(ring_coherence),
        "identity_sweep_cluster": float(identity_cluster),
        "crosstalk_cluster": float(crosstalk_cluster),
        "zero_point_crossover": float(zero_point_crossover),
        "conservation_9d": float(lattice_state.get("conservation_9d", 0.0)),
        "trajectory_stability": float(lattice_state.get("trajectory_stability", 0.0)),
        "utf8_trace_text": utf8_trace_text,
        "utf8_trace_hex": utf8_trace_hex,
    }


def build_decoded_anchor_vectors(prediction: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(prediction or {})
    observed_payload = dict(payload)
    observed_payload["phase_turns_next"] = float(payload.get("phase_turns", 0.0))
    observed_payload["predicted_interference_norm"] = float(
        clamp01(payload.get("noise_pressure_norm", payload.get("predicted_interference_norm", 0.0)))
    )
    observed_lattice = build_temporal_lattice_state(observed_payload)
    predicted_lattice = dict(payload.get("lattice_state", {}) or build_temporal_lattice_state(payload))
    observed_anchor_vector_9d = [
        clamp01(value) for value in list(observed_lattice.get("lattice_state_9d", []) or [])[:9]
    ]
    predicted_anchor_vector_9d = [
        clamp01(value) for value in list(predicted_lattice.get("lattice_state_9d", []) or [])[:9]
    ]
    stability_weight = _temporal_mix({
        "temporal_overlap": clamp01(payload.get("temporal_overlap_norm", 0.0)),
        "observer_feedback": clamp01(payload.get("observer_feedback_norm", 0.0)),
        "axis_resonance": clamp01(dict(payload.get("axis_dynamics", {}) or {}).get("axis_resonance", 0.0)),
        "coupling": clamp01(dict(payload.get("axis_dynamics", {}) or {}).get("coupling_strength", 0.0)),
        "coupling_count": clamp01(safe_float(dict(payload.get("axis_dynamics", {}) or {}).get("temporal_coupling_count", 1.0), 1.0) / 9.0),
        "spin_orientation": clamp01(dict(payload.get("axis_dynamics", {}) or {}).get("spin_orientation_norm", 0.0)),
        "trajectory_stability": clamp01(predicted_lattice.get("trajectory_stability", 0.0)),
        "conservation": clamp01(predicted_lattice.get("conservation_9d", 0.0)),
        "temporal_relativity": clamp01(dict(dict(payload.get("axis_dynamics", {}) or {}).get("temporal_relativity_state", {}) or {}).get("temporal_relativity_norm", 0.0)),
    })[0]
    stable_anchor_vector_9d = [
        float((observed_value * (1.0 - stability_weight)) + (predicted_value * stability_weight))
        for observed_value, predicted_value in zip(observed_anchor_vector_9d, predicted_anchor_vector_9d)
    ]
    interference_vector_9d = [
        float(abs(predicted_value - stable_value))
        for predicted_value, stable_value in zip(predicted_anchor_vector_9d, stable_anchor_vector_9d)
    ]
    anchor_interference_norm = clamp01(
        sum(float(value) for value in interference_vector_9d) / float(len(interference_vector_9d) or 1)
    )
    stable_anchor_score = clamp01(
        1.0
        - (
            sum(
                abs(float(stable_value) - float(observed_value))
                for stable_value, observed_value in zip(stable_anchor_vector_9d, observed_anchor_vector_9d)
            )
            / float(len(stable_anchor_vector_9d) or 1)
        )
    )
    anchor_interference_alignment = clamp01(
        1.0 - abs(anchor_interference_norm - clamp01(payload.get("predicted_interference_norm", 0.0)))
    )
    phase_ring_trace = build_phase_ring_trace(payload)
    phase_ring_trace_vector = [
        clamp01(value) for value in list(phase_ring_trace.get("phase_ring_trace_9d", []) or [])[:9]
    ]
    phase_ring_trace_weight = _temporal_mix({
        "phase_ring_stability": clamp01(phase_ring_trace.get("phase_ring_stability", 0.0)),
        "phase_ring_density": clamp01(phase_ring_trace.get("phase_ring_density", 0.0)),
        "coupling": clamp01(dict(payload.get("axis_dynamics", {}) or {}).get("coupling_strength", 0.0)),
        "stable_anchor": stable_anchor_score,
        "phase_alignment": clamp01(dict(dict(payload.get("axis_dynamics", {}) or {}).get("temporal_relativity_state", {}) or {}).get("phase_alignment_probability", 0.0)),
    })[0]
    traced_anchor_vector_9d = [
        float((stable_value * (1.0 - phase_ring_trace_weight)) + (trace_value * phase_ring_trace_weight))
        for stable_value, trace_value in zip(stable_anchor_vector_9d, phase_ring_trace_vector)
    ]
    phase_ring_alignment = clamp01(
        1.0
        - (
            sum(abs(float(trace_value) - float(stable_value)) for trace_value, stable_value in zip(traced_anchor_vector_9d, stable_anchor_vector_9d))
            / float(len(traced_anchor_vector_9d) or 1)
        )
    )
    return {
        "observed_anchor_vector_9d": observed_anchor_vector_9d,
        "predicted_anchor_vector_9d": predicted_anchor_vector_9d,
        "stable_anchor_vector_9d": stable_anchor_vector_9d,
        "traced_anchor_vector_9d": traced_anchor_vector_9d,
        "interference_vector_9d": interference_vector_9d,
        "stability_weight": float(stability_weight),
        "anchor_interference_norm": float(anchor_interference_norm),
        "stable_anchor_score": float(stable_anchor_score),
        "anchor_interference_alignment": float(anchor_interference_alignment),
        "phase_ring_trace_weight": float(phase_ring_trace_weight),
        "phase_ring_alignment": float(phase_ring_alignment),
        "phase_ring_trace": phase_ring_trace,
    }


def build_harmonic_noise_model(prediction: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(prediction or {})
    axis_dynamics = dict(payload.get("axis_dynamics", {}) or {})
    lattice_state = dict(payload.get("lattice_state", {}) or build_temporal_lattice_state(payload))
    trajectory_state = dict(payload.get("trajectory_state", {}) or build_trajectory_state_9d(payload))
    pulse_interference = dict(payload.get("pulse_interference", {}) or build_pulse_interference_model({**payload, "lattice_state": lattice_state, "trajectory_state": trajectory_state}))
    anchor_vectors = dict(payload.get("anchor_vectors", {}) or build_decoded_anchor_vectors(payload))
    observed_anchor = list(anchor_vectors.get("observed_anchor_vector_9d", []) or [])
    predicted_anchor = list(anchor_vectors.get("predicted_anchor_vector_9d", []) or [])
    stable_anchor = list(anchor_vectors.get("stable_anchor_vector_9d", []) or [])
    temporal_overlap = clamp01(payload.get("temporal_overlap_norm", 0.0))
    thermal_noise = clamp01(payload.get("thermal_noise_norm", 0.0))
    noise_pressure = clamp01(payload.get("noise_pressure_norm", 0.0))
    axis_resonance = clamp01(axis_dynamics.get("axis_resonance", 0.0))
    spin_score = clamp01(axis_dynamics.get("spin_momentum_score", 0.0))
    phase_coherence = clamp01(axis_dynamics.get("phase_coherence", 0.0))
    coupling_strength = clamp01(axis_dynamics.get("coupling_strength", 0.0))
    inertial_force_norm = clamp01(axis_dynamics.get("inertial_force_norm", 0.0))
    temporal_coupling_count_norm = clamp01(safe_float(axis_dynamics.get("temporal_coupling_count", 1.0), 1.0) / 9.0)
    temporal_nonlocal_coupling_norm = clamp01(axis_dynamics.get("temporal_nonlocal_coupling_norm", 0.0))
    identity_sweep_cluster_norm = clamp01(axis_dynamics.get("identity_sweep_cluster_norm", 0.0))
    crosstalk_cluster_norm = clamp01(axis_dynamics.get("crosstalk_cluster_norm", 0.0))
    zero_point_crossover_norm = clamp01(axis_dynamics.get("zero_point_crossover_norm", 0.0))
    conservation_9d = clamp01(lattice_state.get("conservation_9d", 0.0))
    trajectory_stability_norm, trajectory_stability_weights = _temporal_mix({
        "stable_anchor_score": clamp01(anchor_vectors.get("stable_anchor_score", 0.0)),
        "trajectory_stability": clamp01(lattice_state.get("trajectory_stability", 0.0)),
        "phase_ring_stability": clamp01(dict(anchor_vectors.get("phase_ring_trace", {}) or {}).get("phase_ring_stability", 0.0)),
        "ring_coherence": clamp01(dict(anchor_vectors.get("phase_ring_trace", {}) or {}).get("ring_coherence", 0.0)),
        "conservation": conservation_9d,
        "temporal_relativity": clamp01(dict(trajectory_state.get("derived_constants", {}) or {}).get("temporal_sequence_alignment", {}).get("temporal_relativity", 0.0)),
    })
    trajectory_conservation_9d = clamp01(trajectory_state.get("trajectory_conservation_9d", 0.0))
    reverse_causal_flux_coherence = clamp01(trajectory_state.get("reverse_causal_flux_coherence", 0.0))
    hidden_flux_correction_norm = clamp01(trajectory_state.get("hidden_flux_correction_norm", 0.0))
    temporal_occlusion_norm = clamp01(trajectory_state.get("temporal_occlusion_norm", 0.0))
    trajectory_noise_reference = clamp01(trajectory_state.get("trajectory_noise_reference", 0.0))
    gpu_pulse_interference_norm = clamp01(pulse_interference.get("gpu_pulse_interference_norm", 0.0))
    environmental_flux_interference_norm = clamp01(pulse_interference.get("environmental_flux_interference_norm", 0.0))
    harmonic_trajectory_interference_norm = clamp01(pulse_interference.get("harmonic_trajectory_interference_norm", 0.0))
    system_sensitivity_norm = clamp01(pulse_interference.get("system_sensitivity_norm", 0.0))
    pulse_backreaction_norm = clamp01(pulse_interference.get("pulse_backreaction_norm", 0.0))
    pulse_trajectory_alignment = clamp01(pulse_interference.get("pulse_trajectory_alignment", 0.0))
    pulse_phase_alignment = clamp01(pulse_interference.get("pulse_phase_alignment", 0.0))
    rotation_velocity_norm = clamp01(
        vector_energy([
            axis_dynamics.get("spin_rotation_velocity_x", 0.0),
            axis_dynamics.get("spin_rotation_velocity_y", 0.0),
            axis_dynamics.get("spin_rotation_velocity_z", 0.0),
        ])
    )
    orientation_shear_norm = clamp01(
        (
            abs(safe_float(axis_dynamics.get("spin_orientation_x", 0.0), 0.0) - safe_float(axis_dynamics.get("spin_orientation_y", 0.0), 0.0))
            + abs(safe_float(axis_dynamics.get("spin_orientation_y", 0.0), 0.0) - safe_float(axis_dynamics.get("spin_orientation_z", 0.0), 0.0))
            + abs(safe_float(axis_dynamics.get("spin_orientation_x", 0.0), 0.0) - safe_float(axis_dynamics.get("spin_orientation_z", 0.0), 0.0))
        )
        / 3.0
    )
    pair_specs = [
        ("space_xy", 0, 1),
        ("space_xz", 0, 2),
        ("space_yz", 1, 2),
        ("phase_grad_x", 3, 4),
        ("phase_grad_y", 3, 6),
        ("phase_grad_z", 3, 8),
        ("gradient_cross_xy", 4, 7),
        ("gradient_cross_yz", 7, 8),
    ]
    weighted_couplings: list[Dict[str, Any]] = []
    total_weight = 0.0
    total_reaction = 0.0
    total_coupling_energy = 0.0
    total_inertial_collision = 0.0
    pair_span = float(max(len(stable_anchor) - 1, 1))
    for pair_name, lhs_index, rhs_index in pair_specs:
        observed_delta = abs(float(observed_anchor[lhs_index]) - float(observed_anchor[rhs_index]))
        predicted_delta = abs(float(predicted_anchor[lhs_index]) - float(predicted_anchor[rhs_index]))
        stable_delta = abs(float(stable_anchor[lhs_index]) - float(stable_anchor[rhs_index]))
        pair_alignment = clamp01(1.0 - (abs(float(lhs_index) - float(rhs_index)) / pair_span))
        coupling_weight = _temporal_mix({
            "pair_alignment": pair_alignment,
            "coupling": coupling_strength,
            "coupling_count": temporal_coupling_count_norm,
            "phase_coherence": phase_coherence,
            "stable_delta": stable_delta,
            "axis_resonance": axis_resonance,
            "spin": spin_score,
            "nonlocal": temporal_nonlocal_coupling_norm,
            "zero_point": zero_point_crossover_norm,
            "noise_margin": clamp01(1.0 - noise_pressure),
        })[0]
        inertial_collision_norm = clamp01(coupling_weight * _temporal_mix({
            "inertial_force": inertial_force_norm,
            "coupling_count": temporal_coupling_count_norm,
            "prediction_gap": abs(predicted_delta - stable_delta),
            "rotation_velocity": rotation_velocity_norm,
            "identity_sweep_cluster": identity_sweep_cluster_norm,
            "crosstalk_cluster": crosstalk_cluster_norm,
            "intercept_inertia": clamp01(dict(axis_dynamics.get("temporal_relativity_state", {}) or {}).get("intercept_inertia_norm", 0.0)),
        })[0])
        harmonic_response = _temporal_mix({
            "inertial_collision": inertial_collision_norm,
            "prediction_gap": abs(predicted_delta - stable_delta),
            "observation_gap": abs(observed_delta - stable_delta),
            "orientation_shear": orientation_shear_norm,
            "crosstalk_cluster": crosstalk_cluster_norm,
            "identity_sweep_cluster": identity_sweep_cluster_norm,
            "thermal_noise": thermal_noise,
            "pulse_backreaction": pulse_backreaction_norm,
        })[0]
        weighted_couplings.append({
            "pair": pair_name,
            "coupling_weight": float(coupling_weight),
            "stable_delta": float(stable_delta),
            "inertial_collision_norm": float(inertial_collision_norm),
            "rotation_velocity_norm": float(rotation_velocity_norm),
            "orientation_shear_norm": float(orientation_shear_norm),
            "harmonic_response": float(harmonic_response),
        })
        total_weight += float(coupling_weight)
        total_reaction += float(coupling_weight) * float(harmonic_response)
        total_coupling_energy += float(coupling_weight) * float(stable_delta) * (0.5 + 0.5 * inertial_force_norm)
        total_inertial_collision += float(inertial_collision_norm)
    inertial_collision_norm = clamp01(total_inertial_collision / float(len(weighted_couplings) or 1))
    weighted_coupling_energy = clamp01(total_coupling_energy / float(len(weighted_couplings) or 1))
    zero_point_crossover_probability, zero_point_crossover_probability_weights = _temporal_mix({
        "zero_point": zero_point_crossover_norm,
        "conservation_detune": clamp01(1.0 - conservation_9d),
        "nonlocal": temporal_nonlocal_coupling_norm,
        "trajectory_detune": clamp01(1.0 - trajectory_conservation_9d),
        "temporal_occlusion": temporal_occlusion_norm,
        "trajectory_noise": trajectory_noise_reference,
        "gpu_pulse_interference": gpu_pulse_interference_norm,
        "phase_detune": clamp01(1.0 - pulse_phase_alignment),
    })
    environmental_noise_drive, environmental_noise_drive_weights = _temporal_mix({
        "thermal_noise": thermal_noise,
        "noise_pressure": noise_pressure,
        "crosstalk_cluster": crosstalk_cluster_norm,
        "axis_detune": clamp01(1.0 - axis_resonance),
        "temporal_occlusion": temporal_occlusion_norm,
        "trajectory_noise": trajectory_noise_reference,
        "environmental_flux_interference": environmental_flux_interference_norm,
        "system_sensitivity": system_sensitivity_norm,
    })
    environmental_noise_damping = _temporal_mix({
        "trajectory_stability": trajectory_stability_norm,
        "hidden_flux_correction": hidden_flux_correction_norm,
        "phase_alignment": clamp01(dict(axis_dynamics.get("temporal_relativity_state", {}) or {}).get("phase_alignment_probability", 0.0)),
    })[0]
    environmental_noise_susceptibility = clamp01(environmental_noise_drive * clamp01(1.0 - environmental_noise_damping))
    zero_point_noise_floor, zero_point_noise_floor_weights = _temporal_mix({
        "zero_point_probability": zero_point_crossover_probability,
        "conservation_detune": clamp01(1.0 - conservation_9d),
        "trajectory_detune": clamp01(1.0 - trajectory_conservation_9d),
        "temporal_occlusion": temporal_occlusion_norm,
        "reverse_flux_detune": clamp01(1.0 - reverse_causal_flux_coherence),
        "pulse_backreaction": pulse_backreaction_norm,
    })
    coupling_inertia_pressure, coupling_inertia_pressure_weights = _temporal_mix({
        "weighted_coupling_energy": weighted_coupling_energy,
        "inertial_collision": inertial_collision_norm,
        "coupling_count": temporal_coupling_count_norm,
        "temporal_nonlocal": temporal_nonlocal_coupling_norm,
        "zero_point": zero_point_crossover_norm,
        "intercept_inertia": clamp01(dict(axis_dynamics.get("temporal_relativity_state", {}) or {}).get("intercept_inertia_norm", 0.0)),
    })
    harmonic_noise_reaction_norm, harmonic_noise_reaction_weights = _temporal_mix({
        "weighted_reaction": clamp01(total_reaction / max(total_weight, 1.0e-9)),
        "inertial_collision": inertial_collision_norm,
        "coupling_inertia_pressure": coupling_inertia_pressure,
        "crosstalk_cluster": crosstalk_cluster_norm,
        "identity_sweep_cluster": identity_sweep_cluster_norm,
        "zero_point_noise_floor": zero_point_noise_floor,
        "environmental_noise": environmental_noise_susceptibility,
        "trajectory_noise": trajectory_noise_reference,
        "gpu_pulse_interference": gpu_pulse_interference_norm,
        "environmental_flux_interference": environmental_flux_interference_norm,
        "harmonic_trajectory_interference": harmonic_trajectory_interference_norm,
        "pulse_backreaction": pulse_backreaction_norm,
        "hidden_flux_margin": clamp01(1.0 - hidden_flux_correction_norm),
    })
    harmonic_noise_predictability = clamp01(1.0 - _temporal_mix({
        "harmonic_noise_reaction": harmonic_noise_reaction_norm,
        "zero_point_probability": zero_point_crossover_probability,
        "environmental_noise": environmental_noise_susceptibility,
        "conservation_detune": clamp01(1.0 - conservation_9d),
        "temporal_occlusion": temporal_occlusion_norm,
        "reverse_flux_detune": clamp01(1.0 - reverse_causal_flux_coherence),
        "gpu_pulse_interference": gpu_pulse_interference_norm,
        "system_sensitivity": system_sensitivity_norm,
        "pulse_detune": clamp01(1.0 - pulse_trajectory_alignment),
    })[0])
    unwanted_noise_conditions: list[str] = []
    if zero_point_crossover_probability >= 0.18:
        unwanted_noise_conditions.append("zero_point_noise_floor")
    if conservation_9d <= 0.72:
        unwanted_noise_conditions.append("nine_d_conservation_drift")
    if environmental_noise_susceptibility >= 0.15:
        unwanted_noise_conditions.append("environmental_noise_susceptibility")
    if gpu_pulse_interference_norm >= 0.14:
        unwanted_noise_conditions.append("gpu_pulse_interference")
    if pulse_backreaction_norm >= 0.14:
        unwanted_noise_conditions.append("pulse_backreaction")
    if system_sensitivity_norm >= 0.16 and gpu_pulse_interference_norm >= 0.12:
        unwanted_noise_conditions.append("sensitive_pulse_coupling")
    if pulse_trajectory_alignment <= 0.72:
        unwanted_noise_conditions.append("pulse_trajectory_detuning")
    if temporal_occlusion_norm >= 0.15:
        unwanted_noise_conditions.append("temporal_flux_occlusion")
    if inertial_collision_norm >= 0.14:
        unwanted_noise_conditions.append("inertial_coupling_collision")
    if coupling_inertia_pressure >= 0.16:
        unwanted_noise_conditions.append("coupling_inertia_pressure")
    if rotation_velocity_norm >= 0.18 and orientation_shear_norm >= 0.12:
        unwanted_noise_conditions.append("rotational_orientation_shear")
    if identity_sweep_cluster_norm >= 0.14 and zero_point_crossover_norm >= 0.14:
        unwanted_noise_conditions.append("identity_sweep_crossover")
    if crosstalk_cluster_norm >= 0.14:
        unwanted_noise_conditions.append("cluster_crosstalk")
    if clamp01(anchor_vectors.get("anchor_interference_norm", 0.0)) >= 0.12 and coupling_strength >= 0.55:
        unwanted_noise_conditions.append("anchor_interference_collision")
    if harmonic_noise_reaction_norm >= 0.14:
        unwanted_noise_conditions.append("harmonic_coupling")
    if weighted_coupling_energy >= 0.12 and axis_resonance <= 0.72:
        unwanted_noise_conditions.append("weighted_coupling_resonance")
    return {
        "weighted_couplings": weighted_couplings,
        "weighted_coupling_energy": float(weighted_coupling_energy),
        "inertial_collision_norm": float(inertial_collision_norm),
        "coupling_inertia_pressure": float(coupling_inertia_pressure),
        "rotation_velocity_norm": float(rotation_velocity_norm),
        "orientation_shear_norm": float(orientation_shear_norm),
        "identity_sweep_cluster_norm": float(identity_sweep_cluster_norm),
        "crosstalk_cluster_norm": float(crosstalk_cluster_norm),
        "zero_point_crossover_norm": float(zero_point_crossover_norm),
        "zero_point_crossover_probability": float(zero_point_crossover_probability),
        "zero_point_noise_floor": float(zero_point_noise_floor),
        "conservation_9d": float(conservation_9d),
        "trajectory_conservation_9d": float(trajectory_conservation_9d),
        "trajectory_stability_norm": float(trajectory_stability_norm),
        "reverse_causal_flux_coherence": float(reverse_causal_flux_coherence),
        "hidden_flux_correction_norm": float(hidden_flux_correction_norm),
        "temporal_occlusion_norm": float(temporal_occlusion_norm),
        "trajectory_noise_reference": float(trajectory_noise_reference),
        "gpu_pulse_interference_norm": float(gpu_pulse_interference_norm),
        "environmental_flux_interference_norm": float(environmental_flux_interference_norm),
        "harmonic_trajectory_interference_norm": float(harmonic_trajectory_interference_norm),
        "system_sensitivity_norm": float(system_sensitivity_norm),
        "pulse_backreaction_norm": float(pulse_backreaction_norm),
        "pulse_trajectory_alignment": float(pulse_trajectory_alignment),
        "pulse_phase_alignment": float(pulse_phase_alignment),
        "environmental_noise_susceptibility": float(environmental_noise_susceptibility),
        "harmonic_noise_reaction_norm": float(harmonic_noise_reaction_norm),
        "harmonic_noise_predictability": float(harmonic_noise_predictability),
        "derived_constants": {
            "trajectory_stability": trajectory_stability_weights,
            "zero_point_probability": zero_point_crossover_probability_weights,
            "environmental_noise_drive": environmental_noise_drive_weights,
            "zero_point_noise_floor": zero_point_noise_floor_weights,
            "coupling_inertia_pressure": coupling_inertia_pressure_weights,
            "harmonic_noise_reaction": harmonic_noise_reaction_weights,
        },
        "unwanted_noise_conditions": unwanted_noise_conditions,
        "noise_gate_closed": bool(unwanted_noise_conditions),
    }


def score_temporal_accounting(
    prediction: Dict[str, Any],
    telemetry: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = dict(prediction or {})
    telemetry_state = dict(telemetry or {})
    axis_dynamics = dict(payload.get("axis_dynamics", {}) or {})
    lattice_state = build_temporal_lattice_state(payload)
    trajectory_state = dict(payload.get("trajectory_state", {}) or build_trajectory_state_9d({**payload, "lattice_state": lattice_state}))
    pulse_interference = dict(payload.get("pulse_interference", {}) or build_pulse_interference_model({**payload, "lattice_state": lattice_state, "trajectory_state": trajectory_state}))
    anchor_vectors = dict(payload.get("anchor_vectors", {}) or build_decoded_anchor_vectors(payload))
    harmonic_noise = dict(payload.get("harmonic_noise", {}) or build_harmonic_noise_model({**payload, "anchor_vectors": anchor_vectors}))
    anchor_interference_norm = clamp01(anchor_vectors.get("anchor_interference_norm", payload.get("predicted_interference_norm", 0.0)))
    anchor_interference_alignment = clamp01(anchor_vectors.get("anchor_interference_alignment", 0.0))
    stable_anchor_score = clamp01(anchor_vectors.get("stable_anchor_score", 0.0))
    harmonic_noise_reaction_norm = clamp01(harmonic_noise.get("harmonic_noise_reaction_norm", 0.0))
    inertial_collision_norm = clamp01(harmonic_noise.get("inertial_collision_norm", 0.0))
    orientation_shear_norm = clamp01(harmonic_noise.get("orientation_shear_norm", 0.0))
    conservation_9d = clamp01(lattice_state.get("conservation_9d", 0.0))
    trajectory_stability = clamp01(lattice_state.get("trajectory_stability", 0.0))
    trajectory_conservation_9d = clamp01(trajectory_state.get("trajectory_conservation_9d", 0.0))
    temporal_sequence_alignment = clamp01(trajectory_state.get("temporal_sequence_alignment", 0.0))
    reverse_causal_flux_coherence = clamp01(trajectory_state.get("reverse_causal_flux_coherence", 0.0))
    hidden_flux_correction_norm = clamp01(trajectory_state.get("hidden_flux_correction_norm", 0.0))
    gpu_pulse_interference_norm = clamp01(pulse_interference.get("gpu_pulse_interference_norm", 0.0))
    environmental_flux_interference_norm = clamp01(pulse_interference.get("environmental_flux_interference_norm", 0.0))
    system_sensitivity_norm = clamp01(pulse_interference.get("system_sensitivity_norm", 0.0))
    pulse_backreaction_norm = clamp01(pulse_interference.get("pulse_backreaction_norm", 0.0))
    pulse_trajectory_alignment = clamp01(pulse_interference.get("pulse_trajectory_alignment", 0.0))
    pulse_phase_alignment = clamp01(pulse_interference.get("pulse_phase_alignment", 0.0))
    zero_point_crossover_probability = clamp01(harmonic_noise.get("zero_point_crossover_probability", 0.0))
    environmental_noise_susceptibility = clamp01(harmonic_noise.get("environmental_noise_susceptibility", 0.0))
    coupling_strength = clamp01(axis_dynamics.get("coupling_strength", 0.0))
    inertial_force_norm = clamp01(axis_dynamics.get("inertial_force_norm", 0.0))
    temporal_coupling_count_norm = clamp01(safe_float(axis_dynamics.get("temporal_coupling_count", 1.0), 1.0) / 9.0)
    accounting_temporal_state = build_temporal_relativity_state(
        phase_turns=payload.get("phase_turns_next", payload.get("phase_turns", 0.0)),
        frequency_norm=clamp01(dict(payload.get("observed_quartet_norm", {}) or {}).get("F", 0.0)),
        amplitude_norm=clamp01(dict(payload.get("observed_quartet_norm", {}) or {}).get("A", 0.0)),
        resonance_gate=clamp01(axis_dynamics.get("axis_resonance", 0.0)),
        temporal_overlap=clamp01(payload.get("temporal_overlap_norm", 0.0)),
        flux_norm=clamp01(abs(safe_float(payload.get("flux_transport_norm", 0.0), 0.0))),
        vector_x=clamp_signed(axis_dynamics.get("phase_field_vector_x", 0.0)),
        vector_y=clamp_signed(axis_dynamics.get("phase_field_vector_y", 0.0)),
        vector_z=clamp_signed(axis_dynamics.get("phase_field_vector_z", 0.0)),
        speed_norm=clamp01(axis_dynamics.get("speed_measure", 0.0)),
        coupling_strength=coupling_strength,
        spin_momentum_score=clamp01(axis_dynamics.get("spin_momentum_score", 0.0)),
        orientation_shear_norm=orientation_shear_norm,
        observer_feedback_norm=clamp01(payload.get("observer_feedback_norm", 0.0)),
        phase_memory_norm=clamp01(abs(safe_float(payload.get("phase_memory_delta_turns", 0.0), 0.0)) * 2.0),
        zero_point_crossover_norm=zero_point_crossover_probability,
    )
    sample_period_s = max(
        1.0e-6,
        safe_float(
            telemetry_state.get(
                "sample_period_s",
                telemetry_state.get("request_feedback_time_s", 0.02),
            ),
            0.02,
        ),
    )
    request_feedback_time_s = max(
        1.0e-6,
        safe_float(
            telemetry_state.get(
                "request_feedback_time_s",
                telemetry_state.get("actuation_elapsed_s", sample_period_s),
            ),
            sample_period_s,
        ),
    )
    calculation_time_factor = _temporal_mix({
        "observer_feedback": clamp01(payload.get("observer_feedback_norm", 0.0)),
        "subsystem_feedback": clamp01(payload.get("subsystem_feedback_norm", 0.0)),
        "noise_pressure": clamp01(payload.get("noise_pressure_norm", 0.0)),
        "inertial_mass": clamp01(axis_dynamics.get("inertial_mass_proxy", 0.0)),
        "spin_momentum": clamp01(axis_dynamics.get("spin_momentum_score", 0.0)),
        "coupling": coupling_strength,
        "gpu_pulse_interference": gpu_pulse_interference_norm,
        "system_sensitivity": system_sensitivity_norm,
        "coupling_count": temporal_coupling_count_norm,
        "inertial_force": inertial_force_norm,
        "intercept_inertia": accounting_temporal_state.get("intercept_inertia_norm", 0.0),
    })[0]
    calculation_time_s = max(
        1.0e-6,
        sample_period_s * calculation_time_factor,
    )
    next_feedback_factor = _temporal_mix({
        "temporal_overlap": clamp01(payload.get("temporal_overlap_norm", 0.0)),
        "temporal_coupling": clamp01(axis_dynamics.get("temporal_coupling_moment", 0.0)),
        "anchor_interference": anchor_interference_norm,
        "harmonic_noise": harmonic_noise_reaction_norm,
        "inertial_collision": inertial_collision_norm,
        "pulse_backreaction": pulse_backreaction_norm,
        "environmental_flux_interference": environmental_flux_interference_norm,
        "gradient_energy": clamp01(lattice_state.get("gradient_energy", 0.0)),
        "reverse_gate_margin": clamp01(1.0 - clamp01(payload.get("reverse_gate", 0.0))),
        "field_time": accounting_temporal_state.get("field_time_norm", 0.0),
    })[0]
    next_feedback_time_s = max(
        1.0e-6,
        sample_period_s * next_feedback_factor,
    )
    predicted_cycle_time_s = float(request_feedback_time_s + calculation_time_s + next_feedback_time_s)
    baseline_cycle_time_s = float(request_feedback_time_s + calculation_time_s + sample_period_s)
    time_accuracy_score = clamp01(
        1.0
        - abs(predicted_cycle_time_s - baseline_cycle_time_s)
        / max(predicted_cycle_time_s, baseline_cycle_time_s, 1.0e-6)
    )
    phase_accuracy_score = _temporal_mix({
        "phase_continuity": clamp01(1.0 - abs(safe_float(payload.get("phase_delta_turns", 0.0), 0.0)) * 2.0),
        "gradient_alignment": clamp01(lattice_state.get("gradient_alignment", 0.0)),
        "stable_anchor": stable_anchor_score,
        "coupling": coupling_strength,
        "conservation": conservation_9d,
        "temporal_sequence_alignment": temporal_sequence_alignment,
        "reverse_flux": reverse_causal_flux_coherence,
        "pulse_phase_alignment": pulse_phase_alignment,
        "orientation_alignment": clamp01(1.0 - orientation_shear_norm),
        "phase_alignment": accounting_temporal_state.get("phase_alignment_probability", 0.0),
    })[0]
    lattice_alignment = clamp01(
        1.0 - abs(clamp01(lattice_state.get("gradient_energy", 0.0)) - clamp01(axis_dynamics.get("axis_resonance", 0.0)))
    )
    lattice_accuracy_score = _temporal_mix({
        "gradient_alignment": clamp01(lattice_state.get("gradient_alignment", 0.0)),
        "anchor_interference_alignment": anchor_interference_alignment,
        "conservation": conservation_9d,
        "trajectory_conservation": trajectory_conservation_9d,
        "pulse_trajectory_alignment": pulse_trajectory_alignment,
        "collision_margin": clamp01(1.0 - inertial_collision_norm),
        "lattice_alignment": lattice_alignment,
    })[0]
    predictive_confidence = _temporal_mix({
        "time_accuracy": time_accuracy_score,
        "phase_accuracy": phase_accuracy_score,
        "lattice_accuracy": lattice_accuracy_score,
        "noise_margin": clamp01(1.0 - clamp01(payload.get("noise_pressure_norm", 0.0))),
        "stable_anchor": stable_anchor_score,
        "coupling": coupling_strength,
        "trajectory_stability": trajectory_stability,
        "hidden_flux_correction": hidden_flux_correction_norm,
        "pulse_trajectory_alignment": pulse_trajectory_alignment,
        "pulse_margin": clamp01(1.0 - gpu_pulse_interference_norm),
        "sensitivity_margin": clamp01(1.0 - system_sensitivity_norm),
        "collision_margin": clamp01(1.0 - inertial_collision_norm),
        "temporal_relativity": accounting_temporal_state.get("temporal_relativity_norm", 0.0),
    })[0]
    temporal_accuracy_score = _temporal_mix({
        "time_accuracy": time_accuracy_score,
        "phase_accuracy": phase_accuracy_score,
        "lattice_accuracy": lattice_accuracy_score,
        "stable_anchor": stable_anchor_score,
        "harmonic_margin": clamp01(1.0 - harmonic_noise_reaction_norm),
        "collision_margin": clamp01(1.0 - inertial_collision_norm),
        "field_time": accounting_temporal_state.get("field_time_norm", 0.0),
    })[0]
    return {
        "sample_period_s": float(sample_period_s),
        "request_feedback_time_s": float(request_feedback_time_s),
        "calculation_time_s": float(calculation_time_s),
        "next_feedback_time_s": float(next_feedback_time_s),
        "predicted_cycle_time_s": float(predicted_cycle_time_s),
        "baseline_cycle_time_s": float(baseline_cycle_time_s),
        "time_accuracy_score": float(time_accuracy_score),
        "phase_accuracy_score": float(phase_accuracy_score),
        "lattice_accuracy_score": float(lattice_accuracy_score),
        "temporal_accuracy_score": float(temporal_accuracy_score),
        "predictive_confidence": float(predictive_confidence),
        "lattice_state_9d": list(lattice_state.get("lattice_state_9d", []) or []),
        "spatial_axes_3d": dict(lattice_state.get("spatial_axes_3d", {}) or {}),
        "field_gradients_6d": dict(lattice_state.get("field_gradients_6d", {}) or {}),
        "gradient_energy": float(lattice_state.get("gradient_energy", 0.0)),
        "gradient_alignment": float(lattice_state.get("gradient_alignment", 0.0)),
        "conservation_9d": float(conservation_9d),
        "trajectory_stability": float(trajectory_stability),
        "trajectory_conservation_9d": float(trajectory_conservation_9d),
        "temporal_sequence_alignment": float(temporal_sequence_alignment),
        "reverse_causal_flux_coherence": float(reverse_causal_flux_coherence),
        "hidden_flux_correction_norm": float(hidden_flux_correction_norm),
        "temporal_relativity_state": accounting_temporal_state,
        "gpu_pulse_interference_norm": float(gpu_pulse_interference_norm),
        "environmental_flux_interference_norm": float(environmental_flux_interference_norm),
        "system_sensitivity_norm": float(system_sensitivity_norm),
        "pulse_backreaction_norm": float(pulse_backreaction_norm),
        "pulse_trajectory_alignment": float(pulse_trajectory_alignment),
        "pulse_phase_alignment": float(pulse_phase_alignment),
        "anchor_interference_norm": float(anchor_interference_norm),
        "anchor_interference_alignment": float(anchor_interference_alignment),
        "stable_anchor_score": float(stable_anchor_score),
        "harmonic_noise_reaction_norm": float(harmonic_noise_reaction_norm),
        "inertial_collision_norm": float(inertial_collision_norm),
        "zero_point_crossover_probability": float(zero_point_crossover_probability),
        "environmental_noise_susceptibility": float(environmental_noise_susceptibility),
        "coupling_strength": float(coupling_strength),
        "inertial_force_norm": float(inertial_force_norm),
        "temporal_coupling_count": float(safe_float(axis_dynamics.get("temporal_coupling_count", 1.0), 1.0)),
        "unwanted_noise_conditions": list(harmonic_noise.get("unwanted_noise_conditions", []) or []),
    }


def clamp_quartet_to_window(
    quartet: Dict[str, Any],
    normalized_window: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    normalized = normalize_quartet(quartet, normalized_window)
    return denormalize_quartet(normalized, normalized_window)


def _redistribute_energy_residual(
    corrected_norm: Dict[str, float],
    target_total: float,
) -> Dict[str, float]:
    adjusted = {name: clamp01(value) for name, value in corrected_norm.items()}
    for _ in range(6):
        residual = target_total - sum(adjusted.values())
        if abs(residual) <= 1.0e-9:
            break
        if residual > 0.0:
            candidates = [name for name, value in adjusted.items() if value < 1.0 - 1.0e-9]
        else:
            candidates = [name for name, value in adjusted.items() if value > 1.0e-9]
        if not candidates:
            break
        share = residual / float(len(candidates))
        for name in candidates:
            adjusted[name] = clamp01(adjusted[name] + share)
    return adjusted


def apply_9d_photonic_accounting(
    previous_quartet: Dict[str, Any],
    prediction: Dict[str, Any],
    schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    schema_payload = dict(_default_schema())
    schema_payload.update(dict(schema or {}))
    pulse_codes = _schema_section(schema_payload, "pulse_codes", DEFAULT_PULSE_CODES)
    normalized_window = _schema_section(pulse_codes, "normalized_window", DEFAULT_NORMALIZED_WINDOW)

    before_norm = normalize_quartet(previous_quartet, normalized_window)
    raw_next_norm = {name: clamp01(value) for name, value in dict(prediction.get("next_quartet_norm", {}) or {}).items()}
    if not raw_next_norm:
        raw_next_norm = normalize_quartet(prediction.get("next_pulse_quartet", {}), normalized_window)

    energy_before = float(sum(before_norm.values()))
    energy_after_raw = float(sum(raw_next_norm.values()))
    corrected_norm = dict(raw_next_norm)
    correction_ratio = 1.0
    if energy_after_raw > 1.0e-9:
        correction_ratio = float(energy_before / energy_after_raw)
        corrected_norm = {name: clamp01(value * correction_ratio) for name, value in corrected_norm.items()}
        corrected_norm = _redistribute_energy_residual(corrected_norm, energy_before)

    corrected_quartet = denormalize_quartet(corrected_norm, normalized_window)
    corrected_metrics = predict_deviation_metrics(corrected_quartet, schema_payload)
    corrected_power_proxy = float(math.sqrt(max(corrected_norm.get("I", 0.0) * corrected_norm.get("V", 0.0), 0.0)))
    axis_dynamics = dict(prediction.get("axis_dynamics", {}) or {})

    result = dict(prediction)
    result["next_quartet_norm"] = {name: float(value) for name, value in corrected_norm.items()}
    result["next_pulse_quartet"] = corrected_quartet
    result["predicted_metrics"] = corrected_metrics
    result["power_proxy_norm"] = corrected_power_proxy
    lattice_state = build_temporal_lattice_state(result)
    result["lattice_state"] = lattice_state
    result["trajectory_state"] = build_trajectory_state_9d(result)
    result["pulse_interference"] = build_pulse_interference_model(result)
    result["anchor_vectors"] = build_decoded_anchor_vectors(result)
    result["harmonic_noise"] = build_harmonic_noise_model(result)
    result["temporal_accounting"] = score_temporal_accounting(result)
    accounting_vector_9d = list(
        dict(result.get("anchor_vectors", {}) or {}).get("stable_anchor_vector_9d", lattice_state.get("lattice_state_9d", [])) or []
    )
    result["accounting"] = {
        "energy_before_norm": float(energy_before),
        "energy_after_raw_norm": float(energy_after_raw),
        "energy_after_norm": float(sum(corrected_norm.values())),
        "energy_correction_ratio": float(correction_ratio),
        "conservation_error_norm": float(abs(sum(corrected_norm.values()) - energy_before) / max(energy_before, 1.0e-9)),
        "field_path_speed_rating": float(axis_dynamics.get("speed_measure", 0.0)),
        "vector_path_energy_rating": float(
            clamp01(axis_dynamics.get("vector_energy", 0.0))
            * (0.5 + 0.5 * clamp01(axis_dynamics.get("speed_measure", 0.0)))
        ),
        "power_proxy_norm": corrected_power_proxy,
        "accounting_vector_9d": [float(value) for value in accounting_vector_9d],
        "field_gradients_6d": dict(lattice_state.get("field_gradients_6d", {}) or {}),
        "conservation_9d": float(lattice_state.get("conservation_9d", 0.0)),
        "trajectory_stability": float(lattice_state.get("trajectory_stability", 0.0)),
        "trajectory_conservation_9d": float(dict(result.get("trajectory_state", {}) or {}).get("trajectory_conservation_9d", 0.0)),
        "anchor_interference_norm": float(dict(result.get("anchor_vectors", {}) or {}).get("anchor_interference_norm", 0.0)),
        "stable_anchor_score": float(dict(result.get("anchor_vectors", {}) or {}).get("stable_anchor_score", 0.0)),
        "gpu_pulse_interference_norm": float(dict(result.get("pulse_interference", {}) or {}).get("gpu_pulse_interference_norm", 0.0)),
        "system_sensitivity_norm": float(dict(result.get("pulse_interference", {}) or {}).get("system_sensitivity_norm", 0.0)),
        "pulse_backreaction_norm": float(dict(result.get("pulse_interference", {}) or {}).get("pulse_backreaction_norm", 0.0)),
        "zero_point_crossover_probability": float(dict(result.get("harmonic_noise", {}) or {}).get("zero_point_crossover_probability", 0.0)),
        "zero_point_noise_floor": float(dict(result.get("harmonic_noise", {}) or {}).get("zero_point_noise_floor", 0.0)),
        "environmental_noise_susceptibility": float(dict(result.get("harmonic_noise", {}) or {}).get("environmental_noise_susceptibility", 0.0)),
        "harmonic_noise_reaction_norm": float(dict(result.get("harmonic_noise", {}) or {}).get("harmonic_noise_reaction_norm", 0.0)),
        "unwanted_noise_conditions": list(dict(result.get("harmonic_noise", {}) or {}).get("unwanted_noise_conditions", []) or []),
    }
    return result


def build_kernel_scan_order(
    grid_width: int,
    grid_height: int,
    direction: str,
) -> list[Dict[str, int]]:
    width = max(int(grid_width), 1)
    height = max(int(grid_height), 1)
    order: list[Dict[str, int]] = []
    if direction == "left_to_right":
        for y_coord in range(height):
            for x_coord in range(width):
                order.append({"x": x_coord, "y": y_coord})
    elif direction == "right_to_left":
        for y_coord in range(height):
            for x_coord in range(width - 1, -1, -1):
                order.append({"x": x_coord, "y": y_coord})
    elif direction == "top_to_bottom":
        for x_coord in range(width):
            for y_coord in range(height):
                order.append({"x": x_coord, "y": y_coord})
    elif direction == "bottom_to_top":
        for x_coord in range(width):
            for y_coord in range(height - 1, -1, -1):
                order.append({"x": x_coord, "y": y_coord})
    else:
        raise ValueError(f"Unsupported scan direction: {direction}")
    for index, coord in enumerate(order):
        coord["kernel_index"] = int(index)
    return order


def _clamp_window_value(
    quartet_key: str,
    value: float,
    normalized_window: Dict[str, Any],
) -> float:
    lower, upper = _window_limits(normalized_window, QUARTET_TO_WINDOW_KEY[quartet_key])
    return float(max(lower, min(upper, value)))


def _build_interval_activation_quartet(
    base_quartet: Dict[str, Any],
    granularity: Dict[str, float],
    normalized_window: Dict[str, Any],
    interval_index: int,
    interval_count: int,
    direction_progress_norm: float,
    previous_prediction: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    previous_payload = dict(previous_prediction or {})
    interval_norm = 0.0 if interval_count <= 1 else float(interval_index) / float(interval_count - 1)
    base_frequency = float(base_quartet.get("F", DEFAULT_PULSE_CODES["f_code"]))
    base_amplitude = float(base_quartet.get("A", DEFAULT_PULSE_CODES["a_code"]))
    base_amperage = float(base_quartet.get("I", DEFAULT_PULSE_CODES["i_code"]))
    base_voltage = float(base_quartet.get("V", DEFAULT_PULSE_CODES["v_code"]))
    previous_noise = clamp01(previous_payload.get("noise_pressure_norm", 0.0))
    previous_observer = clamp01(previous_payload.get("observer_feedback_norm", 0.0))
    previous_spin = clamp01(dict(previous_payload.get("axis_dynamics", {}) or {}).get("spin_momentum_score", 0.0))

    frequency_increment = granularity["F"] * (1.0 + interval_index)
    target_frequency = _clamp_window_value(
        "F",
        base_frequency + frequency_increment * (0.70 + 0.30 * direction_progress_norm),
        normalized_window,
    )
    wavelength_before = 1.0 / max(base_frequency, 1.0e-9)
    wavelength_after = 1.0 / max(target_frequency, 1.0e-9)
    wavelength_drop_norm = clamp01((wavelength_before - wavelength_after) / max(wavelength_before, 1.0e-9))

    target_amplitude = _clamp_window_value(
        "A",
        base_amplitude
        + granularity["A"] * (0.18 + 0.24 * interval_norm + 0.14 * previous_observer - 0.12 * previous_noise),
        normalized_window,
    )
    voltage_target = _clamp_window_value(
        "V",
        base_voltage + granularity["V"] * (0.16 + 0.18 * direction_progress_norm + 0.08 * previous_spin),
        normalized_window,
    )
    preserved_power = max(base_amperage * base_voltage, 1.0e-9)
    amperage_target = _clamp_window_value(
        "I",
        preserved_power / max(voltage_target, 1.0e-9),
        normalized_window,
    )
    return {
        "F": float(target_frequency),
        "A": float(target_amplitude),
        "I": float(amperage_target),
        "V": float(voltage_target),
        "wavelength_before": float(wavelength_before),
        "wavelength_after": float(wavelength_after),
        "wavelength_drop_norm": float(wavelength_drop_norm),
    }


def build_transport_prediction(
    quartet: Dict[str, Any],
    phase_turns: Any,
    previous_phase_turns: Any,
    telemetry: Dict[str, Any] | None = None,
    schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    schema_payload = dict(_default_schema())
    schema_payload.update(dict(schema or {}))
    pulse_codes = _schema_section(schema_payload, "pulse_codes", DEFAULT_PULSE_CODES)
    normalized_window = _schema_section(pulse_codes, "normalized_window", DEFAULT_NORMALIZED_WINDOW)
    collapse_gates = _schema_section(schema_payload, "collapse_gates", DEFAULT_COLLAPSE_GATES)
    telemetry_state = dict(telemetry or {})
    subsystem_state = dict(telemetry_state.get("observed_subsystems", telemetry_state.get("subsystems", {})) or {})

    quartet_norm = normalize_quartet(quartet, normalized_window)
    deviation_metrics = predict_deviation_metrics(quartet, schema_payload)
    phase_now = wrap_turns(phase_turns)
    phase_prev = wrap_turns(previous_phase_turns)
    phase_memory = signed_turn_delta(phase_now, phase_prev)

    telemetry_coherence = clamp01(
        telemetry_state.get("coherence", telemetry_state.get("predicted_coherence", deviation_metrics.get("coherence", 0.0)))
    )
    trap_ratio = clamp01(
        telemetry_state.get("trap_ratio", telemetry_state.get("predicted_trap_ratio", deviation_metrics.get("trap", 0.0)))
    )
    predicted_interference = clamp01(
        telemetry_state.get("predicted_interference", telemetry_state.get("interference", telemetry_state.get("lattice_interference", 0.0)))
    )
    temporal_overlap = clamp01(
        telemetry_state.get(
            "temporal_overlap",
            telemetry_state.get("temporal_coupling", telemetry_state.get("prediction_lattice_temporal_coupling_norm", 0.0)),
        )
    )
    thermal_noise = clamp01(telemetry_state.get("thermal_noise", telemetry_state.get("source_vibration", 0.0)))
    subsystem_residual = clamp01(telemetry_state.get("subsystem_residual", subsystem_state.get("residual", 0.0)))
    subsystem_spin = clamp01(telemetry_state.get("subsystem_spin", subsystem_state.get("spin", 0.0)))
    subsystem_coupling = clamp01(telemetry_state.get("subsystem_coupling", subsystem_state.get("coupling", 0.0)))
    subsystem_controller = clamp01(telemetry_state.get("subsystem_controller", subsystem_state.get("controller", 0.0)))
    subsystem_feedback_norm, subsystem_feedback_weights = _temporal_mix({
        "residual": subsystem_residual,
        "spin": subsystem_spin,
        "coupling": subsystem_coupling,
        "controller": subsystem_controller,
    })

    axis_dynamics = compute_live_axis_dynamics(
        quartet=quartet,
        phase_turns=phase_now,
        telemetry={
            "coherence": telemetry_coherence,
            "trap_ratio": trap_ratio,
            "predicted_interference": predicted_interference,
            "temporal_coupling": temporal_overlap,
            "thermal_noise": thermal_noise,
            "controller": subsystem_feedback_norm,
        },
        normalized_window=normalized_window,
    )
    dominant_axis_name, dominant_axis_value = dominant_spin_axis(axis_dynamics)
    axis_temporal_state = dict(axis_dynamics.get("temporal_relativity_state", {}) or {})
    coupling_strength = clamp01(axis_dynamics.get("coupling_strength", 0.0))
    inertial_force_norm = clamp01(axis_dynamics.get("inertial_force_norm", 0.0))
    temporal_coupling_count_norm = clamp01(safe_float(axis_dynamics.get("temporal_coupling_count", 1.0), 1.0) / 9.0)
    phase_ring_stability = clamp01(axis_dynamics.get("phase_ring_stability", 0.0))
    zero_point_noise_floor_norm, zero_point_noise_floor_weights = _temporal_mix({
        "zero_point": axis_dynamics.get("zero_point_crossover_norm", 0.0),
        "phase_ring_detune": clamp01(1.0 - phase_ring_stability),
        "axis_detune": clamp01(1.0 - axis_dynamics.get("axis_resonance", 0.0)),
        "cross_talk": axis_temporal_state.get("cross_talk_force_norm", 0.0),
        "entanglement": axis_temporal_state.get("entanglement_probability", 0.0),
    })
    environmental_noise_drive, environmental_noise_drive_weights = _temporal_mix({
        "thermal_noise": thermal_noise,
        "coherence_detune": clamp01(1.0 - telemetry_coherence),
        "subsystem_feedback": subsystem_feedback_norm,
        "cross_talk": axis_temporal_state.get("cross_talk_force_norm", 0.0),
    })
    environmental_noise_damping, environmental_noise_damping_weights = _temporal_mix({
        "phase_ring_stability": phase_ring_stability,
        "temporal_relativity": axis_temporal_state.get("temporal_relativity_norm", 0.0),
        "vector_alignment": axis_temporal_state.get("vector_zero_alignment", 0.0),
    })
    environmental_noise_susceptibility = clamp01(environmental_noise_drive * clamp01(1.0 - environmental_noise_damping))

    observer_feedback_norm, observer_feedback_weights = _temporal_mix({
        "telemetry_coherence": telemetry_coherence,
        "temporal_overlap": temporal_overlap,
        "axis_resonance": axis_dynamics["axis_resonance"],
        "trap_margin": clamp01(1.0 - trap_ratio),
        "subsystem_feedback": subsystem_feedback_norm,
        "predicted_interference": predicted_interference,
        "coupling": coupling_strength,
        "phase_alignment": axis_temporal_state.get("phase_alignment_probability", 0.0),
    })
    constraint_mutation_norm, constraint_mutation_weights = _temporal_mix({
        "trap_ratio": trap_ratio,
        "thermal_noise": thermal_noise,
        "coherence_detune": clamp01(1.0 - telemetry_coherence),
        "subsystem_feedback": subsystem_feedback_norm,
        "phase_memory": clamp01(min(abs(phase_memory) * 2.0, 1.0)),
        "cross_talk": axis_temporal_state.get("cross_talk_force_norm", 0.0),
    })
    gate_coherence_requirement_base, gate_coherence_weights = _temporal_mix({
        "phase_alignment": axis_temporal_state.get("phase_alignment_probability", 0.0),
        "temporal_relativity": axis_temporal_state.get("temporal_relativity_norm", 0.0),
        "axis_resonance": axis_dynamics.get("axis_resonance", 0.0),
        "phase_coherence": axis_dynamics.get("phase_coherence", 0.0),
        "temporal_overlap": temporal_overlap,
        "trap_margin": clamp01(1.0 - trap_ratio),
    })
    gate_coherence_requirement = clamp01(max(
        gate_coherence_requirement_base,
        clamp01(axis_temporal_state.get("phase_alignment_probability", 0.0)),
        clamp01(axis_dynamics.get("phase_coherence", 0.0)),
        clamp01(axis_dynamics.get("axis_resonance", 0.0)),
    ))
    reverse_mutation_tolerance, reverse_mutation_weights = _temporal_mix({
        "trap_margin": clamp01(1.0 - trap_ratio),
        "phase_ring_stability": phase_ring_stability,
        "observer_feedback": observer_feedback_norm,
        "cross_talk_margin": clamp01(1.0 - axis_temporal_state.get("cross_talk_force_norm", 0.0)),
    })
    reverse_gate = 1.0 if (
        telemetry_coherence >= gate_coherence_requirement
        and constraint_mutation_norm <= reverse_mutation_tolerance
    ) else 0.0

    transport_drive_norm, transport_drive_weights = _temporal_signed_mix({
        "amperage_drive": quartet_norm["I"],
        "frequency_drive": quartet_norm["F"],
        "voltage_drive": quartet_norm["V"],
        "predicted_interference": predicted_interference,
        "subsystem_feedback": subsystem_feedback_norm,
        "observer_feedback": observer_feedback_norm,
        "entanglement": axis_temporal_state.get("entanglement_probability", 0.0),
    })
    flux_transport_norm, flux_transport_weights = _temporal_signed_mix({
        "predicted_interference": predicted_interference,
        "temporal_overlap": temporal_overlap,
        "axis_scale_z": axis_dynamics["axis_scale_z"],
        "subsystem_coupling": subsystem_coupling,
        "flux_alignment": clamp01(1.0 - abs(predicted_interference - axis_temporal_state.get("zero_point_line_proximity", 0.0))),
    })
    pulse_vector_energy_norm = clamp01(
        vector_energy([
            clamp_signed(quartet_norm["F"] - quartet_norm["V"]),
            clamp_signed(quartet_norm["A"] - quartet_norm["I"]),
            clamp_signed((quartet_norm["I"] - quartet_norm["A"]) + (quartet_norm["V"] - quartet_norm["F"])),
        ])
    )
    pulse_creation_energy_norm, pulse_creation_energy_weights = _temporal_mix({
        "wave_energy": clamp01(math.sqrt(max(quartet_norm["F"] * quartet_norm["A"], 0.0))),
        "drive_energy": clamp01(math.sqrt(max(quartet_norm["I"] * quartet_norm["V"], 0.0))),
        "pulse_vector_energy": pulse_vector_energy_norm,
        "axis_vector_energy": clamp01(axis_dynamics["vector_energy"]),
        "field_time": axis_temporal_state.get("field_time_norm", 0.0),
    })
    provisional_system_sensitivity_norm, provisional_sensitivity_weights = _temporal_mix({
        "axis_detune": clamp01(1.0 - axis_dynamics["axis_resonance"]),
        "phase_ring_detune": clamp01(1.0 - phase_ring_stability),
        "zero_point": axis_dynamics["zero_point_crossover_norm"],
        "coupling_count": temporal_coupling_count_norm,
        "phase_memory": clamp01(min(abs(phase_memory) * 2.0, 1.0)),
        "intercept_inertia": axis_temporal_state.get("intercept_inertia_norm", 0.0),
    })
    pulse_excitation_pressure_norm, pulse_excitation_pressure_weights = _temporal_mix({
        "pulse_creation": pulse_creation_energy_norm,
        "pulse_vector_energy": pulse_vector_energy_norm,
        "system_sensitivity": provisional_system_sensitivity_norm,
        "flux_transport": clamp01(abs(flux_transport_norm)),
        "entanglement": axis_temporal_state.get("entanglement_probability", 0.0),
    })
    spin_drive_factor, spin_drive_factor_weights = _temporal_mix({
        "spin_momentum": axis_dynamics["spin_momentum_score"],
        "telemetry_coherence": telemetry_coherence,
        "temporal_overlap": temporal_overlap,
        "phase_alignment": axis_temporal_state.get("phase_alignment_probability", 0.0),
    })
    spin_drive_norm = clamp_signed(dominant_axis_value * spin_drive_factor, limit=1.0)
    inertia_drag_norm, inertia_drag_weights = _temporal_mix({
        "inertial_mass": axis_dynamics["inertial_mass_proxy"],
        "relativistic": axis_dynamics["relativistic_correlation"],
        "coupling": coupling_strength,
        "coupling_count": temporal_coupling_count_norm,
        "inertial_force": inertial_force_norm,
        "intercept_inertia": axis_temporal_state.get("intercept_inertia_norm", 0.0),
    })
    reverse_delta_turns = float(
        reverse_gate
        * phase_memory
        * _temporal_mix({
            "observer_feedback": observer_feedback_norm,
            "temporal_overlap": temporal_overlap,
            "spin_momentum": axis_dynamics["spin_momentum_score"],
            "subsystem_feedback": subsystem_feedback_norm,
            "phase_alignment": axis_temporal_state.get("phase_alignment_probability", 0.0),
        })[0]
    )
    phase_delta_turns = float(_temporal_signed_mix({
        "transport_drive": transport_drive_norm,
        "flux_transport": flux_transport_norm,
        "spin_drive": spin_drive_norm,
        "reverse_delta": reverse_delta_turns,
        "inertia_drag": -inertia_drag_norm,
        "temporal_relativity": axis_temporal_state.get("temporal_relativity_norm", 0.0),
    }, limit=1.0)[0] / 8.0)
    phase_next = wrap_turns(phase_now + phase_delta_turns)
    noise_pressure_norm, noise_pressure_weights = _temporal_mix({
        "predicted_interference": predicted_interference,
        "constraint_mutation": constraint_mutation_norm,
        "phase_memory": clamp01(min(abs(phase_memory) * 2.0, 1.0)),
        "subsystem_feedback": subsystem_feedback_norm,
        "coupling": coupling_strength,
        "inertial_force": inertial_force_norm,
        "pulse_excitation": pulse_excitation_pressure_norm,
        "system_sensitivity": provisional_system_sensitivity_norm,
        "zero_point_noise_floor": zero_point_noise_floor_norm,
        "environmental_noise": environmental_noise_susceptibility,
        "cross_talk": axis_temporal_state.get("cross_talk_force_norm", 0.0),
    })

    redistribution_weights = {
        "F": _temporal_mix({
            "axis_scale_x": axis_dynamics["axis_scale_x"],
            "observer_feedback": observer_feedback_norm,
            "temporal_overlap": temporal_overlap,
            "field_time": axis_temporal_state.get("field_time_norm", 0.0),
        })[0],
        "A": _temporal_mix({
            "axis_scale_y": axis_dynamics["axis_scale_y"],
            "telemetry_coherence": telemetry_coherence,
            "trap_margin": clamp01(1.0 - trap_ratio),
            "phase_alignment": axis_temporal_state.get("phase_alignment_probability", 0.0),
        })[0],
        "I": _temporal_mix({
            "axis_scale_z": axis_dynamics["axis_scale_z"],
            "spin_momentum": axis_dynamics["spin_momentum_score"],
            "subsystem_controller": subsystem_controller,
            "entanglement": axis_temporal_state.get("entanglement_probability", 0.0),
        })[0],
        "V": _temporal_mix({
            "axis_resonance": axis_dynamics["axis_resonance"],
            "inertial_mass": axis_dynamics["inertial_mass_proxy"],
            "subsystem_residual": subsystem_residual,
            "intercept_inertia": axis_temporal_state.get("intercept_inertia_norm", 0.0),
        })[0],
    }
    redistribution_total = max(sum(redistribution_weights.values()), 1.0e-9)
    noise_redistribution_norm = {
        name: ((weight / redistribution_total) - 0.25) * noise_pressure_norm * 0.42
        for name, weight in redistribution_weights.items()
    }

    phase_push = clamp_signed(phase_delta_turns * 4.0, limit=1.0)
    reverse_push = clamp_signed(reverse_delta_turns * 4.0, limit=1.0)
    next_quartet_norm = {
        "F": clamp01(quartet_norm["F"] + _temporal_signed_mix({
            "phase_push": phase_push,
            "reverse_push": reverse_push,
            "redistribution": noise_redistribution_norm["F"],
            "noise_backoff": -noise_pressure_norm,
            "path_speed": axis_temporal_state.get("path_speed_norm", 0.0),
        })[0]),
        "A": clamp01(quartet_norm["A"] + _temporal_signed_mix({
            "observer_feedback": observer_feedback_norm,
            "axis_resonance": axis_dynamics["axis_resonance"],
            "trap_backoff": -trap_ratio,
            "redistribution": noise_redistribution_norm["A"],
            "phase_alignment": axis_temporal_state.get("phase_alignment_probability", 0.0),
        })[0]),
        "I": clamp01(quartet_norm["I"] + _temporal_signed_mix({
            "predicted_interference": predicted_interference,
            "spin_momentum": axis_dynamics["spin_momentum_score"],
            "inertia_drag": -inertia_drag_norm,
            "redistribution": noise_redistribution_norm["I"],
            "pulse_pressure": pulse_excitation_pressure_norm,
        })[0]),
        "V": clamp01(quartet_norm["V"] + _temporal_signed_mix({
            "temporal_overlap": temporal_overlap,
            "subsystem_feedback": subsystem_feedback_norm,
            "reverse_gate": reverse_gate,
            "thermal_backoff": -thermal_noise,
            "redistribution": noise_redistribution_norm["V"],
        })[0]),
    }

    current_total = sum(quartet_norm.values())
    desired_total_delta = _temporal_signed_mix({
        "observer_feedback": observer_feedback_norm,
        "transport_drive": transport_drive_norm,
        "noise_backoff": -noise_pressure_norm,
        "pulse_pressure": -pulse_excitation_pressure_norm,
        "entanglement": axis_temporal_state.get("entanglement_probability", 0.0),
    }, limit=1.0)[0]
    desired_total = max(
        0.0,
        min(
            4.0,
            current_total + desired_total_delta,
        ),
    )
    total_adjust = (sum(next_quartet_norm.values()) - desired_total) / 4.0
    for axis_name in next_quartet_norm:
        next_quartet_norm[axis_name] = clamp01(next_quartet_norm[axis_name] - total_adjust)

    current_power_proxy = math.sqrt(max(quartet_norm["I"] * quartet_norm["V"], 0.0))
    next_power_proxy = math.sqrt(max(next_quartet_norm["I"] * next_quartet_norm["V"], 0.0))
    power_retain = clamp_signed(current_power_proxy - next_power_proxy, limit=0.25)
    next_quartet_norm["I"] = clamp01(next_quartet_norm["I"] + 0.5 * power_retain)
    next_quartet_norm["V"] = clamp01(next_quartet_norm["V"] + 0.5 * power_retain)
    next_quartet = denormalize_quartet(next_quartet_norm, normalized_window)
    next_metrics = predict_deviation_metrics(next_quartet, schema_payload)

    result = {
        "observed_quartet": {
            "F": float(safe_float(quartet.get("F", 0.0), 0.0)),
            "A": float(safe_float(quartet.get("A", 0.0), 0.0)),
            "I": float(safe_float(quartet.get("I", 0.0), 0.0)),
            "V": float(safe_float(quartet.get("V", 0.0), 0.0)),
        },
        "observed_quartet_norm": {name: float(value) for name, value in quartet_norm.items()},
        "phase_turns": float(phase_now),
        "phase_turns_previous": float(phase_prev),
        "phase_turns_next": float(phase_next),
        "phase_memory_delta_turns": float(phase_memory),
        "phase_delta_turns": float(phase_delta_turns),
        "reverse_delta_turns": float(reverse_delta_turns),
        "reverse_gate": float(reverse_gate),
        "transport_drive_norm": float(transport_drive_norm),
        "flux_transport_norm": float(flux_transport_norm),
        "observer_feedback_norm": float(observer_feedback_norm),
        "constraint_mutation_norm": float(constraint_mutation_norm),
        "noise_pressure_norm": float(noise_pressure_norm),
        "predicted_interference_norm": float(predicted_interference),
        "temporal_overlap_norm": float(temporal_overlap),
        "thermal_noise_norm": float(thermal_noise),
        "pulse_vector_energy_norm": float(pulse_vector_energy_norm),
        "pulse_creation_energy_norm": float(pulse_creation_energy_norm),
        "pulse_excitation_pressure_norm": float(pulse_excitation_pressure_norm),
        "provisional_system_sensitivity_norm": float(provisional_system_sensitivity_norm),
        "temporal_relativity_state": axis_temporal_state,
        "derived_constants": {
            "subsystem_feedback": subsystem_feedback_weights,
            "zero_point_noise_floor": zero_point_noise_floor_weights,
            "environmental_noise_drive": environmental_noise_drive_weights,
            "environmental_noise_damping": environmental_noise_damping_weights,
            "observer_feedback": observer_feedback_weights,
            "constraint_mutation": constraint_mutation_weights,
            "gate_coherence": gate_coherence_weights,
            "reverse_mutation": reverse_mutation_weights,
            "transport_drive": transport_drive_weights,
            "flux_transport": flux_transport_weights,
            "pulse_creation_energy": pulse_creation_energy_weights,
            "provisional_system_sensitivity": provisional_sensitivity_weights,
            "pulse_excitation_pressure": pulse_excitation_pressure_weights,
            "spin_drive": spin_drive_factor_weights,
            "inertia_drag": inertia_drag_weights,
            "noise_pressure": noise_pressure_weights,
        },
        "zero_point_noise_floor_norm": float(zero_point_noise_floor_norm),
        "environmental_noise_susceptibility": float(environmental_noise_susceptibility),
        "subsystem_feedback_norm": float(subsystem_feedback_norm),
        "dominant_spin_axis": dominant_axis_name,
        "dominant_spin_value": float(dominant_axis_value),
        "axis_dynamics": axis_dynamics,
        "noise_redistribution_norm": {name: float(value) for name, value in noise_redistribution_norm.items()},
        "next_quartet_norm": {name: float(value) for name, value in next_quartet_norm.items()},
        "next_pulse_quartet": next_quartet,
        "deviation_metrics": deviation_metrics,
        "predicted_metrics": next_metrics,
        "power_proxy_norm": float(math.sqrt(max(next_quartet_norm["I"] * next_quartet_norm["V"], 0.0))),
    }
    result["lattice_state"] = build_temporal_lattice_state(result)
    result["trajectory_state"] = build_trajectory_state_9d(result)
    result["pulse_interference"] = build_pulse_interference_model(result)
    result["anchor_vectors"] = build_decoded_anchor_vectors(result)
    result["harmonic_noise"] = build_harmonic_noise_model(result)
    result["anchor_interference_norm"] = float(dict(result.get("anchor_vectors", {}) or {}).get("anchor_interference_norm", 0.0))
    result["harmonic_noise_reaction_norm"] = float(dict(result.get("harmonic_noise", {}) or {}).get("harmonic_noise_reaction_norm", 0.0))
    result["temporal_accounting"] = score_temporal_accounting(result, telemetry_state)
    return result


def encode_photonic_identity(prediction: Dict[str, Any]) -> Dict[str, Any]:
    lattice_state = dict(prediction.get("lattice_state", {}) or {})
    anchor_vectors = dict(prediction.get("anchor_vectors", {}) or {})
    phase_ring_trace = dict(anchor_vectors.get("phase_ring_trace", {}) or build_phase_ring_trace(prediction))
    dominant_spin_axis_name = str(prediction.get("dominant_spin_axis", "x"))
    identity_vector = [
        clamp01(value)
        for value in list(
            anchor_vectors.get(
                "traced_anchor_vector_9d",
                lattice_state.get("lattice_state_9d", build_temporal_lattice_state(prediction).get("lattice_state_9d", [])),
            )
            or []
        )[:9]
    ]
    spectra_q15 = [int(round(clamp01(value) * 32767.0)) for value in identity_vector]
    identity_source = str(phase_ring_trace.get("utf8_trace_text", "")) + "|" + dominant_spin_axis_name
    digest = hashlib.sha256(identity_source.encode("utf-8")).digest()
    trajectory_spectral_id = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return {
        "photonic_identity": "PID-%016X" % trajectory_spectral_id,
        "trajectory_spectral_id_u64": int(trajectory_spectral_id),
        "spectra_axes": [
            "atomic_x",
            "atomic_y",
            "atomic_z",
            "ring_x",
            "ring_y",
            "ring_z",
            "identity_cluster",
            "crosstalk_cluster",
            "zero_point_crossover",
        ],
        "spectra_q15": spectra_q15,
        "dominant_spin_axis": dominant_spin_axis_name,
        "lattice_state_9d": identity_vector,
        "decoded_anchor_vector_9d": list(anchor_vectors.get("predicted_anchor_vector_9d", []) or []),
        "stable_anchor_vector_9d": list(anchor_vectors.get("stable_anchor_vector_9d", identity_vector) or []),
        "traced_anchor_vector_9d": list(anchor_vectors.get("traced_anchor_vector_9d", identity_vector) or []),
        "anchor_interference_norm": float(anchor_vectors.get("anchor_interference_norm", 0.0)),
        "phase_ring_trace_9d": list(phase_ring_trace.get("phase_ring_trace_9d", []) or []),
        "phase_ring_utf8": str(phase_ring_trace.get("utf8_trace_text", "")),
        "phase_ring_utf8_hex": str(phase_ring_trace.get("utf8_trace_hex", "")),
    }


def _u32_word(value: Any, scale: float = 1.0, signed: bool = False) -> int:
    numeric = safe_float(value, 0.0)
    integer = int(round(numeric * float(scale)))
    if signed:
        integer &= 0xFFFFFFFF
    else:
        integer = max(0, integer)
    return int(integer) & 0xFFFFFFFF


def derive_temporal_constant_set(
    dynamics: Dict[str, Any],
    telemetry: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = dict(dynamics or {})
    telemetry_state = dict(telemetry or {})
    coherence = clamp01(telemetry_state.get("coherence", metrics.get("phase_coherence", metrics.get("axis_resonance", 0.0))))
    entropy = clamp01(telemetry_state.get("entropy", metrics.get("crosstalk_cluster_norm", 0.0)))
    overlap = clamp01(telemetry_state.get("temporal_overlap", metrics.get("relative_temporal_coupling", metrics.get("temporal_coupling_moment", 0.0))))
    observer_feedback = clamp01(telemetry_state.get("observer_feedback", metrics.get("identity_sweep_cluster_norm", 0.0)))
    noise_gate = clamp01(telemetry_state.get("noise_gate", metrics.get("crosstalk_cluster_norm", 0.0)))
    phase_delta = clamp01(abs(safe_float(telemetry_state.get("phase_delta_turns", metrics.get("gpu_pulse_phase_effect", 0.0)), 0.0)) * 2.0)
    vector_energy_norm = clamp01(metrics.get("vector_energy", 0.0))
    inertia = clamp01(metrics.get("inertial_mass_proxy", 0.0))
    spin = clamp01(metrics.get("spin_momentum_score", 0.0))
    phase_ring = clamp01(metrics.get("phase_ring_density", 0.0))
    phase_ring_stability = clamp01(metrics.get("phase_ring_stability", 0.0))
    zero_point = clamp01(metrics.get("zero_point_crossover_norm", 0.0))
    temporal_relativity = clamp01(dict(metrics.get("temporal_relativity_state", {}) or {}).get("temporal_relativity_norm", metrics.get("temporal_relativity_norm", 0.0)))
    interception = clamp01(dict(metrics.get("temporal_relativity_state", {}) or {}).get("intercept_inertia_norm", metrics.get("resonant_interception_inertia", 0.0)))
    nonlocal_coupling = clamp01(metrics.get("temporal_nonlocal_coupling_norm", 0.0))

    sent_signal, sent_signal_weights = _temporal_mix({
        "vector_energy": vector_energy_norm,
        "phase_ring": phase_ring,
        "phase_ring_stability": phase_ring_stability,
        "temporal_relativity": temporal_relativity,
        "zero_point": zero_point,
    })
    feedback_gate, feedback_gate_weights = _temporal_mix({
        "coherence": coherence,
        "overlap": overlap,
        "nonlocal_coupling": nonlocal_coupling,
        "phase_ring_stability": phase_ring_stability,
        "temporal_relativity": temporal_relativity,
    })
    kernel_control_gate, kernel_control_weights = _temporal_mix({
        "coherence": coherence,
        "inertia": inertia,
        "phase_ring": phase_ring,
        "zero_point": zero_point,
        "temporal_relativity": temporal_relativity,
        "interception": interception,
    })
    response_energy, response_energy_weights = _temporal_mix({
        "vector_energy": vector_energy_norm,
        "spin": spin,
        "inertia": inertia,
        "observer_feedback": observer_feedback,
        "interception": interception,
        "phase_delta": phase_delta,
    })
    transport_coherence, transport_coherence_weights = _temporal_mix({
        "coherence": coherence,
        "overlap": overlap,
        "phase_ring_stability": phase_ring_stability,
        "nonlocal_coupling": nonlocal_coupling,
        "temporal_relativity": temporal_relativity,
        "noise_inverse": 1.0 - noise_gate,
    })
    load_hint, load_hint_weights = _temporal_mix({
        "vector_energy": vector_energy_norm,
        "coherence": coherence,
        "overlap": overlap,
        "phase_ring": phase_ring,
        "temporal_relativity": temporal_relativity,
        "interception": interception,
    })
    return {
        "sent_signal": float(sent_signal),
        "feedback_gate": float(feedback_gate),
        "kernel_control_gate": float(kernel_control_gate),
        "response_energy": float(response_energy),
        "transport_coherence": float(transport_coherence),
        "load_hint": float(load_hint),
        "derived_constants": {
            "sent_signal": sent_signal_weights,
            "feedback_gate": feedback_gate_weights,
            "kernel_control_gate": kernel_control_weights,
            "response_energy": response_energy_weights,
            "transport_coherence": transport_coherence_weights,
            "load_hint": load_hint_weights,
        },
    }


def encode_axis_dynamics(dynamics: Dict[str, Any]) -> Dict[str, int]:
    metrics = dict(dynamics or {})
    temporal_constants = derive_temporal_constant_set(metrics)
    axis_word = (
        (_u32_word(metrics.get("axis_scale_x", 0.0), scale=255.0) << 24)
        | (_u32_word(metrics.get("axis_scale_y", 0.0), scale=255.0) << 16)
        | (_u32_word(metrics.get("axis_scale_z", 0.0), scale=255.0) << 8)
        | _u32_word(metrics.get("temporal_coupling_moment", 0.0), scale=255.0)
    ) & 0xFFFFFFFF
    spin_word = (
        (_u32_word(abs(metrics.get("spin_axis_x", 0.0)), scale=255.0) << 24)
        | (_u32_word(abs(metrics.get("spin_axis_y", 0.0)), scale=255.0) << 16)
        | (_u32_word(abs(metrics.get("spin_axis_z", 0.0)), scale=255.0) << 8)
        | _u32_word(metrics.get("spin_momentum_score", 0.0), scale=255.0)
    ) & 0xFFFFFFFF
    inertia_word = (
        (_u32_word(metrics.get("vector_energy", 0.0), scale=255.0) << 24)
        | (_u32_word(metrics.get("inertial_mass_proxy", 0.0), scale=255.0) << 16)
        | (_u32_word(metrics.get("relativistic_correlation", 0.0), scale=255.0) << 8)
        | _u32_word(metrics.get("phase_turns", 0.0), scale=255.0)
    ) & 0xFFFFFFFF
    constants_word = (
        (_u32_word(temporal_constants.get("sent_signal", 0.0), scale=255.0) << 24)
        | (_u32_word(temporal_constants.get("feedback_gate", 0.0), scale=255.0) << 16)
        | (_u32_word(temporal_constants.get("kernel_control_gate", 0.0), scale=255.0) << 8)
        | _u32_word(temporal_constants.get("response_energy", 0.0), scale=255.0)
    ) & 0xFFFFFFFF
    return {
        "axis_word": int(axis_word),
        "spin_word": int(spin_word),
        "inertia_word": int(inertia_word),
        "constants_word": int(constants_word),
        "derived_temporal_constants": temporal_constants,
    }


def encode_transport_prediction(prediction: Dict[str, Any]) -> Dict[str, int]:
    metrics = dict(prediction or {})
    predicted_metrics = dict(metrics.get("predicted_metrics", {}) or {})
    transport_word = (
        (_u32_word(metrics.get("phase_turns_next", 0.0), scale=255.0) << 24)
        | (_u32_word(abs(metrics.get("phase_delta_turns", 0.0)), scale=1023.0) << 14)
        | (_u32_word(metrics.get("observer_feedback_norm", 0.0), scale=63.0) << 8)
        | (_u32_word(metrics.get("reverse_gate", 0.0), scale=1.0) << 7)
        | _u32_word(metrics.get("constraint_mutation_norm", 0.0), scale=127.0)
    ) & 0xFFFFFFFF
    telemetry_word = (
        (_u32_word(metrics.get("noise_pressure_norm", 0.0), scale=255.0) << 24)
        | (_u32_word(predicted_metrics.get("coherence", 0.0), scale=255.0) << 16)
        | (_u32_word(predicted_metrics.get("trap", 0.0), scale=255.0) << 8)
        | _u32_word(metrics.get("subsystem_feedback_norm", 0.0), scale=255.0)
    ) & 0xFFFFFFFF
    return {
        "transport_word": int(transport_word),
        "telemetry_word": int(telemetry_word),
    }


def build_live_telemetry_payload(
    quartet: Dict[str, Any],
    phase_turns: Any,
    previous_phase_turns: Any,
    telemetry: Dict[str, Any] | None = None,
    schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    raw_prediction = build_transport_prediction(
        quartet=quartet,
        phase_turns=phase_turns,
        previous_phase_turns=previous_phase_turns,
        telemetry=telemetry,
        schema=schema,
    )
    prediction = apply_9d_photonic_accounting(quartet, raw_prediction, schema=schema)
    photonic_identity = encode_photonic_identity(prediction)
    axis_words = encode_axis_dynamics(prediction["axis_dynamics"])
    transport_words = encode_transport_prediction(prediction)
    temporal_accounting = dict(prediction.get("temporal_accounting", {}) or {})
    harmonic_noise = dict(prediction.get("harmonic_noise", {}) or {})
    anchor_vectors = dict(prediction.get("anchor_vectors", {}) or {})
    lattice_state = dict(prediction.get("lattice_state", {}) or {})
    trajectory_state = dict(prediction.get("trajectory_state", {}) or {})
    pulse_interference = dict(prediction.get("pulse_interference", {}) or {})
    phase_ring_trace = dict(anchor_vectors.get("phase_ring_trace", {}) or {})
    return {
        "transport_prediction": prediction,
        "temporal_accounting": temporal_accounting,
        "photonic_identity": photonic_identity,
        "encoded_words": {
            **axis_words,
            **transport_words,
            "timing_word": int(
                ((_u32_word(temporal_accounting.get("request_feedback_time_s", 0.0), scale=255.0) << 24)
                | (_u32_word(temporal_accounting.get("calculation_time_s", 0.0), scale=255.0) << 16)
                | (_u32_word(temporal_accounting.get("next_feedback_time_s", 0.0), scale=255.0) << 8)
                | _u32_word(temporal_accounting.get("temporal_accuracy_score", 0.0), scale=255.0))
                & 0xFFFFFFFF
            ),
        },
        "encoding_activation_path": {
            "predict_ahead": True,
            "wait_for_observed_pulse": False,
            "transport_mode": "phase_flux_predictive",
            "required_activation_pulse": dict(prediction["next_pulse_quartet"]),
            "basis_phase_turns": float(prediction["phase_turns"]),
            "basis_phase_turns_next": float(prediction["phase_turns_next"]),
            "basis_trajectory_spectral_id_u64": int(photonic_identity["trajectory_spectral_id_u64"]),
            "conservation_error_norm": float(dict(prediction.get("accounting", {}) or {}).get("conservation_error_norm", 0.0)),
            "temporal_accuracy_score": float(temporal_accounting.get("temporal_accuracy_score", 0.0)),
            "stable_anchor_score": float(anchor_vectors.get("stable_anchor_score", 0.0)),
            "conservation_9d": float(lattice_state.get("conservation_9d", 0.0)),
            "trajectory_conservation_9d": float(trajectory_state.get("trajectory_conservation_9d", 0.0)),
            "reverse_causal_flux_coherence": float(trajectory_state.get("reverse_causal_flux_coherence", 0.0)),
            "gpu_pulse_interference_norm": float(pulse_interference.get("gpu_pulse_interference_norm", 0.0)),
            "system_sensitivity_norm": float(pulse_interference.get("system_sensitivity_norm", 0.0)),
            "phase_ring_utf8": str(phase_ring_trace.get("utf8_trace_text", "")),
        },
        "live_telemetry_path": {
            "channel": "research.photonic.live",
            "phase_turns_next": float(prediction["phase_turns_next"]),
            "next_pulse_quartet": dict(prediction["next_pulse_quartet"]),
            "observer_feedback_norm": float(prediction["observer_feedback_norm"]),
            "reverse_delta_turns": float(prediction["reverse_delta_turns"]),
            "noise_redistribution_norm": dict(prediction["noise_redistribution_norm"]),
            "trajectory_spectral_id_u64": int(photonic_identity["trajectory_spectral_id_u64"]),
            "request_feedback_time_s": float(temporal_accounting.get("request_feedback_time_s", 0.0)),
            "calculation_time_s": float(temporal_accounting.get("calculation_time_s", 0.0)),
            "next_feedback_time_s": float(temporal_accounting.get("next_feedback_time_s", 0.0)),
            "temporal_accuracy_score": float(temporal_accounting.get("temporal_accuracy_score", 0.0)),
            "anchor_interference_norm": float(anchor_vectors.get("anchor_interference_norm", 0.0)),
            "conservation_9d": float(lattice_state.get("conservation_9d", 0.0)),
            "trajectory_stability": float(lattice_state.get("trajectory_stability", 0.0)),
            "trajectory_state_9d": list(trajectory_state.get("trajectory_state_9d", []) or []),
            "trajectory_conservation_9d": float(trajectory_state.get("trajectory_conservation_9d", 0.0)),
            "phase_transport_norm": float(trajectory_state.get("phase_transport_norm", 0.0)),
            "trajectory_expansion_norm": float(trajectory_state.get("trajectory_expansion_norm", 0.0)),
            "reverse_causal_flux_coherence": float(trajectory_state.get("reverse_causal_flux_coherence", 0.0)),
            "hidden_flux_correction_norm": float(trajectory_state.get("hidden_flux_correction_norm", 0.0)),
            "temporal_occlusion_norm": float(trajectory_state.get("temporal_occlusion_norm", 0.0)),
            "pulse_interference_state_9d": list(pulse_interference.get("pulse_interference_state_9d", []) or []),
            "gpu_pulse_interference_norm": float(pulse_interference.get("gpu_pulse_interference_norm", 0.0)),
            "environmental_flux_interference_norm": float(pulse_interference.get("environmental_flux_interference_norm", 0.0)),
            "harmonic_trajectory_interference_norm": float(pulse_interference.get("harmonic_trajectory_interference_norm", 0.0)),
            "system_sensitivity_norm": float(pulse_interference.get("system_sensitivity_norm", 0.0)),
            "pulse_backreaction_norm": float(pulse_interference.get("pulse_backreaction_norm", 0.0)),
            "pulse_trajectory_alignment": float(pulse_interference.get("pulse_trajectory_alignment", 0.0)),
            "pulse_interference_utf8": str(pulse_interference.get("pulse_interference_utf8_text", "")),
            "zero_point_crossover_probability": float(harmonic_noise.get("zero_point_crossover_probability", 0.0)),
            "zero_point_noise_floor": float(harmonic_noise.get("zero_point_noise_floor", 0.0)),
            "environmental_noise_susceptibility": float(harmonic_noise.get("environmental_noise_susceptibility", 0.0)),
            "harmonic_noise_reaction_norm": float(harmonic_noise.get("harmonic_noise_reaction_norm", 0.0)),
            "unwanted_noise_conditions": list(harmonic_noise.get("unwanted_noise_conditions", []) or []),
            "phase_ring_trace_9d": list(phase_ring_trace.get("phase_ring_trace_9d", []) or []),
            "trajectory_utf8": str(trajectory_state.get("trajectory_utf8_text", "")),
            "phase_ring_utf8": str(phase_ring_trace.get("utf8_trace_text", "")),
        },
    }


def predict_full_spectrum_calibration(
    surface: Dict[str, Any],
    phase_turns: Any,
    previous_phase_turns: Any,
    interval_count: int = 6,
    kernel_grid_width: int | None = None,
    kernel_grid_height: int | None = None,
    kernel_interval_ms: float | None = None,
    schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    schema_payload = dict(_default_schema())
    schema_payload.update(dict(schema or {}))
    pulse_codes = _schema_section(schema_payload, "pulse_codes", DEFAULT_PULSE_CODES)
    normalized_window = _schema_section(pulse_codes, "normalized_window", DEFAULT_NORMALIZED_WINDOW)
    payload = dict(surface or {})
    axis_resolution = max(int(payload.get("axis_resolution", 6)), 2)
    width = max(int(kernel_grid_width or axis_resolution * 2), 1)
    height = max(int(kernel_grid_height or axis_resolution * 2), 1)
    intervals = max(int(interval_count), 1)
    granularity = infer_quartet_granularity(payload, normalized_window)
    base_quartet = choose_surface_quartet(payload)
    base_quartet_norm = normalize_quartet(base_quartet, normalized_window)
    base_telemetry = build_surface_telemetry(payload)
    base_calibration_state = build_temporal_relativity_state(
        phase_turns=phase_turns,
        frequency_norm=base_quartet_norm.get("F", 0.0),
        amplitude_norm=base_quartet_norm.get("A", 0.0),
        resonance_gate=clamp01(base_telemetry.get("coherence", 0.0)),
        temporal_overlap=clamp01(base_telemetry.get("temporal_coupling", 0.0)),
        flux_norm=clamp01(base_telemetry.get("predicted_interference", 0.0)),
        vector_x=clamp_signed(base_quartet_norm.get("F", 0.0) - base_quartet_norm.get("V", 0.0)),
        vector_y=clamp_signed(base_quartet_norm.get("A", 0.0) - base_quartet_norm.get("I", 0.0)),
        vector_z=clamp_signed(base_quartet_norm.get("I", 0.0) - base_quartet_norm.get("A", 0.0)),
    )
    if kernel_interval_ms is None:
        kernel_interval_ms = (
            0.5
            + 4.0 * clamp01(base_telemetry.get("thermal_noise", 0.0))
            + 2.0 * (1.0 - clamp01(base_telemetry.get("coherence", 0.0)))
        )

    sequences: list[Dict[str, Any]] = []
    last_phase_turns = wrap_turns(phase_turns)
    last_previous_phase_turns = wrap_turns(previous_phase_turns)
    last_quartet = dict(base_quartet)
    last_prediction: Dict[str, Any] | None = None
    last_identity_id = 0
    total_entries = 0
    aggregate_noise_predictability = 0.0
    aggregate_thermal_predictability = 0.0
    aggregate_field_predictability = 0.0
    aggregate_conservation_error = 0.0
    aggregate_coherence = 0.0
    aggregate_temporal_accuracy = 0.0
    aggregate_harmonic_noise_reaction = 0.0
    aggregate_trajectory_conservation = 0.0
    aggregate_temporal_sequence_alignment = 0.0
    aggregate_reverse_causal_flux_coherence = 0.0
    aggregate_hidden_flux_correction = 0.0
    aggregate_gpu_pulse_interference = 0.0
    aggregate_system_sensitivity = 0.0
    aggregate_pulse_backreaction = 0.0
    aggregate_pulse_trajectory_alignment = 0.0
    aggregate_unwanted_noise_events = 0
    identity_repeat_count = 0

    for direction in SCAN_DIRECTIONS:
        scan_order = build_kernel_scan_order(width, height, direction)
        entries: list[Dict[str, Any]] = []
        sequence_noise_predictability = 0.0
        sequence_thermal_predictability = 0.0
        sequence_field_predictability = 0.0
        sequence_conservation_error = 0.0
        sequence_coherence = 0.0
        sequence_harmonic_noise_reaction = 0.0
        sequence_trajectory_conservation = 0.0
        sequence_temporal_sequence_alignment = 0.0
        sequence_reverse_causal_flux_coherence = 0.0
        sequence_gpu_pulse_interference = 0.0
        sequence_system_sensitivity = 0.0
        sequence_pulse_trajectory_alignment = 0.0

        for coord in scan_order:
            x_norm = 0.0 if width <= 1 else float(coord["x"]) / float(width - 1)
            y_norm = 0.0 if height <= 1 else float(coord["y"]) / float(height - 1)
            if direction == "left_to_right":
                direction_progress_norm = x_norm
            elif direction == "right_to_left":
                direction_progress_norm = 1.0 - x_norm
            elif direction == "top_to_bottom":
                direction_progress_norm = y_norm
            else:
                direction_progress_norm = 1.0 - y_norm

            for interval_index in range(intervals):
                activation_quartet = _build_interval_activation_quartet(
                    base_quartet=last_quartet,
                    granularity=granularity,
                    normalized_window=normalized_window,
                    interval_index=interval_index,
                    interval_count=intervals,
                    direction_progress_norm=direction_progress_norm,
                    previous_prediction=last_prediction,
                )
                interval_norm = 0.0 if intervals <= 1 else float(interval_index) / float(intervals - 1)
                sequence_temporal_state = build_temporal_relativity_state(
                    phase_turns=last_phase_turns,
                    frequency_norm=normalize_quartet({
                        "F": activation_quartet["F"],
                        "A": activation_quartet["A"],
                        "I": activation_quartet["I"],
                        "V": activation_quartet["V"],
                    }, normalized_window).get("F", 0.0),
                    amplitude_norm=normalize_quartet({
                        "F": activation_quartet["F"],
                        "A": activation_quartet["A"],
                        "I": activation_quartet["I"],
                        "V": activation_quartet["V"],
                    }, normalized_window).get("A", 0.0),
                    resonance_gate=clamp01(base_telemetry.get("coherence", 0.0)),
                    temporal_overlap=clamp01(base_telemetry.get("temporal_coupling", 0.0)),
                    flux_norm=clamp01(base_telemetry.get("predicted_interference", 0.0)),
                    vector_x=clamp_signed(direction_progress_norm - 0.5),
                    vector_y=clamp_signed(interval_norm - 0.5),
                    vector_z=clamp_signed(clamp01(dict(last_prediction or {}).get("noise_pressure_norm", 0.0)) - 0.5),
                    observer_feedback_norm=clamp01(dict(last_prediction or {}).get("observer_feedback_norm", 0.0)),
                    phase_memory_norm=clamp01(abs(dict(last_prediction or {}).get("phase_delta_turns", 0.0)) * 4.0),
                )
                telemetry = {
                    "coherence": _temporal_mix({
                        "base_coherence": clamp01(base_telemetry.get("coherence", 0.0)),
                        "interval_margin": clamp01(1.0 - interval_norm),
                        "direction_balance": clamp01(1.0 - abs(direction_progress_norm - 0.5) * 2.0),
                        "last_coherence": clamp01(dict(last_prediction or {}).get("predicted_metrics", {}).get("coherence", 0.0)),
                        "observer_feedback": clamp01(dict(last_prediction or {}).get("observer_feedback_norm", 0.0)),
                        "phase_alignment": sequence_temporal_state.get("phase_alignment_probability", 0.0),
                    })[0],
                    "trap_ratio": _temporal_mix({
                        "base_trap": clamp01(base_telemetry.get("trap_ratio", 0.0)),
                        "interval_progress": interval_norm,
                        "direction_edge": clamp01(abs(direction_progress_norm - 0.5) * 2.0),
                        "last_trap": clamp01(dict(last_prediction or {}).get("predicted_metrics", {}).get("trap", 0.0)),
                        "last_noise": clamp01(dict(last_prediction or {}).get("noise_pressure_norm", 0.0)),
                        "cross_talk": sequence_temporal_state.get("cross_talk_force_norm", 0.0),
                    })[0],
                    "predicted_interference": _temporal_mix({
                        "base_interference": clamp01(base_telemetry.get("predicted_interference", 0.0)),
                        "interval_progress": interval_norm,
                        "direction_progress": clamp01(direction_progress_norm),
                        "wavelength_drop": clamp01(activation_quartet["wavelength_drop_norm"]),
                        "last_interference": clamp01(dict(last_prediction or {}).get("predicted_interference_norm", 0.0)),
                        "entanglement": sequence_temporal_state.get("entanglement_probability", 0.0),
                    })[0],
                    "temporal_coupling": _temporal_mix({
                        "base_temporal_coupling": clamp01(base_telemetry.get("temporal_coupling", 0.0)),
                        "interval_progress": interval_norm,
                        "direction_balance": clamp01(1.0 - abs(direction_progress_norm - 0.5) * 2.0),
                        "phase_delta": clamp01(abs(dict(last_prediction or {}).get("phase_delta_turns", 0.0)) * 4.0),
                        "observer_feedback": clamp01(dict(last_prediction or {}).get("observer_feedback_norm", 0.0)),
                        "field_time": sequence_temporal_state.get("field_time_norm", 0.0),
                    })[0],
                    "thermal_noise": _temporal_mix({
                        "base_thermal_noise": clamp01(base_telemetry.get("thermal_noise", 0.0)),
                        "interval_progress": interval_norm,
                        "direction_edge": clamp01(abs(direction_progress_norm - 0.5) * 2.0),
                        "last_noise": clamp01(dict(last_prediction or {}).get("noise_pressure_norm", 0.0)),
                        "system_sensitivity": clamp01(dict(dict(last_prediction or {}).get("pulse_interference", {}) or {}).get("system_sensitivity_norm", 0.0)),
                    })[0],
                    "observed_subsystems": {
                        "residual": _temporal_mix({
                            "base_residual": clamp01(dict(base_telemetry.get("observed_subsystems", {}) or {}).get("residual", 0.0)),
                            "interval_progress": interval_norm,
                            "constraint_mutation": clamp01(dict(last_prediction or {}).get("constraint_mutation_norm", 0.0)),
                            "intercept_inertia": sequence_temporal_state.get("intercept_inertia_norm", 0.0),
                        })[0],
                        "spin": _temporal_mix({
                            "base_spin": clamp01(dict(base_telemetry.get("observed_subsystems", {}) or {}).get("spin", 0.0)),
                            "wavelength_drop": clamp01(activation_quartet["wavelength_drop_norm"]),
                            "spin_momentum": clamp01(dict(last_prediction or {}).get("axis_dynamics", {}).get("spin_momentum_score", 0.0)),
                            "entanglement": sequence_temporal_state.get("entanglement_probability", 0.0),
                        })[0],
                        "coupling": _temporal_mix({
                            "base_coupling": clamp01(dict(base_telemetry.get("observed_subsystems", {}) or {}).get("coupling", 0.0)),
                            "direction_progress": clamp01(direction_progress_norm),
                            "observer_feedback": clamp01(dict(last_prediction or {}).get("observer_feedback_norm", 0.0)),
                            "phase_alignment": sequence_temporal_state.get("phase_alignment_probability", 0.0),
                        })[0],
                        "controller": _temporal_mix({
                            "base_controller": clamp01(dict(base_telemetry.get("observed_subsystems", {}) or {}).get("controller", 0.0)),
                            "interval_progress": interval_norm,
                            "vector_path_energy": clamp01(dict(last_prediction or {}).get("accounting", {}).get("vector_path_energy_rating", 0.0)),
                            "field_time": sequence_temporal_state.get("field_time_norm", 0.0),
                        })[0],
                    },
                }

                raw_prediction = build_transport_prediction(
                    quartet={key: activation_quartet[key] for key in ("F", "A", "I", "V")},
                    phase_turns=last_phase_turns,
                    previous_phase_turns=last_previous_phase_turns,
                    telemetry=telemetry,
                    schema=schema_payload,
                )
                prediction = apply_9d_photonic_accounting(
                    previous_quartet=last_quartet,
                    prediction=raw_prediction,
                    schema=schema_payload,
                )
                photonic_identity = encode_photonic_identity(prediction)
                temporal_accounting = dict(prediction.get("temporal_accounting", {}) or {})
                harmonic_noise = dict(prediction.get("harmonic_noise", {}) or {})
                trajectory_state = dict(prediction.get("trajectory_state", {}) or {})
                pulse_interference = dict(prediction.get("pulse_interference", {}) or {})
                encoded_words = {
                    **encode_axis_dynamics(prediction["axis_dynamics"]),
                    **encode_transport_prediction(prediction),
                }

                noise_predictability = float(1.0 - clamp01(prediction["noise_pressure_norm"]))
                thermal_predictability = float(1.0 - clamp01(telemetry["thermal_noise"]))
                field_predictability = float(1.0 - _temporal_mix({
                    "constraint_mutation": prediction["constraint_mutation_norm"],
                    "conservation_error": dict(prediction.get("accounting", {}) or {}).get("conservation_error_norm", 0.0),
                    "trap": dict(prediction.get("predicted_metrics", {}) or {}).get("trap", 0.0),
                    "trajectory_detune": clamp01(1.0 - clamp01(trajectory_state.get("trajectory_conservation_9d", 0.0))),
                    "sequence_detune": clamp01(1.0 - clamp01(trajectory_state.get("temporal_sequence_alignment", 0.0))),
                    "reverse_flux_detune": clamp01(1.0 - clamp01(trajectory_state.get("reverse_causal_flux_coherence", 0.0))),
                    "gpu_pulse_interference": clamp01(pulse_interference.get("gpu_pulse_interference_norm", 0.0)),
                    "system_sensitivity": clamp01(pulse_interference.get("system_sensitivity_norm", 0.0)),
                    "pulse_detune": clamp01(1.0 - clamp01(pulse_interference.get("pulse_trajectory_alignment", 0.0))),
                })[0])
                if last_identity_id and int(photonic_identity["trajectory_spectral_id_u64"]) == last_identity_id:
                    identity_repeat_count += 1

                entry = {
                    "direction": direction,
                    "kernel_index": int(coord["kernel_index"]),
                    "coord": {"x": int(coord["x"]), "y": int(coord["y"])},
                    "interval_index": int(interval_index),
                    "interval_count": int(intervals),
                    "interval_ms": float(kernel_interval_ms),
                    "direction_progress_norm": float(direction_progress_norm),
                    "activation_quartet": {
                        "F": float(activation_quartet["F"]),
                        "A": float(activation_quartet["A"]),
                        "I": float(activation_quartet["I"]),
                        "V": float(activation_quartet["V"]),
                    },
                    "wavelength_before": float(activation_quartet["wavelength_before"]),
                    "wavelength_after": float(activation_quartet["wavelength_after"]),
                    "wavelength_drop_norm": float(activation_quartet["wavelength_drop_norm"]),
                    "noise_predictability": float(noise_predictability),
                    "thermal_predictability": float(thermal_predictability),
                    "field_predictability": float(field_predictability),
                    "encoding_activation_path": {
                        "predict_ahead": True,
                        "wait_for_observed_pulse": False,
                        "required_activation_pulse": dict(prediction["next_pulse_quartet"]),
                        "trajectory_spectral_id_u64": int(photonic_identity["trajectory_spectral_id_u64"]),
                        "phase_turns_next": float(prediction["phase_turns_next"]),
                    },
                    "transport_prediction": {
                        "phase_turns_next": float(prediction["phase_turns_next"]),
                        "phase_delta_turns": float(prediction["phase_delta_turns"]),
                        "observer_feedback_norm": float(prediction["observer_feedback_norm"]),
                        "reverse_gate": float(prediction["reverse_gate"]),
                        "noise_pressure_norm": float(prediction["noise_pressure_norm"]),
                        "anchor_interference_norm": float(dict(prediction.get("anchor_vectors", {}) or {}).get("anchor_interference_norm", 0.0)),
                        "trajectory_conservation_9d": float(trajectory_state.get("trajectory_conservation_9d", 0.0)),
                        "temporal_sequence_alignment": float(trajectory_state.get("temporal_sequence_alignment", 0.0)),
                        "reverse_causal_flux_coherence": float(trajectory_state.get("reverse_causal_flux_coherence", 0.0)),
                        "hidden_flux_correction_norm": float(trajectory_state.get("hidden_flux_correction_norm", 0.0)),
                        "gpu_pulse_interference_norm": float(pulse_interference.get("gpu_pulse_interference_norm", 0.0)),
                        "system_sensitivity_norm": float(pulse_interference.get("system_sensitivity_norm", 0.0)),
                        "pulse_backreaction_norm": float(pulse_interference.get("pulse_backreaction_norm", 0.0)),
                        "pulse_trajectory_alignment": float(pulse_interference.get("pulse_trajectory_alignment", 0.0)),
                        "harmonic_noise_reaction_norm": float(harmonic_noise.get("harmonic_noise_reaction_norm", 0.0)),
                        "unwanted_noise_conditions": list(harmonic_noise.get("unwanted_noise_conditions", []) or []),
                        "predicted_metrics": dict(prediction["predicted_metrics"]),
                        "accounting": dict(prediction["accounting"]),
                        "trajectory_state": trajectory_state,
                        "pulse_interference": pulse_interference,
                        "temporal_accounting": temporal_accounting,
                    },
                    "photonic_identity": photonic_identity,
                    "encoded_words": encoded_words,
                }
                entries.append(entry)
                total_entries += 1
                sequence_noise_predictability += noise_predictability
                sequence_thermal_predictability += thermal_predictability
                sequence_field_predictability += field_predictability
                sequence_conservation_error += float(dict(prediction.get("accounting", {}) or {}).get("conservation_error_norm", 0.0))
                sequence_coherence += float(dict(prediction.get("predicted_metrics", {}) or {}).get("coherence", 0.0))
                sequence_harmonic_noise_reaction += float(harmonic_noise.get("harmonic_noise_reaction_norm", 0.0))
                sequence_trajectory_conservation += float(trajectory_state.get("trajectory_conservation_9d", 0.0))
                sequence_temporal_sequence_alignment += float(trajectory_state.get("temporal_sequence_alignment", 0.0))
                sequence_reverse_causal_flux_coherence += float(trajectory_state.get("reverse_causal_flux_coherence", 0.0))
                sequence_gpu_pulse_interference += float(pulse_interference.get("gpu_pulse_interference_norm", 0.0))
                sequence_system_sensitivity += float(pulse_interference.get("system_sensitivity_norm", 0.0))
                sequence_pulse_trajectory_alignment += float(pulse_interference.get("pulse_trajectory_alignment", 0.0))
                aggregate_noise_predictability += noise_predictability
                aggregate_thermal_predictability += thermal_predictability
                aggregate_field_predictability += field_predictability
                aggregate_conservation_error += float(dict(prediction.get("accounting", {}) or {}).get("conservation_error_norm", 0.0))
                aggregate_coherence += float(dict(prediction.get("predicted_metrics", {}) or {}).get("coherence", 0.0))
                aggregate_temporal_accuracy += float(temporal_accounting.get("temporal_accuracy_score", 0.0))
                aggregate_harmonic_noise_reaction += float(harmonic_noise.get("harmonic_noise_reaction_norm", 0.0))
                aggregate_trajectory_conservation += float(trajectory_state.get("trajectory_conservation_9d", 0.0))
                aggregate_temporal_sequence_alignment += float(trajectory_state.get("temporal_sequence_alignment", 0.0))
                aggregate_reverse_causal_flux_coherence += float(trajectory_state.get("reverse_causal_flux_coherence", 0.0))
                aggregate_hidden_flux_correction += float(trajectory_state.get("hidden_flux_correction_norm", 0.0))
                aggregate_gpu_pulse_interference += float(pulse_interference.get("gpu_pulse_interference_norm", 0.0))
                aggregate_system_sensitivity += float(pulse_interference.get("system_sensitivity_norm", 0.0))
                aggregate_pulse_backreaction += float(pulse_interference.get("pulse_backreaction_norm", 0.0))
                aggregate_pulse_trajectory_alignment += float(pulse_interference.get("pulse_trajectory_alignment", 0.0))
                if list(harmonic_noise.get("unwanted_noise_conditions", []) or []):
                    aggregate_unwanted_noise_events += 1
                last_previous_phase_turns = last_phase_turns
                last_phase_turns = float(prediction["phase_turns_next"])
                last_quartet = dict(prediction["next_pulse_quartet"])
                last_prediction = prediction
                last_identity_id = int(photonic_identity["trajectory_spectral_id_u64"])

        sequence_entry_count = max(len(entries), 1)
        sequences.append(
            {
                "direction": direction,
                "sequence_complete": bool(len(entries) == width * height * intervals),
                "entry_count": int(len(entries)),
                "mean_noise_predictability": float(sequence_noise_predictability / float(sequence_entry_count)),
                "mean_thermal_predictability": float(sequence_thermal_predictability / float(sequence_entry_count)),
                "mean_field_predictability": float(sequence_field_predictability / float(sequence_entry_count)),
                "mean_conservation_error": float(sequence_conservation_error / float(sequence_entry_count)),
                "mean_coherence": float(sequence_coherence / float(sequence_entry_count)),
                "mean_harmonic_noise_reaction": float(sequence_harmonic_noise_reaction / float(sequence_entry_count)),
                "mean_trajectory_conservation": float(sequence_trajectory_conservation / float(sequence_entry_count)),
                "mean_temporal_sequence_alignment": float(sequence_temporal_sequence_alignment / float(sequence_entry_count)),
                "mean_reverse_causal_flux_coherence": float(sequence_reverse_causal_flux_coherence / float(sequence_entry_count)),
                "mean_gpu_pulse_interference": float(sequence_gpu_pulse_interference / float(sequence_entry_count)),
                "mean_system_sensitivity": float(sequence_system_sensitivity / float(sequence_entry_count)),
                "mean_pulse_trajectory_alignment": float(sequence_pulse_trajectory_alignment / float(sequence_entry_count)),
                "entries": entries,
            }
        )

    count = max(total_entries, 1)
    sequence_coverage = float(sum(1 for sequence in sequences if sequence["sequence_complete"]) / float(len(SCAN_DIRECTIONS)))
    mean_noise_predictability = float(aggregate_noise_predictability / float(count))
    mean_thermal_predictability = float(aggregate_thermal_predictability / float(count))
    mean_field_predictability = float(aggregate_field_predictability / float(count))
    mean_conservation_error = float(aggregate_conservation_error / float(count))
    mean_coherence = float(aggregate_coherence / float(count))
    mean_temporal_accuracy = float(aggregate_temporal_accuracy / float(count))
    mean_harmonic_noise_reaction = float(aggregate_harmonic_noise_reaction / float(count))
    mean_trajectory_conservation = float(aggregate_trajectory_conservation / float(count))
    mean_temporal_sequence_alignment = float(aggregate_temporal_sequence_alignment / float(count))
    mean_reverse_causal_flux_coherence = float(aggregate_reverse_causal_flux_coherence / float(count))
    mean_hidden_flux_correction = float(aggregate_hidden_flux_correction / float(count))
    mean_gpu_pulse_interference = float(aggregate_gpu_pulse_interference / float(count))
    mean_system_sensitivity = float(aggregate_system_sensitivity / float(count))
    mean_pulse_backreaction = float(aggregate_pulse_backreaction / float(count))
    mean_pulse_trajectory_alignment = float(aggregate_pulse_trajectory_alignment / float(count))
    unwanted_noise_condition_ratio = float(aggregate_unwanted_noise_events / float(count))
    identity_repeat_ratio = float(identity_repeat_count / float(max(total_entries - 1, 1)))
    noise_predictability_min = _temporal_mix({
        "base_phase_alignment": base_calibration_state.get("phase_alignment_probability", 0.0),
        "trajectory_conservation": mean_trajectory_conservation,
        "temporal_sequence_alignment": mean_temporal_sequence_alignment,
        "pulse_margin": clamp01(1.0 - mean_gpu_pulse_interference),
        "sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
        "pulse_alignment": mean_pulse_trajectory_alignment,
    })[0]
    thermal_predictability_min = _temporal_mix({
        "base_field_time": base_calibration_state.get("field_time_norm", 0.0),
        "coherence": mean_coherence,
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "hidden_flux": mean_hidden_flux_correction,
        "system_sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
    })[0]
    field_predictability_min = _temporal_mix({
        "trajectory_conservation": mean_trajectory_conservation,
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "hidden_flux": mean_hidden_flux_correction,
        "harmonic_margin": clamp01(1.0 - mean_harmonic_noise_reaction),
    })[0]
    trajectory_conservation_min = _temporal_mix({
        "base_temporal_relativity": base_calibration_state.get("temporal_relativity_norm", 0.0),
        "trajectory_conservation": mean_trajectory_conservation,
        "pulse_alignment": mean_pulse_trajectory_alignment,
    })[0]
    temporal_sequence_alignment_min = _temporal_mix({
        "base_phase_alignment": base_calibration_state.get("phase_alignment_probability", 0.0),
        "temporal_sequence_alignment": mean_temporal_sequence_alignment,
        "reverse_flux": mean_reverse_causal_flux_coherence,
    })[0]
    reverse_causal_flux_coherence_min = _temporal_mix({
        "base_entanglement": base_calibration_state.get("entanglement_probability", 0.0),
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "hidden_flux": mean_hidden_flux_correction,
        "pulse_margin": clamp01(1.0 - mean_gpu_pulse_interference),
    })[0]
    hidden_flux_correction_min = _temporal_mix({
        "base_phase_alignment": base_calibration_state.get("phase_alignment_probability", 0.0),
        "hidden_flux": mean_hidden_flux_correction,
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "trajectory_conservation": mean_trajectory_conservation,
    })[0]
    gpu_pulse_interference_max = clamp01(1.0 - _temporal_mix({
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "system_sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
        "trajectory_conservation": mean_trajectory_conservation,
        "base_cross_talk_margin": clamp01(1.0 - base_calibration_state.get("cross_talk_force_norm", 0.0)),
    })[0])
    system_sensitivity_max = clamp01(1.0 - _temporal_mix({
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "hidden_flux": mean_hidden_flux_correction,
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "base_intercept_margin": clamp01(1.0 - base_calibration_state.get("intercept_inertia_norm", 0.0)),
    })[0])
    pulse_backreaction_max = clamp01(1.0 - _temporal_mix({
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "system_sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "hidden_flux": mean_hidden_flux_correction,
    })[0])
    temporal_accuracy_min = _temporal_mix({
        "base_field_time": base_calibration_state.get("field_time_norm", 0.0),
        "mean_temporal_accuracy": mean_temporal_accuracy,
        "trajectory_conservation": mean_trajectory_conservation,
        "pulse_alignment": mean_pulse_trajectory_alignment,
    })[0]
    harmonic_noise_reaction_max = clamp01(1.0 - _temporal_mix({
        "hidden_flux": mean_hidden_flux_correction,
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "system_sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
    })[0])
    unwanted_noise_condition_ratio_max = clamp01(1.0 - _temporal_mix({
        "trajectory_conservation": mean_trajectory_conservation,
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "system_sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
        "hidden_flux": mean_hidden_flux_correction,
    })[0])
    pulse_trajectory_alignment_min = _temporal_mix({
        "base_phase_alignment": base_calibration_state.get("phase_alignment_probability", 0.0),
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "trajectory_conservation": mean_trajectory_conservation,
        "hidden_flux": mean_hidden_flux_correction,
    })[0]
    coherence_min = _temporal_mix({
        "base_phase_alignment": base_calibration_state.get("phase_alignment_probability", 0.0),
        "mean_coherence": mean_coherence,
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "trajectory_conservation": mean_trajectory_conservation,
    })[0]
    conservation_error_max = clamp01(_temporal_mix({
        "conservation_margin": clamp01(1.0 - mean_conservation_error),
        "trajectory_conservation": mean_trajectory_conservation,
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "hidden_flux": mean_hidden_flux_correction,
    })[0])
    identity_repeat_ratio_max = clamp01(1.0 - _temporal_mix({
        "trajectory_conservation": mean_trajectory_conservation,
        "reverse_flux": mean_reverse_causal_flux_coherence,
        "pulse_alignment": mean_pulse_trajectory_alignment,
        "system_sensitivity_margin": clamp01(1.0 - mean_system_sensitivity),
    })[0])
    feedback_gate_open = bool(
        sequence_coverage >= 1.0
        and mean_noise_predictability >= noise_predictability_min
        and mean_thermal_predictability >= thermal_predictability_min
        and mean_field_predictability >= field_predictability_min
        and mean_conservation_error <= conservation_error_max
        and mean_trajectory_conservation >= trajectory_conservation_min
        and mean_temporal_sequence_alignment >= temporal_sequence_alignment_min
        and mean_reverse_causal_flux_coherence >= reverse_causal_flux_coherence_min
        and mean_hidden_flux_correction >= hidden_flux_correction_min
        and mean_gpu_pulse_interference <= gpu_pulse_interference_max
        and mean_system_sensitivity <= system_sensitivity_max
        and mean_pulse_backreaction <= pulse_backreaction_max
        and mean_pulse_trajectory_alignment >= pulse_trajectory_alignment_min
        and mean_coherence >= coherence_min
        and mean_temporal_accuracy >= temporal_accuracy_min
        and mean_harmonic_noise_reaction <= harmonic_noise_reaction_max
        and unwanted_noise_condition_ratio <= unwanted_noise_condition_ratio_max
        and identity_repeat_ratio <= identity_repeat_ratio_max
    )

    return {
        "model": "predictive_full_spectrum_gpu_pulse_calibration",
        "predict_ahead": True,
        "wait_for_observed_pulse": False,
        "axis_resolution": int(axis_resolution),
        "kernel_grid_width": int(width),
        "kernel_grid_height": int(height),
        "kernel_count": int(width * height),
        "interval_count": int(intervals),
        "kernel_interval_ms": float(kernel_interval_ms),
        "granularity": {name: float(value) for name, value in granularity.items()},
        "base_quartet": dict(base_quartet),
        "base_telemetry": base_telemetry,
        "sequence_coverage": float(sequence_coverage),
        "mean_noise_predictability": mean_noise_predictability,
        "mean_thermal_predictability": mean_thermal_predictability,
        "mean_field_predictability": mean_field_predictability,
        "mean_conservation_error": mean_conservation_error,
        "mean_coherence": mean_coherence,
        "mean_temporal_accuracy": mean_temporal_accuracy,
        "mean_harmonic_noise_reaction": mean_harmonic_noise_reaction,
        "mean_trajectory_conservation": mean_trajectory_conservation,
        "mean_temporal_sequence_alignment": mean_temporal_sequence_alignment,
        "mean_reverse_causal_flux_coherence": mean_reverse_causal_flux_coherence,
        "mean_hidden_flux_correction": mean_hidden_flux_correction,
        "mean_gpu_pulse_interference": mean_gpu_pulse_interference,
        "mean_system_sensitivity": mean_system_sensitivity,
        "mean_pulse_backreaction": mean_pulse_backreaction,
        "mean_pulse_trajectory_alignment": mean_pulse_trajectory_alignment,
        "unwanted_noise_condition_ratio": unwanted_noise_condition_ratio,
        "identity_repeat_ratio": identity_repeat_ratio,
        "feedback_gate": {
            "state": "open" if feedback_gate_open else "gated",
            "gate_open": bool(feedback_gate_open),
            "reason": (
                "full sequence coverage with predictive stability"
                if feedback_gate_open
                else "predictive stability insufficient for live feedback"
            ),
            "requirements": {
                "sequence_coverage": 1.0,
                "noise_predictability_min": noise_predictability_min,
                "thermal_predictability_min": thermal_predictability_min,
                "field_predictability_min": field_predictability_min,
                "conservation_error_max": conservation_error_max,
                "trajectory_conservation_min": trajectory_conservation_min,
                "temporal_sequence_alignment_min": temporal_sequence_alignment_min,
                "reverse_causal_flux_coherence_min": reverse_causal_flux_coherence_min,
                "hidden_flux_correction_min": hidden_flux_correction_min,
                "gpu_pulse_interference_max": gpu_pulse_interference_max,
                "system_sensitivity_max": system_sensitivity_max,
                "pulse_backreaction_max": pulse_backreaction_max,
                "pulse_trajectory_alignment_min": pulse_trajectory_alignment_min,
                "temporal_accuracy_min": temporal_accuracy_min,
                "harmonic_noise_reaction_max": harmonic_noise_reaction_max,
                "unwanted_noise_condition_ratio_max": unwanted_noise_condition_ratio_max,
                "identity_repeat_ratio_max": identity_repeat_ratio_max,
            },
        },
        "encoding_path": {
            "predictive_activation": True,
            "phase_flux_transport": True,
            "uses_last_recorded_trajectory": True,
            "trajectory_spectral_id_u64": int(last_identity_id),
            "required_activation_pulse": dict(last_quartet),
        },
        "sequences": sequences,
    }
