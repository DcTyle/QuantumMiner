from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple
import hashlib
import json
import math

from gpu_pulse_axis_dynamics import (
    build_live_telemetry_payload,
    clamp01,
    safe_float,
    signed_turn_delta,
    wrap_turns,
)


PHOTONIC_VECTOR_AXES = (
    "phase",
    "axis_x",
    "axis_y",
    "axis_z",
    "flux",
    "coherence",
    "spin",
    "inertia",
    "feedback",
)

DEFAULT_STATIC_PROCESS_PROFILES: Dict[str, Dict[str, Any]] = {
    "substrate_idle": {
        "vector_offset": (0.00, 0.01, 0.00, 0.01, 0.00, 0.03, 0.01, 0.02, 0.03),
        "pulse_bias": {"F": -0.004, "A": 0.000, "I": -0.004, "V": 0.000},
        "threshold": 0.86,
    },
    "phase_transport_observer": {
        "vector_offset": (0.03, 0.01, 0.01, 0.00, 0.03, 0.04, 0.01, 0.02, 0.04),
        "pulse_bias": {"F": 0.003, "A": 0.000, "I": 0.000, "V": 0.002},
        "threshold": 0.88,
    },
    "boot_resume": {
        "vector_offset": (0.02, 0.02, 0.00, 0.01, 0.02, 0.04, 0.00, 0.04, 0.05),
        "pulse_bias": {"F": 0.000, "A": 0.002, "I": 0.003, "V": 0.003},
        "threshold": 0.90,
    },
    "memory_latch": {
        "vector_offset": (0.01, 0.00, 0.02, 0.00, 0.01, 0.03, 0.01, 0.05, 0.03),
        "pulse_bias": {"F": 0.000, "A": 0.003, "I": 0.001, "V": 0.002},
        "threshold": 0.89,
    },
    "miner_actuation": {
        "vector_offset": (0.02, 0.03, 0.00, 0.03, 0.05, 0.02, 0.03, 0.04, 0.02),
        "pulse_bias": {"F": 0.003, "A": 0.001, "I": 0.004, "V": 0.004},
        "threshold": 0.91,
    },
}


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True))


def _stable_word(value: Any) -> int:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    else:
        text = str(value or "")
    digest = hashlib.sha256(text.encode("ascii", errors="ignore")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _normalize_quartet(quartet: Dict[str, Any]) -> Dict[str, float]:
    return {
        "F": float(clamp01(quartet.get("F", 0.0))),
        "A": float(clamp01(quartet.get("A", 0.0))),
        "I": float(clamp01(quartet.get("I", 0.0))),
        "V": float(clamp01(quartet.get("V", 0.0))),
    }


def _normalize_vector_9d(vector_9d: Iterable[Any]) -> Tuple[float, ...]:
    values = [safe_float(value, 0.0) for value in list(vector_9d)]
    if len(values) != 9:
        raise ValueError("vector_9d must contain exactly 9 values")
    normalized: List[float] = []
    for index, value in enumerate(values):
        if index == 0:
            normalized.append(float(wrap_turns(value)))
        else:
            normalized.append(float(clamp01(value)))
    return tuple(normalized)


def _mean_abs_gap(lhs: Iterable[float], rhs: Iterable[float], scale: float = 1.0) -> float:
    lhs_values = list(lhs)
    rhs_values = list(rhs)
    if not lhs_values or not rhs_values:
        return 1.0
    count = min(len(lhs_values), len(rhs_values))
    if count <= 0:
        return 1.0
    total = 0.0
    for lhs_value, rhs_value in zip(lhs_values[:count], rhs_values[:count]):
        total += abs(float(lhs_value) - float(rhs_value))
    return clamp01(total / (float(count) * max(float(scale), 1.0e-9)))


def _first_differences(vector_9d: Tuple[float, ...]) -> List[float]:
    return [float(vector_9d[index + 1] - vector_9d[index]) for index in range(len(vector_9d) - 1)]


def _second_differences(vector_9d: Tuple[float, ...]) -> List[float]:
    first = _first_differences(vector_9d)
    return [float(first[index + 1] - first[index]) for index in range(len(first) - 1)]


def _circular_mean_turns(values: Iterable[float], weights: Iterable[float]) -> float:
    vector_x = 0.0
    vector_y = 0.0
    weight_total = 0.0
    for value, weight in zip(values, weights):
        weight_value = max(0.0, float(weight))
        angle = float(wrap_turns(value)) * math.tau
        vector_x += math.cos(angle) * weight_value
        vector_y += math.sin(angle) * weight_value
        weight_total += weight_value
    if weight_total <= 1.0e-9:
        return 0.0
    angle = math.atan2(vector_y / weight_total, vector_x / weight_total)
    return float(wrap_turns(angle / math.tau))


def _blend_scalar(current: float, target: float, fraction: float, phase: bool = False) -> float:
    if phase:
        return float(wrap_turns(current + signed_turn_delta(target, current) * clamp01(fraction)))
    return float(clamp01(current + ((target - current) * clamp01(fraction))))


def _interpolate_vector_9d(current: Iterable[float], target: Iterable[float], fraction: float) -> Tuple[float, ...]:
    current_vector = _normalize_vector_9d(current)
    target_vector = _normalize_vector_9d(target)
    blended: List[float] = []
    for index, (current_value, target_value) in enumerate(zip(current_vector, target_vector)):
        blended.append(_blend_scalar(current_value, target_value, fraction, phase=(index == 0)))
    return tuple(blended)


def _interpolate_quartet(current: Dict[str, Any], target: Dict[str, Any], fraction: float) -> Dict[str, float]:
    current_quartet = _normalize_quartet(current)
    target_quartet = _normalize_quartet(target)
    return {
        name: _blend_scalar(current_quartet[name], target_quartet[name], fraction)
        for name in ("F", "A", "I", "V")
    }


def build_process_vector_9d(payload: Dict[str, Any]) -> Tuple[float, ...]:
    source = dict(payload or {})
    prediction = dict(source.get("transport_prediction", source) or {})
    axis_dynamics = dict(prediction.get("axis_dynamics", {}) or {})
    predicted_metrics = dict(prediction.get("predicted_metrics", {}) or {})
    return _normalize_vector_9d(
        (
            prediction.get("phase_turns_next", prediction.get("phase_turns", 0.0)),
            axis_dynamics.get("axis_scale_x", 0.0),
            axis_dynamics.get("axis_scale_y", 0.0),
            axis_dynamics.get("axis_scale_z", 0.0),
            prediction.get("flux_transport_norm", 0.0),
            predicted_metrics.get("coherence", prediction.get("observer_feedback_norm", 0.0)),
            axis_dynamics.get("spin_momentum_score", 0.0),
            axis_dynamics.get("inertial_mass_proxy", 0.0),
            prediction.get("observer_feedback_norm", prediction.get("subsystem_feedback_norm", 0.0)),
        )
    )


def compare_temporal_trajectory(lhs_vector_9d: Iterable[Any], rhs_vector_9d: Iterable[Any]) -> Dict[str, float]:
    lhs_vector = _normalize_vector_9d(lhs_vector_9d)
    rhs_vector = _normalize_vector_9d(rhs_vector_9d)
    phase_position_score = clamp01(1.0 - (abs(signed_turn_delta(lhs_vector[0], rhs_vector[0])) * 2.0))
    shape_score = clamp01(1.0 - _mean_abs_gap(lhs_vector[1:], rhs_vector[1:]))
    direction_score = clamp01(
        1.0 - _mean_abs_gap(_first_differences(lhs_vector), _first_differences(rhs_vector), scale=2.0)
    )
    curvature_score = clamp01(
        1.0 - _mean_abs_gap(_second_differences(lhs_vector), _second_differences(rhs_vector), scale=2.0)
    )
    speed_score = clamp01(1.0 - _mean_abs_gap(lhs_vector[4:], rhs_vector[4:]))
    overall_similarity = clamp01(
        (phase_position_score * 0.24)
        + (shape_score * 0.24)
        + (direction_score * 0.18)
        + (curvature_score * 0.18)
        + (speed_score * 0.16)
    )
    return {
        "phase_position_score": float(phase_position_score),
        "shape_score": float(shape_score),
        "direction_score": float(direction_score),
        "curvature_score": float(curvature_score),
        "speed_score": float(speed_score),
        "overall_similarity": float(overall_similarity),
    }


def _carrier_id(label: str, vector_9d: Iterable[Any]) -> str:
    word = _stable_word({"label": str(label), "vector_9d": list(_normalize_vector_9d(vector_9d))})
    return "CARRIER-%016X" % word


def make_carrier_wave(
    label: str,
    vector_9d: Iterable[Any],
    activation_quartet: Dict[str, Any],
    resonance_threshold: float = 0.92,
    metadata: Dict[str, Any] | None = None,
) -> "CarrierWave9D":
    return CarrierWave9D(
        carrier_id=_carrier_id(label, vector_9d),
        label=str(label),
        vector_9d=_normalize_vector_9d(vector_9d),
        activation_quartet=_normalize_quartet(activation_quartet),
        resonance_threshold=float(clamp01(resonance_threshold)),
        metadata=_json_clone(metadata or {}),
    )


def _vector_with_offset(reference_vector_9d: Tuple[float, ...], offset: Iterable[Any]) -> Tuple[float, ...]:
    offset_values = list(offset)
    if len(offset_values) != 9:
        raise ValueError("offset must contain exactly 9 values")
    result: List[float] = []
    for index, (base_value, offset_value) in enumerate(zip(reference_vector_9d, offset_values)):
        if index == 0:
            result.append(float(wrap_turns(base_value + safe_float(offset_value, 0.0))))
        else:
            result.append(float(clamp01(base_value + safe_float(offset_value, 0.0))))
    return tuple(result)


def _quartet_with_bias(reference_quartet: Dict[str, Any], bias: Dict[str, Any]) -> Dict[str, float]:
    quartet = _normalize_quartet(reference_quartet)
    return {
        name: float(clamp01(quartet[name] + safe_float(bias.get(name, 0.0), 0.0)))
        for name in quartet
    }


@dataclass
class CarrierWave9D:
    carrier_id: str
    label: str
    vector_9d: Tuple[float, ...]
    activation_quartet: Dict[str, float]
    resonance_threshold: float = 0.92
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector_9d = _normalize_vector_9d(self.vector_9d)
        self.activation_quartet = _normalize_quartet(self.activation_quartet)
        self.resonance_threshold = float(clamp01(self.resonance_threshold))
        self.metadata = _json_clone(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "carrier_id": str(self.carrier_id),
            "label": str(self.label),
            "vector_9d": list(self.vector_9d),
            "activation_quartet": dict(self.activation_quartet),
            "resonance_threshold": float(self.resonance_threshold),
            "metadata": _json_clone(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CarrierWave9D":
        data = dict(payload or {})
        return cls(
            carrier_id=str(data.get("carrier_id", _carrier_id(str(data.get("label", "carrier")), data.get("vector_9d", [0.0] * 9)))),
            label=str(data.get("label", "carrier")),
            vector_9d=_normalize_vector_9d(data.get("vector_9d", [0.0] * 9)),
            activation_quartet=_normalize_quartet(data.get("activation_quartet", {})),
            resonance_threshold=float(clamp01(data.get("resonance_threshold", 0.92))),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class StaticBaseTone:
    tone_label: str = "photonic_base_tone"
    reference_vector_9d: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 9)
    carriers: Dict[str, CarrierWave9D] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reference_vector_9d = _normalize_vector_9d(self.reference_vector_9d)

    def register_carrier(self, carrier: CarrierWave9D) -> None:
        self.carriers[str(carrier.carrier_id)] = carrier

    def ordered_carriers(self) -> List[CarrierWave9D]:
        return [self.carriers[key] for key in sorted(self.carriers)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tone_label": str(self.tone_label),
            "reference_vector_9d": list(self.reference_vector_9d),
            "carriers": [carrier.to_dict() for carrier in self.ordered_carriers()],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StaticBaseTone":
        data = dict(payload or {})
        tone = cls(
            tone_label=str(data.get("tone_label", "photonic_base_tone")),
            reference_vector_9d=_normalize_vector_9d(data.get("reference_vector_9d", [0.0] * 9)),
        )
        for carrier_payload in list(data.get("carriers", []) or []):
            tone.register_carrier(CarrierWave9D.from_dict(dict(carrier_payload or {})))
        return tone


def build_default_static_base_tone(
    reference_payload: Dict[str, Any],
    tone_label: str = "photonic_base_tone",
) -> StaticBaseTone:
    payload = dict(reference_payload or {})
    reference_vector_9d = build_process_vector_9d(payload)
    reference_quartet = dict(
        dict(payload.get("encoding_activation_path", {}) or {}).get(
            "required_activation_pulse",
            dict(dict(payload.get("transport_prediction", {}) or {}).get("next_pulse_quartet", {}) or {}),
        )
        or {}
    )
    photonic_identity = str(dict(payload.get("photonic_identity", {}) or {}).get("photonic_identity", ""))
    tone = StaticBaseTone(tone_label=str(tone_label), reference_vector_9d=reference_vector_9d)
    for index, (label, profile) in enumerate(DEFAULT_STATIC_PROCESS_PROFILES.items()):
        carrier = make_carrier_wave(
            label=label,
            vector_9d=_vector_with_offset(reference_vector_9d, profile.get("vector_offset", [0.0] * 9)),
            activation_quartet=_quartet_with_bias(reference_quartet, profile.get("pulse_bias", {})),
            resonance_threshold=float(clamp01(profile.get("threshold", 0.90))),
            metadata={
                "profile_index": int(index),
                "profile_name": str(label),
                "reference_identity": photonic_identity,
                "reference_vector_axes": list(PHOTONIC_VECTOR_AXES),
            },
        )
        tone.register_carrier(carrier)
    return tone


@dataclass
class PhotonicTextureMap9D:
    texture_id: str = "photonic_texture_9d"
    base_tone: StaticBaseTone = field(default_factory=StaticBaseTone)
    saved_resume_state: CarrierWave9D | None = None
    dynamic_trajectories: Dict[str, CarrierWave9D] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def attach_base_tone(self, base_tone: StaticBaseTone) -> None:
        self.base_tone = base_tone

    def capture_resume_state(
        self,
        payload: Dict[str, Any],
        label: str = "resume_state",
        resonance_threshold: float = 0.92,
        metadata: Dict[str, Any] | None = None,
    ) -> CarrierWave9D:
        data = dict(payload or {})
        activation_quartet = dict(
            dict(data.get("encoding_activation_path", {}) or {}).get(
                "required_activation_pulse",
                dict(dict(data.get("transport_prediction", {}) or {}).get("next_pulse_quartet", {}) or {}),
            )
            or {}
        )
        carrier_metadata = dict(metadata or {})
        carrier_metadata["photonic_identity"] = str(dict(data.get("photonic_identity", {}) or {}).get("photonic_identity", ""))
        carrier_metadata["trajectory_spectral_id_u64"] = int(dict(data.get("photonic_identity", {}) or {}).get("trajectory_spectral_id_u64", 0))
        carrier = make_carrier_wave(
            label=label,
            vector_9d=build_process_vector_9d(data),
            activation_quartet=activation_quartet,
            resonance_threshold=resonance_threshold,
            metadata=carrier_metadata,
        )
        self.saved_resume_state = carrier
        return carrier

    def register_dynamic_trajectory(
        self,
        label: str,
        vector_9d: Iterable[Any],
        activation_quartet: Dict[str, Any],
        resonance_threshold: float = 0.90,
        metadata: Dict[str, Any] | None = None,
    ) -> CarrierWave9D:
        carrier = make_carrier_wave(
            label=label,
            vector_9d=vector_9d,
            activation_quartet=activation_quartet,
            resonance_threshold=resonance_threshold,
            metadata=metadata,
        )
        self.dynamic_trajectories[str(carrier.carrier_id)] = carrier
        return carrier

    def compose_field_map(self) -> Dict[str, Any]:
        weighted_entries: List[Tuple[CarrierWave9D, float, str]] = []
        for carrier in self.base_tone.ordered_carriers():
            weighted_entries.append((carrier, 1.0, "base"))
        if self.saved_resume_state is not None:
            weighted_entries.append((self.saved_resume_state, 1.25, "resume"))
        for carrier in [self.dynamic_trajectories[key] for key in sorted(self.dynamic_trajectories)]:
            weighted_entries.append((carrier, 0.85, "dynamic"))

        if not weighted_entries:
            aggregate_vector_9d = (0.0,) * 9
        else:
            weights = [entry[1] for entry in weighted_entries]
            phase_values = [entry[0].vector_9d[0] for entry in weighted_entries]
            aggregate_vector: List[float] = [_circular_mean_turns(phase_values, weights)]
            total_weight = max(sum(weights), 1.0e-9)
            for index in range(1, 9):
                aggregate_vector.append(
                    float(
                        sum(entry[0].vector_9d[index] * entry[1] for entry in weighted_entries) / total_weight
                    )
                )
            aggregate_vector_9d = _normalize_vector_9d(aggregate_vector)

        carrier_field_energy = float(
            math.sqrt(sum(value * value for value in aggregate_vector_9d)) / math.sqrt(float(len(aggregate_vector_9d) or 1))
        )
        return {
            "texture_id": str(self.texture_id),
            "carrier_field_axes": list(PHOTONIC_VECTOR_AXES),
            "carrier_field_vector_9d": list(aggregate_vector_9d),
            "carrier_field_energy": float(clamp01(carrier_field_energy)),
            "base_carrier_count": int(len(self.base_tone.carriers)),
            "dynamic_carrier_count": int(len(self.dynamic_trajectories)),
            "carrier_count": int(len(self.base_tone.carriers) + len(self.dynamic_trajectories) + (1 if self.saved_resume_state else 0)),
            "saved_resume_present": bool(self.saved_resume_state is not None),
            "saved_resume_label": str(self.saved_resume_state.label) if self.saved_resume_state else "",
            "saved_vector_9d": list(self.saved_resume_state.vector_9d) if self.saved_resume_state else list(aggregate_vector_9d),
            "saved_activation_pulse": dict(self.saved_resume_state.activation_quartet) if self.saved_resume_state else {},
            "saved_identity": str(dict((self.saved_resume_state.metadata if self.saved_resume_state else {}) or {}).get("photonic_identity", "")),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texture_id": str(self.texture_id),
            "base_tone": self.base_tone.to_dict(),
            "saved_resume_state": self.saved_resume_state.to_dict() if self.saved_resume_state else None,
            "dynamic_trajectories": [
                self.dynamic_trajectories[key].to_dict() for key in sorted(self.dynamic_trajectories)
            ],
            "metadata": _json_clone(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PhotonicTextureMap9D":
        data = dict(payload or {})
        texture = cls(
            texture_id=str(data.get("texture_id", "photonic_texture_9d")),
            base_tone=StaticBaseTone.from_dict(dict(data.get("base_tone", {}) or {})),
            metadata=dict(data.get("metadata", {}) or {}),
        )
        saved_payload = data.get("saved_resume_state")
        if isinstance(saved_payload, dict):
            texture.saved_resume_state = CarrierWave9D.from_dict(saved_payload)
        for dynamic_payload in list(data.get("dynamic_trajectories", []) or []):
            carrier = CarrierWave9D.from_dict(dict(dynamic_payload or {}))
            texture.dynamic_trajectories[str(carrier.carrier_id)] = carrier
        return texture


class PhaseTransportObserver:
    def __init__(self, similarity_threshold: float = 0.92, max_resync_steps: int = 4) -> None:
        self.similarity_threshold = float(clamp01(similarity_threshold))
        self.max_resync_steps = max(int(max_resync_steps), 1)

    def observe_live_payload(
        self,
        quartet: Dict[str, Any],
        phase_turns: Any,
        previous_phase_turns: Any,
        telemetry: Dict[str, Any] | None = None,
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return build_live_telemetry_payload(
            quartet=quartet,
            phase_turns=phase_turns,
            previous_phase_turns=previous_phase_turns,
            telemetry=telemetry,
            schema=schema,
        )

    def build_resync_plan(
        self,
        texture_map: PhotonicTextureMap9D,
        quartet: Dict[str, Any],
        phase_turns: Any,
        previous_phase_turns: Any,
        telemetry: Dict[str, Any] | None = None,
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        live_payload = self.observe_live_payload(
            quartet=quartet,
            phase_turns=phase_turns,
            previous_phase_turns=previous_phase_turns,
            telemetry=telemetry,
            schema=schema,
        )
        field_map = texture_map.compose_field_map()
        live_vector_9d = build_process_vector_9d(live_payload)
        target_carrier = texture_map.saved_resume_state
        target_vector_9d = (
            target_carrier.vector_9d
            if target_carrier is not None
            else tuple(field_map.get("carrier_field_vector_9d", [0.0] * 9))
        )
        live_activation = dict(
            dict(live_payload.get("encoding_activation_path", {}) or {}).get("required_activation_pulse", {}) or {}
        )
        target_activation = (
            dict(target_carrier.activation_quartet)
            if target_carrier is not None
            else dict(field_map.get("saved_activation_pulse", {}) or {})
        ) or dict(live_activation)
        similarity = compare_temporal_trajectory(live_vector_9d, target_vector_9d)
        overall_similarity = float(similarity.get("overall_similarity", 0.0))
        similarity_gap = float(clamp01(1.0 - overall_similarity))
        actuation_required = bool(overall_similarity < self.similarity_threshold)
        estimated_resync_steps = 0
        if actuation_required:
            normalized_gap = 0.0
            if self.similarity_threshold < 1.0:
                normalized_gap = similarity_gap / max(1.0 - self.similarity_threshold, 1.0e-9)
            estimated_resync_steps = min(self.max_resync_steps, max(1, int(math.ceil(normalized_gap))))

        pulse_sequence: List[Dict[str, Any]] = []
        for step_index in range(estimated_resync_steps):
            step_fraction = float(step_index + 1) / float(estimated_resync_steps)
            target_step_vector_9d = _interpolate_vector_9d(live_vector_9d, target_vector_9d, step_fraction)
            pulse_quartet = _interpolate_quartet(live_activation, target_activation, step_fraction)
            step_similarity = compare_temporal_trajectory(target_step_vector_9d, target_vector_9d)
            pulse_sequence.append({
                "step_index": int(step_index),
                "phase_turns_target": float(target_step_vector_9d[0]),
                "pulse_quartet": pulse_quartet,
                "expected_similarity": float(step_similarity.get("overall_similarity", 0.0)),
                "target_vector_9d": list(target_step_vector_9d),
            })

        drift_vector_9d = [
            float(signed_turn_delta(target_vector_9d[0], live_vector_9d[0])) if index == 0 else float(target - live)
            for index, (live, target) in enumerate(zip(live_vector_9d, target_vector_9d))
        ]
        return {
            "live_photonic_identity": str(dict(live_payload.get("photonic_identity", {}) or {}).get("photonic_identity", "")),
            "target_photonic_identity": str(dict((target_carrier.metadata if target_carrier else {}) or {}).get("photonic_identity", field_map.get("saved_identity", ""))),
            "target_carrier_id": str(target_carrier.carrier_id) if target_carrier else "",
            "target_label": str(target_carrier.label) if target_carrier else "carrier_field_map",
            "actuation_required": bool(actuation_required),
            "estimated_resync_steps": int(estimated_resync_steps),
            "similarity": similarity,
            "similarity_gap": float(similarity_gap),
            "live_vector_9d": list(live_vector_9d),
            "target_vector_9d": list(target_vector_9d),
            "drift_vector_9d": drift_vector_9d,
            "pulse_sequence": pulse_sequence,
            "field_map": field_map,
            "live_payload": live_payload,
        }


class TemporalAccountingScheduler:
    def __init__(self, similarity_threshold: float = 0.90, minimum_temporal_accuracy: float = 0.65) -> None:
        self.similarity_threshold = float(clamp01(similarity_threshold))
        self.minimum_temporal_accuracy = float(clamp01(minimum_temporal_accuracy))

    def _iter_carriers(self, texture_map: PhotonicTextureMap9D) -> List[Tuple[str, CarrierWave9D]]:
        carriers: List[Tuple[str, CarrierWave9D]] = []
        for carrier in texture_map.base_tone.ordered_carriers():
            carriers.append(("base", carrier))
        if texture_map.saved_resume_state is not None:
            carriers.append(("resume", texture_map.saved_resume_state))
        for key in sorted(texture_map.dynamic_trajectories):
            carriers.append(("dynamic", texture_map.dynamic_trajectories[key]))
        return carriers

    def evaluate_matches(self, live_payload: Dict[str, Any], texture_map: PhotonicTextureMap9D) -> List[Dict[str, Any]]:
        payload = dict(live_payload or {})
        live_vector_9d = build_process_vector_9d(payload)
        live_temporal_accuracy = float(
            dict(payload.get("temporal_accounting", {}) or {}).get("temporal_accuracy_score", 0.0)
        )
        live_activation = dict(
            dict(payload.get("encoding_activation_path", {}) or {}).get("required_activation_pulse", {}) or {}
        )
        source_priority = {"dynamic": 0, "resume": 1, "base": 2}
        matches: List[Dict[str, Any]] = []
        for carrier_source, carrier in self._iter_carriers(texture_map):
            similarity = compare_temporal_trajectory(live_vector_9d, carrier.vector_9d)
            threshold = max(self.similarity_threshold, float(carrier.resonance_threshold))
            gate_open = bool(live_temporal_accuracy >= self.minimum_temporal_accuracy)
            should_actuate = bool(gate_open and float(similarity.get("overall_similarity", 0.0)) >= threshold)
            pulse_fraction = clamp01((float(similarity.get("overall_similarity", 0.0)) + live_temporal_accuracy) * 0.5)
            recommended_pulse = _interpolate_quartet(live_activation or carrier.activation_quartet, carrier.activation_quartet, pulse_fraction)
            matches.append({
                "carrier_id": str(carrier.carrier_id),
                "label": str(carrier.label),
                "carrier_source": str(carrier_source),
                "threshold": float(threshold),
                "scheduler_gate_open": bool(gate_open),
                "actuate": bool(should_actuate),
                "live_temporal_accuracy": float(live_temporal_accuracy),
                "activation_quartet": recommended_pulse,
                "carrier_vector_9d": list(carrier.vector_9d),
                **similarity,
            })
        matches.sort(
            key=lambda item: (
                -float(item.get("overall_similarity", 0.0)),
                source_priority.get(str(item.get("carrier_source", "base")), 9),
                str(item.get("carrier_id", "")),
            )
        )
        return matches

    def schedule(self, live_payload: Dict[str, Any], texture_map: PhotonicTextureMap9D) -> Dict[str, Any]:
        matches = self.evaluate_matches(live_payload, texture_map)
        field_map = texture_map.compose_field_map()
        actuation_events = [match for match in matches if bool(match.get("actuate", False))]
        recommended_match = actuation_events[0] if actuation_events else (matches[0] if matches else {})
        return {
            "scheduler_gate_open": bool(
                float(dict(live_payload.get("temporal_accounting", {}) or {}).get("temporal_accuracy_score", 0.0))
                >= self.minimum_temporal_accuracy
            ),
            "live_vector_9d": list(build_process_vector_9d(live_payload)),
            "field_map": field_map,
            "carrier_count": int(len(matches)),
            "recommended_match": _json_clone(recommended_match),
            "actuation_events": _json_clone(actuation_events),
            "matches": _json_clone(matches),
        }


__all__ = [
    "PHOTONIC_VECTOR_AXES",
    "CarrierWave9D",
    "StaticBaseTone",
    "PhotonicTextureMap9D",
    "PhaseTransportObserver",
    "TemporalAccountingScheduler",
    "build_process_vector_9d",
    "build_default_static_base_tone",
    "compare_temporal_trajectory",
    "make_carrier_wave",
]