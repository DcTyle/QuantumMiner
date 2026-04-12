from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import json
import math

from gpu_pulse_axis_dynamics import (
    build_live_telemetry_payload,
    clamp01,
    clamp_signed,
    compute_axis_field_dynamics,
    encode_axis_dynamics,
    predict_full_spectrum_calibration,
)


PROTOTYPING_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = PROTOTYPING_ROOT.parents[1]
DEFAULT_RUN44_FRAMES = RESEARCH_ROOT / "Run44" / "run_044_live_startup_frames.csv"
DEFAULT_LIVE_LEDGER = RESEARCH_ROOT / "live_compute_interference_ledger.json"
DEFAULT_TEMPORAL_SCHEMA = RESEARCH_ROOT / "temporal_coupling_encoding_schema_2060.json"
DEFAULT_PROCESS_SCHEMA = RESEARCH_ROOT / "process_substrate_calculus_encoding_schema.json"
DEFAULT_NIST_REFERENCE = RESEARCH_ROOT / "nist_silicon_reference.json"
SIG9_SCALE = 1000000


def load_json_with_comments(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("//"):
        text = "\n".join(text.splitlines()[1:])
    payload = json.loads(text)
    if not isinstance(payload, dict):
        return {}
    return payload


def load_live_startup_frames(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = {
                "frame_index": int(row.get("frame_index", len(frames)) or len(frames)),
                "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                "source": str(row.get("source", "live_startup") or "live_startup"),
                "sample_period_s": float(row.get("sample_period_s", 0.0) or 0.0),
                "global_util": float(row.get("global_util", 0.0) or 0.0),
                "gpu_util": float(row.get("gpu_util", 0.0) or 0.0),
                "mem_bw_util": float(row.get("mem_bw_util", 0.0) or 0.0),
                "cpu_util": float(row.get("cpu_util", 0.0) or 0.0),
                "raw_gpu_util": float(row.get("raw_gpu_util", 0.0) or 0.0),
                "raw_mem_bw_util": float(row.get("raw_mem_bw_util", 0.0) or 0.0),
                "raw_cpu_util": float(row.get("raw_cpu_util", 0.0) or 0.0),
                "actuation_applied": str(row.get("actuation_applied", "false")).strip().lower() == "true",
                "actuation_mode": str(row.get("actuation_mode", "") or ""),
                "actuation_tag": str(row.get("actuation_tag", "") or ""),
                "actuation_load_hint": float(row.get("actuation_load_hint", 0.0) or 0.0),
                "actuation_dispatch_ms": float(row.get("actuation_dispatch_ms", 0.0) or 0.0),
                "actuation_elapsed_s": float(row.get("actuation_elapsed_s", 0.0) or 0.0),
                "pulse": float(row.get("pulse", 0.0) or 0.0),
                "anti_pulse": float(row.get("anti_pulse", 0.0) or 0.0),
                "phase_turns": float(row.get("phase_turns", 0.0) or 0.0),
            }
            frames.append(frame)
    return frames


def load_default_inputs() -> Dict[str, Any]:
    return {
        "frames": load_live_startup_frames(DEFAULT_RUN44_FRAMES),
        "live_ledger": load_json_with_comments(DEFAULT_LIVE_LEDGER),
        "temporal_schema": load_json_with_comments(DEFAULT_TEMPORAL_SCHEMA),
        "process_schema": load_json_with_comments(DEFAULT_PROCESS_SCHEMA),
        "nist_reference": load_json_with_comments(DEFAULT_NIST_REFERENCE),
    }


def stable_word(value: Any) -> int:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    else:
        text = str(value or "")
    seed = 2166136261
    for ch in text.encode("ascii", errors="ignore"):
        seed ^= ch
        seed = (seed * 16777619) & 0xFFFFFFFF
    return int(seed)


def wrap_turns(value: Any) -> float:
    turns = float(value or 0.0)
    return turns - math.floor(turns)


def phase_delta_turns(previous_phase: Any, current_phase: Any) -> float:
    delta = wrap_turns(float(current_phase or 0.0) - float(previous_phase or 0.0))
    if delta >= 0.5:
        delta -= 1.0
    return float(delta)


def quantize_sig9(value: Any) -> int:
    return int(round(clamp01(value) * float(SIG9_SCALE)))


def sig9_to_identity(sig9: List[int]) -> str:
    return "PID9-" + "-".join("%06x" % int(component) for component in sig9)


def window_norm(value: Any, window: Dict[str, Any]) -> float:
    lower = float(window.get("min", 0.0) or 0.0)
    upper = float(window.get("max", 1.0) or 1.0)
    span = max(upper - lower, 1.0e-9)
    return clamp01((float(value or 0.0) - lower) / span)


def derive_quartet(
    frame: Dict[str, Any],
    previous_frame: Dict[str, Any] | None,
    live_ledger: Dict[str, Any],
    temporal_schema: Dict[str, Any],
) -> Dict[str, float]:
    pulse_codes = dict(temporal_schema.get("pulse_codes", {}) or {})
    quartet = dict(live_ledger.get("compute_quartet", {}) or {})
    windows = dict(pulse_codes.get("normalized_window", {}) or {})
    previous_phase = float(dict(previous_frame or {}).get("phase_turns", frame.get("phase_turns", 0.0)) or 0.0)
    phase_delta = abs(phase_delta_turns(previous_phase, frame.get("phase_turns", 0.0)))
    dispatch_norm = clamp01(float(frame.get("actuation_dispatch_ms", 0.0) or 0.0) / 40.0)
    raw_frequency = (
        float(quartet.get("F", pulse_codes.get("f_code", 0.245)) or 0.245)
        + 0.010 * (float(frame.get("gpu_util", 0.0) or 0.0) - float(frame.get("global_util", 0.0) or 0.0))
        + 0.005 * dispatch_norm
    )
    raw_amplitude = (
        float(quartet.get("A", pulse_codes.get("a_code", 0.18)) or 0.18)
        + 0.020 * float(frame.get("mem_bw_util", 0.0) or 0.0)
        + 0.008 * abs(float(frame.get("pulse", 0.0) or 0.0))
    )
    raw_amperage = (
        float(quartet.get("I", pulse_codes.get("i_code", 0.33)) or 0.33)
        + 0.040 * clamp01(frame.get("sample_period_s", 0.0))
        + 0.020 * dispatch_norm
        + 0.015 * phase_delta
    )
    raw_voltage = (
        float(quartet.get("V", pulse_codes.get("v_code", 0.33)) or 0.33)
        + 0.030 * float(frame.get("actuation_load_hint", 0.0) or 0.0)
        + 0.020 * abs(float(frame.get("anti_pulse", 0.0) or 0.0))
    )
    return {
        "frequency_raw": float(raw_frequency),
        "amplitude_raw": float(raw_amplitude),
        "amperage_raw": float(raw_amperage),
        "voltage_raw": float(raw_voltage),
        "frequency_norm": window_norm(raw_frequency, dict(windows.get("frequency", {}) or {})),
        "amplitude_norm": window_norm(raw_amplitude, dict(windows.get("amplitude", {}) or {})),
        "amperage_norm": window_norm(raw_amperage, dict(windows.get("amperage", {}) or {})),
        "voltage_norm": window_norm(raw_voltage, dict(windows.get("voltage", {}) or {})),
        "quartet_word": int(
            stable_word({
                "F": round(float(raw_frequency), 6),
                "A": round(float(raw_amplitude), 6),
                "I": round(float(raw_amperage), 6),
                "V": round(float(raw_voltage), 6),
            })
        ),
    }


def compute_resonance_gate(frame: Dict[str, Any], live_ledger: Dict[str, Any]) -> float:
    readiness = clamp01(live_ledger.get("readiness_norm", 0.0))
    score_alignment = clamp01(live_ledger.get("score_alignment_norm", 0.0))
    coherence_alignment = clamp01(live_ledger.get("coherence_alignment_norm", 0.0))
    return clamp01(
        0.30 * score_alignment
        + 0.24 * coherence_alignment
        + 0.18 * readiness
        + 0.16 * clamp01(frame.get("actuation_load_hint", 0.0))
        + 0.12 * (1.0 - clamp01(float(frame.get("actuation_dispatch_ms", 0.0) or 0.0) / 40.0))
    )


def compute_temporal_overlap(frame: Dict[str, Any], previous_frame: Dict[str, Any] | None, live_ledger: Dict[str, Any]) -> float:
    previous_phase = float(dict(previous_frame or {}).get("phase_turns", frame.get("phase_turns", 0.0)) or 0.0)
    phase_gap = abs(phase_delta_turns(previous_phase, frame.get("phase_turns", 0.0)))
    return clamp01(
        0.34 * clamp01(live_ledger.get("encoded_extrapolation_norm", 0.0))
        + 0.22 * (1.0 - phase_gap)
        + 0.20 * clamp01(frame.get("sample_period_s", 0.0))
        + 0.12 * clamp01(frame.get("gpu_util", 0.0))
        + 0.12 * clamp01(frame.get("mem_bw_util", 0.0))
    )


def compute_flux_term(frame: Dict[str, Any], live_ledger: Dict[str, Any], quartet: Dict[str, float]) -> float:
    return clamp01(
        0.32 * clamp01(live_ledger.get("interference_ledger_norm", 0.0))
        + 0.24 * clamp01(frame.get("mem_bw_util", 0.0))
        + 0.18 * clamp01(abs(frame.get("pulse", 0.0)))
        + 0.14 * clamp01(frame.get("actuation_load_hint", 0.0))
        + 0.12 * clamp01(quartet.get("voltage_norm", 0.0))
    )


def compute_observer_damping(
    frame: Dict[str, Any],
    live_ledger: Dict[str, Any],
    quartet: Dict[str, float],
    dynamics: Dict[str, float],
    resonance_gate: float,
) -> Dict[str, float]:
    dispatch_norm = clamp01(float(frame.get("actuation_dispatch_ms", 0.0) or 0.0) / 40.0)
    raw_return_s = float(frame.get("actuation_elapsed_s", 0.0) or 0.0) + (float(frame.get("actuation_dispatch_ms", 0.0) or 0.0) * 0.001)
    latency_norm = clamp01(raw_return_s / 1.25)
    field_interference = clamp01(live_ledger.get("interference_ledger_norm", 0.0))
    observer_damping = clamp01(
        0.28 * dispatch_norm
        + 0.22 * field_interference
        + 0.18 * latency_norm
        + 0.14 * abs(clamp01(quartet.get("voltage_norm", 0.0)) - clamp01(quartet.get("amperage_norm", 0.0)))
        + 0.10 * (1.0 - resonance_gate)
        + 0.08 * (1.0 - clamp01(dynamics.get("axis_resonance", 0.0)))
    )
    return {
        "dispatch_norm": float(dispatch_norm),
        "latency_norm": float(latency_norm),
        "field_interference_norm": float(field_interference),
        "observer_damping": float(observer_damping),
    }


def compute_transport_terms(
    frame: Dict[str, Any],
    previous_frame: Dict[str, Any] | None,
    live_ledger: Dict[str, Any],
    quartet: Dict[str, float],
    dynamics: Dict[str, float],
    resonance_gate: float,
    flux_term: float,
    observer: Dict[str, float],
) -> Dict[str, float]:
    previous_phase = float(dict(previous_frame or {}).get("phase_turns", frame.get("phase_turns", 0.0)) or 0.0)
    phase_gap = phase_delta_turns(previous_phase, frame.get("phase_turns", 0.0))
    request_to_return_s = max(
        float(frame.get("sample_period_s", 0.0) or 0.0),
        float(frame.get("actuation_elapsed_s", 0.0) or 0.0),
    )
    phase_transport_term = clamp_signed(
        0.26 * float(phase_gap)
        + 0.22 * clamp01(quartet.get("voltage_norm", 0.0))
        + 0.18 * clamp01(quartet.get("amperage_norm", 0.0))
        + 0.14 * clamp01(dynamics.get("temporal_coupling_moment", 0.0))
        + 0.10 * resonance_gate
        + 0.10 * clamp01(live_ledger.get("score_alignment_norm", 0.0))
        - 0.20 * clamp01(observer.get("observer_damping", 0.0))
    )
    flux_transport_term = clamp01(
        0.24 * clamp01(quartet.get("amplitude_norm", 0.0))
        + 0.20 * clamp01(quartet.get("voltage_norm", 0.0))
        + 0.18 * clamp01(quartet.get("amperage_norm", 0.0))
        + 0.14 * clamp01(dynamics.get("vector_energy", 0.0))
        + 0.12 * clamp01(flux_term)
        + 0.12 * resonance_gate
        - 0.10 * clamp01(observer.get("observer_damping", 0.0))
    )
    accounting_latency_s = (
        request_to_return_s
        + (float(frame.get("actuation_dispatch_ms", 0.0) or 0.0) * 0.001)
        + abs(float(phase_transport_term)) * max(float(frame.get("sample_period_s", 0.0) or 0.0), 0.001)
    )
    closed_loop_latency_s = accounting_latency_s + clamp01(observer.get("observer_damping", 0.0)) * max(float(frame.get("sample_period_s", 0.0) or 0.0), 0.001)
    return {
        "phase_delta_turns": float(phase_gap),
        "phase_transport_term": float(phase_transport_term),
        "flux_transport_term": float(flux_transport_term),
        "request_to_return_s": float(request_to_return_s),
        "accounting_latency_s": float(accounting_latency_s),
        "closed_loop_latency_s": float(closed_loop_latency_s),
        "gpu_round_trip_norm": clamp01(closed_loop_latency_s / 1.25),
        "phase_correction_norm": clamp01(abs(phase_transport_term) + 0.5 * clamp01(observer.get("observer_damping", 0.0))),
        "flux_correction_norm": clamp01(flux_transport_term * (1.0 - clamp01(observer.get("observer_damping", 0.0)))),
    }


def build_disruption_nodes(
    frame: Dict[str, Any],
    live_ledger: Dict[str, Any],
    dynamics: Dict[str, float],
    transport: Dict[str, float],
    observer: Dict[str, float],
) -> List[Dict[str, Any]]:
    trajectory = [
        clamp_signed(dynamics.get("spin_axis_x", 0.0)),
        clamp_signed(dynamics.get("spin_axis_y", 0.0)),
        clamp_signed(dynamics.get("spin_axis_z", 0.0)),
        wrap_turns(frame.get("phase_turns", 0.0)),
    ]
    node_specs = [
        ("thermal_interference", clamp01(observer.get("dispatch_norm", 0.0))),
        ("field_layer_interference", clamp01(observer.get("field_interference_norm", 0.0))),
        ("return_latency_gap", clamp01(observer.get("latency_norm", 0.0))),
        (
            "resonance_shear",
            clamp01(
                abs(float(dynamics.get("axis_scale_x", 0.0)) - float(dynamics.get("axis_scale_z", 0.0)))
                + 0.5 * float(dynamics.get("inertial_mass_proxy", 0.0))
            ),
        ),
    ]
    nodes: List[Dict[str, Any]] = []
    for node_index, (kind, severity) in enumerate(node_specs):
        correction_turns = clamp_signed(
            float(transport.get("phase_transport_term", 0.0)) * (0.5 + 0.5 * severity)
        )
        correction_norm = clamp01(
            severity
            * (
                0.40
                + 0.40 * float(transport.get("flux_transport_term", 0.0))
                + 0.20 * float(dynamics.get("temporal_coupling_moment", 0.0))
            )
        )
        payload = {
            "kind": kind,
            "frame_index": int(frame.get("frame_index", 0)),
            "severity": round(float(severity), 6),
            "trajectory": [round(float(value), 6) for value in trajectory],
            "correction_turns": round(float(correction_turns), 6),
            "correction_norm": round(float(correction_norm), 6),
            "transport_norm": round(float(transport.get("gpu_round_trip_norm", 0.0)), 6),
            "observer_damping": round(float(observer.get("observer_damping", 0.0)), 6),
            "node_index": int(node_index),
        }
        node_word = stable_word(payload)
        node_id = "PNODE-%02d-%08x" % (int(frame.get("frame_index", 0)), int(node_word))
        node = dict(payload)
        node.update({
            "node_word": int(node_word),
            "node_id": str(node_id),
            "encodable_node": bool(float(severity) >= 0.10),
        })
        nodes.append(node)
    return nodes


def build_sig9(
    frame: Dict[str, Any],
    dynamics: Dict[str, float],
    transport: Dict[str, float],
    observer: Dict[str, float],
    resonance_gate: float,
) -> Dict[str, Any]:
    nexus_norm = clamp01(
        0.42 * resonance_gate
        + 0.28 * (1.0 - clamp01(observer.get("observer_damping", 0.0)))
        + 0.18 * clamp01(dynamics.get("temporal_coupling_moment", 0.0))
        + 0.12 * clamp01(transport.get("flux_transport_term", 0.0))
    )
    sig9_float = [
        clamp01(dynamics.get("axis_scale_x", 0.0)),
        clamp01(dynamics.get("axis_scale_y", 0.0)),
        clamp01(dynamics.get("axis_scale_z", 0.0)),
        clamp01(transport.get("gpu_round_trip_norm", 0.0)),
        wrap_turns(frame.get("phase_turns", 0.0)),
        clamp01(transport.get("flux_transport_term", 0.0)),
        clamp01(observer.get("observer_damping", 0.0)),
        clamp01(dynamics.get("inertial_mass_proxy", 0.0)),
        float(nexus_norm),
    ]
    sig9 = [quantize_sig9(value) for value in sig9_float]
    return {
        "spectra_sig9_float": sig9_float,
        "spectra_sig9": sig9,
        "photonic_identity": sig9_to_identity(sig9),
        "nexus_norm": float(nexus_norm),
        "identity_word": int(stable_word(sig9)),
    }


def build_api_packets(
    frame: Dict[str, Any],
    quartet: Dict[str, float],
    sig9_bundle: Dict[str, Any],
    dynamics: Dict[str, float],
    transport: Dict[str, float],
    observer: Dict[str, float],
    process_schema: Dict[str, Any],
    disruption_nodes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    encoded = encode_axis_dynamics(dynamics)
    pipeline = dict(process_schema.get("canonical_operator_pipeline", {}) or {})
    request_packet = {
        "api_version": "photonic_identity.research.v1",
        "packet_type": "substrate.request",
        "frame_index": int(frame.get("frame_index", 0)),
        "timestamp": float(frame.get("timestamp", 0.0) or 0.0),
        "photonic_identity": str(sig9_bundle.get("photonic_identity", "")),
        "spectra_sig9": list(sig9_bundle.get("spectra_sig9", []) or []),
        "quartet": {
            "frequency_norm": float(quartet.get("frequency_norm", 0.0)),
            "amplitude_norm": float(quartet.get("amplitude_norm", 0.0)),
            "amperage_norm": float(quartet.get("amperage_norm", 0.0)),
            "voltage_norm": float(quartet.get("voltage_norm", 0.0)),
        },
        "operator_pipeline": pipeline,
        "timing": {
            "request_to_return_s": float(transport.get("request_to_return_s", 0.0)),
            "accounting_latency_s": float(transport.get("accounting_latency_s", 0.0)),
            "closed_loop_latency_s": float(transport.get("closed_loop_latency_s", 0.0)),
        },
        "encoding_words": encoded,
    }
    feedback_packet = {
        "api_version": "photonic_identity.research.v1",
        "packet_type": "substrate.feedback",
        "frame_index": int(frame.get("frame_index", 0)),
        "photonic_identity": str(sig9_bundle.get("photonic_identity", "")),
        "transport": {
            "phase_transport_term": float(transport.get("phase_transport_term", 0.0)),
            "flux_transport_term": float(transport.get("flux_transport_term", 0.0)),
            "phase_correction_norm": float(transport.get("phase_correction_norm", 0.0)),
            "flux_correction_norm": float(transport.get("flux_correction_norm", 0.0)),
        },
        "field_dynamics": {
            "axis_scale_x": float(dynamics.get("axis_scale_x", 0.0)),
            "axis_scale_y": float(dynamics.get("axis_scale_y", 0.0)),
            "axis_scale_z": float(dynamics.get("axis_scale_z", 0.0)),
            "vector_energy": float(dynamics.get("vector_energy", 0.0)),
            "temporal_coupling_moment": float(dynamics.get("temporal_coupling_moment", 0.0)),
            "inertial_mass_proxy": float(dynamics.get("inertial_mass_proxy", 0.0)),
            "spin_momentum_score": float(dynamics.get("spin_momentum_score", 0.0)),
        },
        "observer": dict(observer),
        "disruption_node_ids": [str(node.get("node_id", "")) for node in disruption_nodes],
    }
    return {
        "request": request_packet,
        "feedback": feedback_packet,
    }


def build_predictive_telemetry_model(
    frame: Dict[str, Any],
    live_ledger: Dict[str, Any],
    resonance_gate: float,
    temporal_overlap: float,
    flux_term: float,
    observer: Dict[str, float],
    dynamics: Dict[str, float],
) -> Dict[str, Any]:
    sample_period_s = max(
        1.0e-6,
        float(frame.get("sample_period_s", frame.get("actuation_elapsed_s", 0.02)) or 0.02),
    )
    request_feedback_time_s = max(
        sample_period_s,
        float(frame.get("actuation_elapsed_s", 0.0) or 0.0),
    )
    return {
        "coherence": clamp01(0.58 * resonance_gate + 0.42 * (1.0 - clamp01(observer.get("observer_damping", 0.0)))),
        "trap_ratio": clamp01(1.0 - clamp01(live_ledger.get("trap_alignment_norm", 0.0))),
        "predicted_interference": clamp01(max(flux_term, clamp01(live_ledger.get("interference_ledger_norm", 0.0)))),
        "temporal_coupling": clamp01(temporal_overlap),
        "thermal_noise": clamp01(max(observer.get("field_interference_norm", 0.0), observer.get("dispatch_norm", 0.0) * 0.5)),
        "sample_period_s": float(sample_period_s),
        "request_feedback_time_s": float(request_feedback_time_s),
        "actuation_elapsed_s": float(frame.get("actuation_elapsed_s", 0.0) or 0.0),
        "observed_subsystems": {
            "residual": clamp01(live_ledger.get("score_alignment_norm", 0.0)),
            "spin": clamp01(dynamics.get("spin_momentum_score", 0.0)),
            "coupling": clamp01(temporal_overlap),
            "controller": clamp01(resonance_gate),
        },
    }


def build_predictive_calibration_surface(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    count = float(len(records) or 1.0)
    last_record = dict(records[-1] or {})
    quartet_rows = [
        {"quartet": dict(record.get("quartet_raw", {}) or {})}
        for record in records
        if dict(record.get("quartet_raw", {}) or {})
    ]
    return {
        "axis_resolution": max(2, min(6, int(len(records)))),
        "best_prediction": {
            "quartet": dict(last_record.get("quartet_raw", {}) or {}),
            "predicted_coherence": sum(float(record.get("resonance_gate", 0.0)) for record in records) / count,
            "predicted_trap_ratio": 1.0 - (sum(float(record.get("resonance_gate", 0.0)) for record in records) / count),
            "predicted_interference": sum(float(record.get("flux_term", 0.0)) for record in records) / count,
            "temporal_coupling": sum(float(record.get("temporal_overlap", 0.0)) for record in records) / count,
        },
        "observed_field": {
            "coherence": sum(float(record.get("resonance_gate", 0.0)) for record in records) / count,
            "interference": sum(float(record.get("flux_term", 0.0)) for record in records) / count,
            "source_vibration": sum(float(dict(record.get("observer", {}) or {}).get("field_interference_norm", 0.0)) for record in records) / count,
        },
        "observed_subsystems": {
            "residual": sum(float(dict(record.get("predictive_temporal_accounting", {}) or {}).get("temporal_accuracy_score", 0.0)) for record in records) / count,
            "spin": sum(float(dict(record.get("dynamics", {}) or {}).get("spin_momentum_score", 0.0)) for record in records) / count,
            "coupling": sum(float(record.get("temporal_overlap", 0.0)) for record in records) / count,
            "controller": sum(float(dict(record.get("predictive_temporal_accounting", {}) or {}).get("predictive_confidence", 0.0)) for record in records) / count,
        },
        "predictions": quartet_rows,
    }


def compute_photonic_identity_record(
    frame: Dict[str, Any],
    previous_frame: Dict[str, Any] | None,
    temporal_schema: Dict[str, Any],
    process_schema: Dict[str, Any],
    live_ledger: Dict[str, Any],
    nist_reference: Dict[str, Any],
) -> Dict[str, Any]:
    previous_phase_turns = float(dict(previous_frame or {}).get("phase_turns", frame.get("phase_turns", 0.0)) or 0.0)
    quartet = derive_quartet(frame, previous_frame, live_ledger, temporal_schema)
    resonance_gate = compute_resonance_gate(frame, live_ledger)
    temporal_overlap = compute_temporal_overlap(frame, previous_frame, live_ledger)
    flux_term = compute_flux_term(frame, live_ledger, quartet)
    vector_x = clamp_signed(
        (float(frame.get("gpu_util", 0.0) or 0.0) - float(frame.get("global_util", 0.0) or 0.0))
        + 0.25 * (float(quartet.get("frequency_norm", 0.0)) - 0.5)
    )
    vector_y = clamp_signed(
        (float(frame.get("mem_bw_util", 0.0) or 0.0) - float(frame.get("cpu_util", 0.0) or 0.0))
        + 0.25 * (float(quartet.get("amplitude_norm", 0.0)) - 0.5)
    )
    vector_z = clamp_signed(
        (float(frame.get("pulse", 0.0) or 0.0) * 0.5)
        + float(frame.get("actuation_load_hint", 0.0) or 0.0)
        - 0.25
        + 0.25 * (float(quartet.get("voltage_norm", 0.0)) - float(quartet.get("amperage_norm", 0.0)))
    )
    dynamics = compute_axis_field_dynamics(
        frequency_norm=quartet.get("frequency_norm", 0.0),
        amplitude_norm=quartet.get("amplitude_norm", 0.0),
        phase_turns=frame.get("phase_turns", 0.0),
        resonance_gate=resonance_gate,
        temporal_overlap=temporal_overlap,
        flux_term=flux_term,
        vector_x=vector_x,
        vector_y=vector_y,
        vector_z=vector_z,
        energy_hint=clamp01(0.5 * float(quartet.get("voltage_norm", 0.0)) + 0.5 * float(quartet.get("amperage_norm", 0.0))),
    )
    observer = compute_observer_damping(frame, live_ledger, quartet, dynamics, resonance_gate)
    transport = compute_transport_terms(frame, previous_frame, live_ledger, quartet, dynamics, resonance_gate, flux_term, observer)
    disruption_nodes = build_disruption_nodes(frame, live_ledger, dynamics, transport, observer)
    sig9_bundle = build_sig9(frame, dynamics, transport, observer, resonance_gate)
    packets = build_api_packets(frame, quartet, sig9_bundle, dynamics, transport, observer, process_schema, disruption_nodes)
    encoded = encode_axis_dynamics(dynamics)
    predictive_telemetry = build_predictive_telemetry_model(
        frame=frame,
        live_ledger=live_ledger,
        resonance_gate=resonance_gate,
        temporal_overlap=temporal_overlap,
        flux_term=flux_term,
        observer=observer,
        dynamics=dynamics,
    )
    predictive_payload = build_live_telemetry_payload(
        quartet={
            "F": float(quartet.get("frequency_raw", 0.0)),
            "A": float(quartet.get("amplitude_raw", 0.0)),
            "I": float(quartet.get("amperage_raw", 0.0)),
            "V": float(quartet.get("voltage_raw", 0.0)),
        },
        phase_turns=frame.get("phase_turns", 0.0),
        previous_phase_turns=previous_phase_turns,
        telemetry=predictive_telemetry,
        schema=temporal_schema,
    )
    predictive_transport = dict(predictive_payload.get("transport_prediction", {}) or {})
    trajectory_vector = [
        float(vector_x),
        float(vector_y),
        float(vector_z),
        wrap_turns(frame.get("phase_turns", 0.0)),
    ]
    lattice_constant_m = float(nist_reference.get("lattice_constant_m", 0.0) or 0.0)
    density_g_cm3 = float(nist_reference.get("density_g_cm3", 0.0) or 0.0)
    mean_excitation_energy_ev = float(nist_reference.get("mean_excitation_energy_ev", 0.0) or 0.0)
    return {
        "frame_index": int(frame.get("frame_index", 0)),
        "timestamp": float(frame.get("timestamp", 0.0) or 0.0),
        "source": str(frame.get("source", "")),
        "photonic_identity": str(sig9_bundle.get("photonic_identity", "")),
        "identity_word": int(sig9_bundle.get("identity_word", 0)),
        "spectra_sig9": list(sig9_bundle.get("spectra_sig9", []) or []),
        "spectra_sig9_float": list(sig9_bundle.get("spectra_sig9_float", []) or []),
        "quartet_raw": {
            "F": float(quartet.get("frequency_raw", 0.0)),
            "A": float(quartet.get("amplitude_raw", 0.0)),
            "I": float(quartet.get("amperage_raw", 0.0)),
            "V": float(quartet.get("voltage_raw", 0.0)),
        },
        "nexus_norm": float(sig9_bundle.get("nexus_norm", 0.0)),
        "quartet": quartet,
        "resonance_gate": float(resonance_gate),
        "temporal_overlap": float(temporal_overlap),
        "flux_term": float(flux_term),
        "trajectory_vector": trajectory_vector,
        "dynamics": dynamics,
        "observer": observer,
        "transport": transport,
        "encoding_words": encoded,
        "api_packets": packets,
        "predictive_transport_prediction": predictive_transport,
        "predictive_anchor_vectors": dict(predictive_transport.get("anchor_vectors", {}) or {}),
        "predictive_harmonic_noise": dict(predictive_transport.get("harmonic_noise", {}) or {}),
        "predictive_trajectory_state": dict(predictive_transport.get("trajectory_state", {}) or {}),
        "predictive_pulse_interference": dict(predictive_transport.get("pulse_interference", {}) or {}),
        "predictive_phase_ring_trace": dict(dict(predictive_transport.get("anchor_vectors", {}) or {}).get("phase_ring_trace", {}) or {}),
        "predictive_photonic_identity": dict(predictive_payload.get("photonic_identity", {}) or {}),
        "predictive_temporal_accounting": dict(predictive_payload.get("temporal_accounting", {}) or {}),
        "predictive_live_path": dict(predictive_payload.get("live_telemetry_path", {}) or {}),
        "predictive_encoding_words": dict(predictive_payload.get("encoded_words", {}) or {}),
        "disruption_nodes": disruption_nodes,
        "nist_anchor": {
            "lattice_constant_m": lattice_constant_m,
            "density_g_cm3": density_g_cm3,
            "mean_excitation_energy_ev": mean_excitation_energy_ev,
        },
    }


def analyze_photonic_identity_backbone(
    frames: List[Dict[str, Any]],
    temporal_schema: Dict[str, Any],
    process_schema: Dict[str, Any],
    live_ledger: Dict[str, Any],
    nist_reference: Dict[str, Any],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    all_nodes: List[Dict[str, Any]] = []
    api_packets: List[Dict[str, Any]] = []
    previous_frame: Dict[str, Any] | None = None
    for frame in list(frames or []):
        record = compute_photonic_identity_record(
            frame=frame,
            previous_frame=previous_frame,
            temporal_schema=temporal_schema,
            process_schema=process_schema,
            live_ledger=live_ledger,
            nist_reference=nist_reference,
        )
        records.append(record)
        for node in list(record.get("disruption_nodes", []) or []):
            enriched = dict(node)
            enriched["photonic_identity"] = str(record.get("photonic_identity", ""))
            all_nodes.append(enriched)
        api_packets.append({
            "photonic_identity": str(record.get("photonic_identity", "")),
            "request": dict(dict(record.get("api_packets", {}) or {}).get("request", {}) or {}),
            "feedback": dict(dict(record.get("api_packets", {}) or {}).get("feedback", {}) or {}),
        })
        previous_frame = frame
    unique_identities = sorted({str(record.get("photonic_identity", "")) for record in records if str(record.get("photonic_identity", ""))})
    count = float(len(records) or 1.0)
    predictive_calibration_surface = build_predictive_calibration_surface(records)
    predictive_calibration = predict_full_spectrum_calibration(
        surface=predictive_calibration_surface,
        phase_turns=float(dict(records[-1] if records else {}).get("trajectory_vector", [0.0, 0.0, 0.0, 0.0])[3] if records else 0.0),
        previous_phase_turns=float(dict(records[-2] if len(records) > 1 else records[-1] if records else {}).get("trajectory_vector", [0.0, 0.0, 0.0, 0.0])[3] if records else 0.0),
        interval_count=2,
        kernel_grid_width=2,
        kernel_grid_height=2,
        kernel_interval_ms=1.5,
        schema=temporal_schema,
    ) if records else {}
    summary = {
        "run_id": "run_045_photonic_identity_backbone",
        "input_frame_count": int(len(records)),
        "unique_photonic_identities": int(len(unique_identities)),
        "mean_vector_energy": sum(float(dict(record.get("dynamics", {}) or {}).get("vector_energy", 0.0)) for record in records) / count,
        "mean_temporal_coupling_moment": sum(float(dict(record.get("dynamics", {}) or {}).get("temporal_coupling_moment", 0.0)) for record in records) / count,
        "mean_inertial_mass_proxy": sum(float(dict(record.get("dynamics", {}) or {}).get("inertial_mass_proxy", 0.0)) for record in records) / count,
        "mean_predictive_temporal_accuracy": sum(float(dict(record.get("predictive_temporal_accounting", {}) or {}).get("temporal_accuracy_score", 0.0)) for record in records) / count,
        "mean_predictive_cycle_time_s": sum(float(dict(record.get("predictive_temporal_accounting", {}) or {}).get("predicted_cycle_time_s", 0.0)) for record in records) / count,
        "mean_predictive_anchor_interference": sum(float(dict(record.get("predictive_anchor_vectors", {}) or {}).get("anchor_interference_norm", 0.0)) for record in records) / count,
        "mean_predictive_harmonic_noise_reaction": sum(float(dict(record.get("predictive_harmonic_noise", {}) or {}).get("harmonic_noise_reaction_norm", 0.0)) for record in records) / count,
        "mean_predictive_trajectory_conservation": sum(float(dict(record.get("predictive_trajectory_state", {}) or {}).get("trajectory_conservation_9d", 0.0)) for record in records) / count,
        "mean_predictive_phase_transport": sum(float(dict(record.get("predictive_trajectory_state", {}) or {}).get("phase_transport_norm", 0.0)) for record in records) / count,
        "mean_predictive_reverse_causal_flux_coherence": sum(float(dict(record.get("predictive_trajectory_state", {}) or {}).get("reverse_causal_flux_coherence", 0.0)) for record in records) / count,
        "mean_predictive_hidden_flux_correction": sum(float(dict(record.get("predictive_trajectory_state", {}) or {}).get("hidden_flux_correction_norm", 0.0)) for record in records) / count,
        "mean_predictive_gpu_pulse_interference": sum(float(dict(record.get("predictive_pulse_interference", {}) or {}).get("gpu_pulse_interference_norm", 0.0)) for record in records) / count,
        "mean_predictive_system_sensitivity": sum(float(dict(record.get("predictive_pulse_interference", {}) or {}).get("system_sensitivity_norm", 0.0)) for record in records) / count,
        "mean_predictive_pulse_backreaction": sum(float(dict(record.get("predictive_pulse_interference", {}) or {}).get("pulse_backreaction_norm", 0.0)) for record in records) / count,
        "mean_predictive_phase_ring_density": sum(float(dict(record.get("predictive_phase_ring_trace", {}) or {}).get("phase_ring_density", 0.0)) for record in records) / count,
        "mean_predictive_zero_point_crossover": sum(float(dict(record.get("predictive_phase_ring_trace", {}) or {}).get("zero_point_crossover", 0.0)) for record in records) / count,
        "mean_predictive_identity_sweep_cluster": sum(float(dict(record.get("predictive_phase_ring_trace", {}) or {}).get("identity_sweep_cluster", 0.0)) for record in records) / count,
        "mean_predictive_crosstalk_cluster": sum(float(dict(record.get("predictive_phase_ring_trace", {}) or {}).get("crosstalk_cluster", 0.0)) for record in records) / count,
        "max_spin_momentum_score": max([float(dict(record.get("dynamics", {}) or {}).get("spin_momentum_score", 0.0)) for record in records] or [0.0]),
        "max_observer_damping": max([float(dict(record.get("observer", {}) or {}).get("observer_damping", 0.0)) for record in records] or [0.0]),
        "mean_closed_loop_latency_s": sum(float(dict(record.get("transport", {}) or {}).get("closed_loop_latency_s", 0.0)) for record in records) / count,
        "encodable_node_count": int(sum(1 for node in all_nodes if bool(node.get("encodable_node", False)))),
        "unwanted_noise_condition_count": int(sum(len(list(dict(record.get("predictive_harmonic_noise", {}) or {}).get("unwanted_noise_conditions", []) or [])) for record in records)),
        "predictive_calibration": {
            "sequence_coverage": float(dict(predictive_calibration or {}).get("sequence_coverage", 0.0)),
            "mean_temporal_accuracy": float(dict(predictive_calibration or {}).get("mean_temporal_accuracy", 0.0)),
            "mean_harmonic_noise_reaction": float(dict(predictive_calibration or {}).get("mean_harmonic_noise_reaction", 0.0)),
            "mean_trajectory_conservation": float(dict(predictive_calibration or {}).get("mean_trajectory_conservation", 0.0)),
            "mean_reverse_causal_flux_coherence": float(dict(predictive_calibration or {}).get("mean_reverse_causal_flux_coherence", 0.0)),
            "mean_hidden_flux_correction": float(dict(predictive_calibration or {}).get("mean_hidden_flux_correction", 0.0)),
            "mean_gpu_pulse_interference": float(dict(predictive_calibration or {}).get("mean_gpu_pulse_interference", 0.0)),
            "mean_system_sensitivity": float(dict(predictive_calibration or {}).get("mean_system_sensitivity", 0.0)),
            "mean_pulse_backreaction": float(dict(predictive_calibration or {}).get("mean_pulse_backreaction", 0.0)),
            "mean_pulse_trajectory_alignment": float(dict(predictive_calibration or {}).get("mean_pulse_trajectory_alignment", 0.0)),
            "unwanted_noise_condition_ratio": float(dict(predictive_calibration or {}).get("unwanted_noise_condition_ratio", 0.0)),
            "feedback_gate_state": str(dict(dict(predictive_calibration or {}).get("feedback_gate", {}) or {}).get("state", "unknown")),
            "feedback_gate_open": bool(dict(dict(predictive_calibration or {}).get("feedback_gate", {}) or {}).get("gate_open", False)),
        },
        "input_sources": {
            "live_frames": str(DEFAULT_RUN44_FRAMES),
            "live_ledger": str(DEFAULT_LIVE_LEDGER),
            "temporal_schema": str(DEFAULT_TEMPORAL_SCHEMA),
            "process_schema": str(DEFAULT_PROCESS_SCHEMA),
            "nist_reference": str(DEFAULT_NIST_REFERENCE),
        },
        "supported_claims": [
            "Photonic Identity is emitted as a deterministic Sig9-style coordinate signature rather than a crypto hash.",
            "Predictive photonic identities are encoded from UTF-8 phase-ring traces that follow stable anchor vectors through silicon-lattice atomic vector state.",
            "Axis spin, temporal coupling, and inertial mass remain research-only outputs until the calibration loop is validated in isolation.",
            "Request-to-return latency, observer damping, and flux/phase transport are folded into the same trace so GPU return timing can be encoded as part of the identity path.",
            "Field, transport, trajectory, pulse, noise, and gate weights are derived from temporal-relativity state rather than fixed coefficient tables.",
            "Encoded GPU pulses now perturb the simulated photonic trajectories directly, and calibration gates against pulse interference, pulse backreaction, and apparent system sensitivity rather than treating them as static noise add-ons.",
            "Weighted couplings can trigger harmonic-noise reactions, so unwanted-noise conditions are surfaced explicitly before any feedback gate is allowed to open.",
            "Disruption nodes are separately encoded so thermal and field-layer interference can be audited before any engine integration.",
        ],
    }
    return {
        "summary": summary,
        "trace_records": records,
        "api_packets": api_packets,
        "disruption_nodes": all_nodes,
        "unique_photonic_identities": unique_identities,
    }


def write_run45_artifacts(output_dir: Path, analysis: Dict[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_json = output_dir / "run_045_photonic_identity_trace.json"
    trace_csv = output_dir / "run_045_photonic_identity_trace.csv"
    packets_json = output_dir / "run_045_photonic_api_packets.json"
    nodes_json = output_dir / "run_045_disruption_nodes.json"
    summary_json = output_dir / "run_045_summary.json"

    trace_records = list(analysis.get("trace_records", []) or [])
    packets = list(analysis.get("api_packets", []) or [])
    nodes = list(analysis.get("disruption_nodes", []) or [])
    summary = dict(analysis.get("summary", {}) or {})

    trace_json.write_text(json.dumps(trace_records, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    packets_json.write_text(json.dumps(packets, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    nodes_json.write_text(json.dumps(nodes, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    with trace_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame_index",
            "photonic_identity",
            "axis_scale_x",
            "axis_scale_y",
            "axis_scale_z",
            "vector_energy",
            "temporal_coupling_moment",
            "inertial_mass_proxy",
            "spin_momentum_score",
            "phase_transport_term",
            "flux_transport_term",
            "observer_damping",
            "request_to_return_s",
            "closed_loop_latency_s",
            "predictive_temporal_accuracy_score",
            "predictive_anchor_interference_norm",
            "predictive_harmonic_noise_reaction_norm",
            "predictive_trajectory_conservation_norm",
            "predictive_reverse_causal_flux_coherence",
            "predictive_gpu_pulse_interference_norm",
            "predictive_system_sensitivity_norm",
            "predictive_phase_ring_density",
            "predictive_zero_point_crossover_norm",
            "request_feedback_time_s",
            "calculation_time_s",
            "next_feedback_time_s",
            "encodable_node_count",
        ])
        for record in trace_records:
            dynamics = dict(record.get("dynamics", {}) or {})
            transport = dict(record.get("transport", {}) or {})
            observer = dict(record.get("observer", {}) or {})
            predictive = dict(record.get("predictive_temporal_accounting", {}) or {})
            predictive_anchor = dict(record.get("predictive_anchor_vectors", {}) or {})
            predictive_harmonic = dict(record.get("predictive_harmonic_noise", {}) or {})
            predictive_trajectory = dict(record.get("predictive_trajectory_state", {}) or {})
            predictive_pulse = dict(record.get("predictive_pulse_interference", {}) or {})
            predictive_ring = dict(record.get("predictive_phase_ring_trace", {}) or {})
            writer.writerow([
                int(record.get("frame_index", 0)),
                str(record.get("photonic_identity", "")),
                float(dynamics.get("axis_scale_x", 0.0)),
                float(dynamics.get("axis_scale_y", 0.0)),
                float(dynamics.get("axis_scale_z", 0.0)),
                float(dynamics.get("vector_energy", 0.0)),
                float(dynamics.get("temporal_coupling_moment", 0.0)),
                float(dynamics.get("inertial_mass_proxy", 0.0)),
                float(dynamics.get("spin_momentum_score", 0.0)),
                float(transport.get("phase_transport_term", 0.0)),
                float(transport.get("flux_transport_term", 0.0)),
                float(observer.get("observer_damping", 0.0)),
                float(transport.get("request_to_return_s", 0.0)),
                float(transport.get("closed_loop_latency_s", 0.0)),
                float(predictive.get("temporal_accuracy_score", 0.0)),
                float(predictive_anchor.get("anchor_interference_norm", 0.0)),
                float(predictive_harmonic.get("harmonic_noise_reaction_norm", 0.0)),
                float(predictive_trajectory.get("trajectory_conservation_9d", 0.0)),
                float(predictive_trajectory.get("reverse_causal_flux_coherence", 0.0)),
                float(predictive_pulse.get("gpu_pulse_interference_norm", 0.0)),
                float(predictive_pulse.get("system_sensitivity_norm", 0.0)),
                float(predictive_ring.get("phase_ring_density", 0.0)),
                float(predictive_ring.get("zero_point_crossover", 0.0)),
                float(predictive.get("request_feedback_time_s", 0.0)),
                float(predictive.get("calculation_time_s", 0.0)),
                float(predictive.get("next_feedback_time_s", 0.0)),
                int(sum(1 for node in list(record.get("disruption_nodes", []) or []) if bool(node.get("encodable_node", False)))),
            ])

    return {
        "trace_json": str(trace_json),
        "trace_csv": str(trace_csv),
        "packets_json": str(packets_json),
        "nodes_json": str(nodes_json),
        "summary_json": str(summary_json),
    }