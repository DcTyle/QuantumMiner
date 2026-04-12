from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Dict, List
import csv
import json


ROOT = Path(__file__).resolve().parents[2]
RUN45_TRACE = ROOT / "ResearchConfinement" / "Run45" / "run_045_photonic_identity_trace.csv"
RUN45_PACKETS = ROOT / "ResearchConfinement" / "Run45" / "run_045_photonic_api_packets.json"
OUTPUT_SUMMARY = Path(__file__).resolve().parent / "run45_qbit_validation_summary.json"


def _read_trace_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "frame_index": int(row.get("frame_index", 0) or 0),
                "photonic_identity": str(row.get("photonic_identity", "") or ""),
                "axis_scale_x": float(row.get("axis_scale_x", 0.0) or 0.0),
                "axis_scale_y": float(row.get("axis_scale_y", 0.0) or 0.0),
                "axis_scale_z": float(row.get("axis_scale_z", 0.0) or 0.0),
                "vector_energy": float(row.get("vector_energy", 0.0) or 0.0),
                "temporal_coupling_moment": float(row.get("temporal_coupling_moment", 0.0) or 0.0),
                "inertial_mass_proxy": float(row.get("inertial_mass_proxy", 0.0) or 0.0),
                "spin_momentum_score": float(row.get("spin_momentum_score", 0.0) or 0.0),
                "phase_transport_term": float(row.get("phase_transport_term", 0.0) or 0.0),
                "flux_transport_term": float(row.get("flux_transport_term", 0.0) or 0.0),
                "observer_damping": float(row.get("observer_damping", 0.0) or 0.0),
                "request_to_return_s": float(row.get("request_to_return_s", 0.0) or 0.0),
                "closed_loop_latency_s": float(row.get("closed_loop_latency_s", 0.0) or 0.0),
                "predictive_temporal_accuracy_score": float(row.get("predictive_temporal_accuracy_score", 0.0) or 0.0),
                "predictive_anchor_interference_norm": float(row.get("predictive_anchor_interference_norm", 0.0) or 0.0),
                "predictive_harmonic_noise_reaction_norm": float(row.get("predictive_harmonic_noise_reaction_norm", 0.0) or 0.0),
                "request_feedback_time_s": float(row.get("request_feedback_time_s", 0.0) or 0.0),
                "calculation_time_s": float(row.get("calculation_time_s", 0.0) or 0.0),
                "next_feedback_time_s": float(row.get("next_feedback_time_s", 0.0) or 0.0),
                "encodable_node_count": int(row.get("encodable_node_count", 0) or 0),
            })
    return rows


def _read_packets(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return []


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return float(mean(float(row.get(key, 0.0) or 0.0) for row in rows)) if rows else 0.0


def _all_between_zero_and_one(rows: List[Dict[str, Any]], keys: List[str]) -> bool:
    for row in rows:
        for key in keys:
            value = float(row.get(key, 0.0) or 0.0)
            if value < 0.0 or value > 1.0:
                return False
    return True


def validate_run45() -> Dict[str, Any]:
    trace_rows = _read_trace_rows(RUN45_TRACE)
    packets = _read_packets(RUN45_PACKETS)

    if not trace_rows:
        raise AssertionError("Run45 trace is empty")
    if not packets:
        raise AssertionError("Run45 packets are empty")

    request_packets = [packet for packet in packets if dict(packet.get("request", {}) or {})]
    feedback_packets = [packet for packet in packets if dict(packet.get("feedback", {}) or {})]
    valid_identities = [row for row in trace_rows if str(row.get("photonic_identity", "")).startswith("PID9-")]
    valid_nodes = [row for row in trace_rows if int(row.get("encodable_node_count", 0)) > 0]

    summary = {
        "frame_count": len(trace_rows),
        "packet_count": len(packets),
        "request_packet_count": len(request_packets),
        "feedback_packet_count": len(feedback_packets),
        "valid_identity_count": len(valid_identities),
        "valid_node_count": len(valid_nodes),
        "mean_axis_scale_x": _mean(trace_rows, "axis_scale_x"),
        "mean_axis_scale_y": _mean(trace_rows, "axis_scale_y"),
        "mean_axis_scale_z": _mean(trace_rows, "axis_scale_z"),
        "mean_vector_energy": _mean(trace_rows, "vector_energy"),
        "mean_temporal_coupling_moment": _mean(trace_rows, "temporal_coupling_moment"),
        "mean_inertial_mass_proxy": _mean(trace_rows, "inertial_mass_proxy"),
        "mean_spin_momentum_score": _mean(trace_rows, "spin_momentum_score"),
        "mean_phase_transport_term": _mean(trace_rows, "phase_transport_term"),
        "mean_flux_transport_term": _mean(trace_rows, "flux_transport_term"),
        "mean_observer_damping": _mean(trace_rows, "observer_damping"),
        "mean_predictive_temporal_accuracy_score": _mean(trace_rows, "predictive_temporal_accuracy_score"),
        "mean_predictive_anchor_interference_norm": _mean(trace_rows, "predictive_anchor_interference_norm"),
        "mean_predictive_harmonic_noise_reaction_norm": _mean(trace_rows, "predictive_harmonic_noise_reaction_norm"),
        "max_closed_loop_latency_s": max(float(row.get("closed_loop_latency_s", 0.0) or 0.0) for row in trace_rows),
        "max_next_feedback_time_s": max(float(row.get("next_feedback_time_s", 0.0) or 0.0) for row in trace_rows),
    }

    assert summary["frame_count"] >= 8
    assert summary["request_packet_count"] == summary["frame_count"]
    assert summary["feedback_packet_count"] == summary["frame_count"]
    assert summary["valid_identity_count"] == summary["frame_count"]
    assert summary["valid_node_count"] == summary["frame_count"]
    assert _all_between_zero_and_one(trace_rows, [
        "axis_scale_x",
        "axis_scale_y",
        "axis_scale_z",
        "vector_energy",
        "temporal_coupling_moment",
        "inertial_mass_proxy",
        "spin_momentum_score",
        "flux_transport_term",
        "observer_damping",
        "predictive_temporal_accuracy_score",
        "predictive_anchor_interference_norm",
        "predictive_harmonic_noise_reaction_norm",
    ])
    assert all(float(row.get("closed_loop_latency_s", 0.0) or 0.0) >= float(row.get("request_to_return_s", 0.0) or 0.0) for row in trace_rows)
    assert all(float(row.get("next_feedback_time_s", 0.0) or 0.0) >= float(row.get("calculation_time_s", 0.0) or 0.0) for row in trace_rows)
    assert summary["mean_inertial_mass_proxy"] >= summary["mean_spin_momentum_score"]
    assert summary["mean_temporal_coupling_moment"] >= summary["mean_phase_transport_term"]
    assert summary["mean_flux_transport_term"] >= summary["mean_phase_transport_term"]
    assert summary["mean_predictive_temporal_accuracy_score"] >= summary["mean_predictive_anchor_interference_norm"]
    assert summary["max_closed_loop_latency_s"] >= summary["max_next_feedback_time_s"]

    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="ascii")
    return summary


if __name__ == "__main__":
    print(json.dumps(validate_run45(), indent=2, sort_keys=True))