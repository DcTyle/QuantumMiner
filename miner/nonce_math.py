from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import os
import random
import time

from neural_object import (
    neural_objectPacket,
    neural_objectSchema,
)


class NonceMath:
    # Developer-only debug flag (env). When disabled, zero overhead paths.
    _DEBUG = os.getenv("NONCEMATH_DEBUG", "0") == "1"

    # Require VSD appender; no fallbacks allowed
    from VHW.system_utils import vsd_append_bounded as _vsd_append

    @staticmethod
    def snapshot_lane_state(lane_state: Optional[Any]) -> Dict[str, Any]:
        """Best-effort read-only snapshot of lane NonceMath state."""
        try:
            state = dict(getattr(lane_state, "_noncemath_state", {}))
        except Exception:
            state = {}
        try:
            last_nonce = int(state.get("last_nonce", 0))
        except Exception:
            last_nonce = 0
        return {
            "entropy_score": float(state.get("entropy_score", 0.0)),
            "last_nonce": last_nonce,
            "psi": float(state.get("psi", 0.0)),
            "flux": float(state.get("flux", 0.0)),
            "harmonic": float(state.get("harmonic", 0.0)),
            "phase": float(state.get("phase", 0.0)),
            "amplitude_cap": float(state.get("amplitude_cap", 0.0)),
            "coherence_peak": float(state.get("coherence_peak", 0.0)),
            "target_interval": int(state.get("target_interval", 0) or 0),
            "candidate_count": int(state.get("candidate_count", 0) or 0),
            "validated_candidate_count": int(state.get("validated_candidate_count", 0) or 0),
            "valid_share_count": int(state.get("valid_share_count", 0) or 0),
            "valid_ratio": float(state.get("valid_ratio", 0.0)),
            "atomic_vector_x": float(state.get("atomic_vector_x", 0.0)),
            "atomic_vector_y": float(state.get("atomic_vector_y", 0.0)),
            "atomic_vector_z": float(state.get("atomic_vector_z", 0.0)),
            "axis_scale_x": float(state.get("axis_scale_x", 0.0)),
            "axis_scale_y": float(state.get("axis_scale_y", 0.0)),
            "axis_scale_z": float(state.get("axis_scale_z", 0.0)),
            "vector_energy": float(state.get("vector_energy", 0.0)),
            "temporal_coupling_moment": float(state.get("temporal_coupling_moment", 0.0)),
            "inertial_mass_proxy": float(state.get("inertial_mass_proxy", 0.0)),
            "relativistic_correlation": float(state.get("relativistic_correlation", 0.0)),
            "spin_axis_x": float(state.get("spin_axis_x", 0.0)),
            "spin_axis_y": float(state.get("spin_axis_y", 0.0)),
            "spin_axis_z": float(state.get("spin_axis_z", 0.0)),
            "spin_momentum_score": float(state.get("spin_momentum_score", 0.0)),
            "phase_ring_closure": float(state.get("phase_ring_closure", 0.0)),
            "phase_ring_density": float(state.get("phase_ring_density", 0.0)),
            "phase_ring_strength": float(state.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": float(state.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": float(state.get("shared_vector_collapse_gate", 0.0)),
            "shared_vector_phase_lock": float(state.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": float(state.get("inertial_basin_strength", 0.0)),
            "temporal_relativity_norm": float(state.get("temporal_relativity_norm", 0.0)),
            "zero_point_line_distance": float(state.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": float(state.get("field_interference_norm", 0.0)),
            "resonant_interception_inertia": float(state.get("resonant_interception_inertia", 0.0)),
            "process_resonance": float(state.get("process_resonance", 0.0)),
            "mining_resonance_gate": float(state.get("mining_resonance_gate", 0.0)),
            "collapse_readiness": float(state.get("collapse_readiness", 0.0)),
            "mining_resonance_score": float(state.get("mining_resonance_score", 0.0)),
            "search_backend": str(state.get("search_backend", "")),
            "process_mode": str(state.get("process_mode", "")),
            "scheduler_mode": str(state.get("scheduler_mode", "")),
            "active_zone_name": str(state.get("active_zone_name", "")),
            "active_basin_name": str(state.get("active_basin_name", "")),
        }

    @staticmethod
    def _lane_id_str(lane_state: Optional[Any]) -> str:
        try:
            return str(getattr(lane_state, "lane_id", "lane"))
        except Exception:
            return "lane"

    @staticmethod
    def _debug_dump(enabled: bool, lane_id: str, nonce: int, state: Dict[str, Any], tag: str) -> None:
        if not enabled:
            return
        try:
            rec = {
                "ts": time.time(),
                "lane": lane_id,
                "tag": str(tag),
                "nonce": int(nonce) & 0xFFFFFFFF,
                "psi": float(state.get("psi", 0.0)),
                "flux": float(state.get("flux", 0.0)),
                "harmonic": float(state.get("harmonic", 0.0)),
                "phase": float(state.get("phase", 0.0)),
            }
            NonceMath._vsd_append("debug/nonce_evolution", rec, 1000)
        except Exception:
            return

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clamp_signed(value: float, limit: float = 1.0) -> float:
        bound = abs(float(limit))
        return max(-bound, min(bound, float(value)))

    @staticmethod
    def _flag_enabled(value: Any) -> bool:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(int(value))
        try:
            text = str(value or "").strip().lower()
        except Exception:
            text = ""
        if text in ("", "0", "false", "off", "no", "none", "disable", "disabled"):
            return False
        if text in ("1", "true", "on", "yes", "enable", "enabled"):
            return True
        return bool(text)

    @staticmethod
    def _safe_hex(text: Any) -> str:
        try:
            out = str(text or "").strip().lower()
        except Exception:
            out = ""
        if out.startswith("0x"):
            out = out[2:]
        return "".join(ch for ch in out if ch in "0123456789abcdef")

    @staticmethod
    def _stable_fraction(*parts: Any) -> float:
        seed = 2166136261
        for part in parts:
            for ch in str(part or "").encode("ascii", errors="ignore"):
                seed ^= ch
                seed = (seed * 16777619) & 0xFFFFFFFF
        return float(seed & 0xFFFF) / 65535.0

    @staticmethod
    def _init_lane_state(lane_state: Optional[Any]) -> Dict[str, Any]:
        try:
            state = dict(getattr(lane_state, "_noncemath_state", {}))
        except Exception:
            state = {}
        if "base_nonce" not in state:
            state["base_nonce"] = random.getrandbits(32)
        if "psi" not in state:
            state["psi"] = 0.0
        if "flux" not in state:
            state["flux"] = 0.0
        if "harmonic" not in state:
            state["harmonic"] = 0.0
        if "phase" not in state:
            state["phase"] = random.random()
        if "d1" not in state:
            state["d1"] = 0
        if "entropy_score" not in state:
            state["entropy_score"] = 0.0
        if "coherence_peak" not in state:
            state["coherence_peak"] = 0.0
        if "amplitude_cap" not in state:
            state["amplitude_cap"] = 0.0
        if "target_interval" not in state:
            state["target_interval"] = 0
        if "candidate_count" not in state:
            state["candidate_count"] = 0
        if "validated_candidate_count" not in state:
            state["validated_candidate_count"] = 0
        if "valid_share_count" not in state:
            state["valid_share_count"] = 0
        if "valid_ratio" not in state:
            state["valid_ratio"] = 0.0
        if "atomic_vector_x" not in state:
            state["atomic_vector_x"] = 0.0
        if "atomic_vector_y" not in state:
            state["atomic_vector_y"] = 0.0
        if "atomic_vector_z" not in state:
            state["atomic_vector_z"] = 0.0
        if "axis_scale_x" not in state:
            state["axis_scale_x"] = 0.0
        if "axis_scale_y" not in state:
            state["axis_scale_y"] = 0.0
        if "axis_scale_z" not in state:
            state["axis_scale_z"] = 0.0
        if "vector_energy" not in state:
            state["vector_energy"] = 0.0
        if "temporal_coupling_moment" not in state:
            state["temporal_coupling_moment"] = 0.0
        if "inertial_mass_proxy" not in state:
            state["inertial_mass_proxy"] = 0.0
        if "relativistic_correlation" not in state:
            state["relativistic_correlation"] = 0.0
        if "spin_axis_x" not in state:
            state["spin_axis_x"] = 0.0
        if "spin_axis_y" not in state:
            state["spin_axis_y"] = 0.0
        if "spin_axis_z" not in state:
            state["spin_axis_z"] = 0.0
        if "spin_momentum_score" not in state:
            state["spin_momentum_score"] = 0.0
        if "phase_ring_closure" not in state:
            state["phase_ring_closure"] = 0.0
        if "phase_ring_density" not in state:
            state["phase_ring_density"] = 0.0
        if "phase_ring_strength" not in state:
            state["phase_ring_strength"] = 0.0
        if "zero_point_crossover_gate" not in state:
            state["zero_point_crossover_gate"] = 0.0
        if "shared_vector_collapse_gate" not in state:
            state["shared_vector_collapse_gate"] = 0.0
        if "shared_vector_phase_lock" not in state:
            state["shared_vector_phase_lock"] = 0.0
        if "inertial_basin_strength" not in state:
            state["inertial_basin_strength"] = 0.0
        if "temporal_relativity_norm" not in state:
            state["temporal_relativity_norm"] = 0.0
        if "zero_point_line_distance" not in state:
            state["zero_point_line_distance"] = 0.0
        if "field_interference_norm" not in state:
            state["field_interference_norm"] = 0.0
        if "resonant_interception_inertia" not in state:
            state["resonant_interception_inertia"] = 0.0
        if "process_resonance" not in state:
            state["process_resonance"] = 0.0
        if "mining_resonance_gate" not in state:
            state["mining_resonance_gate"] = 0.0
        if "collapse_readiness" not in state:
            state["collapse_readiness"] = 0.0
        if "mining_resonance_score" not in state:
            state["mining_resonance_score"] = 0.0
        if "search_backend" not in state:
            state["search_backend"] = ""
        if "process_mode" not in state:
            state["process_mode"] = ""
        if "scheduler_mode" not in state:
            state["scheduler_mode"] = ""
        if "active_zone_name" not in state:
            state["active_zone_name"] = ""
        if "active_basin_name" not in state:
            state["active_basin_name"] = ""
        return state

    @staticmethod
    def _persist_lane_state(lane_state: Optional[Any], state: Dict[str, Any]) -> None:
        try:
            setattr(lane_state, "_noncemath_state", dict(state))
        except Exception:
            return

    @staticmethod
    def _dmt_update(state: Dict[str, Any], env: Dict[str, float]) -> None:
        # Extract environment signals (bounded [0,1])
        global_util = float(env.get("global_util", 0.5))
        gpu_util = float(env.get("gpu_util", 0.5))
        mem_bw_util = float(env.get("mem_bw_util", 0.5))
        cpu_util = float(env.get("cpu_util", 0.5))

        # Tunable coefficients
        alpha_flux = float(env.get("alpha_flux", 0.15))
        flux_coeff = float(env.get("flux_coeff", 0.20))
        drift_coeff = float(env.get("drift_coeff", 0.35))
        phase_step = float(env.get("phase_step", 0.07))

        # State pull
        psi = float(state.get("psi", 0.0))
        phase = float(state.get("phase", 0.0))

        # Model equations
        psi_next = psi + alpha_flux * (global_util - gpu_util)
        flux = flux_coeff * (mem_bw_util - cpu_util)
        harmonic = drift_coeff * math.sin(2.0 * math.pi * phase)
        phase_next = (phase + phase_step) % 1.0

        state["psi"] = psi_next
        state["flux"] = flux
        state["harmonic"] = harmonic
        state["phase"] = phase_next

    @staticmethod
    def _derive_nonce(state: Dict[str, Any], lane_id_seed: int) -> int:
        base_nonce = int(state.get("base_nonce", 0)) & 0xFFFFFFFF
        d1 = int(state.get("d1", 0)) & 0xFFFFFFFF
        psi = float(state.get("psi", 0.0))
        flux = float(state.get("flux", 0.0))
        harmonic = float(state.get("harmonic", 0.0))
        phase = float(state.get("phase", 0.0))

        psi_flux_term = int((psi + flux) * (1 << 16)) & 0xFFFFFFFF
        lane_phase_shift = int((phase * 977) + (lane_id_seed % 4093)) & 0xFFFFFFFF
        harmonic_drift = int(harmonic * (1 << 18)) & 0xFFFFFFFF
        nonce_next = (base_nonce + d1 + psi_flux_term + harmonic_drift + lane_phase_shift) & 0xFFFFFFFF

        state["base_nonce"] = (base_nonce * 1664525 + 1013904223) & 0xFFFFFFFF
        state["d1"] = (d1 + 1) & 0xFFFFFFFF
        state["last_nonce"] = int(nonce_next)
        return nonce_next

    @staticmethod
    def _lane_seed(lane_state: Optional[Any]) -> int:
        try:
            lid = getattr(lane_state, "lane_id", "lane")
        except Exception:
            lid = "lane"
        h = 2166136261
        for ch in str(lid).encode("ascii", errors="ignore"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    @staticmethod
    def _env_from_norm(norm: Dict[str, Any]) -> Dict[str, float]:
        sp = norm.get("system_payload", {}) if isinstance(norm, dict) else {}
        if not isinstance(sp, dict):
            sp = {}
        return {
            "global_util": float(sp.get("global_util", 0.5)),
            "gpu_util": float(sp.get("gpu_util", 0.5)),
            "mem_bw_util": float(sp.get("mem_bw_util", 0.5)),
            "cpu_util": float(sp.get("cpu_util", 0.5)),
            "alpha_flux": float(sp.get("alpha_flux", 0.15)),
            "flux_coeff": float(sp.get("flux_coeff", 0.20)),
            "drift_coeff": float(sp.get("drift_coeff", 0.35)),
            "phase_step": float(sp.get("phase_step", 0.07)),
            "trace_support": float(sp.get("trace_support", 0.0)),
            "trace_resonance": float(sp.get("trace_resonance", 0.0)),
            "trace_alignment": float(sp.get("trace_alignment", 0.0)),
            "trace_memory": float(sp.get("trace_memory", 0.0)),
            "trace_flux": float(sp.get("trace_flux", 0.0)),
            "trace_temporal_persistence": float(sp.get("trace_temporal_persistence", 0.0)),
            "trace_temporal_overlap": float(sp.get("trace_temporal_overlap", 0.0)),
            "trace_voltage_frequency_flux": float(sp.get("trace_voltage_frequency_flux", 0.0)),
            "trace_frequency_voltage_flux": float(sp.get("trace_frequency_voltage_flux", 0.0)),
            "trace_phase_anchor_turns": float(sp.get("trace_phase_anchor_turns", 0.0)),
        }

    @staticmethod
    def _vector_list(value: Any, size: int) -> List[float]:
        if isinstance(value, (list, tuple)):
            out = [0.0] * size
            for idx in range(min(size, len(value))):
                try:
                    out[idx] = float(value[idx])
                except Exception:
                    out[idx] = 0.0
            return out
        return [0.0] * size

    @staticmethod
    def _substrate_trace_feedback(packet_norm: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        sp = packet_norm.get("system_payload", {}) if isinstance(packet_norm, dict) else {}
        if not isinstance(sp, dict):
            sp = {}
        raw = {}
        try:
            raw = dict(params.get("substrate_trace_state", {}) or {}) if isinstance(params, dict) else {}
        except Exception:
            raw = {}
        for key in (
            "trace_support",
            "trace_resonance",
            "trace_alignment",
            "trace_memory",
            "trace_flux",
            "trace_temporal_persistence",
            "trace_temporal_overlap",
            "trace_voltage_frequency_flux",
            "trace_frequency_voltage_flux",
            "trace_phase_anchor_turns",
            "trace_phase_ring_closure",
            "trace_phase_ring_density",
            "trace_phase_ring_strength",
            "trace_zero_point_crossover",
            "trace_shared_vector_collapse",
            "trace_shared_vector_phase_lock",
            "trace_inertial_basin_strength",
            "trace_temporal_relativity_norm",
            "trace_zero_point_line_distance",
            "trace_field_interference_norm",
            "trace_resonant_interception_inertia",
            "process_resonance",
            "mining_resonance_gate",
            "collapse_readiness",
            "process_mode",
            "scheduler_mode",
            "active_zone_name",
            "active_basin_name",
        ):
            if key not in raw and key in sp:
                raw[key] = sp.get(key)
        for key in ("trace_vector", "trace_axis_vector", "trace_dof_vector", "trace_frequency_gradient_9d"):
            if key not in raw and key in sp:
                raw[key] = sp.get(key)

        trace = {
            "trace_support": NonceMath._clamp01(raw.get("trace_support", 0.0)),
            "trace_resonance": NonceMath._clamp01(raw.get("trace_resonance", 0.0)),
            "trace_alignment": NonceMath._clamp01(raw.get("trace_alignment", 0.0)),
            "trace_memory": NonceMath._clamp01(raw.get("trace_memory", 0.0)),
            "trace_flux": NonceMath._clamp01(raw.get("trace_flux", 0.0)),
            "trace_temporal_persistence": NonceMath._clamp01(raw.get("trace_temporal_persistence", 0.0)),
            "trace_temporal_overlap": NonceMath._clamp01(raw.get("trace_temporal_overlap", 0.0)),
            "trace_voltage_frequency_flux": NonceMath._clamp01(raw.get("trace_voltage_frequency_flux", 0.0)),
            "trace_frequency_voltage_flux": NonceMath._clamp01(raw.get("trace_frequency_voltage_flux", 0.0)),
            "trace_phase_anchor_turns": float(raw.get("trace_phase_anchor_turns", 0.0)) % 1.0,
            "trace_phase_ring_closure": NonceMath._clamp01(raw.get("trace_phase_ring_closure", 0.0)),
            "trace_phase_ring_density": NonceMath._clamp01(raw.get("trace_phase_ring_density", 0.0)),
            "trace_phase_ring_strength": NonceMath._clamp01(raw.get("trace_phase_ring_strength", 0.0)),
            "trace_zero_point_crossover": NonceMath._clamp01(raw.get("trace_zero_point_crossover", 0.0)),
            "trace_shared_vector_collapse": NonceMath._clamp01(raw.get("trace_shared_vector_collapse", 0.0)),
            "trace_shared_vector_phase_lock": NonceMath._clamp01(raw.get("trace_shared_vector_phase_lock", 0.0)),
            "trace_inertial_basin_strength": NonceMath._clamp01(raw.get("trace_inertial_basin_strength", 0.0)),
            "trace_temporal_relativity_norm": NonceMath._clamp01(raw.get("trace_temporal_relativity_norm", 0.0)),
            "trace_zero_point_line_distance": NonceMath._clamp01(raw.get("trace_zero_point_line_distance", 0.0)),
            "trace_field_interference_norm": NonceMath._clamp01(raw.get("trace_field_interference_norm", 0.0)),
            "trace_resonant_interception_inertia": NonceMath._clamp01(raw.get("trace_resonant_interception_inertia", 0.0)),
            "trace_vector": NonceMath._vector_list(raw.get("trace_vector", []), 4),
            "trace_axis_vector": [NonceMath._clamp01(v) for v in NonceMath._vector_list(raw.get("trace_axis_vector", []), 4)],
            "trace_dof_vector": [NonceMath._clamp01(v) for v in NonceMath._vector_list(raw.get("trace_dof_vector", []), 10)],
            "trace_frequency_gradient_9d": [NonceMath._clamp01(v) for v in NonceMath._vector_list(raw.get("trace_frequency_gradient_9d", []), 9)],
            "process_resonance": NonceMath._clamp01(raw.get("process_resonance", 0.0)),
            "mining_resonance_gate": NonceMath._clamp01(raw.get("mining_resonance_gate", 0.0)),
            "collapse_readiness": NonceMath._clamp01(raw.get("collapse_readiness", 0.0)),
            "process_mode": str(raw.get("process_mode", "")),
            "scheduler_mode": str(raw.get("scheduler_mode", "")),
            "active_zone_name": str(raw.get("active_zone_name", "")),
            "active_basin_name": str(raw.get("active_basin_name", "")),
            "feedback_weight": NonceMath._clamp01((params or {}).get("substrate_feedback_weight", 0.35)) if isinstance(params, dict) else 0.35,
            "scan_boost": int((params or {}).get("substrate_scan_boost", 0) or 0) if isinstance(params, dict) else 0,
        }
        trace["trace_gate"] = max(
            trace["trace_support"],
            trace["trace_resonance"],
            trace["trace_alignment"],
            trace["trace_memory"],
            trace["trace_flux"],
            trace["trace_temporal_persistence"],
            trace["trace_temporal_overlap"],
            trace["trace_voltage_frequency_flux"],
            trace["trace_frequency_voltage_flux"],
            trace["trace_phase_ring_strength"],
            trace["trace_shared_vector_phase_lock"],
            trace["trace_temporal_relativity_norm"],
            max([abs(v) for v in trace["trace_axis_vector"]] or [0.0]),
        )
        if trace["trace_gate"] <= 0.0:
            trace["feedback_weight"] = 0.0
        return trace

    @staticmethod
    def _atomic_vector(env: Dict[str, float], state: Dict[str, Any], trace: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        global_util = NonceMath._clamp01(env.get("global_util", 0.5))
        gpu_util = NonceMath._clamp01(env.get("gpu_util", global_util))
        mem_bw_util = NonceMath._clamp01(env.get("mem_bw_util", global_util))
        cpu_util = NonceMath._clamp01(env.get("cpu_util", global_util))
        phase = float(state.get("phase", 0.0))
        vec_x = gpu_util - global_util
        vec_y = mem_bw_util - cpu_util
        vec_z = math.sin(2.0 * math.pi * phase) * (0.5 + 0.5 * gpu_util)
        trace_state = dict(trace or {})
        trace_gate = NonceMath._clamp01(trace_state.get("trace_gate", 0.0))
        feedback_weight = NonceMath._clamp01(trace_state.get("feedback_weight", 0.0)) * trace_gate
        if feedback_weight > 0.0:
            axis = trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0])
            phase_anchor = float(trace_state.get("trace_phase_anchor_turns", phase)) % 1.0
            support = NonceMath._clamp01(trace_state.get("trace_support", 0.0))
            resonance = NonceMath._clamp01(trace_state.get("trace_resonance", 0.0))
            alignment = NonceMath._clamp01(trace_state.get("trace_alignment", 0.0))
            vf_flux = NonceMath._clamp01(trace_state.get("trace_voltage_frequency_flux", 0.0))
            fv_flux = NonceMath._clamp01(trace_state.get("trace_frequency_voltage_flux", 0.0))
            vec_x += ((axis[0] - 0.5) * 0.45 + (fv_flux - 0.5) * 0.20) * feedback_weight
            vec_y += ((axis[1] - 0.5) * 0.45 + (support - 0.5) * 0.20) * feedback_weight
            vec_z = (
                vec_z * (1.0 - 0.30 * feedback_weight)
                + (math.sin(2.0 * math.pi * phase_anchor) * (0.30 + 0.30 * resonance) * feedback_weight)
                + ((axis[3] - 0.5) * 0.35 * feedback_weight)
                + ((vf_flux - 0.5) * 0.25 * feedback_weight)
                + ((alignment - 0.5) * 0.20 * feedback_weight)
            )
        magnitude = min(1.0, math.sqrt(vec_x * vec_x + vec_y * vec_y + vec_z * vec_z))
        axis_scale_x = NonceMath._clamp01(0.52 * gpu_util + 0.28 * NonceMath._clamp01(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0])[0]) + 0.20 * trace_gate)
        axis_scale_y = NonceMath._clamp01(0.52 * mem_bw_util + 0.28 * NonceMath._clamp01(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0])[1]) + 0.20 * trace_gate)
        axis_scale_z = NonceMath._clamp01(0.52 * cpu_util + 0.18 * abs(vec_z) + 0.15 * NonceMath._clamp01(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0])[2]) + 0.15 * trace_gate)
        vector_energy = NonceMath._clamp01((magnitude + axis_scale_x + axis_scale_y + axis_scale_z) / 4.0)
        temporal_coupling = NonceMath._clamp01(
            0.28 * NonceMath._clamp01(trace_state.get("trace_temporal_overlap", 0.0))
            + 0.24 * NonceMath._clamp01(trace_state.get("trace_temporal_persistence", 0.0))
            + 0.18 * magnitude
            + 0.16 * NonceMath._clamp01(trace_state.get("trace_alignment", 0.0))
            + 0.14 * NonceMath._clamp01(trace_state.get("trace_resonance", 0.0))
        )
        spin_axis_x = NonceMath._clamp_signed((vec_y * axis_scale_z) - (vec_z * axis_scale_y))
        spin_axis_y = NonceMath._clamp_signed((vec_z * axis_scale_x) - (vec_x * axis_scale_z))
        spin_axis_z = NonceMath._clamp_signed((vec_x * axis_scale_y) - (vec_y * axis_scale_x))
        spin_score = NonceMath._clamp01((abs(spin_axis_x) + abs(spin_axis_y) + abs(spin_axis_z)) / 3.0)
        relativistic_correlation = NonceMath._clamp01(0.44 * vector_energy + 0.22 * temporal_coupling + 0.18 * spin_score)
        inertial_mass = NonceMath._clamp01(0.38 * vector_energy + 0.22 * temporal_coupling + 0.20 * relativistic_correlation + 0.20 * spin_score)
        phase_ring_closure = NonceMath._clamp01(max(trace_state.get("trace_phase_ring_closure", 0.0), 0.26 * temporal_coupling + 0.24 * axis_scale_x + 0.18 * axis_scale_y + 0.16 * vector_energy + 0.16 * trace_gate))
        phase_ring_density = NonceMath._clamp01(max(trace_state.get("trace_phase_ring_density", 0.0), 0.30 * magnitude + 0.22 * axis_scale_z + 0.18 * inertial_mass + 0.16 * temporal_coupling + 0.14 * trace_gate))
        phase_ring_strength = NonceMath._clamp01(max(trace_state.get("trace_phase_ring_strength", 0.0), 0.34 * phase_ring_closure + 0.28 * phase_ring_density + 0.18 * temporal_coupling + 0.20 * trace_gate))
        zero_point_crossover = NonceMath._clamp01(max(trace_state.get("trace_zero_point_crossover", 0.0), 0.34 * phase_ring_strength + 0.22 * axis_scale_z + 0.22 * (1.0 - min(1.0, abs(vec_x + vec_y + vec_z) / 3.0)) + 0.22 * trace_gate))
        shared_vector_collapse = NonceMath._clamp01(max(trace_state.get("trace_shared_vector_collapse", 0.0), 0.30 * zero_point_crossover + 0.24 * vector_energy + 0.22 * NonceMath._clamp01(trace_state.get("trace_memory", 0.0)) + 0.24 * trace_gate))
        shared_vector_phase_lock = NonceMath._clamp01(max(trace_state.get("trace_shared_vector_phase_lock", 0.0), 0.34 * phase_ring_closure + 0.24 * NonceMath._clamp01(trace_state.get("trace_alignment", 0.0)) + 0.22 * temporal_coupling + 0.20 * trace_gate))
        inertial_basin_strength = NonceMath._clamp01(max(trace_state.get("trace_inertial_basin_strength", 0.0), 0.36 * inertial_mass + 0.28 * phase_ring_strength + 0.18 * zero_point_crossover + 0.18 * shared_vector_collapse))
        temporal_relativity_norm = NonceMath._clamp01(max(trace_state.get("trace_temporal_relativity_norm", 0.0), 0.26 * temporal_coupling + 0.22 * phase_ring_strength + 0.18 * NonceMath._clamp01(trace_state.get("trace_temporal_overlap", 0.0)) + 0.18 * NonceMath._clamp01(trace_state.get("trace_alignment", 0.0)) + 0.16 * (1.0 - min(1.0, abs(phase - (trace_state.get("trace_phase_anchor_turns", phase) % 1.0)) * 2.0))))
        zero_point_line_distance = NonceMath._clamp01(max(trace_state.get("trace_zero_point_line_distance", 0.0), 0.34 * (1.0 - phase_ring_strength) + 0.22 * abs(vec_z) + 0.22 * inertial_mass + 0.22 * (1.0 - NonceMath._clamp01(trace_state.get("trace_alignment", 0.0)))))
        field_interference_norm = NonceMath._clamp01(max(trace_state.get("trace_field_interference_norm", 0.0), 0.34 * abs(vec_x - vec_y) + 0.22 * abs(vec_z) + 0.22 * spin_score + 0.22 * (1.0 - NonceMath._clamp01(trace_state.get("trace_support", 0.0)))))
        resonant_interception_inertia = NonceMath._clamp01(max(trace_state.get("trace_resonant_interception_inertia", 0.0), 0.34 * inertial_mass + 0.24 * temporal_relativity_norm + 0.22 * field_interference_norm + 0.20 * zero_point_line_distance))
        process_resonance = NonceMath._clamp01(max(trace_state.get("process_resonance", 0.0), 0.28 * phase_ring_strength + 0.20 * temporal_relativity_norm + 0.18 * NonceMath._clamp01(trace_state.get("trace_resonance", 0.0)) + 0.18 * NonceMath._clamp01(trace_state.get("trace_alignment", 0.0)) + 0.16 * shared_vector_phase_lock))
        mining_resonance_gate = NonceMath._clamp01(max(trace_state.get("mining_resonance_gate", 0.0), 0.30 * process_resonance + 0.18 * temporal_relativity_norm + 0.14 * (1.0 - zero_point_line_distance) + 0.12 * (1.0 - field_interference_norm) + 0.12 * shared_vector_phase_lock + 0.14 * phase_ring_strength))
        collapse_readiness = NonceMath._clamp01(max(trace_state.get("collapse_readiness", 0.0), 0.28 * shared_vector_collapse + 0.22 * zero_point_crossover + 0.18 * field_interference_norm + 0.16 * resonant_interception_inertia + 0.16 * inertial_basin_strength))
        return {
            "x": float(vec_x),
            "y": float(vec_y),
            "z": float(vec_z),
            "amplitude": float(magnitude),
            "axis_scale_x": float(axis_scale_x),
            "axis_scale_y": float(axis_scale_y),
            "axis_scale_z": float(axis_scale_z),
            "vector_energy": float(vector_energy),
            "temporal_coupling_moment": float(temporal_coupling),
            "inertial_mass_proxy": float(inertial_mass),
            "relativistic_correlation": float(relativistic_correlation),
            "spin_axis_x": float(spin_axis_x),
            "spin_axis_y": float(spin_axis_y),
            "spin_axis_z": float(spin_axis_z),
            "spin_momentum_score": float(spin_score),
            "phase_ring_closure": float(phase_ring_closure),
            "phase_ring_density": float(phase_ring_density),
            "phase_ring_strength": float(phase_ring_strength),
            "zero_point_crossover_gate": float(zero_point_crossover),
            "shared_vector_collapse_gate": float(shared_vector_collapse),
            "shared_vector_phase_lock": float(shared_vector_phase_lock),
            "inertial_basin_strength": float(inertial_basin_strength),
            "temporal_relativity_norm": float(temporal_relativity_norm),
            "zero_point_line_distance": float(zero_point_line_distance),
            "field_interference_norm": float(field_interference_norm),
            "resonant_interception_inertia": float(resonant_interception_inertia),
            "process_resonance": float(process_resonance),
            "mining_resonance_gate": float(mining_resonance_gate),
            "collapse_readiness": float(collapse_readiness),
            "process_mode": str(trace_state.get("process_mode", "phase_transport" if mining_resonance_gate >= 0.60 else "memory_retain")),
            "scheduler_mode": str(trace_state.get("scheduler_mode", "transport")),
            "active_zone_name": str(trace_state.get("active_zone_name", "")),
            "active_basin_name": str(trace_state.get("active_basin_name", "")),
        }

    @staticmethod
    def _candidate_bucket_fields(
        lane_id: str,
        mode: str,
        candidate: Dict[str, Any],
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trace_state = dict(trace or {})
        coherence = NonceMath._clamp01(candidate.get("coherence", candidate.get("coherence_peak", 0.0)))
        target_alignment = NonceMath._clamp01(candidate.get("target_alignment", 0.0))
        trace_alignment = NonceMath._clamp01(candidate.get("trace_alignment", trace_state.get("trace_alignment", 0.0)))
        temporal_coupling = NonceMath._clamp01(candidate.get("temporal_coupling_moment", 0.0))
        inertial_mass = NonceMath._clamp01(candidate.get("inertial_mass_proxy", 0.0))
        spin_score = NonceMath._clamp01(candidate.get("spin_momentum_score", 0.0))
        amplitude_ratio = max(0.0, float(candidate.get("amplitude_ratio", 0.0)))
        temporal_relativity_norm = NonceMath._clamp01(candidate.get("temporal_relativity_norm", trace_state.get("trace_temporal_relativity_norm", 0.0)))
        zero_point_line_distance = NonceMath._clamp01(candidate.get("zero_point_line_distance", trace_state.get("trace_zero_point_line_distance", 0.0)))
        field_interference_norm = NonceMath._clamp01(candidate.get("field_interference_norm", trace_state.get("trace_field_interference_norm", 0.0)))
        process_resonance = NonceMath._clamp01(candidate.get("process_resonance", trace_state.get("process_resonance", 0.0)))
        mining_resonance_gate = NonceMath._clamp01(candidate.get("mining_resonance_gate", trace_state.get("mining_resonance_gate", 0.0)))
        collapse_readiness = NonceMath._clamp01(candidate.get("collapse_readiness", trace_state.get("collapse_readiness", 0.0)))
        vector_path_score = NonceMath._clamp01(
            0.26 * coherence
            + 0.16 * target_alignment
            + 0.12 * trace_alignment
            + 0.10 * temporal_coupling
            + 0.08 * spin_score
            + 0.08 * temporal_relativity_norm
            + 0.08 * (1.0 - zero_point_line_distance)
            + 0.06 * (1.0 - field_interference_norm)
            + 0.06 * process_resonance
            + 0.06 * mining_resonance_gate
            + 0.04 * (1.0 - min(1.0, amplitude_ratio + inertial_mass * 0.5))
        )
        mining_resonance_score = NonceMath._clamp01(
            0.22 * coherence
            + 0.14 * target_alignment
            + 0.12 * trace_alignment
            + 0.12 * temporal_relativity_norm
            + 0.10 * (1.0 - zero_point_line_distance)
            + 0.10 * (1.0 - field_interference_norm)
            + 0.08 * process_resonance
            + 0.08 * mining_resonance_gate
            + 0.08 * collapse_readiness
            + 0.08 * vector_path_score
            + 0.08 * (1.0 - inertial_mass)
        )
        bucket_prefix = "vg" if str(mode).lower() == "gpu_vectorized" else "pc"
        sequence_index = max(0, int(candidate.get("sequence_index", 0) or 0))
        target_interval = max(0, int(candidate.get("target_interval", 0) or 0))
        bucket_band = 1 + int(round(coherence * 8.0))
        trace_band = 1 + int(round(trace_alignment * 8.0))
        interval_band = 1 + (target_interval % 16)
        bucket_id = "%s-%02d-%02d" % (bucket_prefix, sequence_index % 64, target_interval % 64)
        worker_bucket = "p%02d-t%02d-i%02d" % (bucket_band, trace_band, interval_band)
        lane_bias = NonceMath._stable_fraction(lane_id, bucket_id, worker_bucket)
        bucket_priority = NonceMath._clamp01(0.58 * vector_path_score + 0.34 * mining_resonance_score + 0.08 * lane_bias)
        return {
            "bucket_id": bucket_id,
            "worker_bucket": worker_bucket,
            "bucket_priority": float(bucket_priority),
            "vector_path_score": float(vector_path_score),
            "mining_resonance_score": float(mining_resonance_score),
            "search_backend": "runtime_pulse_hash" if bucket_prefix == "vg" else "runtime_phase_coherence",
        }

    @staticmethod
    def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, ...]:
        return (
            float(candidate.get("mining_resonance_score", 0.0)),
            float(candidate.get("process_resonance", 0.0)),
            float(candidate.get("temporal_probe_score", 0.0)),
            float(candidate.get("gpu_feedback_delta_score", 0.0)),
            float(candidate.get("cuda_temporal_score", 0.0)),
            float(candidate.get("vector_path_score", candidate.get("coherence", 0.0))),
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
            float(candidate.get("phase_alignment_score", candidate.get("coherence", 0.0))),
            1.0 - float(candidate.get("phase_confinement_cost", candidate.get("amplitude_ratio", 1.0))),
        )

    @staticmethod
    def _target_profile(
        packet_norm: Dict[str, Any],
        env: Dict[str, float],
        batch_size: int,
        params: Dict[str, Any],
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw = packet_norm.get("raw_payload", {}) if isinstance(packet_norm, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        target_hex = NonceMath._safe_hex(raw.get("target", ""))
        if not target_hex:
            target_hex = "f" * 64
        target_hex = target_hex[-64:].rjust(64, "0")
        target_val = int(target_hex, 16)
        max256 = float((1 << 256) - 1)
        target_norm = (target_val / max256) if max256 > 0.0 else 0.0
        difficulty_norm = 1.0 - target_norm
        words = [int(target_hex[idx:idx + 8], 16) for idx in range(0, 64, 8)]
        windows: List[float] = []
        intervals: List[int] = []
        for idx, word in enumerate(words):
            primary = ((word >> ((idx % 4) * 8)) & 0xFF) / 255.0
            alt = ((word ^ ((idx + 1) * 0x9E3779B9)) & 0xFFFF) / 65535.0
            windows.append(float((0.65 * primary) + (0.35 * alt)))
            intervals.append(int(1 + (((word >> 5) ^ (idx * 131)) & 0x3F)))

        amplitude_floor = float(params.get("amplitude_floor", 0.12))
        amplitude_ceiling = float(params.get("amplitude_ceiling", 0.88))
        amplitude_cap = amplitude_floor + (target_norm * max(0.05, amplitude_ceiling - amplitude_floor))
        amplitude_cap = max(0.05, min(0.95, amplitude_cap))

        scan_override = int(params.get("sequence_scan", 0) or 0)
        if scan_override > 0:
            scan_span = max(batch_size, min(512, scan_override))
        else:
            scan_span = max(96, min(384, batch_size * (2 + int(round(difficulty_norm * 4.0)))))

        stride_scale = float(params.get("phase_stride_scale", 0.35))
        phase_stride = (float(env.get("phase_step", 0.07)) * stride_scale) + 0.002 + (difficulty_norm * 0.020)
        phase_stride = max(0.001, min(0.045, phase_stride))

        trace_state = dict(trace or {})
        trace_gate = NonceMath._clamp01(trace_state.get("trace_gate", 0.0))
        feedback_weight = NonceMath._clamp01(trace_state.get("feedback_weight", 0.0)) * trace_gate
        if feedback_weight > 0.0:
            support = NonceMath._clamp01(trace_state.get("trace_support", 0.0))
            resonance = NonceMath._clamp01(trace_state.get("trace_resonance", 0.0))
            alignment = NonceMath._clamp01(trace_state.get("trace_alignment", 0.0))
            persistence = NonceMath._clamp01(trace_state.get("trace_temporal_persistence", 0.0))
            overlap = NonceMath._clamp01(trace_state.get("trace_temporal_overlap", 0.0))
            mining_resonance_gate = NonceMath._clamp01(trace_state.get("mining_resonance_gate", 0.0))
            temporal_relativity_norm = NonceMath._clamp01(trace_state.get("trace_temporal_relativity_norm", 0.0))
            phase_anchor = float(trace_state.get("trace_phase_anchor_turns", 0.0)) % 1.0
            axis = list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]))
            scan_boost = int(trace_state.get("scan_boost", 0) or 0)
            dynamic_boost = int(round((32.0 + 96.0 * persistence + 64.0 * overlap + 72.0 * mining_resonance_gate) * feedback_weight))
            scan_span = max(batch_size, min(512, scan_span + max(scan_boost, dynamic_boost)))
            amplitude_cap = max(
                0.05,
                min(
                    0.95,
                    amplitude_cap
                    + ((support - 0.5) * 0.20 * feedback_weight)
                    + ((resonance - 0.5) * 0.12 * feedback_weight),
                ),
            )
            phase_stride *= max(0.75, min(1.35, 0.92 + (alignment * 0.30 * feedback_weight) + (overlap * 0.18 * feedback_weight) + (temporal_relativity_norm * 0.16 * feedback_weight)))
            adj_windows: List[float] = []
            adj_intervals: List[int] = []
            for idx, (window, interval) in enumerate(zip(windows, intervals)):
                axis_hint = axis[idx % len(axis)] if axis else 0.0
                phase_hint = phase_anchor if (idx % 2) == 0 else (1.0 - phase_anchor)
                adj_windows.append(
                    NonceMath._clamp01(
                        (window * (1.0 - 0.25 * feedback_weight))
                        + (axis_hint * 0.15 * feedback_weight)
                        + (phase_hint * 0.10 * feedback_weight)
                        + (support * 0.10 * feedback_weight)
                    )
                )
                interval_shift = int(round((alignment + persistence + overlap - 1.0) * 6.0 * feedback_weight))
                adj_intervals.append(max(1, min(96, int(interval) + interval_shift)))
            windows = adj_windows
            intervals = adj_intervals

        return {
            "target_hex": target_hex,
            "target_norm": float(target_norm),
            "difficulty_norm": float(difficulty_norm),
            "words": words,
            "windows": windows,
            "intervals": intervals,
            "dominant_interval": int(max(1, round(sum(intervals) / float(len(intervals) or 1)))),
            "amplitude_cap": float(amplitude_cap),
            "scan_span": int(scan_span),
            "phase_stride": float(phase_stride),
        }

    @staticmethod
    def _coherence_candidate(
        preview_state: Dict[str, Any],
        base_nonce: int,
        lane_seed: int,
        profile: Dict[str, Any],
        atomic_vector: Dict[str, float],
        scan_index: int,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        words = list(profile.get("words", []))
        windows = list(profile.get("windows", []))
        intervals = list(profile.get("intervals", []))
        if not words or not windows or not intervals:
            return None

        seq_index = scan_index % len(intervals)
        interval = int(intervals[seq_index])
        window = float(windows[seq_index])
        phase_ring = (
            float(preview_state.get("phase", 0.0))
            + (scan_index * float(profile.get("phase_stride", 0.01)))
            + (atomic_vector.get("x", 0.0) * 0.03125)
            + (window * 0.0625)
        ) % 1.0
        phase_error = abs(phase_ring - window)
        phase_error = min(phase_error, 1.0 - phase_error)

        amplitude = (
            abs(math.sin(2.0 * math.pi * phase_ring)) * (0.55 + (0.45 * atomic_vector.get("amplitude", 0.0)))
            + (abs(atomic_vector.get("y", 0.0)) * 0.35)
            + (abs(float(preview_state.get("harmonic", 0.0))) * 0.60)
            + (float(atomic_vector.get("phase_ring_strength", 0.0)) * 0.10)
            + (float(atomic_vector.get("shared_vector_phase_lock", 0.0)) * 0.08)
        )
        amplitude_cap = float(profile.get("amplitude_cap", 1.0))
        amplitude_ratio = amplitude / max(amplitude_cap, 1.0e-9)
        if amplitude_ratio > 1.0:
            return None

        target_word = int(words[seq_index])
        ring_mix = ((base_nonce >> (seq_index * 3)) ^ lane_seed ^ target_word) & 0xFFFFFFFF
        ring_norm = ring_mix / float(0xFFFFFFFF)
        target_alignment = 1.0 - min(1.0, abs(ring_norm - window))
        phase_alignment = 1.0 - min(1.0, phase_error * 2.0)
        vector_goal = (window * 2.0) - 1.0
        vector_alignment = 1.0 - min(1.0, abs(atomic_vector.get("z", 0.0) - vector_goal) * 0.5)
        temporal_relativity_norm = NonceMath._clamp01(atomic_vector.get("temporal_relativity_norm", 0.0))
        zero_point_line_distance = NonceMath._clamp01(atomic_vector.get("zero_point_line_distance", 0.0))
        field_interference_norm = NonceMath._clamp01(atomic_vector.get("field_interference_norm", 0.0))
        resonant_interception_inertia = NonceMath._clamp01(atomic_vector.get("resonant_interception_inertia", 0.0))
        process_resonance = NonceMath._clamp01(atomic_vector.get("process_resonance", 0.0))
        mining_resonance_gate = NonceMath._clamp01(atomic_vector.get("mining_resonance_gate", 0.0))
        collapse_readiness = NonceMath._clamp01(atomic_vector.get("collapse_readiness", 0.0))
        base_coherence = NonceMath._clamp01(
            (0.34 * phase_alignment)
            + (0.22 * target_alignment)
            + (0.14 * vector_alignment)
            + (0.10 * temporal_relativity_norm)
            + (0.08 * (1.0 - zero_point_line_distance))
            + (0.06 * (1.0 - field_interference_norm))
            + (0.06 * mining_resonance_gate)
        )

        trace_state = dict(trace or {})
        trace_gate = NonceMath._clamp01(trace_state.get("trace_gate", 0.0))
        feedback_weight = NonceMath._clamp01(trace_state.get("feedback_weight", 0.0)) * trace_gate
        coherence = base_coherence
        trace_axis = [0.0, 0.0, 0.0, 0.0]
        trace_memory = 0.0
        trace_flux = 0.0
        trace_vf_flux = 0.0
        trace_fv_flux = 0.0
        if feedback_weight > 0.0:
            trace_axis = list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]))
            trace_phase_anchor = float(trace_state.get("trace_phase_anchor_turns", phase_ring)) % 1.0
            trace_memory = NonceMath._clamp01(trace_state.get("trace_memory", 0.0))
            trace_flux = NonceMath._clamp01(trace_state.get("trace_flux", 0.0))
            trace_vf_flux = NonceMath._clamp01(trace_state.get("trace_voltage_frequency_flux", 0.0))
            trace_fv_flux = NonceMath._clamp01(trace_state.get("trace_frequency_voltage_flux", 0.0))
            persistence = NonceMath._clamp01(trace_state.get("trace_temporal_persistence", 0.0))
            overlap = NonceMath._clamp01(trace_state.get("trace_temporal_overlap", 0.0))
            trace_phase_ring_strength = NonceMath._clamp01(trace_state.get("trace_phase_ring_strength", 0.0))
            trace_shared_phase_lock = NonceMath._clamp01(trace_state.get("trace_shared_vector_phase_lock", 0.0))
            trace_zero_point = NonceMath._clamp01(trace_state.get("trace_zero_point_crossover", 0.0))
            trace_phase_error = abs(phase_ring - trace_phase_anchor)
            trace_phase_error = min(trace_phase_error, 1.0 - trace_phase_error)
            trace_phase_alignment = 1.0 - min(1.0, trace_phase_error * 2.0)
            axis_goal = 0.5 * (window + ring_norm)
            axis_alignment = 1.0 - min(1.0, abs(trace_axis[0] - axis_goal))
            substrate_alignment = NonceMath._clamp01(
                (0.34 * trace_phase_alignment)
                + (0.18 * axis_alignment)
                + (0.16 * persistence)
                + (0.12 * overlap)
                + (0.10 * trace_memory)
                + (0.05 * trace_vf_flux)
                + (0.05 * trace_fv_flux)
                + (0.05 * trace_phase_ring_strength)
                + (0.03 * trace_shared_phase_lock)
                + (0.03 * trace_zero_point)
            )
            trace_coherence = NonceMath._clamp01(
                (0.42 * phase_alignment)
                + (0.24 * target_alignment)
                + (0.14 * vector_alignment)
                + (0.20 * substrate_alignment)
            )
            coherence = NonceMath._clamp01(
                ((1.0 - feedback_weight) * base_coherence)
                + (feedback_weight * trace_coherence)
            )

        vector_mix = int(round(
            (atomic_vector.get("x", 0.0) + atomic_vector.get("y", 0.0) + atomic_vector.get("z", 0.0)) * 2048.0
        ))
        ring_offset = int(round(phase_ring * 65535.0))
        window_offset = int(round(window * 4095.0))
        if feedback_weight > 0.0:
            vector_mix += int(round(((trace_memory - 0.5) + (trace_flux - 0.5) + (trace_vf_flux - trace_fv_flux)) * 2048.0 * feedback_weight))
            ring_offset += int(round(((trace_state.get("trace_phase_anchor_turns", 0.0) % 1.0) * 2047.0) * feedback_weight))
            window_offset += int(round((trace_axis[1] * 1023.0) * feedback_weight))
        window_offset += int(round((temporal_relativity_norm - 0.5) * 1024.0))
        vector_mix += int(round(((mining_resonance_gate - 0.5) - (field_interference_norm - 0.5)) * 512.0))
        nonce = (base_nonce + (scan_index * interval) + ring_offset + window_offset + vector_mix) & 0xFFFFFFFF

        return {
            "nonce": int(nonce),
            "coherence": float(coherence),
            "amplitude_ratio": float(amplitude_ratio),
            "phase_ring": float(phase_ring),
            "target_interval": int(interval),
            "sequence_index": int(seq_index),
            "target_alignment": float(target_alignment),
            "phase_alignment_score": float(phase_alignment),
            "trace_alignment": float(trace_state.get("trace_alignment", 0.0)),
            "motif_alignment": float(NonceMath._clamp01((coherence + target_alignment) * 0.5)),
            "phase_confinement_cost": float(NonceMath._clamp01(amplitude_ratio)),
            "axis_scale_x": float(atomic_vector.get("axis_scale_x", 0.0)),
            "axis_scale_y": float(atomic_vector.get("axis_scale_y", 0.0)),
            "axis_scale_z": float(atomic_vector.get("axis_scale_z", 0.0)),
            "vector_energy": float(atomic_vector.get("vector_energy", 0.0)),
            "temporal_coupling_moment": float(atomic_vector.get("temporal_coupling_moment", 0.0)),
            "inertial_mass_proxy": float(atomic_vector.get("inertial_mass_proxy", 0.0)),
            "relativistic_correlation": float(atomic_vector.get("relativistic_correlation", 0.0)),
            "spin_axis_x": float(atomic_vector.get("spin_axis_x", 0.0)),
            "spin_axis_y": float(atomic_vector.get("spin_axis_y", 0.0)),
            "spin_axis_z": float(atomic_vector.get("spin_axis_z", 0.0)),
            "spin_momentum_score": float(atomic_vector.get("spin_momentum_score", 0.0)),
            "phase_ring_closure": float(atomic_vector.get("phase_ring_closure", 0.0)),
            "phase_ring_density": float(atomic_vector.get("phase_ring_density", 0.0)),
            "phase_ring_strength": float(atomic_vector.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": float(atomic_vector.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": float(atomic_vector.get("shared_vector_collapse_gate", 0.0)),
            "shared_vector_phase_lock": float(atomic_vector.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": float(atomic_vector.get("inertial_basin_strength", 0.0)),
            "temporal_relativity_norm": float(temporal_relativity_norm),
            "zero_point_line_distance": float(zero_point_line_distance),
            "field_interference_norm": float(field_interference_norm),
            "resonant_interception_inertia": float(resonant_interception_inertia),
            "process_resonance": float(process_resonance),
            "mining_resonance_gate": float(mining_resonance_gate),
            "collapse_readiness": float(collapse_readiness),
            "process_mode": str(atomic_vector.get("process_mode", "")),
            "scheduler_mode": str(atomic_vector.get("scheduler_mode", "")),
            "active_zone_name": str(atomic_vector.get("active_zone_name", "")),
            "active_basin_name": str(atomic_vector.get("active_basin_name", "")),
        }

    @staticmethod
    def _entropy_from_nonces(nonces: List[int]) -> float:
        if not nonces:
            return 0.0
        uniq = len(set(int(n) & 0xFFFFF for n in nonces))
        spread = max(nonces) - min(nonces) if len(nonces) > 1 else 0
        uniq_score = uniq / float(len(nonces))
        spread_score = min(1.0, float(spread) / float(0xFFFFFFFF))
        return NonceMath._clamp01((0.60 * uniq_score) + (0.40 * spread_score))

    @staticmethod
    def _emit_phase_coherence_nonces(
        packet_norm: Dict[str, Any],
        env: Dict[str, float],
        state: Dict[str, Any],
        lane_seed: int,
        batch_size: int,
        params: Dict[str, Any],
    ) -> Tuple[List[int], Dict[str, Any]]:
        preview_state = dict(state)
        NonceMath._dmt_update(preview_state, env)
        base_nonce = NonceMath._derive_nonce(preview_state, lane_seed)
        trace = NonceMath._substrate_trace_feedback(packet_norm, params)
        atomic_vector = NonceMath._atomic_vector(env, preview_state, trace=trace)
        profile = NonceMath._target_profile(packet_norm, env, batch_size, params, trace=trace)

        candidates: List[Dict[str, Any]] = []
        seen = set()
        for scan_index in range(int(profile.get("scan_span", batch_size))):
            preview_state["phase"] = (
                float(preview_state.get("phase", 0.0)) + float(profile.get("phase_stride", 0.01))
            ) % 1.0
            candidate = NonceMath._coherence_candidate(
                preview_state=preview_state,
                base_nonce=base_nonce,
                lane_seed=lane_seed,
                profile=profile,
                atomic_vector=atomic_vector,
                scan_index=scan_index,
                trace=trace,
            )
            if candidate is None:
                continue
            nonce = int(candidate.get("nonce", 0)) & 0xFFFFFFFF
            if nonce in seen:
                continue
            seen.add(nonce)
            candidate["coherence_peak"] = float(candidate.get("coherence", 0.0))
            candidate["sequence_persistence_score"] = NonceMath._clamp01(
                (float(trace.get("trace_temporal_persistence", 0.0))
                + float(candidate.get("coherence", 0.0))
                + float(atomic_vector.get("temporal_coupling_moment", 0.0))) / 3.0
            )
            candidate["temporal_index_overlap"] = NonceMath._clamp01(
                max(float(trace.get("trace_temporal_overlap", 0.0)), float(atomic_vector.get("temporal_coupling_moment", 0.0)))
            )
            candidate.update(
                NonceMath._candidate_bucket_fields(
                    lane_id="phase",
                    mode="phase_coherence",
                    candidate=candidate,
                    trace=trace,
                )
            )
            candidates.append(candidate)

        if not candidates:
            state["candidate_count"] = 0
            state["coherence_peak"] = 0.0
            state["amplitude_cap"] = float(profile.get("amplitude_cap", 0.0))
            state["target_interval"] = int(profile.get("dominant_interval", 0))
            state["atomic_vector_x"] = float(atomic_vector.get("x", 0.0))
            state["atomic_vector_y"] = float(atomic_vector.get("y", 0.0))
            state["atomic_vector_z"] = float(atomic_vector.get("z", 0.0))
            return [], {"selected": [], "profile": profile, "atomic_vector": atomic_vector}

        candidates.sort(key=NonceMath._candidate_sort_key, reverse=True)
        selected = candidates[:max(1, batch_size)]
        nonces = [int(item.get("nonce", 0)) & 0xFFFFFFFF for item in selected]

        state["base_nonce"] = int(preview_state.get("base_nonce", state.get("base_nonce", 0)))
        state["d1"] = int(preview_state.get("d1", state.get("d1", 0)))
        state["psi"] = float(preview_state.get("psi", state.get("psi", 0.0)))
        state["flux"] = float(preview_state.get("flux", state.get("flux", 0.0)))
        state["harmonic"] = float(preview_state.get("harmonic", state.get("harmonic", 0.0)))
        state["phase"] = float(preview_state.get("phase", state.get("phase", 0.0)))
        state["last_nonce"] = int(nonces[-1])
        state["entropy_score"] = float(NonceMath._entropy_from_nonces(nonces))
        state["amplitude_cap"] = float(profile.get("amplitude_cap", 0.0))
        state["coherence_peak"] = float(max(item.get("coherence", 0.0) for item in selected))
        state["target_interval"] = int(profile.get("dominant_interval", 0))
        state["candidate_count"] = int(len(candidates))
        state["validated_candidate_count"] = int(len(selected))
        state["valid_share_count"] = 0
        state["valid_ratio"] = 0.0
        state["atomic_vector_x"] = float(atomic_vector.get("x", 0.0))
        state["atomic_vector_y"] = float(atomic_vector.get("y", 0.0))
        state["atomic_vector_z"] = float(atomic_vector.get("z", 0.0))
        best_candidate = dict(selected[0] if selected else {})
        for key in (
            "axis_scale_x",
            "axis_scale_y",
            "axis_scale_z",
            "vector_energy",
            "temporal_coupling_moment",
            "inertial_mass_proxy",
            "relativistic_correlation",
            "spin_axis_x",
            "spin_axis_y",
            "spin_axis_z",
            "spin_momentum_score",
            "phase_ring_closure",
            "phase_ring_density",
            "phase_ring_strength",
            "zero_point_crossover_gate",
            "shared_vector_collapse_gate",
            "shared_vector_phase_lock",
            "inertial_basin_strength",
            "temporal_relativity_norm",
            "zero_point_line_distance",
            "field_interference_norm",
            "resonant_interception_inertia",
            "process_resonance",
            "mining_resonance_gate",
            "collapse_readiness",
        ):
            state[key] = float(best_candidate.get(key, atomic_vector.get(key, 0.0)))
        state["search_backend"] = str(best_candidate.get("search_backend", "runtime_phase_coherence"))
        state["vector_path_score"] = float(max([item.get("vector_path_score", 0.0) for item in selected] or [0.0]))
        state["mining_resonance_score"] = float(max([item.get("mining_resonance_score", 0.0) for item in selected] or [0.0]))
        state["process_mode"] = str(best_candidate.get("process_mode", atomic_vector.get("process_mode", "")))
        state["scheduler_mode"] = str(best_candidate.get("scheduler_mode", atomic_vector.get("scheduler_mode", "")))
        state["active_zone_name"] = str(best_candidate.get("active_zone_name", atomic_vector.get("active_zone_name", "")))
        state["active_basin_name"] = str(best_candidate.get("active_basin_name", atomic_vector.get("active_basin_name", "")))

        return nonces, {
            "selected": selected,
            "profile": profile,
            "atomic_vector": atomic_vector,
            "trace": trace,
        }

    @staticmethod
    def _build_runtime_simulation_field(
        atomic_vector: Dict[str, Any],
        state: Dict[str, Any],
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        coherence = NonceMath._clamp01(state.get("coherence_peak", 0.0))
        entropy = NonceMath._clamp01(state.get("entropy_score", 0.0))
        return {
            "simulation_field_vector": [
                float(atomic_vector.get("x", 0.0)) * (0.5 + 0.5 * NonceMath._clamp01(atomic_vector.get("axis_scale_x", 0.0))),
                float(atomic_vector.get("y", 0.0)) * (0.5 + 0.5 * NonceMath._clamp01(atomic_vector.get("axis_scale_y", 0.0))),
                float(atomic_vector.get("z", 0.0)) * (0.5 + 0.5 * NonceMath._clamp01(atomic_vector.get("axis_scale_z", 0.0))),
                float(state.get("phase", 0.0)) % 1.0,
            ],
            "sequence_persistence_score": NonceMath._clamp01(max(float(trace.get("trace_temporal_persistence", 0.0)), entropy, float(atomic_vector.get("temporal_coupling_moment", 0.0)))),
            "temporal_index_overlap": NonceMath._clamp01(max(float(trace.get("trace_temporal_overlap", 0.0)), float(atomic_vector.get("temporal_coupling_moment", 0.0)))),
            "voltage_frequency_flux": NonceMath._clamp01(max(float(trace.get("trace_voltage_frequency_flux", 0.0)), abs(float(atomic_vector.get("z", 0.0))))),
            "frequency_voltage_flux": NonceMath._clamp01(max(float(trace.get("trace_frequency_voltage_flux", 0.0)), abs(float(atomic_vector.get("x", 0.0))))),
            "field_alignment_score": NonceMath._clamp01((float(trace.get("trace_alignment", 0.0)) + coherence + float(atomic_vector.get("spin_momentum_score", 0.0))) / 3.0),
            "kernel_control_gate": NonceMath._clamp01(max(float(trace.get("trace_gate", 0.0)), coherence, float(atomic_vector.get("inertial_mass_proxy", 0.0)), float(atomic_vector.get("mining_resonance_gate", 0.0)))),
            "axis_scale_x": float(atomic_vector.get("axis_scale_x", 0.0)),
            "axis_scale_y": float(atomic_vector.get("axis_scale_y", 0.0)),
            "axis_scale_z": float(atomic_vector.get("axis_scale_z", 0.0)),
            "temporal_coupling_moment": float(atomic_vector.get("temporal_coupling_moment", 0.0)),
            "inertial_mass_proxy": float(atomic_vector.get("inertial_mass_proxy", 0.0)),
            "spin_momentum_score": float(atomic_vector.get("spin_momentum_score", 0.0)),
            "temporal_relativity_norm": float(atomic_vector.get("temporal_relativity_norm", 0.0)),
            "zero_point_line_distance": float(atomic_vector.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": float(atomic_vector.get("field_interference_norm", 0.0)),
            "resonant_interception_inertia": float(atomic_vector.get("resonant_interception_inertia", 0.0)),
            "process_resonance": float(atomic_vector.get("process_resonance", 0.0)),
            "mining_resonance_gate": float(atomic_vector.get("mining_resonance_gate", 0.0)),
            "collapse_readiness": float(atomic_vector.get("collapse_readiness", 0.0)),
            "process_mode": str(atomic_vector.get("process_mode", "")),
            "scheduler_mode": str(atomic_vector.get("scheduler_mode", "")),
        }

    @staticmethod
    def _expand_temporal_probe_candidates(
        candidate_pool: List[Dict[str, Any]],
        trace: Dict[str, Any],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        enabled_default = params.get("pulse_hash_audit_mode", params.get("pow_test_mode", False))
        if not NonceMath._flag_enabled(params.get("temporal_probe_enabled", enabled_default)):
            return list(candidate_pool or [])
        working = [dict(item or {}) for item in list(candidate_pool or [])]
        if not working:
            return working
        seed_budget = max(2, min(16, int(params.get("temporal_probe_seed_budget", 8) or 8)))
        variants_per_seed = max(2, min(6, int(params.get("temporal_probe_variants_per_seed", 4) or 4)))
        offset_scale = max(128, int(params.get("temporal_probe_offset_scale", 2048) or 2048))
        ranked = sorted(working, key=NonceMath._candidate_sort_key, reverse=True)
        seeds = ranked[:min(len(ranked), seed_budget)]
        seen = {int(item.get("nonce", 0)) & 0xFFFFFFFF for item in working}
        expanded = list(working)
        trace_alignment = NonceMath._clamp01(trace.get("trace_alignment", 0.0))
        for seed_rank, candidate in enumerate(seeds):
            temporal_probe_score = NonceMath._clamp01(
                0.20 * NonceMath._clamp01(candidate.get("sequence_persistence_score", 0.0))
                + 0.18 * NonceMath._clamp01(candidate.get("temporal_index_overlap", 0.0))
                + 0.16 * NonceMath._clamp01(candidate.get("temporal_relativity_norm", 0.0))
                + 0.12 * NonceMath._clamp01(candidate.get("mining_resonance_gate", 0.0))
                + 0.10 * NonceMath._clamp01(candidate.get("process_resonance", 0.0))
                + 0.10 * (1.0 - NonceMath._clamp01(candidate.get("zero_point_line_distance", 0.0)))
                + 0.08 * (1.0 - NonceMath._clamp01(candidate.get("field_interference_norm", 0.0)))
                + 0.06 * trace_alignment
            )
            base_offset = max(8, int(round((0.24 + temporal_probe_score * 0.76) * float(offset_scale))))
            offsets = [
                base_offset,
                -base_offset,
                base_offset // 2,
                -(base_offset // 2),
                base_offset + (1 + seed_rank) * 17,
                -(base_offset + (1 + seed_rank) * 17),
            ]
            for offset in offsets[:variants_per_seed]:
                probe_nonce = (int(candidate.get("nonce", 0)) + int(offset)) & 0xFFFFFFFF
                if probe_nonce in seen:
                    continue
                seen.add(probe_nonce)
                variant = dict(candidate)
                variant["nonce"] = int(probe_nonce)
                variant["temporal_probe_seed_rank"] = int(seed_rank)
                variant["temporal_probe_offset"] = int(offset)
                variant["temporal_probe_score"] = float(temporal_probe_score)
                variant["cuda_temporal_score"] = NonceMath._clamp01(float(candidate.get("cuda_temporal_score", 0.0)) + 0.10 * temporal_probe_score)
                variant["gpu_feedback_delta_score"] = NonceMath._clamp01(float(candidate.get("gpu_feedback_delta_score", 0.0)) + 0.08 * temporal_probe_score)
                variant["vector_path_score"] = NonceMath._clamp01(float(candidate.get("vector_path_score", 0.0)) + 0.06 * temporal_probe_score)
                variant["mining_resonance_score"] = NonceMath._clamp01(float(candidate.get("mining_resonance_score", 0.0)) + 0.08 * temporal_probe_score)
                expanded.append(variant)
        return expanded

    @staticmethod
    def _emit_gpu_vectorized_nonces(
        packet_norm: Dict[str, Any],
        env: Dict[str, float],
        state: Dict[str, Any],
        lane_id: str,
        lane_seed: int,
        batch_size: int,
        params: Dict[str, Any],
    ) -> Tuple[List[int], Dict[str, Any]]:
        from miner.gpu_vectorized_nonce_search import rank_candidates

        preview_state = dict(state)
        NonceMath._dmt_update(preview_state, env)
        base_nonce = NonceMath._derive_nonce(preview_state, lane_seed)
        trace = NonceMath._substrate_trace_feedback(packet_norm, params)
        atomic_vector = NonceMath._atomic_vector(env, preview_state, trace=trace)
        profile = NonceMath._target_profile(packet_norm, env, batch_size, params, trace=trace)

        scan_override = int(params.get("gpu_batch_size", 0) or 0)
        search_span = max(batch_size, min(2048, scan_override if scan_override > 0 else int(profile.get("scan_span", batch_size))))
        candidate_pool: List[Dict[str, Any]] = []
        seen = set()
        for scan_index in range(int(search_span)):
            preview_state["phase"] = (
                float(preview_state.get("phase", 0.0)) + float(profile.get("phase_stride", 0.01))
            ) % 1.0
            candidate = NonceMath._coherence_candidate(
                preview_state=preview_state,
                base_nonce=base_nonce,
                lane_seed=lane_seed,
                profile=profile,
                atomic_vector=atomic_vector,
                scan_index=scan_index,
                trace=trace,
            )
            if candidate is None:
                continue
            nonce = int(candidate.get("nonce", 0)) & 0xFFFFFFFF
            if nonce in seen:
                continue
            seen.add(nonce)
            candidate["coherence_peak"] = float(candidate.get("coherence", 0.0))
            candidate["sequence_persistence_score"] = NonceMath._clamp01(
                (float(trace.get("trace_temporal_persistence", 0.0))
                + float(candidate.get("coherence", 0.0))
                + float(atomic_vector.get("temporal_coupling_moment", 0.0))) / 3.0
            )
            candidate["temporal_index_overlap"] = NonceMath._clamp01(
                max(float(trace.get("trace_temporal_overlap", 0.0)), float(atomic_vector.get("temporal_coupling_moment", 0.0)))
            )
            candidate["gpu_feedback_delta_score"] = NonceMath._clamp01(
                (float(trace.get("trace_alignment", 0.0))
                + float(trace.get("trace_flux", 0.0))
                + float(candidate.get("target_alignment", 0.0))
                + float(candidate.get("mining_resonance_gate", 0.0))) / 4.0
            )
            candidate["cuda_temporal_score"] = NonceMath._clamp01(
                (float(candidate.get("phase_alignment_score", 0.0))
                + float(candidate.get("coherence", 0.0))
                + float(trace.get("trace_temporal_persistence", 0.0))
                + float(candidate.get("temporal_relativity_norm", 0.0))) / 4.0
            )
            candidate.update(
                NonceMath._candidate_bucket_fields(
                    lane_id=lane_id,
                    mode="gpu_vectorized",
                    candidate=candidate,
                    trace=trace,
                )
            )
            candidate_pool.append(candidate)

        candidate_pool = NonceMath._expand_temporal_probe_candidates(
            candidate_pool=candidate_pool,
            trace=trace,
            params=params,
        )

        if not candidate_pool:
            state["candidate_count"] = 0
            state["validated_candidate_count"] = 0
            state["valid_share_count"] = 0
            state["valid_ratio"] = 0.0
            state["search_backend"] = "runtime_pulse_hash"
            return [], {"selected": [], "profile": profile, "atomic_vector": atomic_vector, "trace": trace}

        simulation_field_state = NonceMath._build_runtime_simulation_field(
            atomic_vector=atomic_vector,
            state=preview_state,
            trace=trace,
        )
        validation_count = int(params.get("validation_window_count", 0) or 0)
        if validation_count <= batch_size and "gpu_batch_size" in params:
            try:
                validation_count = max(batch_size + 1, min(32, int(params.get("gpu_batch_size", batch_size) or batch_size) // 4))
            except Exception:
                validation_count = batch_size
        validation_count = max(batch_size, min(len(candidate_pool), validation_count if validation_count > 0 else batch_size))
        ranked = rank_candidates(
            candidate_pool=candidate_pool,
            batch_size=validation_count,
            simulation_field_state=simulation_field_state,
            target_profile=profile,
            ranking_params=params,
        )
        selected = list(ranked.get("selected", []) or [])
        nonces = [int(item.get("nonce", 0)) & 0xFFFFFFFF for item in selected]

        state["base_nonce"] = int(preview_state.get("base_nonce", state.get("base_nonce", 0)))
        state["d1"] = int(preview_state.get("d1", state.get("d1", 0)))
        state["psi"] = float(preview_state.get("psi", state.get("psi", 0.0)))
        state["flux"] = float(preview_state.get("flux", state.get("flux", 0.0)))
        state["harmonic"] = float(preview_state.get("harmonic", state.get("harmonic", 0.0)))
        state["phase"] = float(preview_state.get("phase", state.get("phase", 0.0)))
        if nonces:
            state["last_nonce"] = int(nonces[-1])
        state["entropy_score"] = float(NonceMath._entropy_from_nonces(nonces))
        state["amplitude_cap"] = float(profile.get("amplitude_cap", 0.0))
        state["coherence_peak"] = float(max([item.get("coherence", 0.0) for item in selected] or [0.0]))
        state["target_interval"] = int(profile.get("dominant_interval", 0))
        state["candidate_count"] = int(len(candidate_pool))
        state["validated_candidate_count"] = int(len(selected))
        state["valid_share_count"] = 0
        state["valid_ratio"] = 0.0
        state["atomic_vector_x"] = float(atomic_vector.get("x", 0.0))
        state["atomic_vector_y"] = float(atomic_vector.get("y", 0.0))
        state["atomic_vector_z"] = float(atomic_vector.get("z", 0.0))
        best_candidate = dict(selected[0] if selected else {})
        for key in (
            "axis_scale_x",
            "axis_scale_y",
            "axis_scale_z",
            "vector_energy",
            "temporal_coupling_moment",
            "inertial_mass_proxy",
            "relativistic_correlation",
            "spin_axis_x",
            "spin_axis_y",
            "spin_axis_z",
            "spin_momentum_score",
            "phase_ring_closure",
            "phase_ring_density",
            "phase_ring_strength",
            "zero_point_crossover_gate",
            "shared_vector_collapse_gate",
            "shared_vector_phase_lock",
            "inertial_basin_strength",
            "temporal_relativity_norm",
            "zero_point_line_distance",
            "field_interference_norm",
            "resonant_interception_inertia",
            "process_resonance",
            "mining_resonance_gate",
            "collapse_readiness",
        ):
            state[key] = float(best_candidate.get(key, atomic_vector.get(key, 0.0)))
        state["search_backend"] = str(best_candidate.get("search_backend", "runtime_pulse_hash"))
        state["vector_path_score"] = float(max([item.get("vector_path_score", 0.0) for item in selected] or [0.0]))
        state["mining_resonance_score"] = float(max([item.get("mining_resonance_score", 0.0) for item in selected] or [0.0]))
        state["process_mode"] = str(best_candidate.get("process_mode", atomic_vector.get("process_mode", "")))
        state["scheduler_mode"] = str(best_candidate.get("scheduler_mode", atomic_vector.get("scheduler_mode", "")))
        state["active_zone_name"] = str(best_candidate.get("active_zone_name", atomic_vector.get("active_zone_name", "")))
        state["active_basin_name"] = str(best_candidate.get("active_basin_name", atomic_vector.get("active_basin_name", "")))

        return nonces, {
            "selected": selected,
            "profile": profile,
            "atomic_vector": atomic_vector,
            "trace": trace,
            "telemetry": dict(ranked.get("telemetry", {}) or {}),
            "simulation_field_state": simulation_field_state,
        }

    @staticmethod
    def compute(
        packet: neural_objectPacket,
        lane_state: Optional[Any] = None,
        mode: str = "derivative",
        count: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[neural_objectPacket]:
        schema = neural_objectSchema.get(packet.packet_type)
        if not schema:
            return []

        packet_norm = schema["convert_incoming"](packet.raw_payload)
        if not isinstance(packet_norm, dict):
            return []
        merged_sp = dict(packet_norm.get("system_payload", {}) or {})
        merged_sp.update(dict(packet.system_payload or {}))
        packet_norm["system_payload"] = merged_sp
        env = NonceMath._env_from_norm(packet_norm)
        phase_params = dict(params or {})

        state = NonceMath._init_lane_state(lane_state)
        lane_seed = NonceMath._lane_seed(lane_state)
        lane_id_str = NonceMath._lane_id_str(lane_state)

        try:
            batch_size = int(schema["derive_batch_size"](packet_norm))
        except Exception:
            batch_size = 1
        try:
            if count is not None:
                override_count = int(count)
                if override_count > 0:
                    batch_size = override_count
        except Exception:
            pass
        batch_size = max(1, int(batch_size))

        nonces: List[int] = []
        coherence_meta: Dict[str, Any] = {
            "selected": [],
            "profile": {},
            "atomic_vector": {},
        }
        selected_by_nonce: Dict[int, Dict[str, Any]] = {}
        selected_by_index: List[Dict[str, Any]] = []

        m = (mode or "derivative").lower()
        if m == "single":
            NonceMath._dmt_update(state, env)
            nonce = NonceMath._derive_nonce(state, lane_seed)
            nonces.append(nonce)
            NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "single")
        elif m == "vector":
            NonceMath._dmt_update(state, env)
            base = NonceMath._derive_nonce(state, lane_seed)
            for idx in range(batch_size):
                state["phase"] = (float(state.get("phase", 0.0)) + 0.003) % 1.0
                nonce = (base + idx) & 0xFFFFFFFF
                nonces.append(nonce)
                NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "vector")
        elif m == "hybrid":
            NonceMath._dmt_update(state, env)
            base = NonceMath._derive_nonce(state, lane_seed)
            vec: List[int] = []
            for idx in range(max(1, batch_size // 2)):
                state["phase"] = (float(state.get("phase", 0.0)) + 0.005) % 1.0
                nonce = (base + (idx * 2 + 1)) & 0xFFFFFFFF
                vec.append(nonce)
                NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "hybrid:phase")
            for _ in range(batch_size - len(vec)):
                NonceMath._dmt_update(state, env)
                nonce = NonceMath._derive_nonce(state, lane_seed)
                vec.append(nonce)
                NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "hybrid:derive")
            nonces = vec
        elif m == "phase_coherence":
            nonces, coherence_meta = NonceMath._emit_phase_coherence_nonces(
                packet_norm=packet_norm,
                env=env,
                state=state,
                lane_seed=lane_seed,
                batch_size=batch_size,
                params=phase_params,
            )
            selected_by_index = list(coherence_meta.get("selected", []) or [])
            selected_by_nonce = {
                int(item.get("nonce", 0)) & 0xFFFFFFFF: dict(item) for item in selected_by_index
            }
            for nonce in nonces:
                NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "phase_coherence")
        elif m == "gpu_vectorized":
            nonces, coherence_meta = NonceMath._emit_gpu_vectorized_nonces(
                packet_norm=packet_norm,
                env=env,
                state=state,
                lane_id=lane_id_str,
                lane_seed=lane_seed,
                batch_size=batch_size,
                params=phase_params,
            )
            selected_by_index = list(coherence_meta.get("selected", []) or [])
            selected_by_nonce = {
                int(item.get("nonce", 0)) & 0xFFFFFFFF: dict(item) for item in selected_by_index
            }
            for nonce in nonces:
                NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "gpu_vectorized")
        else:
            for _ in range(batch_size):
                NonceMath._dmt_update(state, env)
                nonce = NonceMath._derive_nonce(state, lane_seed)
                nonces.append(nonce)
                NonceMath._debug_dump(NonceMath._DEBUG, lane_id_str, nonce, state, "derivative")

        if m not in ("phase_coherence", "gpu_vectorized"):
            state["candidate_count"] = len(nonces)
            state["entropy_score"] = float(NonceMath._entropy_from_nonces(nonces))
            if nonces:
                state["last_nonce"] = int(nonces[-1])
            state["valid_ratio"] = 0.0
            if "validated_candidate_count" not in state:
                state["validated_candidate_count"] = int(len(nonces))
            if "valid_share_count" not in state:
                state["valid_share_count"] = 0

        if not nonces:
            NonceMath._persist_lane_state(lane_state, state)
            return []

        results: List[neural_objectPacket] = []
        valid_count = 0
        normalized_job = dict(packet_norm.get("raw_payload", {}) or {})
        for idx, nonce in enumerate(nonces):
            try:
                pow_hex = schema["hash_function"](packet_norm, int(nonce))
            except Exception:
                pow_hex = ""
            try:
                is_valid = bool(schema["verify_target"](packet_norm, pow_hex))
            except Exception:
                is_valid = False
            if is_valid:
                valid_count += 1

            coherence_info = dict(selected_by_nonce.get(int(nonce) & 0xFFFFFFFF, {}))
            if not coherence_info and idx < len(selected_by_index):
                coherence_info = dict(selected_by_index[idx])

            packet_out = neural_objectPacket(
                packet_type=packet.packet_type,
                network=packet.network,
                raw_payload=dict(packet.raw_payload or {}),
                system_payload={
                    "nonce": int(nonce),
                    "pow": str(pow_hex),
                    "valid": bool(is_valid),
                    "job": dict(normalized_job),
                    "lane": lane_id_str,
                    "coherence": float(coherence_info.get("coherence", 0.0)),
                    "phase_ring": float(coherence_info.get("phase_ring", 0.0)),
                    "amplitude_ratio": float(coherence_info.get("amplitude_ratio", 0.0)),
                    "target_interval": int(coherence_info.get("target_interval", state.get("target_interval", 0))),
                    "sequence_index": int(coherence_info.get("sequence_index", 0)),
                    "target_alignment": float(coherence_info.get("target_alignment", 0.0)),
                    "phase_alignment_score": float(coherence_info.get("phase_alignment_score", 0.0)),
                    "trace_alignment": float(coherence_info.get("trace_alignment", 0.0)),
                    "motif_alignment": float(coherence_info.get("motif_alignment", 0.0)),
                    "phase_confinement_cost": float(coherence_info.get("phase_confinement_cost", coherence_info.get("amplitude_ratio", 0.0))),
                    "bucket_id": str(coherence_info.get("bucket_id", "")),
                    "worker_bucket": str(coherence_info.get("worker_bucket", "")),
                    "bucket_priority": float(coherence_info.get("bucket_priority", 0.0)),
                    "search_backend": str(coherence_info.get("search_backend", state.get("search_backend", ""))),
                    "vector_path_score": float(coherence_info.get("vector_path_score", 0.0)),
                    "mining_resonance_score": float(coherence_info.get("mining_resonance_score", 0.0)),
                    "axis_scale_x": float(coherence_info.get("axis_scale_x", state.get("axis_scale_x", 0.0))),
                    "axis_scale_y": float(coherence_info.get("axis_scale_y", state.get("axis_scale_y", 0.0))),
                    "axis_scale_z": float(coherence_info.get("axis_scale_z", state.get("axis_scale_z", 0.0))),
                    "vector_energy": float(coherence_info.get("vector_energy", state.get("vector_energy", 0.0))),
                    "temporal_coupling_moment": float(coherence_info.get("temporal_coupling_moment", state.get("temporal_coupling_moment", 0.0))),
                    "inertial_mass_proxy": float(coherence_info.get("inertial_mass_proxy", state.get("inertial_mass_proxy", 0.0))),
                    "relativistic_correlation": float(coherence_info.get("relativistic_correlation", state.get("relativistic_correlation", 0.0))),
                    "spin_axis_x": float(coherence_info.get("spin_axis_x", state.get("spin_axis_x", 0.0))),
                    "spin_axis_y": float(coherence_info.get("spin_axis_y", state.get("spin_axis_y", 0.0))),
                    "spin_axis_z": float(coherence_info.get("spin_axis_z", state.get("spin_axis_z", 0.0))),
                    "spin_momentum_score": float(coherence_info.get("spin_momentum_score", state.get("spin_momentum_score", 0.0))),
                    "phase_ring_closure": float(coherence_info.get("phase_ring_closure", state.get("phase_ring_closure", 0.0))),
                    "phase_ring_density": float(coherence_info.get("phase_ring_density", state.get("phase_ring_density", 0.0))),
                    "phase_ring_strength": float(coherence_info.get("phase_ring_strength", state.get("phase_ring_strength", 0.0))),
                    "zero_point_crossover_gate": float(coherence_info.get("zero_point_crossover_gate", state.get("zero_point_crossover_gate", 0.0))),
                    "shared_vector_collapse_gate": float(coherence_info.get("shared_vector_collapse_gate", state.get("shared_vector_collapse_gate", 0.0))),
                    "shared_vector_phase_lock": float(coherence_info.get("shared_vector_phase_lock", state.get("shared_vector_phase_lock", 0.0))),
                    "inertial_basin_strength": float(coherence_info.get("inertial_basin_strength", state.get("inertial_basin_strength", 0.0))),
                    "temporal_relativity_norm": float(coherence_info.get("temporal_relativity_norm", state.get("temporal_relativity_norm", 0.0))),
                    "zero_point_line_distance": float(coherence_info.get("zero_point_line_distance", state.get("zero_point_line_distance", 0.0))),
                    "field_interference_norm": float(coherence_info.get("field_interference_norm", state.get("field_interference_norm", 0.0))),
                    "resonant_interception_inertia": float(coherence_info.get("resonant_interception_inertia", state.get("resonant_interception_inertia", 0.0))),
                    "process_resonance": float(coherence_info.get("process_resonance", state.get("process_resonance", 0.0))),
                    "mining_resonance_gate": float(coherence_info.get("mining_resonance_gate", state.get("mining_resonance_gate", 0.0))),
                    "collapse_readiness": float(coherence_info.get("collapse_readiness", state.get("collapse_readiness", 0.0))),
                    "process_mode": str(coherence_info.get("process_mode", state.get("process_mode", ""))),
                    "scheduler_mode": str(coherence_info.get("scheduler_mode", state.get("scheduler_mode", ""))),
                    "active_zone_name": str(coherence_info.get("active_zone_name", state.get("active_zone_name", ""))),
                    "active_basin_name": str(coherence_info.get("active_basin_name", state.get("active_basin_name", ""))),
                },
                metadata=dict(packet.metadata or {}),
                derived_state=dict(state),
            )
            results.append(packet_out)

        state["valid_ratio"] = float(valid_count / float(len(results) or 1))
        state["validated_candidate_count"] = int(len(results))
        state["valid_share_count"] = int(valid_count)
        if selected_by_index:
            if valid_count > 0:
                valid_coherence = [
                    float(result.system_payload.get("coherence", 0.0))
                    for result in results
                    if bool(result.system_payload.get("valid", False))
                ]
                if valid_coherence:
                    state["coherence_peak"] = float(max(valid_coherence))

        NonceMath._persist_lane_state(lane_state, state)
        return results
