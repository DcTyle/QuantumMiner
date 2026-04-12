from __future__ import annotations

from typing import Any, Dict
import hashlib
import json
import math
import os
import subprocess
import threading
import time

from VHW.system_utils import (
    ada_effective_constants,
    ada_latency_kernel,
    ada_phase_update,
    device_snapshot,
)


_LOCK = threading.RLock()
_TRACE_VRAM_CACHE: Dict[str, Dict[str, Any]] = {}
_VULKAN_ACTUATION_CACHE: Dict[str, Any] = {}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _flag_enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "none"):
        return bool(default)
    if text in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    if text in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    return bool(value)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vulkan_calibration_paths() -> Dict[str, str]:
    project_dir = os.path.join(
        _repo_root(),
        "ResearchConfinement",
        "Calibrationkernals",
        "vulkan_gpu_calibration_kernels",
    )
    shader_dir = os.path.join(project_dir, "shaders", "bin")
    build_dir = os.path.join(project_dir, "build_runtime")
    exe_path = os.path.join(build_dir, "Release", "vulkan_gpu_calibration.exe")
    return {
        "project_dir": project_dir,
        "shader_dir": shader_dir,
        "build_dir": build_dir,
        "exe_path": exe_path,
    }


def _run_command(command: list[str], workdir: str, timeout_s: float = 180.0) -> str:
    result = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1.0, float(timeout_s)),
        check=True,
    )
    return str(result.stdout or "")


def _ensure_vulkan_calibration_binary(payload: Dict[str, Any]) -> str:
    paths = _vulkan_calibration_paths()
    project_dir = str(paths["project_dir"])
    shader_dir = str(paths["shader_dir"])
    build_dir = str(paths["build_dir"])
    exe_path = str(paths["exe_path"])
    cache_key = "vulkan_calibration_exe"
    force_rebuild = _flag_enabled(payload.get("force_vulkan_rebuild", False), False)
    with _LOCK:
        cached_path = str(_VULKAN_ACTUATION_CACHE.get(cache_key, "") or "")
    if cached_path and os.path.isfile(cached_path) and not force_rebuild:
        return cached_path

    os.makedirs(shader_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)

    shader_jobs = [
        (
            os.path.join(project_dir, "shaders", "gpu_calibration.comp"),
            os.path.join(shader_dir, "gpu_calibration.spv"),
        ),
        (
            os.path.join(project_dir, "shaders", "trajectory_update.comp"),
            os.path.join(shader_dir, "trajectory_update.spv"),
        ),
    ]
    for source_path, output_path in shader_jobs:
        if force_rebuild or not os.path.isfile(output_path) or os.path.getmtime(output_path) < os.path.getmtime(source_path):
            _run_command(
                ["glslangValidator", "-V", source_path, "-o", output_path],
                workdir=project_dir,
                timeout_s=120.0,
            )

    need_configure = force_rebuild or not os.path.isfile(os.path.join(build_dir, "CMakeCache.txt"))
    if need_configure:
        _run_command(
            [
                "cmake",
                "-S",
                project_dir,
                "-B",
                build_dir,
                "-G",
                "Visual Studio 17 2022",
            ],
            workdir=project_dir,
            timeout_s=240.0,
        )
    if force_rebuild or not os.path.isfile(exe_path):
        _run_command(
            [
                "cmake",
                "--build",
                build_dir,
                "--config",
                "Release",
            ],
            workdir=project_dir,
            timeout_s=240.0,
        )
    if not os.path.isfile(exe_path):
        raise RuntimeError("vulkan calibration executable missing after build")

    with _LOCK:
        _VULKAN_ACTUATION_CACHE[cache_key] = exe_path
    return exe_path


def _vulkan_calibration_actuation(meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(meta.get("payload", {}) or {})
    exe_path = _ensure_vulkan_calibration_binary(payload)
    shader_dir = _vulkan_calibration_paths()["shader_dir"]
    phase_turns = _clamp01(meta.get("phase_turns", 0.0))
    phase_rad = float(phase_turns) * (2.0 * math.pi)
    sample_period_s = max(0.0005, _safe_float(meta.get("sample_period_s", 0.005), 0.005))
    element_count = max(256, int(payload.get("vulkan_element_count", 8192) or 8192))
    iterations = max(4, int(payload.get("vulkan_iterations", 24) or 24))
    amplitude = _clamp01(payload.get("vulkan_amplitude", 0.18))
    voltage = _clamp01(payload.get("vulkan_voltage", 0.36))
    current = _clamp01(payload.get("vulkan_current", 0.36))
    frequency = _clamp01(payload.get("vulkan_frequency", 0.245))
    lattice_closure = _clamp01(payload.get("vulkan_lattice_closure", 0.88))
    phase_closure = _clamp01(payload.get("vulkan_phase_closure", 0.81))
    recurrence_alignment = _clamp01(payload.get("vulkan_recurrence_alignment", 0.76))
    conservation_alignment = _clamp01(payload.get("vulkan_conservation_alignment", 0.999))
    thermal_noise = _clamp01(payload.get("vulkan_thermal_noise", 0.08))
    field_noise = _clamp01(payload.get("vulkan_field_noise", 0.06))
    task_mode = max(0, min(2, int(payload.get("vulkan_task_mode", 0) or 0)))
    input_headroom = _clamp01(1.0 - max(frequency, amplitude, voltage, current, thermal_noise, field_noise))
    input_resonance, input_resonance_weights = _derived_temporal_mix({
        "frequency": frequency,
        "amplitude": amplitude,
        "voltage": voltage,
        "current": current,
        "recurrence_alignment": recurrence_alignment,
        "lattice_closure": lattice_closure,
    })
    input_temporal_overlap, input_temporal_overlap_weights = _derived_temporal_mix({
        "voltage": voltage,
        "current": current,
        "phase_closure": phase_closure,
        "lattice_closure": lattice_closure,
        "noise_inverse": 1.0 - _clamp01((thermal_noise + field_noise) * 0.5),
        "headroom": input_headroom,
    })
    input_flux_term, input_flux_term_weights = _derived_temporal_mix({
        "frequency": frequency,
        "voltage": voltage,
        "current": current,
        "phase_closure": phase_closure,
        "thermal_noise": thermal_noise,
        "field_noise": field_noise,
    })
    input_energy_hint, input_energy_hint_weights = _derived_temporal_mix({
        "amplitude": amplitude,
        "frequency": frequency,
        "current": current,
        "conservation_alignment": conservation_alignment,
        "headroom": input_headroom,
    })
    input_field_dynamics = _compute_axis_field_dynamics(
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        phase_turns=phase_turns,
        resonance=input_resonance,
        temporal_overlap=input_temporal_overlap,
        flux_term=input_flux_term,
        vector_x=voltage - current,
        vector_y=amplitude - thermal_noise,
        vector_z=frequency - field_noise,
        energy_hint=input_energy_hint,
    )
    input_temporal_constants = _derive_temporal_constant_set(
        field_dynamics=input_field_dynamics,
        phase_turns=phase_turns,
        gpu_util=frequency,
        mem_bw_util=amplitude,
        cpu_util=current,
        global_util=voltage,
        headroom=input_headroom,
        latency_load=_clamp01(sample_period_s * 200.0),
        resonance=input_resonance,
        temporal_overlap=input_temporal_overlap,
        flux_term=input_flux_term,
        persistence=lattice_closure,
        leakage=_clamp01((thermal_noise + field_noise) * 0.5),
        actuation_gain=amplitude,
        pulse_signal=amplitude,
        coherence=recurrence_alignment,
        entropy=phase_closure,
        valid_ratio=conservation_alignment,
        observer_feedback=lattice_closure,
        noise_gate=_clamp01((thermal_noise + field_noise) * 0.5),
        phase_delta_turns=input_field_dynamics.get("phase_injection_delta_turns", 0.0),
    )

    command = [
        exe_path,
        "--shader-dir",
        shader_dir,
        "--task-mode",
        str(task_mode),
        "--element-count",
        str(element_count),
        "--iterations",
        str(iterations),
        "--dt",
        str(max(0.0001, sample_period_s * 0.5)),
        "--phase-scale",
        str(max(0.25, min(4.0, sample_period_s * 240.0))),
        "--frequency",
        str(frequency),
        "--amplitude",
        str(amplitude),
        "--voltage",
        str(voltage),
        "--current",
        str(current),
        "--phase-rad",
        str(phase_rad),
        "--sent-signal",
        str(float(input_temporal_constants.get("sent_signal", 0.0))),
        "--measured-signal",
        str(float(input_temporal_constants.get("measured_signal", 0.0))),
        "--integrated-feedback",
        str(float(input_temporal_constants.get("integrated_feedback", 0.0))),
        "--derivative-signal",
        str(float(input_temporal_constants.get("derivative_signal", 0.0))),
        "--lattice-closure",
        str(lattice_closure),
        "--phase-closure",
        str(phase_closure),
        "--recurrence-alignment",
        str(recurrence_alignment),
        "--conservation-alignment",
        str(conservation_alignment),
        "--thermal-noise",
        str(thermal_noise),
        "--field-noise",
        str(field_noise),
    ]
    raw_output = _run_command(command, workdir=os.path.dirname(exe_path), timeout_s=180.0)
    json_line = ""
    for line in reversed(str(raw_output or "").splitlines()):
        text = str(line or "").strip()
        if text.startswith("{") and text.endswith("}"):
            json_line = text
            break
    if not json_line:
        raise RuntimeError("vulkan calibration output missing json payload")
    result = json.loads(json_line)
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error", "vulkan calibration failed")))
    calibration_dynamics = _compute_axis_field_dynamics(
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        phase_turns=phase_turns,
        resonance=_clamp01(0.5 * _safe_float(result.get("mean_actuation_gain", 0.0), 0.0) + 0.5 * recurrence_alignment),
        temporal_overlap=_clamp01(result.get("mean_persistence", 0.0)),
        flux_term=_clamp01(_safe_float(result.get("mean_leakage", 0.0), 0.0) + 0.5 * abs(_safe_float(result.get("mean_pulse_signal", 0.0), 0.0))),
        vector_x=_safe_float(result.get("mean_actuation_gain", 0.0), 0.0) - _safe_float(result.get("mean_leakage", 0.0), 0.0),
        vector_y=_safe_float(result.get("mean_persistence", 0.0), 0.0) - _clamp01(_safe_float(result.get("mean_position_radius", 0.0), 0.0)),
        vector_z=_safe_float(result.get("mean_pulse_signal", 0.0), 0.0),
        energy_hint=_clamp01((0.50 * amplitude) + (0.25 * frequency) + (0.25 * _clamp01(_safe_float(result.get("peak_velocity", 0.0), 0.0)))),
    )
    for key, value in calibration_dynamics.items():
        result[key] = float(result.get(key, value)) if isinstance(value, float) else value
    calibration_phase_ring_state = _compute_phase_ring_state(
        phase_turns=phase_turns,
        field_dynamics=calibration_dynamics,
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        resonance=_clamp01(0.5 * _safe_float(result.get("mean_actuation_gain", 0.0), 0.0) + 0.5 * recurrence_alignment),
        temporal_overlap=_clamp01(result.get("mean_persistence", 0.0)),
        flux_term=_clamp01(_safe_float(result.get("mean_leakage", 0.0), 0.0) + 0.5 * abs(_safe_float(result.get("mean_pulse_signal", 0.0), 0.0))),
    )
    calibration_dynamics.update(calibration_phase_ring_state)
    result.update(calibration_phase_ring_state)
    calibration_gradient_state = _compose_frequency_gradient_9d(
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        phase_turns=calibration_dynamics.get("phase_injection_turns", phase_turns),
        field_dynamics=calibration_dynamics,
    )
    calibration_dynamics.update(calibration_gradient_state)
    result.update(calibration_gradient_state)
    calibration_temporal_constants = _derive_temporal_constant_set(
        field_dynamics=calibration_dynamics,
        phase_turns=calibration_dynamics.get("phase_injection_turns", phase_turns),
        gpu_util=frequency,
        mem_bw_util=amplitude,
        cpu_util=current,
        global_util=voltage,
        headroom=_clamp01(1.0 - _safe_float(result.get("mean_actuation_gain", 0.0), 0.0)),
        latency_load=_clamp01(_safe_float(result.get("dispatch_elapsed_ms", 0.0), 0.0) / 16.0),
        resonance=_clamp01(0.5 * _safe_float(result.get("mean_actuation_gain", 0.0), 0.0) + 0.5 * recurrence_alignment),
        temporal_overlap=_clamp01(result.get("mean_persistence", 0.0)),
        flux_term=_clamp01(_safe_float(result.get("mean_leakage", 0.0), 0.0) + 0.5 * abs(_safe_float(result.get("mean_pulse_signal", 0.0), 0.0))),
        persistence=_clamp01(_safe_float(result.get("mean_persistence", 0.0), 0.0)),
        leakage=_clamp01(_safe_float(result.get("mean_leakage", 0.0), 0.0)),
        actuation_gain=_safe_float(result.get("mean_actuation_gain", 0.0), 0.0),
        pulse_signal=_safe_float(result.get("mean_pulse_signal", 0.0), 0.0),
        coherence=recurrence_alignment,
        entropy=phase_closure,
        valid_ratio=conservation_alignment,
        observer_feedback=lattice_closure,
        noise_gate=_clamp01(0.5 * thermal_noise + 0.5 * field_noise),
        phase_delta_turns=calibration_dynamics.get("phase_injection_delta_turns", 0.0),
    )
    calibration_feedback_gate = float(calibration_temporal_constants.get("feedback_gate", 0.0))
    calibration_coherence_gate = float(calibration_temporal_constants.get("coherence_gate", 0.0))
    calibration_observer_damping = float(calibration_temporal_constants.get("observer_damping", 0.0))
    calibration_transport_prediction = {
        "phase_transport_term": float(calibration_temporal_constants.get("phase_transport_term", 0.0)),
        "flux_transport_term": float(calibration_temporal_constants.get("flux_transport_term", 0.0)),
        "observer_damping": float(calibration_observer_damping),
        "transport_coherence": float(calibration_temporal_constants.get("transport_coherence", 0.0)),
        "transport_damping_gate": float(calibration_temporal_constants.get("transport_damping_gate", 0.0)),
        "reverse_transport_gate": float(calibration_temporal_constants.get("reverse_transport_gate", 0.0)),
        "calibration_temporal_constant_weights": dict(calibration_temporal_constants.get("weights", {})),
    }
    calibration_trajectory = _compose_photonic_trajectory_9d(
        phase_turns=calibration_dynamics.get("phase_injection_turns", phase_turns),
        field_dynamics=calibration_dynamics,
        flux_term=calibration_transport_prediction["flux_transport_term"],
        coherence_gate=calibration_coherence_gate,
        feedback_gate=calibration_feedback_gate,
    )
    predicted_calibration_trajectory = _predict_photonic_trajectory_9d(
        current_trajectory_9d=list(calibration_trajectory.get("trajectory_9d", []) or []),
        previous_predicted_trajectory_9d=None,
        field_dynamics=calibration_dynamics,
        transport_prediction=calibration_transport_prediction,
        headroom=_clamp01(1.0 - _safe_float(result.get("mean_actuation_gain", 0.0), 0.0)),
        latency_load=_clamp01(_safe_float(result.get("dispatch_elapsed_ms", 0.0), 0.0) / 16.0),
        noise_gate=_clamp01(0.5 * thermal_noise + 0.5 * field_noise),
    )
    result["trajectory_9d"] = list(calibration_trajectory.get("trajectory_9d", []) or [])
    result["predicted_trajectory_9d"] = list(predicted_calibration_trajectory.get("predicted_trajectory_9d", []) or [])
    result["trajectory_velocity_9d"] = list(predicted_calibration_trajectory.get("trajectory_velocity_9d", []) or [])
    result["trajectory_spectral_id"] = str(calibration_trajectory.get("trajectory_spectral_id", ""))
    result["predicted_trajectory_spectral_id"] = str(predicted_calibration_trajectory.get("trajectory_spectral_id", ""))
    result["trajectory_conservation_alignment"] = float(predicted_calibration_trajectory.get("trajectory_conservation_alignment", 0.0))
    result["trajectory_prediction_alignment"] = float(predicted_calibration_trajectory.get("trajectory_prediction_alignment", 0.0))
    result["trajectory_expansion_term"] = float(predicted_calibration_trajectory.get("trajectory_expansion_term", 0.0))
    result["trajectory_sequence_density"] = float(predicted_calibration_trajectory.get("trajectory_sequence_density", 0.0))
    result["trajectory_noise_feedback_norm"] = float(predicted_calibration_trajectory.get("trajectory_noise_feedback_norm", 0.0))
    load_hint = float(calibration_temporal_constants.get("load_hint", 0.0))
    return {
        "mode": "vulkan_calibration",
        "tag": str(result.get("device_name", "vulkan_calibration")),
        "load_hint": float(load_hint),
        "dispatch_elapsed_ms": float(result.get("dispatch_elapsed_ms", 0.0)),
        "axis_scale_x": float(result.get("axis_scale_x", 0.0)),
        "axis_scale_y": float(result.get("axis_scale_y", 0.0)),
        "axis_scale_z": float(result.get("axis_scale_z", 0.0)),
        "vector_energy": float(result.get("vector_energy", 0.0)),
        "temporal_coupling_moment": float(result.get("temporal_coupling_moment", 0.0)),
        "inertial_mass_proxy": float(result.get("inertial_mass_proxy", 0.0)),
        "spin_momentum_score": float(result.get("spin_momentum_score", 0.0)),
        "gpu_pulse_phase_effect": float(result.get("gpu_pulse_phase_effect", 0.0)),
        "phase_injection_delta_turns": float(result.get("phase_injection_delta_turns", 0.0)),
        "phase_injection_turns": float(result.get("phase_injection_turns", phase_turns)),
        "phase_ring_closure": float(result.get("phase_ring_closure", 0.0)),
        "phase_ring_density": float(result.get("phase_ring_density", 0.0)),
        "phase_ring_strength": float(result.get("phase_ring_strength", 0.0)),
        "zero_point_crossover_gate": float(result.get("zero_point_crossover_gate", 0.0)),
        "shared_vector_collapse_gate": float(result.get("shared_vector_collapse_gate", 0.0)),
        "shared_vector_phase_lock": float(result.get("shared_vector_phase_lock", 0.0)),
        "inertial_basin_strength": float(result.get("inertial_basin_strength", 0.0)),
        "wavelength_norm": float(result.get("wavelength_norm", 0.0)),
        "orientation_alignment": float(result.get("orientation_alignment", 0.0)),
        "rotational_velocity_norm": float(result.get("rotational_velocity_norm", 0.0)),
        "relative_temporal_position": float(result.get("relative_temporal_position", 0.0)),
        "zero_point_line_distance": float(result.get("zero_point_line_distance", 0.0)),
        "field_interference_norm": float(result.get("field_interference_norm", 0.0)),
        "resonant_interception_inertia": float(result.get("resonant_interception_inertia", 0.0)),
        "temporal_relativity_norm": float(result.get("temporal_relativity_norm", 0.0)),
        "temporal_relativity_vector": [float(_clamp01(v)) for v in list(result.get("temporal_relativity_vector", []))[:4]],
        "phase_ring_zone_id": int(result.get("phase_ring_zone_id", 0)),
        "frequency_gradient_9d": [float(_clamp01(v)) for v in list(result.get("frequency_gradient_9d", []))[:9]],
        "field_gradient_9d": [float(_clamp01(v)) for v in list(result.get("field_gradient_9d", result.get("frequency_gradient_9d", [])))[:9]],
        "gradient_spectral_id": str(result.get("gradient_spectral_id", "")),
        "trajectory_spectral_id": str(result.get("trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": float(result.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_expansion_term": float(result.get("trajectory_expansion_term", 0.0)),
        "trajectory_9d": [float(_clamp01(v)) for v in list(result.get("trajectory_9d", []))[:9]],
        "predicted_trajectory_9d": [float(_clamp01(v)) for v in list(result.get("predicted_trajectory_9d", []))[:9]],
        "calibration_temporal_constant_weights": {
            "input_resonance": dict(input_resonance_weights),
            "input_temporal_overlap": dict(input_temporal_overlap_weights),
            "input_flux_term": dict(input_flux_term_weights),
            "input_energy_hint": dict(input_energy_hint_weights),
            **{str(key): dict(value) for key, value in dict(calibration_temporal_constants.get("weights", {})).items()},
        },
        "kernel_summary": result,
    }


def _runtime_packet_actuation(meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(meta.get("payload", {}) or {})
    packet = payload.get("packet")
    raw_payload = {}
    packet_network = ""
    packet_type = ""
    try:
        raw_payload = dict(getattr(packet, "raw_payload", {}) or {})
    except Exception:
        raw_payload = {}
    try:
        packet_network = str(getattr(getattr(packet, "network", ""), "name", getattr(packet, "network", "")) or "")
    except Exception:
        packet_network = ""
    try:
        packet_type = str(getattr(getattr(packet, "packet_type", ""), "name", getattr(packet, "packet_type", "")) or "")
    except Exception:
        packet_type = ""
    lane_id = str(payload.get("lane_id", "lane") or "lane")
    tick = int(payload.get("tick", 0) or 0)
    phase_turns = _wrap_turns(meta.get("phase_turns", 0.0))
    sample_period_s = max(0.00025, _safe_float(meta.get("sample_period_s", 0.004), 0.004))
    header_hex = str(raw_payload.get("header_hex", raw_payload.get("header", "")) or "")
    target_hex = str(
        raw_payload.get(
            "share_target",
            raw_payload.get("active_target", raw_payload.get("target", "")),
        )
        or ""
    )
    noise_seed = "%s|%s|%s|%s|%s|%08x" % (
        lane_id,
        packet_network,
        packet_type,
        header_hex[:64],
        target_hex[:64],
        int(tick) & 0xFFFFFFFF,
    )
    digest = hashlib.sha256(noise_seed.encode("ascii", errors="ignore")).digest()

    global_util = _clamp01(payload.get("global_util", 0.0))
    gpu_util = _clamp01(payload.get("gpu_util", global_util))
    mem_bw_util = _clamp01(payload.get("mem_bw_util", global_util))
    cpu_util = _clamp01(payload.get("cpu_util", global_util))
    headroom = _clamp01(1.0 - max(global_util, gpu_util, mem_bw_util, cpu_util))
    frequency = _clamp01(payload.get("runtime_frequency", payload.get("vulkan_frequency", 0.245)))
    amplitude = _clamp01(payload.get("runtime_amplitude", payload.get("vulkan_amplitude", 0.18)))
    voltage = _clamp01(payload.get("runtime_voltage", payload.get("vulkan_voltage", 0.36)))
    current = _clamp01(payload.get("runtime_current", payload.get("vulkan_current", 0.36)))
    phase_seed_delta, phase_seed_weights = _derived_temporal_mix({
        "digest_phase": ((float(digest[0]) / 255.0) - 0.5),
        "gpu_mem_delta": _clamp_signed(gpu_util - mem_bw_util),
        "headroom_delta": _clamp_signed(headroom - 0.5),
        "cpu_global_delta": _clamp_signed(cpu_util - global_util),
        "noise_inverse": _clamp_signed((float(digest[1]) / 255.0) - 0.5),
    }, signed=True, limit=0.25)
    phase_injection_delta_turns = _clamp_signed(phase_seed_delta, 0.25)
    phase_injection_turns = _wrap_turns(float(phase_turns) + float(phase_injection_delta_turns))

    resonance, resonance_weights = _derived_temporal_mix({
        "gpu_util": gpu_util,
        "memory": mem_bw_util,
        "amplitude": amplitude,
        "frequency": frequency,
        "voltage": voltage,
        "current": current,
        "digest": float(digest[1]) / 255.0,
    })
    temporal_overlap, temporal_overlap_weights = _derived_temporal_mix({
        "voltage": voltage,
        "current": current,
        "gpu_mem_alignment": 1.0 - abs(gpu_util - mem_bw_util),
        "digest": float(digest[2]) / 255.0,
        "headroom": headroom,
        "phase_origin": _clamp01(1.0 - abs(_phase_delta_turns(phase_injection_turns, 0.0)) * 2.0),
    })
    flux_term, flux_term_weights = _derived_temporal_mix({
        "frequency": frequency,
        "voltage": voltage,
        "current": current,
        "amplitude": amplitude,
        "digest": float(digest[3]) / 255.0,
        "headroom": headroom,
    })
    energy_hint, energy_hint_weights = _derived_temporal_mix({
        "gpu_util": gpu_util,
        "memory": mem_bw_util,
        "amplitude": amplitude,
        "current": current,
        "digest": float(digest[7]) / 255.0,
        "headroom": headroom,
    })
    field_dynamics = _compute_axis_field_dynamics(
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        phase_turns=phase_injection_turns,
        resonance=resonance,
        temporal_overlap=temporal_overlap,
        flux_term=flux_term,
        vector_x=(gpu_util - global_util) + (((float(digest[4]) / 255.0) - 0.5) * 0.35),
        vector_y=(mem_bw_util - cpu_util) + (((float(digest[5]) / 255.0) - 0.5) * 0.35),
        vector_z=(resonance - temporal_overlap) + (((float(digest[6]) / 255.0) - 0.5) * 0.35),
        energy_hint=energy_hint,
    )
    runtime_temporal_constants = _derive_temporal_constant_set(
        field_dynamics=field_dynamics,
        phase_turns=phase_injection_turns,
        gpu_util=gpu_util,
        mem_bw_util=mem_bw_util,
        cpu_util=cpu_util,
        global_util=global_util,
        headroom=headroom,
        latency_load=_clamp01(1.0 - headroom),
        resonance=resonance,
        temporal_overlap=temporal_overlap,
        flux_term=flux_term,
        persistence=temporal_overlap,
        leakage=_clamp01(1.0 - headroom),
        actuation_gain=gpu_util,
        pulse_signal=amplitude,
        coherence=resonance,
        entropy=temporal_overlap,
        valid_ratio=1.0 - _clamp01(abs(gpu_util - mem_bw_util)),
        observer_feedback=_clamp01(float(digest[2]) / 255.0),
        noise_gate=_clamp01(abs(float(digest[0]) / 255.0 - float(digest[3]) / 255.0)),
        phase_delta_turns=phase_injection_delta_turns,
    )
    phase_ring_state = _compute_phase_ring_state(
        phase_turns=phase_injection_turns,
        field_dynamics=field_dynamics,
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        resonance=resonance,
        temporal_overlap=temporal_overlap,
        flux_term=flux_term,
    )
    field_dynamics.update(phase_ring_state)
    gradient_state = _compose_frequency_gradient_9d(
        frequency_norm=frequency,
        amplitude_norm=amplitude,
        phase_turns=phase_injection_turns,
        field_dynamics=field_dynamics,
    )
    field_dynamics.update(gradient_state)
    transport_prediction = {
        "phase_transport_term": float(runtime_temporal_constants.get("phase_transport_term", 0.0)),
        "flux_transport_term": float(runtime_temporal_constants.get("flux_transport_term", 0.0)),
        "observer_damping": float(runtime_temporal_constants.get("observer_damping", 0.0)),
        "transport_coherence": float(runtime_temporal_constants.get("transport_coherence", 0.0)),
        "transport_damping_gate": float(runtime_temporal_constants.get("transport_damping_gate", 0.0)),
        "reverse_transport_gate": float(runtime_temporal_constants.get("reverse_transport_gate", 0.0)),
        "reverse_delta_theta_hat": float(runtime_temporal_constants.get("phase_transport_term", 0.0)) * 0.75,
        "transported_phase_turns": float(phase_injection_turns),
        "runtime_temporal_constant_weights": dict(runtime_temporal_constants.get("weights", {})),
    }
    trajectory_state = _compose_photonic_trajectory_9d(
        phase_turns=phase_injection_turns,
        field_dynamics=field_dynamics,
        flux_term=flux_term,
        coherence_gate=float(runtime_temporal_constants.get("trajectory_coherence_gate", 0.0)),
        feedback_gate=float(runtime_temporal_constants.get("trajectory_feedback_gate", 0.0)),
    )
    predicted_trajectory = _predict_photonic_trajectory_9d(
        current_trajectory_9d=list(trajectory_state.get("trajectory_9d", []) or []),
        previous_predicted_trajectory_9d=list(payload.get("trace_predicted_trajectory_9d", []) or [])[:9],
        field_dynamics=field_dynamics,
        transport_prediction=transport_prediction,
        headroom=headroom,
        latency_load=_clamp01(1.0 - headroom),
        noise_gate=_clamp01(abs(phase_injection_delta_turns) * 2.0),
    )
    dispatch_scale, dispatch_scale_weights = _derived_temporal_mix({
        "headroom": headroom,
        "phase_ring": field_dynamics.get("phase_ring_strength", 0.0),
        "temporal_relativity": field_dynamics.get("temporal_relativity_norm", 0.0),
        "transport_coherence": runtime_temporal_constants.get("transport_coherence", 0.0),
        "shared_phase_lock": field_dynamics.get("shared_vector_phase_lock", 0.0),
    })
    dispatch_elapsed_ms = max(0.0, sample_period_s * 1000.0 * float(dispatch_scale))
    load_hint = float(runtime_temporal_constants.get("load_hint", 0.0))
    return {
        "mode": "runtime_packet_actuation",
        "tag": "%s:%s" % (packet_network or "packet", lane_id),
        "load_hint": float(load_hint),
        "dispatch_elapsed_ms": float(dispatch_elapsed_ms),
        "axis_scale_x": float(field_dynamics.get("axis_scale_x", 0.0)),
        "axis_scale_y": float(field_dynamics.get("axis_scale_y", 0.0)),
        "axis_scale_z": float(field_dynamics.get("axis_scale_z", 0.0)),
        "vector_energy": float(field_dynamics.get("vector_energy", 0.0)),
        "temporal_coupling_moment": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
        "inertial_mass_proxy": float(field_dynamics.get("inertial_mass_proxy", 0.0)),
        "spin_momentum_score": float(field_dynamics.get("spin_momentum_score", 0.0)),
        "gpu_pulse_phase_effect": float(phase_injection_delta_turns),
        "phase_injection_delta_turns": float(phase_injection_delta_turns),
        "phase_injection_turns": float(phase_injection_turns),
        "phase_ring_closure": float(field_dynamics.get("phase_ring_closure", 0.0)),
        "phase_ring_density": float(field_dynamics.get("phase_ring_density", 0.0)),
        "phase_ring_strength": float(field_dynamics.get("phase_ring_strength", 0.0)),
        "zero_point_crossover_gate": float(field_dynamics.get("zero_point_crossover_gate", 0.0)),
        "shared_vector_collapse_gate": float(field_dynamics.get("shared_vector_collapse_gate", 0.0)),
        "shared_vector_phase_lock": float(field_dynamics.get("shared_vector_phase_lock", 0.0)),
        "inertial_basin_strength": float(field_dynamics.get("inertial_basin_strength", 0.0)),
        "wavelength_norm": float(field_dynamics.get("wavelength_norm", 0.0)),
        "orientation_alignment": float(field_dynamics.get("orientation_alignment", 0.0)),
        "rotational_velocity_norm": float(field_dynamics.get("rotational_velocity_norm", 0.0)),
        "relative_temporal_position": float(field_dynamics.get("relative_temporal_position", 0.0)),
        "zero_point_line_distance": float(field_dynamics.get("zero_point_line_distance", 0.0)),
        "field_interference_norm": float(field_dynamics.get("field_interference_norm", 0.0)),
        "resonant_interception_inertia": float(field_dynamics.get("resonant_interception_inertia", 0.0)),
        "temporal_relativity_norm": float(field_dynamics.get("temporal_relativity_norm", 0.0)),
        "temporal_relativity_vector": [float(_clamp01(v)) for v in list(field_dynamics.get("temporal_relativity_vector", []))[:4]],
        "phase_ring_zone_id": int(field_dynamics.get("phase_ring_zone_id", 0) or 0),
        "frequency_gradient_9d": [float(_clamp01(v)) for v in list(field_dynamics.get("frequency_gradient_9d", []))[:9]],
        "field_gradient_9d": [float(_clamp01(v)) for v in list(field_dynamics.get("field_gradient_9d", []))[:9]],
        "gradient_spectral_id": str(field_dynamics.get("gradient_spectral_id", "")),
        "trajectory_9d": [float(_clamp01(v)) for v in list(trajectory_state.get("trajectory_9d", []))[:9]],
        "predicted_trajectory_9d": [float(_clamp01(v)) for v in list(predicted_trajectory.get("predicted_trajectory_9d", []))[:9]],
        "trajectory_velocity_9d": [float(_clamp01(v)) for v in list(predicted_trajectory.get("trajectory_velocity_9d", []))[:9]],
        "trajectory_spectral_id": str(trajectory_state.get("trajectory_spectral_id", "")),
        "predicted_trajectory_spectral_id": str(predicted_trajectory.get("trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": float(predicted_trajectory.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_prediction_alignment": float(predicted_trajectory.get("trajectory_prediction_alignment", 0.0)),
        "trajectory_expansion_term": float(predicted_trajectory.get("trajectory_expansion_term", 0.0)),
        "trajectory_sequence_density": float(predicted_trajectory.get("trajectory_sequence_density", 0.0)),
        "runtime_temporal_constant_weights": {
            "phase_seed": dict(phase_seed_weights),
            "resonance": dict(resonance_weights),
            "temporal_overlap": dict(temporal_overlap_weights),
            "flux_term": dict(flux_term_weights),
            "energy_hint": dict(energy_hint_weights),
            **{str(key): dict(value) for key, value in dict(runtime_temporal_constants.get("weights", {})).items()},
            "dispatch_scale": dict(dispatch_scale_weights),
        },
    }


def _stable_seed(text: str) -> int:
    seed = 2166136261
    for ch in str(text or "lane").encode("ascii", errors="ignore"):
        seed ^= ch
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _lerp(prev: Any, cur: Any, alpha: float) -> float:
    p = _safe_float(prev, 0.0)
    c = _safe_float(cur, 0.0)
    a = _clamp01(alpha)
    return ((1.0 - a) * p) + (a * c)


def _vector_list(value: Any, size: int) -> list[float]:
    out = [0.0] * int(size)
    if isinstance(value, (list, tuple)):
        for idx in range(min(len(value), int(size))):
            out[idx] = _safe_float(value[idx], 0.0)
    return out


def _vector_energy(values: list[float]) -> float:
    if not values:
        return 0.0
    total = sum(float(v) * float(v) for v in values)
    return _clamp01(math.sqrt(max(total, 0.0)) / math.sqrt(float(len(values))))


def _clamp_signed(value: Any, limit: float = 1.0) -> float:
    bound = abs(float(limit))
    return max(-bound, min(bound, _safe_float(value, 0.0)))


def _normalize_dynamic_weights(components: Dict[str, Any]) -> Dict[str, float]:
    payload = dict(components or {})
    cleaned: Dict[str, float] = {}
    for key, value in payload.items():
        cleaned[str(key)] = max(1.0e-9, abs(_safe_float(value, 0.0)))
    if not cleaned:
        return {}
    total = sum(cleaned.values())
    if total <= 1.0e-9:
        uniform = 1.0 / float(len(cleaned))
        return {key: float(uniform) for key in cleaned}
    return {key: float(value / total) for key, value in cleaned.items()}


def _derived_temporal_mix(
    components: Dict[str, Any],
    signed: bool = False,
    limit: float = 1.0,
) -> tuple[float, Dict[str, float]]:
    payload = dict(components or {})
    weights = _normalize_dynamic_weights(payload)
    total = 0.0
    for key, weight in weights.items():
        total += float(weight) * _safe_float(payload.get(key, 0.0), 0.0)
    if signed:
        return float(_clamp_signed(total, limit)), weights
    return float(_clamp01(total)), weights


def _derive_temporal_constant_set(
    field_dynamics: Dict[str, Any],
    phase_turns: Any = 0.0,
    gpu_util: Any = 0.0,
    mem_bw_util: Any = 0.0,
    cpu_util: Any = 0.0,
    global_util: Any = 0.0,
    headroom: Any = 0.0,
    latency_load: Any = 0.0,
    resonance: Any = 0.0,
    temporal_overlap: Any = 0.0,
    flux_term: Any = 0.0,
    persistence: Any = 0.0,
    leakage: Any = 0.0,
    actuation_gain: Any = 0.0,
    pulse_signal: Any = 0.0,
    coherence: Any = 0.0,
    entropy: Any = 0.0,
    valid_ratio: Any = 0.0,
    observer_feedback: Any = 0.0,
    noise_gate: Any = 0.0,
    phase_delta_turns: Any = 0.0,
) -> Dict[str, Any]:
    dynamics = dict(field_dynamics or {})
    phase = _wrap_turns(
        phase_turns if phase_turns is not None else dynamics.get("phase_injection_turns", dynamics.get("phase_turns", 0.0))
    )
    gpu = _clamp01(gpu_util)
    mem = _clamp01(mem_bw_util)
    cpu = _clamp01(cpu_util)
    glob = _clamp01(global_util)
    head = _clamp01(headroom)
    latency = _clamp01(latency_load)
    resonance_gate = _clamp01(resonance)
    overlap = _clamp01(temporal_overlap)
    flux = _clamp01(abs(_safe_float(flux_term, 0.0)))
    persistence_gate = _clamp01(persistence)
    leakage_gate = _clamp01(leakage)
    actuation = _clamp01(abs(_safe_float(actuation_gain, 0.0)))
    pulse = _clamp01(abs(_safe_float(pulse_signal, 0.0)))
    coherence_gate = _clamp01(coherence)
    entropy_gate = _clamp01(entropy)
    valid_gate = _clamp01(valid_ratio)
    observer = _clamp01(observer_feedback)
    noise = _clamp01(noise_gate)
    phase_delta = _clamp01(abs(_safe_float(phase_delta_turns, 0.0)) * 2.0)
    pressure = _clamp01(max(gpu, mem, cpu, glob))
    phase_center = _clamp01(1.0 - abs(_phase_delta_turns(phase, 0.5)) * 2.0)
    phase_origin = _clamp01(1.0 - abs(_phase_delta_turns(phase, 0.0)) * 2.0)

    axis_resonance = _clamp01(dynamics.get("axis_resonance", 0.0))
    vector_energy = _clamp01(dynamics.get("vector_energy", 0.0))
    temporal_coupling = _clamp01(dynamics.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(dynamics.get("inertial_mass_proxy", 0.0))
    relativistic = _clamp01(dynamics.get("relativistic_correlation", 0.0))
    spin = _clamp01(dynamics.get("spin_momentum_score", 0.0))
    phase_ring_closure = _clamp01(dynamics.get("phase_ring_closure", 0.0))
    phase_ring_density = _clamp01(dynamics.get("phase_ring_density", 0.0))
    phase_ring_strength = _clamp01(dynamics.get("phase_ring_strength", 0.0))
    shared_phase_lock = _clamp01(dynamics.get("shared_vector_phase_lock", 0.0))
    zero_point = _clamp01(dynamics.get("zero_point_crossover_gate", dynamics.get("zero_point_crossover_norm", 0.0)))
    temporal_relativity = _clamp01(dynamics.get("temporal_relativity_norm", 0.0))
    interception = _clamp01(
        dynamics.get("resonant_interception_inertia", dynamics.get("intercept_inertia_norm", 0.0))
    )
    field_interference = _clamp01(dynamics.get("field_interference_norm", 0.0))
    trajectory_conservation = _clamp01(dynamics.get("trajectory_conservation_alignment", 0.0))
    trajectory_prediction = _clamp01(dynamics.get("trajectory_prediction_alignment", 0.0))
    trajectory_expansion = _clamp01(dynamics.get("trajectory_expansion_term", 0.0))
    resonance_detune = _clamp01(abs(resonance_gate - overlap))

    sent_signal, sent_signal_weights = _derived_temporal_mix({
        "phase_center": phase_center,
        "vector_energy": vector_energy,
        "phase_ring": phase_ring_strength,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity,
        "resonance": resonance_gate,
    })
    measured_signal, measured_signal_weights = _derived_temporal_mix({
        "pulse_signal": pulse,
        "persistence": persistence_gate,
        "actuation": actuation,
        "zero_point": zero_point,
        "interference_inverse": 1.0 - field_interference,
        "phase_origin": phase_origin,
    })
    integrated_feedback, integrated_feedback_weights = _derived_temporal_mix({
        "actuation": actuation,
        "persistence": persistence_gate,
        "shared_phase_lock": shared_phase_lock,
        "zero_point": zero_point,
        "temporal_relativity": temporal_relativity,
        "trajectory_conservation": trajectory_conservation,
    })
    derivative_signal, derivative_signal_weights = _derived_temporal_mix({
        "phase_delta": phase_delta,
        "flux": flux,
        "relativistic": relativistic,
        "spin": spin,
        "resonance_detune": resonance_detune,
        "interception": interception,
    })
    feedback_gate, feedback_gate_weights = _derived_temporal_mix({
        "actuation": actuation,
        "persistence": persistence_gate,
        "leakage_inverse": 1.0 - leakage_gate,
        "phase_ring": phase_ring_strength,
        "temporal_relativity": temporal_relativity,
        "trajectory_conservation": trajectory_conservation,
    })
    coherence_gate_out, coherence_gate_weights = _derived_temporal_mix({
        "resonance": resonance_gate,
        "persistence": persistence_gate,
        "axis_resonance": axis_resonance,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity,
        "phase_ring_closure": phase_ring_closure,
    })
    observer_damping, observer_damping_weights = _derived_temporal_mix({
        "noise": noise,
        "leakage": leakage_gate,
        "relativistic": relativistic,
        "interference": field_interference,
        "persistence_detune": 1.0 - persistence_gate,
        "trajectory_detune": 1.0 - trajectory_prediction,
    })
    phase_transport_term, phase_transport_term_weights = _derived_temporal_mix({
        "actuation_delta": _clamp_signed(actuation - leakage_gate),
        "resonance_delta": _clamp_signed(resonance_gate - overlap),
        "phase_delta": _clamp_signed(phase_delta_turns, 0.5),
        "shared_phase_lock": _clamp_signed(shared_phase_lock - 0.5),
        "temporal_relativity": _clamp_signed(temporal_relativity - 0.5),
        "phase_origin": _clamp_signed(phase_origin - 0.5),
    }, signed=True, limit=0.5)
    flux_transport_term, flux_transport_term_weights = _derived_temporal_mix({
        "persistence": persistence_gate,
        "pulse_signal": pulse,
        "leakage": leakage_gate,
        "vector_energy": vector_energy,
        "temporal_relativity": temporal_relativity,
        "interference_inverse": 1.0 - field_interference,
    })
    transport_coherence, transport_coherence_weights = _derived_temporal_mix({
        "coherence_gate": coherence_gate_out,
        "leakage_inverse": 1.0 - leakage_gate,
        "persistence": persistence_gate,
        "axis_resonance": axis_resonance,
        "shared_phase_lock": shared_phase_lock,
        "phase_ring": phase_ring_strength,
    })
    transport_damping_gate, transport_damping_gate_weights = _derived_temporal_mix({
        "observer_inverse": 1.0 - observer_damping,
        "shared_phase_lock": shared_phase_lock,
        "zero_point": zero_point,
        "temporal_relativity": temporal_relativity,
        "headroom": head,
    })
    reverse_transport_gate, reverse_transport_gate_weights = _derived_temporal_mix({
        "persistence": persistence_gate,
        "shared_phase_lock": shared_phase_lock,
        "zero_point": zero_point,
        "temporal_relativity": temporal_relativity,
        "interception_inverse": 1.0 - interception,
        "trajectory_prediction": trajectory_prediction,
    })
    trajectory_coherence_gate, trajectory_coherence_gate_weights = _derived_temporal_mix({
        "resonance": resonance_gate,
        "overlap": overlap,
        "shared_phase_lock": shared_phase_lock,
        "phase_ring": phase_ring_strength,
        "trajectory_conservation": trajectory_conservation,
        "temporal_relativity": temporal_relativity,
    })
    trajectory_feedback_gate, trajectory_feedback_gate_weights = _derived_temporal_mix({
        "overlap": overlap,
        "resonance": resonance_gate,
        "headroom": head,
        "phase_ring": phase_ring_strength,
        "zero_point": zero_point,
        "trajectory_prediction": trajectory_prediction,
    })
    load_hint, load_hint_weights = _derived_temporal_mix({
        "gpu": gpu,
        "memory": mem,
        "resonance": resonance_gate,
        "overlap": overlap,
        "phase_ring": phase_ring_strength,
        "vector_energy": vector_energy,
        "temporal_relativity": temporal_relativity,
        "trajectory_conservation": trajectory_conservation,
        "trajectory_prediction": trajectory_prediction,
    })
    field_alignment_score, field_alignment_score_weights = _derived_temporal_mix({
        "coherence": coherence_gate,
        "entropy": entropy_gate,
        "spin": spin,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity,
        "zero_point": zero_point,
    })
    kernel_control_gate, kernel_control_gate_weights = _derived_temporal_mix({
        "coherence": coherence_gate,
        "inertia": inertia,
        "temporal_relativity": temporal_relativity,
        "phase_ring": phase_ring_strength,
        "pressure_inverse": 1.0 - pressure,
        "zero_point": zero_point,
    })
    sequence_persistence_score, sequence_persistence_weights = _derived_temporal_mix({
        "persistence": persistence_gate,
        "entropy": entropy_gate,
        "temporal_coupling": temporal_coupling,
        "phase_ring_density": phase_ring_density,
        "trajectory_conservation": trajectory_conservation,
    })
    observation_freshness_gate, observation_freshness_weights = _derived_temporal_mix({
        "entropy": entropy_gate,
        "phase_origin": phase_origin,
        "temporal_relativity": temporal_relativity,
        "phase_ring": phase_ring_strength,
        "zero_point": zero_point,
    })
    temporal_overlap_target, temporal_overlap_target_weights = _derived_temporal_mix({
        "overlap": overlap,
        "coherence": coherence_gate,
        "temporal_coupling": temporal_coupling,
        "shared_phase_lock": shared_phase_lock,
        "trajectory_prediction": trajectory_prediction,
    })
    temporal_drive, temporal_drive_weights = _derived_temporal_mix({
        "observer_feedback": observer,
        "coherence": coherence_gate,
        "temporal_coupling": temporal_coupling,
        "phase_ring": phase_ring_strength,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity,
    })
    response_gate, response_gate_weights = _derived_temporal_mix({
        "coherence": coherence_gate,
        "temporal_coupling": temporal_coupling,
        "spin": spin,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity,
    })
    response_energy, response_energy_weights = _derived_temporal_mix({
        "valid_ratio": valid_gate,
        "inertia": inertia,
        "spin": spin,
        "vector_energy": vector_energy,
        "interception": interception,
        "trajectory_expansion": trajectory_expansion,
    })
    latency_norm, latency_norm_weights = _derived_temporal_mix({
        "gpu_mem_delta": abs(gpu - mem),
        "inertia": inertia,
        "observer_damping": observer_damping,
        "field_interference": field_interference,
        "latency_load": latency,
    })
    return {
        "sent_signal": float(sent_signal),
        "measured_signal": float(measured_signal),
        "integrated_feedback": float(integrated_feedback),
        "derivative_signal": float(derivative_signal),
        "feedback_gate": float(feedback_gate),
        "coherence_gate": float(coherence_gate_out),
        "observer_damping": float(observer_damping),
        "phase_transport_term": float(phase_transport_term),
        "flux_transport_term": float(flux_transport_term),
        "transport_coherence": float(transport_coherence),
        "transport_damping_gate": float(transport_damping_gate),
        "reverse_transport_gate": float(reverse_transport_gate),
        "trajectory_coherence_gate": float(trajectory_coherence_gate),
        "trajectory_feedback_gate": float(trajectory_feedback_gate),
        "load_hint": float(load_hint),
        "field_alignment_score": float(field_alignment_score),
        "kernel_control_gate": float(kernel_control_gate),
        "sequence_persistence_score": float(sequence_persistence_score),
        "observation_freshness_gate": float(observation_freshness_gate),
        "temporal_overlap_target": float(temporal_overlap_target),
        "temporal_drive": float(temporal_drive),
        "response_gate": float(response_gate),
        "response_energy": float(response_energy),
        "latency_norm": float(latency_norm),
        "weights": {
            "sent_signal": dict(sent_signal_weights),
            "measured_signal": dict(measured_signal_weights),
            "integrated_feedback": dict(integrated_feedback_weights),
            "derivative_signal": dict(derivative_signal_weights),
            "feedback_gate": dict(feedback_gate_weights),
            "coherence_gate": dict(coherence_gate_weights),
            "observer_damping": dict(observer_damping_weights),
            "phase_transport_term": dict(phase_transport_term_weights),
            "flux_transport_term": dict(flux_transport_term_weights),
            "transport_coherence": dict(transport_coherence_weights),
            "transport_damping_gate": dict(transport_damping_gate_weights),
            "reverse_transport_gate": dict(reverse_transport_gate_weights),
            "trajectory_coherence_gate": dict(trajectory_coherence_gate_weights),
            "trajectory_feedback_gate": dict(trajectory_feedback_gate_weights),
            "load_hint": dict(load_hint_weights),
            "field_alignment_score": dict(field_alignment_score_weights),
            "kernel_control_gate": dict(kernel_control_gate_weights),
            "sequence_persistence_score": dict(sequence_persistence_weights),
            "observation_freshness_gate": dict(observation_freshness_weights),
            "temporal_overlap_target": dict(temporal_overlap_target_weights),
            "temporal_drive": dict(temporal_drive_weights),
            "response_gate": dict(response_gate_weights),
            "response_energy": dict(response_energy_weights),
            "latency_norm": dict(latency_norm_weights),
        },
    }


def _compose_frequency_gradient_9d(
    frequency_norm: Any,
    amplitude_norm: Any,
    phase_turns: Any,
    field_dynamics: Dict[str, Any],
) -> Dict[str, Any]:
    freq = _clamp01(frequency_norm)
    amp = _clamp01(amplitude_norm)
    phase = _wrap_turns(phase_turns)
    phase_center = _clamp01(1.0 - abs(_phase_delta_turns(phase, 0.5)) * 2.0)
    joint_gate = _clamp01(math.sqrt(max(freq * amp, 0.0)))
    axis_x = _clamp01(field_dynamics.get("axis_scale_x", 0.0))
    axis_y = _clamp01(field_dynamics.get("axis_scale_y", 0.0))
    axis_z = _clamp01(field_dynamics.get("axis_scale_z", 0.0))
    axis_resonance = _clamp01(field_dynamics.get("axis_resonance", 0.0))
    temporal = _clamp01(field_dynamics.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(field_dynamics.get("inertial_mass_proxy", 0.0))
    vector_energy = _clamp01(field_dynamics.get("vector_energy", 0.0))
    spin_score = _clamp01(field_dynamics.get("spin_momentum_score", 0.0))
    orientation_alignment = _clamp01(field_dynamics.get("orientation_alignment", axis_resonance))
    wavelength_norm = _clamp01(field_dynamics.get("wavelength_norm", 1.0 - joint_gate))

    freq_x, freq_x_weights = _derived_temporal_mix({
        "frequency": freq,
        "phase_center": phase_center,
        "temporal": temporal,
        "axis_x": axis_x,
        "orientation": orientation_alignment,
        "wavelength": wavelength_norm,
    })
    freq_y, freq_y_weights = _derived_temporal_mix({
        "amplitude": amp,
        "phase_center": phase_center,
        "vector_energy": vector_energy,
        "axis_y": axis_y,
        "temporal": temporal,
        "orientation": orientation_alignment,
    })
    freq_z, freq_z_weights = _derived_temporal_mix({
        "joint_gate": joint_gate,
        "temporal": temporal,
        "spin": spin_score,
        "axis_z": axis_z,
        "inertia": inertia,
        "axis_resonance": axis_resonance,
    })

    gradient_00, weights_00 = _derived_temporal_mix({
        "primary": freq_x,
        "axis": axis_x,
        "phase_center": phase_center,
        "temporal": temporal,
        "wavelength": wavelength_norm,
    })
    gradient_01, weights_01 = _derived_temporal_mix({
        "primary": freq_x,
        "cross_axis": axis_y,
        "temporal": temporal,
        "inertia": inertia,
        "orientation": orientation_alignment,
    })
    gradient_02, weights_02 = _derived_temporal_mix({
        "primary": freq_x,
        "cross_axis": axis_z,
        "vector_energy": vector_energy,
        "spin": spin_score,
        "axis_resonance": axis_resonance,
    })
    gradient_10, weights_10 = _derived_temporal_mix({
        "primary": freq_y,
        "axis": axis_x,
        "temporal": temporal,
        "inertia": inertia,
        "orientation": orientation_alignment,
    })
    gradient_11, weights_11 = _derived_temporal_mix({
        "primary": freq_y,
        "axis": axis_y,
        "phase_center": phase_center,
        "vector_energy": vector_energy,
        "temporal": temporal,
    })
    gradient_12, weights_12 = _derived_temporal_mix({
        "primary": freq_y,
        "cross_axis": axis_z,
        "vector_energy": vector_energy,
        "spin": spin_score,
        "wavelength": wavelength_norm,
    })
    gradient_20, weights_20 = _derived_temporal_mix({
        "primary": freq_z,
        "axis": axis_x,
        "temporal": temporal,
        "inertia": inertia,
        "axis_resonance": axis_resonance,
    })
    gradient_21, weights_21 = _derived_temporal_mix({
        "primary": freq_z,
        "axis": axis_y,
        "vector_energy": vector_energy,
        "spin": spin_score,
        "orientation": orientation_alignment,
    })
    gradient_22, weights_22 = _derived_temporal_mix({
        "primary": freq_z,
        "axis": axis_z,
        "phase_center": phase_center,
        "wavelength": wavelength_norm,
        "axis_resonance": axis_resonance,
    })
    gradient_9d = [
        float(gradient_00),
        float(gradient_01),
        float(gradient_02),
        float(gradient_10),
        float(gradient_11),
        float(gradient_12),
        float(gradient_20),
        float(gradient_21),
        float(gradient_22),
    ]
    gradient_sig9 = [_quantize_sig9(value) for value in gradient_9d]
    return {
        "frequency_x": float(freq_x),
        "frequency_y": float(freq_y),
        "frequency_z": float(freq_z),
        "frequency_gradient_9d": [float(value) for value in gradient_9d],
        "field_gradient_9d": [float(value) for value in gradient_9d],
        "gradient_spectral_id": str(_build_photonic_identity(gradient_sig9)),
        "gradient_temporal_constant_weights": {
            "frequency_x": dict(freq_x_weights),
            "frequency_y": dict(freq_y_weights),
            "frequency_z": dict(freq_z_weights),
            "g00": dict(weights_00),
            "g01": dict(weights_01),
            "g02": dict(weights_02),
            "g10": dict(weights_10),
            "g11": dict(weights_11),
            "g12": dict(weights_12),
            "g20": dict(weights_20),
            "g21": dict(weights_21),
            "g22": dict(weights_22),
        },
    }


def _compute_phase_ring_state(
    phase_turns: Any,
    field_dynamics: Dict[str, Any],
    frequency_norm: Any,
    amplitude_norm: Any,
    resonance: Any,
    temporal_overlap: Any,
    flux_term: Any,
) -> Dict[str, Any]:
    temporal_relativity_state = _compute_temporal_relativity_state(
        phase_turns=phase_turns,
        field_dynamics=field_dynamics,
        frequency_norm=frequency_norm,
        amplitude_norm=amplitude_norm,
        resonance=resonance,
        temporal_overlap=temporal_overlap,
        flux_term=flux_term,
    )
    phase = _wrap_turns(phase_turns)
    freq = _clamp01(frequency_norm)
    amp = _clamp01(amplitude_norm)
    resonance_gate = _clamp01(resonance)
    overlap = _clamp01(temporal_overlap)
    flux_gate = _clamp01(flux_term)
    axis_resonance = _clamp01(field_dynamics.get("axis_resonance", 0.0))
    vector_energy = _clamp01(field_dynamics.get("vector_energy", 0.0))
    temporal_coupling = _clamp01(field_dynamics.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(field_dynamics.get("inertial_mass_proxy", 0.0))
    spin_score = _clamp01(field_dynamics.get("spin_momentum_score", 0.0))
    spin_x = _clamp_signed(field_dynamics.get("spin_axis_x", 0.0))
    spin_y = _clamp_signed(field_dynamics.get("spin_axis_y", 0.0))
    spin_z = _clamp_signed(field_dynamics.get("spin_axis_z", 0.0))
    wavelength_norm = _clamp01(temporal_relativity_state.get("wavelength_norm", 0.0))
    orientation_alignment = _clamp01(temporal_relativity_state.get("orientation_alignment", 0.0))
    relative_temporal_position = _clamp01(temporal_relativity_state.get("relative_temporal_position", 0.0))
    zero_point_line_distance = _clamp01(temporal_relativity_state.get("zero_point_line_distance", 0.0))
    field_interference_norm = _clamp01(temporal_relativity_state.get("field_interference_norm", 0.0))
    resonant_interception_inertia = _clamp01(temporal_relativity_state.get("resonant_interception_inertia", 0.0))
    temporal_relativity_norm = _clamp01(temporal_relativity_state.get("temporal_relativity_norm", 0.0))
    phase_bias, phase_bias_weights = _derived_temporal_mix({
        "axis_resonance": _clamp_signed(axis_resonance - 0.5),
        "spin": _clamp_signed(spin_score - 0.5),
        "flux": _clamp_signed(flux_gate - 0.5),
        "resonance_overlap": _clamp_signed(resonance_gate - overlap),
        "spin_z": float(spin_z),
        "spin_delta": _clamp_signed(spin_x - spin_y),
        "temporal_relativity": _clamp_signed(temporal_relativity_norm - 0.5),
        "relative_position": _clamp_signed(relative_temporal_position - 0.5),
        "orientation": _clamp_signed(orientation_alignment - 0.5),
        "zero_point_alignment": _clamp_signed((1.0 - zero_point_line_distance) - 0.5),
        "interference_alignment": _clamp_signed((1.0 - field_interference_norm) - 0.5),
    }, signed=True, limit=0.25)
    phase_injection_delta_turns = _clamp_signed(phase_bias, 0.25)
    phase_injection_turns = _wrap_turns(phase + phase_injection_delta_turns)
    phase_lock = _clamp01(1.0 - abs(_phase_delta_turns(phase, phase_injection_turns)) * 2.0)
    phase_ring_closure, phase_ring_closure_weights = _derived_temporal_mix({
        "axis_resonance": axis_resonance,
        "temporal": temporal_coupling,
        "spin": spin_score,
        "phase_lock": phase_lock,
        "resonance": resonance_gate,
        "temporal_relativity": temporal_relativity_norm,
        "orientation": orientation_alignment,
    })
    phase_ring_density, phase_ring_density_weights = _derived_temporal_mix({
        "amplitude": amp,
        "frequency": freq,
        "vector_energy": vector_energy,
        "inertia": inertia,
        "overlap": overlap,
        "wavelength": wavelength_norm,
        "relative_position": relative_temporal_position,
    })
    phase_ring_strength, phase_ring_strength_weights = _derived_temporal_mix({
        "closure": phase_ring_closure,
        "density": phase_ring_density,
        "temporal": temporal_coupling,
        "spin": spin_score,
        "interception_inertia": resonant_interception_inertia,
        "temporal_relativity": temporal_relativity_norm,
    })
    zero_point_crossover_gate, zero_point_weights = _derived_temporal_mix({
        "phase_ring_strength": phase_ring_strength,
        "phase_lock": phase_lock,
        "axis_resonance": axis_resonance,
        "temporal": temporal_coupling,
        "zero_point_alignment": 1.0 - zero_point_line_distance,
        "interference": field_interference_norm,
        "interception_inertia": resonant_interception_inertia,
        "spin_balance": 1.0 - min(1.0, abs(spin_x + spin_y + spin_z) / 3.0),
    })
    shared_vector_collapse_gate, collapse_weights = _derived_temporal_mix({
        "zero_point_crossover": zero_point_crossover_gate,
        "axis_resonance": axis_resonance,
        "overlap": overlap,
        "inertia": inertia,
        "phase_ring_closure": phase_ring_closure,
        "interference": field_interference_norm,
        "temporal_relativity": temporal_relativity_norm,
    })
    shared_vector_phase_lock, phase_lock_weights = _derived_temporal_mix({
        "phase_ring_closure": phase_ring_closure,
        "phase_lock": phase_lock,
        "resonance": resonance_gate,
        "overlap": overlap,
        "orientation": orientation_alignment,
        "zero_point_alignment": 1.0 - zero_point_line_distance,
    })
    inertial_basin_strength, inertial_basin_weights = _derived_temporal_mix({
        "inertia": inertia,
        "phase_ring_strength": phase_ring_strength,
        "zero_point_crossover": zero_point_crossover_gate,
        "collapse": shared_vector_collapse_gate,
        "interception_inertia": resonant_interception_inertia,
        "temporal_relativity": temporal_relativity_norm,
    })
    phase_ring_zone_id = _stable_word({
        "phase": round(float(phase_injection_turns), 6),
        "closure": round(float(phase_ring_closure), 6),
        "density": round(float(phase_ring_density), 6),
        "strength": round(float(phase_ring_strength), 6),
        "collapse": round(float(shared_vector_collapse_gate), 6),
    })
    result = {
        "gpu_pulse_phase_effect": float(phase_injection_delta_turns),
        "phase_injection_delta_turns": float(phase_injection_delta_turns),
        "phase_injection_turns": float(phase_injection_turns),
        "phase_ring_closure": float(phase_ring_closure),
        "phase_ring_density": float(phase_ring_density),
        "phase_ring_strength": float(phase_ring_strength),
        "zero_point_crossover_gate": float(zero_point_crossover_gate),
        "shared_vector_collapse_gate": float(shared_vector_collapse_gate),
        "shared_vector_phase_lock": float(shared_vector_phase_lock),
        "inertial_basin_strength": float(inertial_basin_strength),
        "phase_ring_zone_id": int(phase_ring_zone_id),
        "phase_ring_temporal_constant_weights": {
            "phase_bias": dict(phase_bias_weights),
            "phase_ring_closure": dict(phase_ring_closure_weights),
            "phase_ring_density": dict(phase_ring_density_weights),
            "phase_ring_strength": dict(phase_ring_strength_weights),
            "zero_point_crossover": dict(zero_point_weights),
            "collapse": dict(collapse_weights),
            "phase_lock": dict(phase_lock_weights),
            "inertial_basin": dict(inertial_basin_weights),
        },
    }
    result.update(temporal_relativity_state)
    return result


def _compute_temporal_relativity_state(
    phase_turns: Any,
    field_dynamics: Dict[str, Any],
    frequency_norm: Any,
    amplitude_norm: Any,
    resonance: Any,
    temporal_overlap: Any,
    flux_term: Any,
) -> Dict[str, Any]:
    phase = _wrap_turns(phase_turns)
    freq = _clamp01(frequency_norm)
    amp = _clamp01(amplitude_norm)
    resonance_gate = _clamp01(resonance)
    overlap = _clamp01(temporal_overlap)
    flux_gate = _clamp01(flux_term)
    axis_x = _clamp01(field_dynamics.get("axis_scale_x", 0.0))
    axis_y = _clamp01(field_dynamics.get("axis_scale_y", 0.0))
    axis_z = _clamp01(field_dynamics.get("axis_scale_z", 0.0))
    vector_energy = _clamp01(field_dynamics.get("vector_energy", 0.0))
    temporal_coupling = _clamp01(field_dynamics.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(field_dynamics.get("inertial_mass_proxy", 0.0))
    spin_score = _clamp01(field_dynamics.get("spin_momentum_score", 0.0))
    spin_x = _clamp_signed(field_dynamics.get("spin_axis_x", 0.0))
    spin_y = _clamp_signed(field_dynamics.get("spin_axis_y", 0.0))
    spin_z = _clamp_signed(field_dynamics.get("spin_axis_z", 0.0))
    orientation_x = _clamp_signed((axis_x - axis_y) + 0.50 * spin_x, 1.0)
    orientation_y = _clamp_signed((axis_y - axis_z) + 0.50 * spin_y, 1.0)
    orientation_z = _clamp_signed((axis_z - axis_x) + 0.50 * spin_z, 1.0)
    orientation_alignment = _clamp01(
        1.0 - (abs(orientation_x) + abs(orientation_y) + abs(orientation_z)) / 3.0
    )
    rotational_velocity_norm = _clamp01(_vector_energy([spin_x, spin_y, spin_z]) * (0.78 + 0.22 * amp))
    vector_x = _clamp_signed((axis_x - 0.5) + 0.25 * spin_x, 1.0)
    vector_y = _clamp_signed((axis_y - 0.5) + 0.25 * spin_y, 1.0)
    vector_z = _clamp_signed((axis_z - 0.5) + 0.25 * spin_z, 1.0)
    vector_energy_norm = _clamp01(_vector_energy([vector_x, vector_y, vector_z]))
    zero_point_line = _clamp01(1.0 - abs(_phase_delta_turns(phase, 0.0)) * 2.0)
    phase_peak = _clamp01(max(
        1.0 - abs(_phase_delta_turns(phase, 0.25)) * 2.0,
        1.0 - abs(_phase_delta_turns(phase, 0.75)) * 2.0,
    ))
    wavelength_time_norm = _clamp01(1.0 - freq)
    amplitude_excursion_norm = _clamp01(amp * max(phase_peak, zero_point_line))
    observer_feedback = _clamp01(field_dynamics.get("observer_damping", field_dynamics.get("observer_factor", temporal_coupling)))
    phase_memory = _clamp01(field_dynamics.get("phase_alignment_probability", overlap))
    zero_point_crossover = _clamp01(field_dynamics.get("zero_point_crossover_norm", zero_point_line))
    field_time_norm, field_time_weights = _derived_temporal_mix({
        "wavelength_time": wavelength_time_norm,
        "amplitude_excursion": amplitude_excursion_norm,
        "resonance": resonance_gate,
        "overlap": overlap,
    })
    vector_zero_alignment = _clamp01(
        1.0
        - (
            abs(abs(vector_x) - zero_point_line)
            + abs(abs(vector_y) - zero_point_line)
            + abs(abs(vector_z) - zero_point_line)
        )
        / 3.0
    )
    path_speed_norm, path_speed_weights = _derived_temporal_mix({
        "input_speed": rotational_velocity_norm,
        "vector_energy": vector_energy_norm,
        "field_time": field_time_norm,
        "flux": flux_gate,
    })
    phase_alignment_probability, phase_alignment_weights = _derived_temporal_mix({
        "zero_point_line": zero_point_line,
        "phase_peak": phase_peak,
        "overlap": overlap,
        "vector_zero_alignment": vector_zero_alignment,
        "resonance": resonance_gate,
        "observer_feedback": observer_feedback,
    })
    entanglement_probability, entanglement_weights = _derived_temporal_mix({
        "phase_alignment": phase_alignment_probability,
        "vector_alignment": vector_zero_alignment,
        "flux": flux_gate,
        "spin": spin_score,
        "orientation": orientation_alignment,
        "phase_memory": phase_memory,
    })
    cross_talk_force_norm, cross_talk_weights = _derived_temporal_mix({
        "entanglement": entanglement_probability,
        "vector_energy": vector_energy_norm,
        "flux": flux_gate,
        "zero_point": max(zero_point_crossover, zero_point_line),
        "phase_memory": phase_memory,
        "overlap": overlap,
    })
    resonant_interception_inertia, interception_weights = _derived_temporal_mix({
        "field_time": field_time_norm,
        "vector_energy": vector_energy_norm,
        "path_speed": path_speed_norm,
        "coupling": temporal_coupling,
        "spin": spin_score,
        "zero_point": max(zero_point_crossover, zero_point_line),
    })
    relative_temporal_position, relative_position_weights = _derived_temporal_mix({
        "field_time": field_time_norm,
        "phase_alignment": phase_alignment_probability,
        "phase_peak": phase_peak,
        "path_speed": path_speed_norm,
        "zero_point": max(zero_point_crossover, zero_point_line),
        "orientation": orientation_alignment,
    })
    field_interference_norm, interference_weights = _derived_temporal_mix({
        "cross_talk_force": cross_talk_force_norm,
        "resonance_offset": abs(resonance_gate - overlap),
        "flux_offset": abs(vector_energy - flux_gate),
        "orientation_shear": 1.0 - orientation_alignment,
        "zero_point_detune": 1.0 - max(zero_point_crossover, zero_point_line),
    })
    temporal_relativity_norm, temporal_relativity_weights = _derived_temporal_mix({
        "field_time": field_time_norm,
        "phase_alignment": phase_alignment_probability,
        "entanglement": entanglement_probability,
        "path_speed": path_speed_norm,
        "interception": resonant_interception_inertia,
        "zero_point": max(zero_point_crossover, zero_point_line),
    })
    zero_point_line_distance = _clamp01(1.0 - max(zero_point_crossover, zero_point_line))
    wavelength_norm = wavelength_time_norm
    phase_origin_alignment = zero_point_line
    return {
        "phase_position_turns": float(phase),
        "wavelength_norm": float(wavelength_norm),
        "phase_origin_alignment": float(phase_origin_alignment),
        "orientation_x": float(orientation_x),
        "orientation_y": float(orientation_y),
        "orientation_z": float(orientation_z),
        "orientation_alignment": float(orientation_alignment),
        "rotational_velocity_norm": float(rotational_velocity_norm),
        "relative_temporal_position": float(relative_temporal_position),
        "zero_point_line_distance": float(zero_point_line_distance),
        "field_interference_norm": float(field_interference_norm),
        "resonant_interception_inertia": float(resonant_interception_inertia),
        "temporal_relativity_norm": float(temporal_relativity_norm),
        "phase_peak_proximity": float(phase_peak),
        "field_time_norm": float(field_time_norm),
        "amplitude_excursion_norm": float(amplitude_excursion_norm),
        "vector_energy_norm": float(vector_energy_norm),
        "vector_zero_alignment": float(vector_zero_alignment),
        "path_speed_norm": float(path_speed_norm),
        "phase_alignment_probability": float(phase_alignment_probability),
        "entanglement_probability": float(entanglement_probability),
        "cross_talk_force_norm": float(cross_talk_force_norm),
        "intercept_inertia_norm": float(resonant_interception_inertia),
        "temporal_relativity_constant_weights": {
            "field_time": dict(field_time_weights),
            "path_speed": dict(path_speed_weights),
            "phase_alignment": dict(phase_alignment_weights),
            "entanglement": dict(entanglement_weights),
            "cross_talk": dict(cross_talk_weights),
            "relative_position": dict(relative_position_weights),
            "interference": dict(interference_weights),
            "interception": dict(interception_weights),
            "temporal_relativity": dict(temporal_relativity_weights),
        },
        "temporal_relativity_vector": [
            float(phase),
            float(wavelength_norm),
            float(resonant_interception_inertia),
            float(max(zero_point_crossover, zero_point_line)),
        ],
    }


def _compute_axis_field_dynamics(
    frequency_norm: Any,
    amplitude_norm: Any,
    phase_turns: Any,
    resonance: Any,
    temporal_overlap: Any,
    flux_term: Any,
    vector_x: Any = 0.0,
    vector_y: Any = 0.0,
    vector_z: Any = 0.0,
    energy_hint: Any = 0.0,
) -> Dict[str, float]:
    freq = _clamp01(frequency_norm)
    amp = _clamp01(amplitude_norm)
    phase = _clamp01(phase_turns)
    resonance_gate = _clamp01(resonance)
    overlap = _clamp01(temporal_overlap)
    flux_gate = _clamp01(flux_term)
    energy_gate = _clamp01(energy_hint)
    x = _clamp_signed(vector_x)
    y = _clamp_signed(vector_y)
    z = _clamp_signed(vector_z)
    phase_center = _clamp01(1.0 - abs(_phase_delta_turns(phase, 0.5)) * 2.0)

    joint_gate = _clamp01(math.sqrt(max(freq * amp, 0.0)))
    axis_scale_x, axis_scale_x_weights = _derived_temporal_mix({
        "baseline": phase_center,
        "frequency": freq,
        "resonance": resonance_gate,
        "vector": abs(x),
    })
    axis_scale_y, axis_scale_y_weights = _derived_temporal_mix({
        "baseline": phase_center,
        "amplitude": amp,
        "overlap": overlap,
        "vector": abs(y),
    })
    axis_scale_z, axis_scale_z_weights = _derived_temporal_mix({
        "baseline": phase_center,
        "joint_gate": joint_gate,
        "flux": flux_gate,
        "vector": abs(z),
    })
    axis_resonance = _clamp01(
        1.0 - (
            abs(axis_scale_x - axis_scale_y)
            + abs(axis_scale_y - axis_scale_z)
            + abs(axis_scale_x - axis_scale_z)
        ) / 3.0
    )

    scaled_x = x * (0.5 + 0.5 * axis_scale_x)
    scaled_y = y * (0.5 + 0.5 * axis_scale_y)
    scaled_z = z * (0.5 + 0.5 * axis_scale_z)
    vector_energy = _clamp01(_vector_energy([scaled_x, scaled_y, scaled_z]) + 0.20 * energy_gate)
    speed_measure, speed_measure_weights = _derived_temporal_mix({
        "vector_energy": vector_energy,
        "frequency": freq,
        "amplitude": amp,
        "phase_center": phase_center,
    })
    gamma = 1.0 / math.sqrt(max(0.08, 1.0 - (0.92 * speed_measure * speed_measure)))
    relativistic_correlation = _clamp01((gamma - 1.0) / 2.5)
    temporal_coupling_moment, temporal_coupling_weights = _derived_temporal_mix({
        "resonance": resonance_gate,
        "axis_resonance": axis_resonance,
        "overlap": overlap,
        "joint_gate": joint_gate,
        "flux": flux_gate,
    })

    spin_axis_x = _clamp_signed((scaled_y * axis_scale_z) - (scaled_z * axis_scale_y))
    spin_axis_y = _clamp_signed((scaled_z * axis_scale_x) - (scaled_x * axis_scale_z))
    spin_axis_z = _clamp_signed((scaled_x * axis_scale_y) - (scaled_y * axis_scale_x))
    spin_momentum_score = _clamp01(_vector_energy([spin_axis_x, spin_axis_y, spin_axis_z]))
    inertial_mass_proxy, inertial_mass_weights = _derived_temporal_mix({
        "vector_energy": vector_energy,
        "relativistic": relativistic_correlation,
        "spin": spin_momentum_score,
        "temporal": temporal_coupling_moment,
    })
    field_dynamics = {
        "axis_scale_x": float(axis_scale_x),
        "axis_scale_y": float(axis_scale_y),
        "axis_scale_z": float(axis_scale_z),
        "axis_resonance": float(axis_resonance),
        "vector_energy": float(vector_energy),
        "speed_measure": float(speed_measure),
        "relativistic_correlation": float(relativistic_correlation),
        "temporal_coupling_moment": float(temporal_coupling_moment),
        "spin_axis_x": float(spin_axis_x),
        "spin_axis_y": float(spin_axis_y),
        "spin_axis_z": float(spin_axis_z),
        "spin_momentum_score": float(spin_momentum_score),
        "inertial_mass_proxy": float(inertial_mass_proxy),
        "phase_turns": float(phase),
        "axis_temporal_constant_weights": {
            "axis_scale_x": dict(axis_scale_x_weights),
            "axis_scale_y": dict(axis_scale_y_weights),
            "axis_scale_z": dict(axis_scale_z_weights),
            "speed_measure": dict(speed_measure_weights),
            "temporal_coupling": dict(temporal_coupling_weights),
            "inertial_mass": dict(inertial_mass_weights),
        },
    }
    field_dynamics.update(
        _compose_frequency_gradient_9d(
            frequency_norm=freq,
            amplitude_norm=amp,
            phase_turns=phase,
            field_dynamics=field_dynamics,
        )
    )
    field_dynamics.update(
        _compute_phase_ring_state(
            phase_turns=phase,
            field_dynamics=field_dynamics,
            frequency_norm=freq,
            amplitude_norm=amp,
            resonance=resonance_gate,
            temporal_overlap=overlap,
            flux_term=flux_gate,
        )
    )
    return field_dynamics


def _quantize_sig9(value: Any) -> int:
    return int(round(_clamp01(value) * 1000000.0)) & 0xFFFFFFFF


def _build_photonic_identity(sig9: list[int]) -> str:
    out = ["PID9"]
    for component in list(sig9 or [])[:9]:
        out.append("%06x" % (int(component) & 0xFFFFFF))
    while len(out) < 10:
        out.append("000000")
    return "-".join(out)


def _trajectory_budget(values: list[float]) -> float:
    return float(sum(_clamp01(value) for value in list(values or [])))


def _redistribute_trajectory_budget(values: list[float], budget: Any) -> list[float]:
    working = [_clamp01(value) for value in list(values or [])]
    if not working:
        return []
    target_budget = max(0.0, min(float(len(working)), _safe_float(budget, 0.0)))
    if target_budget <= 0.0:
        return [0.0 for _ in working]
    weights = [max(1.0e-9, float(value)) for value in working]
    redistributed = [0.0 for _ in working]
    remaining = set(range(len(working)))
    remaining_budget = float(target_budget)
    while remaining and remaining_budget > 1.0e-9:
        total_weight = sum(weights[index] for index in remaining)
        saturated: list[int] = []
        for index in list(remaining):
            share = remaining_budget * (weights[index] / max(total_weight, 1.0e-9))
            if share >= 1.0:
                redistributed[index] = 1.0
                remaining_budget -= 1.0
                saturated.append(index)
        if not saturated:
            for index in remaining:
                redistributed[index] = _clamp01(
                    remaining_budget * (weights[index] / max(total_weight, 1.0e-9))
                )
            remaining_budget = 0.0
            break
        for index in saturated:
            remaining.discard(index)
    return [_clamp01(value) for value in redistributed]


def _compose_photonic_trajectory_9d(
    phase_turns: Any,
    field_dynamics: Dict[str, Any],
    flux_term: Any,
    coherence_gate: Any,
    feedback_gate: Any,
) -> Dict[str, Any]:
    trajectory_9d = [
        _wrap_turns(phase_turns),
        _clamp01(field_dynamics.get("axis_scale_x", 0.0)),
        _clamp01(field_dynamics.get("axis_scale_y", 0.0)),
        _clamp01(field_dynamics.get("axis_scale_z", 0.0)),
        _clamp01(flux_term),
        _clamp01(coherence_gate),
        _clamp01(field_dynamics.get("spin_momentum_score", 0.0)),
        _clamp01(field_dynamics.get("inertial_mass_proxy", 0.0)),
        _clamp01(feedback_gate),
    ]
    spectra_sig9 = [_quantize_sig9(value) for value in trajectory_9d]
    return {
        "trajectory_9d": [float(value) for value in trajectory_9d],
        "spectra_9d": [float(value) for value in trajectory_9d],
        "spectra_sig9": [int(value) for value in spectra_sig9],
        "trajectory_spectral_id": _build_photonic_identity(spectra_sig9),
        "trajectory_budget": float(_trajectory_budget(trajectory_9d)),
        "trajectory_energy": float(_vector_energy(trajectory_9d)),
    }


def _predict_photonic_trajectory_9d(
    current_trajectory_9d: list[float],
    previous_predicted_trajectory_9d: list[float] | None,
    field_dynamics: Dict[str, Any],
    transport_prediction: Dict[str, Any],
    headroom: float,
    latency_load: float,
    noise_gate: float,
) -> Dict[str, Any]:
    current = [_clamp01(value) for value in list(current_trajectory_9d or [])[:9]]
    while len(current) < 9:
        current.append(0.0)
    previous = [_clamp01(value) for value in list(previous_predicted_trajectory_9d or [])[:9]]
    while len(previous) < 9:
        previous.append(0.0)

    phase_transport_term = _safe_float(transport_prediction.get("phase_transport_term", 0.0), 0.0)
    flux_transport_term = _clamp01(transport_prediction.get("flux_transport_term", 0.0))
    observer_damping = _clamp01(transport_prediction.get("observer_damping", 0.0))
    transport_coherence = _clamp01(transport_prediction.get("transport_coherence", 0.0))
    transport_damping_gate = _clamp01(transport_prediction.get("transport_damping_gate", 0.0))
    reverse_transport_gate = _clamp01(transport_prediction.get("reverse_transport_gate", 0.0))
    temporal_coupling = _clamp01(field_dynamics.get("temporal_coupling_moment", 0.0))
    inertial_mass = _clamp01(field_dynamics.get("inertial_mass_proxy", 0.0))
    relativistic_correlation = _clamp01(field_dynamics.get("relativistic_correlation", 0.0))
    axis_resonance = _clamp01(field_dynamics.get("axis_resonance", 0.0))
    phase_ring_strength = _clamp01(field_dynamics.get("phase_ring_strength", 0.0))
    shared_phase_lock = _clamp01(field_dynamics.get("shared_vector_phase_lock", 0.0))
    temporal_relativity = _clamp01(field_dynamics.get("temporal_relativity_norm", 0.0))

    previous_mix = 0.0
    previous_mix_weights: Dict[str, float] = {}
    if any(value > 0.0 for value in previous):
        previous_mix, previous_mix_weights = _derived_temporal_mix({
            "history_alignment": 1.0 - _clamp01(abs(_phase_delta_turns(current[0], previous[0])) * 2.0),
            "temporal_coupling": temporal_coupling,
            "shared_phase_lock": shared_phase_lock,
            "headroom": headroom,
            "latency_inverse": 1.0 - latency_load,
        })

    trajectory_expansion_term, trajectory_expansion_weights = _derived_temporal_mix({
        "headroom": headroom,
        "phase_transport": _clamp01(abs(float(phase_transport_term)) * 2.0),
        "flux_transport": float(flux_transport_term),
        "temporal_coupling": float(temporal_coupling),
        "relativistic": float(relativistic_correlation),
        "transport_coherence": float(transport_coherence),
        "reverse_transport": float(reverse_transport_gate),
        "observer_inverse": 1.0 - float(observer_damping),
        "temporal_relativity": float(temporal_relativity),
    })
    trajectory_sequence_density, trajectory_sequence_weights = _derived_temporal_mix({
        "noise_inverse": 1.0 - float(noise_gate),
        "transport_coherence": float(transport_coherence),
        "temporal_coupling": float(temporal_coupling),
        "axis_resonance": float(axis_resonance),
        "transport_damping": float(transport_damping_gate),
        "latency_inverse": 1.0 - float(latency_load),
        "phase_ring": float(phase_ring_strength),
    })
    trajectory_noise_feedback_norm, trajectory_noise_feedback_weights = _derived_temporal_mix({
        "transport_damping": float(transport_damping_gate),
        "reverse_transport": float(reverse_transport_gate),
        "noise_inverse": 1.0 - float(noise_gate),
        "axis_resonance": float(axis_resonance),
        "feedback_slot": float(current[8]),
        "shared_phase_lock": float(shared_phase_lock),
    })

    phase_axis_drive, phase_axis_drive_weights = _derived_temporal_mix({
        "phase_transport": _clamp01(abs(float(phase_transport_term)) * 2.0),
        "temporal_coupling": temporal_coupling,
        "axis_resonance": axis_resonance,
        "shared_phase_lock": shared_phase_lock,
        "phase_ring": phase_ring_strength,
    })
    flux_axis_drive, flux_axis_drive_weights = _derived_temporal_mix({
        "flux_transport": flux_transport_term,
        "temporal_coupling": temporal_coupling,
        "axis_resonance": axis_resonance,
        "phase_ring": phase_ring_strength,
        "headroom": headroom,
    })
    expansion_axis_drive, expansion_axis_drive_weights = _derived_temporal_mix({
        "trajectory_expansion": trajectory_expansion_term,
        "temporal_relativity": temporal_relativity,
        "headroom": headroom,
        "reverse_transport": reverse_transport_gate,
        "transport_coherence": transport_coherence,
    })
    damping_axis_drive, damping_axis_drive_weights = _derived_temporal_mix({
        "observer_damping": observer_damping,
        "noise_gate": noise_gate,
        "latency_load": latency_load,
        "inertial_mass": inertial_mass,
        "temporal_relativity": temporal_relativity,
    })
    density_axis_drive, density_axis_drive_weights = _derived_temporal_mix({
        "trajectory_sequence": trajectory_sequence_density,
        "noise_feedback": trajectory_noise_feedback_norm,
        "phase_ring": phase_ring_strength,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity,
    })

    phase_history_delta = 0.0
    if previous_mix > 0.0:
        phase_history_delta = previous_mix * _phase_delta_turns(current[0], previous[0])
    predicted_phase = _wrap_turns(
        float(current[0])
        + float(phase_transport_term) * 0.50
        + float(trajectory_expansion_term) * 0.04
        + float(phase_history_delta)
    )
    raw_predicted = [
        float(predicted_phase),
        _clamp01(current[1] + phase_transport_term * phase_axis_drive + trajectory_expansion_term * expansion_axis_drive + previous_mix * (previous[1] - current[1]) - observer_damping * damping_axis_drive),
        _clamp01(current[2] + flux_transport_term * flux_axis_drive + trajectory_expansion_term * expansion_axis_drive + previous_mix * (previous[2] - current[2]) - observer_damping * damping_axis_drive),
        _clamp01(current[3] + phase_transport_term * phase_axis_drive + flux_transport_term * flux_axis_drive + trajectory_expansion_term * expansion_axis_drive + previous_mix * (previous[3] - current[3]) - observer_damping * damping_axis_drive),
        _clamp01(current[4] + flux_transport_term * flux_axis_drive + trajectory_expansion_term * expansion_axis_drive + previous_mix * (previous[4] - current[4]) - observer_damping * damping_axis_drive),
        _clamp01(current[5] + trajectory_sequence_density * density_axis_drive + trajectory_noise_feedback_norm * density_axis_drive + previous_mix * (previous[5] - current[5]) - observer_damping * damping_axis_drive),
        _clamp01(current[6] + _clamp01(abs(phase_transport_term) * 2.0) * phase_axis_drive + axis_resonance * flux_axis_drive + relativistic_correlation * expansion_axis_drive + previous_mix * (previous[6] - current[6]) - observer_damping * damping_axis_drive),
        _clamp01(current[7] + temporal_coupling * density_axis_drive + headroom * expansion_axis_drive + previous_mix * (previous[7] - current[7]) - latency_load * damping_axis_drive),
        _clamp01(current[8] + trajectory_noise_feedback_norm * density_axis_drive + trajectory_sequence_density * expansion_axis_drive + previous_mix * (previous[8] - current[8]) - observer_damping * damping_axis_drive),
    ]
    predicted_trajectory_9d = _redistribute_trajectory_budget(raw_predicted, _trajectory_budget(current))
    trajectory_velocity_9d = [
        float(_phase_delta_turns(current[0], predicted_trajectory_9d[0])),
        *[float(predicted_trajectory_9d[index] - current[index]) for index in range(1, 9)],
    ]
    trajectory_conservation_alignment = _clamp01(
        1.0 - abs(_trajectory_budget(predicted_trajectory_9d) - _trajectory_budget(current)) / max(_trajectory_budget(current), 1.0)
    )
    trajectory_prediction_alignment, trajectory_prediction_weights = _derived_temporal_mix({
        "conservation": float(trajectory_conservation_alignment),
        "noise_feedback": float(trajectory_noise_feedback_norm),
        "sequence_density": float(trajectory_sequence_density),
        "phase_velocity_inverse": 1.0 - _clamp01(abs(trajectory_velocity_9d[0]) * 2.0),
        "velocity_inverse": 1.0 - float(_vector_energy(trajectory_velocity_9d)),
        "temporal_relativity": float(temporal_relativity),
    })
    predicted_state = _compose_photonic_trajectory_9d(
        phase_turns=predicted_trajectory_9d[0],
        field_dynamics={
            "axis_scale_x": predicted_trajectory_9d[1],
            "axis_scale_y": predicted_trajectory_9d[2],
            "axis_scale_z": predicted_trajectory_9d[3],
            "spin_momentum_score": predicted_trajectory_9d[6],
            "inertial_mass_proxy": predicted_trajectory_9d[7],
        },
        flux_term=predicted_trajectory_9d[4],
        coherence_gate=predicted_trajectory_9d[5],
        feedback_gate=predicted_trajectory_9d[8],
    )
    predicted_state.update({
        "predicted_trajectory_9d": [float(value) for value in predicted_trajectory_9d],
        "trajectory_velocity_9d": [float(value) for value in trajectory_velocity_9d],
        "trajectory_expansion_term": float(trajectory_expansion_term),
        "trajectory_sequence_density": float(trajectory_sequence_density),
        "trajectory_noise_feedback_norm": float(trajectory_noise_feedback_norm),
        "trajectory_conservation_alignment": float(trajectory_conservation_alignment),
        "trajectory_prediction_alignment": float(trajectory_prediction_alignment),
        "trajectory_temporal_constant_weights": {
            "previous_mix": dict(previous_mix_weights),
            "trajectory_expansion": dict(trajectory_expansion_weights),
            "trajectory_sequence_density": dict(trajectory_sequence_weights),
            "trajectory_noise_feedback": dict(trajectory_noise_feedback_weights),
            "phase_axis_drive": dict(phase_axis_drive_weights),
            "flux_axis_drive": dict(flux_axis_drive_weights),
            "expansion_axis_drive": dict(expansion_axis_drive_weights),
            "damping_axis_drive": dict(damping_axis_drive_weights),
            "density_axis_drive": dict(density_axis_drive_weights),
            "trajectory_prediction_alignment": dict(trajectory_prediction_weights),
        },
    })
    return predicted_state


def _dominant_alignment(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    span = min(len(lhs), len(rhs))
    if span <= 0:
        return 0.0
    delta = 0.0
    for idx in range(span):
        delta += abs(float(lhs[idx]) - float(rhs[idx]))
    return _clamp01(1.0 - (delta / float(span)))


def _build_feedback_axis_vector(system_payload: Dict[str, Any], nonce_snapshot: Dict[str, Any]) -> list[float]:
    gpu_util = _clamp01(system_payload.get("gpu_util", 0.0))
    mem_bw_util = _clamp01(system_payload.get("mem_bw_util", 0.0))
    cpu_util = _clamp01(system_payload.get("cpu_util", 0.0))
    phase = _clamp01(nonce_snapshot.get("phase", 0.0))
    axis_scale_x = _clamp01(nonce_snapshot.get("axis_scale_x", gpu_util))
    axis_scale_y = _clamp01(nonce_snapshot.get("axis_scale_y", mem_bw_util))
    axis_scale_z = _clamp01(nonce_snapshot.get("axis_scale_z", cpu_util))
    phase_ring_strength = _clamp01(nonce_snapshot.get("phase_ring_strength", 0.0))
    temporal_relativity = _clamp01(nonce_snapshot.get("temporal_relativity_norm", 0.0))
    feedback_axis_x, _ = _derived_temporal_mix({
        "gpu_util": gpu_util,
        "axis_scale": axis_scale_x,
        "phase_ring": phase_ring_strength,
        "temporal_relativity": temporal_relativity,
    })
    feedback_axis_y, _ = _derived_temporal_mix({
        "memory": mem_bw_util,
        "axis_scale": axis_scale_y,
        "phase_ring": phase_ring_strength,
        "temporal_relativity": temporal_relativity,
    })
    feedback_axis_z, _ = _derived_temporal_mix({
        "cpu": cpu_util,
        "axis_scale": axis_scale_z,
        "phase_ring": phase_ring_strength,
        "temporal_relativity": temporal_relativity,
    })
    return [
        float(feedback_axis_x),
        float(feedback_axis_y),
        float(feedback_axis_z),
        phase,
    ]


def _build_feedback_dof_vector(system_payload: Dict[str, Any], nonce_snapshot: Dict[str, Any]) -> list[float]:
    gpu_util = _clamp01(system_payload.get("gpu_util", 0.0))
    mem_bw_util = _clamp01(system_payload.get("mem_bw_util", 0.0))
    cpu_util = _clamp01(system_payload.get("cpu_util", 0.0))
    global_util = _clamp01(system_payload.get("global_util", 0.0))
    atomic_x = _safe_float(nonce_snapshot.get("atomic_vector_x", 0.0))
    atomic_y = _safe_float(nonce_snapshot.get("atomic_vector_y", 0.0))
    atomic_z = _safe_float(nonce_snapshot.get("atomic_vector_z", 0.0))
    phase = _clamp01(nonce_snapshot.get("phase", 0.0))
    spin_x = _clamp01(abs(_safe_float(nonce_snapshot.get("spin_axis_x", 0.0), 0.0)))
    spin_y = _clamp01(abs(_safe_float(nonce_snapshot.get("spin_axis_y", 0.0), 0.0)))
    spin_z = _clamp01(abs(_safe_float(nonce_snapshot.get("spin_axis_z", 0.0), 0.0)))
    vector_energy = _clamp01(nonce_snapshot.get("vector_energy", 0.0))
    temporal_coupling = _clamp01(nonce_snapshot.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(nonce_snapshot.get("inertial_mass_proxy", 0.0))
    return [
        global_util,
        gpu_util,
        phase,
        cpu_util,
        _clamp01(max(abs(atomic_x), spin_x)),
        _clamp01(max(abs(atomic_y), spin_y)),
        _clamp01(max(abs(atomic_z), spin_z)),
        vector_energy,
        temporal_coupling,
        inertia,
    ]


def _build_simulation_field_state(system_payload: Dict[str, Any], nonce_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    atomic_x = _safe_float(nonce_snapshot.get("atomic_vector_x", 0.0))
    atomic_y = _safe_float(nonce_snapshot.get("atomic_vector_y", 0.0))
    atomic_z = _safe_float(nonce_snapshot.get("atomic_vector_z", 0.0))
    phase = _clamp01(nonce_snapshot.get("phase", 0.0))
    coherence = _clamp01(nonce_snapshot.get("coherence_peak", 0.0))
    entropy = _clamp01(nonce_snapshot.get("entropy_score", 0.0))
    valid_ratio = _clamp01(nonce_snapshot.get("valid_ratio", 0.0))
    axis_scale_x = _clamp01(nonce_snapshot.get("axis_scale_x", 0.0))
    axis_scale_y = _clamp01(nonce_snapshot.get("axis_scale_y", 0.0))
    axis_scale_z = _clamp01(nonce_snapshot.get("axis_scale_z", 0.0))
    vector_energy = _clamp01(nonce_snapshot.get("vector_energy", 0.0))
    temporal_coupling = _clamp01(nonce_snapshot.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(nonce_snapshot.get("inertial_mass_proxy", 0.0))
    relativistic_correlation = _clamp01(nonce_snapshot.get("relativistic_correlation", 0.0))
    spin_score = _clamp01(nonce_snapshot.get("spin_momentum_score", 0.0))
    temporal_relativity_norm = _clamp01(nonce_snapshot.get("temporal_relativity_norm", 0.0))
    gpu_pulse_phase_effect = _clamp_signed(nonce_snapshot.get("gpu_pulse_phase_effect", nonce_snapshot.get("phase_injection_delta_turns", 0.0)), 0.5)
    phase_ring_strength = _clamp01(nonce_snapshot.get("phase_ring_strength", 0.0))
    shared_phase_lock = _clamp01(nonce_snapshot.get("shared_vector_phase_lock", 0.0))
    zero_point_crossover = _clamp01(nonce_snapshot.get("zero_point_crossover_gate", 0.0))
    resonant_interception_inertia = _clamp01(nonce_snapshot.get("resonant_interception_inertia", 0.0))
    vector_scale_x, vector_scale_x_weights = _derived_temporal_mix({
        "axis_scale": axis_scale_x,
        "phase_ring": phase_ring_strength,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity_norm,
    })
    vector_scale_y, vector_scale_y_weights = _derived_temporal_mix({
        "axis_scale": axis_scale_y,
        "phase_ring": phase_ring_strength,
        "shared_phase_lock": shared_phase_lock,
        "temporal_relativity": temporal_relativity_norm,
    })
    vector_scale_z, vector_scale_z_weights = _derived_temporal_mix({
        "axis_scale": axis_scale_z,
        "phase_ring": phase_ring_strength,
        "zero_point": zero_point_crossover,
        "temporal_relativity": temporal_relativity_norm,
        "interception": resonant_interception_inertia,
    })
    temporal_constants = _derive_temporal_constant_set(
        field_dynamics=nonce_snapshot,
        phase_turns=phase,
        gpu_util=_clamp01(system_payload.get("gpu_util", 0.0)),
        mem_bw_util=_clamp01(system_payload.get("mem_bw_util", 0.0)),
        cpu_util=_clamp01(system_payload.get("cpu_util", 0.0)),
        global_util=_clamp01(system_payload.get("global_util", 0.0)),
        headroom=_clamp01(1.0 - max(_clamp01(system_payload.get("gpu_util", 0.0)), _clamp01(system_payload.get("mem_bw_util", 0.0)), _clamp01(system_payload.get("cpu_util", 0.0)), _clamp01(system_payload.get("global_util", 0.0)))),
        latency_load=_clamp01(1.0 - valid_ratio),
        resonance=coherence,
        temporal_overlap=_clamp01(max((entropy + coherence) * 0.5, temporal_coupling)),
        flux_term=_clamp01(max(abs(atomic_x), abs(atomic_z))),
        persistence=entropy,
        leakage=_clamp01(1.0 - valid_ratio),
        actuation_gain=coherence,
        pulse_signal=vector_energy,
        coherence=coherence,
        entropy=entropy,
        valid_ratio=valid_ratio,
        observer_feedback=_clamp01(abs(_safe_float(nonce_snapshot.get("psi", 0.0), 0.0))),
        noise_gate=_clamp01(1.0 - valid_ratio),
        phase_delta_turns=nonce_snapshot.get("phase_injection_delta_turns", gpu_pulse_phase_effect),
    )
    return {
        "simulation_field_vector": [
            atomic_x * float(vector_scale_x),
            atomic_y * float(vector_scale_y),
            atomic_z * float(vector_scale_z),
            phase,
        ],
        "feedback_axis_vector": _build_feedback_axis_vector(system_payload, nonce_snapshot),
        "sequence_persistence_score": float(temporal_constants.get("sequence_persistence_score", 0.0)),
        "temporal_index_overlap": float(temporal_constants.get("temporal_overlap_target", 0.0)),
        "voltage_frequency_flux": _clamp01(abs(atomic_z) * (0.5 + 0.5 * coherence)),
        "frequency_voltage_flux": _clamp01(abs(atomic_x) * (0.5 + 0.5 * entropy)),
        "field_alignment_score": float(temporal_constants.get("field_alignment_score", 0.0)),
        "kernel_control_gate": float(temporal_constants.get("kernel_control_gate", 0.0)),
        "feedback_phase_anchor_turns": phase,
        "axis_scale_x": axis_scale_x,
        "axis_scale_y": axis_scale_y,
        "axis_scale_z": axis_scale_z,
        "vector_energy": vector_energy,
        "temporal_coupling_moment": temporal_coupling,
        "inertial_mass_proxy": inertia,
        "relativistic_correlation": relativistic_correlation,
        "spin_momentum_score": spin_score,
        "gpu_pulse_phase_effect": float(gpu_pulse_phase_effect),
        "phase_injection_delta_turns": float(_clamp_signed(nonce_snapshot.get("phase_injection_delta_turns", gpu_pulse_phase_effect), 0.5)),
        "phase_injection_turns": float(_wrap_turns(nonce_snapshot.get("phase_injection_turns", phase))),
        "phase_ring_closure": float(_clamp01(nonce_snapshot.get("phase_ring_closure", 0.0))),
        "phase_ring_density": float(_clamp01(nonce_snapshot.get("phase_ring_density", 0.0))),
        "phase_ring_strength": float(_clamp01(nonce_snapshot.get("phase_ring_strength", 0.0))),
        "zero_point_crossover_gate": float(_clamp01(nonce_snapshot.get("zero_point_crossover_gate", 0.0))),
        "shared_vector_collapse_gate": float(_clamp01(nonce_snapshot.get("shared_vector_collapse_gate", 0.0))),
        "shared_vector_phase_lock": float(_clamp01(nonce_snapshot.get("shared_vector_phase_lock", 0.0))),
        "inertial_basin_strength": float(_clamp01(nonce_snapshot.get("inertial_basin_strength", 0.0))),
        "wavelength_norm": float(_clamp01(nonce_snapshot.get("wavelength_norm", 0.0))),
        "orientation_alignment": float(_clamp01(nonce_snapshot.get("orientation_alignment", 0.0))),
        "rotational_velocity_norm": float(_clamp01(nonce_snapshot.get("rotational_velocity_norm", 0.0))),
        "relative_temporal_position": float(_clamp01(nonce_snapshot.get("relative_temporal_position", 0.0))),
        "zero_point_line_distance": float(_clamp01(nonce_snapshot.get("zero_point_line_distance", 0.0))),
        "field_interference_norm": float(_clamp01(nonce_snapshot.get("field_interference_norm", 0.0))),
        "resonant_interception_inertia": float(_clamp01(nonce_snapshot.get("resonant_interception_inertia", 0.0))),
        "temporal_relativity_norm": float(temporal_relativity_norm),
        "temporal_relativity_vector": [float(_clamp01(v)) for v in list(nonce_snapshot.get("temporal_relativity_vector", []))[:4]],
        "phase_ring_zone_id": int(nonce_snapshot.get("phase_ring_zone_id", 0) or 0),
        "frequency_gradient_9d": [float(_clamp01(v)) for v in list(nonce_snapshot.get("frequency_gradient_9d", []))[:9]],
        "field_gradient_9d": [float(_clamp01(v)) for v in list(nonce_snapshot.get("field_gradient_9d", nonce_snapshot.get("frequency_gradient_9d", [])))[:9]],
        "gradient_spectral_id": str(nonce_snapshot.get("gradient_spectral_id", "")),
        "substrate_material": "silicon_wafer",
        "simulation_temporal_constant_weights": {
            "vector_scale_x": dict(vector_scale_x_weights),
            "vector_scale_y": dict(vector_scale_y_weights),
            "vector_scale_z": dict(vector_scale_z_weights),
            **{str(key): dict(value) for key, value in dict(temporal_constants.get("weights", {})).items()},
        },
    }


def _build_gpu_feedback(system_payload: Dict[str, Any], nonce_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    gpu_util = _clamp01(system_payload.get("gpu_util", 0.0))
    mem_bw_util = _clamp01(system_payload.get("mem_bw_util", 0.0))
    cpu_util = _clamp01(system_payload.get("cpu_util", 0.0))
    global_util = _clamp01(system_payload.get("global_util", 0.0))
    flux = abs(_safe_float(nonce_snapshot.get("flux", 0.0)))
    harmonic = abs(_safe_float(nonce_snapshot.get("harmonic", 0.0)))
    entropy = _clamp01(nonce_snapshot.get("entropy_score", 0.0))
    coherence = _clamp01(nonce_snapshot.get("coherence_peak", 0.0))
    phase = _clamp01(nonce_snapshot.get("phase", 0.0))
    temporal_coupling = _clamp01(nonce_snapshot.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(nonce_snapshot.get("inertial_mass_proxy", 0.0))
    spin_score = _clamp01(nonce_snapshot.get("spin_momentum_score", 0.0))
    temporal_constants = _derive_temporal_constant_set(
        field_dynamics=nonce_snapshot,
        phase_turns=phase,
        gpu_util=gpu_util,
        mem_bw_util=mem_bw_util,
        cpu_util=cpu_util,
        global_util=global_util,
        headroom=_clamp01(1.0 - max(gpu_util, mem_bw_util, cpu_util, global_util)),
        latency_load=_clamp01(abs(mem_bw_util - gpu_util)),
        resonance=coherence,
        temporal_overlap=temporal_coupling,
        flux_term=flux,
        persistence=entropy,
        leakage=harmonic,
        actuation_gain=gpu_util,
        pulse_signal=temporal_coupling,
        coherence=coherence,
        entropy=entropy,
        valid_ratio=_clamp01(1.0 - abs(gpu_util - mem_bw_util)),
        observer_feedback=_clamp01(abs(_safe_float(nonce_snapshot.get("psi", 0.0), 0.0))),
        noise_gate=harmonic,
        phase_delta_turns=nonce_snapshot.get("phase_injection_delta_turns", nonce_snapshot.get("gpu_pulse_phase_effect", 0.0)),
    )
    temperature_norm, temperature_norm_weights = _derived_temporal_mix({
        "gpu_util": gpu_util,
        "memory": mem_bw_util,
        "phase_ring": _clamp01(nonce_snapshot.get("phase_ring_strength", 0.0)),
        "interference": _clamp01(nonce_snapshot.get("field_interference_norm", 0.0)),
    })
    environment_pressure, environment_pressure_weights = _derived_temporal_mix({
        "global_util": global_util,
        "gpu_util": gpu_util,
        "inertia": inertia,
        "temporal_relativity": _clamp01(nonce_snapshot.get("temporal_relativity_norm", 0.0)),
    })
    environment_stability, environment_stability_weights = _derived_temporal_mix({
        "gpu_cpu_alignment": 1.0 - abs(gpu_util - cpu_util),
        "shared_phase_lock": _clamp01(nonce_snapshot.get("shared_vector_phase_lock", 0.0)),
        "zero_point": _clamp01(nonce_snapshot.get("zero_point_crossover_gate", 0.0)),
        "temporal_relativity": _clamp01(nonce_snapshot.get("temporal_relativity_norm", 0.0)),
    })
    return {
        "feedback_axis_vector": _build_feedback_axis_vector(system_payload, nonce_snapshot),
        "feedback_dof_vector": _build_feedback_dof_vector(system_payload, nonce_snapshot),
        "phase_anchor_turns": phase,
        "phase_alignment": coherence,
        "memory_proxy": entropy,
        "flux_proxy": _clamp01(flux),
        "stability_proxy": _clamp01(1.0 - min(harmonic, 1.0)),
        "temporal_drive": float(temporal_constants.get("temporal_drive", 0.0)),
        "temperature_norm": float(temperature_norm),
        "environment_pressure": float(environment_pressure),
        "environment_stability": float(environment_stability),
        "latency_norm": float(temporal_constants.get("latency_norm", 0.0)),
        "spin_momentum_score": spin_score,
        "temporal_coupling_moment": temporal_coupling,
        "inertial_mass_proxy": inertia,
        "temporal_relativity_norm": _clamp01(nonce_snapshot.get("temporal_relativity_norm", 0.0)),
        "resonant_interception_inertia": _clamp01(nonce_snapshot.get("resonant_interception_inertia", 0.0)),
        "zero_point_line_distance": _clamp01(nonce_snapshot.get("zero_point_line_distance", 0.0)),
        "field_interference_norm": _clamp01(nonce_snapshot.get("field_interference_norm", 0.0)),
        "gpu_pulse_phase_effect": float(_clamp_signed(nonce_snapshot.get("gpu_pulse_phase_effect", nonce_snapshot.get("phase_injection_delta_turns", 0.0)), 0.5)),
        "phase_ring_strength": float(_clamp01(nonce_snapshot.get("phase_ring_strength", 0.0))),
        "shared_vector_phase_lock": float(_clamp01(nonce_snapshot.get("shared_vector_phase_lock", 0.0))),
        "inertial_basin_strength": float(_clamp01(nonce_snapshot.get("inertial_basin_strength", 0.0))),
        "feedback_temporal_constant_weights": {
            **{str(key): dict(value) for key, value in dict(temporal_constants.get("weights", {})).items()},
            "temperature_norm": dict(temperature_norm_weights),
            "environment_pressure": dict(environment_pressure_weights),
            "environment_stability": dict(environment_stability_weights),
        },
    }


def _build_gpu_delta_feedback(system_payload: Dict[str, Any], nonce_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    coherence = _clamp01(nonce_snapshot.get("coherence_peak", 0.0))
    entropy = _clamp01(nonce_snapshot.get("entropy_score", 0.0))
    valid_ratio = _clamp01(nonce_snapshot.get("valid_ratio", 0.0))
    atomic_x = _safe_float(nonce_snapshot.get("atomic_vector_x", 0.0))
    atomic_y = _safe_float(nonce_snapshot.get("atomic_vector_y", 0.0))
    atomic_z = _safe_float(nonce_snapshot.get("atomic_vector_z", 0.0))
    phase = _clamp01(nonce_snapshot.get("phase", 0.0))
    temporal_coupling = _clamp01(nonce_snapshot.get("temporal_coupling_moment", 0.0))
    inertia = _clamp01(nonce_snapshot.get("inertial_mass_proxy", 0.0))
    spin_score = _clamp01(nonce_snapshot.get("spin_momentum_score", 0.0))
    temporal_constants = _derive_temporal_constant_set(
        field_dynamics=nonce_snapshot,
        phase_turns=phase,
        gpu_util=_clamp01(system_payload.get("gpu_util", 0.0)),
        mem_bw_util=_clamp01(system_payload.get("mem_bw_util", 0.0)),
        cpu_util=_clamp01(system_payload.get("cpu_util", 0.0)),
        global_util=_clamp01(system_payload.get("global_util", 0.0)),
        headroom=_clamp01(1.0 - max(_clamp01(system_payload.get("gpu_util", 0.0)), _clamp01(system_payload.get("mem_bw_util", 0.0)), _clamp01(system_payload.get("cpu_util", 0.0)), _clamp01(system_payload.get("global_util", 0.0)))),
        latency_load=_clamp01(abs(_clamp01(system_payload.get("gpu_util", 0.0)) - _clamp01(system_payload.get("mem_bw_util", 0.0)))),
        resonance=coherence,
        temporal_overlap=temporal_coupling,
        flux_term=_clamp01(abs(atomic_z)),
        persistence=entropy,
        leakage=_clamp01(1.0 - valid_ratio),
        actuation_gain=coherence,
        pulse_signal=spin_score,
        coherence=coherence,
        entropy=entropy,
        valid_ratio=valid_ratio,
        observer_feedback=_clamp01(abs(_safe_float(nonce_snapshot.get("psi", 0.0), 0.0))),
        noise_gate=_clamp01(1.0 - valid_ratio),
        phase_delta_turns=nonce_snapshot.get("phase_injection_delta_turns", nonce_snapshot.get("gpu_pulse_phase_effect", 0.0)),
    )
    return {
        "delta_target_vector": [atomic_x, atomic_y, atomic_z, phase],
        "phase_shift_turns": phase,
        "observation_freshness_gate": float(temporal_constants.get("observation_freshness_gate", 0.0)),
        "response_gate": float(temporal_constants.get("response_gate", 0.0)),
        "response_energy": float(temporal_constants.get("response_energy", 0.0)),
        "memory_retention": entropy,
        "latency_norm": float(temporal_constants.get("latency_norm", 0.0)),
        "sequence_persistence_target": float(temporal_constants.get("sequence_persistence_score", 0.0)),
        "temporal_overlap_target": float(temporal_constants.get("temporal_overlap_target", 0.0)),
        "voltage_frequency_flux_target": _clamp01(abs(atomic_z)),
        "frequency_voltage_flux_target": _clamp01(abs(atomic_x)),
        "temporal_coupling_moment": temporal_coupling,
        "inertial_mass_proxy": inertia,
        "spin_momentum_score": spin_score,
        "temporal_relativity_norm": float(_clamp01(nonce_snapshot.get("temporal_relativity_norm", 0.0))),
        "resonant_interception_inertia": float(_clamp01(nonce_snapshot.get("resonant_interception_inertia", 0.0))),
        "zero_point_line_distance": float(_clamp01(nonce_snapshot.get("zero_point_line_distance", 0.0))),
        "field_interference_norm": float(_clamp01(nonce_snapshot.get("field_interference_norm", 0.0))),
        "gpu_pulse_phase_effect": float(_clamp_signed(nonce_snapshot.get("gpu_pulse_phase_effect", nonce_snapshot.get("phase_injection_delta_turns", 0.0)), 0.5)),
        "phase_ring_strength": float(_clamp01(nonce_snapshot.get("phase_ring_strength", 0.0))),
        "zero_point_crossover_gate": float(_clamp01(nonce_snapshot.get("zero_point_crossover_gate", 0.0))),
        "shared_vector_collapse_gate": float(_clamp01(nonce_snapshot.get("shared_vector_collapse_gate", 0.0))),
        "delta_temporal_constant_weights": {
            **{str(key): dict(value) for key, value in dict(temporal_constants.get("weights", {})).items()},
        },
    }


def _update_substrate_trace_state(
    pulse_index: int,
    previous_trace_state: Dict[str, Any],
    simulation_field_state: Dict[str, Any],
    gpu_feedback: Dict[str, Any],
    gpu_pulse_delta_feedback: Dict[str, Any],
    interference_field: Dict[str, Any],
    effective_vector: Dict[str, Any],
    kernel_execution_event: Dict[str, Any],
    trace_label: str,
) -> Dict[str, Any]:
    prev = dict(previous_trace_state or {})
    seed = _stable_seed(trace_label)
    sim_vec = _vector_list(simulation_field_state.get("simulation_field_vector", []), 4)
    axis_vec = _vector_list(gpu_feedback.get("feedback_axis_vector", []), 4)
    dof_vec = _vector_list(gpu_feedback.get("feedback_dof_vector", []), 10)
    delta_vec = _vector_list(gpu_pulse_delta_feedback.get("delta_target_vector", []), 4)
    dominant_vec = _vector_list(
        dict(interference_field.get("dominant_vector", {}) or {}).get("vector", []),
        4,
    )
    field_alignment = _clamp01(
        kernel_execution_event.get(
            "field_alignment_score",
            simulation_field_state.get("field_alignment_score", 0.0),
        )
    )
    kernel_gate = _clamp01(
        kernel_execution_event.get(
            "kernel_control_gate",
            simulation_field_state.get("kernel_control_gate", 0.0),
        )
    )
    field_resonance = _clamp01(interference_field.get("field_resonance", 0.0))
    target_alignment = _clamp01(
        dict(interference_field.get("dominant_vector", {}) or {}).get("target_alignment", 0.0)
    )
    temporal_overlap = _clamp01(simulation_field_state.get("temporal_index_overlap", 0.0))
    sequence_persistence = _clamp01(simulation_field_state.get("sequence_persistence_score", 0.0))
    voltage_frequency_flux = _clamp01(simulation_field_state.get("voltage_frequency_flux", 0.0))
    frequency_voltage_flux = _clamp01(simulation_field_state.get("frequency_voltage_flux", 0.0))
    response_gate = _clamp01(gpu_pulse_delta_feedback.get("response_gate", 0.0))
    observation_freshness = _clamp01(gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0))
    gpu_pulse_phase_effect = _clamp_signed(
        simulation_field_state.get(
            "gpu_pulse_phase_effect",
            gpu_feedback.get("gpu_pulse_phase_effect", gpu_pulse_delta_feedback.get("gpu_pulse_phase_effect", 0.0)),
        ),
        0.5,
    )
    phase_ring_closure = _clamp01(simulation_field_state.get("phase_ring_closure", 0.0))
    phase_ring_density = _clamp01(simulation_field_state.get("phase_ring_density", 0.0))
    phase_ring_strength = _clamp01(
        simulation_field_state.get("phase_ring_strength", gpu_feedback.get("phase_ring_strength", 0.0))
    )
    zero_point_crossover_gate = _clamp01(
        simulation_field_state.get("zero_point_crossover_gate", interference_field.get("zero_point_crossover_gate", gpu_pulse_delta_feedback.get("zero_point_crossover_gate", 0.0)))
    )
    shared_vector_collapse_gate = _clamp01(
        simulation_field_state.get("shared_vector_collapse_gate", interference_field.get("shared_vector_collapse_gate", gpu_pulse_delta_feedback.get("shared_vector_collapse_gate", 0.0)))
    )
    shared_vector_phase_lock = _clamp01(
        simulation_field_state.get("shared_vector_phase_lock", interference_field.get("shared_vector_phase_lock", gpu_feedback.get("shared_vector_phase_lock", 0.0)))
    )
    inertial_basin_strength = _clamp01(
        simulation_field_state.get("inertial_basin_strength", gpu_feedback.get("inertial_basin_strength", 0.0))
    )
    temporal_relativity_norm = _clamp01(simulation_field_state.get("temporal_relativity_norm", gpu_feedback.get("temporal_relativity_norm", 0.0)))
    zero_point_line_distance = _clamp01(simulation_field_state.get("zero_point_line_distance", gpu_feedback.get("zero_point_line_distance", 0.0)))
    field_interference_norm = _clamp01(simulation_field_state.get("field_interference_norm", gpu_feedback.get("field_interference_norm", gpu_pulse_delta_feedback.get("field_interference_norm", 0.0))))
    resonant_interception_inertia = _clamp01(simulation_field_state.get("resonant_interception_inertia", gpu_feedback.get("resonant_interception_inertia", gpu_pulse_delta_feedback.get("resonant_interception_inertia", 0.0))))
    frequency_gradient_9d = [
        _clamp01(v) for v in list(simulation_field_state.get("frequency_gradient_9d", simulation_field_state.get("field_gradient_9d", [])))[:9]
    ]
    while len(frequency_gradient_9d) < 9:
        frequency_gradient_9d.append(0.0)
    alignment_cross = _dominant_alignment(sim_vec, dominant_vec)
    axis_alignment = _dominant_alignment(sim_vec, axis_vec)
    delta_alignment = _dominant_alignment(sim_vec, delta_vec)
    vector_energy = _vector_energy(sim_vec)
    axis_energy = _vector_energy(axis_vec)
    dof_energy = _vector_energy(dof_vec)

    support_raw, support_weights = _derived_temporal_mix({
        "field_alignment": field_alignment,
        "sequence_persistence": sequence_persistence,
        "target_alignment": target_alignment,
        "kernel_gate": kernel_gate,
        "vector_energy": vector_energy,
        "axis_alignment": axis_alignment,
        "phase_ring_strength": phase_ring_strength,
        "inertial_basin": inertial_basin_strength,
        "temporal_relativity": temporal_relativity_norm,
        "interception_inertia": resonant_interception_inertia,
    })
    resonance_raw, resonance_weights = _derived_temporal_mix({
        "field_resonance": field_resonance,
        "axis_energy": axis_energy,
        "dof_energy": dof_energy,
        "response_gate": response_gate,
        "observation_freshness": observation_freshness,
        "phase_ring_closure": phase_ring_closure,
        "shared_phase_lock": shared_vector_phase_lock,
        "zero_point_alignment": 1.0 - zero_point_line_distance,
    })
    alignment_raw, alignment_weights = _derived_temporal_mix({
        "target_alignment": target_alignment,
        "field_alignment": field_alignment,
        "alignment_cross": alignment_cross,
        "delta_alignment": delta_alignment,
        "temporal_overlap": temporal_overlap,
        "zero_point_crossover": zero_point_crossover_gate,
        "collapse_gate": shared_vector_collapse_gate,
        "interference_alignment": 1.0 - field_interference_norm,
    })
    memory_raw, memory_weights = _derived_temporal_mix({
        "sequence_persistence": sequence_persistence,
        "observation_freshness": observation_freshness,
        "target_alignment": target_alignment,
        "memory_history": _clamp01(prev.get("trace_memory", 0.0)),
        "temporal_relativity": temporal_relativity_norm,
    })
    flux_raw, flux_weights = _derived_temporal_mix({
        "voltage_frequency_flux": voltage_frequency_flux,
        "frequency_voltage_flux": frequency_voltage_flux,
        "feedback_flux": _clamp01(gpu_feedback.get("flux_proxy", 0.0)),
        "response_gate": response_gate,
        "phase_ring_density": phase_ring_density,
        "zero_point_alignment": 1.0 - zero_point_line_distance,
    })
    persistence_raw, persistence_weights = _derived_temporal_mix({
        "sequence_persistence": sequence_persistence,
        "memory": memory_raw,
        "field_resonance": field_resonance,
        "kernel_gate": kernel_gate,
        "inertial_basin": inertial_basin_strength,
        "interception_inertia": resonant_interception_inertia,
    })
    overlap_raw, overlap_weights = _derived_temporal_mix({
        "temporal_overlap": temporal_overlap,
        "alignment": alignment_raw,
        "support": support_raw,
        "response_gate": response_gate,
        "shared_phase_lock": shared_vector_phase_lock,
        "temporal_relativity": temporal_relativity_norm,
    })

    anchor_base = _clamp01(gpu_feedback.get("phase_anchor_turns", simulation_field_state.get("feedback_phase_anchor_turns", 0.0)))
    anchor_jitter = ((seed + int(pulse_index)) % 97) / 97.0
    phase_anchor = (
        _safe_float(prev.get("trace_phase_anchor_turns", anchor_base), anchor_base) * 0.62
        + anchor_base * 0.28
        + anchor_jitter * 0.10
    ) % 1.0

    trace_state = {
        "pulse_index": int(pulse_index),
        "trace_label": str(trace_label),
        "trace_support": _lerp(prev.get("trace_support", support_raw), support_raw, 0.42),
        "trace_resonance": _lerp(prev.get("trace_resonance", resonance_raw), resonance_raw, 0.44),
        "trace_alignment": _lerp(prev.get("trace_alignment", alignment_raw), alignment_raw, 0.46),
        "trace_memory": _lerp(prev.get("trace_memory", memory_raw), memory_raw, 0.34),
        "trace_flux": _lerp(prev.get("trace_flux", flux_raw), flux_raw, 0.38),
        "trace_temporal_persistence": _lerp(prev.get("trace_temporal_persistence", persistence_raw), persistence_raw, 0.36),
        "trace_temporal_overlap": _lerp(prev.get("trace_temporal_overlap", overlap_raw), overlap_raw, 0.36),
        "trace_voltage_frequency_flux": _lerp(prev.get("trace_voltage_frequency_flux", voltage_frequency_flux), voltage_frequency_flux, 0.34),
        "trace_frequency_voltage_flux": _lerp(prev.get("trace_frequency_voltage_flux", frequency_voltage_flux), frequency_voltage_flux, 0.34),
        "trace_phase_anchor_turns": float(phase_anchor),
        "trace_vector": [
            _clamp01((sim_vec[0] + 1.0) * 0.5),
            _clamp01((sim_vec[1] + 1.0) * 0.5),
            _clamp01((sim_vec[2] + 1.0) * 0.5),
            _clamp01(field_alignment),
        ],
        "trace_axis_vector": [
            _clamp01(axis_vec[0]),
            _clamp01(axis_vec[1]),
            _clamp01(axis_vec[2]),
            _clamp01(axis_vec[3]),
        ],
        "trace_dof_vector": [_clamp01(v) for v in dof_vec],
        "trace_trajectory_9d": [
            _clamp01(v) for v in list(simulation_field_state.get("trajectory_9d", []))[:9]
        ],
        "trace_predicted_trajectory_9d": [
            _clamp01(v) for v in list(simulation_field_state.get("predicted_trajectory_9d", []))[:9]
        ],
        "trace_frequency_gradient_9d": list(frequency_gradient_9d),
        "trace_gradient_spectral_id": str(simulation_field_state.get("gradient_spectral_id", "")),
        "trace_trajectory_spectral_id": str(simulation_field_state.get("trajectory_spectral_id", "")),
        "trace_predicted_trajectory_spectral_id": str(simulation_field_state.get("predicted_trajectory_spectral_id", "")),
        "trace_trajectory_conservation_alignment": _clamp01(simulation_field_state.get("trajectory_conservation_alignment", 0.0)),
        "trace_trajectory_prediction_alignment": _clamp01(simulation_field_state.get("trajectory_prediction_alignment", 0.0)),
        "trace_trajectory_expansion_term": _clamp01(simulation_field_state.get("trajectory_expansion_term", 0.0)),
        "trace_gpu_pulse_phase_effect": float(gpu_pulse_phase_effect),
        "trace_phase_ring_closure": float(phase_ring_closure),
        "trace_phase_ring_density": float(phase_ring_density),
        "trace_phase_ring_strength": float(phase_ring_strength),
        "trace_zero_point_crossover": float(zero_point_crossover_gate),
        "trace_shared_vector_collapse": float(shared_vector_collapse_gate),
        "trace_shared_vector_phase_lock": float(shared_vector_phase_lock),
        "trace_inertial_basin_strength": float(inertial_basin_strength),
        "trace_temporal_relativity_norm": float(temporal_relativity_norm),
        "trace_zero_point_line_distance": float(zero_point_line_distance),
        "trace_field_interference_norm": float(field_interference_norm),
        "trace_resonant_interception_inertia": float(resonant_interception_inertia),
        "trace_temporal_constant_weights": {
            "support": dict(support_weights),
            "resonance": dict(resonance_weights),
            "alignment": dict(alignment_weights),
            "memory": dict(memory_weights),
            "flux": dict(flux_weights),
            "persistence": dict(persistence_weights),
            "overlap": dict(overlap_weights),
        },
        "effective_vector": {
            "x": _safe_float(effective_vector.get("x", 0.0)),
            "y": _safe_float(effective_vector.get("y", 0.0)),
            "z": _safe_float(effective_vector.get("z", 0.0)),
            "t_eff": _clamp01(effective_vector.get("t_eff", 0.0)),
        },
        "trace_material": str(simulation_field_state.get("substrate_material", "silicon_wafer")),
    }
    trace_gate = max(
        float(trace_state["trace_support"]),
        float(trace_state["trace_resonance"]),
        float(trace_state["trace_alignment"]),
        float(trace_state["trace_memory"]),
        float(trace_state["trace_flux"]),
        float(trace_state["trace_temporal_persistence"]),
        float(trace_state["trace_temporal_overlap"]),
        float(trace_state["trace_voltage_frequency_flux"]),
        float(trace_state["trace_frequency_voltage_flux"]),
        float(trace_state["trace_phase_ring_strength"]),
        float(trace_state["trace_shared_vector_phase_lock"]),
        float(trace_state["trace_temporal_relativity_norm"]),
        max([abs(v) for v in trace_state["trace_axis_vector"]] or [0.0]),
    )
    trace_state["trace_gate"] = _clamp01(trace_gate)
    trace_state["feedback_weight"] = _clamp01(0.18 + 0.72 * trace_state["trace_gate"])
    trace_state["scan_boost"] = int(
        round(
            24.0
            + 48.0 * float(trace_state["trace_support"])
            + 56.0 * float(trace_state["trace_alignment"])
            + 64.0 * float(trace_state["trace_temporal_persistence"])
        )
    )
    trace_state["update_count"] = int(prev.get("update_count", 0) or 0) + 1
    return trace_state


def _sync_substrate_trace_state_to_vram(trace_state: Dict[str, Any]) -> Dict[str, Any]:
    label = str(trace_state.get("trace_label", "lane"))
    with _LOCK:
        update_count = int(trace_state.get("update_count", 0) or 0)
        _TRACE_VRAM_CACHE[label] = {
            "resident": True,
            "reason": "runtime_trace_cache",
            "update_count": update_count,
            "trace_gate": float(trace_state.get("trace_gate", 0.0)),
            "trace_alignment": float(trace_state.get("trace_alignment", 0.0)),
            "trace_support": float(trace_state.get("trace_support", 0.0)),
            "trace_phase_ring_strength": float(trace_state.get("trace_phase_ring_strength", 0.0)),
            "trace_shared_vector_collapse": float(trace_state.get("trace_shared_vector_collapse", 0.0)),
            "trace_zero_point_crossover": float(trace_state.get("trace_zero_point_crossover", 0.0)),
            "trace_temporal_relativity_norm": float(trace_state.get("trace_temporal_relativity_norm", 0.0)),
            "trace_field_interference_norm": float(trace_state.get("trace_field_interference_norm", 0.0)),
        }
        return dict(_TRACE_VRAM_CACHE[label])


def build_substrate_trace_runtime(
    lane_id: str,
    tick: int,
    system_payload: Dict[str, Any],
    nonce_snapshot: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
    sync_vram: bool = True,
    packet: Any | None = None,
    runtime_payload: Dict[str, Any] | None = None,
    previous_profile: Dict[str, Any] | None = None,
    frame_history: list[Dict[str, Any]] | None = None,
    previous_memory_basin_state: Dict[str, Any] | None = None,
    previous_scheduler_state: Dict[str, Any] | None = None,
    previous_process_state: Dict[str, Any] | None = None,
    live_cycle: Any | None = None,
) -> Dict[str, Any]:
    runtime_map = dict(runtime_payload or {})
    prior_trace_state = dict(previous_trace_state or {})
    cycle_enabled = _flag_enabled(
        live_cycle if live_cycle is not None else runtime_map.get("enable_live_substrate_cycle", runtime_map.get("enable_live_gpu_pulse", True)),
        True,
    )
    cycle_result: Dict[str, Any] = {}
    if cycle_enabled:
        cycle_payload = dict(runtime_map)
        cycle_payload["lane_id"] = str(lane_id)
        cycle_payload["tick"] = int(tick)
        cycle_payload["packet"] = packet
        cycle_payload["nonce_snapshot"] = dict(nonce_snapshot or {})
        cycle_payload.setdefault("program_id", "miner_live_cycle_%s" % str(lane_id))
        cycle_payload.setdefault("telemetry_mode", str(runtime_map.get("telemetry_mode", "live_startup") or "live_startup"))
        cycle_payload.setdefault("use_live_telemetry", True)
        cycle_payload.setdefault("capture_sleep", False)
        cycle_payload.setdefault("pulse_cycles", 1)
        cycle_payload.setdefault("horizon_frames", max(1, int(runtime_map.get("telemetry_horizon_frames", runtime_map.get("horizon_frames", 2)) or 2)))
        cycle_payload.setdefault("history_size", max(4, int(runtime_map.get("telemetry_history_size", runtime_map.get("history_size", 8)) or 8)))
        cycle_payload.setdefault(
            "telemetry_sample_period_s",
            max(0.00025, _safe_float(runtime_map.get("telemetry_sample_period_s", runtime_map.get("tick_s", system_payload.get("tick_s", 0.004))), 0.004)),
        )
        cycle_payload.setdefault(
            "actuation_backend",
            str(runtime_map.get("actuation_backend", system_payload.get("actuation_backend", runtime_map.get("gpu_pulse_backend", ""))) or "").strip(),
        )
        for key in (
            "global_util",
            "gpu_util",
            "mem_bw_util",
            "cpu_util",
            "phase_step",
            "alpha_flux",
            "flux_coeff",
            "drift_coeff",
        ):
            if key in system_payload and key not in cycle_payload:
                cycle_payload[key] = system_payload.get(key)
        if "_snapshot_provider" not in cycle_payload and callable(runtime_map.get("_snapshot_provider")):
            cycle_payload["_snapshot_provider"] = runtime_map.get("_snapshot_provider")
        if "_actuation_hook" not in cycle_payload:
            if callable(runtime_map.get("_actuation_hook")):
                cycle_payload["_actuation_hook"] = runtime_map.get("_actuation_hook")
            else:
                cycle_payload["_actuation_hook"] = _runtime_packet_actuation
        try:
            cycle_result = run_live_photonic_substrate_cycle(
                cycle_payload,
                previous_trace_state=prior_trace_state,
                previous_profile=dict(previous_profile or {}),
                frame_history=[dict(item or {}) for item in list(frame_history or [])],
                previous_memory_basin_state=dict(previous_memory_basin_state or {}),
                previous_scheduler_state=dict(previous_scheduler_state or {}),
                previous_process_state=dict(previous_process_state or {}),
            )
        except Exception as exc:
            cycle_result = {
                "ok": False,
                "path": "live_photonic_cycle_failed",
                "error": str(exc),
                "trace_state": dict(prior_trace_state),
                "frames": [dict(item or {}) for item in list(frame_history or [])],
                "profile": dict(previous_profile or {}),
                "actuation_summary": {"applied": False, "call_count": 0, "applied_count": 0, "mode": "error"},
            }

    working_previous = dict(cycle_result.get("trace_state", prior_trace_state) or prior_trace_state)
    snapshot_seed = dict(nonce_snapshot or {})
    snapshot_map = dict(snapshot_seed)
    cycle_latency = dict(dict(cycle_result.get("result", {}) or {}).get("latency_calibration", {}) or {})
    cycle_trace = dict(cycle_result.get("trace_state", {}) or {})
    for src in (cycle_latency, cycle_trace):
        for key in (
            "gpu_pulse_phase_effect",
            "phase_injection_delta_turns",
            "phase_injection_turns",
            "phase_ring_closure",
            "phase_ring_density",
            "phase_ring_strength",
            "zero_point_crossover_gate",
            "shared_vector_collapse_gate",
            "shared_vector_phase_lock",
            "inertial_basin_strength",
            "wavelength_norm",
            "orientation_alignment",
            "rotational_velocity_norm",
            "relative_temporal_position",
            "zero_point_line_distance",
            "field_interference_norm",
            "resonant_interception_inertia",
            "temporal_relativity_norm",
            "trajectory_conservation_alignment",
            "trajectory_prediction_alignment",
            "trajectory_expansion_term",
            "gradient_spectral_id",
            "trajectory_spectral_id",
            "predicted_trajectory_spectral_id",
        ):
            if key in src:
                snapshot_map[key] = src.get(key)
    if "frequency_gradient_9d" not in snapshot_map:
        snapshot_map["frequency_gradient_9d"] = list(
            cycle_latency.get("frequency_gradient_9d", cycle_trace.get("trace_frequency_gradient_9d", [])) or []
        )[:9]
    if "field_gradient_9d" not in snapshot_map:
        snapshot_map["field_gradient_9d"] = list(
            cycle_latency.get("field_gradient_9d", snapshot_map.get("frequency_gradient_9d", [])) or []
        )[:9]

    simulation_field_state = _build_simulation_field_state(system_payload, snapshot_map)
    gpu_feedback = _build_gpu_feedback(system_payload, snapshot_map)
    gpu_delta_feedback = _build_gpu_delta_feedback(system_payload, snapshot_map)
    effective_vector = {
        "x": _safe_float(snapshot_map.get("atomic_vector_x", 0.0)),
        "y": _safe_float(snapshot_map.get("atomic_vector_y", 0.0)),
        "z": _safe_float(snapshot_map.get("atomic_vector_z", 0.0)),
        "t_eff": _clamp01(snapshot_map.get("phase", 0.0)),
    }
    interference_field = {
        "field_resonance": _clamp01(snapshot_map.get("coherence_peak", 0.0)),
        "dominant_vector": {
            "vector": [
                float(effective_vector["x"]),
                float(effective_vector["y"]),
                float(effective_vector["z"]),
                float(effective_vector["t_eff"]),
            ],
            "latent_vector": list(simulation_field_state.get("simulation_field_vector", [0.0, 0.0, 0.0, 0.0])),
            "resonance": _clamp01(snapshot_map.get("coherence_peak", 0.0)),
            "target_alignment": _clamp01(snapshot_map.get("valid_ratio", 0.0)),
        },
    }
    kernel_execution_event = {
        "feedback_phase_anchor_turns": _clamp01(snapshot_map.get("phase", 0.0)),
        "field_alignment_score": float(simulation_field_state.get("field_alignment_score", 0.0)),
        "kernel_control_gate": float(simulation_field_state.get("kernel_control_gate", 0.0)),
        "substrate_material": "silicon_wafer",
    }
    if snapshot_seed:
        trace_state = _update_substrate_trace_state(
            pulse_index=int(tick),
            previous_trace_state=working_previous,
            simulation_field_state=simulation_field_state,
            gpu_feedback=gpu_feedback,
            gpu_pulse_delta_feedback=gpu_delta_feedback,
            interference_field=interference_field,
            effective_vector=effective_vector,
            kernel_execution_event=kernel_execution_event,
            trace_label=str(lane_id),
        )
    else:
        trace_state = dict(working_previous)

    if sync_vram:
        trace_vram = _sync_substrate_trace_state_to_vram(trace_state)
    else:
        trace_vram = {"resident": False, "reason": "sync_disabled", "update_count": 0}

    return {
        "active": True,
        "error": str(cycle_result.get("error", "")),
        "path": str(cycle_result.get("path", "runtime_snapshot") if cycle_result else "runtime_snapshot"),
        "trace_state": dict(trace_state or {}),
        "trace_vram": dict(trace_vram or {}),
        "simulation_field_state": simulation_field_state,
        "gpu_feedback": gpu_feedback,
        "gpu_delta_feedback": gpu_delta_feedback,
        "effective_vector": effective_vector,
        "kernel_execution_event": kernel_execution_event,
        "frame": dict(cycle_result.get("frame", {}) or {}),
        "frames": [dict(item or {}) for item in list(cycle_result.get("frames", []) or [])],
        "profile": dict(cycle_result.get("profile", previous_profile or {}) or {}),
        "result": dict(cycle_result.get("result", {}) or {}),
        "memory_basin_state": dict(cycle_result.get("memory_basin_state", {}) or {}),
        "scheduler_state": dict(cycle_result.get("scheduler_state", {}) or {}),
        "process_state": dict(cycle_result.get("process_state", {}) or {}),
        "forecast_preview": [dict(item or {}) for item in list(cycle_result.get("forecast_preview", []) or [])],
        "actuation_summary": dict(cycle_result.get("actuation_summary", {"applied": False, "call_count": 0, "applied_count": 0, "mode": "passive"}) or {}),
    }


_MICROPROCESS_OPCODE_IDS = {
    "enum_switch": 0,
    "calc_mix": 1,
    "string_fold": 2,
    "struct_route": 3,
    "dispatch_gate": 4,
}

_MICROPROCESS_OPCODE_NAMES = {
    0: "enum_switch",
    1: "calc_mix",
    2: "string_fold",
    3: "struct_route",
    4: "dispatch_gate",
}

_MICROPROCESS_COEFFICIENTS = {
    0: (3, 5, 11, 0x13579BDF),
    1: (7, 9, 13, 0x2468ACE1),
    2: (5, 11, 17, 0x10293847),
    3: (9, 13, 19, 0x89ABCDEF),
    4: (11, 15, 23, 0x55AA55AA),
}


def _stable_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return str(value)
    return str(value or "")


def _stable_word(value: Any) -> int:
    text = _stable_text(value)
    seed = 2166136261
    for ch in text.encode("ascii", errors="ignore"):
        seed ^= ch
        seed = (seed * 16777619) & 0xFFFFFFFF
    return int(seed & 0xFFFFFFFF)


def _rotl32(value: int, shift: int) -> int:
    word = int(value) & 0xFFFFFFFF
    bits = int(shift) & 31
    if bits <= 0:
        return word
    return ((word << bits) | (word >> (32 - bits))) & 0xFFFFFFFF


def _word_axis(word: int) -> float:
    unit = float(int(word) & 0xFFFF) / 65535.0
    return (unit * 2.0) - 1.0


def _microprocess_opcode_id(opcode: Any) -> int:
    text = str(opcode or "enum_switch").strip().lower()
    return int(_MICROPROCESS_OPCODE_IDS.get(text, 0))


def _microprocess_coefficients(opcode_id: int) -> tuple[int, int, int, int]:
    return tuple(_MICROPROCESS_COEFFICIENTS.get(int(opcode_id), _MICROPROCESS_COEFFICIENTS[0]))


def _default_microprocess_artifacts(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    artifact_count = max(1, int(payload.get("artifact_count", 192) or 192))
    modules = list(payload.get(
        "module_names",
        [
            "miner.nonce_math",
            "VHW.compute_manager",
            "VHW.gpu_pulse_runtime",
            "neural_object",
        ],
    ) or [])
    enum_cases = list(payload.get(
        "enum_cases",
        ["SINGLE", "VECTOR", "BATCH", "CUSTOM", "GPU_VECTOR"],
    ) or [])
    opcode_names = list(payload.get(
        "opcodes",
        ["enum_switch", "calc_mix", "string_fold", "struct_route", "dispatch_gate"],
    ) or [])
    if not modules:
        modules = ["runtime.default"]
    if not enum_cases:
        enum_cases = ["DEFAULT"]
    if not opcode_names:
        opcode_names = ["enum_switch"]
    artifacts: list[Dict[str, Any]] = []
    for idx in range(artifact_count):
        module = str(modules[idx % len(modules)])
        enum_case = str(enum_cases[(idx * 3) % len(enum_cases)])
        opcode = str(opcode_names[idx % len(opcode_names)])
        calc_a = ((_stable_word(module) >> (idx % 13)) ^ (idx * 97)) & 0xFFFFFFFF
        calc_b = ((_stable_word(enum_case) >> (idx % 11)) ^ (idx * 131)) & 0xFFFFFFFF
        fields = {
            "module_index": idx % max(1, len(modules)),
            "enum_index": (idx * 3) % max(1, len(enum_cases)),
            "lane": idx % 8,
            "bucket": (idx * 7) % 31,
            "flags": (idx ^ 0x5A) & 0xFF,
        }
        artifacts.append({
            "opcode": opcode,
            "module": module,
            "enum_case": enum_case,
            "text": module + "::" + enum_case + "::artifact_" + str(idx),
            "calc_a": int(calc_a),
            "calc_b": int(calc_b),
            "fields": fields,
        })
    return artifacts


def _prepare_microprocess_artifacts(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    artifacts = list(payload.get("artifacts", []) or [])
    if not artifacts:
        return _default_microprocess_artifacts(payload)
    out: list[Dict[str, Any]] = []
    for idx, artifact in enumerate(artifacts):
        if isinstance(artifact, dict):
            item = dict(artifact)
        else:
            item = {"text": _stable_text(artifact)}
        item.setdefault("opcode", "enum_switch")
        item.setdefault("module", "runtime.module_" + str(idx % 4))
        item.setdefault("enum_case", "CASE_" + str(idx % 7))
        item.setdefault("text", str(item.get("module")) + "::" + str(item.get("enum_case")) + "::" + str(idx))
        item.setdefault("calc_a", (_stable_word(item.get("module")) ^ idx) & 0xFFFFFFFF)
        item.setdefault("calc_b", (_stable_word(item.get("enum_case")) ^ (idx * 17)) & 0xFFFFFFFF)
        if not isinstance(item.get("fields"), dict):
            item["fields"] = {
                "index": idx,
                "module_word": _stable_word(item.get("module")) & 0xFFFF,
            }
        out.append(item)
    return out


def _encode_microprocess_artifact(
    artifact: Dict[str, Any],
    opcode_id: int | None = None,
    coefficients: tuple[int, int, int, int] | None = None,
) -> Dict[str, Any]:
    opcode_text = str(artifact.get("opcode", "enum_switch")).strip().lower()
    op_id = int(opcode_id if opcode_id is not None else _microprocess_opcode_id(opcode_text))
    coeffs = tuple(coefficients or _microprocess_coefficients(op_id))
    module_text = _stable_text(artifact.get("module", "runtime.module"))
    enum_text = _stable_text(artifact.get("enum_case", "CASE_0"))
    text = _stable_text(artifact.get("text", module_text + "::" + enum_text))
    fields = artifact.get("fields", {})
    if not isinstance(fields, dict):
        fields = {"value": _stable_text(fields)}
    calc_a = int(artifact.get("calc_a", _stable_word(module_text)) or 0) & 0xFFFFFFFF
    calc_b = int(artifact.get("calc_b", _stable_word(enum_text)) or 0) & 0xFFFFFFFF
    module_word = _stable_word(module_text)
    enum_word = _stable_word(enum_text)
    text_word = _stable_word(text)
    struct_word = _stable_word(fields)
    field_count = len(fields)
    string_len = len(text)
    calc_word = (calc_a ^ calc_b ^ coeffs[3]) & 0xFFFFFFFF
    control_word = (
        text_word
        ^ struct_word
        ^ module_word
        ^ enum_word
        ^ calc_word
        ^ ((field_count & 0xFF) << 24)
        ^ ((string_len & 0xFF) << 16)
        ^ ((op_id & 0xFF) << 8)
        ^ (coeffs[0] & 0xFF)
    ) & 0xFFFFFFFF
    phase_anchor_turns = float(((control_word >> 8) & 0xFFFF) / 65535.0)
    phase_center = _clamp01(1.0 - abs(_phase_delta_turns(phase_anchor_turns, 0.5)) * 2.0)
    field_alignment, field_alignment_weights = _derived_temporal_mix({
        "text_struct_alignment": float((text_word ^ struct_word) & 0xFF) / 255.0,
        "phase_center": phase_center,
        "field_density": _clamp01(float(field_count) / 16.0),
        "string_density": _clamp01(float(string_len) / 64.0),
    })
    kernel_gate, kernel_gate_weights = _derived_temporal_mix({
        "control_gate": float((control_word >> 24) & 0xFF) / 255.0,
        "phase_center": phase_center,
        "field_density": _clamp01(float(field_count) / 16.0),
        "opcode_density": _clamp01(float(op_id + 1) / 8.0),
    })
    sequence_persistence, sequence_persistence_weights = _derived_temporal_mix({
        "module_density": float((module_word >> 16) & 0xFF) / 255.0,
        "field_density": _clamp01(float(field_count) / 16.0),
        "string_density": _clamp01(float(string_len) / 64.0),
        "phase_center": phase_center,
    })
    temporal_overlap, temporal_overlap_weights = _derived_temporal_mix({
        "enum_density": float((enum_word >> 8) & 0xFF) / 255.0,
        "phase_center": phase_center,
        "opcode_density": _clamp01(float(op_id + 1) / 8.0),
        "string_density": _clamp01(float(string_len) / 64.0),
    })
    vector_z = _word_axis(text_word ^ struct_word ^ calc_word)
    vector_t = _clamp01((float(string_len) + float(field_count) + float(op_id)) / 64.0)
    return {
        "opcode": opcode_text,
        "opcode_id": int(op_id),
        "opcode_name": str(_MICROPROCESS_OPCODE_NAMES.get(op_id, "enum_switch")),
        "coefficients": coeffs,
        "module_text": module_text,
        "enum_text": enum_text,
        "text": text,
        "fields": dict(fields),
        "calc_a": int(calc_a),
        "calc_b": int(calc_b),
        "module_word": int(module_word),
        "enum_word": int(enum_word),
        "text_word": int(text_word),
        "struct_word": int(struct_word),
        "calc_word": int(calc_word),
        "control_word": int(control_word),
        "field_count": int(field_count),
        "string_len": int(string_len),
        "phase_anchor_turns": float(phase_anchor_turns),
        "field_alignment_score": float(field_alignment),
        "kernel_control_gate": float(kernel_gate),
        "sequence_persistence_score": float(sequence_persistence),
        "temporal_index_overlap": float(temporal_overlap),
        "simulation_vector": [
            _word_axis(module_word ^ coeffs[0]),
            _word_axis(enum_word ^ coeffs[1]),
            float(vector_z),
            float(vector_t),
        ],
        "feedback_axis_vector": [
            _clamp01(float((module_word >> 24) & 0xFF) / 255.0),
            _clamp01(float((enum_word >> 16) & 0xFF) / 255.0),
            _clamp01(float((text_word >> 8) & 0xFF) / 255.0),
            float(phase_anchor_turns),
        ],
        "feedback_dof_vector": [
            _clamp01(float((module_word >> 24) & 0xFF) / 255.0),
            _clamp01(float((module_word >> 16) & 0xFF) / 255.0),
            _clamp01(float((module_word >> 8) & 0xFF) / 255.0),
            _clamp01(float(module_word & 0xFF) / 255.0),
            _clamp01(float((enum_word >> 24) & 0xFF) / 255.0),
            _clamp01(float((enum_word >> 16) & 0xFF) / 255.0),
            _clamp01(float((text_word >> 24) & 0xFF) / 255.0),
            _clamp01(float((struct_word >> 16) & 0xFF) / 255.0),
            _clamp01(float((calc_word >> 8) & 0xFF) / 255.0),
            _clamp01(float(control_word & 0xFF) / 255.0),
        ],
        "encoding_temporal_constant_weights": {
            "field_alignment": dict(field_alignment_weights),
            "kernel_control": dict(kernel_gate_weights),
            "sequence_persistence": dict(sequence_persistence_weights),
            "temporal_overlap": dict(temporal_overlap_weights),
        },
    }


def _classical_encode_microprocess_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    opcode_text = str(artifact.get("opcode", "enum_switch")).strip().lower()
    if opcode_text == "enum_switch":
        opcode_id = 0
    elif opcode_text == "calc_mix":
        opcode_id = 1
    elif opcode_text == "string_fold":
        opcode_id = 2
    elif opcode_text == "struct_route":
        opcode_id = 3
    elif opcode_text == "dispatch_gate":
        opcode_id = 4
    else:
        opcode_id = 0
    return _encode_microprocess_artifact(
        artifact=artifact,
        opcode_id=opcode_id,
        coefficients=_microprocess_coefficients(opcode_id),
    )


def _microprocess_trace_inputs(
    encoded_artifact: Dict[str, Any],
    pulse_index: int,
    program_label: str,
    previous_trace_state: Dict[str, Any],
) -> Dict[str, Any]:
    latency_calibration = dict(encoded_artifact.get("latency_calibration", {}) or {})
    sim_vec = list(encoded_artifact.get("simulation_vector", [0.0, 0.0, 0.0, 0.0]))
    axis_vec = list(encoded_artifact.get("feedback_axis_vector", [0.0, 0.0, 0.0, 0.0]))
    dof_vec = list(encoded_artifact.get("feedback_dof_vector", [0.0] * 10))
    target_alignment = _clamp01(float((encoded_artifact.get("control_word", 0) >> 12) & 0xFF) / 255.0)
    response_gate = _clamp01(float((encoded_artifact.get("control_word", 0) >> 20) & 0xFF) / 255.0)
    response_energy = _clamp01(float((encoded_artifact.get("calc_word", 0) >> 8) & 0xFF) / 255.0)
    interference_vector = [
        float(sim_vec[0]),
        float(sim_vec[1]),
        float(sim_vec[2]),
        float(encoded_artifact.get("phase_anchor_turns", 0.0)),
    ]
    effective_vector = {
        "x": float(sim_vec[0]),
        "y": float(sim_vec[1]),
        "z": float(sim_vec[2]),
        "t_eff": float(encoded_artifact.get("phase_anchor_turns", 0.0)),
    }
    return {
        "pulse_index": int(pulse_index),
        "previous_trace_state": dict(previous_trace_state or {}),
        "simulation_field_state": {
            "simulation_field_vector": interference_vector,
            "feedback_axis_vector": axis_vec,
            "sequence_persistence_score": float(encoded_artifact.get("sequence_persistence_score", 0.0)),
            "temporal_index_overlap": float(encoded_artifact.get("temporal_index_overlap", 0.0)),
            "voltage_frequency_flux": _clamp01(abs(float(sim_vec[2]))),
            "frequency_voltage_flux": _clamp01(abs(float(sim_vec[0]))),
            "field_alignment_score": float(encoded_artifact.get("field_alignment_score", 0.0)),
            "kernel_control_gate": float(encoded_artifact.get("kernel_control_gate", 0.0)),
            "feedback_phase_anchor_turns": float(encoded_artifact.get("phase_anchor_turns", 0.0)),
            "gpu_pulse_phase_effect": float(latency_calibration.get("gpu_pulse_phase_effect", 0.0)),
            "phase_injection_delta_turns": float(latency_calibration.get("phase_injection_delta_turns", 0.0)),
            "phase_injection_turns": float(latency_calibration.get("phase_injection_turns", encoded_artifact.get("phase_anchor_turns", 0.0))),
            "phase_ring_closure": float(latency_calibration.get("phase_ring_closure", 0.0)),
            "phase_ring_density": float(latency_calibration.get("phase_ring_density", 0.0)),
            "phase_ring_strength": float(latency_calibration.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": float(latency_calibration.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": float(latency_calibration.get("shared_vector_collapse_gate", 0.0)),
            "shared_vector_phase_lock": float(latency_calibration.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": float(latency_calibration.get("inertial_basin_strength", 0.0)),
            "wavelength_norm": float(latency_calibration.get("wavelength_norm", 0.0)),
            "orientation_alignment": float(latency_calibration.get("orientation_alignment", 0.0)),
            "rotational_velocity_norm": float(latency_calibration.get("rotational_velocity_norm", 0.0)),
            "relative_temporal_position": float(latency_calibration.get("relative_temporal_position", 0.0)),
            "zero_point_line_distance": float(latency_calibration.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": float(latency_calibration.get("field_interference_norm", 0.0)),
            "resonant_interception_inertia": float(latency_calibration.get("resonant_interception_inertia", 0.0)),
            "temporal_relativity_norm": float(latency_calibration.get("temporal_relativity_norm", 0.0)),
            "temporal_relativity_vector": list(latency_calibration.get("temporal_relativity_vector", []))[:4],
            "phase_ring_zone_id": int(latency_calibration.get("phase_ring_zone_id", 0) or 0),
            "frequency_gradient_9d": list(latency_calibration.get("frequency_gradient_9d", []))[:9],
            "field_gradient_9d": list(latency_calibration.get("field_gradient_9d", latency_calibration.get("frequency_gradient_9d", [])))[:9],
            "gradient_spectral_id": str(latency_calibration.get("gradient_spectral_id", "")),
            "substrate_material": "silicon_wafer",
        },
        "gpu_feedback": {
            "feedback_axis_vector": axis_vec,
            "feedback_dof_vector": dof_vec,
            "phase_anchor_turns": float(encoded_artifact.get("phase_anchor_turns", 0.0)),
            "phase_alignment": float(encoded_artifact.get("field_alignment_score", 0.0)),
            "memory_proxy": float(encoded_artifact.get("sequence_persistence_score", 0.0)),
            "flux_proxy": _clamp01(abs(float(sim_vec[2]))),
            "stability_proxy": _clamp01(1.0 - abs(float(sim_vec[2])) * 0.5),
            "temporal_drive": float(encoded_artifact.get("temporal_index_overlap", 0.0)),
            "temperature_norm": _clamp01((float(axis_vec[0]) + float(axis_vec[1])) * 0.5),
            "environment_pressure": _clamp01((float(axis_vec[0]) + float(axis_vec[2])) * 0.5),
            "environment_stability": _clamp01(1.0 - abs(float(axis_vec[0]) - float(axis_vec[2]))),
            "latency_norm": _clamp01(float(latency_calibration.get("actuation_horizon_frames", 0.0)) / 4.0),
            "kernel_request_s": float(latency_calibration.get("kernel_request_s", 0.0)),
            "pulse_generation_s": float(latency_calibration.get("pulse_generation_s", 0.0)),
            "actuation_compensation": float(latency_calibration.get("actuation_compensation", 0.0)),
            "gpu_pulse_phase_effect": float(latency_calibration.get("gpu_pulse_phase_effect", 0.0)),
            "phase_ring_strength": float(latency_calibration.get("phase_ring_strength", 0.0)),
            "shared_vector_phase_lock": float(latency_calibration.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": float(latency_calibration.get("inertial_basin_strength", 0.0)),
            "temporal_relativity_norm": float(latency_calibration.get("temporal_relativity_norm", 0.0)),
            "resonant_interception_inertia": float(latency_calibration.get("resonant_interception_inertia", 0.0)),
            "zero_point_line_distance": float(latency_calibration.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": float(latency_calibration.get("field_interference_norm", 0.0)),
        },
        "gpu_pulse_delta_feedback": {
            "delta_target_vector": interference_vector,
            "phase_shift_turns": float(encoded_artifact.get("phase_anchor_turns", 0.0)),
            "observation_freshness_gate": _clamp01(0.20 + 0.80 * float(axis_vec[3])),
            "response_gate": float(response_gate),
            "response_energy": float(response_energy),
            "memory_retention": float(encoded_artifact.get("sequence_persistence_score", 0.0)),
            "latency_norm": _clamp01(float(latency_calibration.get("latency_load", 0.0))),
            "sequence_persistence_target": float(encoded_artifact.get("sequence_persistence_score", 0.0)),
            "temporal_overlap_target": float(encoded_artifact.get("temporal_index_overlap", 0.0)),
            "voltage_frequency_flux_target": _clamp01(abs(float(sim_vec[2]))),
            "frequency_voltage_flux_target": _clamp01(abs(float(sim_vec[0]))),
            "gpu_pulse_phase_effect": float(latency_calibration.get("gpu_pulse_phase_effect", 0.0)),
            "phase_ring_strength": float(latency_calibration.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": float(latency_calibration.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": float(latency_calibration.get("shared_vector_collapse_gate", 0.0)),
            "temporal_relativity_norm": float(latency_calibration.get("temporal_relativity_norm", 0.0)),
            "resonant_interception_inertia": float(latency_calibration.get("resonant_interception_inertia", 0.0)),
            "zero_point_line_distance": float(latency_calibration.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": float(latency_calibration.get("field_interference_norm", 0.0)),
        },
        "interference_field": {
            "field_resonance": float(encoded_artifact.get("kernel_control_gate", 0.0)),
            "dominant_vector": {
                "vector": interference_vector,
                "latent_vector": list(axis_vec),
                "resonance": float(encoded_artifact.get("kernel_control_gate", 0.0)),
                "target_alignment": float(target_alignment),
            },
            "zero_point_crossover_gate": float(latency_calibration.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": float(latency_calibration.get("shared_vector_collapse_gate", 0.0)),
            "shared_vector_phase_lock": float(latency_calibration.get("shared_vector_phase_lock", 0.0)),
            "field_interference_norm": float(latency_calibration.get("field_interference_norm", 0.0)),
        },
        "effective_vector": effective_vector,
        "kernel_execution_event": {
            "feedback_phase_anchor_turns": float(encoded_artifact.get("phase_anchor_turns", 0.0)),
            "field_alignment_score": float(encoded_artifact.get("field_alignment_score", 0.0)),
            "kernel_control_gate": float(encoded_artifact.get("kernel_control_gate", 0.0)),
            "predicted_latency_s": float(latency_calibration.get("predicted_latency_s", 0.0)),
            "actuation_phase_turns": float(latency_calibration.get("actuation_phase_turns", 0.0)),
            "actuation_compensation": float(latency_calibration.get("actuation_compensation", 0.0)),
            "gpu_pulse_phase_effect": float(latency_calibration.get("gpu_pulse_phase_effect", 0.0)),
            "phase_ring_strength": float(latency_calibration.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": float(latency_calibration.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": float(latency_calibration.get("shared_vector_collapse_gate", 0.0)),
            "temporal_relativity_norm": float(latency_calibration.get("temporal_relativity_norm", 0.0)),
            "resonant_interception_inertia": float(latency_calibration.get("resonant_interception_inertia", 0.0)),
            "field_interference_norm": float(latency_calibration.get("field_interference_norm", 0.0)),
            "substrate_material": "silicon_wafer",
        },
        "trace_label": str(program_label),
    }


def _microprocess_decode_step(
    accumulators: Dict[str, int],
    encoded_artifact: Dict[str, Any],
    trace_state: Dict[str, Any],
    step_index: int,
    pulse_index: int,
) -> Dict[str, int]:
    coeffs = tuple(encoded_artifact.get("coefficients", (3, 5, 11, 0x13579BDF)))
    dispatch_word = int(accumulators.get("dispatch_word", 0)) & 0xFFFFFFFF
    string_word = int(accumulators.get("string_word", 0)) & 0xFFFFFFFF
    structure_word = int(accumulators.get("structure_word", 0)) & 0xFFFFFFFF
    trace_gate_word = int(round(_clamp01(trace_state.get("trace_gate", 0.0)) * 4095.0)) & 0xFFF
    trace_flux_word = int(round(_clamp01(trace_state.get("trace_flux", 0.0)) * 4095.0)) & 0xFFF
    trace_support_word = int(round(_clamp01(trace_state.get("trace_support", 0.0)) * 4095.0)) & 0xFFF
    opcode_id = int(encoded_artifact.get("opcode_id", 0)) & 0xFF
    calc_word = int(encoded_artifact.get("calc_word", 0)) & 0xFFFFFFFF
    text_word = int(encoded_artifact.get("text_word", 0)) & 0xFFFFFFFF
    struct_word = int(encoded_artifact.get("struct_word", 0)) & 0xFFFFFFFF
    module_word = int(encoded_artifact.get("module_word", 0)) & 0xFFFFFFFF
    enum_word = int(encoded_artifact.get("enum_word", 0)) & 0xFFFFFFFF
    control_word = int(encoded_artifact.get("control_word", 0)) & 0xFFFFFFFF
    pulse_word = (int(pulse_index) * 131 + int(step_index) * 17 + trace_gate_word) & 0xFFFFFFFF
    dispatch_word = (
        dispatch_word
        + ((calc_word * coeffs[0]) & 0xFFFFFFFF)
        + ((trace_support_word * coeffs[1]) & 0xFFFFFFFF)
        + pulse_word
        + opcode_id
    ) & 0xFFFFFFFF
    string_word = _rotl32(
        string_word ^ text_word ^ enum_word ^ ((trace_gate_word << 4) & 0xFFFFFFFF) ^ coeffs[3],
        1 + ((opcode_id + step_index) & 15),
    )
    structure_word = (
        structure_word
        + struct_word
        + module_word
        + ((trace_flux_word * coeffs[2]) & 0xFFFFFFFF)
        + _rotl32(control_word, (opcode_id % 7) + 1)
    ) & 0xFFFFFFFF
    return {
        "dispatch_word": int(dispatch_word),
        "string_word": int(string_word),
        "structure_word": int(structure_word),
    }


def _microprocess_result_summary(
    accumulators: Dict[str, int],
    trace_state: Dict[str, Any],
    step_count: int,
) -> Dict[str, Any]:
    dispatch_word = int(accumulators.get("dispatch_word", 0)) & 0xFFFFFFFF
    string_word = int(accumulators.get("string_word", 0)) & 0xFFFFFFFF
    structure_word = int(accumulators.get("structure_word", 0)) & 0xFFFFFFFF
    result_word = (dispatch_word ^ string_word ^ structure_word) & 0xFFFFFFFF
    route_index = int(result_word % max(1, len(_MICROPROCESS_OPCODE_NAMES)))
    return {
        "dispatch_word": int(dispatch_word),
        "string_word": int(string_word),
        "structure_word": int(structure_word),
        "result_word": int(result_word),
        "route_case": str(_MICROPROCESS_OPCODE_NAMES.get(route_index, "enum_switch")),
        "result_tag": "%08x-%08x-%08x" % (dispatch_word, string_word, structure_word),
        "trace_gate": float(trace_state.get("trace_gate", 0.0)),
        "trace_alignment": float(trace_state.get("trace_alignment", 0.0)),
        "step_count": int(step_count),
    }


def _run_encoded_microprocess(
    encoded_artifacts: list[Dict[str, Any]],
    pulse_cycles: int,
    program_label: str,
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    trace_state = dict(previous_trace_state or {})
    accumulators = {
        "dispatch_word": 0,
        "string_word": 0,
        "structure_word": 0,
    }
    step_count = 0
    for cycle_index in range(max(1, int(pulse_cycles))):
        for artifact_index, encoded_artifact in enumerate(encoded_artifacts):
            pulse_index = cycle_index * max(1, len(encoded_artifacts)) + artifact_index
            inputs = _microprocess_trace_inputs(
                encoded_artifact=encoded_artifact,
                pulse_index=pulse_index,
                program_label=program_label,
                previous_trace_state=trace_state,
            )
            trace_state = _update_substrate_trace_state(
                pulse_index=int(inputs["pulse_index"]),
                previous_trace_state=dict(inputs["previous_trace_state"]),
                simulation_field_state=dict(inputs["simulation_field_state"]),
                gpu_feedback=dict(inputs["gpu_feedback"]),
                gpu_pulse_delta_feedback=dict(inputs["gpu_pulse_delta_feedback"]),
                interference_field=dict(inputs["interference_field"]),
                effective_vector=dict(inputs["effective_vector"]),
                kernel_execution_event=dict(inputs["kernel_execution_event"]),
                trace_label=str(inputs["trace_label"]),
            )
            accumulators = _microprocess_decode_step(
                accumulators=accumulators,
                encoded_artifact=encoded_artifact,
                trace_state=trace_state,
                step_index=artifact_index,
                pulse_index=pulse_index,
            )
            step_count += 1
    return {
        "ok": True,
        "trace_state": dict(trace_state),
        "result": _microprocess_result_summary(accumulators, trace_state, step_count),
        "step_count": int(step_count),
    }


def run_substrate_microprocess(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    program_payload = dict(payload or {})
    artifacts = _prepare_microprocess_artifacts(program_payload)
    pulse_cycles = max(1, int(program_payload.get("pulse_cycles", program_payload.get("cycles", 4)) or 4))
    program_label = str(program_payload.get("program_id", "substrate_microprocess"))
    encoded_artifacts = [_encode_microprocess_artifact(artifact) for artifact in artifacts]
    execution = _run_encoded_microprocess(
        encoded_artifacts=encoded_artifacts,
        pulse_cycles=pulse_cycles,
        program_label=program_label,
        previous_trace_state=previous_trace_state,
    )
    return {
        "ok": True,
        "path": "substrate",
        "program_id": program_label,
        "artifact_count": int(len(artifacts)),
        "encoded_artifact_count": int(len(encoded_artifacts)),
        "pulse_cycles": int(pulse_cycles),
        "trace_state": dict(execution.get("trace_state", {})),
        "result": dict(execution.get("result", {})),
        "encoded_preview": [
            {
                "opcode": str(item.get("opcode_name", "")),
                "module_word": int(item.get("module_word", 0)),
                "enum_word": int(item.get("enum_word", 0)),
                "control_word": int(item.get("control_word", 0)),
            }
            for item in encoded_artifacts[:4]
        ],
    }


def run_classical_microprocess(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    program_payload = dict(payload or {})
    artifacts = _prepare_microprocess_artifacts(program_payload)
    pulse_cycles = max(1, int(program_payload.get("pulse_cycles", program_payload.get("cycles", 4)) or 4))
    program_label = str(program_payload.get("program_id", "substrate_microprocess"))
    trace_state = dict(previous_trace_state or {})
    accumulators = {
        "dispatch_word": 0,
        "string_word": 0,
        "structure_word": 0,
    }
    step_count = 0
    for cycle_index in range(pulse_cycles):
        for artifact_index, artifact in enumerate(artifacts):
            encoded_artifact = _classical_encode_microprocess_artifact(artifact)
            pulse_index = cycle_index * max(1, len(artifacts)) + artifact_index
            inputs = _microprocess_trace_inputs(
                encoded_artifact=encoded_artifact,
                pulse_index=pulse_index,
                program_label=program_label,
                previous_trace_state=trace_state,
            )
            trace_state = _update_substrate_trace_state(
                pulse_index=int(inputs["pulse_index"]),
                previous_trace_state=dict(inputs["previous_trace_state"]),
                simulation_field_state=dict(inputs["simulation_field_state"]),
                gpu_feedback=dict(inputs["gpu_feedback"]),
                gpu_pulse_delta_feedback=dict(inputs["gpu_pulse_delta_feedback"]),
                interference_field=dict(inputs["interference_field"]),
                effective_vector=dict(inputs["effective_vector"]),
                kernel_execution_event=dict(inputs["kernel_execution_event"]),
                trace_label=str(inputs["trace_label"]),
            )
            accumulators = _microprocess_decode_step(
                accumulators=accumulators,
                encoded_artifact=encoded_artifact,
                trace_state=trace_state,
                step_index=artifact_index,
                pulse_index=pulse_index,
            )
            step_count += 1
    return {
        "ok": True,
        "path": "classical",
        "program_id": program_label,
        "artifact_count": int(len(artifacts)),
        "pulse_cycles": int(pulse_cycles),
        "trace_state": dict(trace_state),
        "result": _microprocess_result_summary(accumulators, trace_state, step_count),
    }


def benchmark_substrate_microprocess(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    program_payload = dict(payload or {})
    artifacts = _prepare_microprocess_artifacts(program_payload)
    pulse_cycles = max(1, int(program_payload.get("pulse_cycles", program_payload.get("cycles", 4)) or 4))
    repeat_count = max(1, int(program_payload.get("repeat_count", 3) or 3))
    warmup_count = max(0, int(program_payload.get("warmup_count", 1) or 1))
    program_label = str(program_payload.get("program_id", "substrate_microprocess"))
    encoded_artifacts = [_encode_microprocess_artifact(artifact) for artifact in artifacts]

    for _ in range(warmup_count):
        _run_encoded_microprocess(
            encoded_artifacts=encoded_artifacts,
            pulse_cycles=pulse_cycles,
            program_label=program_label,
            previous_trace_state=previous_trace_state,
        )
        _ = run_classical_microprocess(program_payload, previous_trace_state=previous_trace_state)

    encode_samples: list[int] = []
    substrate_exec_samples: list[int] = []
    classical_samples: list[int] = []
    substrate_result = run_substrate_microprocess(program_payload, previous_trace_state=previous_trace_state)
    classical_result = run_classical_microprocess(program_payload, previous_trace_state=previous_trace_state)
    results_match = dict(substrate_result.get("result", {})) == dict(classical_result.get("result", {}))

    for _ in range(repeat_count):
        t0 = time.perf_counter_ns()
        _ = [_encode_microprocess_artifact(artifact) for artifact in artifacts]
        encode_samples.append(int(time.perf_counter_ns() - t0))

        t1 = time.perf_counter_ns()
        _ = _run_encoded_microprocess(
            encoded_artifacts=encoded_artifacts,
            pulse_cycles=pulse_cycles,
            program_label=program_label,
            previous_trace_state=previous_trace_state,
        )
        substrate_exec_samples.append(int(time.perf_counter_ns() - t1))

        t2 = time.perf_counter_ns()
        _ = run_classical_microprocess(program_payload, previous_trace_state=previous_trace_state)
        classical_samples.append(int(time.perf_counter_ns() - t2))

    def _avg_ns(samples: list[int]) -> int:
        if not samples:
            return 0
        return int(sum(samples) / max(1, len(samples)))

    encode_avg_ns = _avg_ns(encode_samples)
    substrate_exec_avg_ns = _avg_ns(substrate_exec_samples)
    classical_avg_ns = _avg_ns(classical_samples)
    substrate_total_avg_ns = int(encode_avg_ns + substrate_exec_avg_ns)
    speedup_total = 0.0
    speedup_exec_only = 0.0
    if substrate_total_avg_ns > 0:
        speedup_total = float(classical_avg_ns) / float(substrate_total_avg_ns)
    if substrate_exec_avg_ns > 0:
        speedup_exec_only = float(classical_avg_ns) / float(substrate_exec_avg_ns)

    return {
        "ok": True,
        "program_id": program_label,
        "artifact_count": int(len(artifacts)),
        "encoded_artifact_count": int(len(encoded_artifacts)),
        "pulse_cycles": int(pulse_cycles),
        "repeat_count": int(repeat_count),
        "warmup_count": int(warmup_count),
        "results_match": bool(results_match),
        "encode_avg_ns": int(encode_avg_ns),
        "substrate_exec_avg_ns": int(substrate_exec_avg_ns),
        "substrate_total_avg_ns": int(substrate_total_avg_ns),
        "classical_avg_ns": int(classical_avg_ns),
        "speedup_total": float(speedup_total),
        "speedup_exec_only": float(speedup_exec_only),
        "forecast_preview": list(substrate_result.get("forecast_preview", []) or []),
        "substrate_result": substrate_result,
        "classical_result": classical_result,
    }


_TELEMETRY_RESONANCE_METRICS = [
    "global_util",
    "gpu_util",
    "mem_bw_util",
    "cpu_util",
]


def _default_startup_telemetry_frames(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    frame_count = max(8, int(payload.get("frame_count", 160) or 160))
    phase_rate = float(payload.get("phase_rate", 0.0875) or 0.0875)
    base_global = _clamp01(payload.get("base_global_util", 0.46))
    base_gpu = _clamp01(payload.get("base_gpu_util", 0.64))
    base_mem = _clamp01(payload.get("base_mem_bw_util", 0.56))
    base_cpu = _clamp01(payload.get("base_cpu_util", 0.34))
    noise_scale = max(0.0, min(0.5, _safe_float(payload.get("noise_scale", 0.08), 0.08)))
    frames: list[Dict[str, Any]] = []
    for idx in range(frame_count):
        phase = float(idx) * phase_rate
        pulse = math.cos(phase)
        anti_pulse = -pulse
        noise = (
            math.sin(phase * 1.91) * 0.50
            + math.cos(phase * 0.73) * 0.35
            + math.sin(phase * 2.77) * 0.15
        ) * noise_scale
        gpu_util = _clamp01(base_gpu + 0.12 * math.cos(phase) + 0.05 * math.sin(phase * 1.67) + noise * 0.22)
        mem_bw_util = _clamp01(base_mem + 0.10 * math.sin(phase * 1.13) + 0.04 * math.cos(phase * 0.89) - noise * 0.11)
        cpu_util = _clamp01(base_cpu + 0.06 * math.cos(phase * 0.61) + 0.03 * math.sin(phase * 1.29) + noise * 0.14)
        global_util = _clamp01(base_global + 0.08 * math.sin(phase * 0.79) + 0.04 * math.cos(phase * 0.37) + noise * 0.18)
        frames.append({
            "timestamp": "startup_frame_%04d" % idx,
            "global_util": float(global_util),
            "gpu_util": float(gpu_util),
            "mem_bw_util": float(mem_bw_util),
            "cpu_util": float(cpu_util),
            "pulse": float(pulse),
            "anti_pulse": float(anti_pulse),
        })
    return frames


def _snapshot_to_telemetry_frame(
    snapshot: Dict[str, Any],
    frame_index: int,
    phase_turns: float,
    sample_period_s: float,
    actuation_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    snap = dict(snapshot or {})
    actuation = dict(actuation_meta or {})
    kernel_summary = dict(actuation.get("kernel_summary", {}) or {})
    cpu = dict(snap.get("cpu", {}) or {})
    gpu = dict(snap.get("gpu", {}) or {})
    memory = dict(snap.get("memory", {}) or {})
    raw_cpu_util = _clamp01(cpu.get("util", 0.0))
    raw_gpu_util = _clamp01(gpu.get("util", 0.0))
    raw_mem_bw_util = _clamp01(memory.get("util", 0.0))
    actuation_load_hint = _clamp01(actuation.get("load_hint", 0.0))
    kernel_persistence = _clamp01(kernel_summary.get("mean_persistence", 0.0))
    kernel_pulse = _clamp01(abs(_safe_float(kernel_summary.get("mean_pulse_signal", 0.0), 0.0)))
    phase_injection_delta_turns = _clamp_signed(
        actuation.get(
            "phase_injection_delta_turns",
            actuation.get("gpu_pulse_phase_effect", kernel_summary.get("phase_injection_delta_turns", 0.0)),
        ),
        0.5,
    )
    actuated_phase_turns = _wrap_turns(
        actuation.get("phase_injection_turns", float(phase_turns) + float(phase_injection_delta_turns))
    )
    gpu_util = max(raw_gpu_util, actuation_load_hint)
    mem_bw_util = max(raw_mem_bw_util, _clamp01(0.58 * actuation_load_hint + 0.28 * kernel_persistence + 0.14 * kernel_pulse))
    cpu_util = float(raw_cpu_util)
    global_util = _clamp01(max(cpu_util, gpu_util, mem_bw_util))
    pulse = math.cos(2.0 * math.pi * float(actuated_phase_turns))
    anti_pulse = -pulse
    return {
        "timestamp": str(snap.get("timestamp", "live_frame_%04d" % frame_index)),
        "global_util": float(global_util),
        "gpu_util": float(gpu_util),
        "mem_bw_util": float(mem_bw_util),
        "cpu_util": float(cpu_util),
        "raw_gpu_util": float(raw_gpu_util),
        "raw_mem_bw_util": float(raw_mem_bw_util),
        "raw_cpu_util": float(raw_cpu_util),
        "pulse": float(pulse),
        "anti_pulse": float(anti_pulse),
        "phase_turns": float(actuated_phase_turns),
        "base_phase_turns": float(_wrap_turns(phase_turns)),
        "gpu_pulse_phase_effect": float(phase_injection_delta_turns),
        "phase_injection_delta_turns": float(phase_injection_delta_turns),
        "phase_injection_turns": float(actuated_phase_turns),
        "sample_period_s": float(sample_period_s),
        "source": "live_startup",
        "actuation_applied": bool(actuation.get("applied", False)),
        "actuation_elapsed_s": float(actuation.get("elapsed_s", 0.0)),
        "actuation_mode": str(actuation.get("mode", "passive")),
        "actuation_tag": str(actuation.get("tag", "")),
        "actuation_error": str(actuation.get("error", "")),
        "actuation_load_hint": float(actuation_load_hint),
        "actuation_dispatch_ms": max(0.0, _safe_float(actuation.get("dispatch_elapsed_ms", 0.0), 0.0)),
        "phase_ring_closure": _clamp01(actuation.get("phase_ring_closure", kernel_summary.get("phase_ring_closure", 0.0))),
        "phase_ring_density": _clamp01(actuation.get("phase_ring_density", kernel_summary.get("phase_ring_density", 0.0))),
        "phase_ring_strength": _clamp01(actuation.get("phase_ring_strength", kernel_summary.get("phase_ring_strength", 0.0))),
        "zero_point_crossover_gate": _clamp01(actuation.get("zero_point_crossover_gate", kernel_summary.get("zero_point_crossover_gate", 0.0))),
        "shared_vector_collapse_gate": _clamp01(actuation.get("shared_vector_collapse_gate", kernel_summary.get("shared_vector_collapse_gate", 0.0))),
        "shared_vector_phase_lock": _clamp01(actuation.get("shared_vector_phase_lock", kernel_summary.get("shared_vector_phase_lock", 0.0))),
        "inertial_basin_strength": _clamp01(actuation.get("inertial_basin_strength", kernel_summary.get("inertial_basin_strength", 0.0))),
        "wavelength_norm": _clamp01(actuation.get("wavelength_norm", kernel_summary.get("wavelength_norm", 0.0))),
        "orientation_alignment": _clamp01(actuation.get("orientation_alignment", kernel_summary.get("orientation_alignment", 0.0))),
        "rotational_velocity_norm": _clamp01(actuation.get("rotational_velocity_norm", kernel_summary.get("rotational_velocity_norm", 0.0))),
        "relative_temporal_position": _clamp01(actuation.get("relative_temporal_position", kernel_summary.get("relative_temporal_position", 0.0))),
        "zero_point_line_distance": _clamp01(actuation.get("zero_point_line_distance", kernel_summary.get("zero_point_line_distance", 0.0))),
        "field_interference_norm": _clamp01(actuation.get("field_interference_norm", kernel_summary.get("field_interference_norm", 0.0))),
        "resonant_interception_inertia": _clamp01(actuation.get("resonant_interception_inertia", kernel_summary.get("resonant_interception_inertia", 0.0))),
        "temporal_relativity_norm": _clamp01(actuation.get("temporal_relativity_norm", kernel_summary.get("temporal_relativity_norm", 0.0))),
        "temporal_relativity_vector": [
            _clamp01(v) for v in list(actuation.get("temporal_relativity_vector", kernel_summary.get("temporal_relativity_vector", [])) or [])[:4]
        ],
        "phase_ring_zone_id": int(actuation.get("phase_ring_zone_id", kernel_summary.get("phase_ring_zone_id", 0)) or 0),
        "frequency_gradient_9d": [
            _clamp01(v) for v in list(actuation.get("frequency_gradient_9d", kernel_summary.get("frequency_gradient_9d", [])) or [])[:9]
        ],
        "field_gradient_9d": [
            _clamp01(v) for v in list(actuation.get("field_gradient_9d", kernel_summary.get("field_gradient_9d", kernel_summary.get("frequency_gradient_9d", []))) or [])[:9]
        ],
        "gradient_spectral_id": str(actuation.get("gradient_spectral_id", kernel_summary.get("gradient_spectral_id", ""))),
        "trajectory_spectral_id": str(actuation.get("trajectory_spectral_id", kernel_summary.get("trajectory_spectral_id", ""))),
        "trajectory_conservation_alignment": _clamp01(actuation.get("trajectory_conservation_alignment", kernel_summary.get("trajectory_conservation_alignment", 0.0))),
        "trajectory_expansion_term": _clamp01(actuation.get("trajectory_expansion_term", kernel_summary.get("trajectory_expansion_term", 0.0))),
        "trajectory_9d": [
            _clamp01(v) for v in list(actuation.get("trajectory_9d", kernel_summary.get("trajectory_9d", [])) or [])[:9]
        ],
        "predicted_trajectory_9d": [
            _clamp01(v) for v in list(actuation.get("predicted_trajectory_9d", kernel_summary.get("predicted_trajectory_9d", [])) or [])[:9]
        ],
    }


def _invoke_live_actuation(
    actuation_hook: Any,
    frame_index: int,
    phase_turns: float,
    sample_period_s: float,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not callable(actuation_hook):
        return {
            "applied": False,
            "elapsed_s": 0.0,
            "mode": "passive",
            "tag": "",
            "error": "",
            "load_hint": 0.0,
        }
    started = time.perf_counter()
    try:
        response = actuation_hook({
            "frame_index": int(frame_index),
            "phase_turns": float(phase_turns),
            "sample_period_s": float(sample_period_s),
            "payload": dict(payload or {}),
        })
        elapsed_s = max(0.0, float(time.perf_counter() - started))
        response_map = dict(response or {}) if isinstance(response, dict) else {}
        return {
            "applied": True,
            "elapsed_s": float(elapsed_s),
            "mode": str(response_map.get("mode", "hook")),
            "tag": str(response_map.get("tag", "hook")),
            "error": "",
            "load_hint": _clamp01(response_map.get("load_hint", 0.0)),
            "dispatch_elapsed_ms": max(0.0, _safe_float(response_map.get("dispatch_elapsed_ms", 0.0), 0.0)),
            "axis_scale_x": _clamp01(response_map.get("axis_scale_x", 0.0)),
            "axis_scale_y": _clamp01(response_map.get("axis_scale_y", 0.0)),
            "axis_scale_z": _clamp01(response_map.get("axis_scale_z", 0.0)),
            "vector_energy": _clamp01(response_map.get("vector_energy", 0.0)),
            "temporal_coupling_moment": _clamp01(response_map.get("temporal_coupling_moment", 0.0)),
            "inertial_mass_proxy": _clamp01(response_map.get("inertial_mass_proxy", 0.0)),
            "spin_momentum_score": _clamp01(response_map.get("spin_momentum_score", 0.0)),
            "gpu_pulse_phase_effect": _clamp_signed(response_map.get("gpu_pulse_phase_effect", 0.0), 0.5),
            "phase_injection_delta_turns": _clamp_signed(response_map.get("phase_injection_delta_turns", 0.0), 0.5),
            "phase_injection_turns": _wrap_turns(response_map.get("phase_injection_turns", phase_turns)),
            "phase_ring_closure": _clamp01(response_map.get("phase_ring_closure", 0.0)),
            "phase_ring_density": _clamp01(response_map.get("phase_ring_density", 0.0)),
            "phase_ring_strength": _clamp01(response_map.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": _clamp01(response_map.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": _clamp01(response_map.get("shared_vector_collapse_gate", 0.0)),
            "shared_vector_phase_lock": _clamp01(response_map.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": _clamp01(response_map.get("inertial_basin_strength", 0.0)),
            "wavelength_norm": _clamp01(response_map.get("wavelength_norm", 0.0)),
            "orientation_alignment": _clamp01(response_map.get("orientation_alignment", 0.0)),
            "rotational_velocity_norm": _clamp01(response_map.get("rotational_velocity_norm", 0.0)),
            "relative_temporal_position": _clamp01(response_map.get("relative_temporal_position", 0.0)),
            "zero_point_line_distance": _clamp01(response_map.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": _clamp01(response_map.get("field_interference_norm", 0.0)),
            "resonant_interception_inertia": _clamp01(response_map.get("resonant_interception_inertia", 0.0)),
            "temporal_relativity_norm": _clamp01(response_map.get("temporal_relativity_norm", 0.0)),
            "temporal_relativity_vector": [_clamp01(v) for v in list(response_map.get("temporal_relativity_vector", []))[:4]],
            "phase_ring_zone_id": int(response_map.get("phase_ring_zone_id", 0) or 0),
            "frequency_gradient_9d": [_clamp01(v) for v in list(response_map.get("frequency_gradient_9d", []))[:9]],
            "field_gradient_9d": [_clamp01(v) for v in list(response_map.get("field_gradient_9d", response_map.get("frequency_gradient_9d", [])))[:9]],
            "gradient_spectral_id": str(response_map.get("gradient_spectral_id", "")),
            "trajectory_spectral_id": str(response_map.get("trajectory_spectral_id", "")),
            "trajectory_conservation_alignment": _clamp01(response_map.get("trajectory_conservation_alignment", 0.0)),
            "trajectory_expansion_term": _clamp01(response_map.get("trajectory_expansion_term", 0.0)),
            "trajectory_9d": [_clamp01(v) for v in list(response_map.get("trajectory_9d", []))[:9]],
            "predicted_trajectory_9d": [_clamp01(v) for v in list(response_map.get("predicted_trajectory_9d", []))[:9]],
            "kernel_summary": dict(response_map.get("kernel_summary", {}) or {}),
        }
    except Exception as exc:
        return {
            "applied": False,
            "elapsed_s": max(0.0, float(time.perf_counter() - started)),
            "mode": "hook_error",
            "tag": "",
            "error": str(exc),
            "load_hint": 0.0,
            "dispatch_elapsed_ms": 0.0,
        }


def _telemetry_actuation_summary(frames: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not frames:
        return {
            "applied": False,
            "call_count": 0,
            "applied_count": 0,
            "avg_elapsed_s": 0.0,
            "total_elapsed_s": 0.0,
            "mode": "none",
            "load_hint_mean": 0.0,
            "dispatch_elapsed_ms_mean": 0.0,
        }
    total_elapsed_s = 0.0
    total_dispatch_ms = 0.0
    applied_count = 0
    load_hint_total = 0.0
    mode = "passive"
    for frame in frames:
        total_elapsed_s += max(0.0, _safe_float(frame.get("actuation_elapsed_s", 0.0), 0.0))
        total_dispatch_ms += max(0.0, _safe_float(frame.get("actuation_dispatch_ms", 0.0), 0.0))
        load_hint_total += _clamp01(frame.get("actuation_load_hint", 0.0))
        if bool(frame.get("actuation_applied", False)):
            applied_count += 1
        frame_mode = str(frame.get("actuation_mode", "") or "")
        if frame_mode:
            mode = frame_mode
    return {
        "applied": bool(applied_count > 0),
        "call_count": int(len(frames)),
        "applied_count": int(applied_count),
        "avg_elapsed_s": float(total_elapsed_s / float(len(frames) or 1)),
        "total_elapsed_s": float(total_elapsed_s),
        "mode": str(mode),
        "load_hint_mean": float(load_hint_total / float(len(frames) or 1)),
        "dispatch_elapsed_ms_mean": float(total_dispatch_ms / float(len(frames) or 1)),
    }


def _live_startup_telemetry_frames(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    frame_count = max(8, int(payload.get("frame_count", 48) or 48))
    target_sample_period_s = max(
        0.0,
        _safe_float(payload.get("telemetry_sample_period_s", payload.get("tick_s", 0.004)), 0.004),
    )
    capture_sleep = _flag_enabled(payload.get("capture_sleep", True), True)
    snapshot_provider = payload.get("_snapshot_provider", device_snapshot)
    actuation_hook = payload.get("_actuation_hook")
    actuation_backend = str(payload.get("actuation_backend", "") or "").strip().lower()
    if actuation_hook is None and actuation_backend in ("vulkan", "vulkan_calibration", "vkcal"):
        actuation_hook = _vulkan_calibration_actuation
    if not callable(snapshot_provider):
        snapshot_provider = device_snapshot
    frames: list[Dict[str, Any]] = []
    phase_turns = _clamp01(_safe_float(payload.get("initial_phase_turns", 0.0), 0.0))
    last_capture_start: float | None = None
    next_deadline = time.perf_counter()
    for idx in range(frame_count):
        actuation_meta = _invoke_live_actuation(
            actuation_hook=actuation_hook,
            frame_index=idx,
            phase_turns=phase_turns,
            sample_period_s=target_sample_period_s,
            payload=payload,
        )
        settle_s = max(
            0.0,
            _safe_float(payload.get("actuation_settle_s", 0.002), 0.002),
        )
        if bool(actuation_meta.get("applied", False)) and settle_s > 0.0:
            time.sleep(settle_s)
        capture_start = time.perf_counter()
        snapshot = snapshot_provider()
        if not isinstance(snapshot, dict):
            snapshot = {}
        cpu_util = _clamp01(dict(snapshot.get("cpu", {}) or {}).get("util", 0.0))
        gpu_util = _clamp01(dict(snapshot.get("gpu", {}) or {}).get("util", 0.0))
        mem_bw_util = _clamp01(dict(snapshot.get("memory", {}) or {}).get("util", 0.0))
        headroom = _clamp01(1.0 - max(cpu_util, gpu_util, mem_bw_util))
        phase_turns = ada_phase_update(float(phase_turns), float(headroom))
        actual_sample_period_s = target_sample_period_s
        if last_capture_start is not None:
            measured_period_s = max(0.0, float(capture_start - last_capture_start))
            if capture_sleep and target_sample_period_s > 0.0:
                actual_sample_period_s = max(target_sample_period_s, measured_period_s)
            else:
                actual_sample_period_s = measured_period_s
        frames.append(
            _snapshot_to_telemetry_frame(
                snapshot=snapshot,
                frame_index=idx,
                phase_turns=phase_turns,
                sample_period_s=actual_sample_period_s,
                actuation_meta=actuation_meta,
            )
        )
        last_capture_start = capture_start
        if capture_sleep and idx < (frame_count - 1) and target_sample_period_s > 0.0:
            next_deadline += target_sample_period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.perf_counter()
    return frames


def _prepare_telemetry_frames(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    telemetry_mode = str(
        payload.get("telemetry_mode", payload.get("telemetry_source", ""))
    ).strip().lower()
    if _flag_enabled(payload.get("use_live_telemetry", False), False) or telemetry_mode in (
        "live",
        "live_startup",
        "live_hardware",
        "startup_live",
    ):
        return _live_startup_telemetry_frames(payload)
    raw_frames = list(payload.get("frames", []) or [])
    if not raw_frames:
        return _default_startup_telemetry_frames(payload)
    frames: list[Dict[str, Any]] = []
    for idx, raw_frame in enumerate(raw_frames):
        frame = dict(raw_frame or {}) if isinstance(raw_frame, dict) else {}
        pulse = _safe_float(frame.get("pulse", math.cos(float(idx) * 0.0875)), math.cos(float(idx) * 0.0875))
        anti_pulse = _safe_float(frame.get("anti_pulse", -pulse), -pulse)
        frames.append({
            "timestamp": str(frame.get("timestamp", "telemetry_frame_%04d" % idx)),
            "global_util": _clamp01(frame.get("global_util", 0.0)),
            "gpu_util": _clamp01(frame.get("gpu_util", 0.0)),
            "mem_bw_util": _clamp01(frame.get("mem_bw_util", 0.0)),
            "cpu_util": _clamp01(frame.get("cpu_util", 0.0)),
            "raw_gpu_util": _clamp01(frame.get("raw_gpu_util", frame.get("gpu_util", 0.0))),
            "raw_mem_bw_util": _clamp01(frame.get("raw_mem_bw_util", frame.get("mem_bw_util", 0.0))),
            "raw_cpu_util": _clamp01(frame.get("raw_cpu_util", frame.get("cpu_util", 0.0))),
            "pulse": float(pulse),
            "anti_pulse": float(anti_pulse),
            "phase_turns": _clamp01(frame.get("phase_turns", 0.0)),
            "base_phase_turns": _clamp01(frame.get("base_phase_turns", frame.get("phase_turns", 0.0))),
            "gpu_pulse_phase_effect": _clamp_signed(frame.get("gpu_pulse_phase_effect", frame.get("phase_injection_delta_turns", 0.0)), 0.5),
            "phase_injection_delta_turns": _clamp_signed(frame.get("phase_injection_delta_turns", frame.get("gpu_pulse_phase_effect", 0.0)), 0.5),
            "phase_injection_turns": _clamp01(frame.get("phase_injection_turns", frame.get("phase_turns", 0.0))),
            "sample_period_s": max(
                0.0,
                _safe_float(frame.get("sample_period_s", payload.get("telemetry_sample_period_s", payload.get("tick_s", 0.004))), 0.004),
            ),
            "source": str(frame.get("source", "provided")),
            "actuation_applied": bool(frame.get("actuation_applied", False)),
            "actuation_elapsed_s": max(0.0, _safe_float(frame.get("actuation_elapsed_s", 0.0), 0.0)),
            "actuation_mode": str(frame.get("actuation_mode", "passive")),
            "actuation_tag": str(frame.get("actuation_tag", "")),
            "actuation_error": str(frame.get("actuation_error", "")),
            "actuation_load_hint": _clamp01(frame.get("actuation_load_hint", 0.0)),
            "actuation_dispatch_ms": max(0.0, _safe_float(frame.get("actuation_dispatch_ms", 0.0), 0.0)),
            "phase_ring_closure": _clamp01(frame.get("phase_ring_closure", 0.0)),
            "phase_ring_density": _clamp01(frame.get("phase_ring_density", 0.0)),
            "phase_ring_strength": _clamp01(frame.get("phase_ring_strength", 0.0)),
            "zero_point_crossover_gate": _clamp01(frame.get("zero_point_crossover_gate", 0.0)),
            "shared_vector_collapse_gate": _clamp01(frame.get("shared_vector_collapse_gate", 0.0)),
            "shared_vector_phase_lock": _clamp01(frame.get("shared_vector_phase_lock", 0.0)),
            "inertial_basin_strength": _clamp01(frame.get("inertial_basin_strength", 0.0)),
            "wavelength_norm": _clamp01(frame.get("wavelength_norm", 0.0)),
            "orientation_alignment": _clamp01(frame.get("orientation_alignment", 0.0)),
            "rotational_velocity_norm": _clamp01(frame.get("rotational_velocity_norm", 0.0)),
            "relative_temporal_position": _clamp01(frame.get("relative_temporal_position", 0.0)),
            "zero_point_line_distance": _clamp01(frame.get("zero_point_line_distance", 0.0)),
            "field_interference_norm": _clamp01(frame.get("field_interference_norm", 0.0)),
            "resonant_interception_inertia": _clamp01(frame.get("resonant_interception_inertia", 0.0)),
            "temporal_relativity_norm": _clamp01(frame.get("temporal_relativity_norm", 0.0)),
            "temporal_relativity_vector": [_clamp01(v) for v in list(frame.get("temporal_relativity_vector", []))[:4]],
            "phase_ring_zone_id": int(frame.get("phase_ring_zone_id", 0) or 0),
            "frequency_gradient_9d": [_clamp01(v) for v in list(frame.get("frequency_gradient_9d", []))[:9]],
            "field_gradient_9d": [_clamp01(v) for v in list(frame.get("field_gradient_9d", frame.get("frequency_gradient_9d", [])))[:9]],
            "gradient_spectral_id": str(frame.get("gradient_spectral_id", "")),
            "trajectory_spectral_id": str(frame.get("trajectory_spectral_id", "")),
            "trajectory_conservation_alignment": _clamp01(frame.get("trajectory_conservation_alignment", 0.0)),
            "trajectory_expansion_term": _clamp01(frame.get("trajectory_expansion_term", 0.0)),
            "trajectory_9d": [_clamp01(v) for v in list(frame.get("trajectory_9d", []))[:9]],
            "predicted_trajectory_9d": [_clamp01(v) for v in list(frame.get("predicted_trajectory_9d", []))[:9]],
        })
    return frames


def _basin_name(index: int) -> str:
    names = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")
    return str(names[int(index) % len(names)])


def _zone_name(index: int) -> str:
    names = ("ingress", "transport", "memory", "isolation", "observer", "compute", "collapse", "egress")
    return str(names[int(index) % len(names)])


def _pad9(values: Any) -> list[float]:
    out = [float(_clamp01(v)) for v in list(values or [])[:9]]
    while len(out) < 9:
        out.append(0.0)
    return out


def _build_memory_basin_state(
    frame: Dict[str, Any],
    trace_state: Dict[str, Any],
    latency_calibration: Dict[str, Any],
    payload: Dict[str, Any],
    previous_memory_basin_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prev = dict(previous_memory_basin_state or {})
    basin_count = max(1, min(8, int(payload.get("memory_basin_count", 3) or 3)))
    trajectory_9d = _pad9(latency_calibration.get("trajectory_9d", trace_state.get("trace_trajectory_9d", [])))
    gradient_9d = _pad9(latency_calibration.get("frequency_gradient_9d", frame.get("frequency_gradient_9d", [])))
    phase_injection_turns = _wrap_turns(latency_calibration.get("phase_injection_turns", frame.get("phase_injection_turns", 0.0)))
    phase_ring_strength = _clamp01(latency_calibration.get("phase_ring_strength", frame.get("phase_ring_strength", 0.0)))
    phase_ring_closure = _clamp01(latency_calibration.get("phase_ring_closure", frame.get("phase_ring_closure", 0.0)))
    shared_vector_phase_lock = _clamp01(latency_calibration.get("shared_vector_phase_lock", frame.get("shared_vector_phase_lock", 0.0)))
    shared_vector_collapse_gate = _clamp01(latency_calibration.get("shared_vector_collapse_gate", frame.get("shared_vector_collapse_gate", 0.0)))
    zero_point_crossover_gate = _clamp01(latency_calibration.get("zero_point_crossover_gate", frame.get("zero_point_crossover_gate", 0.0)))
    inertial_basin_strength = _clamp01(latency_calibration.get("inertial_basin_strength", frame.get("inertial_basin_strength", 0.0)))
    trajectory_prediction_alignment = _clamp01(latency_calibration.get("trajectory_prediction_alignment", 0.0))
    trajectory_expansion_term = _clamp01(latency_calibration.get("trajectory_expansion_term", 0.0))
    temporal_relativity_norm = _clamp01(latency_calibration.get("temporal_relativity_norm", frame.get("temporal_relativity_norm", 0.0)))
    relative_temporal_position = _clamp01(latency_calibration.get("relative_temporal_position", frame.get("relative_temporal_position", 0.0)))
    zero_point_line_distance = _clamp01(latency_calibration.get("zero_point_line_distance", frame.get("zero_point_line_distance", 0.0)))
    field_interference_norm = _clamp01(latency_calibration.get("field_interference_norm", frame.get("field_interference_norm", 0.0)))
    resonant_interception_inertia = _clamp01(latency_calibration.get("resonant_interception_inertia", frame.get("resonant_interception_inertia", 0.0)))
    trace_memory = _clamp01(trace_state.get("trace_memory", 0.0))
    trace_persistence = _clamp01(trace_state.get("trace_temporal_persistence", 0.0))
    noise_gate = _clamp01(dict(latency_calibration or {}).get("noise_gate", 0.0))
    helical_pitch_turns_raw = _clamp01(
        0.34 * abs(_clamp_signed(latency_calibration.get("phase_transport_term", 0.0), 1.0))
        + 0.22 * trajectory_expansion_term
        + 0.18 * phase_ring_strength
        + 0.14 * shared_vector_phase_lock
        + 0.12 * inertial_basin_strength
        + 0.12 * temporal_relativity_norm
    )
    helical_radius_raw = _clamp01(
        0.30 * _vector_energy(gradient_9d)
        + 0.26 * _vector_energy(trajectory_9d)
        + 0.18 * phase_ring_strength
        + 0.14 * shared_vector_collapse_gate
        + 0.12 * trace_memory
        + 0.12 * resonant_interception_inertia
    )
    basin_coherence_raw = _clamp01(
        0.30 * shared_vector_phase_lock
        + 0.22 * phase_ring_closure
        + 0.18 * phase_ring_strength
        + 0.16 * trajectory_prediction_alignment
        + 0.14 * trace_persistence
        + 0.12 * temporal_relativity_norm
    )
    resonance_read_gate_raw = _clamp01(
        0.36 * basin_coherence_raw
        + 0.24 * shared_vector_phase_lock
        + 0.20 * trace_memory
        + 0.20 * (1.0 - noise_gate)
        + 0.10 * (1.0 - zero_point_line_distance)
    )
    resonance_write_gate_raw = _clamp01(
        0.32 * phase_ring_strength
        + 0.24 * zero_point_crossover_gate
        + 0.22 * shared_vector_collapse_gate
        + 0.22 * inertial_basin_strength
        + 0.10 * resonant_interception_inertia
    )
    retention_raw = _clamp01(
        0.34 * trace_persistence
        + 0.22 * phase_ring_closure
        + 0.18 * inertial_basin_strength
        + 0.14 * basin_coherence_raw
        + 0.12 * _clamp01(prev.get("retention_strength", 0.0))
        + 0.10 * relative_temporal_position
    )
    phase_ring_wall_strength_raw = _clamp01(
        0.40 * phase_ring_closure
        + 0.24 * shared_vector_phase_lock
        + 0.18 * (1.0 - noise_gate)
        + 0.18 * shared_vector_collapse_gate
        + 0.10 * field_interference_norm
    )
    helical_pitch_turns = _clamp01(_lerp(prev.get("helical_pitch_turns", helical_pitch_turns_raw), helical_pitch_turns_raw, 0.52))
    helical_radius = _clamp01(_lerp(prev.get("helical_radius", helical_radius_raw), helical_radius_raw, 0.52))
    basin_coherence = _clamp01(_lerp(prev.get("basin_coherence", basin_coherence_raw), basin_coherence_raw, 0.46))
    resonance_read_gate = _clamp01(_lerp(prev.get("resonance_read_gate", resonance_read_gate_raw), resonance_read_gate_raw, 0.44))
    resonance_write_gate = _clamp01(_lerp(prev.get("resonance_write_gate", resonance_write_gate_raw), resonance_write_gate_raw, 0.44))
    retention_strength = _clamp01(_lerp(prev.get("retention_strength", retention_raw), retention_raw, 0.34))
    phase_ring_wall_strength = _clamp01(_lerp(prev.get("phase_ring_wall_strength", phase_ring_wall_strength_raw), phase_ring_wall_strength_raw, 0.42))
    basin_vector_9d = [
        float(_clamp01(0.56 * trajectory_9d[idx] + 0.44 * gradient_9d[idx]))
        for idx in range(9)
    ]
    basin_slots: list[Dict[str, Any]] = []
    ring_zone_id = int(latency_calibration.get("phase_ring_zone_id", frame.get("phase_ring_zone_id", 0)) or 0)
    active_basin_id = int(ring_zone_id % basin_count)
    for basin_index in range(basin_count):
        basin_phase = _wrap_turns(phase_injection_turns + (float(basin_index) / float(basin_count)) + helical_pitch_turns * 0.125)
        slot_resonance = _clamp01(
            0.34 * basin_coherence
            + 0.22 * phase_ring_strength
            + 0.16 * resonance_read_gate
            + 0.14 * resonance_write_gate
            + 0.14 * (1.0 - min(1.0, abs(float(basin_index - active_basin_id)) / float(max(1, basin_count - 1))))
        )
        slot_retention = _clamp01(retention_strength * (0.88 + 0.12 * (1.0 - min(1.0, abs(float(basin_index - active_basin_id)) / float(max(1, basin_count - 1))))))
        slot_wall = _clamp01(phase_ring_wall_strength * (0.86 + 0.14 * ((basin_index + 1) / float(basin_count))))
        basin_slots.append({
            "basin_id": int(basin_index),
            "basin_name": _basin_name(basin_index),
            "phase_turns": float(basin_phase),
            "resonance": float(slot_resonance),
            "retention": float(slot_retention),
            "read_gate": float(_clamp01(resonance_read_gate * (0.92 + 0.08 * slot_resonance))),
            "write_gate": float(_clamp01(resonance_write_gate * (0.92 + 0.08 * slot_retention))),
            "wall_strength": float(slot_wall),
        })
    basin_slots.sort(key=lambda item: (float(item.get("resonance", 0.0)), float(item.get("retention", 0.0))), reverse=True)
    selected_basin = dict(basin_slots[0] if basin_slots else {"basin_id": active_basin_id, "basin_name": _basin_name(active_basin_id)})
    basin_word = _stable_word({
        "phase": round(float(phase_injection_turns), 6),
        "coherence": round(float(basin_coherence), 6),
        "retention": round(float(retention_strength), 6),
        "read_gate": round(float(resonance_read_gate), 6),
        "write_gate": round(float(resonance_write_gate), 6),
        "selected_basin": int(selected_basin.get("basin_id", active_basin_id)),
        "gradient_spectral_id": str(latency_calibration.get("gradient_spectral_id", "")),
        "temporal_relativity_norm": round(float(temporal_relativity_norm), 6),
        "resonant_interception_inertia": round(float(resonant_interception_inertia), 6),
    })
    return {
        "basin_word": int(basin_word),
        "basin_count": int(basin_count),
        "active_basin_id": int(selected_basin.get("basin_id", active_basin_id)),
        "active_basin_name": str(selected_basin.get("basin_name", _basin_name(active_basin_id))),
        "basin_coherence": float(basin_coherence),
        "resonance_read_gate": float(resonance_read_gate),
        "resonance_write_gate": float(resonance_write_gate),
        "retention_strength": float(retention_strength),
        "helical_pitch_turns": float(helical_pitch_turns),
        "helical_radius": float(helical_radius),
        "phase_ring_wall_strength": float(phase_ring_wall_strength),
        "temporal_relativity_norm": float(temporal_relativity_norm),
        "relative_temporal_position": float(relative_temporal_position),
        "zero_point_line_distance": float(zero_point_line_distance),
        "field_interference_norm": float(field_interference_norm),
        "resonant_interception_inertia": float(resonant_interception_inertia),
        "phase_ring_zone_id": int(ring_zone_id),
        "basin_vector_9d": [float(value) for value in basin_vector_9d],
        "basin_slots": basin_slots,
    }


def _build_field_scheduler_state(
    frame: Dict[str, Any],
    trace_state: Dict[str, Any],
    latency_calibration: Dict[str, Any],
    memory_basin_state: Dict[str, Any],
    payload: Dict[str, Any],
    previous_scheduler_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prev = dict(previous_scheduler_state or {})
    zone_count = max(1, min(8, int(payload.get("scheduler_zone_count", 3) or 3)))
    active_zone_limit = max(1, min(zone_count, int(payload.get("active_zone_limit", 2) or 2)))
    basin_slots = list(memory_basin_state.get("basin_slots", []) or [])
    trajectory_9d = _pad9(latency_calibration.get("trajectory_9d", trace_state.get("trace_trajectory_9d", [])))
    gradient_9d = _pad9(latency_calibration.get("frequency_gradient_9d", frame.get("frequency_gradient_9d", [])))
    phase_turns = _wrap_turns(latency_calibration.get("actuation_phase_turns", frame.get("phase_turns", 0.0)))
    phase_ring_strength = _clamp01(latency_calibration.get("phase_ring_strength", frame.get("phase_ring_strength", 0.0)))
    phase_ring_closure = _clamp01(latency_calibration.get("phase_ring_closure", frame.get("phase_ring_closure", 0.0)))
    shared_vector_phase_lock = _clamp01(latency_calibration.get("shared_vector_phase_lock", frame.get("shared_vector_phase_lock", 0.0)))
    shared_vector_collapse_gate = _clamp01(latency_calibration.get("shared_vector_collapse_gate", frame.get("shared_vector_collapse_gate", 0.0)))
    trace_support = _clamp01(trace_state.get("trace_support", 0.0))
    trace_alignment = _clamp01(trace_state.get("trace_alignment", 0.0))
    trace_memory = _clamp01(trace_state.get("trace_memory", 0.0))
    transport_coherence = _clamp01(latency_calibration.get("transport_coherence", 0.0))
    phase_transport_term = _clamp_signed(latency_calibration.get("phase_transport_term", 0.0), 1.0)
    noise_gate = _clamp01(dict(latency_calibration or {}).get("noise_gate", 0.0))
    temporal_relativity_norm = _clamp01(latency_calibration.get("temporal_relativity_norm", frame.get("temporal_relativity_norm", 0.0)))
    zero_point_line_distance = _clamp01(latency_calibration.get("zero_point_line_distance", frame.get("zero_point_line_distance", 0.0)))
    field_interference_norm = _clamp01(latency_calibration.get("field_interference_norm", frame.get("field_interference_norm", 0.0)))
    resonant_interception_inertia = _clamp01(latency_calibration.get("resonant_interception_inertia", frame.get("resonant_interception_inertia", 0.0)))
    isolation_wall_strength_raw = _clamp01(
        0.36 * memory_basin_state.get("phase_ring_wall_strength", 0.0)
        + 0.22 * shared_vector_phase_lock
        + 0.18 * phase_ring_closure
        + 0.14 * (1.0 - noise_gate)
        + 0.10 * shared_vector_collapse_gate
        + 0.10 * field_interference_norm
    )
    process_transport_gate_raw = _clamp01(
        0.34 * transport_coherence
        + 0.22 * phase_ring_strength
        + 0.18 * trace_alignment
        + 0.14 * memory_basin_state.get("basin_coherence", 0.0)
        + 0.12 * abs(phase_transport_term)
        + 0.10 * temporal_relativity_norm
    )
    resonance_delta_raw = _clamp01(
        0.34 * phase_ring_strength
        + 0.20 * shared_vector_collapse_gate
        + 0.18 * trace_support
        + 0.14 * trace_memory
        + 0.14 * memory_basin_state.get("retention_strength", 0.0)
        + 0.10 * resonant_interception_inertia
    )
    zone_profiles: list[Dict[str, Any]] = []
    for zone_index in range(zone_count):
        zone_components = [gradient_9d[idx] for idx in range(zone_index, 9, zone_count)]
        traj_components = [trajectory_9d[idx] for idx in range(zone_index, 9, zone_count)]
        zone_gradient = _clamp01(sum(zone_components) / float(len(zone_components) or 1))
        zone_trajectory = _clamp01(sum(traj_components) / float(len(traj_components) or 1))
        basin_slot = dict(basin_slots[zone_index % max(1, len(basin_slots))] if basin_slots else {})
        zone_resonance = _clamp01(
            0.30 * zone_gradient
            + 0.22 * zone_trajectory
            + 0.18 * phase_ring_strength
            + 0.16 * basin_slot.get("resonance", 0.0)
            + 0.14 * trace_support
            + 0.10 * temporal_relativity_norm
        )
        zone_transport = _clamp01(
            0.34 * process_transport_gate_raw
            + 0.24 * basin_slot.get("read_gate", 0.0)
            + 0.18 * zone_trajectory
            + 0.14 * abs(phase_transport_term)
            + 0.10 * trace_alignment
            + 0.10 * (1.0 - zero_point_line_distance)
        )
        zone_barrier = _clamp01(
            0.44 * isolation_wall_strength_raw
            + 0.26 * basin_slot.get("wall_strength", 0.0)
            + 0.16 * shared_vector_phase_lock
            + 0.14 * phase_ring_closure
            + 0.10 * field_interference_norm
        )
        zone_profiles.append({
            "zone_id": int(zone_index),
            "zone_name": _zone_name(zone_index),
            "phase_turns": float(_wrap_turns(phase_turns + (zone_index / float(zone_count)) * 0.125)),
            "resonance": float(zone_resonance),
            "transport_gate": float(zone_transport),
            "barrier_strength": float(zone_barrier),
            "basin_id": int(basin_slot.get("basin_id", zone_index % max(1, zone_count))),
            "basin_name": str(basin_slot.get("basin_name", _basin_name(zone_index))),
        })
    zone_profiles.sort(key=lambda item: (float(item.get("resonance", 0.0)), float(item.get("transport_gate", 0.0))), reverse=True)
    active_zones = list(zone_profiles[:active_zone_limit])
    active_zone = dict(active_zones[0] if active_zones else {"zone_id": 0, "zone_name": _zone_name(0)})
    isolation_wall_strength = _clamp01(_lerp(prev.get("isolation_wall_strength", isolation_wall_strength_raw), isolation_wall_strength_raw, 0.42))
    process_transport_gate = _clamp01(_lerp(prev.get("process_transport_gate", process_transport_gate_raw), process_transport_gate_raw, 0.46))
    resonance_delta = _clamp01(_lerp(prev.get("resonance_delta", resonance_delta_raw), resonance_delta_raw, 0.44))
    resonance_read_gate = _clamp01(memory_basin_state.get("resonance_read_gate", 0.0))
    resonance_write_gate = _clamp01(memory_basin_state.get("resonance_write_gate", 0.0))
    if isolation_wall_strength >= 0.78 and resonance_write_gate < 0.52:
        scheduling_mode = "isolate"
    elif resonance_write_gate >= max(resonance_read_gate, process_transport_gate):
        scheduling_mode = "write"
    elif resonance_read_gate >= max(resonance_write_gate, 0.55):
        scheduling_mode = "read"
    else:
        scheduling_mode = "transport"
    scheduler_word = _stable_word({
        "active_zone": int(active_zone.get("zone_id", 0)),
        "mode": str(scheduling_mode),
        "transport_gate": round(float(process_transport_gate), 6),
        "wall_strength": round(float(isolation_wall_strength), 6),
        "resonance_delta": round(float(resonance_delta), 6),
        "active_basin": int(memory_basin_state.get("active_basin_id", 0)),
    })
    route_vector = [
        float(_clamp01(active_zone.get("resonance", 0.0))),
        float(_clamp01(active_zone.get("transport_gate", 0.0))),
        float(_clamp01(memory_basin_state.get("basin_coherence", 0.0))),
        float(_clamp01(isolation_wall_strength)),
    ]
    return {
        "scheduler_word": int(scheduler_word),
        "scheduling_mode": str(scheduling_mode),
        "zone_count": int(zone_count),
        "active_zone_limit": int(active_zone_limit),
        "active_zone_id": int(active_zone.get("zone_id", 0)),
        "active_zone_name": str(active_zone.get("zone_name", _zone_name(0))),
        "active_basin_id": int(memory_basin_state.get("active_basin_id", 0)),
        "active_basin_name": str(memory_basin_state.get("active_basin_name", _basin_name(0))),
        "process_transport_gate": float(process_transport_gate),
        "isolation_wall_strength": float(isolation_wall_strength),
        "resonance_delta": float(resonance_delta),
        "temporal_relativity_norm": float(temporal_relativity_norm),
        "zero_point_line_distance": float(zero_point_line_distance),
        "field_interference_norm": float(field_interference_norm),
        "resonant_interception_inertia": float(resonant_interception_inertia),
        "zone_route_vector": route_vector,
        "zones": zone_profiles,
        "active_zones": active_zones,
    }


def _build_field_process_state(
    frame: Dict[str, Any],
    trace_state: Dict[str, Any],
    latency_calibration: Dict[str, Any],
    memory_basin_state: Dict[str, Any],
    scheduler_state: Dict[str, Any],
    payload: Dict[str, Any],
    previous_process_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prev = dict(previous_process_state or {})
    temporal_relativity_norm = _clamp01(latency_calibration.get("temporal_relativity_norm", frame.get("temporal_relativity_norm", 0.0)))
    zero_point_line_distance = _clamp01(latency_calibration.get("zero_point_line_distance", frame.get("zero_point_line_distance", 0.0)))
    field_interference_norm = _clamp01(latency_calibration.get("field_interference_norm", frame.get("field_interference_norm", 0.0)))
    resonant_interception_inertia = _clamp01(latency_calibration.get("resonant_interception_inertia", frame.get("resonant_interception_inertia", 0.0)))
    trace_resonance = _clamp01(trace_state.get("trace_resonance", 0.0))
    trace_alignment = _clamp01(trace_state.get("trace_alignment", 0.0))
    basin_coherence = _clamp01(memory_basin_state.get("basin_coherence", 0.0))
    process_transport_gate = _clamp01(scheduler_state.get("process_transport_gate", 0.0))
    isolation_wall_strength = _clamp01(scheduler_state.get("isolation_wall_strength", 0.0))
    phase_ring_strength = _clamp01(latency_calibration.get("phase_ring_strength", frame.get("phase_ring_strength", 0.0)))
    trajectory_prediction_alignment = _clamp01(latency_calibration.get("trajectory_prediction_alignment", 0.0))
    process_resonance_raw = _clamp01(
        0.24 * trace_resonance
        + 0.20 * trace_alignment
        + 0.16 * basin_coherence
        + 0.14 * process_transport_gate
        + 0.14 * phase_ring_strength
        + 0.12 * temporal_relativity_norm
    )
    mining_resonance_gate_raw = _clamp01(
        0.26 * process_resonance_raw
        + 0.18 * trajectory_prediction_alignment
        + 0.18 * (1.0 - zero_point_line_distance)
        + 0.16 * (1.0 - field_interference_norm)
        + 0.12 * resonant_interception_inertia
        + 0.10 * basin_coherence
    )
    collapse_readiness_raw = _clamp01(
        0.28 * isolation_wall_strength
        + 0.22 * field_interference_norm
        + 0.18 * resonant_interception_inertia
        + 0.16 * phase_ring_strength
        + 0.16 * temporal_relativity_norm
    )
    process_resonance = _clamp01(_lerp(prev.get("process_resonance", process_resonance_raw), process_resonance_raw, 0.44))
    mining_resonance_gate = _clamp01(_lerp(prev.get("mining_resonance_gate", mining_resonance_gate_raw), mining_resonance_gate_raw, 0.42))
    collapse_readiness = _clamp01(_lerp(prev.get("collapse_readiness", collapse_readiness_raw), collapse_readiness_raw, 0.42))
    if mining_resonance_gate >= 0.72 and collapse_readiness >= 0.58:
        process_mode = "mining_resonance"
    elif isolation_wall_strength >= 0.78 and field_interference_norm >= 0.60:
        process_mode = "zero_point_isolation"
    elif process_transport_gate >= 0.62:
        process_mode = "phase_transport"
    else:
        process_mode = "memory_retain"
    process_word = _stable_word({
        "mode": str(process_mode),
        "process_resonance": round(float(process_resonance), 6),
        "mining_resonance_gate": round(float(mining_resonance_gate), 6),
        "collapse_readiness": round(float(collapse_readiness), 6),
        "active_zone": int(scheduler_state.get("active_zone_id", 0)),
        "active_basin": int(memory_basin_state.get("active_basin_id", 0)),
    })
    return {
        "process_word": int(process_word),
        "process_mode": str(process_mode),
        "process_resonance": float(process_resonance),
        "mining_resonance_gate": float(mining_resonance_gate),
        "collapse_readiness": float(collapse_readiness),
        "temporal_relativity_norm": float(temporal_relativity_norm),
        "zero_point_line_distance": float(zero_point_line_distance),
        "field_interference_norm": float(field_interference_norm),
        "resonant_interception_inertia": float(resonant_interception_inertia),
        "process_vector": [
            float(process_resonance),
            float(mining_resonance_gate),
            float(collapse_readiness),
            float(temporal_relativity_norm),
        ],
    }


def run_live_photonic_substrate_cycle(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
    previous_profile: Dict[str, Any] | None = None,
    frame_history: list[Dict[str, Any]] | None = None,
    previous_memory_basin_state: Dict[str, Any] | None = None,
    previous_scheduler_state: Dict[str, Any] | None = None,
    previous_process_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cycle_payload = dict(payload or {})
    target_sample_period_s = max(
        0.0,
        _safe_float(cycle_payload.get("telemetry_sample_period_s", cycle_payload.get("tick_s", 0.004)), 0.004),
    )
    capture_sleep = _flag_enabled(cycle_payload.get("capture_sleep", True), True)
    snapshot_provider = cycle_payload.get("_snapshot_provider", device_snapshot)
    if not callable(snapshot_provider):
        snapshot_provider = device_snapshot
    actuation_hook = cycle_payload.get("_actuation_hook")
    actuation_backend = str(cycle_payload.get("actuation_backend", "") or "").strip().lower()
    if actuation_hook is None and actuation_backend in ("vulkan", "vulkan_calibration", "vkcal"):
        actuation_hook = _vulkan_calibration_actuation
    previous_profile_map = dict(previous_profile or {})
    previous_latency = dict(previous_profile_map.get("latency_calibration", {}) or {})
    previous_trace_map = dict(previous_trace_state or {})
    phase_turns = _wrap_turns(
        cycle_payload.get(
            "initial_phase_turns",
            previous_latency.get("actuation_phase_turns", previous_trace_map.get("trace_phase_anchor_turns", 0.0)),
        )
    )
    actuation_meta = _invoke_live_actuation(
        actuation_hook=actuation_hook,
        frame_index=int(len(list(frame_history or []))),
        phase_turns=phase_turns,
        sample_period_s=max(0.00025, target_sample_period_s),
        payload=cycle_payload,
    )
    settle_s = max(0.0, _safe_float(cycle_payload.get("actuation_settle_s", 0.002), 0.002))
    if capture_sleep and settle_s > 0.0:
        time.sleep(settle_s)
    snapshot = snapshot_provider()
    frame = _snapshot_to_telemetry_frame(
        snapshot=dict(snapshot or {}),
        frame_index=int(len(list(frame_history or []))),
        phase_turns=phase_turns,
        sample_period_s=max(0.00025, target_sample_period_s),
        actuation_meta=actuation_meta,
    )
    history_limit = max(4, min(64, int(cycle_payload.get("history_size", 8) or 8)))
    frames = list(frame_history or [])[-(history_limit - 1):]
    frames.append(dict(frame))
    horizon_frames = max(1, int(cycle_payload.get("horizon_frames", 2) or 2))
    program_label = str(cycle_payload.get("program_id", "live_photonic_cycle"))
    pulse_cycles = max(1, int(cycle_payload.get("pulse_cycles", 1) or 1))
    profile = _telemetry_frame_profile(
        frames=frames,
        frame_index=max(0, len(frames) - 1),
        horizon_frames=horizon_frames,
        payload=cycle_payload,
        previous_profile=previous_profile_map,
    )
    encoded_profile = _encode_telemetry_frame_profile(profile)
    execution = _run_encoded_telemetry_resonance(
        encoded_profiles=[encoded_profile],
        pulse_cycles=pulse_cycles,
        program_label=program_label,
        previous_trace_state=previous_trace_map,
    )
    latency_calibration = dict(dict(execution.get("result", {}) or {}).get("latency_calibration", {}) or {})
    memory_basin_state = _build_memory_basin_state(
        frame=frame,
        trace_state=dict(execution.get("trace_state", {}) or {}),
        latency_calibration=latency_calibration,
        payload=cycle_payload,
        previous_memory_basin_state=previous_memory_basin_state,
    )
    scheduler_state = _build_field_scheduler_state(
        frame=frame,
        trace_state=dict(execution.get("trace_state", {}) or {}),
        latency_calibration=latency_calibration,
        memory_basin_state=memory_basin_state,
        payload=cycle_payload,
        previous_scheduler_state=previous_scheduler_state,
    )
    process_state = _build_field_process_state(
        frame=frame,
        trace_state=dict(execution.get("trace_state", {}) or {}),
        latency_calibration=latency_calibration,
        memory_basin_state=memory_basin_state,
        scheduler_state=scheduler_state,
        payload=cycle_payload,
        previous_process_state=previous_process_state,
    )
    return {
        "ok": True,
        "path": "live_photonic_cycle",
        "program_id": program_label,
        "telemetry_source": str(frame.get("source", "live_startup")),
        "frame": dict(frame),
        "frames": [dict(item) for item in frames],
        "profile": dict(profile),
        "encoded_profile": {
            "control_word": int(encoded_profile.get("control_word", 0)),
            "timing_word": int(encoded_profile.get("timing_word", 0)),
            "trajectory_word": int(encoded_profile.get("trajectory_word", 0)),
            "phase_ring_zone_id": int(latency_calibration.get("phase_ring_zone_id", 0) or 0),
        },
        "actuation_summary": _telemetry_actuation_summary([frame]),
        "trace_state": dict(execution.get("trace_state", {})),
        "result": dict(execution.get("result", {})),
        "memory_basin_state": memory_basin_state,
        "scheduler_state": scheduler_state,
        "process_state": process_state,
        "forecast_preview": [
            {
                "timestamp": str(profile.get("timestamp", "")),
                "source": str(profile.get("source", "live_startup")),
                "dominant_metric": str(dict(list(profile.get("dominant_nodes", [{}]))[0]).get("metric", "")),
                "resonance_gate": float(profile.get("resonance_gate", 0.0)),
                "noise_gate": float(profile.get("noise_gate", 0.0)),
                "predicted_latency_s": float(latency_calibration.get("predicted_latency_s", 0.0)),
                "phase_ring_strength": float(latency_calibration.get("phase_ring_strength", 0.0)),
                "zero_point_crossover_gate": float(latency_calibration.get("zero_point_crossover_gate", 0.0)),
                "shared_vector_collapse_gate": float(latency_calibration.get("shared_vector_collapse_gate", 0.0)),
                "scheduler_mode": str(scheduler_state.get("scheduling_mode", "")),
                "active_zone_name": str(scheduler_state.get("active_zone_name", "")),
                "active_basin_name": str(memory_basin_state.get("active_basin_name", "")),
                "process_mode": str(process_state.get("process_mode", "")),
                "mining_resonance_gate": float(process_state.get("mining_resonance_gate", 0.0)),
                "transport_input_mode": str(latency_calibration.get("transport_input_mode", "telemetry_only")),
            }
        ],
        "latency_calibration": latency_calibration,
    }


def _telemetry_metric_value(frame: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return _clamp01(frame.get(key, default))


def _telemetry_metric_companion(frame: Dict[str, Any], key: str) -> float:
    if key == "global_util":
        return _clamp01((_telemetry_metric_value(frame, "gpu_util") + _telemetry_metric_value(frame, "cpu_util")) * 0.5)
    if key == "gpu_util":
        return _telemetry_metric_value(frame, "mem_bw_util")
    if key == "mem_bw_util":
        return _telemetry_metric_value(frame, "gpu_util")
    if key == "cpu_util":
        return _telemetry_metric_value(frame, "global_util")
    return 0.0


def _phase_delta_turns(origin: Any, target: Any) -> float:
    delta = (float(target) - float(origin)) % 1.0
    if delta > 0.5:
        delta -= 1.0
    return float(delta)


def _wrap_turns(value: Any) -> float:
    try:
        return float(value) % 1.0
    except Exception:
        return 0.0


def _phase_flux_transport_predictor(
    frame: Dict[str, Any],
    dominant_nodes: list[Dict[str, Any]],
    predicted_phase_turns: float,
    phase_delta_turns: float,
    headroom: float,
    latency_load: float,
    resonance_gate: float,
    noise_level: float,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    def _normalized_weight(name: str, default: float, weight_sum: float) -> float:
        return _safe_float(payload.get(name, default), default) / max(weight_sum, 1.0e-9)

    node_count = float(len(dominant_nodes) or 1.0)
    dominant_resonance = _clamp01(
        sum(
            float(node.get("actuation_resonance", node.get("resonance", 0.0)))
            for node in dominant_nodes
        )
        / node_count
    )
    dominant_near_term = _clamp01(
        sum(
            float(node.get("actuation_term", node.get("near_term", 0.0)))
            for node in dominant_nodes
        )
        / node_count
    )
    pulse_gate = _clamp01((float(frame.get("pulse", 0.0)) + 1.0) * 0.5)
    anti_pulse_gate = _clamp01((float(frame.get("anti_pulse", 0.0)) + 1.0) * 0.5)
    actuation_load_hint = _clamp01(frame.get("actuation_load_hint", 0.0))
    transport_weight_sum = max(
        1.0e-9,
        _safe_float(payload.get("transport_resonance_weight", 0.36), 0.36)
        + _safe_float(payload.get("transport_gate_weight", 0.22), 0.22)
        + _safe_float(payload.get("transport_noise_inverse_weight", 0.18), 0.18)
        + _safe_float(payload.get("transport_headroom_weight", 0.14), 0.14)
        + _safe_float(payload.get("transport_actuation_weight", 0.10), 0.10),
    )
    transport_resonance_weight = _normalized_weight("transport_resonance_weight", 0.36, transport_weight_sum)
    transport_gate_weight = _normalized_weight("transport_gate_weight", 0.22, transport_weight_sum)
    transport_noise_inverse_weight = _normalized_weight("transport_noise_inverse_weight", 0.18, transport_weight_sum)
    transport_headroom_weight = _normalized_weight("transport_headroom_weight", 0.14, transport_weight_sum)
    transport_actuation_weight = _normalized_weight("transport_actuation_weight", 0.10, transport_weight_sum)
    transport_coherence = _clamp01(
        transport_resonance_weight * float(dominant_resonance)
        + transport_gate_weight * float(resonance_gate)
        + transport_noise_inverse_weight * (1.0 - float(noise_level))
        + transport_headroom_weight * float(headroom)
        + transport_actuation_weight * float(actuation_load_hint)
    )
    observer_damping_min = max(
        0.0,
        _safe_float(payload.get("observer_damping_min", 0.010), 0.010),
    )
    observer_damping_max = max(
        observer_damping_min,
        _safe_float(payload.get("observer_damping_max", 0.020), 0.020),
    )
    observer_damping = observer_damping_min + (
        (observer_damping_max - observer_damping_min)
        * _clamp01(
            0.62 * float(noise_level)
            + 0.38 * (1.0 - float(dominant_resonance))
        )
    )
    transport_damping_gate = _clamp01(1.0 - float(observer_damping) * 24.0)
    flux_weight_sum = max(
        1.0e-9,
        _safe_float(payload.get("flux_gpu_weight", 0.30), 0.30)
        + _safe_float(payload.get("flux_mem_weight", 0.22), 0.22)
        + _safe_float(payload.get("flux_global_weight", 0.14), 0.14)
        + _safe_float(payload.get("flux_cpu_weight", 0.12), 0.12)
        + _safe_float(payload.get("flux_actuation_weight", 0.12), 0.12)
        + _safe_float(payload.get("flux_near_term_weight", 0.10), 0.10),
    )
    flux_gpu_weight = _normalized_weight("flux_gpu_weight", 0.30, flux_weight_sum)
    flux_mem_weight = _normalized_weight("flux_mem_weight", 0.22, flux_weight_sum)
    flux_global_weight = _normalized_weight("flux_global_weight", 0.14, flux_weight_sum)
    flux_cpu_weight = _normalized_weight("flux_cpu_weight", 0.12, flux_weight_sum)
    flux_actuation_weight = _normalized_weight("flux_actuation_weight", 0.12, flux_weight_sum)
    flux_near_term_weight = _normalized_weight("flux_near_term_weight", 0.10, flux_weight_sum)
    flux_transport_term = _clamp01(
        flux_gpu_weight * _telemetry_metric_value(frame, "gpu_util")
        + flux_mem_weight * _telemetry_metric_value(frame, "mem_bw_util")
        + flux_global_weight * _telemetry_metric_value(frame, "global_util")
        + flux_cpu_weight * _telemetry_metric_value(frame, "cpu_util")
        + flux_actuation_weight * float(actuation_load_hint)
        + flux_near_term_weight * float(dominant_near_term)
    )
    flux_transport_delta_turns = (
        (
            (float(flux_transport_term) - float(latency_load))
            * (0.05 + 0.10 * float(transport_coherence))
            * float(transport_damping_gate)
        )
        + ((float(pulse_gate) - float(anti_pulse_gate)) * 0.01 * (0.5 + 0.5 * float(headroom)))
    )
    transport_candidate_phase_turns = _wrap_turns(
        float(predicted_phase_turns)
        + float(phase_delta_turns) * 0.25
        + float(flux_transport_delta_turns)
    )
    reverse_transport_min_coherence = _clamp01(
        payload.get("reverse_transport_min_coherence", 0.58)
    )
    reverse_transport_max_noise = _clamp01(
        payload.get("reverse_transport_max_noise", 0.18)
    )
    reverse_transport_min_resonance = _clamp01(
        payload.get("reverse_transport_min_resonance", 0.45)
    )
    reverse_transport_gate = 1.0 if (
        float(transport_coherence) >= float(reverse_transport_min_coherence)
        and float(noise_level) <= float(reverse_transport_max_noise)
        and float(resonance_gate) >= float(reverse_transport_min_resonance)
    ) else 0.0
    reverse_delta_theta_hat = float(reverse_transport_gate) * _phase_delta_turns(
        float(predicted_phase_turns),
        float(transport_candidate_phase_turns),
    )
    phase_transport_term = (
        float(phase_delta_turns) * (0.70 + 0.20 * float(pulse_gate))
        + float(reverse_delta_theta_hat) * (0.45 + 0.35 * float(transport_coherence))
    )
    transported_phase_turns = _wrap_turns(
        float(predicted_phase_turns)
        + float(phase_transport_term) * 0.50
        + float(flux_transport_delta_turns) * (0.40 + 0.20 * float(actuation_load_hint))
    )
    return {
        "transport_input_mode": "actuation_adjusted_telemetry" if actuation_load_hint > 0.0 else "telemetry_only",
        "raw_telemetry_only": bool(actuation_load_hint <= 0.0),
        "transport_coherence": float(transport_coherence),
        "observer_damping": float(observer_damping),
        "transport_damping_gate": float(transport_damping_gate),
        "flux_transport_term": float(flux_transport_term),
        "flux_transport_delta_turns": float(flux_transport_delta_turns),
        "transport_candidate_phase_turns": float(transport_candidate_phase_turns),
        "reverse_transport_gate": float(reverse_transport_gate),
        "reverse_delta_theta_hat": float(reverse_delta_theta_hat),
        "phase_transport_term": float(phase_transport_term),
        "transported_phase_turns": float(transported_phase_turns),
    }


def _telemetry_latency_load(frame: Dict[str, Any]) -> float:
    global_util = _telemetry_metric_value(frame, "global_util")
    gpu_util = _telemetry_metric_value(frame, "gpu_util")
    mem_bw_util = _telemetry_metric_value(frame, "mem_bw_util")
    cpu_util = _telemetry_metric_value(frame, "cpu_util")
    headroom = _clamp01(1.0 - max(global_util, gpu_util, mem_bw_util, cpu_util))
    return _clamp01(
        0.34 * gpu_util
        + 0.26 * mem_bw_util
        + 0.18 * global_util
        + 0.10 * cpu_util
        + 0.12 * (1.0 - headroom)
    )


def _telemetry_latency_calibration(
    frame: Dict[str, Any],
    dominant_nodes: list[Dict[str, Any]],
    base_phase_turns: float,
    noise_gate: float,
    payload: Dict[str, Any],
    previous_predicted_trajectory_9d: list[float] | None = None,
) -> Dict[str, Any]:
    global_util = _telemetry_metric_value(frame, "global_util")
    gpu_util = _telemetry_metric_value(frame, "gpu_util")
    mem_bw_util = _telemetry_metric_value(frame, "mem_bw_util")
    cpu_util = _telemetry_metric_value(frame, "cpu_util")
    headroom = _clamp01(1.0 - max(global_util, gpu_util, mem_bw_util, cpu_util))
    latency_load = _telemetry_latency_load(frame)
    sample_period_s = max(
        0.00025,
        _safe_float(
            frame.get("sample_period_s", payload.get("telemetry_sample_period_s", payload.get("tick_s", 0.004))),
            0.004,
        ),
    )
    pulse_generation_ns = max(
        5000.0,
        _safe_float(payload.get("assumed_pulse_generation_ns", 180000.0), 180000.0),
    )
    kernel_request_ns = max(
        5000.0,
        _safe_float(payload.get("assumed_kernel_request_ns", 260000.0), 260000.0),
    )
    kernel_actuation_ns = max(
        5000.0,
        _safe_float(payload.get("assumed_kernel_actuation_ns", 120000.0), 120000.0),
    )
    node_count = float(len(dominant_nodes) or 1)
    dominant_resonance = _clamp01(
        sum(float(node.get("resonance", 0.0)) for node in dominant_nodes) / node_count
    )
    dominant_noise = _clamp01(
        sum(float(node.get("noise", 0.0)) for node in dominant_nodes) / node_count
    )
    ada_constants = ada_effective_constants({
        "global_util": float(global_util),
        "gpu_util": float(gpu_util),
        "mem_bw_util": float(mem_bw_util),
        "cpu_util": float(cpu_util),
    })
    predicted_phase_turns = ada_phase_update(float(base_phase_turns), float(headroom))
    phase_delta_turns = _phase_delta_turns(base_phase_turns, predicted_phase_turns)
    kernel_latency_s = ada_latency_kernel(float(headroom), float(latency_load), float(predicted_phase_turns))
    pulse_generation_s = (pulse_generation_ns * 1e-9) * (
        1.0
        + 0.48 * float(dominant_noise)
        + 0.24 * (1.0 - float(dominant_resonance))
        + 0.18 * float(latency_load)
    )
    kernel_request_s = (kernel_request_ns * 1e-9) * (
        1.0
        + 0.52 * float(latency_load)
        + 0.22 * float(dominant_noise)
        + 0.14 * (1.0 - float(headroom))
    )
    transport_prediction = _phase_flux_transport_predictor(
        frame=frame,
        dominant_nodes=dominant_nodes,
        predicted_phase_turns=predicted_phase_turns,
        phase_delta_turns=phase_delta_turns,
        headroom=headroom,
        latency_load=latency_load,
        resonance_gate=dominant_resonance,
        noise_level=dominant_noise,
        payload=payload,
    )
    field_dynamics = _compute_axis_field_dynamics(
        frequency_norm=gpu_util,
        amplitude_norm=mem_bw_util,
        phase_turns=predicted_phase_turns,
        resonance=dominant_resonance,
        temporal_overlap=1.0 - dominant_noise,
        flux_term=_clamp01(0.5 * latency_load + 0.5 * float(transport_prediction.get("flux_transport_term", 0.0))),
        vector_x=gpu_util - global_util,
        vector_y=mem_bw_util - cpu_util,
        vector_z=(dominant_resonance - dominant_noise) + (0.5 * float(transport_prediction.get("phase_transport_term", 0.0))),
        energy_hint=_clamp01(frame.get("actuation_load_hint", 0.0)),
    )
    phase_ring_state = _compute_phase_ring_state(
        phase_turns=predicted_phase_turns,
        field_dynamics=field_dynamics,
        frequency_norm=gpu_util,
        amplitude_norm=mem_bw_util,
        resonance=dominant_resonance,
        temporal_overlap=1.0 - dominant_noise,
        flux_term=_clamp01(0.5 * latency_load + 0.5 * float(transport_prediction.get("flux_transport_term", 0.0))),
    )
    field_dynamics.update(phase_ring_state)
    gradient_state = _compose_frequency_gradient_9d(
        frequency_norm=gpu_util,
        amplitude_norm=mem_bw_util,
        phase_turns=field_dynamics.get("phase_injection_turns", predicted_phase_turns),
        field_dynamics=field_dynamics,
    )
    field_dynamics.update(gradient_state)
    if any(
        (
            _clamp01(frame.get("phase_ring_strength", 0.0)) > 0.0,
            _clamp01(frame.get("shared_vector_collapse_gate", 0.0)) > 0.0,
            abs(_clamp_signed(frame.get("phase_injection_delta_turns", frame.get("gpu_pulse_phase_effect", 0.0)), 0.5)) > 0.0,
        )
    ):
        field_dynamics["gpu_pulse_phase_effect"] = _clamp_signed(
            _lerp(
                field_dynamics.get("gpu_pulse_phase_effect", 0.0),
                frame.get("gpu_pulse_phase_effect", frame.get("phase_injection_delta_turns", 0.0)),
                0.55,
            ),
            0.5,
        )
        field_dynamics["phase_injection_delta_turns"] = _clamp_signed(
            _lerp(
                field_dynamics.get("phase_injection_delta_turns", 0.0),
                frame.get("phase_injection_delta_turns", field_dynamics.get("gpu_pulse_phase_effect", 0.0)),
                0.55,
            ),
            0.5,
        )
        field_dynamics["phase_injection_turns"] = _wrap_turns(
            _lerp(
                field_dynamics.get("phase_injection_turns", predicted_phase_turns),
                frame.get("phase_injection_turns", predicted_phase_turns),
                0.55,
            )
        )
        for key in (
            "phase_ring_closure",
            "phase_ring_density",
            "phase_ring_strength",
            "zero_point_crossover_gate",
            "shared_vector_collapse_gate",
            "shared_vector_phase_lock",
            "inertial_basin_strength",
            "wavelength_norm",
            "orientation_alignment",
            "rotational_velocity_norm",
            "relative_temporal_position",
            "zero_point_line_distance",
            "field_interference_norm",
            "resonant_interception_inertia",
            "temporal_relativity_norm",
        ):
            field_dynamics[key] = _clamp01(_lerp(field_dynamics.get(key, 0.0), frame.get(key, 0.0), 0.55))
        incoming_gradient = [float(_clamp01(v)) for v in list(frame.get("frequency_gradient_9d", []))[:9]]
        if incoming_gradient:
            base_gradient = [float(_clamp01(v)) for v in list(field_dynamics.get("frequency_gradient_9d", []))[:9]]
            while len(base_gradient) < 9:
                base_gradient.append(0.0)
            while len(incoming_gradient) < 9:
                incoming_gradient.append(0.0)
            blended_gradient = [
                float(_clamp01(_lerp(base_gradient[idx], incoming_gradient[idx], 0.55)))
                for idx in range(9)
            ]
            field_dynamics["frequency_gradient_9d"] = blended_gradient
            field_dynamics["field_gradient_9d"] = list(blended_gradient)
        if str(frame.get("gradient_spectral_id", "")):
            field_dynamics["gradient_spectral_id"] = str(frame.get("gradient_spectral_id", ""))
        if int(frame.get("phase_ring_zone_id", 0) or 0):
            field_dynamics["phase_ring_zone_id"] = int(frame.get("phase_ring_zone_id", 0) or 0)
    provisional_feedback_gate = _clamp01(
        0.40 * float(transport_prediction.get("transport_damping_gate", 0.0))
        + 0.24 * float(transport_prediction.get("reverse_transport_gate", 0.0))
        + 0.20 * (1.0 - float(dominant_noise))
        + 0.16 * float(headroom)
    )
    provisional_coherence_gate = _clamp01(
        0.54 * float(dominant_resonance)
        + 0.46 * float(transport_prediction.get("transport_coherence", 0.0))
    )
    provisional_trajectory = _compose_photonic_trajectory_9d(
        phase_turns=predicted_phase_turns,
        field_dynamics=field_dynamics,
        flux_term=transport_prediction.get("flux_transport_term", 0.0),
        coherence_gate=provisional_coherence_gate,
        feedback_gate=provisional_feedback_gate,
    )
    trajectory_prediction = _predict_photonic_trajectory_9d(
        current_trajectory_9d=list(provisional_trajectory.get("trajectory_9d", []) or []),
        previous_predicted_trajectory_9d=previous_predicted_trajectory_9d,
        field_dynamics=field_dynamics,
        transport_prediction=transport_prediction,
        headroom=headroom,
        latency_load=latency_load,
        noise_gate=dominant_noise,
    )
    pulse_time_scale = _clamp01(sample_period_s / max(sample_period_s + kernel_latency_s + pulse_generation_s, 1.0e-9))
    request_time_scale = _clamp01(kernel_latency_s / max(sample_period_s + kernel_latency_s + kernel_request_s, 1.0e-9))
    actuation_time_scale = _clamp01(1.0 - min(1.0, request_time_scale * 0.5))
    pulse_generation_scale, pulse_generation_weights = _derived_temporal_mix({
        "temporal_coupling": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
        "spin": float(field_dynamics.get("spin_momentum_score", 0.0)),
        "temporal_relativity": float(field_dynamics.get("temporal_relativity_norm", 0.0)),
        "phase_ring_strength": float(field_dynamics.get("phase_ring_strength", 0.0)),
        "zero_point_alignment": 1.0 - float(field_dynamics.get("zero_point_line_distance", 0.0)),
    })
    pulse_generation_s *= (1.0 + (pulse_generation_scale * pulse_time_scale))
    kernel_request_scale, kernel_request_weights = _derived_temporal_mix({
        "inertial_mass": float(field_dynamics.get("inertial_mass_proxy", 0.0)),
        "relativistic": float(field_dynamics.get("relativistic_correlation", 0.0)),
        "interception_inertia": float(field_dynamics.get("resonant_interception_inertia", 0.0)),
        "transport_damping": float(transport_prediction.get("transport_damping_gate", 0.0)),
        "shared_phase_lock": float(field_dynamics.get("shared_vector_phase_lock", 0.0)),
    })
    kernel_request_s *= (1.0 + (kernel_request_scale * request_time_scale))
    kernel_actuation_drive, kernel_actuation_weights = _derived_temporal_mix({
        "phase_delta": abs(float(phase_delta_turns)),
        "resonance_inverse": 1.0 - float(dominant_resonance),
        "noise_gate": float(noise_gate),
        "reverse_delta": abs(float(transport_prediction.get("reverse_delta_theta_hat", 0.0))),
        "transport_inverse": 1.0 - float(transport_prediction.get("transport_coherence", 0.0)),
        "spin": float(field_dynamics.get("spin_momentum_score", 0.0)),
        "temporal": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
        "relativistic": float(field_dynamics.get("relativistic_correlation", 0.0)),
        "phase_ring_strength": float(field_dynamics.get("phase_ring_strength", 0.0)),
        "collapse_gate": float(field_dynamics.get("shared_vector_collapse_gate", 0.0)),
        "interference": float(field_dynamics.get("field_interference_norm", 0.0)),
        "interception_inertia": float(field_dynamics.get("resonant_interception_inertia", 0.0)),
    })
    kernel_actuation_s = (kernel_actuation_ns * 1e-9) * (1.0 + (kernel_actuation_drive * actuation_time_scale))
    predicted_latency_s = (
        float(kernel_latency_s)
        + float(pulse_generation_s)
        + float(kernel_request_s)
        + float(kernel_actuation_s)
    )
    actuation_horizon_frames = min(
        8.0,
        max(0.0, float(predicted_latency_s) / max(sample_period_s, 1e-9)),
    )
    horizon_gate = _clamp01(1.0 - min(1.0, float(actuation_horizon_frames) / 4.0))
    actuation_compensation, actuation_compensation_weights = _derived_temporal_mix({
        "dominant_resonance": float(dominant_resonance),
        "noise_inverse": 1.0 - float(dominant_noise),
        "headroom": float(headroom),
        "latency_inverse": 1.0 - float(latency_load),
        "horizon_gate": float(horizon_gate),
        "transport_coherence": float(transport_prediction.get("transport_coherence", 0.0)),
        "transport_damping": float(transport_prediction.get("transport_damping_gate", 0.0)),
        "temporal_coupling": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
        "inertia_inverse": 1.0 - float(field_dynamics.get("inertial_mass_proxy", 0.0)),
        "trajectory_conservation": float(trajectory_prediction.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_density": float(trajectory_prediction.get("trajectory_sequence_density", 0.0)),
        "trajectory_expansion": float(trajectory_prediction.get("trajectory_expansion_term", 0.0)),
        "trajectory_prediction": float(trajectory_prediction.get("trajectory_prediction_alignment", 0.0)),
        "phase_ring_strength": float(field_dynamics.get("phase_ring_strength", 0.0)),
        "shared_phase_lock": float(field_dynamics.get("shared_vector_phase_lock", 0.0)),
        "zero_point_crossover": float(field_dynamics.get("zero_point_crossover_gate", 0.0)),
        "temporal_relativity": float(field_dynamics.get("temporal_relativity_norm", 0.0)),
        "zero_point_alignment": 1.0 - float(field_dynamics.get("zero_point_line_distance", 0.0)),
    })
    actuation_phase_turns = _wrap_turns(
        float(transport_prediction.get("transported_phase_turns", predicted_phase_turns))
        + float(actuation_compensation) * 0.125
        + float(transport_prediction.get("reverse_delta_theta_hat", 0.0)) * 0.20
        + float(field_dynamics.get("gpu_pulse_phase_effect", 0.0)) * 0.25
    )
    feedback_gate, feedback_gate_weights = _derived_temporal_mix({
        "actuation_compensation": float(actuation_compensation),
        "transport_damping": float(transport_prediction.get("transport_damping_gate", 0.0)),
        "reverse_transport": float(transport_prediction.get("reverse_transport_gate", 0.0)),
        "headroom": float(headroom),
        "inertial_basin": float(field_dynamics.get("inertial_basin_strength", 0.0)),
        "temporal_relativity": float(field_dynamics.get("temporal_relativity_norm", 0.0)),
    })
    coherence_gate, coherence_gate_weights = _derived_temporal_mix({
        "dominant_resonance": float(dominant_resonance),
        "transport_coherence": float(transport_prediction.get("transport_coherence", 0.0)),
        "trajectory_prediction": float(trajectory_prediction.get("trajectory_prediction_alignment", 0.0)),
        "phase_ring_closure": float(field_dynamics.get("phase_ring_closure", 0.0)),
        "shared_phase_lock": float(field_dynamics.get("shared_vector_phase_lock", 0.0)),
        "orientation": float(field_dynamics.get("orientation_alignment", 0.0)),
    })
    trajectory_state = _compose_photonic_trajectory_9d(
        phase_turns=actuation_phase_turns,
        field_dynamics=field_dynamics,
        flux_term=transport_prediction.get("flux_transport_term", 0.0),
        coherence_gate=coherence_gate,
        feedback_gate=feedback_gate,
    )
    trajectory_prediction = _predict_photonic_trajectory_9d(
        current_trajectory_9d=list(trajectory_state.get("trajectory_9d", []) or []),
        previous_predicted_trajectory_9d=previous_predicted_trajectory_9d,
        field_dynamics=field_dynamics,
        transport_prediction=transport_prediction,
        headroom=headroom,
        latency_load=latency_load,
        noise_gate=dominant_noise,
    )
    field_dynamics = dict(field_dynamics)
    field_dynamics.update({
        "trajectory_9d": list(trajectory_state.get("trajectory_9d", []) or []),
        "predicted_trajectory_9d": list(trajectory_prediction.get("predicted_trajectory_9d", []) or []),
        "trajectory_velocity_9d": list(trajectory_prediction.get("trajectory_velocity_9d", []) or []),
        "trajectory_spectral_id": str(trajectory_state.get("trajectory_spectral_id", "")),
        "predicted_trajectory_spectral_id": str(trajectory_prediction.get("trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": float(trajectory_prediction.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_prediction_alignment": float(trajectory_prediction.get("trajectory_prediction_alignment", 0.0)),
        "trajectory_expansion_term": float(trajectory_prediction.get("trajectory_expansion_term", 0.0)),
        "trajectory_sequence_density": float(trajectory_prediction.get("trajectory_sequence_density", 0.0)),
        "trajectory_noise_feedback_norm": float(trajectory_prediction.get("trajectory_noise_feedback_norm", 0.0)),
        "spectra_sig9": list(trajectory_state.get("spectra_sig9", []) or []),
        "spectra_9d": list(trajectory_state.get("spectra_9d", []) or []),
    })
    throughput_hz = 1.0 / max(float(predicted_latency_s), 1e-9)
    timing_word = _stable_word({
        "headroom": round(float(headroom), 6),
        "latency_load": round(float(latency_load), 6),
        "predicted_latency_s": round(float(predicted_latency_s), 6),
        "actuation_compensation": round(float(actuation_compensation), 6),
        "phase": round(float(actuation_phase_turns), 6),
        "trajectory_spectral_id": str(trajectory_state.get("trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": round(float(trajectory_prediction.get("trajectory_conservation_alignment", 0.0)), 6),
    })
    return {
        "ada_constants": {
            "h_eff": float(ada_constants.get("h_eff", 0.0)),
            "k_B_eff": float(ada_constants.get("k_B_eff", 0.0)),
            "c_eff": float(ada_constants.get("c_eff", 0.0)),
        },
        "headroom": float(headroom),
        "latency_load": float(latency_load),
        "sample_period_s": float(sample_period_s),
        "kernel_latency_s": float(kernel_latency_s),
        "pulse_generation_s": float(pulse_generation_s),
        "kernel_request_s": float(kernel_request_s),
        "kernel_actuation_s": float(kernel_actuation_s),
        "predicted_latency_s": float(predicted_latency_s),
        "predicted_phase_turns": float(predicted_phase_turns),
        "phase_delta_turns": float(phase_delta_turns),
        "actuation_phase_turns": float(actuation_phase_turns),
        "actuation_horizon_frames": float(actuation_horizon_frames),
        "actuation_compensation": float(actuation_compensation),
        "latency_temporal_constant_weights": {
            "pulse_generation": dict(pulse_generation_weights),
            "kernel_request": dict(kernel_request_weights),
            "kernel_actuation": dict(kernel_actuation_weights),
            "actuation_compensation": dict(actuation_compensation_weights),
            "feedback_gate": dict(feedback_gate_weights),
            "coherence_gate": dict(coherence_gate_weights),
        },
        "throughput_hz": float(throughput_hz),
        "transport_input_mode": str(transport_prediction.get("transport_input_mode", "telemetry_only")),
        "raw_telemetry_only": bool(transport_prediction.get("raw_telemetry_only", True)),
        "transport_coherence": float(transport_prediction.get("transport_coherence", 0.0)),
        "observer_damping": float(transport_prediction.get("observer_damping", 0.0)),
        "transport_damping_gate": float(transport_prediction.get("transport_damping_gate", 0.0)),
        "flux_transport_term": float(transport_prediction.get("flux_transport_term", 0.0)),
        "flux_transport_delta_turns": float(transport_prediction.get("flux_transport_delta_turns", 0.0)),
        "transport_candidate_phase_turns": float(transport_prediction.get("transport_candidate_phase_turns", 0.0)),
        "reverse_transport_gate": float(transport_prediction.get("reverse_transport_gate", 0.0)),
        "reverse_delta_theta_hat": float(transport_prediction.get("reverse_delta_theta_hat", 0.0)),
        "phase_transport_term": float(transport_prediction.get("phase_transport_term", 0.0)),
        "transported_phase_turns": float(transport_prediction.get("transported_phase_turns", 0.0)),
        "axis_scale_x": float(field_dynamics.get("axis_scale_x", 0.0)),
        "axis_scale_y": float(field_dynamics.get("axis_scale_y", 0.0)),
        "axis_scale_z": float(field_dynamics.get("axis_scale_z", 0.0)),
        "vector_energy": float(field_dynamics.get("vector_energy", 0.0)),
        "temporal_coupling_moment": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
        "inertial_mass_proxy": float(field_dynamics.get("inertial_mass_proxy", 0.0)),
        "relativistic_correlation": float(field_dynamics.get("relativistic_correlation", 0.0)),
        "spin_axis_x": float(field_dynamics.get("spin_axis_x", 0.0)),
        "spin_axis_y": float(field_dynamics.get("spin_axis_y", 0.0)),
        "spin_axis_z": float(field_dynamics.get("spin_axis_z", 0.0)),
        "spin_momentum_score": float(field_dynamics.get("spin_momentum_score", 0.0)),
        "gpu_pulse_phase_effect": float(field_dynamics.get("gpu_pulse_phase_effect", 0.0)),
        "phase_injection_delta_turns": float(field_dynamics.get("phase_injection_delta_turns", 0.0)),
        "phase_injection_turns": float(field_dynamics.get("phase_injection_turns", predicted_phase_turns)),
        "phase_ring_closure": float(field_dynamics.get("phase_ring_closure", 0.0)),
        "phase_ring_density": float(field_dynamics.get("phase_ring_density", 0.0)),
        "phase_ring_strength": float(field_dynamics.get("phase_ring_strength", 0.0)),
        "zero_point_crossover_gate": float(field_dynamics.get("zero_point_crossover_gate", 0.0)),
        "shared_vector_collapse_gate": float(field_dynamics.get("shared_vector_collapse_gate", 0.0)),
        "shared_vector_phase_lock": float(field_dynamics.get("shared_vector_phase_lock", 0.0)),
        "inertial_basin_strength": float(field_dynamics.get("inertial_basin_strength", 0.0)),
        "wavelength_norm": float(field_dynamics.get("wavelength_norm", 0.0)),
        "orientation_alignment": float(field_dynamics.get("orientation_alignment", 0.0)),
        "rotational_velocity_norm": float(field_dynamics.get("rotational_velocity_norm", 0.0)),
        "relative_temporal_position": float(field_dynamics.get("relative_temporal_position", 0.0)),
        "zero_point_line_distance": float(field_dynamics.get("zero_point_line_distance", 0.0)),
        "field_interference_norm": float(field_dynamics.get("field_interference_norm", 0.0)),
        "resonant_interception_inertia": float(field_dynamics.get("resonant_interception_inertia", 0.0)),
        "temporal_relativity_norm": float(field_dynamics.get("temporal_relativity_norm", 0.0)),
        "temporal_relativity_vector": list(field_dynamics.get("temporal_relativity_vector", []) or []),
        "phase_ring_zone_id": int(field_dynamics.get("phase_ring_zone_id", 0) or 0),
        "frequency_gradient_9d": list(field_dynamics.get("frequency_gradient_9d", []) or []),
        "field_gradient_9d": list(field_dynamics.get("field_gradient_9d", []) or []),
        "gradient_spectral_id": str(field_dynamics.get("gradient_spectral_id", "")),
        "trajectory_9d": list(field_dynamics.get("trajectory_9d", []) or []),
        "predicted_trajectory_9d": list(field_dynamics.get("predicted_trajectory_9d", []) or []),
        "trajectory_velocity_9d": list(field_dynamics.get("trajectory_velocity_9d", []) or []),
        "trajectory_spectral_id": str(field_dynamics.get("trajectory_spectral_id", "")),
        "predicted_trajectory_spectral_id": str(field_dynamics.get("predicted_trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": float(field_dynamics.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_prediction_alignment": float(field_dynamics.get("trajectory_prediction_alignment", 0.0)),
        "trajectory_expansion_term": float(field_dynamics.get("trajectory_expansion_term", 0.0)),
        "trajectory_sequence_density": float(field_dynamics.get("trajectory_sequence_density", 0.0)),
        "trajectory_noise_feedback_norm": float(field_dynamics.get("trajectory_noise_feedback_norm", 0.0)),
        "spectra_sig9": list(field_dynamics.get("spectra_sig9", []) or []),
        "spectra_9d": list(field_dynamics.get("spectra_9d", []) or []),
        "field_dynamics": field_dynamics,
        "timing_word": int(timing_word),
    }


def _telemetry_node_profile(
    frames: list[Dict[str, Any]],
    frame_index: int,
    metric_key: str,
    horizon_frames: int,
) -> Dict[str, Any]:
    current = dict(frames[frame_index])
    prev = dict(frames[max(0, frame_index - 1)])
    nxt = dict(frames[min(len(frames) - 1, frame_index + max(1, horizon_frames))])
    cur = _telemetry_metric_value(current, metric_key)
    prev_v = _telemetry_metric_value(prev, metric_key, cur)
    next_v = _telemetry_metric_value(nxt, metric_key, cur)
    slope = next_v - prev_v
    noise = abs(cur - prev_v)
    companion = _telemetry_metric_companion(current, metric_key)
    coupling = _clamp01(1.0 - abs(cur - companion))
    pulse_gate = _clamp01((float(current.get("pulse", 0.0)) + 1.0) * 0.5)
    anti_pulse_gate = _clamp01((float(current.get("anti_pulse", 0.0)) + 1.0) * 0.5)
    slope_gate = _clamp01(1.0 - min(1.0, abs(slope) * 2.0))
    noise_gate = _clamp01(1.0 - min(1.0, noise * 3.0))
    near_term = _clamp01(
        cur
        + slope * (0.30 + 0.45 * pulse_gate)
        - noise * (0.18 + 0.34 * (1.0 - noise_gate))
        + (coupling - 0.5) * 0.08
    )
    resonance = _clamp01(
        0.24 * cur
        + 0.22 * near_term
        + 0.18 * noise_gate
        + 0.14 * slope_gate
        + 0.12 * pulse_gate
        + 0.10 * coupling
    )
    node_word = _stable_word({
        "metric": metric_key,
        "frame": frame_index,
        "resonance": round(resonance, 6),
        "near_term": round(near_term, 6),
        "noise": round(noise, 6),
    })
    return {
        "metric": str(metric_key),
        "frame_index": int(frame_index),
        "value": float(cur),
        "prev_value": float(prev_v),
        "next_value": float(next_v),
        "slope": float(slope),
        "noise": float(noise),
        "coupling": float(coupling),
        "pulse_gate": float(pulse_gate),
        "anti_pulse_gate": float(anti_pulse_gate),
        "near_term": float(near_term),
        "resonance": float(resonance),
        "node_word": int(node_word),
    }


def _telemetry_frame_profile(
    frames: list[Dict[str, Any]],
    frame_index: int,
    horizon_frames: int,
    payload: Dict[str, Any] | None = None,
    previous_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    profile_payload = dict(payload or {})
    frame = dict(frames[frame_index])
    nodes = [
        _telemetry_node_profile(frames, frame_index, metric_key, horizon_frames)
        for metric_key in _TELEMETRY_RESONANCE_METRICS
    ]
    nodes.sort(key=lambda item: (float(item.get("resonance", 0.0)), float(item.get("near_term", 0.0))), reverse=True)
    seed_nodes = list(nodes[:3])
    seed_resonance_gate = _clamp01(sum(float(node.get("resonance", 0.0)) for node in seed_nodes) / float(len(seed_nodes) or 1))
    noise_gate = _clamp01(sum(float(node.get("noise", 0.0)) for node in nodes) / float(len(nodes) or 1))
    pulse_gate = _clamp01((float(frame.get("pulse", 0.0)) + 1.0) * 0.5)
    base_phase_turns = (
        0.44 * pulse_gate
        + 0.28 * seed_resonance_gate
        + 0.18 * _clamp01(_telemetry_metric_value(frame, "gpu_util") - _telemetry_metric_value(frame, "cpu_util") + 0.5)
        + 0.10 * _clamp01(_telemetry_metric_value(frame, "mem_bw_util") - _telemetry_metric_value(frame, "global_util") + 0.5)
    ) % 1.0
    previous_latency = dict(previous_profile.get("latency_calibration", {}) or {}) if isinstance(previous_profile, dict) else {}
    previous_predicted_trajectory_9d = list(
        previous_latency.get("predicted_trajectory_9d", previous_latency.get("trajectory_9d", [])) or []
    )
    latency_calibration = _telemetry_latency_calibration(
        frame=frame,
        dominant_nodes=seed_nodes,
        base_phase_turns=base_phase_turns,
        noise_gate=noise_gate,
        payload=profile_payload,
        previous_predicted_trajectory_9d=previous_predicted_trajectory_9d,
    )
    field_dynamics = dict(latency_calibration.get("field_dynamics", {}) or {})
    actuation_horizon = float(latency_calibration.get("actuation_horizon_frames", 0.0))
    actuation_compensation = float(latency_calibration.get("actuation_compensation", 0.0))
    phase_transport_term = float(latency_calibration.get("phase_transport_term", 0.0))
    flux_transport_term = float(latency_calibration.get("flux_transport_term", 0.0))
    transport_coherence = float(latency_calibration.get("transport_coherence", 0.0))
    transport_damping_gate = float(latency_calibration.get("transport_damping_gate", 0.0))
    reverse_transport_gate = float(latency_calibration.get("reverse_transport_gate", 0.0))
    observer_damping = float(latency_calibration.get("observer_damping", 0.0))
    temporal_coupling = float(field_dynamics.get("temporal_coupling_moment", 0.0))
    inertial_mass = float(field_dynamics.get("inertial_mass_proxy", 0.0))
    spin_score = float(field_dynamics.get("spin_momentum_score", 0.0))
    phase_shift = _phase_delta_turns(
        base_phase_turns,
        float(latency_calibration.get("actuation_phase_turns", base_phase_turns)),
    )
    enriched_nodes: list[Dict[str, Any]] = []
    for node in nodes:
        actuation_term_raw, actuation_term_weights = _derived_temporal_mix({
            "near_term": float(node.get("near_term", 0.0)),
            "slope_drive": _clamp_signed(float(node.get("slope", 0.0)) * min(1.0, actuation_horizon / float(max(1, horizon_frames)))),
            "noise_drag": _clamp_signed(-float(node.get("noise", 0.0)) * (1.0 - actuation_compensation)),
            "observer_drag": _clamp_signed(-float(observer_damping)),
            "compensation_bias": _clamp_signed(actuation_compensation - 0.5),
            "phase_shift": _clamp_signed(float(phase_shift), 1.0),
            "phase_transport": _clamp_signed(float(phase_transport_term), 1.0),
            "flux_transport": _clamp_signed(float(flux_transport_term), 1.0),
            "temporal_coupling": float(temporal_coupling),
            "inertia_inverse": 1.0 - float(inertial_mass),
            "spin": float(spin_score),
        }, signed=True, limit=1.0)
        actuation_term = _clamp01(actuation_term_raw)
        actuation_resonance, actuation_resonance_weights = _derived_temporal_mix({
            "resonance": float(node.get("resonance", 0.0)),
            "actuation_term": float(actuation_term),
            "compensation": float(actuation_compensation),
            "noise_inverse": 1.0 - float(noise_gate),
            "horizon_gate": _clamp01(1.0 - min(1.0, actuation_horizon / 4.0)),
            "transport_coherence": float(transport_coherence),
            "reverse_transport": float(reverse_transport_gate),
            "temporal_coupling": float(temporal_coupling),
            "spin": float(spin_score),
        })
        enriched_node = dict(node)
        enriched_node["actuation_term"] = float(actuation_term)
        enriched_node["actuation_resonance"] = float(actuation_resonance)
        enriched_node["latency_compensation"] = float(actuation_compensation)
        enriched_node["phase_transport_term"] = float(phase_transport_term)
        enriched_node["flux_transport_term"] = float(flux_transport_term)
        enriched_node["transport_coherence"] = float(transport_coherence)
        enriched_node["transport_damping_gate"] = float(transport_damping_gate)
        enriched_node["reverse_transport_gate"] = float(reverse_transport_gate)
        enriched_node["actuation_temporal_constant_weights"] = {
            "actuation_term": dict(actuation_term_weights),
            "actuation_resonance": dict(actuation_resonance_weights),
        }
        enriched_nodes.append(enriched_node)
    enriched_nodes.sort(
        key=lambda item: (
            float(item.get("actuation_resonance", 0.0)),
            float(item.get("actuation_term", 0.0)),
        ),
        reverse=True,
    )
    dominant_nodes = list(enriched_nodes[:3])
    resonance_gate = _clamp01(
        sum(float(node.get("actuation_resonance", node.get("resonance", 0.0))) for node in dominant_nodes)
        / float(len(dominant_nodes) or 1)
    )
    phase_turns = float(latency_calibration.get("actuation_phase_turns", base_phase_turns))
    forecast_frame = {
        node["metric"]: float(node.get("actuation_term", node.get("near_term", 0.0)))
        for node in enriched_nodes
    }
    forecast_frame["pulse"] = float(frame.get("pulse", 0.0))
    forecast_frame["anti_pulse"] = float(frame.get("anti_pulse", 0.0))
    return {
        "frame_index": int(frame_index),
        "timestamp": str(frame.get("timestamp", "")),
        "source": str(frame.get("source", "unknown")),
        "frame": frame,
        "nodes": enriched_nodes,
        "dominant_nodes": dominant_nodes,
        "forecast_frame": forecast_frame,
        "resonance_gate": float(resonance_gate),
        "noise_gate": float(noise_gate),
        "base_phase_turns": float(base_phase_turns),
        "phase_turns": float(phase_turns),
        "field_dynamics": field_dynamics,
        "latency_calibration": latency_calibration,
        "profile_temporal_constant_weights": dict(latency_calibration.get("latency_temporal_constant_weights", {}) or {}),
    }


def _build_telemetry_profiles(
    frames: list[Dict[str, Any]],
    horizon_frames: int,
    payload: Dict[str, Any],
) -> list[Dict[str, Any]]:
    profiles: list[Dict[str, Any]] = []
    previous_profile: Dict[str, Any] = {}
    for frame_index in range(len(frames)):
        profile = _telemetry_frame_profile(
            frames,
            frame_index,
            horizon_frames,
            payload,
            previous_profile=previous_profile,
        )
        profiles.append(profile)
        previous_profile = profile
    return profiles


def _encode_telemetry_frame_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    dominant_nodes = list(profile.get("dominant_nodes", []) or [])
    if not dominant_nodes:
        dominant_nodes = [{
            "metric": "global_util",
            "resonance": 0.0,
            "near_term": 0.0,
            "node_word": 0,
        }]
    dominant_metric = str(dominant_nodes[0].get("metric", "global_util"))
    opcode_id = _microprocess_opcode_id({
        "global_util": "enum_switch",
        "gpu_util": "calc_mix",
        "mem_bw_util": "string_fold",
        "cpu_util": "struct_route",
    }.get(dominant_metric, "dispatch_gate"))
    coeffs = _microprocess_coefficients(opcode_id)
    forecast_frame = dict(profile.get("forecast_frame", {}) or {})
    latency_calibration = dict(profile.get("latency_calibration", {}) or {})
    field_dynamics = dict(profile.get("field_dynamics", latency_calibration.get("field_dynamics", {})) or {})
    trajectory_9d = [float(_clamp01(value)) for value in list(latency_calibration.get("trajectory_9d", field_dynamics.get("trajectory_9d", [])) or [])[:9]]
    while len(trajectory_9d) < 9:
        trajectory_9d.append(0.0)
    predicted_trajectory_9d = [float(_clamp01(value)) for value in list(latency_calibration.get("predicted_trajectory_9d", field_dynamics.get("predicted_trajectory_9d", [])) or [])[:9]]
    while len(predicted_trajectory_9d) < 9:
        predicted_trajectory_9d.append(0.0)
    node_mix_word = 0
    for node in dominant_nodes:
        node_mix_word ^= int(node.get("node_word", 0)) & 0xFFFFFFFF
    frame_word = _stable_word({
        "timestamp": profile.get("timestamp", ""),
        "forecast": {k: round(float(v), 6) for k, v in forecast_frame.items()},
        "dominant": [str(node.get("metric", "")) for node in dominant_nodes],
    })
    noise_word = _stable_word({
        "noise_gate": round(float(profile.get("noise_gate", 0.0)), 6),
        "phase_turns": round(float(profile.get("phase_turns", 0.0)), 6),
        "predicted_latency_s": round(float(latency_calibration.get("predicted_latency_s", 0.0)), 6),
        "actuation_compensation": round(float(latency_calibration.get("actuation_compensation", 0.0)), 6),
        "observer_damping": round(float(latency_calibration.get("observer_damping", 0.0)), 6),
    })
    resonance_word = _stable_word({
        "resonance_gate": round(float(profile.get("resonance_gate", 0.0)), 6),
        "node_mix_word": int(node_mix_word),
        "kernel_latency_s": round(float(latency_calibration.get("kernel_latency_s", 0.0)), 6),
        "actuation_horizon_frames": round(float(latency_calibration.get("actuation_horizon_frames", 0.0)), 6),
        "phase_transport_term": round(float(latency_calibration.get("phase_transport_term", 0.0)), 6),
        "reverse_transport_gate": round(float(latency_calibration.get("reverse_transport_gate", 0.0)), 6),
    })
    transport_word = _stable_word({
        "flux_transport_term": round(float(latency_calibration.get("flux_transport_term", 0.0)), 6),
        "flux_transport_delta_turns": round(float(latency_calibration.get("flux_transport_delta_turns", 0.0)), 6),
        "transport_coherence": round(float(latency_calibration.get("transport_coherence", 0.0)), 6),
        "transported_phase_turns": round(float(latency_calibration.get("transported_phase_turns", 0.0)), 6),
    })
    field_word = _stable_word({
        "axis_scale_x": round(float(field_dynamics.get("axis_scale_x", 0.0)), 6),
        "axis_scale_y": round(float(field_dynamics.get("axis_scale_y", 0.0)), 6),
        "axis_scale_z": round(float(field_dynamics.get("axis_scale_z", 0.0)), 6),
        "temporal_coupling_moment": round(float(field_dynamics.get("temporal_coupling_moment", 0.0)), 6),
        "inertial_mass_proxy": round(float(field_dynamics.get("inertial_mass_proxy", 0.0)), 6),
        "spin_momentum_score": round(float(field_dynamics.get("spin_momentum_score", 0.0)), 6),
    })
    trajectory_word = _stable_word({
        "trajectory_9d": [round(value, 6) for value in trajectory_9d],
        "predicted_trajectory_9d": [round(value, 6) for value in predicted_trajectory_9d],
        "trajectory_spectral_id": str(latency_calibration.get("trajectory_spectral_id", field_dynamics.get("trajectory_spectral_id", ""))),
        "predicted_trajectory_spectral_id": str(latency_calibration.get("predicted_trajectory_spectral_id", field_dynamics.get("predicted_trajectory_spectral_id", ""))),
        "trajectory_conservation_alignment": round(float(latency_calibration.get("trajectory_conservation_alignment", field_dynamics.get("trajectory_conservation_alignment", 0.0))), 6),
        "trajectory_prediction_alignment": round(float(latency_calibration.get("trajectory_prediction_alignment", field_dynamics.get("trajectory_prediction_alignment", 0.0))), 6),
        "trajectory_expansion_term": round(float(latency_calibration.get("trajectory_expansion_term", field_dynamics.get("trajectory_expansion_term", 0.0))), 6),
    })
    timing_word = int(latency_calibration.get("timing_word", 0)) & 0xFFFFFFFF
    control_word = (frame_word ^ node_mix_word ^ noise_word ^ resonance_word ^ transport_word ^ field_word ^ trajectory_word ^ timing_word ^ coeffs[3]) & 0xFFFFFFFF
    sim_x = (
        (_telemetry_metric_value(forecast_frame, "gpu_util") - _telemetry_metric_value(forecast_frame, "global_util"))
        * (0.5 + 0.5 * float(field_dynamics.get("axis_scale_x", 0.0)))
        + float(field_dynamics.get("spin_axis_x", 0.0)) * 0.10
    )
    sim_y = (
        (_telemetry_metric_value(forecast_frame, "mem_bw_util") - _telemetry_metric_value(forecast_frame, "cpu_util"))
        * (0.5 + 0.5 * float(field_dynamics.get("axis_scale_y", 0.0)))
        + float(field_dynamics.get("spin_axis_y", 0.0)) * 0.10
    )
    sim_z = (
        float(profile.get("resonance_gate", 0.0))
        - float(profile.get("noise_gate", 0.0))
        + float(latency_calibration.get("actuation_compensation", 0.0))
        - float(latency_calibration.get("latency_load", 0.0))
        + float(latency_calibration.get("phase_transport_term", 0.0)) * 0.50
        + float(latency_calibration.get("flux_transport_term", 0.0)) * 0.25
        - float(latency_calibration.get("observer_damping", 0.0)) * 4.0
        + float(field_dynamics.get("temporal_coupling_moment", 0.0)) * 0.20
        - float(field_dynamics.get("inertial_mass_proxy", 0.0)) * 0.15
        + float(field_dynamics.get("relativistic_correlation", 0.0)) * 0.10
    )
    phase_turns = float(latency_calibration.get("actuation_phase_turns", profile.get("phase_turns", 0.0)))
    phase_delta = abs(_phase_delta_turns(profile.get("base_phase_turns", phase_turns), phase_turns))
    top_resonance = _clamp01(
        float(dominant_nodes[0].get("actuation_resonance", dominant_nodes[0].get("resonance", 0.0)))
    )
    axis_vector = [
        _clamp01(float(field_dynamics.get("axis_scale_x", _telemetry_metric_value(forecast_frame, "gpu_util")))),
        _clamp01(float(field_dynamics.get("axis_scale_y", _telemetry_metric_value(forecast_frame, "mem_bw_util")))),
        _clamp01(float(field_dynamics.get("axis_scale_z", _telemetry_metric_value(forecast_frame, "cpu_util")))),
        float(phase_turns),
    ]
    dof_vector = [
        _telemetry_metric_value(forecast_frame, "global_util"),
        _telemetry_metric_value(forecast_frame, "gpu_util"),
        _telemetry_metric_value(forecast_frame, "mem_bw_util"),
        _telemetry_metric_value(forecast_frame, "cpu_util"),
        _clamp01(abs(float(field_dynamics.get("spin_axis_x", 0.0)))),
        _clamp01(abs(float(field_dynamics.get("spin_axis_y", 0.0)))),
        _clamp01(abs(float(field_dynamics.get("spin_axis_z", 0.0)))),
        _clamp01(float(field_dynamics.get("vector_energy", 0.0))),
        _clamp01(float(field_dynamics.get("temporal_coupling_moment", 0.0))),
        _clamp01(float(field_dynamics.get("inertial_mass_proxy", 0.0))),
    ]
    field_alignment_score, field_alignment_weights = _derived_temporal_mix({
        "resonance_gate": float(profile.get("resonance_gate", 0.0)),
        "top_resonance": top_resonance,
        "noise_inverse": 1.0 - float(profile.get("noise_gate", 0.0)),
        "simulation_energy": _clamp01(abs(sim_x) + abs(sim_y)),
        "actuation_compensation": float(latency_calibration.get("actuation_compensation", 0.0)),
        "transport_coherence": float(latency_calibration.get("transport_coherence", 0.0)),
        "temporal_coupling": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
    })
    kernel_control_gate, kernel_control_weights = _derived_temporal_mix({
        "resonance_gate": float(profile.get("resonance_gate", 0.0)),
        "top_resonance": top_resonance,
        "simulation_depth": _clamp01(abs(sim_z)),
        "axis_gate": float(axis_vector[0]),
        "latency_inverse": 1.0 - float(latency_calibration.get("latency_load", 0.0)),
        "reverse_transport": float(latency_calibration.get("reverse_transport_gate", 0.0)),
        "inertial_mass": float(field_dynamics.get("inertial_mass_proxy", 0.0)),
    })
    return {
        "opcode_id": int(opcode_id),
        "coefficients": coeffs,
        "module_word": int(frame_word),
        "enum_word": int(resonance_word),
        "text_word": int(node_mix_word),
        "struct_word": int(noise_word),
        "calc_word": int(control_word ^ resonance_word),
        "control_word": int(control_word),
        "phase_anchor_turns": float(phase_turns),
        "field_alignment_score": float(field_alignment_score),
        "kernel_control_gate": float(kernel_control_gate),
        "sequence_persistence_score": _clamp01(
            0.30 * (1.0 - float(profile.get("noise_gate", 0.0)))
            + 0.24 * float(profile.get("resonance_gate", 0.0))
            + 0.24 * _clamp01(float(dominant_nodes[0].get("actuation_term", dominant_nodes[0].get("near_term", 0.0))))
            + 0.22 * float(latency_calibration.get("actuation_compensation", 0.0))
            + 0.08 * float(latency_calibration.get("transport_damping_gate", 0.0))
            + 0.08 * float(latency_calibration.get("trajectory_sequence_density", field_dynamics.get("trajectory_sequence_density", 0.0)))
        ),
        "temporal_index_overlap": _clamp01(
            0.34 * float(profile.get("resonance_gate", 0.0))
            + 0.22 * _clamp01(abs(float(dominant_nodes[0].get("slope", 0.0))) * 2.0)
            + 0.20 * float(axis_vector[3])
            + 0.14 * (1.0 - _clamp01(float(latency_calibration.get("actuation_horizon_frames", 0.0)) / 4.0))
            + 0.10 * _clamp01(float(phase_delta) * 2.0)
            + 0.08 * float(latency_calibration.get("reverse_transport_gate", 0.0))
            + 0.08 * float(latency_calibration.get("trajectory_prediction_alignment", field_dynamics.get("trajectory_prediction_alignment", 0.0)))
        ),
        "simulation_vector": [float(sim_x), float(sim_y), float(sim_z), float(phase_turns)],
        "feedback_axis_vector": axis_vector,
        "feedback_dof_vector": dof_vector,
        "resonant_nodes": dominant_nodes,
        "forecast_frame": forecast_frame,
        "noise_gate": float(profile.get("noise_gate", 0.0)),
        "resonance_gate": float(profile.get("resonance_gate", 0.0)),
        "axis_scale_x": float(field_dynamics.get("axis_scale_x", 0.0)),
        "axis_scale_y": float(field_dynamics.get("axis_scale_y", 0.0)),
        "axis_scale_z": float(field_dynamics.get("axis_scale_z", 0.0)),
        "vector_energy": float(field_dynamics.get("vector_energy", 0.0)),
        "temporal_coupling_moment": float(field_dynamics.get("temporal_coupling_moment", 0.0)),
        "inertial_mass_proxy": float(field_dynamics.get("inertial_mass_proxy", 0.0)),
        "spin_momentum_score": float(field_dynamics.get("spin_momentum_score", 0.0)),
        "trajectory_word": int(trajectory_word),
        "trajectory_9d": [float(value) for value in trajectory_9d],
        "predicted_trajectory_9d": [float(value) for value in predicted_trajectory_9d],
        "trajectory_spectral_id": str(latency_calibration.get("trajectory_spectral_id", field_dynamics.get("trajectory_spectral_id", ""))),
        "predicted_trajectory_spectral_id": str(latency_calibration.get("predicted_trajectory_spectral_id", field_dynamics.get("predicted_trajectory_spectral_id", ""))),
        "trajectory_conservation_alignment": float(latency_calibration.get("trajectory_conservation_alignment", field_dynamics.get("trajectory_conservation_alignment", 0.0))),
        "trajectory_prediction_alignment": float(latency_calibration.get("trajectory_prediction_alignment", field_dynamics.get("trajectory_prediction_alignment", 0.0))),
        "trajectory_expansion_term": float(latency_calibration.get("trajectory_expansion_term", field_dynamics.get("trajectory_expansion_term", 0.0))),
        "field_dynamics": field_dynamics,
        "latency_calibration": latency_calibration,
        "encoding_temporal_constant_weights": {
            "field_alignment": dict(field_alignment_weights),
            "kernel_control": dict(kernel_control_weights),
        },
        "timing_word": int(timing_word),
    }


def _telemetry_result_summary(
    accumulators: Dict[str, int],
    trace_state: Dict[str, Any],
    step_count: int,
    dominant_nodes: list[Dict[str, Any]],
    resonance_gate: float,
    noise_gate: float,
    latency_calibration: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    summary = _microprocess_result_summary(accumulators, trace_state, step_count)
    summary["dominant_nodes"] = [
        {
            "metric": str(node.get("metric", "")),
            "resonance": float(node.get("actuation_resonance", node.get("resonance", 0.0))),
            "near_term": float(node.get("actuation_term", node.get("near_term", 0.0))),
            "noise": float(node.get("noise", 0.0)),
        }
        for node in list(dominant_nodes or [])[:4]
    ]
    summary["resonance_gate"] = float(resonance_gate)
    summary["noise_gate"] = float(noise_gate)
    timing = dict(latency_calibration or {})
    summary["latency_calibration"] = {
        "headroom": float(timing.get("headroom", 0.0)),
        "latency_load": float(timing.get("latency_load", 0.0)),
        "sample_period_s": float(timing.get("sample_period_s", 0.0)),
        "kernel_latency_s": float(timing.get("kernel_latency_s", 0.0)),
        "pulse_generation_s": float(timing.get("pulse_generation_s", 0.0)),
        "kernel_request_s": float(timing.get("kernel_request_s", 0.0)),
        "kernel_actuation_s": float(timing.get("kernel_actuation_s", 0.0)),
        "predicted_latency_s": float(timing.get("predicted_latency_s", 0.0)),
        "actuation_horizon_frames": float(timing.get("actuation_horizon_frames", 0.0)),
        "actuation_compensation": float(timing.get("actuation_compensation", 0.0)),
        "predicted_phase_turns": float(timing.get("predicted_phase_turns", 0.0)),
        "actuation_phase_turns": float(timing.get("actuation_phase_turns", 0.0)),
        "transport_input_mode": str(timing.get("transport_input_mode", "telemetry_only")),
        "raw_telemetry_only": bool(timing.get("raw_telemetry_only", True)),
        "transport_coherence": float(timing.get("transport_coherence", 0.0)),
        "observer_damping": float(timing.get("observer_damping", 0.0)),
        "transport_damping_gate": float(timing.get("transport_damping_gate", 0.0)),
        "flux_transport_term": float(timing.get("flux_transport_term", 0.0)),
        "flux_transport_delta_turns": float(timing.get("flux_transport_delta_turns", 0.0)),
        "transport_candidate_phase_turns": float(timing.get("transport_candidate_phase_turns", 0.0)),
        "reverse_transport_gate": float(timing.get("reverse_transport_gate", 0.0)),
        "reverse_delta_theta_hat": float(timing.get("reverse_delta_theta_hat", 0.0)),
        "phase_transport_term": float(timing.get("phase_transport_term", 0.0)),
        "transported_phase_turns": float(timing.get("transported_phase_turns", 0.0)),
        "gpu_pulse_phase_effect": float(timing.get("gpu_pulse_phase_effect", 0.0)),
        "phase_injection_delta_turns": float(timing.get("phase_injection_delta_turns", 0.0)),
        "phase_injection_turns": float(timing.get("phase_injection_turns", 0.0)),
        "phase_ring_closure": float(timing.get("phase_ring_closure", 0.0)),
        "phase_ring_density": float(timing.get("phase_ring_density", 0.0)),
        "phase_ring_strength": float(timing.get("phase_ring_strength", 0.0)),
        "zero_point_crossover_gate": float(timing.get("zero_point_crossover_gate", 0.0)),
        "shared_vector_collapse_gate": float(timing.get("shared_vector_collapse_gate", 0.0)),
        "shared_vector_phase_lock": float(timing.get("shared_vector_phase_lock", 0.0)),
        "inertial_basin_strength": float(timing.get("inertial_basin_strength", 0.0)),
        "wavelength_norm": float(timing.get("wavelength_norm", 0.0)),
        "orientation_alignment": float(timing.get("orientation_alignment", 0.0)),
        "rotational_velocity_norm": float(timing.get("rotational_velocity_norm", 0.0)),
        "relative_temporal_position": float(timing.get("relative_temporal_position", 0.0)),
        "zero_point_line_distance": float(timing.get("zero_point_line_distance", 0.0)),
        "field_interference_norm": float(timing.get("field_interference_norm", 0.0)),
        "resonant_interception_inertia": float(timing.get("resonant_interception_inertia", 0.0)),
        "temporal_relativity_norm": float(timing.get("temporal_relativity_norm", 0.0)),
        "temporal_relativity_vector": [float(_clamp01(value)) for value in list(timing.get("temporal_relativity_vector", []))[:4]],
        "phase_ring_zone_id": int(timing.get("phase_ring_zone_id", 0) or 0),
        "frequency_gradient_9d": [float(_clamp01(value)) for value in list(timing.get("frequency_gradient_9d", []))[:9]],
        "field_gradient_9d": [float(_clamp01(value)) for value in list(timing.get("field_gradient_9d", timing.get("frequency_gradient_9d", [])))[:9]],
        "gradient_spectral_id": str(timing.get("gradient_spectral_id", "")),
        "axis_scale_x": float(timing.get("axis_scale_x", 0.0)),
        "axis_scale_y": float(timing.get("axis_scale_y", 0.0)),
        "axis_scale_z": float(timing.get("axis_scale_z", 0.0)),
        "vector_energy": float(timing.get("vector_energy", 0.0)),
        "temporal_coupling_moment": float(timing.get("temporal_coupling_moment", 0.0)),
        "inertial_mass_proxy": float(timing.get("inertial_mass_proxy", 0.0)),
        "relativistic_correlation": float(timing.get("relativistic_correlation", 0.0)),
        "spin_axis_x": float(timing.get("spin_axis_x", 0.0)),
        "spin_axis_y": float(timing.get("spin_axis_y", 0.0)),
        "spin_axis_z": float(timing.get("spin_axis_z", 0.0)),
        "spin_momentum_score": float(timing.get("spin_momentum_score", 0.0)),
        "trajectory_spectral_id": str(timing.get("trajectory_spectral_id", "")),
        "predicted_trajectory_spectral_id": str(timing.get("predicted_trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": float(timing.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_prediction_alignment": float(timing.get("trajectory_prediction_alignment", 0.0)),
        "trajectory_expansion_term": float(timing.get("trajectory_expansion_term", 0.0)),
        "trajectory_sequence_density": float(timing.get("trajectory_sequence_density", 0.0)),
        "trajectory_noise_feedback_norm": float(timing.get("trajectory_noise_feedback_norm", 0.0)),
        "trajectory_9d": [float(_clamp01(value)) for value in list(timing.get("trajectory_9d", []))[:9]],
        "predicted_trajectory_9d": [float(_clamp01(value)) for value in list(timing.get("predicted_trajectory_9d", []))[:9]],
        "throughput_hz": float(timing.get("throughput_hz", 0.0)),
        "ada_constants": {
            "h_eff": float(dict(timing.get("ada_constants", {}) or {}).get("h_eff", 0.0)),
            "k_B_eff": float(dict(timing.get("ada_constants", {}) or {}).get("k_B_eff", 0.0)),
            "c_eff": float(dict(timing.get("ada_constants", {}) or {}).get("c_eff", 0.0)),
        },
    }
    return summary


def _normalize_hash_hex(value: Any, default: str = "") -> str:
    try:
        text = str(value or "").strip().lower()
    except Exception:
        text = ""
    if text.startswith("0x"):
        text = text[2:]
    text = "".join(ch for ch in text if ch in "0123456789abcdef")
    if not text:
        text = str(default or "")
    text = "".join(ch for ch in text if ch in "0123456789abcdef")
    return text[-64:].rjust(64, "0") if text else ("f" * 64)


def _leading_zero_nibbles_hex(text: str) -> int:
    count = 0
    for ch in _normalize_hash_hex(text, default=""):
        if ch != "0":
            break
        count += 1
    return int(count)


def _pow_distance_metrics(pow_hex: str, target_hex: str) -> Dict[str, Any]:
    normalized_pow = _normalize_hash_hex(pow_hex)
    normalized_target = _normalize_hash_hex(target_hex)
    prefix_match = 0
    for lhs, rhs in zip(normalized_pow, normalized_target):
        if lhs != rhs:
            break
        prefix_match += 1
    pow_zero_nibbles = _leading_zero_nibbles_hex(normalized_pow)
    target_zero_nibbles = _leading_zero_nibbles_hex(normalized_target)
    try:
        pow_value = int(normalized_pow, 16)
        target_value = int(normalized_target, 16)
    except Exception:
        pow_value = 0
        target_value = 1
    valid = bool(target_value > 0 and pow_value <= target_value)
    if valid:
        over_target_ratio = 0.0
        distance_score = 1.0
    else:
        over_target_ratio = float(max(pow_value - target_value, 0)) / max(float(target_value), 1.0)
        distance_score = 1.0 / (1.0 + math.log10(1.0 + over_target_ratio))
    zero_alignment = _clamp01(
        float(pow_zero_nibbles + prefix_match * 0.25) / max(float(target_zero_nibbles + 1), 1.0)
    )
    nibble_span = max(target_zero_nibbles + 8, 1)
    normalized_prefix_match = _clamp01(float(prefix_match) / float(nibble_span))
    return {
        "valid": bool(valid),
        "distance_score": float(distance_score),
        "zero_alignment": float(zero_alignment),
        "prefix_match": int(prefix_match),
        "normalized_prefix_match": float(normalized_prefix_match),
        "pow_zero_nibbles": int(pow_zero_nibbles),
        "target_zero_nibbles": int(target_zero_nibbles),
        "over_target_ratio": float(over_target_ratio),
    }


def _pow_hex_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("hash_hex", "powhash", "hash"):
            normalized = _normalize_hash_hex(value.get(key, ""), default="")
            if normalized and normalized != ("f" * 64):
                return normalized
        return ""
    normalized = _normalize_hash_hex(value, default="")
    if normalized == ("f" * 64):
        return ""
    return normalized


def _candidate_packet_network(raw_payload: Dict[str, Any], network_name: str = "") -> str:
    network_u = str(network_name or raw_payload.get("network", "") or "").upper()
    if network_u in ("BTC", "LTC", "ETC", "RVN"):
        return network_u
    if str(raw_payload.get("seed_hash", "") or raw_payload.get("seed", "")):
        return "ETC"
    if str(raw_payload.get("epoch", "") or raw_payload.get("height", "")) and str(raw_payload.get("header_hash", raw_payload.get("header", ""))):
        return "RVN"
    if str(raw_payload.get("header_hex", raw_payload.get("header", ""))):
        return "BTC"
    return ""


def _iter_hash_probe_packet_candidates(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    lane_id = str(payload.get("hash_probe_lane_id", "") or "")
    direct_sources = [
        ("payload.hash_probe_packet", payload.get("hash_probe_packet")),
        ("payload.hash_probe_job", payload.get("hash_probe_job")),
    ]
    jobs_map = dict(payload.get("jobs_map", {}) or {})
    if lane_id and lane_id in jobs_map:
        direct_sources.append(("payload.jobs_map[%s]" % lane_id, jobs_map.get(lane_id)))
    for key, value in jobs_map.items():
        direct_sources.append(("payload.jobs_map[%s]" % str(key), value))

    for source_name, value in direct_sources:
        if value is not None:
            candidates.append({"source": str(source_name), "value": value})

    try:
        from VHW.vsd_manager import VSD

        telemetry_global = dict(VSD.get("telemetry/global", {}) or {})
        vsd_jobs = dict(telemetry_global.get("jobs_map", {}) or {})
        if lane_id and lane_id in vsd_jobs:
            candidates.append({"source": "VSD.telemetry/global.jobs_map[%s]" % lane_id, "value": vsd_jobs.get(lane_id)})
        for key, value in vsd_jobs.items():
            candidates.append({"source": "VSD.telemetry/global.jobs_map[%s]" % str(key), "value": value})
    except Exception:
        pass

    try:
        import bios.main_runtime as main_runtime

        telemetry_global = dict(main_runtime._vsd.get("telemetry/global", {}) or {})
        runtime_jobs = dict(telemetry_global.get("jobs_map", {}) or {})
        if lane_id and lane_id in runtime_jobs:
            candidates.append({"source": "bios.main_runtime._vsd.jobs_map[%s]" % lane_id, "value": runtime_jobs.get(lane_id)})
        for key, value in runtime_jobs.items():
            candidates.append({"source": "bios.main_runtime._vsd.jobs_map[%s]" % str(key), "value": value})
    except Exception:
        pass
    return candidates


def _resolve_hash_probe_packet(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from neural_object import neural_objectPacket, neural_objectSchema, network_to_packet_type
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    packet_network = str(payload.get("hash_probe_network", "") or "")
    seen: set[tuple[str, str]] = set()
    for candidate in _iter_hash_probe_packet_candidates(payload):
        source = str(candidate.get("source", "candidate"))
        value = candidate.get("value")
        raw_payload: Dict[str, Any] = {}
        packet_type = None
        if isinstance(value, neural_objectPacket):
            packet_type = value.packet_type
            raw_payload = dict(value.raw_payload or {})
            if not packet_network:
                packet_network = str(getattr(value.network, "name", "") or "")
        elif isinstance(value, dict):
            if isinstance(value.get("raw_payload"), dict):
                raw_payload = dict(value.get("raw_payload", {}) or {})
            else:
                raw_payload = dict(value or {})
            network_u = _candidate_packet_network(raw_payload, packet_network)
            if network_u:
                try:
                    packet_type = network_to_packet_type(network_u)
                except Exception:
                    packet_type = None
        if packet_type is None:
            continue
        schema = dict(neural_objectSchema.get(packet_type, {}) or {})
        convert_incoming = schema.get("convert_incoming")
        hash_function = schema.get("hash_function")
        verify_target = schema.get("verify_target")
        if not callable(convert_incoming) or not callable(hash_function) or not callable(verify_target):
            continue
        packet_norm = dict(convert_incoming(raw_payload) or {})
        norm_raw = dict(packet_norm.get("raw_payload", {}) or {})
        if not norm_raw:
            continue
        dedupe_key = (str(packet_type), json.dumps(norm_raw, sort_keys=True, ensure_ascii=True, default=str))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        target_hex = _normalize_hash_hex(
            norm_raw.get("share_target", norm_raw.get("active_target", norm_raw.get("target", ""))),
            default="",
        )
        if not target_hex or target_hex == ("f" * 64):
            continue
        return {
            "ok": True,
            "source": source,
            "packet_type": str(packet_type),
            "packet_norm": packet_norm,
            "schema": schema,
            "target_hex": target_hex,
        }
    return {"ok": False, "error": "no hash probe packet available"}


def _u32_word(value: Any, scale: float = 1.0, signed: bool = False) -> int:
    try:
        number = float(value) * float(scale)
    except Exception:
        number = 0.0
    integer = int(round(number))
    if signed:
        integer &= 0xFFFFFFFF
    return int(integer) & 0xFFFFFFFF


def _rotl32(value: Any, bits: int) -> int:
    word = int(value) & 0xFFFFFFFF
    shift = int(bits) % 32
    if shift <= 0:
        return int(word)
    return int(((word << shift) | (word >> (32 - shift))) & 0xFFFFFFFF)


def _hash_probe_target_hex(payload: Dict[str, Any]) -> Dict[str, Any]:
    custom_target = _normalize_hash_hex(payload.get("hash_probe_target_hex", ""), default="")
    if custom_target and custom_target != ("f" * 64):
        return {
            "target_hex": custom_target,
            "target_source": "custom_target_hex",
        }
    zero_nibbles = max(1, min(63, int(payload.get("hash_probe_zero_nibbles", 8) or 8)))
    target_hex = ("0" * zero_nibbles) + ("f" * (64 - zero_nibbles))
    return {
        "target_hex": str(target_hex[:64]),
        "target_source": "zero_nibble_probe",
    }


def _temporal_probe_nonce_candidates(
    result: Dict[str, Any],
    payload: Dict[str, Any],
    profiles: list[Dict[str, Any]] | None = None,
    encoded_profiles: list[Dict[str, Any]] | None = None,
) -> list[int]:
    result_map = dict(result or {})
    profiles_list = list(profiles or [])
    encoded_list = list(encoded_profiles or [])
    packet_network = _resolve_hash_probe_packet(payload)
    packet_type = str(packet_network.get("packet_type", ""))
    default_limit = 48
    if "ETC" in packet_type or "RVN" in packet_type:
        default_limit = 16
    limit = max(8, min(256, int(payload.get("hash_probe_candidate_limit", default_limit) or default_limit)))
    harmonic_scale = max(1, min(7, int(payload.get("hash_probe_harmonic_scale", 3) or 3)))
    temporal_offset_scale = max(128, min(8192, int(payload.get("hash_probe_temporal_offset_scale", 1536) or 1536)))
    variants_per_profile = max(2, min(8, int(payload.get("hash_probe_variants_per_profile", 4) or 4)))

    candidates: list[int] = []
    seen: set[int] = set()

    def _add_candidate(value: Any) -> None:
        try:
            nonce = int(value) & 0xFFFFFFFF
        except Exception:
            return
        if nonce in seen:
            return
        seen.add(nonce)
        candidates.append(nonce)

    for value in (
        result_map.get("result_word", 0),
        result_map.get("dispatch_word", 0),
        result_map.get("string_word", 0),
        result_map.get("structure_word", 0),
        int(result_map.get("result_word", 0)) ^ int(result_map.get("dispatch_word", 0)),
        int(result_map.get("string_word", 0)) ^ int(result_map.get("structure_word", 0)),
    ):
        _add_candidate(value)
        if len(candidates) >= limit:
            return candidates[:limit]

    previous_phase_word = int(result_map.get("dispatch_word", 0)) & 0xFFFFFFFF
    previous_temporal_word = int(result_map.get("string_word", 0)) & 0xFFFFFFFF
    for idx, encoded_profile in enumerate(encoded_list):
        if len(candidates) >= limit:
            break
        profile = dict(profiles_list[idx] if idx < len(profiles_list) else {})
        encoded = dict(encoded_profile or {})
        latency = dict(encoded.get("latency_calibration", {}) or {})
        resonant_nodes = list(encoded.get("resonant_nodes", profile.get("dominant_nodes", [])) or [])

        node_mix_word = int(encoded.get("text_word", 0)) & 0xFFFFFFFF
        node_pressure = 0.0
        for node_idx, node in enumerate(resonant_nodes[:4]):
            node_metric = {
                "metric": str(node.get("metric", "")),
                "resonance": round(float(node.get("actuation_resonance", node.get("resonance", 0.0))), 6),
                "near_term": round(float(node.get("actuation_term", node.get("near_term", 0.0))), 6),
                "noise": round(float(node.get("noise", 0.0)), 6),
            }
            node_word = int(node.get("node_word", _stable_word(node_metric))) & 0xFFFFFFFF
            node_mix_word ^= _rotl32(node_word, ((idx + 1) * (node_idx + 1)) % 13 + 1)
            node_pressure += 0.25 * float(node_metric["resonance"])

        phase_word = _u32_word(
            latency.get("transported_phase_turns", latency.get("actuation_phase_turns", encoded.get("phase_anchor_turns", 0.0))),
            scale=float(0xFFFFFFFF),
        )
        trajectory_word = int(encoded.get("trajectory_word", 0)) & 0xFFFFFFFF
        flux_word = _u32_word(latency.get("flux_transport_term", 0.0), scale=float(0xFFFFFFFF))
        persistence_word = _u32_word(
            encoded.get("sequence_persistence_score", profile.get("sequence_persistence_score", 0.0)),
            scale=float(0xFFFFFFFF),
        )
        overlap_word = _u32_word(
            encoded.get("temporal_index_overlap", profile.get("temporal_index_overlap", 0.0)),
            scale=float(0xFFFFFFFF),
        )
        resonance_word = _u32_word(encoded.get("resonance_gate", profile.get("resonance_gate", 0.0)), scale=float(0xFFFFFFFF))
        decoherence_word = _u32_word(profile.get("noise_gate", encoded.get("noise_gate", 0.0)), scale=float(0xFFFFFFFF))
        damping_word = _u32_word(latency.get("transport_damping_gate", 0.0), scale=float(0xFFFFFFFF))
        reverse_word = _u32_word(latency.get("reverse_delta_theta_hat", 0.0), scale=1.0e9, signed=True)
        temporal_word = _u32_word(latency.get("phase_transport_term", 0.0), scale=1.0e9, signed=True)
        flux_delta_word = _u32_word(latency.get("flux_transport_delta_turns", 0.0), scale=1.0e9, signed=True)

        harmonic_order = 1 + ((idx * harmonic_scale) % 7)
        base_word = int(encoded.get("control_word", 0)) & 0xFFFFFFFF
        recurrence_word = (
            base_word
            + _rotl32(phase_word, harmonic_order)
            + _rotl32(flux_word, harmonic_order + 3)
            + _rotl32(persistence_word ^ overlap_word, harmonic_order + 5)
            + _rotl32(trajectory_word, harmonic_order + 7)
            + (idx * 0x9E3779B9)
        ) & 0xFFFFFFFF
        temporal_offset = _u32_word(
            float(node_pressure)
            + float(latency.get("phase_transport_term", 0.0))
            + float(latency.get("flux_transport_delta_turns", 0.0)),
            scale=float(temporal_offset_scale),
            signed=True,
        )
        phase_orbit_word = (
            _rotl32(phase_word ^ previous_phase_word, harmonic_order + 7)
            ^ _rotl32(temporal_word ^ previous_temporal_word, harmonic_order + 11)
            ^ _rotl32(node_mix_word, harmonic_order + 13)
        ) & 0xFFFFFFFF
        candidate_words = [
            recurrence_word,
            (recurrence_word ^ node_mix_word ^ _rotl32(resonance_word, harmonic_order + 7)) & 0xFFFFFFFF,
            (recurrence_word + temporal_offset + _rotl32(damping_word ^ decoherence_word, harmonic_order + 11)) & 0xFFFFFFFF,
            (_rotl32(recurrence_word ^ reverse_word ^ previous_phase_word, harmonic_order + 9) + phase_orbit_word) & 0xFFFFFFFF,
            (_rotl32(base_word ^ flux_delta_word ^ overlap_word, harmonic_order + 5) + temporal_offset) & 0xFFFFFFFF,
            (phase_orbit_word + _rotl32(flux_word ^ persistence_word, harmonic_order + 3) + _rotl32(reverse_word, harmonic_order + 1)) & 0xFFFFFFFF,
            (_rotl32(trajectory_word ^ phase_word ^ node_mix_word, harmonic_order + 11) + temporal_offset) & 0xFFFFFFFF,
        ]
        for candidate_word in candidate_words[:variants_per_profile + 2]:
            _add_candidate(candidate_word)
            if len(candidates) >= limit:
                break
        previous_phase_word = phase_word
        previous_temporal_word = temporal_word

    return candidates[:limit]


def _telemetry_hash_probe(
    result: Dict[str, Any],
    payload: Dict[str, Any],
    profiles: list[Dict[str, Any]] | None = None,
    encoded_profiles: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    result_map = dict(result or {})
    latency = dict(result_map.get("latency_calibration", {}) or {})
    candidate_nonces = _temporal_probe_nonce_candidates(
        result=result_map,
        payload=payload,
        profiles=profiles,
        encoded_profiles=encoded_profiles,
    )

    real_packet = _resolve_hash_probe_packet(payload)
    if bool(real_packet.get("ok", False)):
        packet_norm = dict(real_packet.get("packet_norm", {}) or {})
        schema = dict(real_packet.get("schema", {}) or {})
        target_hex = str(real_packet.get("target_hex", "f" * 64))
        preview: list[Dict[str, Any]] = []
        best_entry: Dict[str, Any] | None = None
        for nonce in candidate_nonces:
            try:
                pow_raw = schema["hash_function"](packet_norm, int(nonce))
            except Exception:
                pow_raw = ""
            pow_hex = _pow_hex_text(pow_raw)
            if not pow_hex:
                continue
            metrics = _pow_distance_metrics(pow_hex, target_hex)
            entry = {
                "nonce": int(nonce),
                "pow_hex": str(pow_hex),
                **metrics,
            }
            if len(preview) < 6:
                preview.append({
                    "nonce": int(nonce),
                    "hash_prefix": str(pow_hex[:16]),
                    "distance_score": float(metrics.get("distance_score", 0.0)),
                    "pow_zero_nibbles": int(metrics.get("pow_zero_nibbles", 0)),
                    "valid": bool(metrics.get("valid", False)),
                })
            if best_entry is None or (
                bool(entry.get("valid", False)),
                int(entry.get("pow_zero_nibbles", 0)),
                float(entry.get("distance_score", 0.0)),
            ) > (
                bool(best_entry.get("valid", False)),
                int(best_entry.get("pow_zero_nibbles", 0)),
                float(best_entry.get("distance_score", 0.0)),
            ):
                best_entry = entry
        if best_entry is not None:
            return {
                "probe_source": "real_job_packet",
                "packet_source": str(real_packet.get("source", "")),
                "packet_type": str(real_packet.get("packet_type", "")),
                "target_hex": str(target_hex),
                "target_source": "packet_share_target",
                "candidate_count": int(len(candidate_nonces)),
                "header_word_count": 20,
                "sample_preview": preview,
                "probe_hash_hex": str(best_entry.get("pow_hex", "")),
                "probe_hash_prefix": str(str(best_entry.get("pow_hex", ""))[:16]),
                "selected_nonce": int(best_entry.get("nonce", 0)),
                "selected_nonce_hex": "%08x" % (int(best_entry.get("nonce", 0)) & 0xFFFFFFFF),
                "valid": bool(best_entry.get("valid", False)),
                "distance_score": float(best_entry.get("distance_score", 0.0)),
                "zero_alignment": float(best_entry.get("zero_alignment", 0.0)),
                "prefix_match": int(best_entry.get("prefix_match", 0)),
                "normalized_prefix_match": float(best_entry.get("normalized_prefix_match", 0.0)),
                "pow_zero_nibbles": int(best_entry.get("pow_zero_nibbles", 0)),
                "target_zero_nibbles": int(best_entry.get("target_zero_nibbles", 0)),
                "over_target_ratio": float(best_entry.get("over_target_ratio", 0.0)),
            }

    target_info = _hash_probe_target_hex(payload)
    header_words = [
        int(result_map.get("dispatch_word", 0)) & 0xFFFFFFFF,
        int(result_map.get("string_word", 0)) & 0xFFFFFFFF,
        int(result_map.get("structure_word", 0)) & 0xFFFFFFFF,
        int(result_map.get("result_word", 0)) & 0xFFFFFFFF,
        _u32_word(result_map.get("resonance_gate", 0.0), scale=1.0e9),
        _u32_word(result_map.get("noise_gate", 0.0), scale=1.0e9),
        _u32_word(latency.get("predicted_latency_s", 0.0), scale=1.0e9),
        _u32_word(latency.get("actuation_compensation", 0.0), scale=1.0e9),
        _u32_word(latency.get("predicted_phase_turns", 0.0), scale=1.0e9),
        _u32_word(latency.get("actuation_phase_turns", 0.0), scale=1.0e9),
        _u32_word(latency.get("phase_transport_term", 0.0), scale=1.0e9, signed=True),
        _u32_word(latency.get("flux_transport_term", 0.0), scale=1.0e9),
        _u32_word(latency.get("transport_coherence", 0.0), scale=1.0e9),
        _u32_word(latency.get("observer_damping", 0.0), scale=1.0e9),
        _u32_word(latency.get("transport_damping_gate", 0.0), scale=1.0e9),
        _u32_word(latency.get("reverse_transport_gate", 0.0), scale=1.0e9),
        _u32_word(latency.get("reverse_delta_theta_hat", 0.0), scale=1.0e9, signed=True),
        _u32_word(latency.get("transported_phase_turns", 0.0), scale=1.0e9),
        int(_stable_word(result_map.get("route_case", ""))) & 0xFFFFFFFF,
        int(result_map.get("step_count", 0)) & 0xFFFFFFFF,
    ]
    header_bytes = b"".join(int(word).to_bytes(4, byteorder="little", signed=False) for word in header_words)
    hash_bytes = hashlib.sha256(hashlib.sha256(header_bytes).digest()).digest()
    pow_hex = hash_bytes.hex()
    metrics = _pow_distance_metrics(pow_hex, str(target_info.get("target_hex", "f" * 64)))
    return {
        "probe_source": "synthetic_transport_header",
        "probe_hash_hex": str(pow_hex),
        "probe_hash_prefix": str(pow_hex[:16]),
        "target_hex": str(target_info.get("target_hex", "f" * 64)),
        "target_source": str(target_info.get("target_source", "zero_nibble_probe")),
        "header_word_count": int(len(header_words)),
        **metrics,
    }


def _raw_transport_replay_frames(frames: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    raw_frames: list[Dict[str, Any]] = []
    for frame in list(frames or []):
        current = dict(frame or {})
        raw_gpu_util = _clamp01(current.get("raw_gpu_util", current.get("gpu_util", 0.0)))
        raw_mem_bw_util = _clamp01(current.get("raw_mem_bw_util", current.get("mem_bw_util", 0.0)))
        raw_cpu_util = _clamp01(current.get("raw_cpu_util", current.get("cpu_util", 0.0)))
        raw_frames.append({
            "timestamp": str(current.get("timestamp", "")),
            "global_util": _clamp01(max(raw_gpu_util, raw_mem_bw_util, raw_cpu_util)),
            "gpu_util": float(raw_gpu_util),
            "mem_bw_util": float(raw_mem_bw_util),
            "cpu_util": float(raw_cpu_util),
            "raw_gpu_util": float(raw_gpu_util),
            "raw_mem_bw_util": float(raw_mem_bw_util),
            "raw_cpu_util": float(raw_cpu_util),
            "pulse": float(current.get("pulse", 0.0)),
            "anti_pulse": float(current.get("anti_pulse", 0.0)),
            "phase_turns": float(_clamp01(current.get("phase_turns", 0.0))),
            "sample_period_s": max(0.0, _safe_float(current.get("sample_period_s", 0.0), 0.0)),
            "source": "live_startup_raw_replay",
            "actuation_applied": False,
            "actuation_elapsed_s": 0.0,
            "actuation_mode": "raw_telemetry_replay",
            "actuation_tag": str(current.get("actuation_tag", "")),
            "actuation_error": "",
            "actuation_load_hint": 0.0,
            "actuation_dispatch_ms": 0.0,
        })
    return raw_frames


def _transport_mode_benchmark_summary(benchmark: Dict[str, Any]) -> Dict[str, Any]:
    substrate_result = dict(benchmark.get("substrate_result", {}) or {})
    result = dict(substrate_result.get("result", {}) or {})
    latency = dict(result.get("latency_calibration", {}) or {})
    hash_probe = dict(benchmark.get("hash_probe", {}) or {})
    return {
        "telemetry_source": str(benchmark.get("telemetry_source", "")),
        "results_match": bool(benchmark.get("results_match", False)),
        "speedup_total": float(benchmark.get("speedup_total", 0.0)),
        "speedup_exec_only": float(benchmark.get("speedup_exec_only", 0.0)),
        "predicted_latency_s": float(latency.get("predicted_latency_s", 0.0)),
        "phase_transport_term": float(latency.get("phase_transport_term", 0.0)),
        "flux_transport_term": float(latency.get("flux_transport_term", 0.0)),
        "reverse_transport_gate": float(latency.get("reverse_transport_gate", 0.0)),
        "transport_input_mode": str(latency.get("transport_input_mode", "telemetry_only")),
        "trajectory_spectral_id": str(latency.get("trajectory_spectral_id", "")),
        "trajectory_conservation_alignment": float(latency.get("trajectory_conservation_alignment", 0.0)),
        "trajectory_prediction_alignment": float(latency.get("trajectory_prediction_alignment", 0.0)),
        "trajectory_expansion_term": float(latency.get("trajectory_expansion_term", 0.0)),
        "temporal_relativity_norm": float(latency.get("temporal_relativity_norm", 0.0)),
        "field_interference_norm": float(latency.get("field_interference_norm", 0.0)),
        "resonant_interception_inertia": float(latency.get("resonant_interception_inertia", 0.0)),
        "hash_distance_score": float(hash_probe.get("distance_score", 0.0)),
        "hash_zero_nibbles": int(hash_probe.get("pow_zero_nibbles", 0)),
        "hash_valid": bool(hash_probe.get("valid", False)),
        "hash_prefix": str(hash_probe.get("probe_hash_prefix", "")),
        "hash_probe_source": str(hash_probe.get("probe_source", "")),
        "hash_candidate_count": int(hash_probe.get("candidate_count", 0)),
    }


def optimize_telemetry_transport_hash(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base_payload = dict(payload or {})
    frames = _prepare_telemetry_frames(base_payload)
    frozen_payload = dict(base_payload)
    frozen_payload["frames"] = list(frames)
    frozen_payload["telemetry_mode"] = "provided"
    frozen_payload["telemetry_source"] = "provided"
    frozen_payload["use_live_telemetry"] = False
    frozen_payload["compare_raw_transport_modes"] = False

    variants = [
        ("baseline", {}),
        ("low_damping_temporal", {
            "observer_damping_min": 0.006,
            "observer_damping_max": 0.014,
            "hash_probe_temporal_offset_scale": 2048,
            "hash_probe_harmonic_scale": 4,
        }),
        ("actuation_trim", {
            "transport_actuation_weight": 0.04,
            "flux_actuation_weight": 0.04,
            "flux_near_term_weight": 0.16,
            "hash_probe_temporal_offset_scale": 2048,
        }),
        ("raw_bias_bridge", {
            "transport_actuation_weight": 0.0,
            "flux_actuation_weight": 0.0,
            "observer_damping_min": 0.008,
            "observer_damping_max": 0.016,
            "flux_near_term_weight": 0.18,
        }),
        ("reverse_relaxed", {
            "reverse_transport_min_coherence": 0.52,
            "reverse_transport_max_noise": 0.22,
            "reverse_transport_min_resonance": 0.38,
            "hash_probe_candidate_limit": 64,
            "hash_probe_harmonic_scale": 5,
        }),
        ("headroom_resonance_bias", {
            "transport_resonance_weight": 0.42,
            "transport_headroom_weight": 0.20,
            "transport_actuation_weight": 0.04,
            "flux_global_weight": 0.10,
            "flux_near_term_weight": 0.18,
            "hash_probe_temporal_offset_scale": 3072,
        }),
    ]

    ranked: list[Dict[str, Any]] = []
    for variant_name, overrides in variants:
        variant_payload = dict(frozen_payload)
        variant_payload.update(dict(overrides))
        variant_payload["program_id"] = str(base_payload.get("program_id", "telemetry_resonance")) + "_" + str(variant_name)
        bench = benchmark_telemetry_resonance(variant_payload, previous_trace_state=previous_trace_state)
        summary = _transport_mode_benchmark_summary(bench)
        summary["variant"] = str(variant_name)
        summary["overrides"] = dict(overrides)
        ranked.append(summary)

    ranked.sort(
        key=lambda item: (
            bool(item.get("hash_valid", False)),
            int(item.get("hash_zero_nibbles", 0)),
            float(item.get("hash_distance_score", 0.0)),
            float(item.get("speedup_total", 0.0)),
        ),
        reverse=True,
    )
    best = dict(ranked[0] if ranked else {})
    baseline = next((item for item in ranked if str(item.get("variant", "")) == "baseline"), {})
    return {
        "ok": True,
        "program_id": str(base_payload.get("program_id", "telemetry_resonance")),
        "frame_count": int(len(frames)),
        "telemetry_source": str(dict(frames[0] if frames else {}).get("source", "synthetic")),
        "best": best,
        "baseline": dict(baseline or {}),
        "improvement": {
            "hash_distance_score": float(best.get("hash_distance_score", 0.0) - float(dict(baseline or {}).get("hash_distance_score", 0.0))),
            "hash_zero_nibbles": int(best.get("hash_zero_nibbles", 0) - int(dict(baseline or {}).get("hash_zero_nibbles", 0))),
            "speedup_total": float(best.get("speedup_total", 0.0) - float(dict(baseline or {}).get("speedup_total", 0.0))),
        },
        "ranked": ranked,
    }


def _run_encoded_telemetry_resonance(
    encoded_profiles: list[Dict[str, Any]],
    pulse_cycles: int,
    program_label: str,
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    trace_state = dict(previous_trace_state or {})
    accumulators = {
        "dispatch_word": 0,
        "string_word": 0,
        "structure_word": 0,
    }
    step_count = 0
    dominant_nodes: list[Dict[str, Any]] = []
    resonance_gate = 0.0
    noise_gate = 0.0
    latency_calibration: Dict[str, Any] = {}
    for cycle_index in range(max(1, int(pulse_cycles))):
        for profile_index, encoded_profile in enumerate(encoded_profiles):
            pulse_index = cycle_index * max(1, len(encoded_profiles)) + profile_index
            inputs = _microprocess_trace_inputs(
                encoded_artifact=encoded_profile,
                pulse_index=pulse_index,
                program_label=program_label,
                previous_trace_state=trace_state,
            )
            trace_state = _update_substrate_trace_state(
                pulse_index=int(inputs["pulse_index"]),
                previous_trace_state=dict(inputs["previous_trace_state"]),
                simulation_field_state=dict(inputs["simulation_field_state"]),
                gpu_feedback=dict(inputs["gpu_feedback"]),
                gpu_pulse_delta_feedback=dict(inputs["gpu_pulse_delta_feedback"]),
                interference_field=dict(inputs["interference_field"]),
                effective_vector=dict(inputs["effective_vector"]),
                kernel_execution_event=dict(inputs["kernel_execution_event"]),
                trace_label=str(inputs["trace_label"]),
            )
            accumulators = _microprocess_decode_step(
                accumulators=accumulators,
                encoded_artifact=encoded_profile,
                trace_state=trace_state,
                step_index=profile_index,
                pulse_index=pulse_index,
            )
            dominant_nodes = list(encoded_profile.get("resonant_nodes", dominant_nodes))
            resonance_gate = float(encoded_profile.get("resonance_gate", resonance_gate))
            noise_gate = float(encoded_profile.get("noise_gate", noise_gate))
            latency_calibration = dict(encoded_profile.get("latency_calibration", latency_calibration))
            step_count += 1
    return {
        "ok": True,
        "trace_state": dict(trace_state),
        "result": _telemetry_result_summary(
            accumulators=accumulators,
            trace_state=trace_state,
            step_count=step_count,
            dominant_nodes=dominant_nodes,
            resonance_gate=resonance_gate,
            noise_gate=noise_gate,
            latency_calibration=latency_calibration,
        ),
        "step_count": int(step_count),
    }


def run_substrate_telemetry_resonance(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    program_payload = dict(payload or {})
    frames = _prepare_telemetry_frames(program_payload)
    horizon_frames = max(1, int(program_payload.get("horizon_frames", 3) or 3))
    pulse_cycles = max(1, int(program_payload.get("pulse_cycles", program_payload.get("cycles", 4)) or 4))
    program_label = str(program_payload.get("program_id", "telemetry_resonance"))
    profiles = _build_telemetry_profiles(frames, horizon_frames, program_payload)
    encoded_profiles = [_encode_telemetry_frame_profile(profile) for profile in profiles]
    execution = _run_encoded_telemetry_resonance(
        encoded_profiles=encoded_profiles,
        pulse_cycles=pulse_cycles,
        program_label=program_label,
        previous_trace_state=previous_trace_state,
    )
    return {
        "ok": True,
        "path": "substrate_telemetry",
        "program_id": program_label,
        "frame_count": int(len(frames)),
        "pulse_cycles": int(pulse_cycles),
        "horizon_frames": int(horizon_frames),
        "telemetry_source": str(dict(frames[0] if frames else {}).get("source", "synthetic")),
        "actuation_summary": _telemetry_actuation_summary(frames),
        "trace_state": dict(execution.get("trace_state", {})),
        "result": dict(execution.get("result", {})),
        "forecast_preview": [
            {
                "timestamp": str(profile.get("timestamp", "")),
                "source": str(profile.get("source", "unknown")),
                "dominant_metric": str(dict(list(profile.get("dominant_nodes", [{}]))[0]).get("metric", "")),
                "resonance_gate": float(profile.get("resonance_gate", 0.0)),
                "noise_gate": float(profile.get("noise_gate", 0.0)),
                "predicted_latency_s": float(dict(profile.get("latency_calibration", {}) or {}).get("predicted_latency_s", 0.0)),
                "actuation_compensation": float(dict(profile.get("latency_calibration", {}) or {}).get("actuation_compensation", 0.0)),
                "actuation_horizon_frames": float(dict(profile.get("latency_calibration", {}) or {}).get("actuation_horizon_frames", 0.0)),
                "phase_transport_term": float(dict(profile.get("latency_calibration", {}) or {}).get("phase_transport_term", 0.0)),
                "reverse_transport_gate": float(dict(profile.get("latency_calibration", {}) or {}).get("reverse_transport_gate", 0.0)),
                "trajectory_spectral_id": str(dict(profile.get("latency_calibration", {}) or {}).get("trajectory_spectral_id", "")),
                "trajectory_conservation_alignment": float(dict(profile.get("latency_calibration", {}) or {}).get("trajectory_conservation_alignment", 0.0)),
                "trajectory_expansion_term": float(dict(profile.get("latency_calibration", {}) or {}).get("trajectory_expansion_term", 0.0)),
                "transport_input_mode": str(dict(profile.get("latency_calibration", {}) or {}).get("transport_input_mode", "telemetry_only")),
            }
            for profile in profiles[:4]
        ],
    }


def run_classical_telemetry_resonance(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    program_payload = dict(payload or {})
    frames = _prepare_telemetry_frames(program_payload)
    horizon_frames = max(1, int(program_payload.get("horizon_frames", 3) or 3))
    pulse_cycles = max(1, int(program_payload.get("pulse_cycles", program_payload.get("cycles", 4)) or 4))
    program_label = str(program_payload.get("program_id", "telemetry_resonance"))
    profiles = _build_telemetry_profiles(frames, horizon_frames, program_payload)
    encoded_profiles = [_encode_telemetry_frame_profile(profile) for profile in profiles]
    trace_state = dict(previous_trace_state or {})
    accumulators = {
        "dispatch_word": 0,
        "string_word": 0,
        "structure_word": 0,
    }
    step_count = 0
    dominant_nodes: list[Dict[str, Any]] = []
    resonance_gate = 0.0
    noise_gate = 0.0
    latency_calibration: Dict[str, Any] = {}
    for cycle_index in range(pulse_cycles):
        for frame_index in range(len(frames)):
            profile = dict(profiles[frame_index])
            encoded_profile = dict(encoded_profiles[frame_index])
            pulse_index = cycle_index * max(1, len(frames)) + frame_index
            inputs = _microprocess_trace_inputs(
                encoded_artifact=encoded_profile,
                pulse_index=pulse_index,
                program_label=program_label,
                previous_trace_state=trace_state,
            )
            trace_state = _update_substrate_trace_state(
                pulse_index=int(inputs["pulse_index"]),
                previous_trace_state=dict(inputs["previous_trace_state"]),
                simulation_field_state=dict(inputs["simulation_field_state"]),
                gpu_feedback=dict(inputs["gpu_feedback"]),
                gpu_pulse_delta_feedback=dict(inputs["gpu_pulse_delta_feedback"]),
                interference_field=dict(inputs["interference_field"]),
                effective_vector=dict(inputs["effective_vector"]),
                kernel_execution_event=dict(inputs["kernel_execution_event"]),
                trace_label=str(inputs["trace_label"]),
            )
            accumulators = _microprocess_decode_step(
                accumulators=accumulators,
                encoded_artifact=encoded_profile,
                trace_state=trace_state,
                step_index=frame_index,
                pulse_index=pulse_index,
            )
            dominant_nodes = list(encoded_profile.get("resonant_nodes", dominant_nodes))
            resonance_gate = float(encoded_profile.get("resonance_gate", resonance_gate))
            noise_gate = float(encoded_profile.get("noise_gate", noise_gate))
            latency_calibration = dict(encoded_profile.get("latency_calibration", latency_calibration))
            step_count += 1
    return {
        "ok": True,
        "path": "classical_telemetry",
        "program_id": program_label,
        "frame_count": int(len(frames)),
        "pulse_cycles": int(pulse_cycles),
        "horizon_frames": int(horizon_frames),
        "telemetry_source": str(dict(frames[0] if frames else {}).get("source", "synthetic")),
        "actuation_summary": _telemetry_actuation_summary(frames),
        "trace_state": dict(trace_state),
        "result": _telemetry_result_summary(
            accumulators=accumulators,
            trace_state=trace_state,
            step_count=step_count,
            dominant_nodes=dominant_nodes,
            resonance_gate=resonance_gate,
            noise_gate=noise_gate,
            latency_calibration=latency_calibration,
        ),
    }


def benchmark_telemetry_resonance(
    payload: Dict[str, Any],
    previous_trace_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    program_payload = dict(payload or {})
    frames = _prepare_telemetry_frames(program_payload)
    actuation_summary = _telemetry_actuation_summary(frames)
    frozen_payload = dict(program_payload)
    frozen_payload["frames"] = list(frames)
    frozen_payload["telemetry_mode"] = "provided"
    frozen_payload["telemetry_source"] = "provided"
    frozen_payload["use_live_telemetry"] = False
    horizon_frames = max(1, int(program_payload.get("horizon_frames", 3) or 3))
    pulse_cycles = max(1, int(program_payload.get("pulse_cycles", program_payload.get("cycles", 4)) or 4))
    repeat_count = max(1, int(program_payload.get("repeat_count", 3) or 3))
    warmup_count = max(0, int(program_payload.get("warmup_count", 1) or 1))
    program_label = str(program_payload.get("program_id", "telemetry_resonance"))
    profiles = _build_telemetry_profiles(frames, horizon_frames, frozen_payload)
    encoded_profiles = [_encode_telemetry_frame_profile(profile) for profile in profiles]

    for _ in range(warmup_count):
        _run_encoded_telemetry_resonance(
            encoded_profiles=encoded_profiles,
            pulse_cycles=pulse_cycles,
            program_label=program_label,
            previous_trace_state=previous_trace_state,
        )
        _ = run_classical_telemetry_resonance(frozen_payload, previous_trace_state=previous_trace_state)

    profile_samples: list[int] = []
    substrate_exec_samples: list[int] = []
    classical_samples: list[int] = []
    substrate_result = run_substrate_telemetry_resonance(frozen_payload, previous_trace_state=previous_trace_state)
    classical_result = run_classical_telemetry_resonance(frozen_payload, previous_trace_state=previous_trace_state)
    results_match = dict(substrate_result.get("result", {})) == dict(classical_result.get("result", {}))

    for _ in range(repeat_count):
        t0 = time.perf_counter_ns()
        _ = _build_telemetry_profiles(frames, horizon_frames, frozen_payload)
        profile_samples.append(int(time.perf_counter_ns() - t0))

        t1 = time.perf_counter_ns()
        _ = _run_encoded_telemetry_resonance(
            encoded_profiles=encoded_profiles,
            pulse_cycles=pulse_cycles,
            program_label=program_label,
            previous_trace_state=previous_trace_state,
        )
        substrate_exec_samples.append(int(time.perf_counter_ns() - t1))

        t2 = time.perf_counter_ns()
        _ = run_classical_telemetry_resonance(frozen_payload, previous_trace_state=previous_trace_state)
        classical_samples.append(int(time.perf_counter_ns() - t2))

    def _avg_ns(samples: list[int]) -> int:
        if not samples:
            return 0
        return int(sum(samples) / max(1, len(samples)))

    profile_avg_ns = _avg_ns(profile_samples)
    substrate_exec_avg_ns = _avg_ns(substrate_exec_samples)
    classical_avg_ns = _avg_ns(classical_samples)
    substrate_total_avg_ns = int(profile_avg_ns + substrate_exec_avg_ns)
    speedup_total = 0.0
    speedup_exec_only = 0.0
    if substrate_total_avg_ns > 0:
        speedup_total = float(classical_avg_ns) / float(substrate_total_avg_ns)
    if substrate_exec_avg_ns > 0:
        speedup_exec_only = float(classical_avg_ns) / float(substrate_exec_avg_ns)

    hash_probe = _telemetry_hash_probe(
        dict(dict(substrate_result.get("result", {}) or {})),
        frozen_payload,
        profiles=profiles,
        encoded_profiles=encoded_profiles,
    )

    result_map = {
        "ok": True,
        "program_id": program_label,
        "frame_count": int(len(frames)),
        "horizon_frames": int(horizon_frames),
        "pulse_cycles": int(pulse_cycles),
        "telemetry_source": str(dict(frames[0] if frames else {}).get("source", "synthetic")),
        "actuation_summary": actuation_summary,
        "repeat_count": int(repeat_count),
        "warmup_count": int(warmup_count),
        "results_match": bool(results_match),
        "profile_avg_ns": int(profile_avg_ns),
        "substrate_exec_avg_ns": int(substrate_exec_avg_ns),
        "substrate_total_avg_ns": int(substrate_total_avg_ns),
        "classical_avg_ns": int(classical_avg_ns),
        "speedup_total": float(speedup_total),
        "speedup_exec_only": float(speedup_exec_only),
        "hash_probe": hash_probe,
        "forecast_preview": list(substrate_result.get("forecast_preview", []) or []),
        "substrate_result": substrate_result,
        "classical_result": classical_result,
    }

    compare_default = bool(actuation_summary.get("applied", False))
    compare_modes = _flag_enabled(program_payload.get("compare_raw_transport_modes", compare_default), compare_default)
    if compare_modes and bool(frames):
        raw_payload = dict(frozen_payload)
        raw_payload["program_id"] = program_label + "_raw_replay"
        raw_payload["frames"] = _raw_transport_replay_frames(frames)
        raw_payload["compare_raw_transport_modes"] = False
        raw_payload["use_live_telemetry"] = False
        raw_payload["telemetry_mode"] = "provided"
        raw_payload["telemetry_source"] = "provided"
        raw_benchmark = benchmark_telemetry_resonance(raw_payload, previous_trace_state=previous_trace_state)
        adjusted_summary = _transport_mode_benchmark_summary(result_map)
        raw_summary = _transport_mode_benchmark_summary(raw_benchmark)
        result_map["transport_mode_comparison"] = {
            "adjusted": adjusted_summary,
            "raw_replay": raw_summary,
            "deltas": {
                "predicted_latency_s": float(adjusted_summary.get("predicted_latency_s", 0.0) - raw_summary.get("predicted_latency_s", 0.0)),
                "phase_transport_term": float(adjusted_summary.get("phase_transport_term", 0.0) - raw_summary.get("phase_transport_term", 0.0)),
                "flux_transport_term": float(adjusted_summary.get("flux_transport_term", 0.0) - raw_summary.get("flux_transport_term", 0.0)),
                "trajectory_conservation_alignment": float(adjusted_summary.get("trajectory_conservation_alignment", 0.0) - raw_summary.get("trajectory_conservation_alignment", 0.0)),
                "trajectory_prediction_alignment": float(adjusted_summary.get("trajectory_prediction_alignment", 0.0) - raw_summary.get("trajectory_prediction_alignment", 0.0)),
                "trajectory_expansion_term": float(adjusted_summary.get("trajectory_expansion_term", 0.0) - raw_summary.get("trajectory_expansion_term", 0.0)),
                "temporal_relativity_norm": float(adjusted_summary.get("temporal_relativity_norm", 0.0) - raw_summary.get("temporal_relativity_norm", 0.0)),
                "field_interference_norm": float(adjusted_summary.get("field_interference_norm", 0.0) - raw_summary.get("field_interference_norm", 0.0)),
                "resonant_interception_inertia": float(adjusted_summary.get("resonant_interception_inertia", 0.0) - raw_summary.get("resonant_interception_inertia", 0.0)),
                "speedup_total": float(adjusted_summary.get("speedup_total", 0.0) - raw_summary.get("speedup_total", 0.0)),
                "hash_distance_score": float(adjusted_summary.get("hash_distance_score", 0.0) - raw_summary.get("hash_distance_score", 0.0)),
                "hash_zero_nibbles": int(adjusted_summary.get("hash_zero_nibbles", 0) - raw_summary.get("hash_zero_nibbles", 0)),
                "hash_valid": int(bool(adjusted_summary.get("hash_valid", False))) - int(bool(raw_summary.get("hash_valid", False))),
            },
        }

    return result_map
