# Path: core/constants_eq.py
# Description:
#   Centralized accessors for effective constants under the DMT policy.
#   - NEVER returns raw baseline constants for compute paths.
#   - Produces h_eff, k_b_eff, c_eff, e_charge_eff, mu_0_eff, epsilon_0_eff
#     as derived constants driven by environment and temporal dynamics.
#   - Temporal dynamics uses phase, frequency, wavelength, amplitude, zero-point
#     proximity, and vector-field alignment to shape relative time and resonance.
#   - Caches results per operating point with TTL derived from telemetry tick.
#   - Exposes snapshot/save for telemetry and downstream modules.
#
#   All values are calculated from environment inputs; there are no arbitrary outputs.
#   Reference baselines appear only as internal parameters inside effective transforms.

from typing import Dict, List, Tuple
from core import utils
import math
import hashlib

# Cache entry: (timestamp_s, payload_dict)
_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}

def _now_s() -> float:
    # Use env time if present for determinism; else derive a stable fallback.
    t = utils.get("env/time_s", None)
    if t is None:
        anchor = utils.get("env/temperature_k", 300.0)
        h = hashlib.sha256(f"{anchor}".encode("ascii", "ignore")).hexdigest()
        return (int(h, 16) % 10_000_000) / 10.0
    return float(t)

def _ttl_s() -> float:
    tick_ms = float(utils.get("policy/telemetry_tick_ms", 400.0))
    return max(0.05, 0.25 * tick_ms / 1000.0)

def _wrap_phase(phase_radians: float) -> float:
    return math.atan2(math.sin(float(phase_radians)), math.cos(float(phase_radians)))

def _unit(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    nrm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if nrm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) / nrm for v in vec)

def _alignment(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ua = _unit(a)
    ub = _unit(b)
    dot = sum(ua[i] * ub[i] for i in range(3))
    return max(0.0, min(1.0, 0.5 * (dot + 1.0)))

def _key(temperature_k: float,
         v_frac: float,
         flux_factor: float,
         strain_factor: float,
         phase_radians: float,
         frequency_hz: float,
         wavelength_m: float,
         amplitude: float,
         zero_point_offset: float,
         vector_field: Tuple[float, float, float],
         spin_vector: Tuple[float, float, float],
         orientation_vector: Tuple[float, float, float]) -> str:
    mb = float(utils.get("hw/mem_bandwidth_gbps", 100.0))
    tp = float(utils.get("hw/compute_throughput_tops", 10.0))
    return (
        f"{temperature_k:.8f}|{v_frac:.8f}|{flux_factor:.8f}|{strain_factor:.8f}|"
        f"{phase_radians:.8f}|{frequency_hz:.8f}|{wavelength_m:.8f}|{amplitude:.8f}|"
        f"{zero_point_offset:.8f}|"
        f"{vector_field[0]:.8f},{vector_field[1]:.8f},{vector_field[2]:.8f}|"
        f"{spin_vector[0]:.8f},{spin_vector[1]:.8f},{spin_vector[2]:.8f}|"
        f"{orientation_vector[0]:.8f},{orientation_vector[1]:.8f},{orientation_vector[2]:.8f}|"
        f"{mb:.4f}|{tp:.4f}"
    )

def _temporal_dynamics(phase_radians: float,
                       frequency_hz: float,
                       wavelength_m: float,
                       amplitude: float,
                       zero_point_offset: float,
                       vector_field: Tuple[float, float, float],
                       spin_vector: Tuple[float, float, float],
                       orientation_vector: Tuple[float, float, float]) -> Dict[str, float]:
    phase = _wrap_phase(phase_radians)
    zero_phase = _wrap_phase(zero_point_offset)
    phase_gap = abs(_wrap_phase(phase - zero_phase))
    frequency = max(1.0e-12, float(frequency_hz))
    wavelength = max(1.0e-12, float(wavelength_m))
    amp = max(0.0, float(amplitude))

    phase_alignment = max(0.0, min(1.0, 0.5 * (1.0 + math.cos(phase_gap))))
    zero_point_overlap = 1.0 / (1.0 + amp * abs(math.sin(phase_gap)))
    vector_alignment = _alignment(vector_field, orientation_vector)
    spin_alignment = _alignment(vector_field, spin_vector)
    orientation_alignment = _alignment(orientation_vector, spin_vector)

    wave_path_length = wavelength * (1.0 + amp * abs(math.sin(phase)))
    wave_path_speed = max(1.0e-12, frequency * wave_path_length)
    cycle_time_s = 1.0 / frequency

    resonance_inertia = math.tanh(amp * frequency * wavelength / (1.0 + phase_gap))
    phase_alignment_probability = phase_alignment * (0.4 + 0.6 * vector_alignment)
    entanglement_weight = (
        phase_alignment_probability *
        zero_point_overlap *
        (0.4 + 0.6 * spin_alignment) *
        (0.4 + 0.6 * orientation_alignment)
    )
    temporal_relativity = (
        1.0 +
        0.025 * phase_alignment +
        0.020 * entanglement_weight +
        0.015 * resonance_inertia +
        0.010 * spin_alignment
    )
    relative_temporal_position = phase / (2.0 * math.pi)

    return {
        "phase_radians": phase,
        "frequency_hz": frequency,
        "wavelength_m": wavelength,
        "amplitude": amp,
        "zero_point_offset": zero_phase,
        "phase_gap": phase_gap,
        "phase_alignment": phase_alignment,
        "zero_point_overlap": zero_point_overlap,
        "vector_alignment": vector_alignment,
        "spin_alignment": spin_alignment,
        "orientation_alignment": orientation_alignment,
        "wave_path_length": wave_path_length,
        "wave_path_speed": wave_path_speed,
        "cycle_time_s": cycle_time_s,
        "resonance_inertia": resonance_inertia,
        "phase_alignment_probability": phase_alignment_probability,
        "entanglement_weight": entanglement_weight,
        "temporal_relativity": temporal_relativity,
        "relative_temporal_position": relative_temporal_position,
    }

def _effective_from_env(temperature_k: float,
                        velocity_fraction_c: float,
                        flux_factor: float,
                        strain_factor: float,
                        phase_radians: float,
                        frequency_hz: float,
                        wavelength_m: float,
                        amplitude: float,
                        zero_point_offset: float,
                        vector_field: Tuple[float, float, float],
                        spin_vector: Tuple[float, float, float],
                        orientation_vector: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Compute effective constants from VSD-driven inputs.
    This function keeps physical reference constants only as internal parameters.
    """
    h_ref = 6.62607015e-34
    k_b_ref = 1.380649e-23
    c_ref = 299792458.0
    e_ref = 1.602176634e-19
    mu0_ref = 1.25663706212e-6
    eps0_ref = 8.8541878128e-12

    stochastic = utils.stochastic_dispersion_factor(temperature_k)
    relativistic = utils.relativistic_correlation(velocity_fraction_c, flux_factor, strain_factor)
    temporal = _temporal_dynamics(
        phase_radians,
        frequency_hz,
        wavelength_m,
        amplitude,
        zero_point_offset,
        vector_field,
        spin_vector,
        orientation_vector,
    )

    mb = max(1.0, float(utils.get("hw/mem_bandwidth_gbps", 100.0)))
    tp = max(1.0, float(utils.get("hw/compute_throughput_tops", 10.0)))
    env_scale = 1.0 + 0.002 * math.tanh((tp / mb) - 0.1)
    temporal_scale = temporal["temporal_relativity"] * (1.0 + 0.05 * temporal["entanglement_weight"])

    scale = (relativistic / (1.0 + 0.25 * (stochastic - 1.0))) * env_scale * temporal_scale

    c_eff = (
        c_ref *
        (1.0 + 0.005 * (relativistic - 1.0)) *
        (1.0 + 0.002 * (temporal["temporal_relativity"] - 1.0)) /
        (1.0 + 0.002 * (stochastic - 1.0))
    )
    h_eff = h_ref * scale * (1.0 + 0.02 * temporal["entanglement_weight"])
    k_b_eff = (
        k_b_ref *
        (1.0 + 0.15 * (stochastic - 1.0)) *
        (1.0 + 0.01 * temporal["resonance_inertia"]) /
        (1.0 + 0.01 * (relativistic - 1.0))
    )
    e_eff = (
        e_ref *
        (1.0 + 0.001 * (relativistic - 1.0) + 0.0005 * temporal["phase_alignment"]) *
        (1.0 - 0.0005 * (stochastic - 1.0))
    )
    mu0_eff = mu0_ref * (1.0 + 0.01 * (stochastic - 1.0)) * (1.0 + 0.002 * temporal["vector_alignment"])
    eps0_eff = eps0_ref * (1.0 - 0.01 * (stochastic - 1.0)) / (1.0 + 0.002 * temporal["vector_alignment"])

    return {
        "h_eff": h_eff,
        "k_b_eff": k_b_eff,
        "c_eff": c_eff,
        "e_charge_eff": e_eff,
        "mu_0_eff": mu0_eff,
        "epsilon_0_eff": eps0_eff,
        "scale_eff": scale,
        "stochastic_factor": stochastic,
        "relativistic_corr": relativistic,
        "env_scale": env_scale,
        "temporal_scale": temporal_scale,
        "temporal_relativity": temporal["temporal_relativity"],
        "phase_alignment": temporal["phase_alignment"],
        "phase_alignment_probability": temporal["phase_alignment_probability"],
        "zero_point_overlap": temporal["zero_point_overlap"],
        "vector_alignment": temporal["vector_alignment"],
        "spin_alignment": temporal["spin_alignment"],
        "orientation_alignment": temporal["orientation_alignment"],
        "wave_path_length": temporal["wave_path_length"],
        "wave_path_speed": temporal["wave_path_speed"],
        "cycle_time_s": temporal["cycle_time_s"],
        "resonance_inertia": temporal["resonance_inertia"],
        "entanglement_weight": temporal["entanglement_weight"],
        "relative_temporal_position": temporal["relative_temporal_position"],
        "phase_radians": temporal["phase_radians"],
        "frequency_hz": temporal["frequency_hz"],
        "wavelength_m": temporal["wavelength_m"],
        "amplitude": temporal["amplitude"],
        "zero_point_offset": temporal["zero_point_offset"],
    }

def get_effective(temperature_k: float = None,
                  velocity_fraction_c: float = None,
                  flux_factor: float = None,
                  strain_factor: float = None,
                  phase_radians: float = None,
                  frequency_hz: float = None,
                  wavelength_m: float = None,
                  amplitude: float = None,
                  zero_point_offset: float = None,
                  vector_field: Tuple[float, float, float] = None,
                  spin_vector: Tuple[float, float, float] = None,
                  orientation_vector: Tuple[float, float, float] = None) -> Dict[str, float]:
    """
    Returns a dict with effective constants derived from environment data.
    Any missing argument is read live from VSD via utils.env_* probes.
    Cached with TTL linked to telemetry tick for stability.
    """
    T = float(temperature_k) if temperature_k is not None else utils.env_temperature_k()
    v = float(velocity_fraction_c) if velocity_fraction_c is not None else utils.env_velocity_fraction_c()
    f = float(flux_factor) if flux_factor is not None else utils.env_flux_factor()
    s = float(strain_factor) if strain_factor is not None else utils.env_strain_factor()
    phase = float(phase_radians) if phase_radians is not None else utils.env_phase_radians()
    freq = float(frequency_hz) if frequency_hz is not None else utils.env_frequency_hz()
    wave = float(wavelength_m) if wavelength_m is not None else utils.env_wavelength_m()
    amp = float(amplitude) if amplitude is not None else utils.env_amplitude()
    zero = float(zero_point_offset) if zero_point_offset is not None else utils.env_zero_point_offset()
    vf = tuple(vector_field) if vector_field is not None else utils.env_vector_field()
    spin = tuple(spin_vector) if spin_vector is not None else utils.env_spin_vector()
    orient = tuple(orientation_vector) if orientation_vector is not None else utils.env_orientation_vector()

    k = _key(T, v, f, s, phase, freq, wave, amp, zero, vf, spin, orient)
    now = _now_s()
    ttl = _ttl_s()
    cached = _CACHE.get(k)
    if cached is not None:
        ts, payload = cached
        if (now - ts) <= ttl:
            return payload

    eff = _effective_from_env(T, v, f, s, phase, freq, wave, amp, zero, vf, spin, orient)
    _CACHE[k] = (now, eff)
    return eff

def clear_cache() -> None:
    _CACHE.clear()

def snapshot_to_vsd(vsd_key: str = "telemetry/effective_constants_snapshot") -> Dict[str, float]:
    eff = get_effective()
    utils.store(vsd_key, eff)
    return eff

def export_as_json() -> str:
    return json_dumps_ascii(get_effective())

def json_dumps_ascii(obj: Dict[str, float]) -> str:
    import json as _json
    return _json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

def diagnostic_trend_check(samples: int = 8) -> Dict[str, float]:
    """
    Sweeps small changes in temperature, velocity, and phase to ensure effective constants
    change smoothly. Returns aggregate statistics for monitoring.
    """
    base_T = utils.env_temperature_k()
    base_v = utils.env_velocity_fraction_c()
    base_f = utils.env_flux_factor()
    base_s = utils.env_strain_factor()
    base_phase = utils.env_phase_radians()

    vals_h: List[float] = []
    vals_kb: List[float] = []
    vals_c: List[float] = []
    vals_temporal: List[float] = []
    vals_entangle: List[float] = []

    for i in range(max(1, int(samples))):
        dT = (i - samples / 2.0) * 0.5
        dv = (i - samples / 2.0) * 0.001
        dphase = (i - samples / 2.0) * 0.05
        eff = get_effective(
            base_T + dT,
            max(0.0, min(0.999999, base_v + dv)),
            base_f,
            base_s,
            base_phase + dphase,
        )
        vals_h.append(eff["h_eff"])
        vals_kb.append(eff["k_b_eff"])
        vals_c.append(eff["c_eff"])
        vals_temporal.append(eff["temporal_relativity"])
        vals_entangle.append(eff["entanglement_weight"])

    def _spread(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return max(xs) - min(xs)

    out = {
        "samples": float(samples),
        "h_eff_spread": float(_spread(vals_h)),
        "k_b_eff_spread": float(_spread(vals_kb)),
        "c_eff_spread": float(_spread(vals_c)),
        "temporal_relativity_spread": float(_spread(vals_temporal)),
        "entanglement_weight_spread": float(_spread(vals_entangle)),
    }
    utils.store("telemetry/constants_trend", out)
    return out
