from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import cuda as numba_cuda
except Exception:
    numba_cuda = None


PROTOTYPING_ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUN41_SUMMARY = ROOT / "Run41" / "run_041b_summary.json"
NIST_REFERENCE = ROOT / "nist_silicon_reference.json"
PULSE_OPERATORS = ROOT / "photon_pulse_deviation_operators_2060.json"
TEMPORAL_COUPLING = ROOT / "temporal_coupling_encoding_schema_2060.json"
PROCESS_SUBSTRATE = ROOT / "process_substrate_calculus_encoding_schema.json"
ROOT_STATE = ROOT / "state.json"
MINER_RUNTIME_CONFIG = REPO_ROOT / "miner" / "miner_runtime_config.json"
SUBSTRATE_TRACE_VRAM_CACHE: dict[str, Any] = {}


if numba_cuda is not None:
    @numba_cuda.jit
    def temporal_candidate_cuda_kernel(
        features: Any,
        weights: Any,
        target_vector: Any,
        trace_axis: Any,
        trace_summary: Any,
        harmonic_weight: float,
        scores: Any,
        nonce_biases: Any,
    ) -> None:
        idx = numba_cuda.grid(1)
        if idx >= features.shape[0]:
            return
        accum = 0.0
        proximity = 0.0
        trace_axis_alignment = 0.0
        trace_temporal_alignment = 0.0
        bias = 0
        for feat_idx in range(features.shape[1]):
            feat_value = features[idx, feat_idx]
            accum += feat_value * weights[feat_idx]
            if feat_idx < target_vector.shape[0]:
                delta = feat_value - target_vector[feat_idx]
                if delta < 0.0:
                    delta = -delta
                proximity += delta
            bias = (
                (bias * 1664525)
                + int((feat_value + weights[feat_idx]) * 1000000.0)
                + 1013904223
            ) & 0xFFFFFFFF
        if target_vector.shape[0] > 0:
            proximity = 1.0 - (proximity / target_vector.shape[0])
        if proximity < 0.0:
            proximity = 0.0
        if trace_axis.shape[0] >= 4:
            axis_delta = features[idx, 0] - trace_axis[0]
            if axis_delta < 0.0:
                axis_delta = -axis_delta
            trace_axis_alignment += 0.28 * (1.0 - min(axis_delta, 1.0))
            axis_delta = features[idx, 5] - trace_axis[1]
            if axis_delta < 0.0:
                axis_delta = -axis_delta
            trace_axis_alignment += 0.24 * (1.0 - min(axis_delta, 1.0))
            axis_delta = features[idx, 11] - trace_axis[2]
            if axis_delta < 0.0:
                axis_delta = -axis_delta
            trace_axis_alignment += 0.22 * (1.0 - min(axis_delta, 1.0))
            axis_delta = features[idx, 12] - trace_axis[3]
            if axis_delta < 0.0:
                axis_delta = -axis_delta
            trace_axis_alignment += 0.26 * (1.0 - min(axis_delta, 1.0))
        if trace_summary.shape[0] >= 10:
            temporal_delta = features[idx, 10] - trace_summary[6]
            if temporal_delta < 0.0:
                temporal_delta = -temporal_delta
            trace_temporal_alignment += 0.18 * (1.0 - min(temporal_delta, 1.0))
            temporal_delta = features[idx, 11] - trace_summary[7]
            if temporal_delta < 0.0:
                temporal_delta = -temporal_delta
            trace_temporal_alignment += 0.18 * (1.0 - min(temporal_delta, 1.0))
            temporal_delta = features[idx, 12] - trace_summary[8]
            if temporal_delta < 0.0:
                temporal_delta = -temporal_delta
            trace_temporal_alignment += 0.16 * (1.0 - min(temporal_delta, 1.0))
            temporal_delta = features[idx, 13] - trace_summary[9]
            if temporal_delta < 0.0:
                temporal_delta = -temporal_delta
            trace_temporal_alignment += 0.16 * (1.0 - min(temporal_delta, 1.0))
            temporal_delta = features[idx, 15] - trace_summary[1]
            if temporal_delta < 0.0:
                temporal_delta = -temporal_delta
            trace_temporal_alignment += 0.16 * (1.0 - min(temporal_delta, 1.0))
            temporal_delta = features[idx, 17] - trace_summary[2]
            if temporal_delta < 0.0:
                temporal_delta = -temporal_delta
            trace_temporal_alignment += 0.16 * (1.0 - min(temporal_delta, 1.0))
        trace_axis_alignment = min(max(trace_axis_alignment, 0.0), 1.0)
        trace_temporal_alignment = min(max(trace_temporal_alignment, 0.0), 1.0)
        scores[idx] = (
            accum
            + proximity * (0.18 + 0.17 * harmonic_weight)
            + trace_axis_alignment * (0.10 + 0.14 * harmonic_weight)
            + trace_temporal_alignment * (0.12 + 0.16 * harmonic_weight)
        )
        bias = (
            bias
            + int((trace_axis_alignment + trace_temporal_alignment + harmonic_weight) * 1000000.0)
        ) & 0xFFFFFFFF
        nonce_biases[idx] = bias

def sha256d_compute_share(job: dict[str, Any], nonce_int: int) -> dict[str, Any]:
    header_hex = str(job.get("header_hex", job.get("header", ""))).strip()
    if header_hex.startswith("0x"):
        header_hex = header_hex[2:]
    if len(header_hex) % 2:
        header_hex = "0" + header_hex
    try:
        header = bytes.fromhex(header_hex)
    except Exception:
        header = b""
    hdr = bytearray(header[:80].ljust(80, b"\x00"))
    struct.pack_into("<I", hdr, 76, nonce_int & 0xFFFFFFFF)
    first_digest = hashlib.sha256(bytes(hdr)).digest()
    digest = hashlib.sha256(first_digest).digest()
    function_metrics = sha256d_function_metrics(first_digest, digest)
    return {
        "nonce": "0x%08x" % (nonce_int & 0xFFFFFFFF),
        "header": bytes(hdr).hex(),
        "first_hash_hex": first_digest.hex(),
        "hash_hex": digest.hex(),
        "sha256_function_score": float(function_metrics.get("function_score", 0.0)),
        "sha256_round_coupling": float(function_metrics.get("round_coupling", 0.0)),
        "sha256_first_entropy": float(function_metrics.get("first_entropy", 0.0)),
        "sha256_final_entropy": float(function_metrics.get("final_entropy", 0.0)),
        "sha256_transition_ratio": float(function_metrics.get("transition_ratio", 0.0)),
        "sha256_bit_balance": float(function_metrics.get("bit_balance", 0.0)),
        "sha256_lane_balance": float(function_metrics.get("lane_balance", 0.0)),
    }


@dataclass
class CohortSeed:
    name: str
    freq_norm: float
    amp_norm: float
    volt_norm: float
    curr_norm: float


@dataclass
class PacketState:
    packet_id: int
    cohort: str
    cohort_index: int
    topological_charge: int
    amplitude_drive: float
    frequency_drive: float
    voltage_drive: float
    amperage_drive: float
    retro_gain: float
    spectrum: np.ndarray


@dataclass
class SimulationConfig:
    packet_count: int = 32
    bin_count: int = 128
    steps: int = 48
    recon_samples: int = 256
    equivalent_grid_linear: int = 256
    seed: int = 41
    low_bin_count: int = 12
    kappa_a: float = 0.085
    kappa_f: float = 0.055
    kappa_couple: float = 0.080
    kappa_leak: float = 0.035
    kappa_trap: float = 0.120
    kappa_retro: float = 0.085
    max_amplitude: float = 2.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure frequency-domain photon confinement simulator."
    )
    parser.add_argument("--packet-count", type=int, default=32)
    parser.add_argument("--bin-count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--recon-samples", type=int, default=256)
    parser.add_argument("--equivalent-grid-linear", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "frequency_domain_runs" / "latest"),
        help="Directory for plots, summaries, and debug views.",
    )
    parser.add_argument(
        "--write-root-samples",
        action="store_true",
        help="Refresh the root photon sample JSON/CSV files used by the current archive loader.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        text = handle.read()
    if text.lstrip().startswith("//"):
        text = "\n".join(text.splitlines()[1:])
    return json.loads(text)


def load_cohort_seeds() -> list[CohortSeed]:
    summary = load_json(RUN41_SUMMARY)
    best_cases = summary["best_cases"]
    order = ["D_track", "I_accum", "L_smooth"]
    out: list[CohortSeed] = []
    for name in order:
        case = best_cases[name]
        out.append(
            CohortSeed(
                name=name,
                freq_norm=float(case["freq_norm"]),
                amp_norm=float(case["amp_norm"]),
                volt_norm=float(case["volt_norm"]),
                curr_norm=float(case["curr_norm"]),
            )
        )
    return out


def load_nist_reference() -> dict[str, float]:
    raw = load_json(NIST_REFERENCE)
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / max(width, 1.0e-6)) ** 2)


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coords / max(sigma, 1.0e-6)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def blur_along_bins(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = gaussian_kernel(radius, sigma)
    padded = np.pad(values, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    out = np.empty_like(values)
    for packet_idx in range(values.shape[0]):
        for axis_idx in range(values.shape[1]):
            out[packet_idx, axis_idx] = np.convolve(
                padded[packet_idx, axis_idx], kernel, mode="valid"
            )
    return out


def write_json_with_comment(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"// {path.name}\n")
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv_with_comment(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"// {path.name}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def extract_basin_id(cluster_id: str, default: str = "electron_basin") -> str:
    cluster_text = str(cluster_id or "")
    for basin_id in ("electron_basin", "phonon_basin", "exciton_basin", "hole_basin"):
        if basin_id in cluster_text:
            return basin_id
    return str(default or "electron_basin")


def load_submission_rate_anchor(symbol: str = "BTC") -> dict[str, float]:
    fallback = {
        "allowed_rate_per_second": 2.0,
        "tick_duration_s": 0.5,
        "jitter_fraction": 0.18,
    }
    try:
        payload = load_json(MINER_RUNTIME_CONFIG)
        if not isinstance(payload, dict):
            return fallback
        submission_rate = dict(payload.get("submission_rate", {}) or {})
        coins = dict(payload.get("coins", {}) or {})
        coin_config = dict(coins.get(str(symbol).upper(), {}) or {})
        fallback_allowed = float(submission_rate.get("fallback_allowed_rate_per_second", 2.0))
        pool_ceiling = float(
            coin_config.get(
                "pool_submit_ceiling_per_second",
                submission_rate.get("pool_submit_ceiling_per_second", fallback_allowed),
            )
        )
        network_ceiling = float(
            coin_config.get(
                "network_submit_ceiling_per_second",
                submission_rate.get("network_submit_ceiling_per_second", fallback_allowed),
            )
        )
        guard_fraction = float(
            coin_config.get(
                "pool_submit_guard_fraction",
                submission_rate.get("pool_submit_guard_fraction", 0.985),
            )
        )
        allowed_rate = max(
            0.05,
            min(pool_ceiling * guard_fraction, network_ceiling, fallback_allowed if fallback_allowed > 0.0 else network_ceiling),
        )
        min_tick_duration = float(submission_rate.get("min_tick_duration_s", 0.05))
        tick_duration = max(min_tick_duration, 1.0 / max(allowed_rate, 1.0e-6))
        jitter_fraction = clamp01(float(submission_rate.get("jitter_fraction", 0.18)))
        return {
            "allowed_rate_per_second": float(allowed_rate),
            "tick_duration_s": float(tick_duration),
            "jitter_fraction": float(jitter_fraction),
        }
    except Exception:
        return fallback


def window_norm(value: float, window: dict[str, Any]) -> float:
    lower = float(window.get("min", 0.0))
    upper = float(window.get("max", 1.0))
    return clamp01((float(value) - lower) / max(upper - lower, 1.0e-9))


def pulse_packet_dev(metric: dict[str, Any], deltas: dict[str, float]) -> float:
    center = float(metric.get("center_value", 0.0))
    total = center
    jacobian = dict(metric.get("jacobian", {}) or {})
    hessian_diag = dict(metric.get("hessian_diag", {}) or {})
    hessian_cross = dict(metric.get("hessian_cross", {}) or {})
    for axis, delta in deltas.items():
        total += float(jacobian.get(axis, 0.0)) * float(delta)
    for axis, delta in deltas.items():
        total += 0.5 * float(hessian_diag.get(axis, 0.0)) * float(delta) * float(delta)
    for pair, coeff in hessian_cross.items():
        if len(str(pair)) != 2:
            continue
        lhs = str(pair)[0]
        rhs = str(pair)[1]
        total += float(coeff) * float(deltas.get(lhs, 0.0)) * float(deltas.get(rhs, 0.0))
    return total


def clamp_band(
    amplitude_norm: float,
    amplitude_cap: float,
    trap_ratio: float,
    violation: float,
) -> float:
    overdrive = max(0.0, float(amplitude_norm) - float(amplitude_cap))
    risk = max(float(trap_ratio), float(violation), overdrive)
    return clamp01(risk)


def bits_to_target_hex(nbits_hex: str) -> str:
    try:
        nbits = int(str(nbits_hex), 16)
        exponent = (nbits >> 24) & 0xFF
        mantissa = nbits & 0xFFFFFF
        target = mantissa * (1 << (8 * max(exponent - 3, 0)))
        return f"{target:064x}"
    except Exception:
        return "00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"


def pow_hex_leq_target(pow_hex: str, target_hex: str) -> bool:
    try:
        pow_value = int(str(pow_hex or "").replace("0x", "").strip() or "0", 16)
        target_value = int(str(target_hex or "").replace("0x", "").strip() or "0", 16)
        return pow_value <= target_value
    except Exception:
        return False


def normalize_hex_64(value: str) -> str:
    normalized = str(value or "").replace("0x", "").strip().lower()
    filtered = "".join(ch for ch in normalized if ch in "0123456789abcdef")
    if not filtered:
        filtered = "0"
    return filtered.rjust(64, "0")[-64:]


def count_leading_zero_nibbles(value: str) -> int:
    normalized = normalize_hex_64(value)
    count = 0
    for ch in normalized:
        if ch != "0":
            break
        count += 1
    return count


def digest_entropy_norm(blob: bytes) -> float:
    payload = bytes(blob or b"")
    if not payload:
        return 0.0
    counts = [0] * 256
    for value in payload:
        counts[int(value)] += 1
    total = float(len(payload))
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = float(count) / total
        entropy -= probability * math.log2(max(probability, 1.0e-12))
    max_entropy = min(8.0, math.log2(total) if total > 1.0 else 1.0)
    return clamp01(entropy / max(max_entropy, 1.0e-9))


def digest_transition_ratio(blob: bytes) -> float:
    payload = bytes(blob or b"")
    if len(payload) <= 1:
        return 0.0
    transitions = 0
    previous = payload[0]
    for value in payload[1:]:
        if value != previous:
            transitions += 1
        previous = value
    return float(transitions) / float(len(payload) - 1)


def digest_bit_balance(blob: bytes) -> float:
    payload = bytes(blob or b"")
    if not payload:
        return 0.0
    one_bits = 0
    for value in payload:
        one_bits += int(int(value).bit_count())
    total_bits = float(len(payload) * 8)
    ratio = float(one_bits) / max(total_bits, 1.0)
    return clamp01(1.0 - abs(ratio - 0.5) / 0.5)


def digest_lane_balance(blob: bytes, lane_count: int = 4) -> float:
    payload = bytes(blob or b"")
    if not payload:
        return 0.0
    lanes = max(1, int(lane_count))
    padded = payload.ljust(((len(payload) + lanes - 1) // lanes) * lanes, b"\x00")
    lane_size = len(padded) // lanes
    lane_means: list[float] = []
    for lane_index in range(lanes):
        lane = padded[lane_index * lane_size : (lane_index + 1) * lane_size]
        lane_means.append(float(sum(lane)) / max(float(len(lane) * 255), 1.0))
    variance = float(np.var(np.array(lane_means, dtype=np.float64)))
    return clamp01(1.0 - variance / 0.08)


def sha256d_function_metrics(first_digest: bytes, final_digest: bytes) -> dict[str, Any]:
    first_blob = bytes(first_digest or b"")
    final_blob = bytes(final_digest or b"")
    xor_blob = bytes(lhs ^ rhs for lhs, rhs in zip(first_blob, final_blob))
    first_entropy = digest_entropy_norm(first_blob)
    final_entropy = digest_entropy_norm(final_blob)
    transition_ratio = clamp01(
        0.40 * digest_transition_ratio(final_blob)
        + 0.34 * digest_transition_ratio(first_blob)
        + 0.26 * digest_transition_ratio(xor_blob)
    )
    bit_balance = clamp01(
        0.42 * digest_bit_balance(final_blob)
        + 0.32 * digest_bit_balance(first_blob)
        + 0.26 * digest_bit_balance(xor_blob)
    )
    lane_balance = clamp01(
        0.58 * digest_lane_balance(final_blob)
        + 0.42 * digest_lane_balance(xor_blob)
    )
    final_hex_metrics = hex_symbol_metrics(final_blob.hex(), width=64)
    xor_hex_metrics = hex_symbol_metrics(xor_blob.hex(), width=64)
    round_coupling = clamp01(
        0.32 * float(xor_hex_metrics.get("integrity_score", 0.0))
        + 0.24 * float(xor_hex_metrics.get("symbol_entropy_norm", 0.0))
        + 0.16 * float(final_hex_metrics.get("transition_ratio", 0.0))
        + 0.14 * bit_balance
        + 0.14 * lane_balance
    )
    function_score = clamp01(
        0.24 * final_entropy
        + 0.18 * first_entropy
        + 0.16 * transition_ratio
        + 0.14 * bit_balance
        + 0.12 * lane_balance
        + 0.10 * float(final_hex_metrics.get("integrity_score", 0.0))
        + 0.06 * round_coupling
    )
    return {
        "first_entropy": float(first_entropy),
        "final_entropy": float(final_entropy),
        "transition_ratio": float(transition_ratio),
        "bit_balance": float(bit_balance),
        "lane_balance": float(lane_balance),
        "round_coupling": float(round_coupling),
        "function_score": float(function_score),
    }


def pow_distance_metrics(pow_hex: str, target_hex: str) -> dict[str, Any]:
    normalized_pow = normalize_hex_64(pow_hex)
    normalized_target = normalize_hex_64(target_hex)
    prefix_match = 0
    for lhs, rhs in zip(normalized_pow, normalized_target):
        if lhs != rhs:
            break
        prefix_match += 1
    pow_zero_nibbles = count_leading_zero_nibbles(normalized_pow)
    target_zero_nibbles = count_leading_zero_nibbles(normalized_target)
    try:
        pow_value = int(normalized_pow, 16)
        target_value = int(normalized_target, 16)
    except Exception:
        pow_value = 0
        target_value = 1
    valid = bool(pow_value <= target_value)
    if valid:
        over_target_ratio = 0.0
        distance_score = 1.0
    else:
        over_target_ratio = float(max(pow_value - target_value, 0)) / max(float(target_value), 1.0)
        distance_score = 1.0 / (1.0 + math.log10(1.0 + over_target_ratio))
    zero_alignment = clamp01(
        float(pow_zero_nibbles + prefix_match * 0.25) / max(float(target_zero_nibbles + 1), 1.0)
    )
    nibble_span = max(target_zero_nibbles + 8, 1)
    normalized_prefix_match = clamp01(float(prefix_match) / float(nibble_span))
    return {
        "valid": valid,
        "distance_score": float(distance_score),
        "zero_alignment": float(zero_alignment),
        "prefix_match": int(prefix_match),
        "normalized_prefix_match": float(normalized_prefix_match),
        "pow_zero_nibbles": int(pow_zero_nibbles),
        "target_zero_nibbles": int(target_zero_nibbles),
        "over_target_ratio": float(over_target_ratio),
    }


def hash_target_phase_metrics(pow_hex: str, target_hex: str) -> dict[str, Any]:
    normalized_pow = normalize_hex_64(pow_hex)
    normalized_target = normalize_hex_64(target_hex)
    pow_values = [int(ch, 16) for ch in normalized_pow]
    target_values = [int(ch, 16) for ch in normalized_target]
    prefix_match = 0
    for pow_nibble, target_nibble in zip(pow_values, target_values):
        if pow_nibble != target_nibble:
            break
        prefix_match += 1
    first_divergence = prefix_match if prefix_match < len(pow_values) else len(pow_values) - 1
    frontier_pow = pow_values[first_divergence] if pow_values else 0
    frontier_target = target_values[first_divergence] if target_values else 0
    frontier_delta = float(frontier_target - frontier_pow)
    frontier_slack = clamp01((frontier_delta + 15.0) / 30.0)
    frontier_cost = clamp01(max(float(frontier_pow - frontier_target), 0.0) / 15.0)
    under_window_values: list[float] = []
    band_alignment_values: list[float] = []
    flux_alignment_values: list[float] = []
    for index, (pow_nibble, target_nibble) in enumerate(zip(pow_values, target_values)):
        nibble_gap = abs(float(pow_nibble - target_nibble)) / 15.0
        over_target = max(float(pow_nibble - target_nibble), 0.0) / 15.0
        under_window_values.append(clamp01(1.0 - over_target))
        band_alignment_values.append(clamp01(1.0 - nibble_gap))
        if index > 0:
            pow_step = abs(float(pow_values[index] - pow_values[index - 1])) / 15.0
            target_step = abs(float(target_values[index] - target_values[index - 1])) / 15.0
            flux_alignment_values.append(clamp01(1.0 - abs(pow_step - target_step)))
    window_coverage = clamp01(
        float(np.mean(np.array(under_window_values, dtype=np.float64))) if under_window_values else 0.0
    )
    band_alignment = clamp01(
        float(np.mean(np.array(band_alignment_values, dtype=np.float64))) if band_alignment_values else 0.0
    )
    flux_alignment = clamp01(
        float(np.mean(np.array(flux_alignment_values, dtype=np.float64))) if flux_alignment_values else 0.0
    )
    prefix_lock = clamp01(float(prefix_match) / 64.0)
    phase_cost = clamp01(
        0.46 * frontier_cost
        + 0.24 * (1.0 - window_coverage)
        + 0.18 * (1.0 - band_alignment)
        + 0.12 * (1.0 - flux_alignment)
    )
    phase_pressure = clamp01(
        0.30 * window_coverage
        + 0.22 * band_alignment
        + 0.18 * flux_alignment
        + 0.16 * prefix_lock
        + 0.14 * frontier_slack
    )
    stable_pressure = clamp01(
        0.54 * phase_pressure
        + 0.24 * (1.0 - phase_cost)
        + 0.12 * frontier_slack
        + 0.10 * float(1.0 if pow_hex_leq_target(normalized_pow, normalized_target) else 0.0)
    )
    return {
        "hash_target_phase_pressure": float(phase_pressure),
        "hash_target_phase_cost": float(phase_cost),
        "hash_target_prefix_lock": float(prefix_lock),
        "hash_target_window_coverage": float(window_coverage),
        "hash_target_band_alignment": float(band_alignment),
        "hash_target_flux_alignment": float(flux_alignment),
        "hash_target_frontier_slack": float(frontier_slack),
        "hash_target_frontier_cost": float(frontier_cost),
        "hash_target_stable_pressure": float(stable_pressure),
    }


def rotate_left_32(value: int, shift: int) -> int:
    word = int(value) & 0xFFFFFFFF
    turns = int(shift) & 31
    if turns <= 0:
        return word
    return ((word << turns) | (word >> (32 - turns))) & 0xFFFFFFFF


def hex_symbol_metrics(value: str, width: int = 0) -> dict[str, Any]:
    normalized = str(value or "").replace("0x", "").strip().lower()
    filtered = "".join(ch for ch in normalized if ch in "0123456789abcdef")
    if width > 0:
        filtered = filtered.rjust(width, "0")[-width:]
    length = len(filtered)
    if length <= 0:
        return {
            "normalized": "",
            "unique_symbol_count": 0,
            "unique_symbol_ratio": 0.0,
            "dominant_symbol_ratio": 1.0,
            "longest_run": 0,
            "longest_run_ratio": 1.0,
            "transition_count": 0,
            "transition_ratio": 0.0,
            "symbol_entropy": 0.0,
            "symbol_entropy_norm": 0.0,
            "integrity_score": 0.0,
        }
    counts: dict[str, int] = {}
    longest_run = 1
    current_run = 1
    transition_count = 0
    previous = filtered[0]
    for index, ch in enumerate(filtered):
        counts[ch] = int(counts.get(ch, 0)) + 1
        if index <= 0:
            continue
        if ch != previous:
            transition_count += 1
            current_run = 1
        else:
            current_run += 1
            longest_run = max(longest_run, current_run)
        previous = ch
    probabilities = [float(count) / float(length) for count in counts.values() if count > 0]
    symbol_entropy = 0.0
    for probability in probabilities:
        symbol_entropy -= probability * math.log2(max(probability, 1.0e-12))
    max_entropy = math.log2(float(min(16, length))) if length > 1 else 1.0
    unique_symbol_count = len(counts)
    unique_symbol_ratio = float(unique_symbol_count) / float(max(min(16, length), 1))
    dominant_symbol_ratio = float(max(counts.values())) / float(length)
    longest_run_ratio = float(longest_run) / float(length)
    transition_ratio = float(transition_count) / float(max(length - 1, 1))
    symbol_entropy_norm = clamp01(symbol_entropy / max(max_entropy, 1.0e-9))
    integrity_score = clamp01(
        0.30 * symbol_entropy_norm
        + 0.24 * transition_ratio
        + 0.18 * unique_symbol_ratio
        + 0.16 * (1.0 - longest_run_ratio)
        + 0.12 * (1.0 - dominant_symbol_ratio)
    )
    return {
        "normalized": filtered,
        "unique_symbol_count": int(unique_symbol_count),
        "unique_symbol_ratio": float(unique_symbol_ratio),
        "dominant_symbol_ratio": float(dominant_symbol_ratio),
        "longest_run": int(longest_run),
        "longest_run_ratio": float(longest_run_ratio),
        "transition_count": int(transition_count),
        "transition_ratio": float(transition_ratio),
        "symbol_entropy": float(symbol_entropy),
        "symbol_entropy_norm": float(symbol_entropy_norm),
        "integrity_score": float(integrity_score),
    }


def canonical_utf8_byte_turn(byte_value: int) -> float:
    return wrap_turns(float(int(byte_value) & 0xFF) / 256.0)


def utf8_phase_ring_metrics(
    byte_values: bytes,
    byte_turns: list[float] | None = None,
    phase_deltas: list[float] | None = None,
) -> dict[str, Any]:
    payload = bytes(byte_values or b"")
    if not payload:
        return {
            "byte_hex": "",
            "utf8_preview": "",
            "unique_symbol_count": 0,
            "unique_symbol_ratio": 0.0,
            "dominant_symbol_ratio": 1.0,
            "longest_run_ratio": 1.0,
            "byte_transition_ratio": 0.0,
            "byte_entropy_norm": 0.0,
            "phase_flux_ratio": 0.0,
            "phase_delta_alignment": 0.0,
            "phase_delta_balance": 0.0,
            "phase_integrity_score": 0.0,
        }
    turns = [canonical_utf8_byte_turn(value) for value in payload]
    if byte_turns and len(byte_turns) == len(payload):
        turns = [wrap_turns(float(value)) for value in byte_turns]
    deltas = [signed_turn_delta(turns[idx], turns[idx - 1]) for idx in range(1, len(turns))]
    if phase_deltas and len(phase_deltas) > 0:
        deltas = [float(value) for value in phase_deltas]
    counts: dict[int, int] = {}
    longest_run = 1
    transition_count = 0
    run_length = 1
    for index, value in enumerate(payload):
        counts[int(value)] = counts.get(int(value), 0) + 1
        if index > 0:
            if payload[index - 1] != value:
                transition_count += 1
                run_length = 1
            else:
                run_length += 1
                if run_length > longest_run:
                    longest_run = run_length
    max_entropy = math.log2(float(min(256, len(payload)))) if len(payload) > 1 else 1.0
    byte_entropy = 0.0
    for count in counts.values():
        probability = float(count) / float(len(payload))
        byte_entropy -= probability * math.log2(max(probability, 1.0e-12))
    unique_symbol_count = len(counts)
    unique_symbol_ratio = float(unique_symbol_count) / float(max(min(256, len(payload)), 1))
    dominant_symbol_ratio = float(max(counts.values())) / float(len(payload))
    longest_run_ratio = float(longest_run) / float(len(payload))
    byte_transition_ratio = float(transition_count) / float(max(len(payload) - 1, 1))
    byte_entropy_norm = clamp01(byte_entropy / max(max_entropy, 1.0e-9))
    phase_flux_ratio = clamp01(
        float(np.mean(np.array([clamp01(abs(delta) / 0.5) for delta in deltas], dtype=np.float64)))
        if deltas
        else 0.0
    )
    phase_delta_alignment = clamp01(
        float(np.mean(np.array([clamp01(1.0 - abs(delta) / 0.5) for delta in deltas], dtype=np.float64)))
        if deltas
        else 0.0
    )
    phase_delta_balance = clamp01(
        1.0
        - float(
            np.var(
                np.array([clamp01(abs(delta) / 0.5) for delta in deltas], dtype=np.float64)
            )
        )
        / 0.08
        if deltas
        else 1.0
    )
    phase_integrity_score = clamp01(
        0.24 * byte_entropy_norm
        + 0.18 * byte_transition_ratio
        + 0.12 * unique_symbol_ratio
        + 0.10 * (1.0 - longest_run_ratio)
        + 0.12 * phase_flux_ratio
        + 0.14 * phase_delta_alignment
        + 0.10 * phase_delta_balance
    )
    utf8_preview = payload.decode("utf-8", errors="replace").encode("unicode_escape").decode("ascii")
    return {
        "byte_hex": payload.hex(),
        "utf8_preview": utf8_preview,
        "unique_symbol_count": int(unique_symbol_count),
        "unique_symbol_ratio": float(unique_symbol_ratio),
        "dominant_symbol_ratio": float(dominant_symbol_ratio),
        "longest_run_ratio": float(longest_run_ratio),
        "byte_transition_ratio": float(byte_transition_ratio),
        "byte_entropy_norm": float(byte_entropy_norm),
        "phase_flux_ratio": float(phase_flux_ratio),
        "phase_delta_alignment": float(phase_delta_alignment),
        "phase_delta_balance": float(phase_delta_balance),
        "phase_integrity_score": float(phase_integrity_score),
    }


def target_prefix_phase_metrics(
    target_hex: str,
    byte_values: bytes,
    phase_ring_turns: list[float] | None = None,
) -> dict[str, Any]:
    payload = bytes(byte_values or b"")
    if not payload:
        return {
            "target_prefix_lock": 0.0,
            "target_prefix_phase_alignment": 0.0,
            "target_prefix_byte_match": 0.0,
            "target_prefix_flux_alignment": 0.0,
            "target_prefix_turns": [],
        }
    target_bytes_full = bytes.fromhex(normalize_hex_64(target_hex))
    target_prefix = target_bytes_full[: len(payload)] if target_bytes_full else b""
    if len(target_prefix) < len(payload):
        target_prefix = target_prefix.ljust(len(payload), b"\x00")
    target_turns = [canonical_utf8_byte_turn(value) for value in target_prefix]
    observed_turns = [canonical_utf8_byte_turn(value) for value in payload]
    if phase_ring_turns and len(phase_ring_turns) == len(payload):
        observed_turns = [wrap_turns(float(value)) for value in phase_ring_turns]
    weights = [float(len(payload) - index) for index in range(len(payload))]
    weight_total = max(sum(weights), 1.0)
    alignment_sum = 0.0
    byte_match_sum = 0.0
    flux_sum = 0.0
    for index, (observed_turn, target_turn) in enumerate(zip(observed_turns, target_turns)):
        weight = weights[index] / weight_total
        alignment_sum += weight * turn_alignment(observed_turn, target_turn)
        byte_match_sum += weight * clamp01(
            1.0 - abs(int(payload[index]) - int(target_prefix[index])) / 255.0
        )
        if index > 0:
            observed_delta = signed_turn_delta(observed_turn, observed_turns[index - 1])
            target_delta = signed_turn_delta(target_turn, target_turns[index - 1])
            flux_sum += weight * clamp01(1.0 - abs(observed_delta - target_delta) / 0.5)
    flux_alignment = clamp01(flux_sum / max(1.0 - (weights[0] / weight_total), 1.0e-9)) if len(payload) > 1 else 1.0
    phase_alignment = clamp01(alignment_sum)
    byte_match = clamp01(byte_match_sum)
    prefix_lock = clamp01(
        0.48 * phase_alignment
        + 0.30 * byte_match
        + 0.22 * flux_alignment
    )
    return {
        "target_prefix_lock": float(prefix_lock),
        "target_prefix_phase_alignment": float(phase_alignment),
        "target_prefix_byte_match": float(byte_match),
        "target_prefix_flux_alignment": float(flux_alignment),
        "target_prefix_turns": [float(value) for value in target_turns],
    }


def build_prefix_trajectory_vector(
    target_hex: str,
    target_prefix_turns: list[float] | None = None,
    observed_turns: list[float] | None = None,
    target_phase_pressure: float = 0.0,
    target_phase_alignment: float = 0.0,
    target_flux_alignment: float = 0.0,
    electron_ring_stability: float = 0.0,
    electron_phase_anchor: float = 0.0,
) -> dict[str, Any]:
    prefix_turns = [float(wrap_turns(value)) for value in list(target_prefix_turns or [])]
    if not prefix_turns:
        prefix_turns = [
            float(value)
            for value in target_prefix_phase_metrics(target_hex, b"\x00\x00\x00\x00").get(
                "target_prefix_turns", []
            )
        ]
    if not prefix_turns:
        prefix_turns = [0.0, 0.0, 0.0, 0.0]
    while len(prefix_turns) < 4:
        prefix_turns.append(float(prefix_turns[-1] if prefix_turns else 0.0))
    observed_source = [float(wrap_turns(value)) for value in list(observed_turns or [])]
    if not observed_source:
        observed_source = [float(value) for value in prefix_turns[:4]]
    while len(observed_source) < 4:
        observed_source.append(float(observed_source[-1] if observed_source else 0.0))
    target_phase_pressure = clamp01(float(target_phase_pressure))
    target_phase_alignment = clamp01(float(target_phase_alignment))
    target_flux_alignment = clamp01(float(target_flux_alignment))
    electron_ring_stability = clamp01(float(electron_ring_stability))
    electron_phase_anchor = wrap_turns(float(electron_phase_anchor))
    encoded_vector: list[float] = []
    vector_alignments: list[float] = []
    vector_flux_alignments: list[float] = []
    for index in range(4):
        target_turn = wrap_turns(float(prefix_turns[index]))
        observed_turn = wrap_turns(float(observed_source[index]))
        stable_turn = wrap_turns(
            0.44 * target_turn
            + 0.14 * target_phase_alignment
            + 0.12 * target_phase_pressure
            + 0.10 * target_flux_alignment
            + 0.10 * electron_ring_stability
            + 0.06 * electron_phase_anchor
            + 0.04 * float(index + 1) / 4.0
        )
        pull_turn = signed_turn_delta(target_turn, observed_turn)
        encoded_turn = wrap_turns(
            stable_turn
            + pull_turn * (0.18 + 0.34 * target_phase_pressure + 0.12 * target_phase_alignment)
            + signed_turn_delta(electron_phase_anchor, observed_turn) * 0.06
        )
        encoded_vector.append(float(encoded_turn))
        vector_alignments.append(float(turn_alignment(encoded_turn, target_turn)))
        if index > 0:
            encoded_delta = signed_turn_delta(encoded_turn, encoded_vector[index - 1])
            target_delta = signed_turn_delta(target_turn, prefix_turns[index - 1])
            vector_flux_alignments.append(
                float(clamp01(1.0 - abs(encoded_delta - target_delta) / 0.5))
            )
    vector_alignment = clamp01(
        float(np.mean(np.array(vector_alignments, dtype=np.float64))) if vector_alignments else 0.0
    )
    vector_flux_alignment = clamp01(
        float(np.mean(np.array(vector_flux_alignments, dtype=np.float64)))
        if vector_flux_alignments
        else vector_alignment
    )
    vector_phase_pressure = clamp01(
        0.42 * vector_alignment
        + 0.24 * vector_flux_alignment
        + 0.14 * target_phase_pressure
        + 0.10 * target_phase_alignment
        + 0.10 * electron_ring_stability
    )
    return {
        "prefix_trajectory_vector": [float(value) for value in encoded_vector[:4]],
        "prefix_trajectory_vector_alignment": float(vector_alignment),
        "prefix_trajectory_flux_alignment": float(vector_flux_alignment),
        "prefix_trajectory_phase_pressure": float(vector_phase_pressure),
        "prefix_target_turns": [float(value) for value in prefix_turns[:4]],
    }


def build_coherent_noise_field(
    pre_feedback: dict[str, Any] | None,
    post_feedback: dict[str, Any] | None,
    phase_retention: float,
    response_gate: float,
    observation_freshness_gate: float,
) -> dict[str, Any]:
    pre_feedback = dict(pre_feedback or {})
    post_feedback = dict(post_feedback or {})
    pre_axis_vector = np.array(
        list(pre_feedback.get("feedback_axis_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    post_axis_vector = np.array(
        list(post_feedback.get("feedback_axis_vector", []) or pre_axis_vector.tolist()),
        dtype=np.float64,
    )
    if pre_axis_vector.shape[0] != 4:
        pre_axis_vector = np.zeros(4, dtype=np.float64)
    if post_axis_vector.shape[0] != 4:
        post_axis_vector = np.array(pre_axis_vector, dtype=np.float64)
    pre_dof_vector = np.array(
        list(pre_feedback.get("feedback_dof_vector", []) or [0.0] * len(GPU_PULSE_DOF_LABELS)),
        dtype=np.float64,
    )
    post_dof_vector = np.array(
        list(post_feedback.get("feedback_dof_vector", []) or pre_dof_vector.tolist()),
        dtype=np.float64,
    )
    if pre_dof_vector.shape[0] != len(GPU_PULSE_DOF_LABELS):
        pre_dof_vector = np.zeros(len(GPU_PULSE_DOF_LABELS), dtype=np.float64)
    if post_dof_vector.shape[0] != len(GPU_PULSE_DOF_LABELS):
        post_dof_vector = np.array(pre_dof_vector, dtype=np.float64)
    phase_retention = clamp01(float(phase_retention))
    response_gate = clamp01(float(response_gate))
    observation_freshness_gate = clamp01(float(observation_freshness_gate))
    post_environment_pressure = clamp01(float(post_feedback.get("environment_pressure", 0.0)))
    post_environment_stability = clamp01(float(post_feedback.get("environment_stability", 0.0)))
    post_temperature_velocity = clamp01(float(post_feedback.get("temperature_velocity_norm", 0.0)))
    post_latency_jitter = clamp01(float(post_feedback.get("latency_jitter_norm", 0.0)))
    axis_delta = np.abs(post_axis_vector - pre_axis_vector)
    dof_delta = np.abs(post_dof_vector - pre_dof_vector)
    turbulence_axis_vector = np.clip(
        0.40 * axis_delta
        + 0.18 * np.abs(post_axis_vector)
        + 0.12 * post_environment_pressure
        + 0.12 * (1.0 - post_environment_stability)
        + 0.10 * post_temperature_velocity
        + 0.08 * post_latency_jitter
        + 0.08 * (1.0 - phase_retention),
        0.0,
        1.0,
    )
    turbulence_dof_seed = np.array(
        [
            float(turbulence_axis_vector[0]),
            float(turbulence_axis_vector[1]),
            float(turbulence_axis_vector[3]),
            float(turbulence_axis_vector[2]),
            clamp01(math.sqrt(max(float(turbulence_axis_vector[0]) * float(turbulence_axis_vector[1]), 0.0))),
            clamp01(math.sqrt(max(float(turbulence_axis_vector[0]) * float(turbulence_axis_vector[3]), 0.0))),
            clamp01(math.sqrt(max(float(turbulence_axis_vector[0]) * float(turbulence_axis_vector[2]), 0.0))),
            clamp01(math.sqrt(max(float(turbulence_axis_vector[1]) * float(turbulence_axis_vector[3]), 0.0))),
            clamp01(math.sqrt(max(float(turbulence_axis_vector[1]) * float(turbulence_axis_vector[2]), 0.0))),
            clamp01(math.sqrt(max(float(turbulence_axis_vector[3]) * float(turbulence_axis_vector[2]), 0.0))),
        ],
        dtype=np.float64,
    )
    coherent_noise_dof_vector = np.clip(
        0.46 * dof_delta
        + 0.24 * np.abs(post_dof_vector)
        + 0.18 * turbulence_dof_seed
        + 0.06 * post_environment_pressure
        + 0.06 * (1.0 - post_environment_stability),
        0.0,
        1.0,
    )
    coherent_noise_tensor = np.eye(len(GPU_PULSE_DOF_LABELS), dtype=np.float64) * 0.34
    coherent_noise_tensor += np.diag(coherent_noise_dof_vector * 0.52)
    pair_to_primary = {
        4: (0, 1),
        5: (0, 2),
        6: (0, 3),
        7: (1, 2),
        8: (1, 3),
        9: (2, 3),
    }
    for pair_idx, (lhs_idx, rhs_idx) in pair_to_primary.items():
        coupling_value = float(coherent_noise_dof_vector[pair_idx])
        edge_value = 0.04 + 0.22 * coupling_value
        coherent_noise_tensor[pair_idx, lhs_idx] += edge_value
        coherent_noise_tensor[lhs_idx, pair_idx] += edge_value
        coherent_noise_tensor[pair_idx, rhs_idx] += edge_value
        coherent_noise_tensor[rhs_idx, pair_idx] += edge_value
        coherent_noise_tensor[lhs_idx, rhs_idx] += 0.03 + 0.12 * coupling_value
        coherent_noise_tensor[rhs_idx, lhs_idx] += 0.03 + 0.12 * coupling_value
    coherent_noise_tensor += np.outer(coherent_noise_dof_vector, coherent_noise_dof_vector) * 0.05
    coherent_noise_tensor = np.clip(coherent_noise_tensor, 0.0, 1.45)
    spectral_probe = np.abs(np.fft.rfft(coherent_noise_dof_vector, n=8))
    if spectral_probe.shape[0] < 4:
        spectral_probe = np.pad(spectral_probe, (0, 4 - spectral_probe.shape[0]))
    spectral_norm = spectral_probe[:4]
    if float(np.max(spectral_norm)) > 0.0:
        spectral_norm = spectral_norm / float(np.max(spectral_norm))
    coherent_noise_axis_vector = np.clip(
        0.48 * turbulence_axis_vector
        + 0.20 * np.abs(post_axis_vector)
        + 0.16 * spectral_norm[:4]
        + 0.08 * post_environment_pressure
        + 0.08 * (1.0 - post_environment_stability),
        0.0,
        1.0,
    )
    noise_resonance_nodes = np.clip(
        0.34 * coherent_noise_axis_vector
        + 0.20 * spectral_norm[:4]
        + 0.16 * phase_retention
        + 0.14 * response_gate
        + 0.08 * observation_freshness_gate
        + 0.08 * (1.0 - post_environment_pressure),
        0.0,
        1.0,
    )
    drift_compensation_vector = np.clip(
        0.34 * (1.0 - coherent_noise_axis_vector)
        + 0.22 * noise_resonance_nodes
        + 0.16 * np.abs(post_axis_vector)
        + 0.12 * phase_retention
        + 0.10 * response_gate
        + 0.06 * observation_freshness_gate,
        0.0,
        1.0,
    )
    relative_spatial_field = clamp_vector_norm(
        np.array(
            [
                0.36 * float(post_axis_vector[0])
                + 0.20 * float(drift_compensation_vector[0])
                + 0.18 * float(noise_resonance_nodes[0]),
                0.34 * float(post_axis_vector[1])
                + 0.22 * float(drift_compensation_vector[1])
                + 0.18 * float(noise_resonance_nodes[1]),
                0.32 * float(post_axis_vector[2])
                + 0.20 * float(drift_compensation_vector[2])
                + 0.18 * float(noise_resonance_nodes[2]),
                0.28 * float(post_axis_vector[3])
                + 0.20 * float(drift_compensation_vector[3])
                + 0.16 * float(noise_resonance_nodes[3])
                + 0.12 * phase_retention,
            ],
            dtype=np.float64,
        ),
        max_norm=2.4,
    )
    noise_resonance_gate = clamp01(
        0.32 * float(np.mean(noise_resonance_nodes))
        + 0.18 * response_gate
        + 0.16 * phase_retention
        + 0.14 * observation_freshness_gate
        + 0.10 * (1.0 - post_environment_pressure)
        + 0.10 * post_environment_stability
    )
    noise_orbital_anchor_turns = wrap_turns(
        0.28 * float(post_feedback.get("phase_anchor_turns", 0.0))
        + 0.18 * float(np.mean(noise_resonance_nodes))
        + 0.16 * float(np.mean(drift_compensation_vector))
        + 0.14 * phase_retention
        + 0.12 * response_gate
        + 0.12 * observation_freshness_gate
    )
    return {
        "coherent_noise_axis_vector": [float(value) for value in coherent_noise_axis_vector],
        "coherent_noise_dof_vector": [float(value) for value in coherent_noise_dof_vector],
        "coherent_noise_tensor": coherent_noise_tensor.tolist(),
        "noise_resonance_nodes": [float(value) for value in noise_resonance_nodes],
        "drift_compensation_vector": [float(value) for value in drift_compensation_vector],
        "relative_spatial_field": [float(value) for value in relative_spatial_field],
        "noise_resonance_gate": float(noise_resonance_gate),
        "noise_orbital_anchor_turns": float(noise_orbital_anchor_turns),
        "environment_turbulence": float(np.mean(turbulence_axis_vector)),
    }


def build_phase_orbital_trace(
    phase_ring_turns: list[float] | None,
    phase_ring_deltas: list[float] | None,
    carrier_turns: list[float] | None,
    prefix_trajectory_vector: list[float] | None,
    noise_resonance_nodes: list[float] | None,
    drift_compensation_vector: list[float] | None,
    relative_spatial_field: list[float] | None,
    projected_temporal_dof_vector: list[float] | None,
    utf8_blob: bytes | None = None,
) -> dict[str, Any]:
    phase_turns = [float(wrap_turns(value)) for value in list(phase_ring_turns or [])]
    phase_deltas = [float(value) for value in list(phase_ring_deltas or [])]
    carrier_values = [float(wrap_turns(value)) for value in list(carrier_turns or [])]
    prefix_vector = [float(wrap_turns(value)) for value in list(prefix_trajectory_vector or [])]
    noise_nodes = [float(clamp01(value)) for value in list(noise_resonance_nodes or [])]
    drift_vector = [float(clamp01(value)) for value in list(drift_compensation_vector or [])]
    relative_field = [float(value) for value in list(relative_spatial_field or [])]
    temporal_vector = [float(value) for value in list(projected_temporal_dof_vector or [])]
    while len(prefix_vector) < max(4, len(phase_turns)):
        prefix_vector.append(float(prefix_vector[-1] if prefix_vector else 0.0))
    while len(noise_nodes) < max(4, len(phase_turns)):
        noise_nodes.append(float(noise_nodes[-1] if noise_nodes else 0.0))
    while len(drift_vector) < 4:
        drift_vector.append(float(drift_vector[-1] if drift_vector else 0.0))
    while len(relative_field) < 4:
        relative_field.append(float(relative_field[-1] if relative_field else 0.0))
    while len(temporal_vector) < 4:
        temporal_vector.append(float(temporal_vector[-1] if temporal_vector else 0.0))
    orbital_vectors: list[list[float]] = []
    orbital_alignments: list[float] = []
    orbital_resonances: list[float] = []
    orbital_stabilities: list[float] = []
    payload = bytes(utf8_blob or b"")
    for index, phase_turn in enumerate(phase_turns):
        phase_delta = float(phase_deltas[index]) if index < len(phase_deltas) else 0.0
        carrier_turn = float(carrier_values[index]) if index < len(carrier_values) else phase_turn
        prefix_turn = float(prefix_vector[index]) if index < len(prefix_vector) else phase_turn
        noise_node = float(noise_nodes[index]) if index < len(noise_nodes) else 0.0
        utf8_norm = float(payload[index]) / 255.0 if index < len(payload) else float(index + 1) / 4.0
        orbital_radius = clamp01(
            0.28
            + 0.22 * turn_alignment(phase_turn, prefix_turn)
            + 0.18 * (1.0 - min(abs(phase_delta) * 2.0, 1.0))
            + 0.12 * noise_node
            + 0.10 * drift_vector[index % 4]
            + 0.10 * utf8_norm
        )
        orbital_angle = float(phase_turn) * math.tau + noise_node * math.pi
        x_val = (
            math.cos(orbital_angle) * orbital_radius
            + signed_turn_delta(prefix_turn, carrier_turn) * 0.22
            + relative_field[0] * 0.18
            + temporal_vector[0] * 0.10
        )
        y_val = (
            math.sin(orbital_angle) * orbital_radius
            + signed_turn_delta(phase_turn, prefix_turn) * 0.22
            + relative_field[1] * 0.18
            + temporal_vector[1] * 0.10
        )
        z_val = (
            math.sin((carrier_turn + phase_turn + noise_node) * math.tau) * (0.24 + 0.30 * orbital_radius)
            + relative_field[2] * 0.20
            + temporal_vector[2] * 0.12
            + (drift_vector[2] - 0.5) * 0.18
        )
        t_val = clamp01(
            0.36 * orbital_radius
            + 0.18 * noise_node
            + 0.16 * drift_vector[3]
            + 0.14 * relative_field[3]
            + 0.10 * temporal_vector[3]
            + 0.06 * utf8_norm
        )
        orbital_vectors.append([float(x_val), float(y_val), float(z_val), float(t_val)])
        target_orbit = np.array(
            [
                math.cos(float(prefix_turn) * math.tau) * orbital_radius,
                math.sin(float(prefix_turn) * math.tau) * orbital_radius,
                relative_field[2] * 0.18 + temporal_vector[2] * 0.10,
                t_val,
            ],
            dtype=np.float64,
        )
        current_orbit = np.array([x_val, y_val, z_val, t_val], dtype=np.float64)
        orbital_alignments.append(float(vector_similarity(current_orbit, target_orbit)))
        orbital_resonances.append(
            float(
                clamp01(
                    0.42 * noise_node
                    + 0.22 * turn_alignment(phase_turn, prefix_turn)
                    + 0.18 * (1.0 - min(abs(phase_delta) * 2.0, 1.0))
                    + 0.18 * utf8_norm
                )
            )
        )
        orbital_stabilities.append(
            float(
                clamp01(
                    0.44 * (1.0 - min(abs(phase_delta) * 2.0, 1.0))
                    + 0.20 * turn_alignment(carrier_turn, phase_turn)
                    + 0.18 * (1.0 - min(abs(z_val) / 2.0, 1.0))
                    + 0.18 * drift_vector[index % 4]
                )
            )
        )
    orbital_trace_vector = np.mean(np.array(orbital_vectors, dtype=np.float64), axis=0) if orbital_vectors else np.zeros(4, dtype=np.float64)
    orbital_trace_vector = clamp_vector_norm(np.array(orbital_trace_vector, dtype=np.float64), max_norm=2.2)
    orbital_alignment = clamp01(
        float(np.mean(np.array(orbital_alignments, dtype=np.float64))) if orbital_alignments else 0.0
    )
    orbital_resonance = clamp01(
        float(np.mean(np.array(orbital_resonances, dtype=np.float64))) if orbital_resonances else 0.0
    )
    orbital_stability = clamp01(
        float(np.mean(np.array(orbital_stabilities, dtype=np.float64))) if orbital_stabilities else 0.0
    )
    relative_spatial_field = clamp_vector_norm(
        np.array(relative_field, dtype=np.float64) + orbital_trace_vector * np.array([0.22, 0.22, 0.24, 0.18], dtype=np.float64),
        max_norm=2.5,
    )
    return {
        "phase_orbital_vectors": [[float(component) for component in vector] for vector in orbital_vectors],
        "phase_orbital_trace_vector": [float(value) for value in orbital_trace_vector],
        "phase_orbital_alignment": float(orbital_alignment),
        "phase_orbital_resonance": float(orbital_resonance),
        "phase_orbital_stability": float(orbital_stability),
        "phase_orbital_relative_field": [float(value) for value in relative_spatial_field],
    }


def decode_temporal_nonce(
    mixed_nonce_state: int,
    pulse_seed: int,
    packet_idx: int,
    carrier_idx: int,
    pulse_index: int,
    harmonic_order: int,
    temporal_sequence_index: int,
    temporal_sequence_length: int,
    sequence_persistence_score: float,
    temporal_index_overlap: float,
    target_phase_pressure: float,
    target_phase_alignment: float,
    target_phase_cost: float,
    target_phase_anchor: float,
    target_flux_alignment: float,
    target_hex: str,
    electron_ring_stability: float,
    electron_phase_anchor: float,
    noise_resonance_nodes: list[float] | None,
    drift_compensation_vector: list[float] | None,
    relative_spatial_field: list[float] | None,
    projected_temporal_dof_vector: list[float] | None,
    decode_terms: list[float],
) -> dict[str, Any]:
    sequence_length = max(1, int(temporal_sequence_length))
    term_bank = [clamp01(float(value)) for value in list(decode_terms or [])]
    if not term_bank:
        term_bank = [0.0]
    target_phase_pressure = clamp01(float(target_phase_pressure))
    target_phase_alignment = clamp01(float(target_phase_alignment))
    target_phase_cost = clamp01(float(target_phase_cost))
    target_phase_anchor = wrap_turns(float(target_phase_anchor))
    target_flux_alignment = clamp01(float(target_flux_alignment))
    electron_ring_stability = clamp01(float(electron_ring_stability))
    electron_phase_anchor = wrap_turns(float(electron_phase_anchor))
    noise_resonance_nodes = [float(clamp01(value)) for value in list(noise_resonance_nodes or [])]
    drift_compensation_vector = [float(clamp01(value)) for value in list(drift_compensation_vector or [])]
    relative_spatial_field = [float(value) for value in list(relative_spatial_field or [])]
    projected_temporal_dof_vector = [float(value) for value in list(projected_temporal_dof_vector or [])]
    while len(noise_resonance_nodes) < 4:
        noise_resonance_nodes.append(float(noise_resonance_nodes[-1] if noise_resonance_nodes else 0.0))
    while len(drift_compensation_vector) < 4:
        drift_compensation_vector.append(float(drift_compensation_vector[-1] if drift_compensation_vector else 0.0))
    while len(relative_spatial_field) < 4:
        relative_spatial_field.append(float(relative_spatial_field[-1] if relative_spatial_field else 0.0))
    while len(projected_temporal_dof_vector) < 4:
        projected_temporal_dof_vector.append(
            float(projected_temporal_dof_vector[-1] if projected_temporal_dof_vector else 0.0)
        )
    decoded_bytes: list[int] = []
    sequence_positions: list[int] = []
    phase_ring_turns: list[float] = []
    phase_ring_deltas: list[float] = []
    carrier_turns: list[float] = []
    prefix_drive_turns: list[float] = []
    target_ring_alignments: list[float] = []
    prefix_target_turns = [
        float(value)
        for value in target_prefix_phase_metrics(target_hex, b"\x00\x00\x00\x00").get("target_prefix_turns", [])
    ]
    if not prefix_target_turns:
        prefix_target_turns = [0.0, 0.0, 0.0, 0.0]
    carrier_turn = wrap_turns(
        0.20 * clamp01(float((int(pulse_seed) & 0xFF) / 255.0))
        + 0.15 * clamp01(float(((int(pulse_seed) >> 8) & 0xFF) / 255.0))
        + 0.17 * clamp01(sequence_persistence_score)
        + 0.13 * clamp01(temporal_index_overlap)
        + 0.08 * target_phase_pressure
        + 0.07 * target_phase_alignment
        + 0.05 * target_flux_alignment
        + 0.05 * electron_ring_stability
        + 0.04 * (1.0 - target_phase_cost)
        + 0.03 * target_phase_anchor
        + 0.02 * electron_phase_anchor
        + 0.02 * float(np.mean(np.array(noise_resonance_nodes[:4], dtype=np.float64)))
        + 0.02 * float(np.mean(np.array(drift_compensation_vector[:4], dtype=np.float64)))
        + 0.03 * clamp01(float((int(mixed_nonce_state) & 0xFFFF) / 65535.0))
        + 0.03 * float((int(packet_idx) + int(carrier_idx) + int(pulse_index)) % 16) / 16.0
    )
    target_stability_turn = wrap_turns(
        0.34 * target_phase_anchor
        + 0.18 * target_phase_alignment
        + 0.14 * target_phase_pressure
        + 0.12 * target_flux_alignment
        + 0.12 * electron_phase_anchor
        + 0.10 * electron_ring_stability
        + 0.06 * float(np.mean(np.array(noise_resonance_nodes[:4], dtype=np.float64)))
        + 0.04 * float(np.mean(np.array(drift_compensation_vector[:4], dtype=np.float64)))
    )
    for byte_idx in range(4):
        sequence_position = int(
            (temporal_sequence_index + byte_idx + packet_idx + carrier_idx) % sequence_length
        )
        sequence_positions.append(sequence_position)
        primary_term = term_bank[(sequence_position + byte_idx * 3) % len(term_bank)]
        secondary_term = term_bank[
            (sequence_position * 5 + harmonic_order + pulse_index + byte_idx) % len(term_bank)
        ]
        tertiary_term = term_bank[
            (packet_idx + carrier_idx + pulse_index + byte_idx * 7) % len(term_bank)
        ]
        quaternary_term = term_bank[
            (temporal_sequence_index + packet_idx * 3 + carrier_idx * 5 + byte_idx * 11) % len(term_bank)
        ]
        sequence_window = float(sequence_position + 1) / float(sequence_length)
        seed_byte = float((int(pulse_seed) >> ((byte_idx % 4) * 8)) & 0xFF) / 255.0
        target_turn = wrap_turns(
            0.27 * primary_term
            + 0.23 * secondary_term
            + 0.16 * tertiary_term
            + 0.11 * quaternary_term
            + 0.08 * seed_byte
            + 0.06 * sequence_window
            + 0.04 * clamp01(sequence_persistence_score)
            + 0.03 * clamp01(temporal_index_overlap)
            + 0.02 * carrier_turn
        )
        target_pull = signed_turn_delta(target_stability_turn, target_turn)
        prefix_drive_turn = wrap_turns(
            0.42 * prefix_target_turns[byte_idx % len(prefix_target_turns)]
            + 0.18 * target_stability_turn
            + 0.12 * target_phase_anchor
            + 0.10 * electron_phase_anchor
            + 0.08 * target_phase_alignment
            + 0.06 * target_phase_pressure
            + 0.06 * noise_resonance_nodes[byte_idx % 4]
            + 0.04 * drift_compensation_vector[byte_idx % 4]
            + 0.04 * carrier_turn
        )
        pressure_target_turn = wrap_turns(
            target_turn
            + target_pull * (0.12 + 0.26 * target_phase_pressure)
            + signed_turn_delta(target_phase_anchor, target_turn) * 0.10
            + signed_turn_delta(electron_phase_anchor, target_turn) * 0.08
            + signed_turn_delta(prefix_drive_turn, target_turn) * 0.16
            + signed_turn_delta(wrap_turns(relative_spatial_field[byte_idx % 4]), target_turn) * 0.06
            + signed_turn_delta(wrap_turns(projected_temporal_dof_vector[byte_idx % 4]), target_turn) * 0.05
            + (target_flux_alignment - 0.5) * 0.08
            + (electron_ring_stability - 0.5) * 0.06
            + (noise_resonance_nodes[byte_idx % 4] - 0.5) * 0.10
            + (drift_compensation_vector[byte_idx % 4] - 0.5) * 0.08
        )
        byte_value = int(math.floor(pressure_target_turn * 256.0)) & 0xFF
        byte_turn = canonical_utf8_byte_turn(byte_value)
        byte_delta = signed_turn_delta(byte_turn, carrier_turn)
        anchor_delta = signed_turn_delta(target_stability_turn, carrier_turn)
        phase_delta = float(
            max(
                -0.5,
                min(
                    0.499999999,
                    0.54 * byte_delta
                    + 0.26 * anchor_delta
                    + 0.20 * signed_turn_delta(byte_turn, target_stability_turn),
                ),
            )
        )
        carrier_step = clamp01(
            0.24
            + 0.18 * clamp01(sequence_persistence_score)
            + 0.16 * clamp01(temporal_index_overlap)
            + 0.14 * target_phase_pressure
            + 0.10 * target_phase_alignment
            + 0.08 * target_flux_alignment
            + 0.06 * electron_ring_stability
            + 0.06 * noise_resonance_nodes[byte_idx % 4]
            + 0.04 * drift_compensation_vector[byte_idx % 4]
            + 0.10 * float(byte_idx + 1) / 4.0
            + 0.06 * float(harmonic_order % 5) / 4.0
        )
        carrier_turns.append(float(carrier_turn))
        prefix_drive_turns.append(float(prefix_drive_turn))
        phase_ring_turns.append(float(byte_turn))
        phase_ring_deltas.append(float(phase_delta))
        target_ring_alignments.append(float(turn_alignment(byte_turn, target_stability_turn)))
        carrier_turn = wrap_turns(carrier_turn + carrier_step * phase_delta)
        decoded_bytes.append(int(byte_value))
    decoded_blob = bytes(decoded_bytes)
    decoded_hex = decoded_blob.hex()
    decoded_value = int.from_bytes(decoded_blob, byteorder="big", signed=False) if decoded_blob else 0
    phase_metrics = utf8_phase_ring_metrics(
        decoded_blob,
        byte_turns=phase_ring_turns,
        phase_deltas=phase_ring_deltas,
    )
    prefix_metrics = target_prefix_phase_metrics(
        target_hex,
        decoded_blob,
        phase_ring_turns=phase_ring_turns,
    )
    prefix_vector_metrics = build_prefix_trajectory_vector(
        target_hex=target_hex,
        target_prefix_turns=prefix_drive_turns or prefix_target_turns,
        observed_turns=phase_ring_turns,
        target_phase_pressure=target_phase_pressure,
        target_phase_alignment=target_phase_alignment,
        target_flux_alignment=target_flux_alignment,
        electron_ring_stability=electron_ring_stability,
        electron_phase_anchor=electron_phase_anchor,
    )
    prefix_trajectory_vector = [
        float(value)
        for value in list(prefix_vector_metrics.get("prefix_trajectory_vector", []) or [])
    ]
    if not prefix_trajectory_vector:
        prefix_trajectory_vector = [float(value) for value in prefix_drive_turns[:4]]
    while len(prefix_trajectory_vector) < 4:
        prefix_trajectory_vector.append(float(prefix_trajectory_vector[-1] if prefix_trajectory_vector else 0.0))
    orbital_trace_metrics = build_phase_orbital_trace(
        phase_ring_turns=phase_ring_turns,
        phase_ring_deltas=phase_ring_deltas,
        carrier_turns=carrier_turns,
        prefix_trajectory_vector=prefix_trajectory_vector,
        noise_resonance_nodes=noise_resonance_nodes,
        drift_compensation_vector=drift_compensation_vector,
        relative_spatial_field=relative_spatial_field,
        projected_temporal_dof_vector=projected_temporal_dof_vector,
        utf8_blob=decoded_blob,
    )
    phase_orbital_trace_vector = [
        float(value)
        for value in list(orbital_trace_metrics.get("phase_orbital_trace_vector", []) or [])
    ]
    while len(phase_orbital_trace_vector) < 4:
        phase_orbital_trace_vector.append(
            float(phase_orbital_trace_vector[-1] if phase_orbital_trace_vector else 0.0)
        )
    if sequence_positions:
        sequence_span = (
            float(max(sequence_positions) - min(sequence_positions) + 1) / float(sequence_length)
        )
    else:
        sequence_span = 1.0
    temporal_coverage = float(len(set(sequence_positions))) / float(sequence_length)
    target_ring_alignment = clamp01(
        float(np.mean(np.array(target_ring_alignments, dtype=np.float64))) if target_ring_alignments else 0.0
    )
    target_prefix_vector_alignment = clamp01(
        float(prefix_vector_metrics.get("prefix_trajectory_vector_alignment", 0.0))
    )
    target_prefix_vector_flux_alignment = clamp01(
        float(prefix_vector_metrics.get("prefix_trajectory_flux_alignment", 0.0))
    )
    target_prefix_vector_phase_pressure = clamp01(
        float(prefix_vector_metrics.get("prefix_trajectory_phase_pressure", 0.0))
    )
    target_prefix_lock = clamp01(float(prefix_metrics.get("target_prefix_lock", 0.0)))
    target_prefix_lock = clamp01(
        0.74 * target_prefix_lock
        + 0.18 * target_prefix_vector_alignment
        + 0.08 * target_prefix_vector_flux_alignment
    )
    target_prefix_phase_alignment = clamp01(float(prefix_metrics.get("target_prefix_phase_alignment", 0.0)))
    target_prefix_flux_alignment = clamp01(float(prefix_metrics.get("target_prefix_flux_alignment", 0.0)))
    phase_orbital_alignment = clamp01(float(orbital_trace_metrics.get("phase_orbital_alignment", 0.0)))
    phase_orbital_resonance = clamp01(float(orbital_trace_metrics.get("phase_orbital_resonance", 0.0)))
    phase_orbital_stability = clamp01(float(orbital_trace_metrics.get("phase_orbital_stability", 0.0)))
    prefix_threshold = clamp01(
        0.10
        + 0.20 * target_phase_pressure
        + 0.16 * target_phase_alignment
        + 0.12 * target_flux_alignment
        + 0.12 * electron_ring_stability
        + 0.10 * target_ring_alignment
        + 0.06 * target_prefix_vector_alignment
        + 0.04 * phase_orbital_alignment
        + 0.08 * (1.0 - target_phase_cost)
    )
    prefix_deviation = max(prefix_threshold - target_prefix_lock, 0.0)
    prefix_asymptote_pressure = clamp01(
        0.76 * (prefix_deviation / max(prefix_threshold + prefix_deviation, 1.0e-9))
        + 0.24 * (1.0 - target_prefix_vector_alignment)
    )
    decode_integrity_score = clamp01(
        0.44 * float(phase_metrics.get("phase_integrity_score", 0.0))
        + 0.18 * temporal_coverage
        + 0.12 * clamp01(sequence_span)
        + 0.10 * clamp01(sequence_persistence_score)
        + 0.08 * clamp01(temporal_index_overlap)
        + 0.08 * float(phase_metrics.get("phase_delta_alignment", 0.0))
        + 0.08 * target_ring_alignment
        + 0.06 * target_prefix_lock
        + 0.05 * target_prefix_vector_alignment
        + 0.04 * target_prefix_vector_phase_pressure
        + 0.06 * phase_orbital_alignment
        + 0.04 * phase_orbital_resonance
        + 0.04 * phase_orbital_stability
        - 0.06 * prefix_asymptote_pressure
    )
    phase_delta_word = 0
    for delta_idx, phase_delta in enumerate(phase_ring_deltas):
        delta_value = int(round(clamp01((float(phase_delta) + 0.5)) * 65535.0)) & 0xFFFF
        phase_delta_word ^= rotate_left_32(delta_value, ((delta_idx * 7) + 3) & 31)
    carrier_word = 0
    for turn_idx, turn_value in enumerate(carrier_turns):
        carrier_value = int(round(wrap_turns(float(turn_value)) * 65535.0)) & 0xFFFF
        carrier_word ^= rotate_left_32(carrier_value, ((turn_idx * 5) + 1) & 31)
    prefix_vector_word = 0
    for vector_idx, vector_turn in enumerate(prefix_trajectory_vector[:4]):
        vector_value = int(round(wrap_turns(float(vector_turn)) * 65535.0)) & 0xFFFF
        prefix_vector_word ^= rotate_left_32(vector_value, ((vector_idx * 9) + 5) & 31)
    orbital_word = 0
    for orbital_idx, orbital_value in enumerate(phase_orbital_trace_vector[:4]):
        orbital_lane = int(round(clamp01(0.5 + 0.25 * float(orbital_value)) * 65535.0)) & 0xFFFF
        orbital_word ^= rotate_left_32(orbital_lane, ((orbital_idx * 11) + 7) & 31)
    phase_shift = (
        (
            int(temporal_sequence_index)
            + harmonic_order
            + carrier_idx
            + int(round(float(phase_metrics.get("phase_flux_ratio", 0.0)) * 17.0))
        )
        % 19
    ) + 1
    final_nonce = rotate_left_32(
        int(mixed_nonce_state) ^ decoded_value ^ phase_delta_word ^ carrier_word ^ prefix_vector_word ^ orbital_word,
        phase_shift,
    )
    final_nonce = (
        final_nonce
        + rotate_left_32(decoded_value, ((packet_idx + pulse_index) % 11) + 1)
        + rotate_left_32(prefix_vector_word, ((carrier_idx + harmonic_order) % 13) + 1)
        + rotate_left_32(orbital_word, ((temporal_sequence_index + carrier_idx) % 17) + 1)
        + int(round(decode_integrity_score * 65535.0))
    ) & 0xFFFFFFFF
    if (
        int(phase_metrics.get("unique_symbol_count", 0)) <= 2
        and float(phase_metrics.get("longest_run_ratio", 0.0)) >= 0.50
    ):
        final_nonce ^= 0xA5A5A5A5
        final_nonce = rotate_left_32(final_nonce, ((harmonic_order + pulse_index) % 13) + 1)
    return {
        "nonce": int(final_nonce),
        "mixed_nonce": int(mixed_nonce_state) & 0xFFFFFFFF,
        "decoded_raw_nonce": int(decoded_value) & 0xFFFFFFFF,
        "decoded_sequence_hex": decoded_hex,
        "decoded_sequence_utf8_preview": str(phase_metrics.get("utf8_preview", "")),
        "decode_sequence_positions": [int(position) for position in sequence_positions],
        "decode_phase_ring_turns": [float(value) for value in phase_ring_turns],
        "decode_phase_ring_deltas": [float(value) for value in phase_ring_deltas],
        "decode_carrier_turns": [float(value) for value in carrier_turns],
        "decode_temporal_coverage": float(temporal_coverage),
        "decode_temporal_span": float(clamp01(sequence_span)),
        "decode_integrity_score": float(decode_integrity_score),
        "decode_symbol_entropy": float(phase_metrics.get("byte_entropy_norm", 0.0)),
        "decode_transition_ratio": float(phase_metrics.get("phase_flux_ratio", 0.0)),
        "decode_longest_run_ratio": float(phase_metrics.get("longest_run_ratio", 0.0)),
        "decode_dominant_symbol_ratio": float(phase_metrics.get("dominant_symbol_ratio", 1.0)),
        "decode_unique_symbol_count": int(phase_metrics.get("unique_symbol_count", 0)),
        "decode_unique_symbol_ratio": float(phase_metrics.get("unique_symbol_ratio", 0.0)),
        "decode_phase_delta_alignment": float(phase_metrics.get("phase_delta_alignment", 0.0)),
        "decode_phase_delta_balance": float(phase_metrics.get("phase_delta_balance", 0.0)),
        "decode_phase_integrity_score": float(phase_metrics.get("phase_integrity_score", 0.0)),
        "decode_target_ring_alignment": float(target_ring_alignment),
        "decode_target_phase_pressure": float(target_phase_pressure),
        "decode_target_prefix_lock": float(target_prefix_lock),
        "decode_target_prefix_phase_alignment": float(target_prefix_phase_alignment),
        "decode_target_prefix_flux_alignment": float(target_prefix_flux_alignment),
        "decode_target_prefix_vector": [float(value) for value in prefix_trajectory_vector[:4]],
        "decode_target_prefix_vector_alignment": float(target_prefix_vector_alignment),
        "decode_target_prefix_vector_flux_alignment": float(target_prefix_vector_flux_alignment),
        "decode_target_prefix_vector_phase_pressure": float(target_prefix_vector_phase_pressure),
        "decode_phase_orbital_vectors": list(orbital_trace_metrics.get("phase_orbital_vectors", []) or []),
        "decode_phase_orbital_trace_vector": [float(value) for value in phase_orbital_trace_vector[:4]],
        "decode_phase_orbital_alignment": float(phase_orbital_alignment),
        "decode_phase_orbital_resonance": float(phase_orbital_resonance),
        "decode_phase_orbital_stability": float(phase_orbital_stability),
        "decode_phase_orbital_relative_field": [
            float(value)
            for value in list(orbital_trace_metrics.get("phase_orbital_relative_field", []) or relative_spatial_field[:4])
        ],
        "decode_prefix_asymptote_pressure": float(prefix_asymptote_pressure),
    }


def ensure_temporal_decode_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    decoded_sequence_hex = str(candidate.get("decoded_sequence_hex", "")).strip().lower()
    decoded_sequence_hex = "".join(ch for ch in decoded_sequence_hex if ch in "0123456789abcdef")
    fallback_nonce = int(candidate.get("decoded_raw_nonce", candidate.get("nonce", 0))) & 0xFFFFFFFF
    if not decoded_sequence_hex:
        decoded_sequence_hex = f"{fallback_nonce:08x}"
    if len(decoded_sequence_hex) % 2 == 1:
        decoded_sequence_hex = f"0{decoded_sequence_hex}"
    decoded_blob = bytes.fromhex(decoded_sequence_hex)
    phase_ring_turns = [
        float(value) for value in list(candidate.get("decode_phase_ring_turns", []) or [])
    ]
    phase_ring_deltas = [
        float(value) for value in list(candidate.get("decode_phase_ring_deltas", []) or [])
    ]
    metrics = utf8_phase_ring_metrics(
        decoded_blob,
        byte_turns=phase_ring_turns if phase_ring_turns else None,
        phase_deltas=phase_ring_deltas if phase_ring_deltas else None,
    )
    sequence_positions = [
        int(position)
        for position in list(candidate.get("decode_sequence_positions", []) or [])
    ]
    sequence_length = max(1, int(candidate.get("temporal_sequence_length", len(sequence_positions) or 1)))
    if sequence_positions:
        normalized_positions = [int(position) % sequence_length for position in sequence_positions]
        temporal_coverage = float(len(set(normalized_positions))) / float(sequence_length)
        temporal_span = float(max(normalized_positions) - min(normalized_positions) + 1) / float(sequence_length)
    else:
        temporal_coverage = clamp01(
            0.25
            + 0.45 * float(candidate.get("sequence_persistence_score", 0.0))
            + 0.30 * float(candidate.get("temporal_index_overlap", 0.0))
        )
        temporal_span = temporal_coverage
    decode_integrity_score = float(candidate.get("decode_integrity_score", 0.0))
    if decode_integrity_score <= 0.0:
        decode_integrity_score = clamp01(
            0.44 * float(metrics.get("phase_integrity_score", 0.0))
            + 0.18 * temporal_coverage
            + 0.12 * clamp01(temporal_span)
            + 0.10 * clamp01(float(candidate.get("sequence_persistence_score", 0.0)))
            + 0.08 * clamp01(float(candidate.get("temporal_index_overlap", 0.0)))
            + 0.08 * float(metrics.get("phase_delta_alignment", 0.0))
        )
    candidate["decoded_sequence_hex"] = decoded_sequence_hex
    candidate["decoded_sequence_utf8_preview"] = str(metrics.get("utf8_preview", ""))
    candidate["decode_integrity_score"] = float(decode_integrity_score)
    candidate["decode_symbol_entropy"] = float(metrics.get("byte_entropy_norm", 0.0))
    candidate["decode_transition_ratio"] = float(metrics.get("phase_flux_ratio", 0.0))
    candidate["decode_longest_run_ratio"] = float(metrics.get("longest_run_ratio", 1.0))
    candidate["decode_dominant_symbol_ratio"] = float(metrics.get("dominant_symbol_ratio", 1.0))
    candidate["decode_unique_symbol_count"] = int(metrics.get("unique_symbol_count", 0))
    candidate["decode_unique_symbol_ratio"] = float(metrics.get("unique_symbol_ratio", 0.0))
    candidate["decode_phase_delta_alignment"] = float(metrics.get("phase_delta_alignment", 0.0))
    candidate["decode_phase_delta_balance"] = float(metrics.get("phase_delta_balance", 0.0))
    candidate["decode_phase_integrity_score"] = float(metrics.get("phase_integrity_score", 0.0))
    candidate["decode_target_ring_alignment"] = float(
        candidate.get("decode_target_ring_alignment", metrics.get("phase_delta_alignment", 0.0))
    )
    candidate["decode_target_phase_pressure"] = float(
        candidate.get("decode_target_phase_pressure", 0.0)
    )
    candidate["decode_target_prefix_lock"] = float(
        candidate.get("decode_target_prefix_lock", 0.0)
    )
    candidate["decode_target_prefix_phase_alignment"] = float(
        candidate.get("decode_target_prefix_phase_alignment", metrics.get("phase_delta_alignment", 0.0))
    )
    candidate["decode_target_prefix_flux_alignment"] = float(
        candidate.get("decode_target_prefix_flux_alignment", metrics.get("phase_flux_ratio", 0.0))
    )
    prefix_vector = [
        float(value)
        for value in list(candidate.get("decode_target_prefix_vector", []) or [])
    ]
    if not prefix_vector:
        prefix_vector = [float(value) for value in phase_ring_turns[:4]]
    while len(prefix_vector) < 4:
        prefix_vector.append(float(prefix_vector[-1] if prefix_vector else 0.0))
    candidate["decode_target_prefix_vector"] = [float(wrap_turns(value)) for value in prefix_vector[:4]]
    candidate["decode_target_prefix_vector_alignment"] = float(
        candidate.get(
            "decode_target_prefix_vector_alignment",
            candidate.get("decode_target_prefix_lock", 0.0),
        )
    )
    candidate["decode_target_prefix_vector_flux_alignment"] = float(
        candidate.get(
            "decode_target_prefix_vector_flux_alignment",
            candidate.get("decode_target_prefix_flux_alignment", metrics.get("phase_flux_ratio", 0.0)),
        )
    )
    candidate["decode_target_prefix_vector_phase_pressure"] = float(
        candidate.get(
            "decode_target_prefix_vector_phase_pressure",
            candidate.get("decode_target_phase_pressure", 0.0),
        )
    )
    phase_orbital_trace_vector = [
        float(value)
        for value in list(candidate.get("decode_phase_orbital_trace_vector", []) or [])
    ]
    if not phase_orbital_trace_vector:
        phase_orbital_trace_vector = [float(value) for value in candidate["decode_target_prefix_vector"][:4]]
    while len(phase_orbital_trace_vector) < 4:
        phase_orbital_trace_vector.append(
            float(phase_orbital_trace_vector[-1] if phase_orbital_trace_vector else 0.0)
        )
    candidate["decode_phase_orbital_trace_vector"] = [
        float(value) for value in phase_orbital_trace_vector[:4]
    ]
    candidate["decode_phase_orbital_vectors"] = list(
        candidate.get("decode_phase_orbital_vectors", []) or []
    )
    candidate["decode_phase_orbital_alignment"] = float(
        candidate.get(
            "decode_phase_orbital_alignment",
            candidate.get("decode_target_prefix_vector_alignment", 0.0),
        )
    )
    candidate["decode_phase_orbital_resonance"] = float(
        candidate.get(
            "decode_phase_orbital_resonance",
            candidate.get("decode_target_prefix_vector_flux_alignment", metrics.get("phase_flux_ratio", 0.0)),
        )
    )
    candidate["decode_phase_orbital_stability"] = float(
        candidate.get(
            "decode_phase_orbital_stability",
            candidate.get("decode_phase_integrity_score", metrics.get("phase_integrity_score", 0.0)),
        )
    )
    orbital_relative_field = [
        float(value)
        for value in list(candidate.get("decode_phase_orbital_relative_field", []) or [])
    ]
    if not orbital_relative_field:
        orbital_relative_field = [float(value) for value in phase_orbital_trace_vector[:4]]
    while len(orbital_relative_field) < 4:
        orbital_relative_field.append(float(orbital_relative_field[-1] if orbital_relative_field else 0.0))
    candidate["decode_phase_orbital_relative_field"] = [
        float(value) for value in orbital_relative_field[:4]
    ]
    candidate["decode_prefix_asymptote_pressure"] = float(
        candidate.get("decode_prefix_asymptote_pressure", 1.0 - candidate["decode_target_prefix_lock"])
    )
    candidate["decode_temporal_coverage"] = float(
        candidate.get("decode_temporal_coverage", temporal_coverage)
    )
    candidate["decode_temporal_span"] = float(
        candidate.get("decode_temporal_span", clamp01(temporal_span))
    )
    candidate["decoded_raw_nonce"] = int(fallback_nonce)
    return candidate


def build_prototype_share_target(
    network_target_hex: str,
    sample_budget: int,
    desired_valid_count: int,
) -> dict[str, Any]:
    normalized_target = normalize_hex_64(network_target_hex)
    try:
        network_target_value = int(normalized_target, 16)
    except Exception:
        network_target_value = 1
    max_target_value = (1 << 256) - 1
    sample_budget = max(1, int(sample_budget))
    desired_valid_count = max(1, min(int(desired_valid_count), sample_budget))
    prototype_target_value = max(
        network_target_value,
        min(
            max_target_value,
            (((max_target_value + 1) * desired_valid_count) // sample_budget) - 1,
        ),
    )
    if network_target_value > 0:
        target_multiplier = float(prototype_target_value) / float(network_target_value)
    else:
        target_multiplier = float(max_target_value)
    target_probability = min(
        1.0,
        float(desired_valid_count) / float(sample_budget),
    )
    return {
        "prototype_target_hex": f"{prototype_target_value:064x}",
        "sample_budget": int(sample_budget),
        "desired_valid_count": int(desired_valid_count),
        "target_multiplier": float(target_multiplier),
        "target_probability": float(target_probability),
    }


Q32_32_SCALE = float(1 << 32)
U64_PHASE_MODULUS = float(1 << 64)


def encode_q32_32(value: float) -> int:
    return int(round(float(value) * Q32_32_SCALE))


def decode_q32_32(value: Any) -> float:
    try:
        return float(int(value)) / Q32_32_SCALE
    except Exception:
        return 0.0


def phase_turns_to_u64(turns: float) -> int:
    return int(round(wrap_turns(turns) * (U64_PHASE_MODULUS - 1.0))) & 0xFFFFFFFFFFFFFFFF


def phase_u64_to_turns(value: Any) -> float:
    try:
        return wrap_turns(float(int(value) & 0xFFFFFFFFFFFFFFFF) / (U64_PHASE_MODULUS - 1.0))
    except Exception:
        return 0.0


def wrap_turns(value: float) -> float:
    wrapped = math.fmod(float(value), 1.0)
    if wrapped < 0.0:
        wrapped += 1.0
    return float(wrapped)


def turn_distance(lhs: float, rhs: float) -> float:
    delta = abs(wrap_turns(lhs) - wrap_turns(rhs))
    return float(min(delta, 1.0 - delta))


def turn_alignment(lhs: float, rhs: float) -> float:
    return clamp01(1.0 - turn_distance(lhs, rhs) / 0.5)


def signed_turn_delta(lhs: float, rhs: float) -> float:
    delta = wrap_turns(lhs) - wrap_turns(rhs)
    if delta >= 0.5:
        delta -= 1.0
    elif delta < -0.5:
        delta += 1.0
    return float(delta)


def mean_feedback_metric(
    items: list[dict[str, Any]] | None,
    key: str,
    default: float = 0.0,
) -> float:
    rows = list(items or [])
    if not rows:
        return float(default)
    values = [float(row.get(key, default)) for row in rows]
    if not values:
        return float(default)
    return float(np.mean(values))


def gpu_feedback_projection_tensor_from_rows(rows: Any = None) -> np.ndarray:
    default_rows = [
        [1.0, 0.0, 0.0, 0.0, 0.50, 0.50, 0.50, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.50, 0.50, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 0.50, 0.50],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 0.50, 0.0, 0.50],
    ]
    projection_tensor = np.array(
        rows if rows is not None else default_rows,
        dtype=np.float64,
    )
    if projection_tensor.shape != (4, 10):
        projection_tensor = np.array(default_rows, dtype=np.float64)
    row_sums = np.sum(projection_tensor, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return projection_tensor / row_sums


def build_vector_feedback_context(
    pulse_index: int,
    lattice_calibration: dict[str, Any] | None = None,
    pulse_sweep: dict[str, Any] | None = None,
    feedback_state: dict[str, Any] | None = None,
    target_profile: dict[str, Any] | None = None,
    candidate_pool: list[dict[str, Any]] | None = None,
    cuda_kernel_telemetry: dict[str, Any] | None = None,
    effective_vector: dict[str, Any] | None = None,
    temporal_manifold: dict[str, Any] | None = None,
    kernel_execution_event: dict[str, Any] | None = None,
    feedback_stage: str = "runtime",
) -> dict[str, Any]:
    del pulse_index
    lattice_calibration = dict(lattice_calibration or {})
    pulse_sweep = dict(pulse_sweep or {})
    feedback_state = dict(feedback_state or {})
    target_profile = dict(target_profile or {})
    candidate_pool = list(candidate_pool or [])
    cuda_kernel_telemetry = dict(cuda_kernel_telemetry or {})
    effective_vector = dict(effective_vector or {})
    temporal_manifold = dict(temporal_manifold or {})
    kernel_execution_event = dict(kernel_execution_event or {})
    feedback_stage = str(feedback_stage or "runtime").strip().lower() or "runtime"
    ancilla_summary = dict(feedback_state.get("kernel_ancilla_summary", {}) or {})
    stage_injection_weight = 1.0 if feedback_stage == "injection" else 0.0
    stage_pre_cuda_weight = 1.0 if feedback_stage == "pre_cuda" else 0.0
    stage_post_cuda_weight = 1.0 if feedback_stage == "post_cuda" else 0.0

    effective_spatial_magnitude = clamp01(float(effective_vector.get("spatial_magnitude", 0.0)))
    effective_temporal_projection = clamp01(float(effective_vector.get("t_eff", 0.0)))
    effective_coherence_bias = clamp01(float(effective_vector.get("coherence_bias", 0.0)))
    effective_axis_energy = clamp01(
        (
            abs(float(effective_vector.get("x", 0.0)))
            + abs(float(effective_vector.get("y", 0.0)))
            + abs(float(effective_vector.get("z", 0.0)))
        )
        / 3.0
    )
    manifold_coherence_norm = clamp01(
        float(temporal_manifold.get("coherence_norm", effective_coherence_bias))
    )
    manifold_phase_transport = clamp01(
        float(
            temporal_manifold.get(
                "phase_transport_norm",
                temporal_manifold.get("phase_transport", effective_temporal_projection),
            )
        )
    )
    manifold_energy_norm = clamp01(
        float(
            temporal_manifold.get(
                "energy_norm",
                temporal_manifold.get("temporal_energy", effective_spatial_magnitude),
            )
        )
    )
    entry_calibration_readiness = clamp01(
        float(
            kernel_execution_event.get(
                "calibration_readiness",
                feedback_state.get("calibration_readiness", 0.0),
            )
        )
    )
    entry_target_gate = clamp01(
        float(kernel_execution_event.get("target_gate", feedback_state.get("target_gate", 0.0)))
    )
    entry_field_alignment = clamp01(
        float(
            kernel_execution_event.get(
                "field_alignment_score",
                feedback_state.get("field_alignment_score", 0.0),
            )
        )
    )
    entry_kernel_drive = clamp01(
        float(
            kernel_execution_event.get(
                "kernel_drive_mean",
                feedback_state.get("kernel_drive_mean", 0.0),
            )
        )
    )
    entry_delta_gate = clamp01(
        float(
            kernel_execution_event.get(
                "feedback_delta_response_gate",
                feedback_state.get("feedback_delta_response_gate", 0.0),
            )
        )
    )
    stage_drive = clamp01(
        0.20 * entry_calibration_readiness
        + 0.18 * entry_target_gate
        + 0.16 * entry_field_alignment
        + 0.12 * manifold_coherence_norm
        + 0.10 * manifold_phase_transport
        + 0.08 * effective_spatial_magnitude
        + 0.06 * effective_temporal_projection
        + 0.06 * entry_kernel_drive
        + 0.04 * entry_delta_gate
    )
    trace_state = dict(feedback_state.get("substrate_trace_state", {}) or {})
    trace_axis_vector = np.array(
        list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if trace_axis_vector.shape[0] != 4:
        trace_axis_vector = np.zeros(4, dtype=np.float64)
    trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
    trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
    trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
    trace_memory = clamp01(float(trace_state.get("trace_memory", 0.0)))
    trace_flux = clamp01(float(trace_state.get("trace_flux", 0.0)))
    trace_stability = clamp01(float(trace_state.get("trace_stability", 0.0)))
    trace_temporal_persistence = clamp01(
        float(trace_state.get("trace_temporal_persistence", 0.0))
    )
    trace_temporal_overlap = clamp01(
        float(trace_state.get("trace_temporal_overlap", 0.0))
    )
    trace_voltage_frequency_flux = clamp01(
        float(trace_state.get("trace_voltage_frequency_flux", 0.0))
    )
    trace_frequency_voltage_flux = clamp01(
        float(trace_state.get("trace_frequency_voltage_flux", 0.0))
    )

    base_dof_vector = np.array(
        list(lattice_calibration.get("gpu_pulse_dof_vector", []) or [0.0] * 10),
        dtype=np.float64,
    )
    if base_dof_vector.shape[0] != 10:
        base_dof_vector = np.zeros(10, dtype=np.float64)
    base_dof_tensor = np.array(
        lattice_calibration.get(
            "gpu_pulse_dof_tensor",
            np.zeros((10, 10), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if base_dof_tensor.shape != (10, 10):
        base_dof_tensor = np.zeros((10, 10), dtype=np.float64)
    projection_tensor = gpu_feedback_projection_tensor_from_rows(
        lattice_calibration.get("gpu_pulse_projection_tensor")
    )

    target_window = clamp01(float(target_profile.get("difficulty_window", 0.5)))
    candidate_count = int(len(candidate_pool))
    candidate_count_norm = clamp01(float(candidate_count) / 32768.0)
    expanded_eval_norm = clamp01(
        float(cuda_kernel_telemetry.get("expanded_eval_count", candidate_count))
        / 131072.0
    )
    expanded_keep_norm = clamp01(
        float(cuda_kernel_telemetry.get("expanded_keep_count", candidate_count))
        / 32768.0
    )
    search_volume_norm = clamp01(
        (float(cuda_kernel_telemetry.get("search_volume_gain", 1.0)) - 1.0) / 2.5
    )
    ancilla_commit_ratio = clamp01(float(ancilla_summary.get("commit_ratio", 0.0)))
    ancilla_convergence = clamp01(float(ancilla_summary.get("convergence_mean", 0.0)))
    ancilla_flux = clamp01(float(ancilla_summary.get("flux_mean", 0.0)))
    ancilla_phase_alignment = clamp01(float(ancilla_summary.get("phase_alignment_mean", 0.0)))
    ancilla_current_norm = clamp01(float(ancilla_summary.get("current_norm_mean", 0.0)))

    coherence_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "coherence_peak",
            float(
                pulse_sweep.get(
                    "coherence",
                    feedback_state.get("calibration_readiness", target_window),
                )
            ),
        )
    )
    target_alignment_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "target_alignment",
            float(feedback_state.get("target_gate", target_window)),
        )
    )
    interference_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "interference_resonance",
            float(
                feedback_state.get(
                    "interference_accounting",
                    pulse_sweep.get("score", target_window),
                )
            ),
        )
    )
    vector_alignment_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "vector_alignment",
            float(feedback_state.get("field_alignment_score", target_window)),
        )
    )
    basin_alignment_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "basin_alignment",
            float(feedback_state.get("field_alignment_score", target_window)),
        )
    )
    row_activation_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "row_activation",
            float(feedback_state.get("motif_energy", 0.0)),
        )
    )
    motif_alignment_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "motif_alignment",
            float(feedback_state.get("motif_consistency", 0.0)),
        )
    )
    phase_pressure_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "phase_length_pressure",
            float(target_window),
        )
    )
    phase_span_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "phase_length_span",
            float(max(0.0, 1.0 - target_window)),
        )
    )
    phase_confinement_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "phase_confinement_pressure",
            float(phase_pressure_mean),
        )
    )
    sequence_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "sequence_persistence_score",
            float(feedback_state.get("sequence_persistence_score", target_window)),
        )
    )
    overlap_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "temporal_index_overlap",
            float(feedback_state.get("temporal_index_overlap", target_window)),
        )
    )
    vf_flux_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "voltage_frequency_flux",
            float(feedback_state.get("voltage_frequency_flux", target_window)),
        )
    )
    fv_flux_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "frequency_voltage_flux",
            float(feedback_state.get("frequency_voltage_flux", target_window)),
        )
    )
    shared_score_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "shared_score",
            float(pulse_sweep.get("coherence", coherence_mean)),
        )
    )
    crosstalk_mean = clamp01(
        mean_feedback_metric(candidate_pool, "crosstalk", 0.0)
    )
    temporal_weight_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "temporal_weight",
            float(sequence_mean),
        )
    )
    kernel_balance_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "kernel_balance_score",
            float(feedback_state.get("kernel_balance_mean", 0.0)),
        )
    )
    harmonic_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "harmonic_resonance_score",
            float(feedback_state.get("kernel_harmonic_resonance_mean", 0.0)),
        )
    )
    kernel_phase_alignment_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "kernel_phase_alignment",
            float(feedback_state.get("kernel_resonance_alignment_mean", 0.0)),
        )
    )
    delta_gate_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "kernel_delta_gate",
            float(feedback_state.get("kernel_delta_gate_mean", 0.0)),
        )
    )
    delta_memory_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "kernel_delta_memory",
            float(feedback_state.get("kernel_delta_memory_mean", 0.0)),
        )
    )
    delta_flux_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "kernel_delta_flux",
            float(feedback_state.get("kernel_delta_flux_mean", 0.0)),
        )
    )
    delta_phase_alignment_mean = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "kernel_delta_phase_alignment",
            float(feedback_state.get("kernel_delta_phase_alignment_mean", 0.0)),
        )
    )
    candidate_ancilla_convergence = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "ancilla_convergence",
            ancilla_convergence,
        )
    )
    candidate_ancilla_flux = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "ancilla_flux_norm",
            ancilla_flux,
        )
    )
    candidate_ancilla_phase_alignment = clamp01(
        mean_feedback_metric(
            candidate_pool,
            "ancilla_phase_alignment",
            ancilla_phase_alignment,
        )
    )

    field_pressure = clamp01(
        float(
            feedback_state.get(
                "field_pressure",
                lattice_calibration.get("field_pressure", 0.0),
            )
        )
    )
    field_amplitude_bias = clamp01(
        float(feedback_state.get("field_amplitude_bias", pulse_sweep.get("score", 0.0)))
    )
    field_frequency_bias = clamp01(
        float(feedback_state.get("field_frequency_bias", target_window))
    )
    field_alignment_score = clamp01(
        float(feedback_state.get("field_alignment_score", vector_alignment_mean))
    )
    target_gate = clamp01(float(feedback_state.get("target_gate", target_alignment_mean)))
    calibration_readiness = clamp01(
        float(feedback_state.get("calibration_readiness", coherence_mean))
    )
    interference_accounting = clamp01(
        float(feedback_state.get("interference_accounting", interference_mean))
    )
    motif_consistency = clamp01(
        float(feedback_state.get("motif_consistency", motif_alignment_mean))
    )
    motif_energy = clamp01(float(feedback_state.get("motif_energy", row_activation_mean)))
    motif_stability = clamp01(
        float(
            feedback_state.get(
                "motif_stability",
                0.5 * motif_alignment_mean + 0.5 * sequence_mean,
            )
        )
    )
    kernel_drive_mean = clamp01(
        float(feedback_state.get("kernel_drive_mean", harmonic_mean))
    )
    kernel_control_gate = clamp01(
        float(feedback_state.get("kernel_control_gate", kernel_balance_mean))
    )

    activity_norm = clamp01(
        0.18 * candidate_count_norm
        + 0.16 * expanded_eval_norm
        + 0.12 * expanded_keep_norm
        + 0.16 * interference_mean
        + 0.14 * row_activation_mean
        + 0.12 * kernel_drive_mean
        + 0.12 * target_alignment_mean
        + 0.10 * ancilla_commit_ratio
        + 0.18 * stage_drive * stage_injection_weight
        + 0.10 * stage_drive * stage_pre_cuda_weight
        + 0.06 * stage_drive * stage_post_cuda_weight
        + 0.12 * trace_support
        + 0.10 * trace_resonance
    )
    amplitude_driver = clamp01(
        0.20 * field_amplitude_bias
        + 0.16 * coherence_mean
        + 0.14 * interference_accounting
        + 0.12 * target_gate
        + 0.10 * phase_pressure_mean
        + 0.10 * motif_alignment_mean
        + 0.10 * row_activation_mean
        + 0.08 * shared_score_mean
        + 0.08 * candidate_ancilla_convergence
        + 0.06 * ancilla_current_norm
        + 0.12 * stage_drive * stage_injection_weight
        + 0.06 * effective_temporal_projection * stage_injection_weight
        + 0.06 * entry_calibration_readiness * stage_injection_weight
        + 0.04 * stage_drive * stage_pre_cuda_weight
        + 0.08 * float(trace_axis_vector[1])
        + 0.06 * trace_memory
        + 0.04 * trace_alignment
    )
    frequency_driver = clamp01(
        0.20 * field_frequency_bias
        + 0.16 * fv_flux_mean
        + 0.14 * vf_flux_mean
        + 0.12 * sequence_mean
        + 0.10 * overlap_mean
        + 0.10 * target_alignment_mean
        + 0.10 * kernel_phase_alignment_mean
        + 0.08 * harmonic_mean
        + 0.08 * candidate_ancilla_phase_alignment
        + 0.06 * ancilla_commit_ratio
        + 0.12 * stage_drive * stage_injection_weight
        + 0.06 * manifold_phase_transport * stage_injection_weight
        + 0.06 * entry_target_gate * stage_injection_weight
        + 0.04 * stage_drive * stage_pre_cuda_weight
        + 0.08 * float(trace_axis_vector[0])
        + 0.06 * trace_frequency_voltage_flux
        + 0.04 * trace_temporal_persistence
    )
    voltage_driver = clamp01(
        0.20 * vf_flux_mean
        + 0.16 * field_alignment_score
        + 0.14 * target_alignment_mean
        + 0.12 * kernel_balance_mean
        + 0.10 * field_pressure
        + 0.10 * basin_alignment_mean
        + 0.10 * interference_mean
        + 0.08 * (1.0 - phase_span_mean)
        + 0.08 * ancilla_current_norm
        + 0.06 * candidate_ancilla_flux
        + 0.10 * stage_drive * stage_injection_weight
        + 0.08 * entry_field_alignment * stage_injection_weight
        + 0.06 * effective_axis_energy * stage_injection_weight
        + 0.04 * stage_drive * stage_post_cuda_weight
        + 0.08 * float(trace_axis_vector[3])
        + 0.06 * trace_voltage_frequency_flux
        + 0.04 * trace_resonance
    )
    current_driver = clamp01(
        0.20 * delta_flux_mean
        + 0.16 * delta_gate_mean
        + 0.14 * phase_confinement_mean
        + 0.12 * row_activation_mean
        + 0.12 * shared_score_mean
        + 0.10 * interference_mean
        + 0.08 * temporal_weight_mean
        + 0.08 * delta_memory_mean
        + 0.10 * ancilla_commit_ratio
        + 0.08 * candidate_ancilla_flux
        + 0.06 * candidate_ancilla_convergence
        + 0.10 * stage_drive * stage_injection_weight
        + 0.08 * entry_kernel_drive * stage_injection_weight
        + 0.06 * manifold_energy_norm * stage_injection_weight
        + 0.04 * stage_drive * stage_post_cuda_weight
        + 0.08 * float(trace_axis_vector[2])
        + 0.06 * trace_flux
        + 0.04 * trace_temporal_overlap
    )

    amplitude_observable = clamp01(
        0.44 * amplitude_driver
        + 0.26 * float(base_dof_vector[1])
        + 0.12 * activity_norm
        + 0.10 * calibration_readiness
        + 0.08 * motif_energy
    )
    frequency_observable = clamp01(
        0.44 * frequency_driver
        + 0.26 * float(base_dof_vector[0])
        + 0.12 * activity_norm
        + 0.10 * calibration_readiness
        + 0.08 * search_volume_norm
    )
    voltage_observable = clamp01(
        0.44 * voltage_driver
        + 0.26 * float(base_dof_vector[2])
        + 0.12 * field_pressure
        + 0.10 * field_alignment_score
        + 0.08 * target_gate
    )
    current_observable = clamp01(
        0.44 * current_driver
        + 0.26 * float(base_dof_vector[3])
        + 0.12 * activity_norm
        + 0.10 * delta_memory_mean
        + 0.08 * kernel_control_gate
    )

    util_gpu_norm = clamp01(
        0.46 * activity_norm
        + 0.24 * vector_alignment_mean
        + 0.18 * target_alignment_mean
        + 0.12 * field_alignment_score
    )
    util_mem_norm = clamp01(
        0.38 * sequence_mean
        + 0.24 * overlap_mean
        + 0.18 * delta_memory_mean
        + 0.10 * motif_consistency
        + 0.10 * temporal_weight_mean
    )
    graphics_clock_norm = clamp01(
        0.48 * frequency_observable
        + 0.18 * field_frequency_bias
        + 0.18 * target_alignment_mean
        + 0.16 * search_volume_norm
    )
    sm_clock_norm = clamp01(
        0.42 * frequency_observable
        + 0.20 * coherence_mean
        + 0.18 * kernel_drive_mean
        + 0.10 * harmonic_mean
        + 0.10 * vector_alignment_mean
    )
    power_norm = clamp01(
        0.34 * amplitude_observable
        + 0.22 * current_observable
        + 0.20 * voltage_observable
        + 0.14 * interference_mean
        + 0.10 * field_pressure
    )
    temperature_norm = clamp01(
        0.24 * phase_pressure_mean
        + 0.18 * (1.0 - basin_alignment_mean)
        + 0.16 * (1.0 - field_alignment_score)
        + 0.14 * delta_gate_mean
        + 0.14 * (1.0 - motif_stability)
        + 0.14 * field_pressure
    )
    environment_pressure = clamp01(
        0.28 * phase_pressure_mean
        + 0.18 * crosstalk_mean
        + 0.14 * (1.0 - target_alignment_mean)
        + 0.14 * (1.0 - basin_alignment_mean)
        + 0.14 * delta_flux_mean
        + 0.12 * (1.0 - field_alignment_score)
    )
    environment_stability = clamp01(
        0.24 * calibration_readiness
        + 0.20 * motif_consistency
        + 0.18 * motif_stability
        + 0.14 * field_alignment_score
        + 0.10 * target_alignment_mean
        + 0.08 * (1.0 - crosstalk_mean)
        + 0.06 * sequence_mean
    )
    pstate_norm = clamp01(0.60 * environment_stability + 0.40 * (1.0 - temperature_norm))
    fan_speed_norm = clamp01(
        0.52 * temperature_norm
        + 0.28 * power_norm
        + 0.20 * (1.0 - environment_stability)
    )

    graphics_clock_mhz = 300.0 + 1500.0 * graphics_clock_norm
    sm_clock_mhz = 300.0 + 1500.0 * sm_clock_norm
    power_limit_w = 80.0
    power_draw_w = max(5.0, power_limit_w * power_norm)
    temperature_c = 25.0 + 55.0 * temperature_norm
    fan_speed_pct = 100.0 * fan_speed_norm

    return {
        "source": "vector_runtime_feedback",
        "feedback_stage": feedback_stage,
        "base_dof_vector": [float(value) for value in base_dof_vector],
        "base_dof_tensor": base_dof_tensor.tolist(),
        "projection_tensor": projection_tensor.tolist(),
        "amplitude_observable": float(amplitude_observable),
        "frequency_observable": float(frequency_observable),
        "voltage_observable": float(voltage_observable),
        "current_observable": float(current_observable),
        "utilization_gpu_norm": float(util_gpu_norm),
        "utilization_mem_norm": float(util_mem_norm),
        "graphics_clock_norm": float(graphics_clock_norm),
        "sm_clock_norm": float(sm_clock_norm),
        "power_norm": float(power_norm),
        "power_headroom_norm": float(clamp01(1.0 - power_norm)),
        "temperature_norm": float(temperature_norm),
        "temperature_c": float(temperature_c),
        "fan_speed_norm": float(fan_speed_norm),
        "fan_speed_pct": float(fan_speed_pct),
        "pstate_norm": float(pstate_norm),
        "environment_pressure": float(environment_pressure),
        "environment_stability": float(environment_stability),
        "power_draw_w": float(power_draw_w),
        "power_limit_w": float(power_limit_w),
        "graphics_clock_mhz": float(graphics_clock_mhz),
        "sm_clock_mhz": float(sm_clock_mhz),
        "activity_norm": float(activity_norm),
        "stage_drive": float(stage_drive),
        "manifold_coherence_norm": float(manifold_coherence_norm),
        "manifold_phase_transport": float(manifold_phase_transport),
        "manifold_energy_norm": float(manifold_energy_norm),
        "effective_spatial_magnitude": float(effective_spatial_magnitude),
        "effective_temporal_projection": float(effective_temporal_projection),
        "entry_calibration_readiness": float(entry_calibration_readiness),
        "entry_target_gate": float(entry_target_gate),
        "entry_field_alignment": float(entry_field_alignment),
        "entry_kernel_drive": float(entry_kernel_drive),
        "trace_support": float(trace_support),
        "trace_resonance": float(trace_resonance),
        "trace_alignment": float(trace_alignment),
        "trace_memory": float(trace_memory),
        "trace_flux": float(trace_flux),
        "trace_stability": float(trace_stability),
        "trace_temporal_persistence": float(trace_temporal_persistence),
        "trace_temporal_overlap": float(trace_temporal_overlap),
        "trace_voltage_frequency_flux": float(trace_voltage_frequency_flux),
        "trace_frequency_voltage_flux": float(trace_frequency_voltage_flux),
        "trace_axis_vector": [float(value) for value in trace_axis_vector],
        "candidate_count": int(candidate_count),
        "kernel_delta_phase_alignment_mean": float(delta_phase_alignment_mean),
    }


def sample_gpu_pulse_feedback(
    pulse_index: int,
    previous_feedback: dict[str, Any] | None = None,
    feedback_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_started = time.perf_counter()

    previous_feedback = dict(previous_feedback or {})
    feedback_context = dict(feedback_context or {})
    previous_amplitude = clamp01(float(previous_feedback.get("amplitude_observable", 0.5)))
    previous_frequency = clamp01(float(previous_feedback.get("frequency_observable", 0.5)))
    previous_voltage = clamp01(float(previous_feedback.get("voltage_observable", 0.5)))
    previous_current = clamp01(float(previous_feedback.get("current_observable", 0.5)))
    previous_memory = clamp01(float(previous_feedback.get("memory_proxy", 0.0)))
    previous_flux = clamp01(float(previous_feedback.get("flux_proxy", 0.0)))
    previous_phase_anchor = wrap_turns(float(previous_feedback.get("phase_anchor_turns", 0.0)))
    context_source = str(feedback_context.get("source", "vector_runtime_feedback"))
    base_dof_vector = np.array(
        list(
            feedback_context.get(
                "base_dof_vector",
                previous_feedback.get("feedback_dof_vector", [0.0] * 10),
            )
            or [0.0] * 10
        ),
        dtype=np.float64,
    )
    if base_dof_vector.shape[0] != 10:
        base_dof_vector = np.zeros(10, dtype=np.float64)
    base_dof_tensor = np.array(
        feedback_context.get(
            "base_dof_tensor",
            np.zeros((10, 10), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if base_dof_tensor.shape != (10, 10):
        base_dof_tensor = np.zeros((10, 10), dtype=np.float64)
    projection_tensor = gpu_feedback_projection_tensor_from_rows(
        feedback_context.get("projection_tensor")
    )

    util_gpu_norm = clamp01(
        float(
            feedback_context.get(
                "utilization_gpu_norm",
                previous_feedback.get("utilization_gpu_norm", 0.0),
            )
        )
    )
    util_mem_norm = clamp01(
        float(
            feedback_context.get(
                "utilization_mem_norm",
                previous_feedback.get("utilization_mem_norm", 0.0),
            )
        )
    )
    graphics_clock_norm = clamp01(
        float(
            feedback_context.get(
                "graphics_clock_norm",
                0.55 * util_gpu_norm + 0.45 * previous_frequency,
            )
        )
    )
    sm_clock_norm = clamp01(
        float(
            feedback_context.get(
                "sm_clock_norm",
                0.55 * util_gpu_norm + 0.45 * previous_amplitude,
            )
        )
    )
    power_norm = clamp01(
        float(
            feedback_context.get(
                "power_norm",
                0.40 * util_gpu_norm + 0.30 * util_mem_norm,
            )
        )
    )
    power_headroom_norm = clamp01(
        float(
            feedback_context.get(
                "power_headroom_norm",
                1.0 - power_norm,
            )
        )
    )
    temperature_norm = clamp01(
        float(
            feedback_context.get(
                "temperature_norm",
                0.30 * power_norm + 0.20 * (1.0 - power_headroom_norm),
            )
        )
    )
    temperature_c = float(
        feedback_context.get("temperature_c", 25.0 + 55.0 * temperature_norm)
    )
    fan_speed_norm = clamp01(
        float(
            feedback_context.get(
                "fan_speed_norm",
                0.44 * temperature_norm
                + 0.32 * power_norm
                + 0.24 * util_gpu_norm,
            )
        )
    )
    fan_speed_pct = float(
        feedback_context.get("fan_speed_pct", 100.0 * fan_speed_norm)
    )
    pstate_norm = clamp01(
        float(
            feedback_context.get(
                "pstate_norm",
                0.5 * (1.0 - temperature_norm) + 0.5 * power_headroom_norm,
            )
        )
    )
    thermal_headroom_norm = clamp01(
        float(
            feedback_context.get(
                "thermal_headroom_norm",
                1.0 - temperature_norm,
            )
        )
    )
    previous_temperature_c = float(previous_feedback.get("temperature_c", temperature_c))
    temperature_delta_c = float(temperature_c - previous_temperature_c)
    temperature_velocity_norm = clamp01(abs(temperature_delta_c) / 8.0)
    amplitude_observable = clamp01(
        float(
            feedback_context.get(
                "amplitude_observable",
                0.30 * util_gpu_norm
                + 0.18 * util_mem_norm
                + 0.18 * power_norm
                + 0.14 * sm_clock_norm
                + 0.10 * graphics_clock_norm
                + 0.10 * pstate_norm,
            )
        )
    )
    frequency_observable = clamp01(
        float(
            feedback_context.get(
                "frequency_observable",
                0.52 * sm_clock_norm
                + 0.20 * graphics_clock_norm
                + 0.14 * util_gpu_norm
                + 0.08 * power_norm
                + 0.06 * pstate_norm,
            )
        )
    )
    voltage_observable = clamp01(
        float(
            feedback_context.get(
                "voltage_observable",
                0.34 * power_norm
                + 0.22 * graphics_clock_norm
                + 0.16 * sm_clock_norm
                + 0.14 * temperature_norm
                + 0.14 * pstate_norm,
            )
        )
    )
    current_observable = clamp01(
        float(
            feedback_context.get(
                "current_observable",
                0.34 * util_gpu_norm
                + 0.18 * util_mem_norm
                + 0.20 * power_norm
                + 0.16 * max(power_norm - 0.5 * voltage_observable, 0.0)
                + 0.12 * sm_clock_norm,
            )
        )
    )
    eps = 1.0e-6
    dln_amplitude = float(math.log(max(amplitude_observable, eps) / max(previous_amplitude, eps)))
    dln_frequency = float(math.log(max(frequency_observable, eps) / max(previous_frequency, eps)))
    dln_voltage = float(math.log(max(voltage_observable, eps) / max(previous_voltage, eps)))
    dln_current = float(math.log(max(current_observable, eps) / max(previous_current, eps)))
    ddln_amplitude = dln_amplitude - float(previous_feedback.get("dln_amplitude", 0.0))
    ddln_frequency = dln_frequency - float(previous_feedback.get("dln_frequency", 0.0))
    ddln_voltage = dln_voltage - float(previous_feedback.get("dln_voltage", 0.0))
    ddln_current = dln_current - float(previous_feedback.get("dln_current", 0.0))

    amplitude_ref = max(
        float(previous_feedback.get("amplitude_ref", previous_amplitude)),
        0.25,
    )
    frequency_ref = max(
        float(previous_feedback.get("frequency_ref", previous_frequency)),
        0.25,
    )
    voltage_ref = max(
        float(previous_feedback.get("voltage_ref", previous_voltage)),
        0.25,
    )
    current_ref = max(
        float(previous_feedback.get("current_ref", previous_current)),
        0.25,
    )
    phase_anchor_turns = wrap_turns(
        0.52 * previous_phase_anchor
        + 0.17
        + 0.22 * math.log(max(amplitude_observable, eps) / amplitude_ref)
        + 0.11 * math.log(max(voltage_observable, eps) / voltage_ref)
        + 0.13 * math.log(max(frequency_observable, eps) / frequency_ref)
        + 0.09 * math.log(max(current_observable, eps) / current_ref)
        + pulse_index * 0.03125
    )
    phase_anchor_rad = float(phase_anchor_turns * math.tau)
    phase_alignment = turn_alignment(phase_anchor_turns, previous_phase_anchor)
    flux_proxy = clamp01(
        0.22 * util_mem_norm
        + 0.18 * power_norm
        + 0.14 * temperature_norm
        + 0.14 * clamp01(abs(dln_frequency) * 2.5)
        + 0.12 * clamp01(abs(dln_voltage) * 2.5)
        + 0.10 * clamp01(abs(dln_current) * 2.5)
        + 0.10 * previous_flux
    )
    stability_proxy = clamp01(
        1.0
        - min(
            1.0,
            0.28 * abs(dln_amplitude)
            + 0.24 * abs(dln_frequency)
            + 0.24 * abs(dln_voltage)
            + 0.24 * abs(dln_current),
        )
    )
    memory_proxy = clamp01(
        0.34 * stability_proxy
        + 0.26 * previous_memory
        + 0.16 * util_mem_norm
        + 0.12 * pstate_norm
        + 0.12 * phase_alignment
    )
    vf_flux_proxy = clamp01(
        float(
            feedback_context.get(
                "vf_flux_proxy",
                0.42 * voltage_observable
                + 0.38 * frequency_observable
                + 0.12 * clamp01(abs(dln_voltage) * 2.0)
                + 0.08 * clamp01(abs(dln_frequency) * 2.0),
            )
        )
    )
    environment_pressure = clamp01(
        float(
            feedback_context.get(
                "environment_pressure",
                0.26 * temperature_norm
                + 0.18 * util_mem_norm
                + 0.16 * power_norm
                + 0.12 * temperature_velocity_norm
                + 0.10 * (1.0 - pstate_norm)
                + 0.10 * (1.0 - thermal_headroom_norm)
                + 0.08 * fan_speed_norm,
            )
        )
    )
    environment_stability = clamp01(
        float(
            feedback_context.get(
                "environment_stability",
                1.0
                - min(
                    1.0,
                    0.28 * temperature_velocity_norm
                    + 0.20 * abs(dln_voltage)
                    + 0.18 * abs(dln_frequency)
                    + 0.18 * abs(dln_current)
                    + 0.16 * abs(dln_amplitude),
                ),
            )
        )
    )
    sampling_latency_ms = float((time.perf_counter() - sample_started) * 1000.0)
    previous_sampling_latency_ms = float(
        previous_feedback.get("sampling_latency_ms", sampling_latency_ms)
    )
    latency_jitter_ms = abs(sampling_latency_ms - previous_sampling_latency_ms)
    latency_norm = clamp01(sampling_latency_ms / 50.0)
    latency_jitter_norm = clamp01(latency_jitter_ms / 25.0)
    temporal_latency_gate = clamp01(
        0.42 * (1.0 - latency_norm)
        + 0.20 * (1.0 - latency_jitter_norm)
        + 0.18 * environment_stability
        + 0.12 * thermal_headroom_norm
        + 0.08 * phase_alignment
    )
    temporal_drive = clamp01(
        0.24 * memory_proxy
        + 0.18 * phase_alignment
        + 0.14 * stability_proxy
        + 0.12 * vf_flux_proxy
        + 0.10 * clamp01(1.0 - abs(ddln_frequency) - abs(ddln_voltage) * 0.5)
        + 0.10 * environment_stability
        + 0.07 * temporal_latency_gate
        + 0.05 * thermal_headroom_norm
    )
    power_draw_w = float(
        feedback_context.get("power_draw_w", max(5.0, 80.0 * power_norm))
    )
    power_limit_w = max(
        float(feedback_context.get("power_limit_w", 80.0)),
        power_draw_w,
        1.0,
    )
    graphics_clock_mhz = float(
        feedback_context.get("graphics_clock_mhz", 300.0 + 1500.0 * graphics_clock_norm)
    )
    sm_clock_mhz = float(
        feedback_context.get("sm_clock_mhz", 300.0 + 1500.0 * sm_clock_norm)
    )

    dynamic_dof_vector = np.array(
        [
            frequency_observable,
            amplitude_observable,
            voltage_observable,
            current_observable,
            clamp01(
                0.60 * math.sqrt(max(frequency_observable * amplitude_observable, 0.0))
                + 0.16 * phase_alignment
                + 0.14 * memory_proxy
                + 0.10 * stability_proxy
            ),
            clamp01(
                0.58 * math.sqrt(max(frequency_observable * voltage_observable, 0.0))
                + 0.18 * vf_flux_proxy
                + 0.14 * phase_alignment
                + 0.10 * flux_proxy
            ),
            clamp01(
                0.58 * math.sqrt(max(frequency_observable * current_observable, 0.0))
                + 0.18 * flux_proxy
                + 0.14 * stability_proxy
                + 0.10 * clamp01(abs(dln_current) * 2.0)
            ),
            clamp01(
                0.58 * math.sqrt(max(amplitude_observable * voltage_observable, 0.0))
                + 0.16 * memory_proxy
                + 0.14 * phase_alignment
                + 0.12 * clamp01(abs(dln_voltage) * 2.0)
            ),
            clamp01(
                0.58 * math.sqrt(max(amplitude_observable * current_observable, 0.0))
                + 0.18 * memory_proxy
                + 0.12 * stability_proxy
                + 0.12 * clamp01(abs(dln_current) * 2.0)
            ),
            clamp01(
                0.56 * math.sqrt(max(voltage_observable * current_observable, 0.0))
                + 0.20 * vf_flux_proxy
                + 0.12 * flux_proxy
                + 0.12 * phase_alignment
            ),
        ],
        dtype=np.float64,
    )
    feedback_dof_vector = np.clip(
        0.52 * base_dof_vector + 0.48 * dynamic_dof_vector,
        0.0,
        1.0,
    )
    dynamic_dof_tensor = np.eye(10, dtype=np.float64) * 0.58
    dynamic_dof_tensor += np.diag(feedback_dof_vector * 0.42)
    pair_to_primary = {
        4: (0, 1),
        5: (0, 2),
        6: (0, 3),
        7: (1, 2),
        8: (1, 3),
        9: (2, 3),
    }
    for pair_idx, (lhs_idx, rhs_idx) in pair_to_primary.items():
        coupling_value = float(feedback_dof_vector[pair_idx])
        for primary_idx in (lhs_idx, rhs_idx):
            edge_value = 0.09 + 0.24 * coupling_value
            dynamic_dof_tensor[pair_idx, primary_idx] += edge_value
            dynamic_dof_tensor[primary_idx, pair_idx] += edge_value
        primary_edge = 0.05 + 0.12 * coupling_value
        dynamic_dof_tensor[lhs_idx, rhs_idx] += primary_edge
        dynamic_dof_tensor[rhs_idx, lhs_idx] += primary_edge
    dynamic_dof_tensor += np.outer(feedback_dof_vector, feedback_dof_vector) * 0.04
    dynamic_dof_tensor = np.clip(dynamic_dof_tensor, 0.0, 1.45)
    feedback_dof_tensor = np.clip(
        0.52 * base_dof_tensor + 0.48 * dynamic_dof_tensor,
        0.0,
        1.45,
    )
    feedback_axis_vector = projection_tensor @ feedback_dof_vector
    feedback_axis_tensor = projection_tensor @ feedback_dof_tensor @ projection_tensor.T
    feedback_axis_tensor = np.clip(feedback_axis_tensor, 0.0, 1.45)
    envelope_history = list(previous_feedback.get("envelope_history", []) or [])
    envelope_history.append(
        {
            "pulse_index": int(pulse_index),
            "amplitude": float(amplitude_observable),
            "frequency": float(frequency_observable),
            "voltage": float(voltage_observable),
            "current": float(current_observable),
            "phase_anchor_turns": float(phase_anchor_turns),
            "phase_alignment": float(phase_alignment),
            "memory_proxy": float(memory_proxy),
            "flux_proxy": float(flux_proxy),
            "temperature_norm": float(temperature_norm),
            "environment_pressure": float(environment_pressure),
            "latency_norm": float(latency_norm),
        }
    )
    envelope_history = envelope_history[-8:]
    return {
        "source": context_source,
        "pulse_index": int(pulse_index),
        "tau_delta_ticks": 1,
        "amplitude_observable": float(amplitude_observable),
        "frequency_observable": float(frequency_observable),
        "voltage_observable": float(voltage_observable),
        "current_observable": float(current_observable),
        "amplitude_ref": float(amplitude_ref),
        "frequency_ref": float(frequency_ref),
        "voltage_ref": float(voltage_ref),
        "current_ref": float(current_ref),
        "dln_amplitude": float(dln_amplitude),
        "dln_frequency": float(dln_frequency),
        "dln_voltage": float(dln_voltage),
        "dln_current": float(dln_current),
        "ddln_amplitude": float(ddln_amplitude),
        "ddln_frequency": float(ddln_frequency),
        "ddln_voltage": float(ddln_voltage),
        "ddln_current": float(ddln_current),
        "phase_anchor_turns": float(phase_anchor_turns),
        "phase_anchor_rad": float(phase_anchor_rad),
        "phase_alignment": float(phase_alignment),
        "flux_proxy": float(flux_proxy),
        "memory_proxy": float(memory_proxy),
        "stability_proxy": float(stability_proxy),
        "vf_flux_proxy": float(vf_flux_proxy),
        "temporal_drive": float(temporal_drive),
        "feedback_dof_vector": [float(value) for value in feedback_dof_vector],
        "feedback_dof_tensor": feedback_dof_tensor.tolist(),
        "feedback_axis_vector": [float(value) for value in feedback_axis_vector],
        "feedback_axis_tensor": feedback_axis_tensor.tolist(),
        "utilization_gpu_norm": float(util_gpu_norm),
        "utilization_mem_norm": float(util_mem_norm),
        "graphics_clock_norm": float(graphics_clock_norm),
        "sm_clock_norm": float(sm_clock_norm),
        "power_norm": float(power_norm),
        "power_headroom_norm": float(power_headroom_norm),
        "temperature_norm": float(temperature_norm),
        "temperature_c": float(temperature_c),
        "temperature_delta_c": float(temperature_delta_c),
        "temperature_velocity_norm": float(temperature_velocity_norm),
        "thermal_headroom_norm": float(thermal_headroom_norm),
        "fan_speed_pct": float(fan_speed_pct),
        "fan_speed_norm": float(fan_speed_norm),
        "pstate_norm": float(pstate_norm),
        "environment_pressure": float(environment_pressure),
        "environment_stability": float(environment_stability),
        "sampling_latency_ms": float(sampling_latency_ms),
        "latency_norm": float(latency_norm),
        "latency_jitter_ms": float(latency_jitter_ms),
        "latency_jitter_norm": float(latency_jitter_norm),
        "temporal_latency_gate": float(temporal_latency_gate),
        "power_draw_w": float(power_draw_w),
        "power_limit_w": float(power_limit_w),
        "graphics_clock_mhz": float(graphics_clock_mhz),
        "sm_clock_mhz": float(sm_clock_mhz),
        "candidate_count": int(feedback_context.get("candidate_count", 0)),
        "activity_norm": float(feedback_context.get("activity_norm", 0.0)),
        "ancilla_commit_ratio": float(feedback_context.get("ancilla_commit_ratio", 0.0)),
        "ancilla_convergence": float(feedback_context.get("ancilla_convergence", 0.0)),
        "ancilla_flux": float(feedback_context.get("ancilla_flux", 0.0)),
        "ancilla_phase_alignment": float(feedback_context.get("ancilla_phase_alignment", 0.0)),
        "ancilla_current_norm": float(feedback_context.get("ancilla_current_norm", 0.0)),
        "envelope_history": envelope_history,
    }


def build_gpu_pulse_delta_feedback(
    pulse_index: int,
    pre_feedback: dict[str, Any] | None,
    post_feedback: dict[str, Any] | None,
    cuda_kernel_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pre_feedback = dict(pre_feedback or {})
    post_feedback = dict(post_feedback or {})
    cuda_kernel_telemetry = dict(cuda_kernel_telemetry or {})
    pre_axis_vector = np.array(
        list(pre_feedback.get("feedback_axis_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    post_axis_vector = np.array(
        list(post_feedback.get("feedback_axis_vector", []) or pre_axis_vector.tolist()),
        dtype=np.float64,
    )
    if pre_axis_vector.shape[0] != 4:
        pre_axis_vector = np.zeros(4, dtype=np.float64)
    if post_axis_vector.shape[0] != 4:
        post_axis_vector = np.array(pre_axis_vector, dtype=np.float64)
    pre_dof_vector = np.array(
        list(pre_feedback.get("feedback_dof_vector", []) or [0.0] * 10),
        dtype=np.float64,
    )
    post_dof_vector = np.array(
        list(post_feedback.get("feedback_dof_vector", []) or pre_dof_vector.tolist()),
        dtype=np.float64,
    )
    if pre_dof_vector.shape[0] != 10:
        pre_dof_vector = np.zeros(10, dtype=np.float64)
    if post_dof_vector.shape[0] != 10:
        post_dof_vector = np.array(pre_dof_vector, dtype=np.float64)

    amplitude_delta = float(post_feedback.get("amplitude_observable", 0.0)) - float(
        pre_feedback.get("amplitude_observable", 0.0)
    )
    frequency_delta = float(post_feedback.get("frequency_observable", 0.0)) - float(
        pre_feedback.get("frequency_observable", 0.0)
    )
    voltage_delta = float(post_feedback.get("voltage_observable", 0.0)) - float(
        pre_feedback.get("voltage_observable", 0.0)
    )
    current_delta = float(post_feedback.get("current_observable", 0.0)) - float(
        pre_feedback.get("current_observable", 0.0)
    )
    phase_shift_turns = signed_turn_delta(
        float(post_feedback.get("phase_anchor_turns", 0.0)),
        float(pre_feedback.get("phase_anchor_turns", 0.0)),
    )
    phase_retention = turn_alignment(
        float(post_feedback.get("phase_anchor_turns", 0.0)),
        float(pre_feedback.get("phase_anchor_turns", 0.0)),
    )
    axis_delta_vector = post_axis_vector - pre_axis_vector
    dof_delta_vector = post_dof_vector - pre_dof_vector
    axis_delta_energy = clamp01(float(np.linalg.norm(axis_delta_vector)) / 1.20)
    dof_delta_energy = clamp01(float(np.linalg.norm(dof_delta_vector)) / 2.80)
    pre_environment_pressure = clamp01(float(pre_feedback.get("environment_pressure", 0.0)))
    post_environment_pressure = clamp01(float(post_feedback.get("environment_pressure", 0.0)))
    pre_environment_stability = clamp01(float(pre_feedback.get("environment_stability", 0.0)))
    post_environment_stability = clamp01(float(post_feedback.get("environment_stability", 0.0)))
    post_thermal_headroom = clamp01(float(post_feedback.get("thermal_headroom_norm", 0.0)))
    post_temperature_velocity = clamp01(float(post_feedback.get("temperature_velocity_norm", 0.0)))
    pre_roundtrip_latency_ms = float(
        pre_feedback.get(
            "pulse_roundtrip_latency_ms",
            pre_feedback.get("sampling_latency_ms", 0.0),
        )
    )
    post_roundtrip_latency_ms = float(
        post_feedback.get(
            "pulse_roundtrip_latency_ms",
            post_feedback.get("sampling_latency_ms", 0.0),
        )
    )
    dispatch_latency_ms = float(cuda_kernel_telemetry.get("dispatch_latency_ms", 0.0))
    if post_roundtrip_latency_ms <= 0.0:
        post_roundtrip_latency_ms = float(post_feedback.get("sampling_latency_ms", 0.0))
    latency_delta_ms = float(post_roundtrip_latency_ms - pre_roundtrip_latency_ms)
    feedback_window_ms = float(
        post_feedback.get("pulse_feedback_window_ms", post_roundtrip_latency_ms)
    )
    observation_gap_ms = float(
        max(
            dispatch_latency_ms + float(post_feedback.get("sampling_latency_ms", 0.0)),
            post_roundtrip_latency_ms,
            feedback_window_ms,
            0.0,
        )
    )
    latency_norm = clamp01(post_roundtrip_latency_ms / 32.0)
    latency_jitter_norm = clamp01(abs(latency_delta_ms) / 16.0)
    window_span_ms = float(max(feedback_window_ms, post_roundtrip_latency_ms, 0.0))
    window_span_norm = clamp01(window_span_ms / 24.0)
    observation_gap_norm = clamp01(math.log10(1.0 + observation_gap_ms) / 4.0)
    observation_freshness_gate = clamp01(
        1.0 / (1.0 + math.log10(1.0 + observation_gap_ms))
    )
    dispatch_feedback_ratio = float(
        dispatch_latency_ms / max(feedback_window_ms, 1.0e-6)
    )
    search_volume_norm = clamp01(
        (float(cuda_kernel_telemetry.get("search_volume_gain", 1.0)) - 1.0) / 2.5
    )
    expanded_eval_norm = clamp01(
        float(cuda_kernel_telemetry.get("expanded_eval_count", 0)) / 131072.0
    )
    expanded_keep_norm = clamp01(
        float(cuda_kernel_telemetry.get("expanded_keep_count", 0)) / 32768.0
    )
    window_calibration_steps = max(
        1,
        min(
            16,
            int(round(1.0 + 5.0 * search_volume_norm + 4.0 * expanded_keep_norm)),
        ),
    )
    window_density = clamp01(
        0.42 * expanded_eval_norm
        + 0.34 * expanded_keep_norm
        + 0.24 * search_volume_norm
    )
    activity_norm = clamp01(
        0.34 * float(cuda_kernel_telemetry.get("score_mean", 0.0))
        + 0.24 * expanded_eval_norm
        + 0.20 * expanded_keep_norm
        + 0.22 * clamp01(float(cuda_kernel_telemetry.get("blocks_per_grid", 0)) / 512.0)
    )
    response_energy = clamp01(
        0.42 * dof_delta_energy
        + 0.28 * axis_delta_energy
        + 0.14 * latency_norm
        + 0.10 * post_environment_pressure
        + 0.06 * post_temperature_velocity
    )
    response_stability = clamp01(
        1.0
        - min(
            1.0,
            0.24 * abs(amplitude_delta) * 3.0
            + 0.24 * abs(frequency_delta) * 3.0
            + 0.22 * abs(voltage_delta) * 3.0
            + 0.18 * abs(current_delta) * 3.0
            + 0.12 * abs(phase_shift_turns) * 4.0,
        )
    )
    response_stability = clamp01(
        0.78 * response_stability
        + 0.12 * post_environment_stability
        + 0.10 * (1.0 - latency_jitter_norm)
    )
    response_stability = clamp01(
        0.72 * response_stability
        + 0.18 * observation_freshness_gate
        + 0.10 * (1.0 - observation_gap_norm)
    )
    memory_retention = clamp01(
        0.34 * float(post_feedback.get("memory_proxy", 0.0))
        + 0.22 * phase_retention
        + 0.16 * float(post_feedback.get("stability_proxy", 0.0))
        + 0.14 * response_stability
        + 0.14 * (1.0 - axis_delta_energy)
        + 0.08 * observation_freshness_gate
    )
    latency_gate = clamp01(
        0.34 * (1.0 - latency_norm)
        + 0.20 * (1.0 - latency_jitter_norm)
        + 0.16 * response_stability
        + 0.16 * post_environment_stability
        + 0.14 * post_thermal_headroom
        + 0.12 * observation_freshness_gate
    )
    window_latency_alignment = clamp01(
        0.34 * latency_gate
        + 0.22 * (1.0 - window_span_norm)
        + 0.18 * response_stability
        + 0.14 * activity_norm
        + 0.12 * post_environment_stability
        + 0.12 * observation_freshness_gate
    )
    sequence_persistence_target = clamp01(
        0.30 * memory_retention
        + 0.18 * float(post_feedback.get("temporal_drive", 0.0))
        + 0.18 * phase_retention
        + 0.12 * response_stability
        + 0.12 * activity_norm
        + 0.10 * float(post_feedback.get("vf_flux_proxy", 0.0))
        + 0.10 * latency_gate
        + 0.06 * post_thermal_headroom
    )
    temporal_overlap_target = clamp01(
        0.28 * phase_retention
        + 0.22 * memory_retention
        + 0.18 * float(post_feedback.get("temporal_drive", 0.0))
        + 0.12 * response_stability
        + 0.10 * activity_norm
        + 0.10 * (1.0 - axis_delta_energy)
        + 0.08 * latency_gate
        + 0.06 * (1.0 - latency_jitter_norm)
    )
    voltage_frequency_flux_target = clamp01(
        0.34 * float(post_feedback.get("vf_flux_proxy", 0.0))
        + 0.18 * abs(voltage_delta) * 3.0
        + 0.16 * abs(frequency_delta) * 3.0
        + 0.14 * response_energy
        + 0.10 * activity_norm
        + 0.08 * float(post_feedback.get("voltage_observable", 0.0))
        + 0.08 * post_environment_pressure
    )
    frequency_voltage_flux_target = clamp01(
        0.30 * float(post_feedback.get("vf_flux_proxy", 0.0))
        + 0.18 * float(post_feedback.get("frequency_observable", 0.0))
        + 0.16 * abs(phase_shift_turns) * 4.0
        + 0.14 * response_energy
        + 0.12 * activity_norm
        + 0.10 * abs(frequency_delta) * 3.0
        + 0.08 * latency_gate
    )
    phase_pressure_target = clamp01(
        0.24 * abs(phase_shift_turns) * 4.0
        + 0.20 * response_energy
        + 0.16 * abs(amplitude_delta) * 3.0
        + 0.12 * abs(voltage_delta) * 3.0
        + 0.10 * activity_norm
        + 0.10 * (1.0 - phase_retention)
        + 0.08 * float(post_feedback.get("temporal_drive", 0.0))
        + 0.06 * post_environment_pressure
        + 0.04 * latency_norm
    )
    response_gate = clamp01(
        0.26 * response_energy
        + 0.22 * response_stability
        + 0.18 * memory_retention
        + 0.16 * phase_retention
        + 0.10 * activity_norm
        + 0.08 * float(post_feedback.get("temporal_drive", 0.0))
        + 0.08 * latency_gate
        + 0.06 * post_environment_stability
        + 0.08 * observation_freshness_gate
    )
    environment_pressure_target = clamp01(
        0.42 * post_environment_pressure
        + 0.18 * post_temperature_velocity
        + 0.14 * latency_norm
        + 0.14 * (1.0 - post_environment_stability)
        + 0.12 * float(post_feedback.get("temperature_norm", 0.0))
    )
    thermal_headroom_target = clamp01(
        0.54 * post_thermal_headroom
        + 0.16 * response_stability
        + 0.14 * latency_gate
        + 0.08 * float(post_feedback.get("fan_speed_norm", 0.0))
        + 0.08 * float(post_feedback.get("power_headroom_norm", 0.0))
    )
    delta_target_vector = np.array(
        [
            sequence_persistence_target,
            temporal_overlap_target,
            voltage_frequency_flux_target,
            frequency_voltage_flux_target,
        ],
        dtype=np.float64,
    )
    coherent_noise_field = build_coherent_noise_field(
        pre_feedback=pre_feedback,
        post_feedback=post_feedback,
        phase_retention=phase_retention,
        response_gate=response_gate,
        observation_freshness_gate=observation_freshness_gate,
    )
    return {
        "pulse_index": int(pulse_index),
        "phase_shift_turns": float(phase_shift_turns),
        "phase_retention": float(phase_retention),
        "amplitude_delta": float(amplitude_delta),
        "frequency_delta": float(frequency_delta),
        "voltage_delta": float(voltage_delta),
        "current_delta": float(current_delta),
        "axis_delta_vector": [float(value) for value in axis_delta_vector],
        "dof_delta_vector": [float(value) for value in dof_delta_vector],
        "axis_delta_energy": float(axis_delta_energy),
        "dof_delta_energy": float(dof_delta_energy),
        "response_energy": float(response_energy),
        "response_stability": float(response_stability),
        "memory_retention": float(memory_retention),
        "roundtrip_latency_ms": float(post_roundtrip_latency_ms),
        "feedback_window_ms": float(feedback_window_ms),
        "dispatch_latency_ms": float(dispatch_latency_ms),
        "observation_gap_ms": float(observation_gap_ms),
        "observation_gap_norm": float(observation_gap_norm),
        "observation_freshness_gate": float(observation_freshness_gate),
        "dispatch_feedback_ratio": float(dispatch_feedback_ratio),
        "latency_delta_ms": float(latency_delta_ms),
        "latency_norm": float(latency_norm),
        "latency_jitter_norm": float(latency_jitter_norm),
        "latency_gate": float(latency_gate),
        "window_span_ms": float(window_span_ms),
        "window_span_norm": float(window_span_norm),
        "window_calibration_steps": int(window_calibration_steps),
        "window_density": float(window_density),
        "window_latency_alignment": float(window_latency_alignment),
        "environment_pressure_target": float(environment_pressure_target),
        "thermal_headroom_target": float(thermal_headroom_target),
        "environment_delta": float(post_environment_pressure - pre_environment_pressure),
        "sequence_persistence_target": float(sequence_persistence_target),
        "temporal_overlap_target": float(temporal_overlap_target),
        "voltage_frequency_flux_target": float(voltage_frequency_flux_target),
        "frequency_voltage_flux_target": float(frequency_voltage_flux_target),
        "phase_pressure_target": float(phase_pressure_target),
        "response_gate": float(response_gate),
        "activity_norm": float(activity_norm),
        "delta_target_vector": [float(value) for value in delta_target_vector],
        "coherent_noise_axis_vector": list(
            coherent_noise_field.get("coherent_noise_axis_vector", [])
        ),
        "coherent_noise_dof_vector": list(
            coherent_noise_field.get("coherent_noise_dof_vector", [])
        ),
        "coherent_noise_tensor": list(
            coherent_noise_field.get("coherent_noise_tensor", [])
        ),
        "noise_resonance_nodes": list(
            coherent_noise_field.get("noise_resonance_nodes", [])
        ),
        "drift_compensation_vector": list(
            coherent_noise_field.get("drift_compensation_vector", [])
        ),
        "relative_spatial_field": list(
            coherent_noise_field.get("relative_spatial_field", [])
        ),
        "noise_resonance_gate": float(
            coherent_noise_field.get("noise_resonance_gate", 0.0)
        ),
        "noise_orbital_anchor_turns": float(
            coherent_noise_field.get("noise_orbital_anchor_turns", 0.0)
        ),
        "environment_turbulence": float(
            coherent_noise_field.get("environment_turbulence", 0.0)
        ),
        "pre_source": str(pre_feedback.get("source", "fallback")),
        "post_source": str(post_feedback.get("source", "fallback")),
    }


def integrate_gpu_feedback_into_field_state(
    simulation_field_state: dict[str, Any],
    gpu_feedback: dict[str, Any] | None,
    gpu_pulse_delta_feedback: dict[str, Any] | None,
    blend: float = 0.26,
) -> None:
    gpu_feedback = dict(gpu_feedback or {})
    gpu_pulse_delta_feedback = dict(gpu_pulse_delta_feedback or {})
    if not simulation_field_state:
        return
    blend = clamp01(float(blend))
    keep = 1.0 - blend
    delta_target_vector = list(
        gpu_pulse_delta_feedback.get("delta_target_vector", []) or [0.0, 0.0, 0.0, 0.0]
    )
    if len(delta_target_vector) != 4:
        delta_target_vector = [0.0, 0.0, 0.0, 0.0]

    simulation_field_state["gpu_pulse_feedback"] = dict(gpu_feedback)
    simulation_field_state["gpu_pulse_delta_feedback"] = dict(gpu_pulse_delta_feedback)
    simulation_field_state["feedback_delta_target_vector"] = [
        float(value) for value in delta_target_vector
    ]
    simulation_field_state["feedback_delta_phase_retention"] = float(
        gpu_pulse_delta_feedback.get("phase_retention", 0.0)
    )
    simulation_field_state["feedback_delta_response_gate"] = float(
        gpu_pulse_delta_feedback.get("response_gate", 0.0)
    )
    simulation_field_state["feedback_delta_response_energy"] = float(
        gpu_pulse_delta_feedback.get("response_energy", 0.0)
    )
    simulation_field_state["feedback_delta_memory_retention"] = float(
        gpu_pulse_delta_feedback.get("memory_retention", 0.0)
    )
    simulation_field_state["feedback_delta_latency_gate"] = float(
        gpu_pulse_delta_feedback.get("latency_gate", 0.0)
    )
    simulation_field_state["feedback_delta_window_span_norm"] = float(
        gpu_pulse_delta_feedback.get("window_span_norm", 0.0)
    )
    simulation_field_state["feedback_delta_window_density"] = float(
        gpu_pulse_delta_feedback.get("window_density", 0.0)
    )
    simulation_field_state["feedback_delta_window_latency_alignment"] = float(
        gpu_pulse_delta_feedback.get("window_latency_alignment", 0.0)
    )
    simulation_field_state["feedback_delta_environment_pressure"] = float(
        gpu_pulse_delta_feedback.get("environment_pressure_target", 0.0)
    )
    simulation_field_state["feedback_delta_thermal_headroom"] = float(
        gpu_pulse_delta_feedback.get("thermal_headroom_target", 0.0)
    )
    simulation_field_state["feedback_observation_gap_ms"] = float(
        gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)
    )
    simulation_field_state["feedback_observation_freshness_gate"] = float(
        gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)
    )
    simulation_field_state["feedback_dispatch_feedback_ratio"] = float(
        gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0)
    )
    coherent_noise_axis_vector = list(
        gpu_pulse_delta_feedback.get("coherent_noise_axis_vector", []) or [0.0, 0.0, 0.0, 0.0]
    )
    coherent_noise_dof_vector = list(
        gpu_pulse_delta_feedback.get("coherent_noise_dof_vector", []) or [0.0] * len(GPU_PULSE_DOF_LABELS)
    )
    coherent_noise_tensor = gpu_pulse_delta_feedback.get(
        "coherent_noise_tensor",
        np.zeros((len(GPU_PULSE_DOF_LABELS), len(GPU_PULSE_DOF_LABELS)), dtype=np.float64).tolist(),
    )
    noise_resonance_nodes = list(
        gpu_pulse_delta_feedback.get("noise_resonance_nodes", []) or [0.0, 0.0, 0.0, 0.0]
    )
    drift_compensation_vector = list(
        gpu_pulse_delta_feedback.get("drift_compensation_vector", []) or [0.0, 0.0, 0.0, 0.0]
    )
    relative_spatial_field = list(
        gpu_pulse_delta_feedback.get("relative_spatial_field", []) or [0.0, 0.0, 0.0, 0.0]
    )
    simulation_field_state["coherent_noise_axis_vector"] = [
        float(value) for value in coherent_noise_axis_vector[:4]
    ]
    simulation_field_state["coherent_noise_dof_vector"] = [
        float(value) for value in coherent_noise_dof_vector[: len(GPU_PULSE_DOF_LABELS)]
    ]
    simulation_field_state["coherent_noise_tensor"] = coherent_noise_tensor
    simulation_field_state["noise_resonance_nodes"] = [
        float(value) for value in noise_resonance_nodes[:4]
    ]
    simulation_field_state["drift_compensation_vector"] = [
        float(value) for value in drift_compensation_vector[:4]
    ]
    simulation_field_state["relative_spatial_field"] = [
        float(value) for value in relative_spatial_field[:4]
    ]
    simulation_field_state["noise_resonance_gate"] = float(
        gpu_pulse_delta_feedback.get("noise_resonance_gate", 0.0)
    )
    simulation_field_state["noise_orbital_anchor_turns"] = float(
        gpu_pulse_delta_feedback.get("noise_orbital_anchor_turns", 0.0)
    )
    simulation_field_state["environment_turbulence"] = float(
        gpu_pulse_delta_feedback.get("environment_turbulence", 0.0)
    )

    simulation_field_state["sequence_persistence_score"] = float(
        clamp01(
            keep * float(simulation_field_state.get("sequence_persistence_score", 0.0))
            + blend
            * (
                0.90 * float(gpu_pulse_delta_feedback.get("sequence_persistence_target", 0.0))
                + 0.10 * float(gpu_pulse_delta_feedback.get("noise_resonance_gate", 0.0))
            )
        )
    )
    simulation_field_state["temporal_index_overlap"] = float(
        clamp01(
            keep * float(simulation_field_state.get("temporal_index_overlap", 0.0))
            + blend
            * (
                0.90 * float(gpu_pulse_delta_feedback.get("temporal_overlap_target", 0.0))
                + 0.10
                * float(
                    np.mean(
                        np.array(
                            gpu_pulse_delta_feedback.get("drift_compensation_vector", [0.0, 0.0, 0.0, 0.0]),
                            dtype=np.float64,
                        )
                    )
                )
            )
        )
    )
    simulation_field_state["voltage_frequency_flux"] = float(
        clamp01(
            keep * float(simulation_field_state.get("voltage_frequency_flux", 0.0))
            + blend
            * (
                0.88 * float(gpu_pulse_delta_feedback.get("voltage_frequency_flux_target", 0.0))
                + 0.12
                * float(
                    list(gpu_pulse_delta_feedback.get("coherent_noise_dof_vector", []) or [0.0] * len(GPU_PULSE_DOF_LABELS))[5]
                    if len(list(gpu_pulse_delta_feedback.get("coherent_noise_dof_vector", []) or [])) > 5
                    else 0.0
                )
            )
        )
    )
    simulation_field_state["frequency_voltage_flux"] = float(
        clamp01(
            keep * float(simulation_field_state.get("frequency_voltage_flux", 0.0))
            + blend
            * (
                0.88 * float(gpu_pulse_delta_feedback.get("frequency_voltage_flux_target", 0.0))
                + 0.12
                * float(
                    list(gpu_pulse_delta_feedback.get("coherent_noise_dof_vector", []) or [0.0] * len(GPU_PULSE_DOF_LABELS))[6]
                    if len(list(gpu_pulse_delta_feedback.get("coherent_noise_dof_vector", []) or [])) > 6
                    else 0.0
                )
            )
        )
    )


def update_substrate_trace_state(
    pulse_index: int,
    previous_trace_state: dict[str, Any] | None = None,
    simulation_field_state: dict[str, Any] | None = None,
    gpu_feedback: dict[str, Any] | None = None,
    gpu_pulse_delta_feedback: dict[str, Any] | None = None,
    interference_field: dict[str, Any] | None = None,
    effective_vector: dict[str, Any] | None = None,
    kernel_execution_event: dict[str, Any] | None = None,
    trace_label: str = "runtime",
) -> dict[str, Any]:
    previous_trace_state = dict(previous_trace_state or {})
    simulation_field_state = dict(simulation_field_state or {})
    gpu_feedback = dict(gpu_feedback or {})
    gpu_pulse_delta_feedback = dict(gpu_pulse_delta_feedback or {})
    interference_field = dict(interference_field or {})
    effective_vector = dict(effective_vector or {})
    kernel_execution_event = dict(kernel_execution_event or {})
    trace_label = str(trace_label or "runtime")

    previous_trace_vector = np.array(
        list(
            previous_trace_state.get(
                "trace_vector",
                simulation_field_state.get("simulation_field_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if previous_trace_vector.shape[0] != 4:
        previous_trace_vector = np.zeros(4, dtype=np.float64)
    previous_trace_axis_vector = np.array(
        list(
            previous_trace_state.get(
                "trace_axis_vector",
                simulation_field_state.get("feedback_axis_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if previous_trace_axis_vector.shape[0] != 4:
        previous_trace_axis_vector = np.zeros(4, dtype=np.float64)
    previous_trace_dof_vector = np.array(
        list(
            previous_trace_state.get(
                "trace_dof_vector",
                gpu_feedback.get("feedback_dof_vector", [0.0] * 10),
            )
            or [0.0] * 10
        ),
        dtype=np.float64,
    )
    if previous_trace_dof_vector.shape[0] != 10:
        previous_trace_dof_vector = np.zeros(10, dtype=np.float64)

    dominant_vector_payload = dict(interference_field.get("dominant_vector", {}) or {})
    dominant_trace_vector = np.array(
        list(
            dominant_vector_payload.get(
                "vector",
                [
                    float(effective_vector.get("x", 0.0)),
                    float(effective_vector.get("y", 0.0)),
                    float(effective_vector.get("z", 0.0)),
                    float(effective_vector.get("t_eff", 0.0)),
                ],
            )
            or [
                float(effective_vector.get("x", 0.0)),
                float(effective_vector.get("y", 0.0)),
                float(effective_vector.get("z", 0.0)),
                float(effective_vector.get("t_eff", 0.0)),
            ]
        ),
        dtype=np.float64,
    )
    if dominant_trace_vector.shape[0] != 4:
        dominant_trace_vector = np.zeros(4, dtype=np.float64)
    dominant_latent_vector = np.array(
        list(
            dominant_vector_payload.get(
                "latent_vector",
                simulation_field_state.get("simulation_field_vector", previous_trace_vector.tolist()),
            )
            or simulation_field_state.get("simulation_field_vector", previous_trace_vector.tolist())
        ),
        dtype=np.float64,
    )
    if dominant_latent_vector.shape[0] != 4:
        dominant_latent_vector = np.array(previous_trace_vector, dtype=np.float64)
    field_vector = np.array(
        list(
            simulation_field_state.get(
                "simulation_field_vector",
                dominant_latent_vector.tolist(),
            )
            or dominant_latent_vector.tolist()
        ),
        dtype=np.float64,
    )
    if field_vector.shape[0] != 4:
        field_vector = np.array(previous_trace_vector, dtype=np.float64)
    feedback_axis_vector = np.array(
        list(
            gpu_feedback.get(
                "feedback_axis_vector",
                simulation_field_state.get("feedback_axis_vector", previous_trace_axis_vector.tolist()),
            )
            or previous_trace_axis_vector.tolist()
        ),
        dtype=np.float64,
    )
    if feedback_axis_vector.shape[0] != 4:
        feedback_axis_vector = np.array(previous_trace_axis_vector, dtype=np.float64)
    feedback_dof_vector = np.array(
        list(gpu_feedback.get("feedback_dof_vector", previous_trace_dof_vector.tolist()) or previous_trace_dof_vector.tolist()),
        dtype=np.float64,
    )
    if feedback_dof_vector.shape[0] != 10:
        feedback_dof_vector = np.array(previous_trace_dof_vector, dtype=np.float64)
    delta_target_vector = np.array(
        list(
            gpu_pulse_delta_feedback.get(
                "delta_target_vector",
                simulation_field_state.get("feedback_delta_target_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if delta_target_vector.shape[0] != 4:
        delta_target_vector = np.zeros(4, dtype=np.float64)
    kernel_control_signature = np.array(
        list(simulation_field_state.get("kernel_control_signature", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if kernel_control_signature.shape[0] != 4:
        kernel_control_signature = np.zeros(4, dtype=np.float64)
    coherent_noise_axis_vector = np.array(
        list(
            gpu_pulse_delta_feedback.get(
                "coherent_noise_axis_vector",
                simulation_field_state.get("coherent_noise_axis_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if coherent_noise_axis_vector.shape[0] != 4:
        coherent_noise_axis_vector = np.zeros(4, dtype=np.float64)
    coherent_noise_dof_vector = np.array(
        list(
            gpu_pulse_delta_feedback.get(
                "coherent_noise_dof_vector",
                simulation_field_state.get("coherent_noise_dof_vector", [0.0] * len(GPU_PULSE_DOF_LABELS)),
            )
            or [0.0] * len(GPU_PULSE_DOF_LABELS)
        ),
        dtype=np.float64,
    )
    if coherent_noise_dof_vector.shape[0] != len(GPU_PULSE_DOF_LABELS):
        coherent_noise_dof_vector = np.zeros(len(GPU_PULSE_DOF_LABELS), dtype=np.float64)
    noise_resonance_nodes = np.array(
        list(
            gpu_pulse_delta_feedback.get(
                "noise_resonance_nodes",
                simulation_field_state.get("noise_resonance_nodes", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if noise_resonance_nodes.shape[0] != 4:
        noise_resonance_nodes = np.zeros(4, dtype=np.float64)
    drift_compensation_vector = np.array(
        list(
            gpu_pulse_delta_feedback.get(
                "drift_compensation_vector",
                simulation_field_state.get("drift_compensation_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if drift_compensation_vector.shape[0] != 4:
        drift_compensation_vector = np.zeros(4, dtype=np.float64)
    previous_trace_relative_spatial_field = np.array(
        list(previous_trace_state.get("trace_relative_spatial_field", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if previous_trace_relative_spatial_field.shape[0] != 4:
        previous_trace_relative_spatial_field = np.zeros(4, dtype=np.float64)
    relative_spatial_field = np.array(
        list(
            gpu_pulse_delta_feedback.get(
                "relative_spatial_field",
                simulation_field_state.get("relative_spatial_field", previous_trace_relative_spatial_field.tolist()),
            )
            or previous_trace_relative_spatial_field.tolist()
        ),
        dtype=np.float64,
    )
    if relative_spatial_field.shape[0] != 4:
        relative_spatial_field = np.array(previous_trace_relative_spatial_field, dtype=np.float64)
    noise_resonance_gate = clamp01(
        float(
            gpu_pulse_delta_feedback.get(
                "noise_resonance_gate",
                simulation_field_state.get("noise_resonance_gate", 0.0),
            )
        )
    )
    noise_orbital_anchor_turns = wrap_turns(
        float(
            gpu_pulse_delta_feedback.get(
                "noise_orbital_anchor_turns",
                simulation_field_state.get("noise_orbital_anchor_turns", 0.0),
            )
        )
    )
    environment_turbulence = clamp01(
        float(
            gpu_pulse_delta_feedback.get(
                "environment_turbulence",
                simulation_field_state.get("environment_turbulence", 0.0),
            )
        )
    )

    previous_trace_phase_anchor = wrap_turns(
        float(previous_trace_state.get("trace_phase_anchor_turns", 0.0))
    )
    feedback_phase_anchor = wrap_turns(float(gpu_feedback.get("phase_anchor_turns", 0.0)))
    dominant_trace_phase = wrap_turns(float(dominant_trace_vector[3]))
    event_phase_anchor = wrap_turns(
        float(
            kernel_execution_event.get(
                "feedback_phase_anchor_turns",
                simulation_field_state.get("feedback_phase_anchor_turns", feedback_phase_anchor),
            )
        )
    )
    phase_shift_turns = float(gpu_pulse_delta_feedback.get("phase_shift_turns", 0.0))
    observation_freshness_gate = clamp01(
        float(gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0))
    )
    response_gate = clamp01(float(gpu_pulse_delta_feedback.get("response_gate", 0.0)))
    response_energy = clamp01(float(gpu_pulse_delta_feedback.get("response_energy", 0.0)))
    dominant_resonance = clamp01(
        float(
            dominant_vector_payload.get(
                "resonance",
                interference_field.get(
                    "field_resonance",
                    previous_trace_state.get("trace_resonance", 0.0),
                ),
            )
        )
    )
    dominant_target_alignment = clamp01(
        float(dominant_vector_payload.get("target_alignment", 0.0))
    )
    field_alignment_score = clamp01(
        float(
            kernel_execution_event.get(
                "field_alignment_score",
                simulation_field_state.get("field_alignment_score", 0.0),
            )
        )
    )
    kernel_control_gate = clamp01(
        float(
            kernel_execution_event.get(
                "kernel_control_gate",
                simulation_field_state.get("kernel_control_gate", 0.0),
            )
        )
    )
    previous_trace_support = clamp01(
        float(previous_trace_state.get("trace_support", 0.0))
    )
    previous_trace_resonance = clamp01(
        float(previous_trace_state.get("trace_resonance", 0.0))
    )
    previous_trace_alignment = clamp01(
        float(previous_trace_state.get("trace_alignment", 0.0))
    )
    previous_trace_memory = clamp01(
        float(previous_trace_state.get("trace_memory", 0.0))
    )
    previous_trace_flux = clamp01(
        float(previous_trace_state.get("trace_flux", 0.0))
    )
    previous_trace_stability = clamp01(
        float(previous_trace_state.get("trace_stability", 0.0))
    )
    previous_trace_temporal_persistence = clamp01(
        float(previous_trace_state.get("trace_temporal_persistence", 0.0))
    )
    previous_trace_temporal_overlap = clamp01(
        float(previous_trace_state.get("trace_temporal_overlap", 0.0))
    )
    previous_trace_vf_flux = clamp01(
        float(previous_trace_state.get("trace_voltage_frequency_flux", 0.0))
    )
    previous_trace_fv_flux = clamp01(
        float(previous_trace_state.get("trace_frequency_voltage_flux", 0.0))
    )

    derived_trace_axis = np.array(
        [
            clamp01(abs(float(dominant_trace_vector[0]))),
            clamp01(abs(float(dominant_trace_vector[1]))),
            clamp01(abs(float(dominant_trace_vector[2]))),
            clamp01(abs(float(dominant_trace_vector[3]))),
        ],
        dtype=np.float64,
    )
    trace_support = clamp01(
        0.22 * dominant_resonance
        + 0.18 * observation_freshness_gate
        + 0.14 * response_gate
        + 0.12 * dominant_target_alignment
        + 0.10 * field_alignment_score
        + 0.08 * kernel_control_gate
        + 0.08 * previous_trace_support
        + 0.08 * previous_trace_resonance
    )
    trace_vector = clamp_vector_norm(
        previous_trace_vector * (0.48 + 0.18 * previous_trace_temporal_persistence)
        + dominant_trace_vector * (0.24 + 0.18 * dominant_resonance)
        + dominant_latent_vector * (0.08 + 0.06 * trace_support)
        + field_vector * (0.12 + 0.12 * field_alignment_score)
        + feedback_axis_vector * (0.08 + 0.12 * response_gate)
        + kernel_control_signature * (0.06 + 0.08 * kernel_control_gate)
        + delta_target_vector * (0.06 + 0.10 * observation_freshness_gate)
        + coherent_noise_axis_vector * (0.05 + 0.08 * noise_resonance_gate)
        + drift_compensation_vector * (0.06 + 0.08 * observation_freshness_gate)
        + relative_spatial_field * (0.08 + 0.10 * noise_resonance_gate)
        + noise_resonance_nodes * (0.04 + 0.06 * environment_turbulence),
        max_norm=3.10,
    )
    trace_axis_vector = np.clip(
        0.44 * previous_trace_axis_vector
        + 0.28 * feedback_axis_vector
        + 0.18 * derived_trace_axis
        + 0.06 * np.abs(kernel_control_signature)
        + 0.04 * np.abs(delta_target_vector)
        + 0.08 * coherent_noise_axis_vector
        + 0.04 * np.abs(noise_resonance_nodes),
        0.0,
        1.0,
    )
    trace_dof_seed = np.array(
        [
            float(trace_axis_vector[0]),
            float(trace_axis_vector[1]),
            float(trace_axis_vector[3]),
            float(trace_axis_vector[2]),
            clamp01(math.sqrt(max(float(trace_axis_vector[0]) * float(trace_axis_vector[1]), 0.0))),
            clamp01(math.sqrt(max(float(trace_axis_vector[0]) * float(trace_axis_vector[3]), 0.0))),
            clamp01(math.sqrt(max(float(trace_axis_vector[0]) * float(trace_axis_vector[2]), 0.0))),
            clamp01(math.sqrt(max(float(trace_axis_vector[1]) * float(trace_axis_vector[3]), 0.0))),
            clamp01(math.sqrt(max(float(trace_axis_vector[1]) * float(trace_axis_vector[2]), 0.0))),
            clamp01(math.sqrt(max(float(trace_axis_vector[3]) * float(trace_axis_vector[2]), 0.0))),
        ],
        dtype=np.float64,
    )
    trace_dof_vector = np.clip(
        0.52 * previous_trace_dof_vector
        + 0.28 * feedback_dof_vector
        + 0.20 * trace_dof_seed
        + 0.08 * coherent_noise_dof_vector,
        0.0,
        1.0,
    )
    trace_phase_anchor_turns = wrap_turns(
        0.44 * previous_trace_phase_anchor
        + 0.24 * feedback_phase_anchor
        + 0.14 * dominant_trace_phase
        + 0.10 * event_phase_anchor
        + 0.08 * phase_shift_turns
        + 0.06 * noise_orbital_anchor_turns
    )
    trace_relative_spatial_field = clamp_vector_norm(
        previous_trace_relative_spatial_field * (0.42 + 0.18 * previous_trace_temporal_persistence)
        + relative_spatial_field * (0.28 + 0.16 * noise_resonance_gate)
        + trace_vector * (0.12 + 0.10 * trace_support)
        + drift_compensation_vector * (0.10 + 0.10 * observation_freshness_gate)
        + noise_resonance_nodes * (0.08 + 0.08 * environment_turbulence),
        max_norm=2.55,
    )
    trace_alignment = clamp01(
        0.30 * vector_similarity(trace_vector, dominant_trace_vector)
        + 0.24 * vector_similarity(trace_axis_vector, feedback_axis_vector)
        + 0.18 * vector_similarity(trace_vector, field_vector)
        + 0.08 * vector_similarity(trace_relative_spatial_field, relative_spatial_field)
        + 0.14 * dominant_target_alignment
        + 0.08 * field_alignment_score
        + 0.06 * previous_trace_alignment
    )
    trace_resonance = clamp01(
        0.32 * previous_trace_resonance
        + 0.24 * dominant_resonance
        + 0.16 * trace_alignment
        + 0.08 * noise_resonance_gate
        + 0.12 * response_gate
        + 0.10 * observation_freshness_gate
        + 0.06 * field_alignment_score
    )
    trace_memory = clamp01(
        0.36 * previous_trace_memory
        + 0.20 * clamp01(float(gpu_feedback.get("memory_proxy", 0.0)))
        + 0.12 * clamp01(float(gpu_pulse_delta_feedback.get("memory_retention", 0.0)))
        + 0.10 * trace_resonance
        + 0.10 * trace_alignment
        + 0.06 * observation_freshness_gate
        + 0.06 * dominant_target_alignment
    )
    trace_flux = clamp01(
        0.34 * previous_trace_flux
        + 0.18 * clamp01(float(gpu_feedback.get("flux_proxy", 0.0)))
        + 0.14 * clamp01(float(gpu_pulse_delta_feedback.get("response_energy", 0.0)))
        + 0.12 * dominant_resonance
        + 0.08 * environment_turbulence
        + 0.10 * response_gate
        + 0.06 * field_alignment_score
        + 0.06 * observation_freshness_gate
    )
    trace_stability = clamp01(
        0.38 * previous_trace_stability
        + 0.18 * clamp01(float(gpu_feedback.get("stability_proxy", 0.0)))
        + 0.12 * trace_alignment
        + 0.10 * trace_memory
        + 0.08 * (1.0 - clamp01(float(gpu_pulse_delta_feedback.get("latency_norm", 0.0))))
        + 0.08 * clamp01(float(gpu_feedback.get("environment_stability", 0.0)))
        + 0.04 * (1.0 - environment_turbulence)
        + 0.06 * observation_freshness_gate
    )
    trace_temporal_persistence = clamp01(
        0.34 * previous_trace_temporal_persistence
        + 0.18 * float(simulation_field_state.get("sequence_persistence_score", 0.0))
        + 0.12 * clamp01(float(gpu_pulse_delta_feedback.get("sequence_persistence_target", 0.0)))
        + 0.12 * trace_memory
        + 0.10 * trace_stability
        + 0.08 * trace_alignment
        + 0.06 * observation_freshness_gate
    )
    trace_temporal_overlap = clamp01(
        0.34 * previous_trace_temporal_overlap
        + 0.18 * float(simulation_field_state.get("temporal_index_overlap", 0.0))
        + 0.12 * clamp01(float(gpu_pulse_delta_feedback.get("temporal_overlap_target", 0.0)))
        + 0.12 * trace_alignment
        + 0.10 * trace_stability
        + 0.08 * trace_memory
        + 0.06 * dominant_target_alignment
    )
    trace_voltage_frequency_flux = clamp01(
        0.34 * previous_trace_vf_flux
        + 0.18 * float(simulation_field_state.get("voltage_frequency_flux", 0.0))
        + 0.12 * clamp01(float(gpu_pulse_delta_feedback.get("voltage_frequency_flux_target", 0.0)))
        + 0.12 * trace_flux
        + 0.10 * float(trace_axis_vector[3])
        + 0.08 * float(trace_axis_vector[0])
        + 0.06 * observation_freshness_gate
    )
    trace_frequency_voltage_flux = clamp01(
        0.34 * previous_trace_fv_flux
        + 0.18 * float(simulation_field_state.get("frequency_voltage_flux", 0.0))
        + 0.12 * clamp01(float(gpu_pulse_delta_feedback.get("frequency_voltage_flux_target", 0.0)))
        + 0.12 * trace_flux
        + 0.10 * float(trace_axis_vector[0])
        + 0.08 * float(trace_axis_vector[3])
        + 0.06 * trace_alignment
    )

    trace_history = list(previous_trace_state.get("trace_history", []) or [])
    trace_history.append(
        {
            "pulse_index": int(pulse_index),
            "trace_label": trace_label,
            "trace_support": float(trace_support),
            "trace_resonance": float(trace_resonance),
            "trace_alignment": float(trace_alignment),
            "trace_memory": float(trace_memory),
            "trace_flux": float(trace_flux),
            "trace_stability": float(trace_stability),
            "trace_phase_anchor_turns": float(trace_phase_anchor_turns),
            "trace_relative_spatial_field": [float(value) for value in trace_relative_spatial_field],
            "noise_resonance_gate": float(noise_resonance_gate),
            "environment_turbulence": float(environment_turbulence),
            "observation_freshness_gate": float(observation_freshness_gate),
            "response_gate": float(response_gate),
            "trace_vector": [float(value) for value in trace_vector],
        }
    )
    trace_history = trace_history[-16:]
    return {
        "pulse_index": int(pulse_index),
        "trace_label": trace_label,
        "trace_vector": [float(value) for value in trace_vector],
        "trace_axis_vector": [float(value) for value in trace_axis_vector],
        "trace_dof_vector": [float(value) for value in trace_dof_vector],
        "trace_phase_anchor_turns": float(trace_phase_anchor_turns),
        "trace_relative_spatial_field": [float(value) for value in trace_relative_spatial_field],
        "coherent_noise_axis_vector": [float(value) for value in coherent_noise_axis_vector],
        "coherent_noise_dof_vector": [float(value) for value in coherent_noise_dof_vector],
        "noise_resonance_nodes": [float(value) for value in noise_resonance_nodes],
        "drift_compensation_vector": [float(value) for value in drift_compensation_vector],
        "noise_resonance_gate": float(noise_resonance_gate),
        "noise_orbital_anchor_turns": float(noise_orbital_anchor_turns),
        "environment_turbulence": float(environment_turbulence),
        "trace_support": float(trace_support),
        "trace_resonance": float(trace_resonance),
        "trace_alignment": float(trace_alignment),
        "trace_memory": float(trace_memory),
        "trace_flux": float(trace_flux),
        "trace_stability": float(trace_stability),
        "trace_temporal_persistence": float(trace_temporal_persistence),
        "trace_temporal_overlap": float(trace_temporal_overlap),
        "trace_voltage_frequency_flux": float(trace_voltage_frequency_flux),
        "trace_frequency_voltage_flux": float(trace_frequency_voltage_flux),
        "observation_freshness_gate": float(observation_freshness_gate),
        "response_gate": float(response_gate),
        "trace_history": trace_history,
    }


def upload_array_to_vram(cache_key: str, host_array: np.ndarray) -> Any:
    if numba_cuda is None:
        return None
    device_array = SUBSTRATE_TRACE_VRAM_CACHE.get(cache_key)
    if device_array is None:
        device_array = numba_cuda.to_device(host_array)
    else:
        try:
            if tuple(device_array.shape) != tuple(host_array.shape) or str(device_array.dtype) != str(host_array.dtype):
                device_array = numba_cuda.to_device(host_array)
            else:
                device_array.copy_to_device(host_array)
        except Exception:
            device_array = numba_cuda.to_device(host_array)
    SUBSTRATE_TRACE_VRAM_CACHE[cache_key] = device_array
    return device_array


def sync_substrate_trace_state_to_vram(
    trace_state: dict[str, Any] | None,
) -> dict[str, Any]:
    trace_state = dict(trace_state or {})
    if numba_cuda is None:
        return {"resident": False, "reason": "numba_cuda_unavailable", "update_count": 0}
    try:
        if not bool(numba_cuda.is_available()):
            return {"resident": False, "reason": "cuda_device_unavailable", "update_count": 0}
    except Exception as exc:
        return {
            "resident": False,
            "reason": f"cuda_probe_failed:{type(exc).__name__}",
            "update_count": 0,
        }
    trace_vector = np.array(
        list(trace_state.get("trace_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    trace_axis_vector = np.array(
        list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    trace_dof_vector = np.array(
        list(trace_state.get("trace_dof_vector", [0.0] * 10) or [0.0] * 10),
        dtype=np.float32,
    )
    trace_relative_spatial_field = np.array(
        list(trace_state.get("trace_relative_spatial_field", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    coherent_noise_axis_vector = np.array(
        list(trace_state.get("coherent_noise_axis_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    coherent_noise_dof_vector = np.array(
        list(trace_state.get("coherent_noise_dof_vector", [0.0] * len(GPU_PULSE_DOF_LABELS)) or [0.0] * len(GPU_PULSE_DOF_LABELS)),
        dtype=np.float32,
    )
    noise_resonance_nodes = np.array(
        list(trace_state.get("noise_resonance_nodes", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    drift_compensation_vector = np.array(
        list(trace_state.get("drift_compensation_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    trace_summary_vector = np.array(
        [
            float(trace_state.get("trace_support", 0.0)),
            float(trace_state.get("trace_resonance", 0.0)),
            float(trace_state.get("trace_alignment", 0.0)),
            float(trace_state.get("trace_memory", 0.0)),
            float(trace_state.get("trace_flux", 0.0)),
            float(trace_state.get("trace_stability", 0.0)),
            float(trace_state.get("trace_temporal_persistence", 0.0)),
            float(trace_state.get("trace_temporal_overlap", 0.0)),
            float(trace_state.get("trace_voltage_frequency_flux", 0.0)),
            float(trace_state.get("trace_frequency_voltage_flux", 0.0)),
        ],
        dtype=np.float32,
    )
    upload_array_to_vram("trace_vector", trace_vector)
    upload_array_to_vram("trace_axis_vector", trace_axis_vector)
    upload_array_to_vram("trace_dof_vector", trace_dof_vector)
    upload_array_to_vram("trace_relative_spatial_field", trace_relative_spatial_field)
    upload_array_to_vram("coherent_noise_axis_vector", coherent_noise_axis_vector)
    upload_array_to_vram("coherent_noise_dof_vector", coherent_noise_dof_vector)
    upload_array_to_vram("noise_resonance_nodes", noise_resonance_nodes)
    upload_array_to_vram("drift_compensation_vector", drift_compensation_vector)
    upload_array_to_vram("trace_summary_vector", trace_summary_vector)
    update_count = int(SUBSTRATE_TRACE_VRAM_CACHE.get("update_count", 0)) + 1
    SUBSTRATE_TRACE_VRAM_CACHE["update_count"] = update_count
    device_name = ""
    try:
        raw_name = numba_cuda.get_current_device().name
        device_name = raw_name.decode("ascii", errors="ignore") if isinstance(raw_name, bytes) else str(raw_name)
    except Exception:
        device_name = ""
    return {
        "resident": True,
        "device": device_name,
        "update_count": int(update_count),
        "trace_vector_dim": int(trace_vector.shape[0]),
        "trace_axis_dim": int(trace_axis_vector.shape[0]),
        "trace_dof_dim": int(trace_dof_vector.shape[0]),
        "trace_relative_spatial_dim": int(trace_relative_spatial_field.shape[0]),
        "coherent_noise_axis_dim": int(coherent_noise_axis_vector.shape[0]),
        "coherent_noise_dof_dim": int(coherent_noise_dof_vector.shape[0]),
        "noise_resonance_dim": int(noise_resonance_nodes.shape[0]),
        "drift_compensation_dim": int(drift_compensation_vector.shape[0]),
        "trace_summary_dim": int(trace_summary_vector.shape[0]),
    }


def update_compute_regime(
    simulation_field_state: dict[str, Any],
    kernel_execution_event: dict[str, Any],
    freshness_gate: float,
) -> None:
    trace_state = dict(simulation_field_state.get("substrate_trace_state", {}) or {})
    trace_vram = dict(simulation_field_state.get("substrate_trace_vram", {}) or {})
    calibration_readiness = clamp01(
        float(simulation_field_state.get("calibration_readiness", 0.0))
    )
    target_gate = clamp01(float(simulation_field_state.get("target_gate", 0.0)))
    field_alignment_score = clamp01(
        float(simulation_field_state.get("field_alignment_score", 0.0))
    )
    trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
    trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
    trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
    trace_temporal_persistence = clamp01(
        float(trace_state.get("trace_temporal_persistence", 0.0))
    )
    freshness_gate = clamp01(float(freshness_gate))
    vector_harmonic_gate = clamp01(
        0.26 * calibration_readiness
        + 0.18 * target_gate
        + 0.14 * field_alignment_score
        + 0.12 * trace_support
        + 0.10 * trace_resonance
        + 0.08 * trace_alignment
        + 0.06 * trace_temporal_persistence
        + 0.12 * freshness_gate
        + 0.10 * (1.0 if bool(trace_vram.get("resident", False)) else 0.0)
    )
    compute_regime = (
        "vector_harmonic"
        if (
            vector_harmonic_gate >= 0.68
            and calibration_readiness >= 0.72
            and target_gate >= 0.62
            and bool(trace_vram.get("resident", False))
            and freshness_gate >= 0.72
        )
        else "classical_calibration"
    )
    harmonic_compute_weight = clamp01(
        0.24 + 0.76 * vector_harmonic_gate
        if compute_regime == "vector_harmonic"
        else 0.12 + 0.46 * vector_harmonic_gate
    )
    simulation_field_state["substrate_material"] = "silicon_wafer"
    simulation_field_state["silicon_reference_source"] = str(NIST_REFERENCE.name)
    simulation_field_state["compute_regime"] = compute_regime
    simulation_field_state["vector_harmonic_gate"] = float(vector_harmonic_gate)
    simulation_field_state["harmonic_compute_weight"] = float(harmonic_compute_weight)
    kernel_execution_event["substrate_material"] = "silicon_wafer"
    kernel_execution_event["silicon_reference_source"] = str(NIST_REFERENCE.name)
    kernel_execution_event["compute_regime"] = compute_regime
    kernel_execution_event["vector_harmonic_gate"] = float(vector_harmonic_gate)
    kernel_execution_event["harmonic_compute_weight"] = float(harmonic_compute_weight)


def apply_gpu_feedback_delta_to_candidates(
    candidate_pool: list[dict[str, Any]],
    gpu_pulse_delta_feedback: dict[str, Any] | None,
    simulation_field_state: dict[str, Any] | None = None,
) -> None:
    gpu_pulse_delta_feedback = dict(gpu_pulse_delta_feedback or {})
    simulation_field_state = dict(simulation_field_state or {})
    if not candidate_pool:
        return
    delta_target_vector = np.array(
        list(gpu_pulse_delta_feedback.get("delta_target_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if delta_target_vector.shape[0] != 4:
        delta_target_vector = np.zeros(4, dtype=np.float64)
    phase_pressure_target = clamp01(float(gpu_pulse_delta_feedback.get("phase_pressure_target", 0.0)))
    response_gate = clamp01(float(gpu_pulse_delta_feedback.get("response_gate", 0.0)))
    response_energy = clamp01(float(gpu_pulse_delta_feedback.get("response_energy", 0.0)))
    memory_retention = clamp01(float(gpu_pulse_delta_feedback.get("memory_retention", 0.0)))
    observation_freshness_gate = clamp01(
        float(gpu_pulse_delta_feedback.get("observation_freshness_gate", 1.0))
    )
    trace_state = dict(simulation_field_state.get("substrate_trace_state", {}) or {})
    compute_regime = str(simulation_field_state.get("compute_regime", "classical_calibration"))
    harmonic_compute_weight = clamp01(
        float(simulation_field_state.get("harmonic_compute_weight", 0.0))
    )
    trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
    trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
    trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
    vector_harmonic_weight = (
        harmonic_compute_weight if compute_regime == "vector_harmonic" else 0.0
    )
    for candidate in candidate_pool:
        temporal_vector = np.array(
            [
                clamp01(float(candidate.get("sequence_persistence_score", 0.0))),
                clamp01(float(candidate.get("temporal_index_overlap", 0.0))),
                clamp01(float(candidate.get("voltage_frequency_flux", 0.0))),
                clamp01(float(candidate.get("frequency_voltage_flux", 0.0))),
            ],
            dtype=np.float64,
        )
        temporal_alignment = clamp01(1.0 - float(np.linalg.norm(temporal_vector - delta_target_vector)) / 2.0)
        phase_alignment = clamp01(
            1.0 - abs(clamp01(float(candidate.get("phase_length_pressure", 0.0))) - phase_pressure_target)
        )
        kernel_delta_alignment = clamp01(
            0.32 * response_gate
            + 0.24 * clamp01(float(candidate.get("kernel_delta_gate", 0.0)))
            + 0.22 * clamp01(float(candidate.get("kernel_delta_phase_alignment", 0.0)))
            + 0.12 * clamp01(float(candidate.get("kernel_delta_memory", 0.0)))
            + 0.10 * clamp01(float(candidate.get("kernel_delta_flux", 0.0)))
        )
        ancilla_alignment = clamp01(
            0.26 * clamp01(float(candidate.get("ancilla_convergence", 0.0)))
            + 0.24 * clamp01(float(candidate.get("ancilla_phase_alignment", 0.0)))
            + 0.18 * clamp01(float(candidate.get("ancilla_flux_norm", 0.0)))
            + 0.16 * clamp01(float(candidate.get("ancilla_temporal_persistence", 0.0)))
            + 0.16 * clamp01(float(candidate.get("ancilla_commit_gate", 0.0)))
        )
        raw_cuda_score = float(candidate.get("cuda_temporal_score", 0.0))
        candidate["cuda_temporal_score_raw"] = float(raw_cuda_score)
        gpu_feedback_delta_score = clamp01(
            0.44 * temporal_alignment
            + 0.18 * phase_alignment
            + 0.12 * raw_cuda_score
            + 0.10 * clamp01(float(candidate.get("target_alignment", 0.0)))
            + 0.08 * clamp01(float(candidate.get("coherence_peak", 0.0)))
            + 0.08 * response_gate
            + 0.06 * clamp01(float(candidate.get("kernel_balance_score", 0.0)))
            + 0.06 * clamp01(float(candidate.get("harmonic_resonance_score", 0.0)))
            + 0.04 * clamp01(float(candidate.get("retro_temporal_gain", 0.0)))
            + 0.04 * clamp01(float(candidate.get("kernel_phase_alignment", 0.0)))
            + 0.06 * kernel_delta_alignment
            + 0.06 * ancilla_alignment
            + 0.05 * trace_alignment
            + 0.04 * trace_support
            + 0.03 * trace_resonance
        )
        gpu_feedback_delta_score = clamp01(
            gpu_feedback_delta_score
            * (
                0.42
                + 0.46 * observation_freshness_gate
                + 0.12 * vector_harmonic_weight
            )
        )
        previous_feedback_delta_score = clamp01(float(candidate.get("gpu_feedback_delta_score", 0.0)))
        previous_observation_freshness = clamp01(
            float(candidate.get("gpu_feedback_observation_freshness", observation_freshness_gate))
        )
        if previous_feedback_delta_score > 0.0:
            previous_weight = (
                0.28 + 0.72 * previous_observation_freshness
            ) * (1.0 + 0.35 * vector_harmonic_weight)
            current_weight = (
                0.08 + 0.92 * observation_freshness_gate
            ) * (0.12 + 0.88 * observation_freshness_gate)
            if vector_harmonic_weight > 0.0:
                current_weight *= 0.30 + 0.70 * observation_freshness_gate
            total_weight = max(previous_weight + current_weight, 1.0e-6)
            gpu_feedback_delta_score = clamp01(
                (
                    previous_feedback_delta_score * previous_weight
                    + gpu_feedback_delta_score * current_weight
                )
                / total_weight
            )
            observation_freshness_gate = max(
                observation_freshness_gate,
                previous_observation_freshness,
            )
        candidate["gpu_feedback_delta_score"] = float(gpu_feedback_delta_score)
        candidate["gpu_feedback_temporal_alignment"] = float(temporal_alignment)
        candidate["gpu_feedback_phase_alignment"] = float(phase_alignment)
        candidate["gpu_feedback_kernel_delta_alignment"] = float(kernel_delta_alignment)
        candidate["gpu_feedback_ancilla_alignment"] = float(ancilla_alignment)
        candidate["gpu_feedback_trace_alignment"] = float(trace_alignment)
        candidate["gpu_feedback_trace_support"] = float(trace_support)
        candidate["gpu_feedback_observation_freshness"] = float(observation_freshness_gate)
        candidate["gpu_feedback_response_gate"] = float(response_gate)
        candidate["cuda_temporal_score"] = float(
            clamp01(
                (0.76 - 0.12 * vector_harmonic_weight) * raw_cuda_score
                + (0.24 + 0.12 * vector_harmonic_weight) * gpu_feedback_delta_score
            )
        )
        candidate["sequence_persistence_score"] = float(
            clamp01(
                0.90 * float(candidate.get("sequence_persistence_score", 0.0))
                + 0.10 * float(delta_target_vector[0])
            )
        )
        candidate["temporal_index_overlap"] = float(
            clamp01(
                0.90 * float(candidate.get("temporal_index_overlap", 0.0))
                + 0.10 * float(delta_target_vector[1])
            )
        )
        candidate["voltage_frequency_flux"] = float(
            clamp01(
                0.90 * float(candidate.get("voltage_frequency_flux", 0.0))
                + 0.10 * float(delta_target_vector[2])
            )
        )
        candidate["frequency_voltage_flux"] = float(
            clamp01(
                0.90 * float(candidate.get("frequency_voltage_flux", 0.0))
                + 0.10 * float(delta_target_vector[3])
            )
        )
        candidate["target_alignment"] = float(
            clamp01(
                0.92 * float(candidate.get("target_alignment", 0.0))
                + 0.04 * phase_alignment
                + 0.04 * temporal_alignment
                + 0.02 * clamp01(float(candidate.get("kernel_delta_phase_alignment", 0.0)))
            )
        )
        candidate["phase_length_pressure"] = float(
            clamp01(
                0.92 * float(candidate.get("phase_length_pressure", 0.0))
                + 0.08 * phase_pressure_target
                + 0.04 * clamp01(float(candidate.get("kernel_delta_gate", 0.0)))
            )
        )
        candidate["phase_alignment_score"] = float(candidate["phase_length_pressure"])
        candidate["phase_confinement_cost"] = float(
            clamp01(
                0.90 * float(candidate.get("phase_confinement_cost", 1.0 - candidate["phase_length_pressure"]))
                + 0.10 * (1.0 - float(candidate["phase_length_pressure"]))
            )
        )
        candidate["btc_phase_alignment"] = float(
            clamp01(float(candidate.get("btc_phase_alignment", candidate.get("btc_phase_pressure", 0.0))))
        )
        candidate["btc_phase_cost"] = float(
            clamp01(
                0.90 * float(candidate.get("btc_phase_cost", 1.0 - candidate["btc_phase_alignment"]))
                + 0.10 * (1.0 - float(candidate["btc_phase_alignment"]))
            )
        )
        candidate["gpu_feedback_memory_retention"] = float(memory_retention)
        candidate["gpu_feedback_response_energy"] = float(response_energy)


def run_cuda_temporal_candidate_stage(
    candidate_pool: list[dict[str, Any]],
    simulation_field_state: dict[str, Any],
    target_profile: dict[str, Any],
) -> dict[str, Any]:
    telemetry = {
        "backend": "cuda",
        "enabled": False,
        "device": "",
        "candidate_count": int(len(candidate_pool)),
        "feature_dim": 30,
        "score_mean": 0.0,
        "score_max": 0.0,
        "expansion_factor": 1,
        "expanded_eval_count": int(len(candidate_pool)),
        "expanded_keep_count": 0,
        "blocks_per_grid": 0,
        "threads_per_block": 0,
        "reason": "",
    }
    if not candidate_pool:
        telemetry["reason"] = "empty_candidate_pool"
        return telemetry
    if numba_cuda is None:
        telemetry["reason"] = "numba_cuda_unavailable"
        return telemetry
    try:
        if not bool(numba_cuda.is_available()):
            telemetry["reason"] = "cuda_device_unavailable"
            return telemetry
    except Exception as exc:
        telemetry["reason"] = f"cuda_probe_failed:{type(exc).__name__}"
        return telemetry
    try:
        target_hex = str(
            simulation_field_state.get(
                "network_target_hex",
                target_profile.get("target_hex", target_profile.get("prototype_target_hex", "")),
            )
        )
        target_phase_anchor = wrap_turns(
            float(np.mean(list(target_profile.get("phase_windows", []) or [0.0])))
        )
        target_prefix_vector_metrics = build_prefix_trajectory_vector(
            target_hex=target_hex,
            target_phase_pressure=float(simulation_field_state.get("btc_network_phase_pressure", 0.0)),
            target_phase_alignment=float(
                simulation_field_state.get(
                    "btc_network_phase_alignment",
                    simulation_field_state.get("btc_network_phase_pressure", 0.0),
                )
            ),
            target_flux_alignment=clamp01(
                0.56 * float(simulation_field_state.get("voltage_frequency_flux", 0.0))
                + 0.44 * float(simulation_field_state.get("frequency_voltage_flux", 0.0))
            ),
            electron_ring_stability=clamp01(
                0.62 * float(simulation_field_state.get("trace_support", 0.0))
                + 0.38 * float(simulation_field_state.get("trace_resonance", 0.0))
            ),
            electron_phase_anchor=target_phase_anchor,
        )
        target_prefix_vector = np.array(
            list(target_prefix_vector_metrics.get("prefix_trajectory_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if target_prefix_vector.shape[0] != 4:
            target_prefix_vector = np.zeros(4, dtype=np.float32)
        target_relative_spatial_field = np.array(
            list(
                simulation_field_state.get("relative_spatial_field", [0.0, 0.0, 0.0, 0.0])
                or [0.0, 0.0, 0.0, 0.0]
            ),
            dtype=np.float32,
        )
        if target_relative_spatial_field.shape[0] != 4:
            target_relative_spatial_field = np.zeros(4, dtype=np.float32)
        feature_rows = np.array(
            [
                [
                    float(candidate.get("coherence_peak", 0.0)),
                    float(candidate.get("target_alignment", 0.0)),
                    float(candidate.get("interference_resonance", 0.0)),
                    float(candidate.get("row_activation", 0.0)),
                    float(candidate.get("motif_alignment", 0.0)),
                    float(candidate.get("phase_alignment_score", candidate.get("phase_length_pressure", 0.0))),
                    float(1.0 - clamp01(float(candidate.get("phase_confinement_cost", 1.0 - float(candidate.get("phase_length_pressure", 0.0)))))),
                    float(candidate.get("amplitude_phase_pressure", 0.0)),
                    float(candidate.get("btc_force_alignment", 0.0)),
                    float(candidate.get("btc_phase_alignment", candidate.get("btc_phase_pressure", 0.0))),
                    float(candidate.get("sequence_persistence_score", 0.0)),
                    float(candidate.get("temporal_index_overlap", 0.0)),
                    float(candidate.get("voltage_frequency_flux", 0.0)),
                    float(candidate.get("frequency_voltage_flux", 0.0)),
                    float(candidate.get("kernel_balance_score", 0.0)),
                    float(candidate.get("harmonic_resonance_score", 0.0)),
                    float(candidate.get("retro_temporal_gain", 0.0)),
                    float(candidate.get("kernel_phase_alignment", 0.0)),
                    float(candidate.get("kernel_delta_gate", 0.0)),
                    float(candidate.get("kernel_delta_memory", 0.0)),
                    float(candidate.get("kernel_delta_flux", 0.0)),
                    float(candidate.get("kernel_delta_phase_alignment", 0.0)),
                    float(
                        list(candidate.get("decode_target_prefix_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[0]
                    ),
                    float(
                        list(candidate.get("decode_target_prefix_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[1]
                    ),
                    float(
                        list(candidate.get("decode_target_prefix_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[2]
                    ),
                    float(
                        list(candidate.get("decode_target_prefix_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[3]
                    ),
                    float(
                        list(candidate.get("decode_phase_orbital_trace_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[0]
                    ),
                    float(
                        list(candidate.get("decode_phase_orbital_trace_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[1]
                    ),
                    float(
                        list(candidate.get("decode_phase_orbital_trace_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[2]
                    ),
                    float(
                        list(candidate.get("decode_phase_orbital_trace_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[3]
                    ),
                ]
                for candidate in candidate_pool
            ],
            dtype=np.float32,
        )
        candidate_count = int(feature_rows.shape[0])
        phase_pressure_drive = clamp01(
            float(
                np.mean(
                    [
                        float(
                            candidate.get(
                                "phase_alignment_score",
                                candidate.get("phase_length_pressure", 0.0),
                            )
                        )
                        for candidate in candidate_pool
                    ]
                )
            )
        )
        phase_confinement_drive = clamp01(
            float(
                np.mean(
                    [
                        float(
                            candidate.get(
                                "phase_confinement_cost",
                                1.0
                                - float(candidate.get("phase_length_pressure", 0.0)),
                            )
                        )
                        for candidate in candidate_pool
                    ]
                )
            )
        )
        temporal_overlap_drive = clamp01(
            float(np.mean([float(candidate.get("temporal_index_overlap", 0.0)) for candidate in candidate_pool]))
        )
        sequence_persistence_drive = clamp01(
            float(np.mean([float(candidate.get("sequence_persistence_score", 0.0)) for candidate in candidate_pool]))
        )
        voltage_frequency_drive = clamp01(
            float(np.mean([float(candidate.get("voltage_frequency_flux", 0.0)) for candidate in candidate_pool]))
        )
        frequency_voltage_drive = clamp01(
            float(np.mean([float(candidate.get("frequency_voltage_flux", 0.0)) for candidate in candidate_pool]))
        )
        trace_state = dict(simulation_field_state.get("substrate_trace_state", {}) or {})
        trace_vram = dict(simulation_field_state.get("substrate_trace_vram", {}) or {})
        trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
        trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
        trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
        trace_temporal_persistence = clamp01(
            float(trace_state.get("trace_temporal_persistence", 0.0))
        )
        trace_temporal_overlap = clamp01(
            float(trace_state.get("trace_temporal_overlap", 0.0))
        )
        trace_voltage_frequency_flux = clamp01(
            float(trace_state.get("trace_voltage_frequency_flux", 0.0))
        )
        trace_frequency_voltage_flux = clamp01(
            float(trace_state.get("trace_frequency_voltage_flux", 0.0))
        )
        harmonic_compute_weight = clamp01(
            float(simulation_field_state.get("harmonic_compute_weight", 0.0))
        )
        trace_axis_vector = np.array(
            list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if trace_axis_vector.shape[0] != 4:
            trace_axis_vector = np.zeros(4, dtype=np.float32)
        trace_summary_vector = np.array(
            [
                float(trace_support),
                float(trace_resonance),
                float(trace_alignment),
                float(trace_state.get("trace_memory", 0.0)),
                float(trace_state.get("trace_flux", 0.0)),
                float(trace_state.get("trace_stability", 0.0)),
                float(trace_temporal_persistence),
                float(trace_temporal_overlap),
                float(trace_voltage_frequency_flux),
                float(trace_frequency_voltage_flux),
            ],
            dtype=np.float32,
        )
        trace_relative_spatial_field = np.array(
            list(
                trace_state.get(
                    "trace_relative_spatial_field",
                    target_relative_spatial_field.tolist(),
                )
                or target_relative_spatial_field.tolist()
            ),
            dtype=np.float32,
        )
        if trace_relative_spatial_field.shape[0] != 4:
            trace_relative_spatial_field = np.array(target_relative_spatial_field, dtype=np.float32)
        target_relative_spatial_field = np.clip(
            0.58 * target_relative_spatial_field + 0.42 * trace_relative_spatial_field,
            0.0,
            1.0,
        )
        kernel_delta_drive = clamp01(
            0.62 * float(simulation_field_state.get("kernel_delta_gate_mean", 0.0))
            + 0.38 * float(simulation_field_state.get("kernel_delta_flux_mean", 0.0))
        )
        phase_volume_gain = clamp01(
            0.50 * phase_pressure_drive
            + 0.18 * (1.0 - phase_confinement_drive)
            + 0.22 * sequence_persistence_drive
            + 0.20 * float(simulation_field_state.get("target_gate", 0.0))
            + 0.14 * trace_support
        )
        coupling_volume_gain = clamp01(
            0.32 * temporal_overlap_drive
            + 0.26 * voltage_frequency_drive
            + 0.26 * frequency_voltage_drive
            + 0.16 * sequence_persistence_drive
            + 0.12 * trace_voltage_frequency_flux
            + 0.10 * trace_frequency_voltage_flux
        )
        kernel_volume_gain = clamp01(
            0.56 * kernel_delta_drive
            + 0.24 * float(simulation_field_state.get("kernel_drive_mean", 0.0))
            + 0.20 * float(simulation_field_state.get("kernel_balance_mean", 0.0))
            + 0.16 * trace_resonance
        )
        search_volume_gain = (
            1.0
            + 0.85 * phase_volume_gain
            + 0.95 * coupling_volume_gain
            + 0.70 * kernel_volume_gain
            + 0.60 * harmonic_compute_weight
        )
        target_eval_floor = int(
            round(49152.0 + 24576.0 * phase_volume_gain + 16384.0 * kernel_volume_gain)
        )
        target_eval_cap = int(
            round(131072.0 + 65536.0 * coupling_volume_gain + 32768.0 * phase_volume_gain)
        )
        target_eval_count = max(
            target_eval_floor,
            min(
                target_eval_cap,
                int(
                    round(
                        float(candidate_count)
                        * (
                            26.0
                            + 12.0 * phase_volume_gain
                            + 14.0 * coupling_volume_gain
                            + 10.0 * kernel_volume_gain
                        )
                    )
                ),
            ),
        )
        telemetry["phase_volume_gain"] = float(phase_volume_gain)
        telemetry["phase_confinement_drive"] = float(phase_confinement_drive)
        telemetry["coupling_volume_gain"] = float(coupling_volume_gain)
        telemetry["kernel_volume_gain"] = float(kernel_volume_gain)
        telemetry["search_volume_gain"] = float(search_volume_gain)
        telemetry["trace_support"] = float(trace_support)
        telemetry["trace_resonance"] = float(trace_resonance)
        telemetry["trace_alignment"] = float(trace_alignment)
        telemetry["trace_vram_resident"] = bool(trace_vram.get("resident", False))
        telemetry["trace_vram_updates"] = int(trace_vram.get("update_count", 0))
        telemetry["harmonic_compute_weight"] = float(harmonic_compute_weight)
        telemetry["compute_regime"] = str(
            simulation_field_state.get("compute_regime", "classical_calibration")
        )
        telemetry["substrate_material"] = str(
            simulation_field_state.get("substrate_material", "silicon_wafer")
        )
        telemetry["silicon_reference_source"] = str(
            simulation_field_state.get("silicon_reference_source", NIST_REFERENCE.name)
        )
        telemetry["target_eval_floor"] = int(target_eval_floor)
        telemetry["target_eval_cap"] = int(target_eval_cap)
        expansion_factor = max(1, min(64, int(math.ceil(float(target_eval_count) / max(float(candidate_count), 1.0)))))
        expanded_eval_count = int(candidate_count * expansion_factor)
        telemetry["expansion_factor"] = int(expansion_factor)
        telemetry["expanded_eval_count"] = int(expanded_eval_count)
        features = np.empty((expanded_eval_count, feature_rows.shape[1]), dtype=np.float32)
        expanded_base_index = np.empty(expanded_eval_count, dtype=np.int32)
        expanded_branch_index = np.empty(expanded_eval_count, dtype=np.int32)
        expanded_nonce_offsets = np.empty(expanded_eval_count, dtype=np.uint32)
        expanded_row = 0
        for candidate_index, candidate in enumerate(candidate_pool):
            base = feature_rows[candidate_index]
            base_nonce = int(candidate.get("nonce", 0)) & 0xFFFFFFFF
            phase_length_index = int(candidate.get("phase_length_index", 0))
            temporal_sequence_index = int(candidate.get("temporal_sequence_index", 0))
            sequence_persistence_score = clamp01(float(candidate.get("sequence_persistence_score", 0.0)))
            temporal_index_overlap = clamp01(float(candidate.get("temporal_index_overlap", 0.0)))
            voltage_frequency_flux = clamp01(float(candidate.get("voltage_frequency_flux", 0.0)))
            frequency_voltage_flux = clamp01(float(candidate.get("frequency_voltage_flux", 0.0)))
            for branch_index in range(expansion_factor):
                branch_phase = float(branch_index + 1) / float(expansion_factor + 1)
                branch_wave = 0.5 + 0.5 * math.sin(
                    (branch_index + 1)
                    * (0.73 + 0.11 * sequence_persistence_score + 0.07 * temporal_index_overlap)
                )
                branch_cross = 0.5 + 0.5 * math.cos(
                    (branch_index + 1)
                    * (0.61 + 0.09 * voltage_frequency_flux + 0.07 * frequency_voltage_flux)
                )
                features[expanded_row, 0] = clamp01(base[0] * (0.90 + 0.18 * branch_wave) + 0.04 * branch_phase)
                features[expanded_row, 1] = clamp01(base[1] * (0.88 + 0.20 * branch_cross) + 0.05 * branch_phase)
                features[expanded_row, 2] = clamp01(base[2] * (0.90 + 0.16 * branch_wave) + 0.04 * branch_cross)
                features[expanded_row, 3] = clamp01(base[3] * (0.88 + 0.18 * branch_cross) + 0.04 * branch_wave)
                features[expanded_row, 4] = clamp01(base[4] * (0.90 + 0.16 * branch_phase) + 0.05 * branch_wave)
                features[expanded_row, 5] = clamp01(base[5] * (0.92 + 0.12 * branch_cross) + 0.05 * sequence_persistence_score)
                features[expanded_row, 6] = clamp01(base[6] * (0.90 + 0.14 * branch_wave) + 0.05 * branch_phase)
                features[expanded_row, 7] = clamp01(base[7] * (0.90 + 0.14 * branch_cross) + 0.05 * branch_wave)
                features[expanded_row, 8] = clamp01(base[8] * (0.90 + 0.14 * branch_phase) + 0.04 * branch_cross)
                features[expanded_row, 9] = clamp01(base[9] * (0.90 + 0.14 * branch_wave) + 0.04 * branch_phase)
                features[expanded_row, 10] = clamp01(base[10] * (0.90 + 0.12 * branch_cross) + 0.05 * branch_wave)
                features[expanded_row, 11] = clamp01(base[11] * (0.90 + 0.12 * branch_phase) + 0.05 * branch_cross)
                features[expanded_row, 12] = clamp01(base[12] * (0.90 + 0.12 * branch_wave) + 0.05 * branch_phase)
                features[expanded_row, 13] = clamp01(base[13] * (0.90 + 0.12 * branch_cross) + 0.05 * branch_wave)
                features[expanded_row, 14] = clamp01(base[14] * (0.90 + 0.12 * branch_phase) + 0.05 * branch_cross)
                features[expanded_row, 15] = clamp01(base[15] * (0.90 + 0.12 * branch_wave) + 0.05 * branch_phase)
                features[expanded_row, 16] = clamp01(base[16] * (0.90 + 0.12 * branch_cross) + 0.05 * branch_wave)
                features[expanded_row, 17] = clamp01(base[17] * (0.90 + 0.12 * branch_phase) + 0.05 * branch_cross)
                features[expanded_row, 18] = clamp01(base[18] * (0.90 + 0.12 * branch_cross) + 0.05 * branch_wave)
                features[expanded_row, 19] = clamp01(base[19] * (0.90 + 0.12 * branch_phase) + 0.05 * branch_cross)
                features[expanded_row, 20] = clamp01(base[20] * (0.90 + 0.12 * branch_wave) + 0.05 * branch_phase)
                features[expanded_row, 21] = clamp01(base[21] * (0.90 + 0.12 * branch_cross) + 0.05 * branch_wave)
                features[expanded_row, 22] = clamp01(base[22] * (0.92 + 0.10 * branch_phase) + 0.04 * branch_cross)
                features[expanded_row, 23] = clamp01(base[23] * (0.92 + 0.10 * branch_wave) + 0.04 * branch_phase)
                features[expanded_row, 24] = clamp01(base[24] * (0.92 + 0.10 * branch_cross) + 0.04 * branch_wave)
                features[expanded_row, 25] = clamp01(base[25] * (0.92 + 0.10 * branch_phase) + 0.04 * branch_cross)
                features[expanded_row, 26] = clamp01(base[26] * (0.92 + 0.10 * branch_wave) + 0.04 * branch_phase)
                features[expanded_row, 27] = clamp01(base[27] * (0.92 + 0.10 * branch_cross) + 0.04 * branch_wave)
                features[expanded_row, 28] = clamp01(base[28] * (0.92 + 0.10 * branch_phase) + 0.04 * branch_cross)
                features[expanded_row, 29] = clamp01(base[29] * (0.92 + 0.10 * branch_wave) + 0.04 * branch_phase)
                expanded_base_index[expanded_row] = int(candidate_index)
                expanded_branch_index[expanded_row] = int(branch_index)
                expanded_nonce_offsets[expanded_row] = np.uint32(
                    (
                        base_nonce
                        ^ ((branch_index + 1) * 0x45D9F3B)
                        ^ ((phase_length_index + 1) * 0x27D4EB2D)
                        ^ ((temporal_sequence_index + 1) * 0x165667B1)
                    )
                    & 0xFFFFFFFF
                )
                expanded_row += 1
        weights = np.array(
            [
                0.18,
                0.16,
                0.10,
                0.07,
                0.07,
                0.07,
                0.06,
                0.06,
                0.06,
                0.06,
                0.10,
                0.07,
                0.05,
                0.04,
                0.05,
                0.04,
                0.04,
                0.04,
                0.03,
                0.03,
                0.03,
                0.03,
                0.04,
                0.04,
                0.04,
                0.04,
                0.04,
                0.04,
                0.04,
                0.04,
            ],
            dtype=np.float32,
        )
        btc_force_mean = clamp01(
            float(
                np.mean(
                    list(simulation_field_state.get("btc_network_force_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])
                )
            )
        )
        btc_phase_pressure = clamp01(float(simulation_field_state.get("btc_network_phase_pressure", 0.0)))
        btc_amplitude_pressure = clamp01(
            float(simulation_field_state.get("btc_network_amplitude_pressure", 0.0))
        )
        target_vector = np.array(
            [
                float(simulation_field_state.get("target_gate", 0.0)),
                float(target_profile.get("difficulty_window", 0.0)),
                float(simulation_field_state.get("interference_accounting", 0.0)),
                float(simulation_field_state.get("motif_consistency", 0.0)),
                float(simulation_field_state.get("motif_energy", 0.0)),
                float(np.mean(list(target_profile.get("phase_windows", []) or [0.0]))),
                float(btc_amplitude_pressure),
                float(clamp01(0.52 * btc_phase_pressure + 0.48 * btc_amplitude_pressure)),
                float(btc_force_mean),
                float(btc_phase_pressure),
                float(
                    clamp01(
                        0.70 * float(simulation_field_state.get("sequence_persistence_score", 0.0))
                        + 0.30 * trace_temporal_persistence
                    )
                ),
                float(
                    clamp01(
                        0.70 * float(simulation_field_state.get("temporal_index_overlap", 0.0))
                        + 0.30 * trace_temporal_overlap
                    )
                ),
                float(
                    clamp01(
                        0.68 * float(simulation_field_state.get("voltage_frequency_flux", 0.0))
                        + 0.32 * trace_voltage_frequency_flux
                    )
                ),
                float(
                    clamp01(
                        0.68 * float(simulation_field_state.get("frequency_voltage_flux", 0.0))
                        + 0.32 * trace_frequency_voltage_flux
                    )
                ),
                float(simulation_field_state.get("kernel_balance_mean", 0.0)),
                float(simulation_field_state.get("kernel_harmonic_resonance_mean", 0.0)),
                float(simulation_field_state.get("kernel_retro_gain_mean", 0.0)),
                float(simulation_field_state.get("kernel_resonance_alignment_mean", 0.0)),
                float(simulation_field_state.get("kernel_delta_gate_mean", 0.0)),
                float(simulation_field_state.get("kernel_delta_memory_mean", 0.0)),
                float(simulation_field_state.get("kernel_delta_flux_mean", 0.0)),
                float(
                    clamp01(
                        0.76 * float(simulation_field_state.get("kernel_delta_phase_alignment_mean", 0.0))
                        + 0.24 * trace_alignment
                    )
                ),
                float(target_prefix_vector[0]),
                float(target_prefix_vector[1]),
                float(target_prefix_vector[2]),
                float(target_prefix_vector[3]),
                float(target_relative_spatial_field[0]),
                float(target_relative_spatial_field[1]),
                float(target_relative_spatial_field[2]),
                float(target_relative_spatial_field[3]),
            ],
            dtype=np.float32,
        )
        d_features = numba_cuda.to_device(features)
        d_weights = upload_array_to_vram("cuda_weights", weights)
        d_target = upload_array_to_vram("cuda_target_vector", target_vector)
        d_trace_axis = upload_array_to_vram("trace_axis_vector_runtime", trace_axis_vector)
        d_trace_summary = upload_array_to_vram(
            "trace_summary_vector_runtime", trace_summary_vector
        )
        d_scores = numba_cuda.device_array(features.shape[0], dtype=np.float32)
        d_nonce_biases = numba_cuda.device_array(features.shape[0], dtype=np.uint32)
        threads_per_block = 256
        blocks_per_grid = max(1, (features.shape[0] + threads_per_block - 1) // threads_per_block)
        telemetry["threads_per_block"] = int(threads_per_block)
        telemetry["blocks_per_grid"] = int(blocks_per_grid)
        temporal_candidate_cuda_kernel[blocks_per_grid, threads_per_block](
            d_features,
            d_weights,
            d_target,
            d_trace_axis,
            d_trace_summary,
            float(harmonic_compute_weight),
            d_scores,
            d_nonce_biases,
        )
        numba_cuda.synchronize()
        scores = d_scores.copy_to_host()
        nonce_biases = d_nonce_biases.copy_to_host()
        normalized_scores = np.clip(0.5 + 0.5 * np.tanh(scores.astype(np.float64)), 0.0, 1.0)
        base_best_scores = np.zeros(candidate_count, dtype=np.float64)
        base_best_biases = np.zeros(candidate_count, dtype=np.uint32)
        base_best_branch = np.zeros(candidate_count, dtype=np.int32)
        for index in range(expanded_eval_count):
            base_index = int(expanded_base_index[index])
            score_value = float(normalized_scores[index])
            if score_value >= float(base_best_scores[base_index]):
                base_best_scores[base_index] = score_value
                base_best_biases[base_index] = np.uint32(
                    (int(nonce_biases[index]) ^ int(expanded_nonce_offsets[index])) & 0xFFFFFFFF
                )
                base_best_branch[base_index] = int(expanded_branch_index[index])
        for index, candidate in enumerate(candidate_pool):
            candidate["cuda_temporal_score"] = float(base_best_scores[index])
            candidate["cuda_nonce_bias"] = int(base_best_biases[index]) & 0xFFFFFFFF
            candidate["cuda_branch_index"] = int(base_best_branch[index])
        expanded_keep_floor = int(
            round(16384.0 + 4096.0 * phase_volume_gain + 2048.0 * kernel_volume_gain)
        )
        expanded_keep_scale = int(
            round(
                float(candidate_count)
                * (
                    6.0
                    + 2.0 * coupling_volume_gain
                    + 2.0 * phase_volume_gain
                    + 1.0 * kernel_volume_gain
                )
            )
        )
        expanded_keep_count = min(expanded_eval_count, max(expanded_keep_scale, expanded_keep_floor))
        telemetry["expanded_keep_count"] = int(expanded_keep_count)
        if expanded_keep_count > 0:
            top_indices = np.argpartition(normalized_scores, -expanded_keep_count)[-expanded_keep_count:]
            top_indices = top_indices[np.argsort(normalized_scores[top_indices])[::-1]]
            expanded_candidates: list[dict[str, Any]] = []
            seen_nonces: set[int] = {int(candidate.get("nonce", 0)) & 0xFFFFFFFF for candidate in candidate_pool}
            for rank, expanded_index in enumerate(top_indices):
                base_index = int(expanded_base_index[int(expanded_index)])
                base_candidate = dict(candidate_pool[base_index])
                nonce_offset = int(
                    (
                        int(expanded_nonce_offsets[int(expanded_index)])
                        ^ int(nonce_biases[int(expanded_index)])
                        ^ ((rank + 1) * 0x9E3779B1)
                    )
                    & 0xFFFFFFFF
                )
                probe_nonce = (int(base_candidate.get("nonce", 0)) + nonce_offset) & 0xFFFFFFFF
                if probe_nonce in seen_nonces:
                    continue
                seen_nonces.add(probe_nonce)
                score_value = float(normalized_scores[int(expanded_index)])
                branch_index = int(expanded_branch_index[int(expanded_index)])
                base_candidate["nonce"] = int(probe_nonce)
                base_candidate["cuda_temporal_score"] = float(score_value)
                base_candidate["cuda_nonce_bias"] = int(
                    (int(nonce_biases[int(expanded_index)]) ^ nonce_offset) & 0xFFFFFFFF
                )
                base_candidate["cuda_branch_index"] = int(branch_index)
                base_candidate["cuda_expanded"] = True
                base_candidate["cluster_id"] = (
                    f"{str(base_candidate.get('cluster_id', 'cluster'))}"
                    f"-cx{branch_index:02d}"
                )
                base_candidate["target_alignment"] = clamp01(
                    0.84 * float(base_candidate.get("target_alignment", 0.0))
                    + 0.16 * score_value
                )
                base_candidate["coherence_peak"] = clamp01(
                    0.82 * float(base_candidate.get("coherence_peak", 0.0))
                    + 0.18 * score_value
                )
                base_candidate["sequence_persistence_score"] = clamp01(
                    0.88 * float(base_candidate.get("sequence_persistence_score", 0.0))
                    + 0.12 * score_value
                )
                ensure_temporal_decode_metrics(base_candidate)
                expanded_candidates.append(base_candidate)
            candidate_pool.extend(expanded_candidates)
        device_name = ""
        try:
            device = numba_cuda.get_current_device()
            raw_name = getattr(device, "name", "")
            device_name = raw_name.decode("ascii", errors="ignore") if isinstance(raw_name, bytes) else str(raw_name)
        except Exception:
            device_name = ""
        telemetry["enabled"] = True
        telemetry["device"] = device_name
        telemetry["score_mean"] = float(np.mean(normalized_scores))
        telemetry["score_max"] = float(np.max(normalized_scores))
        telemetry["candidate_count"] = int(len(candidate_pool))
        return telemetry
    except Exception as exc:
        telemetry["reason"] = f"cuda_kernel_failed:{type(exc).__name__}"
        return telemetry


def build_target_probe_offsets(
    candidate: dict[str, Any],
    target_profile: dict[str, Any],
    pulse_index: int,
    cluster_rank: int = 0,
    cluster_weight: float = 0.0,
    prefix_bias: int = 0,
) -> list[int]:
    interval_windows = list(target_profile.get("interval_windows", []) or [0.5])
    phase_windows = list(target_profile.get("phase_windows", []) or interval_windows)
    phase_length_pressure = clamp01(float(candidate.get("phase_length_pressure", 0.0)))
    phase_alignment_score = clamp01(
        float(candidate.get("phase_alignment_score", phase_length_pressure))
    )
    phase_confinement_cost = clamp01(
        float(candidate.get("phase_confinement_cost", 1.0 - phase_alignment_score))
    )
    phase_length_span = clamp01(float(candidate.get("phase_length_span", 0.0)))
    phase_length_index = int(candidate.get("phase_length_index", 0))
    temporal_sequence_index = int(candidate.get("temporal_sequence_index", pulse_index))
    temporal_sequence_length = max(1, int(candidate.get("temporal_sequence_length", len(phase_windows))))
    temporal_index_overlap = clamp01(float(candidate.get("temporal_index_overlap", 0.0)))
    sequence_persistence_score = clamp01(float(candidate.get("sequence_persistence_score", 0.0)))
    voltage_frequency_flux = clamp01(float(candidate.get("voltage_frequency_flux", 0.0)))
    frequency_voltage_flux = clamp01(float(candidate.get("frequency_voltage_flux", 0.0)))
    cuda_nonce_bias = int(candidate.get("cuda_nonce_bias", 0)) & 0xFFFFFFFF
    pressure_scale = max(
        0.32,
        0.42
        + 0.38 * phase_alignment_score
        + 0.20 * (1.0 - phase_confinement_cost)
        + 0.20 * phase_length_span
        + 0.10 * sequence_persistence_score
        + 0.08 * temporal_index_overlap,
    )
    interval_seed = 1 + int(round(float(candidate.get("target_interval", 0.0)) * 4093.0 * pressure_scale))
    phase_seed = 1 + int(round(float(candidate.get("target_alignment", 0.0)) * 8191.0 * pressure_scale))
    motif_seed = 1 + int(
        round(
            (
                float(candidate.get("motif_alignment", 0.0))
                + float(candidate.get("row_activation", 0.0))
            )
            * 12289.0
            * pressure_scale
        )
    )
    resonance_seed = 1 + int(round(float(candidate.get("interference_resonance", 0.0)) * 6143.0 * pressure_scale))
    cascade_seed = 1 + int(round(float(candidate.get("cascade_activation", 0.0)) * 2047.0 * pressure_scale))
    target_window_seed = 1 + int(
        round(
            float(interval_windows[pulse_index % len(interval_windows)]) * 1531.0
            + float(phase_windows[pulse_index % len(phase_windows)]) * 1187.0
        )
    )
    phase_length_seed = 1 + int(round((phase_length_index + 1) * (11.0 + 13.0 * pressure_scale)))
    pressure_seed = 1 + int(
        round(
            (
                0.24
                + phase_alignment_score
                + phase_length_span
                + phase_confinement_cost
            )
            * 887.0
        )
    )
    temporal_seed = 1 + int(
        round(
            (temporal_sequence_index + 1)
            * (9.0 + 11.0 * temporal_index_overlap + 7.0 * sequence_persistence_score)
        )
    )
    overlap_seed = 1 + int(round((0.18 + temporal_index_overlap) * 1531.0))
    persistence_seed = 1 + int(round((0.24 + sequence_persistence_score) * 1777.0))
    flux_seed = 1 + int(
        round(
            (
                0.18
                + 0.44 * voltage_frequency_flux
                + 0.38 * frequency_voltage_flux
            )
            * 1901.0
        )
    )
    cuda_seed = 1 + int((cuda_nonce_bias % 8191))
    temporal_span_seed = 1 + int(
        round(float(temporal_sequence_length) * (1.0 + 0.10 * sequence_persistence_score))
    )
    prefix_seed = max(1, int(prefix_bias) + 1)
    fine_seed = max(1, prefix_seed * (cluster_rank + 1) * 17)
    cluster_seed = 1 + int(round((0.35 + clamp01(cluster_weight)) * 4091.0))
    focus_seed = 1 + int(round((0.20 + clamp01(cluster_weight)) * 1021.0))
    base_offsets = [
        0,
        phase_length_seed,
        -phase_length_seed,
        pressure_seed,
        -pressure_seed,
        temporal_seed,
        -temporal_seed,
        overlap_seed,
        -overlap_seed,
        persistence_seed,
        -persistence_seed,
        flux_seed,
        -flux_seed,
        cuda_seed,
        -cuda_seed,
        fine_seed,
        -fine_seed,
        interval_seed,
        -interval_seed,
        phase_seed,
        -phase_seed,
        motif_seed,
        -motif_seed,
        resonance_seed,
        -resonance_seed,
        interval_seed + phase_seed,
        -(interval_seed + phase_seed),
        temporal_seed + persistence_seed,
        -(temporal_seed + persistence_seed),
        flux_seed + overlap_seed,
        -(flux_seed + overlap_seed),
        cuda_seed + phase_length_seed,
        -(cuda_seed + phase_length_seed),
        motif_seed + cascade_seed + target_window_seed,
        -(motif_seed + cascade_seed + target_window_seed),
    ]
    if clamp01(cluster_weight) >= 0.30:
        base_offsets.extend(
            [
                cluster_seed,
                -cluster_seed,
                cluster_seed + fine_seed,
                -(cluster_seed + fine_seed),
                cluster_seed + phase_length_seed,
                -(cluster_seed + phase_length_seed),
                cluster_seed + interval_seed,
                -(cluster_seed + interval_seed),
                temporal_seed + cluster_seed,
                -(temporal_seed + cluster_seed),
                flux_seed + temporal_span_seed,
                -(flux_seed + temporal_span_seed),
            ]
        )
    if clamp01(cluster_weight) >= 0.48:
        base_offsets.extend(
            [
                focus_seed,
                -focus_seed,
                phase_seed + cluster_seed,
                -(phase_seed + cluster_seed),
                pressure_seed + focus_seed,
                -(pressure_seed + focus_seed),
                resonance_seed + focus_seed,
                -(resonance_seed + focus_seed),
                persistence_seed + focus_seed,
                -(persistence_seed + focus_seed),
                overlap_seed + temporal_seed,
                -(overlap_seed + temporal_seed),
            ]
        )
    offsets: list[int] = []
    seen: set[int] = set()
    for offset in base_offsets:
        normalized = int(offset)
        if normalized in seen:
            continue
        seen.add(normalized)
        offsets.append(normalized)
    return offsets


def build_phase_length_event_profile(
    target_profile: dict[str, Any],
    pulse_index: int,
    packet_idx: int,
    carrier_idx: int,
    simulation_field_state: dict[str, Any],
    target_phase_window: float,
    target_carrier_window: float,
    detected_phase_signature: float,
    detected_row_activation: float,
    detected_motif_alignment: float,
    detected_cascade_activation: float,
    field_amplitude_bias: float,
    target_amplitude_cap: float,
    kernel_balance_score: float = 0.0,
    harmonic_resonance_score: float = 0.0,
    retro_temporal_gain: float = 0.0,
    kernel_phase_alignment: float = 0.0,
    kernel_delta_gate: float = 0.0,
    kernel_delta_memory: float = 0.0,
    kernel_delta_flux: float = 0.0,
    kernel_delta_phase_alignment: float = 0.0,
) -> dict[str, float]:
    interval_windows = list(target_profile.get("interval_windows", []) or [target_carrier_window])
    phase_windows = list(target_profile.get("phase_windows", []) or interval_windows)
    network_algorithm = dict(target_profile.get("network_algorithm", {}) or {})
    btc_force_vector = np.array(
        list(
            simulation_field_state.get(
                "btc_network_force_vector",
                network_algorithm.get("force_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if btc_force_vector.shape[0] != 4:
        btc_force_vector = np.zeros(4, dtype=np.float64)
    btc_phase_turns = [
        wrap_turns(float(value))
        for value in list(
            simulation_field_state.get(
                "btc_network_phase_turns",
                network_algorithm.get("phase_turns", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        )
    ]
    if len(btc_phase_turns) < 4:
        btc_phase_turns = [0.0, 0.0, 0.0, 0.0]
    btc_phase_pressure_curve = [
        clamp01(float(value))
        for value in list(network_algorithm.get("phase_pressure_curve", phase_windows) or phase_windows)
    ]
    if not btc_phase_pressure_curve:
        btc_phase_pressure_curve = [0.0]
    btc_amplitude_pressure_curve = [
        clamp01(float(value))
        for value in list(network_algorithm.get("amplitude_pressure_curve", phase_windows) or phase_windows)
    ]
    if not btc_amplitude_pressure_curve:
        btc_amplitude_pressure_curve = [0.0]
    btc_network_phase_pressure = clamp01(
        float(
            simulation_field_state.get(
                "btc_network_phase_pressure",
                network_algorithm.get("phase_pressure", 0.0),
            )
        )
    )
    btc_network_amplitude_pressure = clamp01(
        float(
            simulation_field_state.get(
                "btc_network_amplitude_pressure",
                network_algorithm.get("amplitude_pressure", 0.0),
            )
        )
    )
    btc_network_algorithm_bias = clamp01(
        float(
            simulation_field_state.get(
                "btc_network_algorithm_bias",
                network_algorithm.get("algorithm_bias", 0.0),
            )
        )
    )
    temporal_sequence_indexes = [
        int(index)
        for index in list(simulation_field_state.get("temporal_sequence_indexes", []) or [])
    ]
    temporal_sequence_index = int(
        simulation_field_state.get("temporal_sequence_index", pulse_index % max(len(phase_windows), 1))
    )
    sequence_persistence_score = clamp01(
        float(simulation_field_state.get("sequence_persistence_score", 0.0))
    )
    temporal_index_overlap = clamp01(
        float(simulation_field_state.get("temporal_index_overlap", 0.0))
    )
    voltage_frequency_flux = clamp01(
        float(simulation_field_state.get("voltage_frequency_flux", 0.0))
    )
    frequency_voltage_flux = clamp01(
        float(simulation_field_state.get("frequency_voltage_flux", 0.0))
    )
    gpu_pulse_feedback = dict(simulation_field_state.get("gpu_pulse_feedback", {}) or {})
    feedback_phase_anchor_turns = wrap_turns(float(gpu_pulse_feedback.get("phase_anchor_turns", 0.0)))
    feedback_phase_alignment = clamp01(float(gpu_pulse_feedback.get("phase_alignment", 0.0)))
    feedback_memory_proxy = clamp01(float(gpu_pulse_feedback.get("memory_proxy", 0.0)))
    feedback_flux_proxy = clamp01(float(gpu_pulse_feedback.get("flux_proxy", 0.0)))
    feedback_stability_proxy = clamp01(float(gpu_pulse_feedback.get("stability_proxy", 0.0)))
    feedback_temporal_drive = clamp01(float(gpu_pulse_feedback.get("temporal_drive", 0.0)))
    feedback_frequency = clamp01(float(gpu_pulse_feedback.get("frequency_observable", 0.0)))
    feedback_amplitude = clamp01(float(gpu_pulse_feedback.get("amplitude_observable", 0.0)))
    feedback_voltage = clamp01(float(gpu_pulse_feedback.get("voltage_observable", 0.0)))
    feedback_current = clamp01(float(gpu_pulse_feedback.get("current_observable", 0.0)))
    feedback_temperature_norm = clamp01(float(gpu_pulse_feedback.get("temperature_norm", 0.0)))
    feedback_thermal_headroom = clamp01(
        float(gpu_pulse_feedback.get("thermal_headroom_norm", 1.0 - feedback_temperature_norm))
    )
    feedback_temperature_velocity = clamp01(
        float(gpu_pulse_feedback.get("temperature_velocity_norm", 0.0))
    )
    feedback_environment_pressure = clamp01(float(gpu_pulse_feedback.get("environment_pressure", 0.0)))
    feedback_environment_stability = clamp01(float(gpu_pulse_feedback.get("environment_stability", 0.0)))
    feedback_latency_norm = clamp01(float(gpu_pulse_feedback.get("latency_norm", 0.0)))
    feedback_latency_gate = clamp01(
        float(gpu_pulse_feedback.get("temporal_latency_gate", 1.0 - feedback_latency_norm))
    )
    feedback_delta_target_vector = np.array(
        list(simulation_field_state.get("feedback_delta_target_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if feedback_delta_target_vector.shape[0] != 4:
        feedback_delta_target_vector = np.zeros(4, dtype=np.float64)
    feedback_delta_phase_retention = clamp01(
        float(simulation_field_state.get("feedback_delta_phase_retention", 0.0))
    )
    feedback_delta_response_gate = clamp01(
        float(simulation_field_state.get("feedback_delta_response_gate", 0.0))
    )
    feedback_delta_memory_retention = clamp01(
        float(simulation_field_state.get("feedback_delta_memory_retention", 0.0))
    )
    feedback_delta_latency_gate = clamp01(float(simulation_field_state.get("feedback_delta_latency_gate", 0.0)))
    feedback_delta_window_span_norm = clamp01(
        float(simulation_field_state.get("feedback_delta_window_span_norm", 0.0))
    )
    feedback_delta_window_density = clamp01(
        float(simulation_field_state.get("feedback_delta_window_density", 0.0))
    )
    feedback_delta_window_latency_alignment = clamp01(
        float(simulation_field_state.get("feedback_delta_window_latency_alignment", 0.0))
    )
    feedback_delta_environment_pressure = clamp01(
        float(simulation_field_state.get("feedback_delta_environment_pressure", 0.0))
    )
    feedback_delta_thermal_headroom = clamp01(
        float(simulation_field_state.get("feedback_delta_thermal_headroom", 0.0))
    )
    trace_state = dict(simulation_field_state.get("substrate_trace_state", {}) or {})
    trace_phase_anchor_turns = wrap_turns(
        float(trace_state.get("trace_phase_anchor_turns", feedback_phase_anchor_turns))
    )
    trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
    trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
    trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
    trace_memory = clamp01(float(trace_state.get("trace_memory", feedback_memory_proxy)))
    trace_flux = clamp01(float(trace_state.get("trace_flux", feedback_flux_proxy)))
    trace_stability = clamp01(
        float(trace_state.get("trace_stability", feedback_stability_proxy))
    )
    trace_temporal_persistence = clamp01(
        float(trace_state.get("trace_temporal_persistence", sequence_persistence_score))
    )
    trace_temporal_overlap = clamp01(
        float(trace_state.get("trace_temporal_overlap", temporal_index_overlap))
    )
    trace_voltage_frequency_flux = clamp01(
        float(trace_state.get("trace_voltage_frequency_flux", voltage_frequency_flux))
    )
    trace_frequency_voltage_flux = clamp01(
        float(trace_state.get("trace_frequency_voltage_flux", frequency_voltage_flux))
    )
    kernel_balance_score = clamp01(float(kernel_balance_score))
    harmonic_resonance_score = clamp01(float(harmonic_resonance_score))
    retro_temporal_gain = clamp01(float(retro_temporal_gain))
    kernel_phase_alignment = clamp01(float(kernel_phase_alignment))
    kernel_delta_gate = clamp01(float(kernel_delta_gate))
    kernel_delta_memory = clamp01(float(kernel_delta_memory))
    kernel_delta_flux = clamp01(float(kernel_delta_flux))
    kernel_delta_phase_alignment = clamp01(float(kernel_delta_phase_alignment))
    envelope_history = list(gpu_pulse_feedback.get("envelope_history", []) or [])
    history_phase_anchor = feedback_phase_anchor_turns
    history_memory_window = feedback_memory_proxy
    history_flux_window = feedback_flux_proxy
    history_stability_window = feedback_stability_proxy
    if envelope_history:
        history_phase_anchor = wrap_turns(
            float(np.mean([float(item.get("phase_anchor_turns", feedback_phase_anchor_turns)) for item in envelope_history]))
        )
        history_memory_window = clamp01(
            float(np.mean([float(item.get("memory_proxy", feedback_memory_proxy)) for item in envelope_history]))
        )
        history_flux_window = clamp01(
            float(np.mean([float(item.get("flux_proxy", feedback_flux_proxy)) for item in envelope_history]))
        )
        history_stability_window = clamp01(
            float(
                np.mean(
                    [float(item.get("stability_proxy", feedback_stability_proxy)) for item in envelope_history]
                )
            )
        )
    temporal_sequence_length = max(
        1,
        int(simulation_field_state.get("temporal_sequence_length", len(phase_windows))),
    )
    if temporal_sequence_indexes:
        event_index = int(
            temporal_sequence_indexes[
                (packet_idx + carrier_idx + pulse_index) % len(temporal_sequence_indexes)
            ]
        )
    else:
        event_index = int((pulse_index * 37 + packet_idx * 11 + carrier_idx) % max(len(phase_windows), 1))
    event_window = float(phase_windows[event_index % len(phase_windows)])
    sequence_window = float(phase_windows[temporal_sequence_index % len(phase_windows)])
    event_feedback_phase = wrap_turns(
        feedback_phase_anchor_turns
        + float(event_index) / max(float(temporal_sequence_length), 1.0)
        * (0.10 + 0.22 * feedback_frequency + 0.10 * feedback_current)
        + float(carrier_idx + 1) * (0.005 + 0.018 * feedback_amplitude)
        + float(packet_idx + 1) * (0.003 + 0.012 * feedback_voltage)
        + float(feedback_delta_target_vector[1]) * (0.04 + 0.08 * feedback_delta_phase_retention)
        + kernel_delta_phase_alignment * 0.06
    )
    event_feedback_alignment = clamp01(
        0.42 * turn_alignment(event_feedback_phase, sequence_window)
        + 0.24 * turn_alignment(event_feedback_phase, history_phase_anchor)
        + 0.12 * turn_alignment(event_feedback_phase, trace_phase_anchor_turns)
        + 0.18 * feedback_delta_phase_retention
        + 0.16 * kernel_delta_phase_alignment
        + 0.10 * trace_alignment
    )
    btc_phase_pressure = clamp01(
        0.42 * float(btc_phase_pressure_curve[event_index % len(btc_phase_pressure_curve)])
        + 0.18 * turn_alignment(event_feedback_phase, btc_phase_turns[event_index % 4])
        + 0.14 * btc_network_phase_pressure
        + 0.12 * feedback_phase_alignment
        + 0.08 * kernel_phase_alignment
        + 0.06 * btc_network_algorithm_bias
    )
    event_feedback_memory = clamp01(
        0.34 * feedback_memory_proxy
        + 0.14 * history_memory_window
        + 0.14 * trace_memory
        + 0.12 * feedback_temporal_drive
        + 0.10 * feedback_stability_proxy
        + 0.10 * event_feedback_alignment
        + 0.08 * feedback_delta_memory_retention
        + 0.06 * float(feedback_delta_target_vector[0])
        + 0.06 * feedback_delta_response_gate
        + 0.06 * feedback_environment_stability
        + 0.04 * feedback_latency_gate
        + 0.04 * feedback_thermal_headroom
        + 0.04 * feedback_delta_window_latency_alignment
        + 0.04 * trace_support
    )
    event_window = clamp01(
        0.40 * event_window
        + 0.16 * sequence_window
        + 0.10 * target_phase_window
        + 0.08 * sequence_persistence_score
        + 0.10 * event_feedback_alignment
        + 0.08 * feedback_phase_alignment
        + 0.08 * kernel_phase_alignment
        + 0.06 * trace_alignment
        + 0.06 * harmonic_resonance_score
        + 0.04 * float(feedback_delta_target_vector[1])
        + 0.04 * kernel_delta_gate
        + 0.04 * kernel_delta_phase_alignment
        + 0.04 * feedback_latency_gate
        + 0.04 * feedback_thermal_headroom
        + 0.02 * (1.0 - feedback_environment_pressure)
        + 0.03 * feedback_delta_window_latency_alignment
        + 0.03 * (1.0 - feedback_delta_window_span_norm)
        + 0.04 * trace_resonance
    )
    prev_window = float(phase_windows[(event_index - 1) % len(phase_windows)])
    next_window = float(phase_windows[(event_index + 1) % len(phase_windows)])
    event_span = clamp01(
        abs(next_window - prev_window) * 0.34
        + abs(event_window - target_phase_window) * 0.28
        + abs(sequence_window - event_window) * 0.18
        + 0.10 * (1.0 - event_feedback_alignment)
        + 0.10 * turn_distance(history_phase_anchor, feedback_phase_anchor_turns)
    )
    event_temporal_overlap = clamp01(
        0.36 * temporal_index_overlap
        + 0.20 * event_feedback_alignment
        + 0.12 * feedback_phase_alignment
        + 0.12 * feedback_temporal_drive
        + 0.10 * event_feedback_memory
        + 0.10 * trace_temporal_overlap
        + 0.10 * turn_alignment(event_feedback_phase, feedback_phase_anchor_turns)
        + 0.08 * retro_temporal_gain
        + 0.06 * kernel_phase_alignment
        + 0.06 * float(feedback_delta_target_vector[1])
        + 0.06 * feedback_delta_phase_retention
        + 0.06 * kernel_delta_gate
        + 0.04 * kernel_delta_phase_alignment
        + 0.04 * feedback_latency_gate
        + 0.03 * feedback_delta_latency_gate
        + 0.04 * trace_alignment
    )
    event_sequence_persistence = clamp01(
        0.26 * sequence_persistence_score
        + 0.18 * event_feedback_memory
        + 0.12 * trace_temporal_persistence
        + 0.10 * feedback_stability_proxy
        + 0.10 * event_feedback_alignment
        + 0.10 * history_stability_window
        + 0.08 * trace_stability
        + 0.08 * feedback_temporal_drive
        + 0.08 * retro_temporal_gain
        + 0.06 * kernel_balance_score
        + 0.06 * float(feedback_delta_target_vector[0])
        + 0.06 * feedback_delta_response_gate
        + 0.06 * kernel_delta_memory
        + 0.06 * feedback_environment_stability
        + 0.04 * feedback_latency_gate
        + 0.03 * feedback_delta_thermal_headroom
        + 0.03 * feedback_delta_window_latency_alignment
        + 0.04 * trace_support
    )
    event_voltage_frequency_flux = clamp01(
        0.28 * voltage_frequency_flux
        + 0.16 * trace_voltage_frequency_flux
        + 0.18 * feedback_voltage
        + 0.16 * feedback_frequency
        + 0.10 * feedback_flux_proxy
        + 0.08 * event_feedback_alignment
        + 0.08 * history_flux_window
        + 0.08 * trace_flux
        + 0.06 * harmonic_resonance_score
        + 0.06 * float(feedback_delta_target_vector[2])
        + 0.06 * kernel_delta_flux
        + 0.04 * feedback_environment_pressure
        + 0.03 * feedback_temperature_velocity
        + 0.03 * feedback_delta_window_density
        + 0.04 * trace_alignment
    )
    event_frequency_voltage_flux = clamp01(
        0.28 * frequency_voltage_flux
        + 0.16 * trace_frequency_voltage_flux
        + 0.20 * feedback_frequency
        + 0.16 * feedback_voltage
        + 0.08 * feedback_phase_alignment
        + 0.08 * feedback_temporal_drive
        + 0.06 * event_feedback_alignment
        + 0.08 * trace_flux
        + 0.06 * harmonic_resonance_score
        + 0.06 * float(feedback_delta_target_vector[3])
        + 0.06 * kernel_delta_flux
        + 0.04 * feedback_latency_gate
        + 0.03 * feedback_delta_latency_gate
        + 0.03 * feedback_delta_window_latency_alignment
        + 0.04 * trace_temporal_persistence
    )
    phase_amplitude_pressure = clamp01(
        0.40 * float(btc_amplitude_pressure_curve[event_index % len(btc_amplitude_pressure_curve)])
        + 0.24 * btc_phase_pressure
        + 0.12 * btc_network_amplitude_pressure
        + 0.10 * field_amplitude_bias
        + 0.08 * target_carrier_window
        + 0.06 * event_feedback_memory
        + 0.06 * trace_memory
    )
    btc_force_alignment = clamp01(
        vector_similarity(
            np.array(
                [
                    event_window,
                    sequence_window,
                    event_temporal_overlap,
                    event_sequence_persistence,
                ],
                dtype=np.float64,
            ),
            btc_force_vector,
        )
    )
    pressure = clamp01(
        0.14 * event_window
        + 0.10 * sequence_window
        + 0.12 * (1.0 - event_span)
        + 0.12 * target_phase_window
        + 0.10 * target_carrier_window
        + 0.08 * detected_phase_signature
        + 0.06 * detected_row_activation
        + 0.06 * detected_motif_alignment
        + 0.06 * detected_cascade_activation
        + 0.08 * event_sequence_persistence
        + 0.07 * event_temporal_overlap
        + 0.06 * event_voltage_frequency_flux
        + 0.05 * event_frequency_voltage_flux
        + 0.04 * event_feedback_alignment
        + 0.04 * event_feedback_memory
        + 0.04 * trace_resonance
        + 0.05 * kernel_balance_score
        + 0.04 * harmonic_resonance_score
        + 0.03 * retro_temporal_gain
        + 0.03 * kernel_phase_alignment
        + 0.04 * kernel_delta_gate
        + 0.03 * kernel_delta_memory
        + 0.03 * kernel_delta_flux
        + 0.02 * kernel_delta_phase_alignment
        + 0.03 * feedback_environment_pressure
        + 0.03 * feedback_latency_gate
        + 0.02 * feedback_delta_environment_pressure
        + 0.02 * feedback_delta_window_latency_alignment
        + 0.08 * btc_phase_pressure
        + 0.06 * btc_force_alignment
        + 0.04 * btc_network_algorithm_bias
    )
    phase_alignment_score = float(pressure)
    phase_confinement_cost = clamp01(
        0.30 * (1.0 - phase_alignment_score)
        + 0.18 * event_span
        + 0.12 * feedback_environment_pressure
        + 0.10 * (1.0 - feedback_latency_gate)
        + 0.10 * (1.0 - event_feedback_alignment)
        + 0.08 * (1.0 - event_feedback_memory)
        + 0.06 * (1.0 - btc_force_alignment)
        + 0.06 * (1.0 - trace_resonance)
    )
    amplitude_pressure = clamp01(
        0.30 * phase_alignment_score
        + 0.18 * field_amplitude_bias
        + 0.14 * target_carrier_window
        + 0.10 * detected_row_activation
        + 0.08 * detected_motif_alignment
        + 0.06 * (1.0 - event_span)
        + 0.05 * event_sequence_persistence
        + 0.05 * event_feedback_memory
        + 0.04 * event_voltage_frequency_flux
        + 0.03 * event_frequency_voltage_flux
        + 0.03 * event_feedback_alignment
        + 0.04 * kernel_balance_score
        + 0.03 * harmonic_resonance_score
        + 0.03 * kernel_delta_gate
        + 0.02 * kernel_delta_memory
        + 0.02 * kernel_delta_flux
        + 0.03 * feedback_environment_pressure
        + 0.02 * feedback_thermal_headroom
        + 0.02 * feedback_latency_gate
        + 0.02 * feedback_delta_window_density
        + 0.14 * phase_amplitude_pressure
        + 0.08 * btc_force_alignment
        + 0.06 * btc_network_amplitude_pressure
        + 0.08 * (1.0 - phase_confinement_cost)
    )
    btc_phase_alignment = float(btc_phase_pressure)
    btc_phase_cost = clamp01(
        0.56 * (1.0 - btc_phase_alignment)
        + 0.24 * phase_confinement_cost
        + 0.12 * (1.0 - target_phase_window)
        + 0.08 * (1.0 - btc_force_alignment)
    )
    amplitude_phase_pressure = clamp01(
        0.34 * phase_alignment_score
        + 0.30 * phase_amplitude_pressure
        + 0.20 * (1.0 - phase_confinement_cost)
        + 0.16 * btc_force_alignment
    )
    amplitude_cap = clamp01(
        float(target_amplitude_cap)
        * (
            0.44
            + 0.28 * amplitude_pressure
            + 0.10 * (1.0 - btc_phase_alignment)
            + 0.08 * (1.0 - phase_confinement_cost)
            + 0.10 * feedback_delta_response_gate
            + 0.06 * kernel_delta_gate
            + 0.04 * feedback_latency_gate
            + 0.04 * (1.0 - feedback_environment_pressure)
            + 0.04 * feedback_thermal_headroom
            + 0.04 * feedback_delta_window_latency_alignment
        )
    )
    return {
        "phase_length_index": float(event_index),
        "phase_length_window": float(event_window),
        "phase_length_span": float(event_span),
        "phase_length_pressure": float(phase_alignment_score),
        "phase_alignment_score": float(phase_alignment_score),
        "phase_confinement_cost": float(phase_confinement_cost),
        "phase_amplitude_cap": float(amplitude_cap),
        "phase_amplitude_pressure": float(phase_amplitude_pressure),
        "amplitude_phase_pressure": float(amplitude_phase_pressure),
        "btc_phase_pressure": float(btc_phase_alignment),
        "btc_phase_alignment": float(btc_phase_alignment),
        "btc_phase_cost": float(btc_phase_cost),
        "btc_force_alignment": float(btc_force_alignment),
        "temporal_sequence_index": float(temporal_sequence_index),
        "temporal_sequence_length": float(temporal_sequence_length),
        "temporal_index_overlap": float(event_temporal_overlap),
        "sequence_persistence_score": float(event_sequence_persistence),
        "voltage_frequency_flux": float(event_voltage_frequency_flux),
        "frequency_voltage_flux": float(event_frequency_voltage_flux),
        "feedback_event_alignment": float(event_feedback_alignment),
        "retro_delta_gate": float(kernel_delta_gate),
        "retro_delta_memory": float(kernel_delta_memory),
        "retro_delta_flux": float(kernel_delta_flux),
        "retro_delta_phase_alignment": float(kernel_delta_phase_alignment),
    }


def build_temporal_sequence_accounting(
    target_profile: dict[str, Any],
    pulse_index: int,
    lattice_calibration: dict[str, Any],
    pulse_sweep: dict[str, Any],
    previous_field_state: dict[str, Any] | None = None,
    encoded_event_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_field_state = dict(previous_field_state or {})
    encoded_event_model = dict(encoded_event_model or {})
    field_environment = dict(lattice_calibration.get("field_environment", {}) or {})
    wave_step_field = dict(lattice_calibration.get("wave_step_field", {}) or {})
    axis_wave_norm = dict(lattice_calibration.get("axis_wave_norm", {}) or {})
    axis_step_interval = dict(lattice_calibration.get("axis_step_interval", {}) or {})
    coupling_gradient_field = dict(lattice_calibration.get("coupling_gradient_field", {}) or {})
    field_pressure = float(lattice_calibration.get("field_pressure", 0.0))
    larger_field_exposure = float(lattice_calibration.get("larger_field_exposure", 0.0))
    base_dof_vector = np.array(
        list(lattice_calibration.get("gpu_pulse_dof_vector", []) or [0.0] * 10),
        dtype=np.float64,
    )
    if base_dof_vector.shape[0] != 10:
        base_dof_vector = np.zeros(10, dtype=np.float64)
    base_dof_tensor = np.array(
        lattice_calibration.get("gpu_pulse_dof_tensor", np.zeros((10, 10), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if base_dof_tensor.shape != (10, 10):
        base_dof_tensor = np.zeros((10, 10), dtype=np.float64)
    projection_tensor = np.array(
        lattice_calibration.get("gpu_pulse_projection_tensor", np.zeros((4, 10), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if projection_tensor.shape != (4, 10):
        projection_tensor = np.zeros((4, 10), dtype=np.float64)
    previous_gpu_feedback = dict(previous_field_state.get("gpu_pulse_feedback", {}) or {})
    previous_delta_feedback = dict(previous_field_state.get("gpu_pulse_delta_feedback", {}) or {})
    previous_ancilla_summary = dict(previous_field_state.get("kernel_ancilla_summary", {}) or {})
    ancilla_commit_ratio_previous = clamp01(float(previous_ancilla_summary.get("commit_ratio", 0.0)))
    ancilla_convergence_previous = clamp01(float(previous_ancilla_summary.get("convergence_mean", 0.0)))
    ancilla_flux_previous = clamp01(float(previous_ancilla_summary.get("flux_mean", 0.0)))
    ancilla_phase_alignment_previous = clamp01(
        float(previous_ancilla_summary.get("phase_alignment_mean", 0.0))
    )
    ancilla_current_norm_previous = clamp01(
        float(previous_ancilla_summary.get("current_norm_mean", 0.0))
    )
    ancilla_tension_headroom_previous = clamp01(
        float(previous_ancilla_summary.get("tension_headroom_mean", 1.0 - ancilla_current_norm_previous))
    )
    ancilla_gradient_headroom_previous = clamp01(
        float(previous_ancilla_summary.get("gradient_headroom_mean", 1.0 - ancilla_flux_previous))
    )
    ancilla_temporal_persistence_previous = clamp01(
        float(
            previous_ancilla_summary.get(
                "temporal_persistence_mean",
                0.5 * ancilla_commit_ratio_previous + 0.5 * ancilla_convergence_previous,
            )
        )
    )
    ancilla_activation_gate_previous = clamp01(
        float(previous_ancilla_summary.get("activation_gate_mean", ancilla_temporal_persistence_previous))
    )
    delta_target_vector = np.array(
        list(previous_delta_feedback.get("delta_target_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if delta_target_vector.shape[0] != 4:
        delta_target_vector = np.zeros(4, dtype=np.float64)
    delta_phase_shift_turns = float(previous_delta_feedback.get("phase_shift_turns", 0.0))
    delta_phase_retention = clamp01(float(previous_delta_feedback.get("phase_retention", 0.0)))
    delta_response_gate = clamp01(float(previous_delta_feedback.get("response_gate", 0.0)))
    delta_response_energy = clamp01(float(previous_delta_feedback.get("response_energy", 0.0)))
    delta_memory_retention = clamp01(float(previous_delta_feedback.get("memory_retention", 0.0)))
    delta_latency_norm = clamp01(float(previous_delta_feedback.get("latency_norm", 0.0)))
    delta_latency_gate = clamp01(float(previous_delta_feedback.get("latency_gate", 0.0)))
    delta_window_span_norm = clamp01(float(previous_delta_feedback.get("window_span_norm", 0.0)))
    delta_window_density = clamp01(float(previous_delta_feedback.get("window_density", 0.0)))
    delta_window_latency_alignment = clamp01(
        float(previous_delta_feedback.get("window_latency_alignment", 0.0))
    )
    delta_window_steps = max(1, int(previous_delta_feedback.get("window_calibration_steps", 1)))
    delta_environment_pressure = clamp01(
        float(previous_delta_feedback.get("environment_pressure_target", 0.0))
    )
    delta_thermal_headroom = clamp01(
        float(previous_delta_feedback.get("thermal_headroom_target", 0.0))
    )
    gpu_pulse_feedback = sample_gpu_pulse_feedback(
        pulse_index=pulse_index,
        previous_feedback=previous_gpu_feedback,
        feedback_context=build_vector_feedback_context(
            pulse_index=pulse_index,
            lattice_calibration=lattice_calibration,
            pulse_sweep=pulse_sweep,
            feedback_state=previous_field_state,
            target_profile=target_profile,
        ),
    )
    feedback_axis_vector = np.array(
        list(gpu_pulse_feedback.get("feedback_axis_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if feedback_axis_vector.shape[0] != 4:
        feedback_axis_vector = np.zeros(4, dtype=np.float64)
    feedback_axis_tensor = np.array(
        gpu_pulse_feedback.get("feedback_axis_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if feedback_axis_tensor.shape != (4, 4):
        feedback_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    feedback_dof_vector = np.array(
        list(gpu_pulse_feedback.get("feedback_dof_vector", []) or base_dof_vector.tolist()),
        dtype=np.float64,
    )
    if feedback_dof_vector.shape[0] != 10:
        feedback_dof_vector = np.array(base_dof_vector, dtype=np.float64)
    feedback_dof_tensor = np.array(
        gpu_pulse_feedback.get("feedback_dof_tensor", base_dof_tensor.tolist()),
        dtype=np.float64,
    )
    if feedback_dof_tensor.shape != (10, 10):
        feedback_dof_tensor = np.array(base_dof_tensor, dtype=np.float64)
    feedback_frequency_axis = clamp01(float(feedback_axis_vector[0]))
    feedback_amplitude_axis = clamp01(float(feedback_axis_vector[1]))
    feedback_current_axis = clamp01(float(feedback_axis_vector[2]))
    feedback_voltage_axis = clamp01(float(feedback_axis_vector[3]))
    feedback_phase_anchor_turns = wrap_turns(float(gpu_pulse_feedback.get("phase_anchor_turns", 0.0)))
    feedback_phase_alignment = clamp01(float(gpu_pulse_feedback.get("phase_alignment", 0.0)))
    feedback_memory_proxy = clamp01(float(gpu_pulse_feedback.get("memory_proxy", 0.0)))
    feedback_flux_proxy = clamp01(float(gpu_pulse_feedback.get("flux_proxy", 0.0)))
    feedback_stability_proxy = clamp01(float(gpu_pulse_feedback.get("stability_proxy", 0.0)))
    feedback_vf_flux_proxy = clamp01(float(gpu_pulse_feedback.get("vf_flux_proxy", 0.0)))
    feedback_temporal_drive = clamp01(float(gpu_pulse_feedback.get("temporal_drive", 0.0)))
    feedback_temperature_norm = clamp01(float(gpu_pulse_feedback.get("temperature_norm", 0.0)))
    feedback_thermal_headroom = clamp01(
        float(gpu_pulse_feedback.get("thermal_headroom_norm", 1.0 - feedback_temperature_norm))
    )
    feedback_temperature_velocity = clamp01(
        float(gpu_pulse_feedback.get("temperature_velocity_norm", 0.0))
    )
    feedback_environment_pressure = clamp01(float(gpu_pulse_feedback.get("environment_pressure", 0.0)))
    feedback_environment_stability = clamp01(float(gpu_pulse_feedback.get("environment_stability", 0.0)))
    feedback_latency_norm = clamp01(float(gpu_pulse_feedback.get("latency_norm", 0.0)))
    feedback_latency_jitter = clamp01(float(gpu_pulse_feedback.get("latency_jitter_norm", 0.0)))
    feedback_latency_gate = clamp01(
        float(gpu_pulse_feedback.get("temporal_latency_gate", 1.0 - feedback_latency_norm))
    )
    feedback_dln_amplitude = float(gpu_pulse_feedback.get("dln_amplitude", 0.0))
    feedback_dln_frequency = float(gpu_pulse_feedback.get("dln_frequency", 0.0))
    feedback_dln_voltage = float(gpu_pulse_feedback.get("dln_voltage", 0.0))
    feedback_ddln_frequency = float(gpu_pulse_feedback.get("ddln_frequency", 0.0))
    feedback_ddln_voltage = float(gpu_pulse_feedback.get("ddln_voltage", 0.0))
    interval_windows = list(target_profile.get("interval_windows", []) or [0.5])
    phase_windows = list(target_profile.get("phase_windows", []) or interval_windows)
    network_algorithm = dict(target_profile.get("network_algorithm", {}) or {})
    network_force_vector = np.array(
        list(network_algorithm.get("force_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if network_force_vector.shape[0] != 4:
        network_force_vector = np.zeros(4, dtype=np.float64)
    network_force_tensor = np.array(
        network_algorithm.get("force_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if network_force_tensor.shape != (4, 4):
        network_force_tensor = np.zeros((4, 4), dtype=np.float64)
    network_phase_turns = [
        wrap_turns(float(value))
        for value in list(network_algorithm.get("phase_turns", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])
    ]
    if len(network_phase_turns) < 4:
        network_phase_turns = [0.0, 0.0, 0.0, 0.0]
    network_phase_pressure = clamp01(float(network_algorithm.get("phase_pressure", 0.0)))
    network_amplitude_pressure = clamp01(float(network_algorithm.get("amplitude_pressure", 0.0)))
    network_algorithm_bias = clamp01(float(network_algorithm.get("algorithm_bias", 0.0)))
    state_axes = list(encoded_event_model.get("state_vector", []) or [])
    sequence_length = max(
        8,
        min(
            max(len(state_axes) * 2, 8),
            max(len(phase_windows), 8),
            max(24, 8 + delta_window_steps),
            32,
        ),
    )
    previous_sequence_index = int(previous_field_state.get("temporal_sequence_index", pulse_index % sequence_length))
    previous_sequence_vector = np.array(
        list(previous_field_state.get("persistent_temporal_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if previous_sequence_vector.shape[0] != 4:
        previous_sequence_vector = np.zeros(4, dtype=np.float64)
    previous_temporal_dof_vector = np.array(
        list(previous_field_state.get("persistent_temporal_dof_vector", []) or base_dof_vector.tolist()),
        dtype=np.float64,
    )
    if previous_temporal_dof_vector.shape[0] != 10:
        previous_temporal_dof_vector = np.array(base_dof_vector, dtype=np.float64)
    previous_overlap = clamp01(float(previous_field_state.get("temporal_index_overlap", 0.0)))
    previous_persistence = clamp01(float(previous_field_state.get("sequence_persistence_score", 0.0)))
    sweep_quality = clamp01(
        float(pulse_sweep.get("coherence", 0.0))
        - float(pulse_sweep.get("deviation", 0.0)) * 0.10
    )
    feedback_sequence_index = (
        int(round(feedback_phase_anchor_turns * max(float(sequence_length - 1), 0.0)))
        if sequence_length > 1
        else 0
    )
    retro_sequence_offset = int(round(delta_phase_shift_turns * float(sequence_length)))
    sequence_index = int(
        (
            pulse_index
            + feedback_sequence_index
            + retro_sequence_offset
            + int(round(abs(feedback_dln_frequency) * 7.0))
            + int(round(float(axis_step_interval.get("V", 0.0)) * 10.0))
            + int(round(float(coupling_gradient_field.get("F_V", 0.0)) * 6.0))
        )
        % sequence_length
    )
    interval_index = int(round(sequence_index * len(interval_windows) / max(sequence_length, 1))) % len(interval_windows)
    phase_index = int(round(sequence_index * len(phase_windows) / max(sequence_length, 1))) % len(phase_windows)
    sequence_interval_window = clamp01(
        0.46 * float(interval_windows[interval_index])
        + 0.14 * feedback_frequency_axis
        + 0.10 * feedback_voltage_axis
        + 0.10 * feedback_memory_proxy
        + 0.10 * feedback_phase_alignment
        + 0.10 * sweep_quality
        + 0.08 * float(delta_target_vector[0])
        + 0.06 * delta_response_gate
        + 0.06 * delta_phase_retention
        + 0.06 * feedback_latency_gate
        + 0.04 * (1.0 - feedback_environment_pressure)
        + 0.04 * delta_latency_gate
        + 0.04 * delta_window_latency_alignment
        + 0.03 * (1.0 - delta_window_span_norm)
        + 0.08 * float(network_force_vector[0])
        + 0.06 * network_phase_pressure
        + 0.04 * turn_alignment(feedback_phase_anchor_turns, network_phase_turns[sequence_index % 4])
        + 0.04 * ancilla_commit_ratio_previous
        + 0.03 * ancilla_gradient_headroom_previous
        + 0.03 * ancilla_activation_gate_previous
    )
    sequence_phase_window = clamp01(
        0.42 * float(phase_windows[phase_index])
        + 0.14 * feedback_phase_anchor_turns
        + 0.12 * feedback_amplitude_axis
        + 0.10 * feedback_stability_proxy
        + 0.10 * feedback_memory_proxy
        + 0.12 * feedback_phase_alignment
        + 0.08 * float(delta_target_vector[1])
        + 0.06 * delta_phase_retention
        + 0.06 * clamp01(abs(delta_phase_shift_turns) * 4.0)
        + 0.04 * delta_memory_retention
        + 0.04 * feedback_thermal_headroom
        + 0.04 * feedback_environment_stability
        + 0.04 * delta_thermal_headroom
        + 0.04 * delta_window_latency_alignment
        + 0.03 * (1.0 - delta_window_span_norm)
        + 0.08 * float(network_force_vector[1])
        + 0.06 * network_amplitude_pressure
        + 0.04 * turn_alignment(feedback_phase_anchor_turns, network_phase_turns[(sequence_index + 1) % 4])
        + 0.04 * ancilla_phase_alignment_previous
        + 0.03 * ancilla_temporal_persistence_previous
        + 0.03 * ancilla_tension_headroom_previous
    )
    voltage_frequency_flux = clamp01(
        0.26 * feedback_vf_flux_proxy
        + 0.18 * feedback_voltage_axis
        + 0.14 * feedback_frequency_axis
        + 0.10 * float(coupling_gradient_field.get("F_V", 0.0))
        + 0.08 * sequence_phase_window
        + 0.08 * sequence_interval_window
        + 0.06 * field_pressure
        + 0.05 * sweep_quality
        + 0.05 * clamp01(abs(feedback_dln_voltage) * 2.0)
        + 0.08 * float(delta_target_vector[2])
        + 0.05 * delta_response_gate
        + 0.05 * feedback_environment_pressure
        + 0.04 * feedback_temperature_velocity
        + 0.04 * delta_window_density
        + 0.06 * float(network_force_vector[2])
        + 0.05 * network_phase_pressure
        + 0.04 * network_algorithm_bias
        + 0.04 * ancilla_flux_previous
        + 0.03 * ancilla_current_norm_previous
        + 0.03 * ancilla_activation_gate_previous
    )
    frequency_voltage_flux = clamp01(
        0.22 * feedback_vf_flux_proxy
        + 0.18 * feedback_frequency_axis
        + 0.16 * feedback_voltage_axis
        + 0.12 * feedback_phase_alignment
        + 0.10 * float(coupling_gradient_field.get("F_V", 0.0))
        + 0.08 * float(wave_step_field.get("cascade_interval", 0.0))
        + 0.08 * field_pressure
        + 0.06 * sequence_phase_window
        + 0.05 * feedback_temporal_drive
        + 0.05 * clamp01(abs(feedback_ddln_frequency) + abs(feedback_ddln_voltage))
        + 0.08 * float(delta_target_vector[3])
        + 0.05 * delta_phase_retention
        + 0.05 * feedback_latency_gate
        + 0.04 * delta_latency_gate
        + 0.04 * delta_window_latency_alignment
        + 0.06 * float(network_force_vector[3])
        + 0.05 * network_amplitude_pressure
        + 0.04 * network_algorithm_bias
        + 0.04 * ancilla_phase_alignment_previous
        + 0.03 * ancilla_temporal_persistence_previous
        + 0.03 * ancilla_gradient_headroom_previous
    )
    current_sequence_vector = np.array(
        [
            clamp01(
                0.54 * feedback_frequency_axis
                + 0.12 * sequence_interval_window
                + 0.10 * feedback_vf_flux_proxy
                + 0.08 * sweep_quality
                + 0.08 * field_pressure
                + 0.08 * feedback_phase_alignment
                + 0.06 * float(delta_target_vector[0])
                + 0.05 * delta_phase_retention
                + 0.05 * feedback_latency_gate
                + 0.04 * feedback_environment_stability
                + 0.03 * delta_window_latency_alignment
                + 0.06 * float(network_force_vector[0])
                + 0.04 * network_phase_pressure
                + 0.04 * ancilla_commit_ratio_previous
                + 0.03 * ancilla_gradient_headroom_previous
            ),
            clamp01(
                0.52 * feedback_amplitude_axis
                + 0.12 * sequence_phase_window
                + 0.10 * feedback_memory_proxy
                + 0.08 * sweep_quality
                + 0.08 * larger_field_exposure
                + 0.10 * feedback_temporal_drive
                + 0.06 * delta_memory_retention
                + 0.04 * float(delta_target_vector[1])
                + 0.04 * feedback_thermal_headroom
                + 0.04 * delta_thermal_headroom
                + 0.03 * (1.0 - delta_window_span_norm)
                + 0.06 * float(network_force_vector[1])
                + 0.04 * network_amplitude_pressure
                + 0.04 * ancilla_convergence_previous
                + 0.03 * ancilla_phase_alignment_previous
            ),
            clamp01(
                0.48 * feedback_current_axis
                + 0.16 * feedback_flux_proxy
                + 0.10 * sequence_interval_window
                + 0.10 * feedback_memory_proxy
                + 0.08 * feedback_phase_alignment
                + 0.08 * feedback_temporal_drive
                + 0.06 * delta_response_gate
                + 0.04 * float(delta_target_vector[2])
                + 0.04 * feedback_environment_pressure
                + 0.04 * feedback_latency_gate
                + 0.03 * delta_window_density
                + 0.06 * float(network_force_vector[2])
                + 0.04 * network_phase_pressure
                + 0.04 * ancilla_current_norm_previous
                + 0.04 * ancilla_flux_previous
            ),
            clamp01(
                0.50 * feedback_voltage_axis
                + 0.16 * feedback_vf_flux_proxy
                + 0.10 * sequence_phase_window
                + 0.08 * field_pressure
                + 0.08 * feedback_phase_alignment
                + 0.08 * feedback_flux_proxy
                + 0.06 * float(delta_target_vector[3])
                + 0.04 * delta_response_energy
                + 0.04 * feedback_latency_gate
                + 0.04 * delta_latency_gate
                + 0.03 * delta_window_latency_alignment
                + 0.06 * float(network_force_vector[3])
                + 0.04 * network_amplitude_pressure
                + 0.04 * ancilla_flux_previous
                + 0.03 * ancilla_gradient_headroom_previous
            ),
        ],
        dtype=np.float64,
    )
    dof_sequence_seed = clamp01(
        0.26 * sweep_quality
        + 0.18 * previous_persistence
        + 0.14 * previous_overlap
        + 0.14 * field_pressure
        + 0.14 * feedback_memory_proxy
        + 0.08 * feedback_phase_alignment
        + 0.06 * feedback_stability_proxy
        + 0.06 * feedback_environment_stability
        + 0.04 * feedback_latency_gate
        + 0.04 * delta_window_latency_alignment
    )
    static_temporal_dof_basis = build_gpu_pulse_dof_basis(
        axis_wave_norm=axis_wave_norm,
        axis_step_interval=axis_step_interval,
        coupling_gradient_field=coupling_gradient_field,
        field_environment=field_environment,
        wave_step_field=wave_step_field,
        field_pressure=field_pressure,
        larger_field_exposure=larger_field_exposure,
        sequence_persistence_score=dof_sequence_seed,
        temporal_index_overlap=previous_overlap,
        voltage_frequency_flux=voltage_frequency_flux,
        frequency_voltage_flux=frequency_voltage_flux,
    )
    current_temporal_dof_vector = np.array(
        list(static_temporal_dof_basis.get("gpu_pulse_dof_vector", []) or base_dof_vector.tolist()),
        dtype=np.float64,
    )
    if current_temporal_dof_vector.shape[0] != 10:
        current_temporal_dof_vector = np.array(base_dof_vector, dtype=np.float64)
    current_temporal_dof_vector = clamp_vector_norm(
        current_temporal_dof_vector * (0.54 + 0.14 * sweep_quality)
        + feedback_dof_vector * (0.46 + 0.18 * feedback_memory_proxy),
        max_norm=5.40,
    )
    current_temporal_dof_tensor = np.array(
        static_temporal_dof_basis.get("gpu_pulse_dof_tensor", base_dof_tensor.tolist()),
        dtype=np.float64,
    )
    if current_temporal_dof_tensor.shape != (10, 10):
        current_temporal_dof_tensor = np.array(base_dof_tensor, dtype=np.float64)
    current_temporal_dof_tensor = np.array(base_dof_tensor, dtype=np.float64) * (
        0.42 + 0.10 * field_pressure
    ) + current_temporal_dof_tensor * (0.32 + 0.12 * sweep_quality) + feedback_dof_tensor * (
        0.26 + 0.14 * feedback_memory_proxy
    )
    current_temporal_dof_tensor += np.outer(feedback_dof_vector, current_temporal_dof_vector) * (
        0.01 + 0.03 * feedback_temporal_drive
    )
    current_temporal_dof_tensor = np.clip(current_temporal_dof_tensor, 0.0, 1.55)
    projected_temporal_dof_vector = (
        projection_tensor @ current_temporal_dof_vector
        if projection_tensor.shape == (4, 10)
        else np.zeros(4, dtype=np.float64)
    )
    current_sequence_vector = clamp_vector_norm(
        current_sequence_vector
        + projected_temporal_dof_vector * (0.10 + 0.10 * sweep_quality)
        + feedback_axis_vector * (0.08 + 0.10 * feedback_temporal_drive)
        + (feedback_axis_tensor @ current_sequence_vector) * (0.03 + 0.05 * feedback_flux_proxy),
        max_norm=2.55,
    )
    current_sequence_vector = clamp_vector_norm(
        current_sequence_vector
        + network_force_vector * (0.10 + 0.10 * network_algorithm_bias)
        + (network_force_tensor @ current_sequence_vector) * (0.02 + 0.04 * network_phase_pressure),
        max_norm=2.72,
    )
    sequence_alignment = vector_similarity(current_sequence_vector, previous_sequence_vector)
    feedback_sequence_alignment = vector_similarity(current_sequence_vector, feedback_axis_vector)
    cyclic_delta = abs(sequence_index - previous_sequence_index)
    cyclic_delta = min(cyclic_delta, max(sequence_length - cyclic_delta, 0))
    sequence_index_alignment = clamp01(
        1.0 - float(cyclic_delta) / max(float(sequence_length) * 0.5, 1.0)
    )
    carryover = clamp01(
        0.14
        + 0.16 * sequence_alignment
        + 0.14 * feedback_sequence_alignment
        + 0.12 * previous_overlap
        + 0.10 * previous_persistence
        + 0.10 * feedback_phase_alignment
        + 0.10 * feedback_memory_proxy
        + 0.08 * feedback_temporal_drive
        + 0.06 * voltage_frequency_flux
        + 0.10 * delta_phase_retention
        + 0.08 * delta_memory_retention
        + 0.06 * delta_response_gate
        + 0.06 * feedback_latency_gate
        + 0.04 * delta_latency_gate
        + 0.04 * feedback_environment_stability
        + 0.04 * delta_window_latency_alignment
        + 0.04 * ancilla_temporal_persistence_previous
        + 0.03 * ancilla_commit_ratio_previous
        + 0.03 * ancilla_activation_gate_previous
    )
    persistent_temporal_dof_vector = clamp_vector_norm(
        previous_temporal_dof_vector * carryover
        + current_temporal_dof_vector * (0.88 - 0.28 * carryover),
        max_norm=5.40,
    )
    projected_persistent_temporal_dof_vector = (
        projection_tensor @ persistent_temporal_dof_vector
        if projection_tensor.shape == (4, 10)
        else np.zeros(4, dtype=np.float64)
    )
    persistent_temporal_vector = clamp_vector_norm(
        previous_sequence_vector * carryover
        + current_sequence_vector * (0.88 - 0.28 * carryover),
        max_norm=2.35,
    )
    persistent_temporal_vector = clamp_vector_norm(
        persistent_temporal_vector
        + projected_persistent_temporal_dof_vector * (0.08 + 0.06 * sequence_alignment)
        + feedback_axis_vector * (0.06 + 0.08 * feedback_temporal_drive),
        max_norm=2.55,
    )
    temporal_index_overlap = clamp01(
        0.24 * sequence_alignment
        + 0.16 * sequence_index_alignment
        + 0.16 * feedback_phase_alignment
        + 0.14 * feedback_sequence_alignment
        + 0.10 * feedback_memory_proxy
        + 0.08 * feedback_temporal_drive
        + 0.06 * feedback_flux_proxy
        + 0.06 * voltage_frequency_flux
        + 0.08 * float(delta_target_vector[1])
        + 0.06 * delta_phase_retention
        + 0.04 * feedback_latency_gate
        + 0.04 * delta_latency_gate
        + 0.04 * (1.0 - delta_window_span_norm)
        + 0.06 * network_phase_pressure
        + 0.04 * float(network_force_vector[1])
        + 0.04 * ancilla_phase_alignment_previous
        + 0.03 * ancilla_temporal_persistence_previous
    )
    sequence_persistence_score = clamp01(
        0.22 * feedback_memory_proxy
        + 0.16 * feedback_stability_proxy
        + 0.12 * feedback_temporal_drive
        + 0.12 * sequence_alignment
        + 0.10 * feedback_sequence_alignment
        + 0.10 * voltage_frequency_flux
        + 0.08 * frequency_voltage_flux
        + 0.05 * field_pressure
        + 0.05 * larger_field_exposure
        + 0.08 * delta_memory_retention
        + 0.06 * delta_response_gate
        + 0.06 * float(delta_target_vector[0])
        + 0.04 * feedback_environment_stability
        + 0.04 * feedback_latency_gate
        + 0.04 * delta_thermal_headroom
        + 0.04 * delta_window_latency_alignment
        + 0.06 * network_amplitude_pressure
        + 0.04 * float(network_force_vector[0])
        + 0.04 * ancilla_convergence_previous
        + 0.03 * ancilla_temporal_persistence_previous
        + 0.03 * ancilla_tension_headroom_previous
    )
    sequence_stride = max(
        1,
        int(
            round(
                1.0
                + voltage_frequency_flux * 1.5
                + frequency_voltage_flux * 1.0
                + temporal_index_overlap
                + feedback_temporal_drive
                + feedback_latency_gate * 0.5
                + delta_response_gate * 0.8
                + 0.10 * float(delta_window_steps)
            )
        ),
    )
    persistence_span = max(
        1,
        min(
            4,
            1
            + int(
                round(
                    sequence_persistence_score * 2.5
                    + temporal_index_overlap
                    + feedback_memory_proxy
                    + feedback_latency_gate * 0.5
                    + delta_memory_retention
                    + 0.5 * delta_response_gate
                    + 0.10 * float(delta_window_steps)
                )
            ),
        ),
    )
    temporal_sequence_indexes: list[int] = []
    seen_indexes: set[int] = set()
    for offset in range(-persistence_span, persistence_span + 1):
        candidate_index = int((sequence_index + offset * sequence_stride) % sequence_length)
        if candidate_index in seen_indexes:
            continue
        seen_indexes.add(candidate_index)
        temporal_sequence_indexes.append(candidate_index)
    voltage_frequency_correlation_tensor = np.array(
        [
            [
                0.18 + 0.42 * voltage_frequency_flux,
                0.04 + 0.08 * float(coupling_gradient_field.get("F_A", 0.0)),
                0.04 + 0.08 * float(coupling_gradient_field.get("F_I", 0.0)),
                0.16 + 0.52 * voltage_frequency_flux + 0.10 * temporal_index_overlap,
            ],
            [
                0.04 + 0.08 * float(coupling_gradient_field.get("F_A", 0.0)),
                0.14 + 0.24 * float(axis_wave_norm.get("A", 0.0)),
                0.05 + 0.08 * float(coupling_gradient_field.get("A_I", 0.0)),
                0.06 + 0.08 * float(coupling_gradient_field.get("A_V", 0.0)),
            ],
            [
                0.04 + 0.08 * float(coupling_gradient_field.get("F_I", 0.0)),
                0.05 + 0.08 * float(coupling_gradient_field.get("A_I", 0.0)),
                0.14 + 0.24 * float(axis_wave_norm.get("I", 0.0)),
                0.06 + 0.08 * float(coupling_gradient_field.get("I_V", 0.0)),
            ],
            [
                0.16 + 0.48 * frequency_voltage_flux + 0.10 * temporal_index_overlap,
                0.06 + 0.08 * float(coupling_gradient_field.get("A_V", 0.0)),
                0.06 + 0.08 * float(coupling_gradient_field.get("I_V", 0.0)),
                0.18 + 0.40 * float(axis_wave_norm.get("V", 0.0)) + 0.10 * sequence_persistence_score,
            ],
        ],
        dtype=np.float64,
    )
    voltage_frequency_correlation_tensor += np.outer(
        persistent_temporal_vector,
        current_sequence_vector,
    ) * (0.02 + 0.04 * sequence_persistence_score)
    voltage_frequency_correlation_tensor += feedback_axis_tensor * (
        0.04 + 0.10 * feedback_memory_proxy + 0.06 * feedback_phase_alignment
    )
    voltage_frequency_correlation_tensor += np.outer(
        feedback_axis_vector,
        persistent_temporal_vector,
    ) * (0.03 + 0.04 * feedback_temporal_drive)
    voltage_frequency_correlation_tensor += np.outer(
        delta_target_vector,
        persistent_temporal_vector,
    ) * (0.02 + 0.04 * delta_response_gate)
    voltage_frequency_correlation_tensor += np.outer(
        persistent_temporal_vector,
        delta_target_vector,
    ) * (0.02 + 0.04 * delta_memory_retention)
    voltage_frequency_correlation_tensor += network_force_tensor * (
        0.04 + 0.06 * network_algorithm_bias + 0.04 * network_phase_pressure
    )
    voltage_frequency_correlation_tensor += np.outer(
        network_force_vector,
        persistent_temporal_vector,
    ) * (0.03 + 0.04 * network_amplitude_pressure)
    persistent_temporal_dof_tensor = np.array(base_dof_tensor, dtype=np.float64) * (
        0.70 + 0.16 * field_pressure
    )
    persistent_temporal_dof_tensor += np.array(current_temporal_dof_tensor, dtype=np.float64) * (
        0.18 + 0.12 * sweep_quality
    )
    persistent_temporal_dof_tensor += np.array(feedback_dof_tensor, dtype=np.float64) * (
        0.14 + 0.10 * feedback_memory_proxy
    )
    persistent_temporal_dof_tensor += np.outer(
        persistent_temporal_dof_vector,
        current_temporal_dof_vector,
    ) * (0.01 + 0.03 * sequence_persistence_score)
    persistent_temporal_dof_tensor = np.clip(persistent_temporal_dof_tensor, 0.0, 1.55)
    projected_temporal_dof_axis_tensor = (
        projection_tensor @ persistent_temporal_dof_tensor @ projection_tensor.T
        if projection_tensor.shape == (4, 10)
        else np.zeros((4, 4), dtype=np.float64)
    )
    voltage_frequency_correlation_tensor += projected_temporal_dof_axis_tensor * (
        0.03 + 0.05 * sequence_persistence_score + 0.03 * temporal_index_overlap
    )
    voltage_frequency_correlation_tensor = np.clip(voltage_frequency_correlation_tensor, 0.0, 1.35)
    return {
        "temporal_sequence_index": int(sequence_index),
        "temporal_sequence_length": int(sequence_length),
        "temporal_sequence_indexes": temporal_sequence_indexes,
        "sequence_stride": int(sequence_stride),
        "temporal_persistence_span": int(persistence_span),
        "sequence_persistence_score": float(sequence_persistence_score),
        "temporal_index_overlap": float(temporal_index_overlap),
        "voltage_frequency_flux": float(voltage_frequency_flux),
        "frequency_voltage_flux": float(frequency_voltage_flux),
        "persistent_temporal_vector": [float(value) for value in persistent_temporal_vector],
        "persistent_temporal_dof_vector": [float(value) for value in persistent_temporal_dof_vector],
        "projected_temporal_dof_vector": [float(value) for value in projected_persistent_temporal_dof_vector],
        "temporal_sequence_signature": [float(value) for value in current_sequence_vector],
        "voltage_frequency_correlation_tensor": voltage_frequency_correlation_tensor.tolist(),
        "voltage_frequency_dof_axis_tensor": projected_temporal_dof_axis_tensor.tolist(),
        "voltage_frequency_dof_tensor": persistent_temporal_dof_tensor.tolist(),
        "gpu_pulse_dof_labels": list(lattice_calibration.get("gpu_pulse_dof_labels", []) or list(GPU_PULSE_DOF_LABELS)),
        "gpu_pulse_feedback": dict(gpu_pulse_feedback),
        "feedback_axis_vector": [float(value) for value in feedback_axis_vector],
        "feedback_axis_tensor": feedback_axis_tensor.tolist(),
        "feedback_temporal_drive": float(feedback_temporal_drive),
        "feedback_temperature_norm": float(feedback_temperature_norm),
        "feedback_thermal_headroom": float(feedback_thermal_headroom),
        "feedback_temperature_velocity": float(feedback_temperature_velocity),
        "feedback_environment_pressure": float(feedback_environment_pressure),
        "feedback_environment_stability": float(feedback_environment_stability),
        "feedback_latency_norm": float(feedback_latency_norm),
        "feedback_latency_jitter": float(feedback_latency_jitter),
        "feedback_latency_gate": float(feedback_latency_gate),
        "feedback_delta_target_vector": [float(value) for value in delta_target_vector],
        "feedback_delta_phase_shift_turns": float(delta_phase_shift_turns),
        "feedback_delta_phase_retention": float(delta_phase_retention),
        "feedback_delta_response_gate": float(delta_response_gate),
        "feedback_delta_response_energy": float(delta_response_energy),
        "feedback_delta_memory_retention": float(delta_memory_retention),
        "feedback_delta_latency_norm": float(delta_latency_norm),
        "feedback_delta_latency_gate": float(delta_latency_gate),
        "feedback_delta_window_span_norm": float(delta_window_span_norm),
        "feedback_delta_window_density": float(delta_window_density),
        "btc_network_force_vector": [float(value) for value in network_force_vector],
        "btc_network_force_tensor": network_force_tensor.tolist(),
        "btc_network_phase_turns": [float(value) for value in network_phase_turns],
        "btc_network_phase_pressure": float(network_phase_pressure),
        "btc_network_amplitude_pressure": float(network_amplitude_pressure),
        "btc_network_algorithm_bias": float(network_algorithm_bias),
        "feedback_delta_window_latency_alignment": float(delta_window_latency_alignment),
        "feedback_delta_window_steps": int(delta_window_steps),
        "feedback_delta_environment_pressure": float(delta_environment_pressure),
        "feedback_delta_thermal_headroom": float(delta_thermal_headroom),
        "ancilla_commit_ratio_previous": float(ancilla_commit_ratio_previous),
        "ancilla_convergence_previous": float(ancilla_convergence_previous),
        "ancilla_flux_previous": float(ancilla_flux_previous),
        "ancilla_phase_alignment_previous": float(ancilla_phase_alignment_previous),
        "ancilla_current_norm_previous": float(ancilla_current_norm_previous),
        "ancilla_tension_headroom_previous": float(ancilla_tension_headroom_previous),
        "ancilla_gradient_headroom_previous": float(ancilla_gradient_headroom_previous),
        "ancilla_temporal_persistence_previous": float(ancilla_temporal_persistence_previous),
        "ancilla_activation_gate_previous": float(ancilla_activation_gate_previous),
    }


def build_cluster_probe_plan(
    cluster_map: dict[str, list[dict[str, Any]]],
    seed_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cluster_profiles: list[dict[str, Any]] = []
    for cluster_id, items in cluster_map.items():
        ranked_items = sorted(
            list(items),
            key=lambda item: (
                float(item.get("decode_phase_integrity_score", 0.0)),
                float(item.get("decode_phase_delta_alignment", 0.0)),
                float(item.get("decode_target_prefix_lock", 0.0)),
                1.0 - float(item.get("decode_prefix_asymptote_pressure", 1.0)),
                float(item.get("coherence_peak", 0.0)),
                float(item.get("interference_resonance", 0.0)),
                float(item.get("motif_alignment", 0.0)),
                float(item.get("row_activation", 0.0)),
                float(item.get("target_alignment", 0.0)),
            ),
            reverse=True,
        )
        count = len(ranked_items)
        mean_coherence = float(np.mean([float(item.get("coherence_peak", 0.0)) for item in ranked_items]))
        mean_resonance = float(np.mean([float(item.get("interference_resonance", 0.0)) for item in ranked_items]))
        mean_motif = float(np.mean([float(item.get("motif_alignment", 0.0)) for item in ranked_items]))
        mean_row_activation = float(np.mean([float(item.get("row_activation", 0.0)) for item in ranked_items]))
        mean_target = float(np.mean([float(item.get("target_alignment", 0.0)) for item in ranked_items]))
        mean_phase_pressure = float(
            np.mean([float(item.get("phase_length_pressure", 0.0)) for item in ranked_items])
        )
        mean_sequence_persistence = float(
            np.mean([float(item.get("sequence_persistence_score", 0.0)) for item in ranked_items])
        )
        mean_temporal_overlap = float(
            np.mean([float(item.get("temporal_index_overlap", 0.0)) for item in ranked_items])
        )
        mean_voltage_frequency_flux = float(
            np.mean([float(item.get("voltage_frequency_flux", 0.0)) for item in ranked_items])
        )
        mean_decode_phase_integrity = float(
            np.mean([float(item.get("decode_phase_integrity_score", 0.0)) for item in ranked_items])
        )
        mean_decode_phase_alignment = float(
            np.mean([float(item.get("decode_phase_delta_alignment", 0.0)) for item in ranked_items])
        )
        mean_decode_target_prefix_lock = float(
            np.mean([float(item.get("decode_target_prefix_lock", 0.0)) for item in ranked_items])
        )
        mean_decode_target_prefix_vector_alignment = float(
            np.mean(
                [
                    float(item.get("decode_target_prefix_vector_alignment", 0.0))
                    for item in ranked_items
                ]
            )
        )
        mean_decode_phase_orbital_alignment = float(
            np.mean([float(item.get("decode_phase_orbital_alignment", 0.0)) for item in ranked_items])
        )
        mean_decode_phase_orbital_resonance = float(
            np.mean([float(item.get("decode_phase_orbital_resonance", 0.0)) for item in ranked_items])
        )
        mean_decode_phase_orbital_stability = float(
            np.mean([float(item.get("decode_phase_orbital_stability", 0.0)) for item in ranked_items])
        )
        mean_decode_prefix_asymptote = float(
            np.mean([float(item.get("decode_prefix_asymptote_pressure", 1.0)) for item in ranked_items])
        )
        profile_score = clamp01(
            0.15 * mean_decode_phase_integrity
            + 0.11 * mean_decode_phase_alignment
            + 0.12 * mean_decode_target_prefix_lock
            + 0.08 * mean_decode_target_prefix_vector_alignment
            + 0.07 * mean_decode_phase_orbital_alignment
            + 0.06 * mean_decode_phase_orbital_resonance
            + 0.05 * mean_decode_phase_orbital_stability
            + 0.09 * (1.0 - mean_decode_prefix_asymptote)
            + 0.15 * mean_coherence
            + 0.14 * mean_resonance
            + 0.11 * mean_motif
            + 0.09 * mean_row_activation
            + 0.08 * mean_target
            + 0.08 * mean_phase_pressure
            + 0.06 * mean_sequence_persistence
            + 0.05 * mean_temporal_overlap
            + 0.04 * mean_voltage_frequency_flux
            + 0.10 * clamp01(float(count) / 8.0)
        )
        cluster_profiles.append(
            {
                "cluster_id": str(cluster_id),
                "items": ranked_items,
                "count": int(count),
                "score": float(profile_score),
                "mean_coherence": float(mean_coherence),
                "mean_resonance": float(mean_resonance),
                "mean_motif_alignment": float(mean_motif),
                "mean_row_activation": float(mean_row_activation),
                "mean_target_alignment": float(mean_target),
            "mean_decode_phase_integrity": float(mean_decode_phase_integrity),
            "mean_decode_phase_alignment": float(mean_decode_phase_alignment),
            "mean_decode_target_prefix_lock": float(mean_decode_target_prefix_lock),
            "mean_decode_target_prefix_vector_alignment": float(
                mean_decode_target_prefix_vector_alignment
            ),
            "mean_decode_phase_orbital_alignment": float(
                mean_decode_phase_orbital_alignment
            ),
            "mean_decode_phase_orbital_resonance": float(
                mean_decode_phase_orbital_resonance
            ),
            "mean_decode_phase_orbital_stability": float(
                mean_decode_phase_orbital_stability
            ),
            "mean_decode_prefix_asymptote": float(mean_decode_prefix_asymptote),
                "mean_phase_pressure": float(mean_phase_pressure),
                "mean_sequence_persistence": float(mean_sequence_persistence),
                "mean_temporal_overlap": float(mean_temporal_overlap),
                "mean_voltage_frequency_flux": float(mean_voltage_frequency_flux),
            }
        )
    cluster_profiles.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("mean_coherence", 0.0)),
            float(item.get("mean_resonance", 0.0)),
            int(item.get("count", 0)),
        ),
        reverse=True,
    )
    if not cluster_profiles or seed_budget <= 0:
        return [], []
    active_profiles = cluster_profiles[: min(8, len(cluster_profiles))]
    base_quota_total = sum(2 if rank < 2 else 1 for rank in range(len(active_profiles)))
    remaining_budget = max(int(seed_budget) - base_quota_total, 0)
    total_score = sum(float(profile.get("score", 0.0)) for profile in active_profiles)
    for rank, profile in enumerate(active_profiles):
        quota = 2 if rank < 2 else 1
        if total_score > 1.0e-9 and remaining_budget > 0:
            quota += int(round(remaining_budget * float(profile.get("score", 0.0)) / total_score))
        quota = min(int(profile.get("count", 0)), max(quota, 1))
        profile["probe_rank"] = int(rank)
        profile["probe_quota"] = int(quota)
    assigned = sum(int(profile.get("probe_quota", 0)) for profile in active_profiles)
    while assigned > seed_budget:
        adjusted = False
        for profile in reversed(active_profiles):
            min_quota = 2 if int(profile.get("probe_rank", 0)) < 2 else 1
            if int(profile.get("probe_quota", 0)) > min_quota:
                profile["probe_quota"] = int(profile.get("probe_quota", 0)) - 1
                assigned -= 1
                adjusted = True
                if assigned <= seed_budget:
                    break
        if not adjusted:
            break
    while assigned < seed_budget:
        adjusted = False
        for profile in active_profiles:
            if int(profile.get("probe_quota", 0)) < int(profile.get("count", 0)):
                profile["probe_quota"] = int(profile.get("probe_quota", 0)) + 1
                assigned += 1
                adjusted = True
                if assigned >= seed_budget:
                    break
        if not adjusted:
            break
    seed_candidates: list[dict[str, Any]] = []
    for profile in active_profiles:
        probe_rank = int(profile.get("probe_rank", 0))
        probe_quota = int(profile.get("probe_quota", 0))
        probe_weight = float(profile.get("score", 0.0))
        for candidate in list(profile.get("items", []) or [])[:probe_quota]:
            seeded = dict(candidate)
            ensure_temporal_decode_metrics(seeded)
            seeded["cluster_probe_rank"] = int(probe_rank)
            seeded["cluster_probe_weight"] = float(probe_weight)
            seeded["cluster_probe_id"] = str(profile.get("cluster_id", ""))
            seed_candidates.append(seeded)
    return seed_candidates, [
        {
            "cluster_id": str(profile.get("cluster_id", "")),
            "score": float(profile.get("score", 0.0)),
            "count": int(profile.get("count", 0)),
            "probe_rank": int(profile.get("probe_rank", 0)),
            "probe_quota": int(profile.get("probe_quota", 0)),
            "mean_coherence": float(profile.get("mean_coherence", 0.0)),
            "mean_resonance": float(profile.get("mean_resonance", 0.0)),
            "mean_motif_alignment": float(profile.get("mean_motif_alignment", 0.0)),
            "mean_row_activation": float(profile.get("mean_row_activation", 0.0)),
            "mean_target_alignment": float(profile.get("mean_target_alignment", 0.0)),
            "mean_phase_pressure": float(profile.get("mean_phase_pressure", 0.0)),
            "mean_sequence_persistence": float(profile.get("mean_sequence_persistence", 0.0)),
            "mean_temporal_overlap": float(profile.get("mean_temporal_overlap", 0.0)),
            "mean_voltage_frequency_flux": float(profile.get("mean_voltage_frequency_flux", 0.0)),
        }
        for profile in active_profiles
    ]


def build_btc_target_profile(nbits_hex: str) -> dict[str, Any]:
    target_hex = bits_to_target_hex(nbits_hex)
    try:
        nbits = int(str(nbits_hex), 16)
    except Exception:
        nbits = 0x1D00FFFF
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0xFFFFFF
    exponent_window = clamp01((float(exponent) - float(0x1B)) / max(float(0x1D - 0x1B), 1.0))
    mantissa_window = clamp01(
        (float(mantissa) - float(0x0404CB)) / max(float(0x00FFFF - 0x0404CB), 1.0)
    )
    difficulty_window = clamp01(0.25 + 0.45 * exponent_window + 0.30 * mantissa_window)
    target_nibbles = [int(ch, 16) for ch in str(target_hex).lower() if ch in "0123456789abcdef"]
    if not target_nibbles:
        target_nibbles = [0]
    interval_windows = [clamp01(0.16 + 0.84 * (float(nibble) / 15.0)) for nibble in target_nibbles]
    phase_windows = [
        clamp01(0.28 * difficulty_window + 0.72 * (float(nibble) / 15.0))
        for nibble in target_nibbles
    ]
    return {
        "target_hex": target_hex,
        "difficulty_window": difficulty_window,
        "interval_windows": interval_windows,
        "phase_windows": phase_windows,
    }


def build_btc_network_algorithm_profile(
    header_hex: str,
    nbits_hex: str,
    target_hex: str,
) -> dict[str, Any]:
    normalized_target = normalize_hex_64(target_hex)
    normalized_header = str(header_hex).strip()
    if normalized_header.startswith("0x"):
        normalized_header = normalized_header[2:]
    if len(normalized_header) % 2:
        normalized_header = "0" + normalized_header
    try:
        header_bytes = bytes.fromhex(normalized_header)
    except Exception:
        header_bytes = b""
    header_bytes = header_bytes[:80].ljust(80, b"\x00")
    try:
        header_words = struct.unpack("<20I", header_bytes[:80])
    except Exception:
        header_words = tuple([0] * 20)
    target_bytes = bytes.fromhex(normalized_target)
    first_hash = hashlib.sha256(header_bytes).digest()
    second_hash = hashlib.sha256(first_hash).digest()
    try:
        nbits = int(str(nbits_hex), 16)
    except Exception:
        nbits = 0x1D00FFFF
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0xFFFFFF
    exponent_window = clamp01((float(exponent) - float(0x1B)) / max(float(0x1D - 0x1B), 1.0))
    mantissa_window = clamp01(
        (float(mantissa) - float(0x0404CB)) / max(float(0x00FFFF - 0x0404CB), 1.0)
    )
    word_windows: list[float] = []
    first_hash_windows: list[float] = []
    second_hash_windows: list[float] = []
    target_windows: list[float] = []
    phase_turns: list[float] = []
    for idx in range(4):
        word_slice = header_words[idx * 5 : (idx + 1) * 5]
        hash_start = idx * 8
        hash_stop = hash_start + 8
        first_chunk = first_hash[hash_start:hash_stop]
        second_chunk = second_hash[hash_start:hash_stop]
        target_chunk = target_bytes[hash_start:hash_stop]
        word_window = clamp01(
            float(np.mean([float(value) / float(0xFFFFFFFF) for value in word_slice])) if word_slice else 0.0
        )
        first_window = clamp01(float(np.mean([float(value) / 255.0 for value in first_chunk])) if first_chunk else 0.0)
        second_window = clamp01(
            float(np.mean([float(value) / 255.0 for value in second_chunk])) if second_chunk else 0.0
        )
        target_window = clamp01(
            float(np.mean([1.0 - float(value) / 255.0 for value in target_chunk])) if target_chunk else 0.0
        )
        word_windows.append(float(word_window))
        first_hash_windows.append(float(first_window))
        second_hash_windows.append(float(second_window))
        target_windows.append(float(target_window))
        phase_turns.append(
            wrap_turns(
                0.56 * (float(int.from_bytes(first_chunk or b"\x00" * 8, byteorder="big")) / float(1 << 64))
                + 0.24 * word_window
                + 0.20 * target_window
            )
        )
    header_bit_density = clamp01(
        float(sum(int(bin(byte).count("1")) for byte in header_bytes)) / max(float(len(header_bytes) * 8), 1.0)
    )
    prefix_pressure = clamp01(float(count_leading_zero_nibbles(normalized_target)) / 16.0)
    target_density = clamp01(float(np.mean(target_windows)) if target_windows else 0.0)
    schedule_pressure = clamp01(
        0.32 * float(np.mean([abs(word_windows[idx] - word_windows[(idx + 1) % 4]) for idx in range(4)]))
        + 0.24 * exponent_window
        + 0.18 * mantissa_window
        + 0.16 * header_bit_density
        + 0.10 * target_density
    )
    phase_pressure_curve: list[float] = []
    amplitude_pressure_curve: list[float] = []
    target_nibbles = [int(ch, 16) for ch in normalized_target]
    if not target_nibbles:
        target_nibbles = [0]
    for idx, nibble in enumerate(target_nibbles):
        axis_idx = idx % 4
        nibble_pressure = 1.0 - (float(nibble) / 15.0)
        phase_curve = clamp01(
            0.30 * nibble_pressure
            + 0.24 * phase_turns[axis_idx]
            + 0.18 * prefix_pressure
            + 0.16 * schedule_pressure
            + 0.12 * first_hash_windows[axis_idx]
        )
        amplitude_curve = clamp01(
            0.34 * phase_curve
            + 0.24 * target_density
            + 0.18 * second_hash_windows[axis_idx]
            + 0.14 * word_windows[axis_idx]
            + 0.10 * mantissa_window
        )
        phase_pressure_curve.append(float(phase_curve))
        amplitude_pressure_curve.append(float(amplitude_curve))
    phase_pressure = clamp01(float(np.mean(phase_pressure_curve)) if phase_pressure_curve else 0.0)
    amplitude_pressure = clamp01(
        0.46 * phase_pressure
        + 0.24 * target_density
        + 0.18 * prefix_pressure
        + 0.12 * schedule_pressure
    )
    algorithm_bias = clamp01(
        0.34 * phase_pressure
        + 0.28 * amplitude_pressure
        + 0.20 * prefix_pressure
        + 0.18 * header_bit_density
    )
    force_vector = np.array(
        [
            clamp01(
                0.26 * word_windows[0]
                + 0.22 * first_hash_windows[0]
                + 0.18 * phase_pressure
                + 0.16 * target_windows[0]
                + 0.10 * phase_turns[0]
                + 0.08 * exponent_window
            ),
            clamp01(
                0.20 * word_windows[1]
                + 0.16 * first_hash_windows[1]
                + 0.24 * amplitude_pressure
                + 0.16 * target_windows[1]
                + 0.12 * second_hash_windows[1]
                + 0.12 * phase_turns[1]
            ),
            clamp01(
                0.18 * word_windows[2]
                + 0.18 * first_hash_windows[2]
                + 0.18 * second_hash_windows[2]
                + 0.18 * phase_pressure
                + 0.14 * target_windows[2]
                + 0.14 * phase_turns[2]
            ),
            clamp01(
                0.18 * word_windows[3]
                + 0.16 * first_hash_windows[3]
                + 0.18 * second_hash_windows[3]
                + 0.18 * prefix_pressure
                + 0.16 * amplitude_pressure
                + 0.14 * phase_turns[3]
            ),
        ],
        dtype=np.float64,
    )
    force_tensor = np.outer(force_vector, force_vector)
    force_tensor += np.diag([phase_pressure, amplitude_pressure, schedule_pressure, prefix_pressure]) * 0.25
    force_tensor = np.clip(force_tensor, 0.0, 1.35)
    return {
        "header_bit_density": float(header_bit_density),
        "target_density": float(target_density),
        "prefix_pressure": float(prefix_pressure),
        "schedule_pressure": float(schedule_pressure),
        "phase_pressure": float(phase_pressure),
        "amplitude_pressure": float(amplitude_pressure),
        "algorithm_bias": float(algorithm_bias),
        "word_windows": [float(value) for value in word_windows],
        "first_hash_windows": [float(value) for value in first_hash_windows],
        "second_hash_windows": [float(value) for value in second_hash_windows],
        "target_windows": [float(value) for value in target_windows],
        "phase_turns": [float(value) for value in phase_turns],
        "force_vector": [float(value) for value in force_vector],
        "force_tensor": force_tensor.tolist(),
        "phase_pressure_curve": [float(value) for value in phase_pressure_curve],
        "amplitude_pressure_curve": [float(value) for value in amplitude_pressure_curve],
    }


def metric_cross_term(metric: dict[str, Any], lhs: str, rhs: str) -> float:
    cross = dict(metric.get("hessian_cross", {}) or {})
    probes = [
        f"{lhs}{rhs}",
        f"{rhs}{lhs}",
        f"{lhs.lower()}{rhs.lower()}",
        f"{rhs.lower()}{lhs.lower()}",
        f"{lhs.upper()}{rhs.upper()}",
        f"{rhs.upper()}{lhs.upper()}",
    ]
    for probe in probes:
        if probe in cross:
            return float(cross.get(probe, 0.0))
    return 0.0


def matrix_offdiag_mean(values: np.ndarray) -> float:
    matrix = np.array(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0.0
    if matrix.shape[0] == 1:
        return float(matrix[0, 0])
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(matrix[mask]))


def clamp_vector_norm(values: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= max(max_norm, 1.0e-9):
        return values
    return values * (float(max_norm) / max(norm, 1.0e-9))


def vector_similarity(lhs: np.ndarray | list[float], rhs: np.ndarray | list[float]) -> float:
    left = np.array(lhs, dtype=np.float64).reshape(-1)
    right = np.array(rhs, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-9 or right_norm <= 1.0e-9:
        return clamp01(1.0 - float(np.mean(np.abs(left - right))))
    cosine = float(np.dot(left, right) / max(left_norm * right_norm, 1.0e-9))
    cosine = clamp01(0.5 + 0.5 * cosine)
    abs_delta = clamp01(float(np.mean(np.abs(left - right))))
    return clamp01(0.58 * cosine + 0.42 * (1.0 - abs_delta))


def quantize_vector_key(values: np.ndarray | list[float], bucket_count: int = 5) -> tuple[int, ...]:
    vector = np.array(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return tuple()
    return tuple(int(round(clamp01(float(value)) * float(bucket_count))) for value in vector)


GPU_PULSE_DOF_LABELS = (
    "frequency",
    "amplitude",
    "voltage",
    "amperage",
    "frequency_amplitude",
    "frequency_voltage",
    "frequency_amperage",
    "amplitude_voltage",
    "amplitude_amperage",
    "voltage_amperage",
)


def build_gpu_pulse_dof_basis(
    axis_wave_norm: dict[str, Any],
    axis_step_interval: dict[str, Any],
    coupling_gradient_field: dict[str, Any],
    field_environment: dict[str, Any],
    wave_step_field: dict[str, Any],
    field_pressure: float,
    larger_field_exposure: float,
    sequence_persistence_score: float = 0.0,
    temporal_index_overlap: float = 0.0,
    voltage_frequency_flux: float = 0.0,
    frequency_voltage_flux: float = 0.0,
) -> dict[str, Any]:
    charge_field = clamp01(float(field_environment.get("charge_field", 0.0)))
    lattice_field = clamp01(float(field_environment.get("lattice_field", 0.0)))
    coherence_field = clamp01(float(field_environment.get("coherence_field", 0.0)))
    vacancy_field = clamp01(float(field_environment.get("vacancy_field", 0.0)))
    unison_interval = clamp01(float(wave_step_field.get("unison_interval", 0.0)))
    lateral_interval = clamp01(float(wave_step_field.get("lateral_interval", 0.0)))
    cascade_interval = clamp01(float(wave_step_field.get("cascade_interval", 0.0)))
    field_pressure = clamp01(float(field_pressure))
    larger_field_exposure = clamp01(float(larger_field_exposure))
    sequence_persistence_score = clamp01(float(sequence_persistence_score))
    temporal_index_overlap = clamp01(float(temporal_index_overlap))
    voltage_frequency_flux = clamp01(float(voltage_frequency_flux))
    frequency_voltage_flux = clamp01(float(frequency_voltage_flux))

    dof_vector = np.array(
        [
            clamp01(
                0.40 * float(axis_wave_norm.get("F", 0.0))
                + 0.16 * charge_field
                + 0.10 * unison_interval
                + 0.10 * field_pressure
                + 0.14 * voltage_frequency_flux
                + 0.10 * sequence_persistence_score
            ),
            clamp01(
                0.40 * float(axis_wave_norm.get("A", 0.0))
                + 0.18 * lattice_field
                + 0.10 * unison_interval
                + 0.10 * larger_field_exposure
                + 0.12 * temporal_index_overlap
                + 0.10 * sequence_persistence_score
            ),
            clamp01(
                0.36 * float(axis_wave_norm.get("V", 0.0))
                + 0.18 * vacancy_field
                + 0.14 * float(axis_step_interval.get("V", 0.0))
                + 0.10 * cascade_interval
                + 0.10 * voltage_frequency_flux
                + 0.12 * frequency_voltage_flux
            ),
            clamp01(
                0.36 * float(axis_wave_norm.get("I", 0.0))
                + 0.18 * coherence_field
                + 0.14 * float(axis_step_interval.get("I", 0.0))
                + 0.10 * lateral_interval
                + 0.12 * sequence_persistence_score
                + 0.10 * temporal_index_overlap
            ),
            clamp01(
                0.52 * float(coupling_gradient_field.get("F_A", 0.0))
                + 0.14 * float(axis_wave_norm.get("F", 0.0))
                + 0.14 * float(axis_wave_norm.get("A", 0.0))
                + 0.10 * sequence_persistence_score
                + 0.10 * temporal_index_overlap
            ),
            clamp01(
                0.50 * float(coupling_gradient_field.get("F_V", 0.0))
                + 0.16 * voltage_frequency_flux
                + 0.12 * float(axis_wave_norm.get("F", 0.0))
                + 0.12 * float(axis_wave_norm.get("V", 0.0))
                + 0.10 * field_pressure
            ),
            clamp01(
                0.50 * float(coupling_gradient_field.get("F_I", 0.0))
                + 0.14 * float(axis_wave_norm.get("F", 0.0))
                + 0.14 * float(axis_wave_norm.get("I", 0.0))
                + 0.12 * sequence_persistence_score
                + 0.10 * coherence_field
            ),
            clamp01(
                0.50 * float(coupling_gradient_field.get("A_V", 0.0))
                + 0.14 * float(axis_wave_norm.get("A", 0.0))
                + 0.14 * float(axis_wave_norm.get("V", 0.0))
                + 0.12 * frequency_voltage_flux
                + 0.10 * larger_field_exposure
            ),
            clamp01(
                0.50 * float(coupling_gradient_field.get("A_I", 0.0))
                + 0.14 * float(axis_wave_norm.get("A", 0.0))
                + 0.14 * float(axis_wave_norm.get("I", 0.0))
                + 0.12 * sequence_persistence_score
                + 0.10 * temporal_index_overlap
            ),
            clamp01(
                0.48 * float(coupling_gradient_field.get("I_V", 0.0))
                + 0.14 * float(axis_wave_norm.get("V", 0.0))
                + 0.14 * float(axis_wave_norm.get("I", 0.0))
                + 0.12 * voltage_frequency_flux
                + 0.12 * frequency_voltage_flux
            ),
        ],
        dtype=np.float64,
    )
    dof_tensor = np.eye(10, dtype=np.float64) * 0.66
    dof_tensor += np.diag(dof_vector * 0.34)
    pair_to_primary = {
        4: (0, 1),
        5: (0, 2),
        6: (0, 3),
        7: (1, 2),
        8: (1, 3),
        9: (2, 3),
    }
    for pair_idx, (lhs_idx, rhs_idx) in pair_to_primary.items():
        coupling_value = float(dof_vector[pair_idx])
        for primary_idx in (lhs_idx, rhs_idx):
            edge_value = 0.10 + 0.22 * coupling_value
            dof_tensor[pair_idx, primary_idx] += edge_value
            dof_tensor[primary_idx, pair_idx] += edge_value
        primary_edge = 0.04 + 0.10 * coupling_value
        dof_tensor[lhs_idx, rhs_idx] += primary_edge
        dof_tensor[rhs_idx, lhs_idx] += primary_edge
    pair_indexes = list(pair_to_primary.keys())
    for lhs_idx in pair_indexes:
        lhs_axes = set(pair_to_primary[lhs_idx])
        for rhs_idx in pair_indexes:
            if rhs_idx <= lhs_idx:
                continue
            rhs_axes = set(pair_to_primary[rhs_idx])
            shared_axes = len(lhs_axes.intersection(rhs_axes))
            if shared_axes <= 0:
                continue
            pair_coupling = 0.03 + 0.08 * shared_axes * float(dof_vector[lhs_idx] + dof_vector[rhs_idx]) * 0.5
            dof_tensor[lhs_idx, rhs_idx] += pair_coupling
            dof_tensor[rhs_idx, lhs_idx] += pair_coupling
    dof_tensor = np.clip(dof_tensor, 0.0, 1.45)
    projection_tensor = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.50, 0.50, 0.50, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.50, 0.50, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 0.50, 0.50],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.50, 0.0, 0.50, 0.0, 0.50],
        ],
        dtype=np.float64,
    )
    projection_tensor /= np.sum(projection_tensor, axis=1, keepdims=True)
    axis_projection = projection_tensor @ dof_vector
    axis_tensor = projection_tensor @ dof_tensor @ projection_tensor.T
    axis_tensor = np.clip(axis_tensor, 0.0, 1.45)
    return {
        "gpu_pulse_dof_labels": list(GPU_PULSE_DOF_LABELS),
        "gpu_pulse_dof_vector": [float(value) for value in dof_vector],
        "gpu_pulse_dof_tensor": dof_tensor.tolist(),
        "gpu_pulse_projection_tensor": projection_tensor.tolist(),
        "gpu_pulse_axis_projection": [float(value) for value in axis_projection],
        "gpu_pulse_axis_tensor": axis_tensor.tolist(),
    }


def build_encoded_wave_state(
    quartet: dict[str, float],
    pulse_sweep: dict[str, Any],
    lattice_calibration: dict[str, Any],
    simulation_field_state: dict[str, Any],
    deviation_ops: dict[str, Any],
    baseline_frequency_norm: float,
    amplitude_cap: float,
    temporal_persistence: float,
    prediction_confidence: float,
    residual_norm: float,
    coupling_norm: float,
    pulse_index: int,
    bin_count: int,
) -> dict[str, Any]:
    score_metric = dict(deviation_ops.get("score", {}) or {})
    coherence_metric = dict(deviation_ops.get("coherence", {}) or {})
    curvature_metric = dict(deviation_ops.get("curvature", {}) or {})
    sweep_deltas = dict(pulse_sweep.get("deltas", {}) or {})
    calibration_vector = np.array(
        list(lattice_calibration.get("calibration_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if calibration_vector.shape[0] != 4:
        calibration_vector = np.zeros(4, dtype=np.float64)
    environment_tensor = np.array(
        lattice_calibration.get("environment_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if environment_tensor.shape != (4, 4):
        environment_tensor = np.zeros((4, 4), dtype=np.float64)
    field_pressure = float(lattice_calibration.get("field_pressure", 0.0))
    larger_field_exposure = float(lattice_calibration.get("larger_field_exposure", 0.0))
    dominant_basin = dict(lattice_calibration.get("dominant_basin", {}) or {})
    dominant_basin_depth = float(dominant_basin.get("depth", 0.0))
    axis_wave_norm = dict(lattice_calibration.get("axis_wave_norm", {}) or {})
    axis_step_interval = dict(lattice_calibration.get("axis_step_interval", {}) or {})
    coupling_gradient_field = dict(lattice_calibration.get("coupling_gradient_field", {}) or {})
    simulation_field_vector = np.array(
        list(simulation_field_state.get("simulation_field_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if simulation_field_vector.shape[0] != 4:
        simulation_field_vector = np.zeros(4, dtype=np.float64)
    trace_state = dict(simulation_field_state.get("substrate_trace_state", {}) or {})
    trace_vector = np.array(
        list(trace_state.get("trace_vector", simulation_field_vector.tolist()) or simulation_field_vector.tolist()),
        dtype=np.float64,
    )
    if trace_vector.shape[0] != 4:
        trace_vector = np.array(simulation_field_vector, dtype=np.float64)
    trace_axis_vector = np.array(
        list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if trace_axis_vector.shape[0] != 4:
        trace_axis_vector = np.zeros(4, dtype=np.float64)
    trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
    trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
    trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
    field_frequency_bias = float(simulation_field_state.get("field_frequency_bias", 0.0))
    field_amplitude_bias = float(simulation_field_state.get("field_amplitude_bias", 0.0))
    calibration_readiness = float(simulation_field_state.get("calibration_readiness", 0.0))
    entry_trigger = bool(simulation_field_state.get("entry_trigger", False))
    temporal_sequence_vector = np.array(
        list(simulation_field_state.get("persistent_temporal_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if temporal_sequence_vector.shape[0] != 4:
        temporal_sequence_vector = np.zeros(4, dtype=np.float64)
    temporal_correlation_tensor = np.array(
        simulation_field_state.get(
            "voltage_frequency_correlation_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if temporal_correlation_tensor.shape != (4, 4):
        temporal_correlation_tensor = np.zeros((4, 4), dtype=np.float64)
    projected_temporal_dof_vector = np.array(
        list(simulation_field_state.get("projected_temporal_dof_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if projected_temporal_dof_vector.shape[0] != 4:
        projected_temporal_dof_vector = np.zeros(4, dtype=np.float64)
    temporal_dof_axis_tensor = np.array(
        simulation_field_state.get(
            "voltage_frequency_dof_axis_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if temporal_dof_axis_tensor.shape != (4, 4):
        temporal_dof_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    btc_network_force_vector = np.array(
        list(simulation_field_state.get("btc_network_force_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if btc_network_force_vector.shape[0] != 4:
        btc_network_force_vector = np.zeros(4, dtype=np.float64)
    btc_network_force_tensor = np.array(
        simulation_field_state.get(
            "btc_network_force_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if btc_network_force_tensor.shape != (4, 4):
        btc_network_force_tensor = np.zeros((4, 4), dtype=np.float64)
    btc_network_phase_turns = [
        wrap_turns(float(value))
        for value in list(
            simulation_field_state.get("btc_network_phase_turns", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]
        )
    ]
    if len(btc_network_phase_turns) < 4:
        btc_network_phase_turns = [0.0, 0.0, 0.0, 0.0]
    btc_network_phase_pressure = clamp01(float(simulation_field_state.get("btc_network_phase_pressure", 0.0)))
    btc_network_amplitude_pressure = clamp01(
        float(simulation_field_state.get("btc_network_amplitude_pressure", 0.0))
    )
    btc_network_algorithm_bias = clamp01(float(simulation_field_state.get("btc_network_algorithm_bias", 0.0)))
    gpu_pulse_feedback = dict(simulation_field_state.get("gpu_pulse_feedback", {}) or {})
    feedback_axis_vector = np.array(
        list(
            simulation_field_state.get(
                "feedback_axis_vector",
                gpu_pulse_feedback.get("feedback_axis_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if feedback_axis_vector.shape[0] != 4:
        feedback_axis_vector = np.zeros(4, dtype=np.float64)
    feedback_axis_tensor = np.array(
        simulation_field_state.get(
            "feedback_axis_tensor",
            gpu_pulse_feedback.get("feedback_axis_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        ),
        dtype=np.float64,
    )
    if feedback_axis_tensor.shape != (4, 4):
        feedback_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    feedback_phase_anchor_turns = wrap_turns(float(gpu_pulse_feedback.get("phase_anchor_turns", 0.0)))
    feedback_phase_alignment = clamp01(float(gpu_pulse_feedback.get("phase_alignment", 0.0)))
    feedback_memory_proxy = clamp01(float(gpu_pulse_feedback.get("memory_proxy", 0.0)))
    feedback_flux_proxy = clamp01(float(gpu_pulse_feedback.get("flux_proxy", 0.0)))
    feedback_stability_proxy = clamp01(float(gpu_pulse_feedback.get("stability_proxy", 0.0)))
    feedback_temporal_drive = clamp01(float(gpu_pulse_feedback.get("temporal_drive", 0.0)))
    feedback_axis_values = np.array(
        [
            clamp01(float(gpu_pulse_feedback.get("frequency_observable", float(feedback_axis_vector[0])))),
            clamp01(float(gpu_pulse_feedback.get("amplitude_observable", float(feedback_axis_vector[1])))),
            clamp01(float(gpu_pulse_feedback.get("current_observable", float(feedback_axis_vector[2])))),
            clamp01(float(gpu_pulse_feedback.get("voltage_observable", float(feedback_axis_vector[3])))),
        ],
        dtype=np.float64,
    )
    feedback_axis_deltas = np.array(
        [
            float(gpu_pulse_feedback.get("dln_frequency", 0.0)),
            float(gpu_pulse_feedback.get("dln_amplitude", 0.0)),
            float(gpu_pulse_feedback.get("dln_current", 0.0)),
            float(gpu_pulse_feedback.get("dln_voltage", 0.0)),
        ],
        dtype=np.float64,
    )
    feedback_axis_curvatures = np.array(
        [
            float(gpu_pulse_feedback.get("ddln_frequency", 0.0)),
            float(gpu_pulse_feedback.get("ddln_amplitude", 0.0)),
            float(gpu_pulse_feedback.get("ddln_current", 0.0)),
            float(gpu_pulse_feedback.get("ddln_voltage", 0.0)),
        ],
        dtype=np.float64,
    )
    projected_temporal_dof_vector = np.array(
        list(simulation_field_state.get("projected_temporal_dof_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if projected_temporal_dof_vector.shape[0] != 4:
        projected_temporal_dof_vector = np.zeros(4, dtype=np.float64)
    temporal_dof_axis_tensor = np.array(
        simulation_field_state.get(
            "voltage_frequency_dof_axis_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if temporal_dof_axis_tensor.shape != (4, 4):
        temporal_dof_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    sequence_persistence_score = float(simulation_field_state.get("sequence_persistence_score", 0.0))
    temporal_index_overlap = float(simulation_field_state.get("temporal_index_overlap", 0.0))
    temporal_sequence_index = int(simulation_field_state.get("temporal_sequence_index", 0))
    sequence_length = max(1, int(simulation_field_state.get("temporal_sequence_length", 1)))
    sequence_index_norm = clamp01(float(temporal_sequence_index) / float(sequence_length))
    voltage_frequency_flux = float(simulation_field_state.get("voltage_frequency_flux", 0.0))
    basis_specs = [
        (
            "F",
            "freq_axis",
            float(quartet.get("f_code", 0.0)),
            baseline_frequency_norm,
            0.07,
            float(axis_wave_norm.get("F", baseline_frequency_norm)),
            float(axis_step_interval.get("F", 0.0)),
        ),
        (
            "A",
            "amp_axis",
            float(quartet.get("a_code", 0.0)),
            amplitude_cap,
            0.19,
            float(axis_wave_norm.get("A", amplitude_cap)),
            float(axis_step_interval.get("A", 0.0)),
        ),
        (
            "I",
            "curr_axis",
            float(quartet.get("i_code", 0.0)),
            temporal_persistence,
            0.31,
            float(axis_wave_norm.get("I", temporal_persistence)),
            float(axis_step_interval.get("I", 0.0)),
        ),
        (
            "V",
            "volt_axis",
            float(quartet.get("v_code", 0.0)),
            prediction_confidence,
            0.43,
            float(axis_wave_norm.get("V", prediction_confidence)),
            float(axis_step_interval.get("V", 0.0)),
        ),
    ]
    basis_modes: list[dict[str, Any]] = []
    latent_components: list[float] = []
    for axis_idx, (axis, label, code_value, anchor, phase_offset, wave_norm, step_interval) in enumerate(basis_specs):
        delta = float(sweep_deltas.get(axis, 0.0))
        calibration_bias = float(calibration_vector[axis_idx])
        feedback_axis_value = float(feedback_axis_values[axis_idx])
        feedback_axis_delta = float(feedback_axis_deltas[axis_idx])
        feedback_axis_curvature = float(feedback_axis_curvatures[axis_idx])
        feedback_axis_cross = float(np.mean(feedback_axis_tensor[axis_idx]))
        phase = 2.0 * math.pi * (
            code_value
            + phase_offset
            + pulse_index * 0.03125
            + step_interval * 0.50
            + float(simulation_field_vector[axis_idx]) * 0.08
            + float(temporal_sequence_vector[axis_idx]) * 0.07
            + float(projected_temporal_dof_vector[axis_idx]) * 0.05
            + sequence_index_norm * 0.05
            + delta * 6.0
            + calibration_bias * 0.045
            + feedback_phase_anchor_turns * 0.14
            + feedback_axis_value * 0.08
            + feedback_phase_alignment * 0.04
            + feedback_axis_delta * 0.025
            - feedback_axis_curvature * 0.012
            + btc_network_phase_turns[axis_idx] * 0.10
            + float(btc_network_force_vector[axis_idx]) * 0.08
        )
        amplitude = clamp01(
            0.30
            + 0.30 * code_value
            + 0.18 * anchor
            + 0.08 * wave_norm
            + 0.08 * field_amplitude_bias
            + 0.04 * calibration_readiness
            + 0.04 * sequence_persistence_score
            + 0.03 * voltage_frequency_flux
            + 0.03 * float(projected_temporal_dof_vector[axis_idx])
            + 0.10 * (1.0 - residual_norm)
            + 0.08 * prediction_confidence
            + 0.08 * field_pressure
            + 0.05 * max(0.0, calibration_bias)
            + 0.06 * feedback_axis_value
            + 0.05 * feedback_memory_proxy
            + 0.04 * feedback_stability_proxy
            + 0.03 * feedback_axis_cross
            + 0.10 * btc_network_amplitude_pressure
            + 0.06 * float(btc_network_force_vector[axis_idx])
            - 0.06 * abs(delta) * 25.0
            - 0.03 * abs(feedback_axis_curvature)
        )
        confinement = clamp01(
            amplitude_cap * (0.62 + 0.38 * anchor)
            + temporal_persistence * 0.12
            + coupling_norm * 0.10
            + 0.06 * wave_norm
            + 0.05 * step_interval
            + 0.08 * calibration_readiness
            + 0.06 * sequence_persistence_score
            + 0.04 * temporal_index_overlap
            + 0.04 * float(projected_temporal_dof_vector[axis_idx])
            + field_pressure * 0.14
            + dominant_basin_depth * 0.10
            + larger_field_exposure * 0.06
            + 0.08 * feedback_memory_proxy
            + 0.05 * feedback_phase_alignment
            + 0.04 * feedback_axis_cross
            + 0.03 * feedback_temporal_drive
            + 0.02 * feedback_flux_proxy
            + 0.08 * btc_network_phase_pressure
            + 0.06 * btc_network_amplitude_pressure
            - abs(delta) * 10.0
            - 0.02 * abs(feedback_axis_curvature)
        )
        frequency = 1.0 + float(bin_count) * (
            0.12
            + 0.52 * anchor
            + 0.10 * code_value
            + 0.08 * wave_norm
            + 0.06 * step_interval
            + 0.05 * field_frequency_bias
            + 0.05 * voltage_frequency_flux
            + 0.04 * float(simulation_field_vector[axis_idx])
            + 0.04 * float(temporal_sequence_vector[axis_idx])
            + 0.04 * float(projected_temporal_dof_vector[axis_idx])
            + 0.06 * feedback_axis_value
            + 0.05 * feedback_phase_alignment
            + 0.04 * feedback_flux_proxy
            + feedback_axis_delta * 0.20
            - abs(feedback_axis_curvature) * 0.08
            + delta * 4.0
            + float(btc_network_force_vector[axis_idx]) * 0.18
            + btc_network_phase_turns[axis_idx] * 0.12
        ) + calibration_bias * float(bin_count) * 0.08 + larger_field_exposure * float(bin_count) * 0.02
        quadrature = amplitude * (0.6 * math.cos(phase) + 0.4 * math.sin(phase))
        latent_components.append(
            quadrature
            + calibration_bias * 0.18
            + (wave_norm - 0.5) * 0.10
            + float(simulation_field_vector[axis_idx]) * 0.12
            + float(temporal_sequence_vector[axis_idx]) * 0.10
            + float(projected_temporal_dof_vector[axis_idx]) * 0.08
            + feedback_axis_value * 0.10
            + feedback_axis_cross * 0.08
            + feedback_phase_alignment * 0.05
            + float(btc_network_force_vector[axis_idx]) * 0.10
            + btc_network_algorithm_bias * 0.06
        )
        basis_modes.append(
            {
                "axis": axis,
                "label": label,
                "code_value": code_value,
                "amplitude": amplitude,
                "phase": phase,
                "frequency": frequency,
                "confinement": confinement,
                "delta": delta,
                "wave_norm": wave_norm,
                "step_interval": step_interval,
                "calibration_bias": calibration_bias,
                "field_bias": float(simulation_field_vector[axis_idx]),
                "feedback_axis_value": float(feedback_axis_value),
                "feedback_axis_delta": float(feedback_axis_delta),
                "feedback_axis_cross": float(feedback_axis_cross),
            }
        )

    coupling_matrix = np.eye(4, dtype=np.float64) * 0.82
    pair_index = {
        ("F", "A"): (0, 1),
        ("F", "I"): (0, 2),
        ("F", "V"): (0, 3),
        ("A", "I"): (1, 2),
        ("A", "V"): (1, 3),
        ("I", "V"): (2, 3),
    }
    for pair, (row_idx, col_idx) in pair_index.items():
        lhs, rhs = pair
        mixed = (
            metric_cross_term(score_metric, lhs, rhs) * 0.35
            + metric_cross_term(coherence_metric, lhs, rhs) * 0.45
            + metric_cross_term(curvature_metric, lhs, rhs) * 0.20
        )
        coupling_gradient = float(
            coupling_gradient_field.get(f"{lhs}_{rhs}", coupling_gradient_field.get(f"{rhs}_{lhs}", 0.0))
        )
        mixed_norm = math.tanh(mixed * 48.0 + coupling_gradient * 0.75)
        coupling_matrix[row_idx, col_idx] = mixed_norm
        coupling_matrix[col_idx, row_idx] = mixed_norm

    coupling_matrix += environment_tensor * (0.08 + 0.12 * field_pressure)
    coupling_matrix += np.outer(simulation_field_vector, simulation_field_vector) * (
        0.04 + 0.04 * calibration_readiness
    )
    coupling_matrix += temporal_correlation_tensor * (
        0.05 + 0.08 * sequence_persistence_score + 0.04 * temporal_index_overlap
    )
    coupling_matrix += temporal_dof_axis_tensor * (
        0.04 + 0.06 * sequence_persistence_score + 0.04 * voltage_frequency_flux
    )
    coupling_matrix += np.outer(temporal_sequence_vector, temporal_sequence_vector) * (
        0.02 + 0.04 * sequence_persistence_score
    )
    coupling_matrix += np.outer(projected_temporal_dof_vector, projected_temporal_dof_vector) * (
        0.02 + 0.03 * temporal_index_overlap
    )
    coupling_matrix += feedback_axis_tensor * (
        0.04 + 0.08 * feedback_memory_proxy + 0.04 * feedback_phase_alignment
    )
    coupling_matrix += np.outer(feedback_axis_vector, feedback_axis_vector) * (
        0.02 + 0.06 * feedback_flux_proxy
    )
    coupling_matrix += btc_network_force_tensor * (
        0.04 + 0.08 * btc_network_phase_pressure + 0.04 * btc_network_algorithm_bias
    )
    coupling_matrix += np.outer(btc_network_force_vector, btc_network_force_vector) * (
        0.02 + 0.06 * btc_network_amplitude_pressure
    )
    coupling_matrix = np.clip(coupling_matrix, -1.25, 1.25)
    latent_vector = clamp_vector_norm(
        np.array(latent_components, dtype=np.float64),
        max_norm=max(0.45, (2.40 + field_pressure * 0.35 + larger_field_exposure * 0.25) * amplitude_cap),
    )
    return {
        "basis_modes": basis_modes,
        "latent_vector": [float(value) for value in latent_vector],
        "coupling_matrix": coupling_matrix.tolist(),
        "state_norm": float(np.linalg.norm(latent_vector)),
        "amplitude_cap": float(amplitude_cap),
        "cross_terms": {
            "F_A": float(coupling_matrix[0, 1]),
            "F_I": float(coupling_matrix[0, 2]),
            "F_V": float(coupling_matrix[0, 3]),
            "A_I": float(coupling_matrix[1, 2]),
            "A_V": float(coupling_matrix[1, 3]),
            "I_V": float(coupling_matrix[2, 3]),
        },
        "field_pressure": float(field_pressure),
        "larger_field_exposure": float(larger_field_exposure),
        "calibration_readiness": float(calibration_readiness),
        "entry_trigger": entry_trigger,
        "temporal_sequence_index": int(temporal_sequence_index),
        "sequence_persistence_score": float(sequence_persistence_score),
        "voltage_frequency_flux": float(voltage_frequency_flux),
        "dominant_basin": dict(dominant_basin),
    }


def evolve_temporal_manifold_state(
    encoded_state: dict[str, Any],
    simulation_field_state: dict[str, Any],
    lattice_calibration: dict[str, Any],
    pulse_index: int,
    nesting_depth: int,
    global_theta: float,
    temporal_persistence: float,
    baseline_coherence: float,
    baseline_shared: float,
    residual_norm: float,
    coupling_norm: float,
) -> dict[str, Any]:
    latent = np.array(list(encoded_state.get("latent_vector", []) or [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    coupling_matrix = np.array(encoded_state.get("coupling_matrix", np.eye(4).tolist()), dtype=np.float64)
    amplitude_cap = float(encoded_state.get("amplitude_cap", 0.5))
    basis_modes = list(encoded_state.get("basis_modes", []) or [])
    calibration_vector = np.array(
        list(lattice_calibration.get("calibration_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if calibration_vector.shape[0] != 4:
        calibration_vector = np.zeros(4, dtype=np.float64)
    environment_tensor = np.array(
        lattice_calibration.get("environment_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if environment_tensor.shape != (4, 4):
        environment_tensor = np.zeros((4, 4), dtype=np.float64)
    field_pressure = float(lattice_calibration.get("field_pressure", 0.0))
    larger_field_exposure = float(lattice_calibration.get("larger_field_exposure", 0.0))
    temporal_gain = float(lattice_calibration.get("temporal_gain", temporal_persistence))
    temporal_sequence_vector = np.array(
        list(simulation_field_state.get("persistent_temporal_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if temporal_sequence_vector.shape[0] != 4:
        temporal_sequence_vector = np.zeros(4, dtype=np.float64)
    temporal_correlation_tensor = np.array(
        simulation_field_state.get(
            "voltage_frequency_correlation_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if temporal_correlation_tensor.shape != (4, 4):
        temporal_correlation_tensor = np.zeros((4, 4), dtype=np.float64)
    projected_temporal_dof_vector = np.array(
        list(simulation_field_state.get("projected_temporal_dof_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if projected_temporal_dof_vector.shape[0] != 4:
        projected_temporal_dof_vector = np.zeros(4, dtype=np.float64)
    temporal_dof_axis_tensor = np.array(
        simulation_field_state.get(
            "voltage_frequency_dof_axis_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if temporal_dof_axis_tensor.shape != (4, 4):
        temporal_dof_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    gpu_pulse_feedback = dict(simulation_field_state.get("gpu_pulse_feedback", {}) or {})
    feedback_axis_vector = np.array(
        list(
            simulation_field_state.get(
                "feedback_axis_vector",
                gpu_pulse_feedback.get("feedback_axis_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if feedback_axis_vector.shape[0] != 4:
        feedback_axis_vector = np.zeros(4, dtype=np.float64)
    feedback_axis_tensor = np.array(
        simulation_field_state.get(
            "feedback_axis_tensor",
            gpu_pulse_feedback.get("feedback_axis_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        ),
        dtype=np.float64,
    )
    if feedback_axis_tensor.shape != (4, 4):
        feedback_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    feedback_phase_anchor_turns = wrap_turns(float(gpu_pulse_feedback.get("phase_anchor_turns", 0.0)))
    feedback_phase_alignment = clamp01(float(gpu_pulse_feedback.get("phase_alignment", 0.0)))
    feedback_memory_proxy = clamp01(float(gpu_pulse_feedback.get("memory_proxy", 0.0)))
    feedback_flux_proxy = clamp01(float(gpu_pulse_feedback.get("flux_proxy", 0.0)))
    feedback_temporal_drive = clamp01(float(gpu_pulse_feedback.get("temporal_drive", 0.0)))
    temporal_sequence_indexes = [
        int(index)
        for index in list(simulation_field_state.get("temporal_sequence_indexes", []) or [])
    ]
    temporal_sequence_index = int(simulation_field_state.get("temporal_sequence_index", pulse_index))
    sequence_persistence_score = float(simulation_field_state.get("sequence_persistence_score", 0.0))
    temporal_index_overlap = float(simulation_field_state.get("temporal_index_overlap", 0.0))
    voltage_frequency_flux = float(simulation_field_state.get("voltage_frequency_flux", 0.0))
    temporal_history: list[list[float]] = []
    for step_idx in range(max(1, int(nesting_depth))):
        sequence_step_index = (
            temporal_sequence_indexes[step_idx % len(temporal_sequence_indexes)]
            if temporal_sequence_indexes
            else temporal_sequence_index
        )
        drive_phase = (
            global_theta
            + feedback_phase_anchor_turns * math.tau
            + pulse_index * 0.23
            + sequence_step_index * (0.07 + voltage_frequency_flux * 0.12)
            + step_idx * (0.11 + temporal_persistence * 0.19 + feedback_temporal_drive * 0.08)
        )
        drive = np.array(
            [
                math.cos(drive_phase + float((basis_modes[0] if len(basis_modes) > 0 else {}).get("phase", 0.0))),
                math.sin(drive_phase + float((basis_modes[1] if len(basis_modes) > 1 else {}).get("phase", 0.0))),
                math.cos(drive_phase + float((basis_modes[2] if len(basis_modes) > 2 else {}).get("phase", 0.0))),
                math.sin(drive_phase + float((basis_modes[3] if len(basis_modes) > 3 else {}).get("phase", 0.0))),
            ],
            dtype=np.float64,
        )
        transport = (coupling_matrix @ latent) * (0.18 + 0.14 * coupling_norm)
        transport += (environment_tensor @ latent) * (0.06 + 0.10 * larger_field_exposure)
        transport += (temporal_correlation_tensor @ latent) * (
            0.04 + 0.08 * sequence_persistence_score + 0.04 * temporal_index_overlap
        )
        transport += (temporal_dof_axis_tensor @ latent) * (
            0.03 + 0.06 * sequence_persistence_score + 0.04 * voltage_frequency_flux
        )
        transport += (feedback_axis_tensor @ latent) * (
            0.03 + 0.06 * feedback_memory_proxy + 0.04 * feedback_phase_alignment
        )
        latent = latent * (0.76 + 0.08 * baseline_shared + 0.05 * field_pressure)
        latent += transport
        latent += drive * (0.08 + 0.08 * temporal_persistence + 0.04 * residual_norm)
        latent += calibration_vector * (0.05 + 0.07 * field_pressure + 0.04 * temporal_gain)
        latent += temporal_sequence_vector * (
            0.04 + 0.06 * sequence_persistence_score + 0.04 * voltage_frequency_flux
        )
        latent += projected_temporal_dof_vector * (
            0.03 + 0.05 * sequence_persistence_score + 0.04 * temporal_index_overlap
        )
        latent += feedback_axis_vector * (
            0.04 + 0.06 * feedback_temporal_drive + 0.04 * feedback_flux_proxy
        )
        latent = clamp_vector_norm(
            latent,
            max_norm=max(0.55, (2.65 + field_pressure * 0.25 + larger_field_exposure * 0.20) * amplitude_cap),
        )
        temporal_history.append([float(value) for value in latent])

    history_matrix = np.array(temporal_history if temporal_history else [latent.tolist()], dtype=np.float64)
    step_energy = np.linalg.norm(np.diff(history_matrix, axis=0), axis=1) if len(history_matrix) > 1 else np.array([0.0])
    coherence_norm = clamp01(
        0.38 * baseline_coherence
        + 0.24 * temporal_persistence
        + 0.18 * baseline_shared
        + 0.10 * field_pressure
        + 0.06 * larger_field_exposure
        + 0.12 * (1.0 - float(np.std(history_matrix)))
        + 0.08 * (1.0 - float(np.mean(step_energy)))
        + 0.08 * feedback_memory_proxy
        + 0.06 * feedback_phase_alignment
    )
    field_resonance = clamp01(
        0.42 * field_pressure
        + 0.28 * larger_field_exposure
        + 0.18 * temporal_gain
        + 0.12 * coherence_norm
        + 0.08 * feedback_flux_proxy
    )
    return {
        "latent_vector": [float(value) for value in latent],
        "temporal_history": temporal_history,
        "coherence_norm": float(coherence_norm),
        "transport_norm": float(clamp01(np.mean(step_energy) / max(amplitude_cap * 1.8, 1.0e-6))),
        "history_norm": float(np.linalg.norm(history_matrix[-1])),
        "field_resonance": float(field_resonance),
        "temporal_sequence_index": int(temporal_sequence_index),
        "temporal_sequence_indexes": temporal_sequence_indexes,
        "sequence_persistence_score": float(sequence_persistence_score),
        "temporal_index_overlap": float(temporal_index_overlap),
        "voltage_frequency_flux": float(voltage_frequency_flux),
        "environment_tensor": environment_tensor.tolist(),
    }


def project_effective_vector(manifold_state: dict[str, Any]) -> dict[str, Any]:
    latent = np.array(list(manifold_state.get("latent_vector", []) or [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    projection_matrix = np.array(
        [
            [0.82, -0.18, 0.29, 0.14],
            [0.16, 0.79, -0.22, 0.24],
            [-0.21, 0.17, 0.76, 0.31],
            [0.12, 0.26, 0.18, 0.87],
        ],
        dtype=np.float64,
    )
    projected = projection_matrix @ latent
    x_val = math.tanh(float(projected[0]))
    y_val = math.tanh(float(projected[1]))
    z_val = math.tanh(float(projected[2]))
    t_eff = clamp01(0.5 + 0.5 * math.tanh(float(projected[3])))
    field_resonance = float(manifold_state.get("field_resonance", 0.0))
    spatial_magnitude = clamp01(
        math.sqrt(x_val * x_val + y_val * y_val + z_val * z_val) / math.sqrt(3.0)
    )
    curvature_pressure = clamp01(
        0.46 * float(manifold_state.get("transport_norm", 0.0))
        + 0.24 * abs(z_val - x_val)
        + 0.18 * abs(y_val - x_val)
        + 0.12 * (1.0 - t_eff)
        + 0.10 * field_resonance
    )
    coherence_bias = clamp01(
        0.50 * float(manifold_state.get("coherence_norm", 0.0))
        + 0.28 * spatial_magnitude
        + 0.22 * t_eff
        + 0.10 * field_resonance
    )
    return {
        "x": float(x_val),
        "y": float(y_val),
        "z": float(z_val),
        "t_eff": float(t_eff),
        "spatial_magnitude": float(spatial_magnitude),
        "curvature_pressure": float(curvature_pressure),
        "coherence_bias": float(coherence_bias),
        "field_resonance": float(field_resonance),
        "projection_matrix": projection_matrix.tolist(),
        "projected_state": [float(value) for value in projected],
    }


def simulate_manifold_path(
    seed_vector: np.ndarray,
    coupling_matrix: np.ndarray,
    path_steps: list[np.ndarray],
    amplitude_cap: float,
) -> np.ndarray:
    latent = np.array(seed_vector, dtype=np.float64)
    for step in path_steps:
        drive = np.array(step, dtype=np.float64)
        latent = latent * 0.81 + (coupling_matrix @ latent) * 0.19 + drive * 0.11
        latent = clamp_vector_norm(latent, max_norm=max(0.55, 2.65 * amplitude_cap))
    return latent


def project_spatial_components(latent: np.ndarray) -> np.ndarray:
    projection_matrix = np.array(
        [
            [0.82, -0.18, 0.29, 0.14],
            [0.16, 0.79, -0.22, 0.24],
            [-0.21, 0.17, 0.76, 0.31],
            [0.12, 0.26, 0.18, 0.87],
        ],
        dtype=np.float64,
    )
    projected = projection_matrix @ np.array(latent, dtype=np.float64)
    return np.array(
        [
            math.tanh(float(projected[0])),
            math.tanh(float(projected[1])),
            math.tanh(float(projected[2])),
            clamp01(0.5 + 0.5 * math.tanh(float(projected[3]))),
        ],
        dtype=np.float64,
    )


def evaluate_manifold_diagnostics(
    encoded_state: dict[str, Any],
    manifold_state: dict[str, Any],
    effective_vector: dict[str, Any],
    pulse_sweep: dict[str, Any],
    residual_norm: float,
    coupling_norm: float,
) -> dict[str, Any]:
    seed_vector = np.array(list(encoded_state.get("latent_vector", []) or [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    coupling_matrix = np.array(encoded_state.get("coupling_matrix", np.eye(4).tolist()), dtype=np.float64)
    amplitude_cap = float(encoded_state.get("amplitude_cap", 0.5))
    sweep_deltas = dict(pulse_sweep.get("deltas", {}) or {})
    drive_axes = np.array(
        [
            float(sweep_deltas.get("F", 0.0)) * 24.0 + residual_norm * 0.12,
            float(sweep_deltas.get("A", 0.0)) * 18.0 + coupling_norm * 0.10,
            residual_norm * 0.18 + float(manifold_state.get("coherence_norm", 0.0)) * 0.06,
            coupling_norm * 0.20 + float(effective_vector.get("t_eff", 0.0)) * 0.05,
        ],
        dtype=np.float64,
    )
    path_a = [
        np.array([drive_axes[0], 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, drive_axes[1], 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, drive_axes[2], 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, drive_axes[3]], dtype=np.float64),
    ]
    path_b = [
        np.array([0.0, drive_axes[1], 0.0, 0.0], dtype=np.float64),
        np.array([drive_axes[0], 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, drive_axes[3]], dtype=np.float64),
        np.array([0.0, 0.0, drive_axes[2], 0.0], dtype=np.float64),
    ]
    path_forward = simulate_manifold_path(seed_vector, coupling_matrix, path_a, amplitude_cap)
    path_alternate = simulate_manifold_path(seed_vector, coupling_matrix, path_b, amplitude_cap)
    path_reverse = simulate_manifold_path(seed_vector, coupling_matrix, list(reversed(path_a)), amplitude_cap)

    proj_forward = project_spatial_components(path_forward)
    proj_alternate = project_spatial_components(path_alternate)
    proj_reverse = project_spatial_components(path_reverse)

    path_equivalence_error = float(np.mean(np.abs(proj_forward - proj_alternate)))
    temporal_ordering_delta = float(np.mean(np.abs(proj_forward - proj_reverse)))

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_seed = np.array(seed_vector, dtype=np.float64)
    rotated_seed[:3] = rotation @ rotated_seed[:3]
    rotated_projection = project_spatial_components(rotated_seed)
    expected_spatial = rotation @ np.array(
        [
            float(effective_vector.get("x", 0.0)),
            float(effective_vector.get("y", 0.0)),
            float(effective_vector.get("z", 0.0)),
        ],
        dtype=np.float64,
    )
    basis_rotation_residual = float(
        np.mean(np.abs(rotated_projection[:3] - expected_spatial))
    )
    return {
        "path_equivalence_error": path_equivalence_error,
        "temporal_ordering_delta": temporal_ordering_delta,
        "basis_rotation_residual": basis_rotation_residual,
        "basis_rotation_stability": float(clamp01(1.0 - basis_rotation_residual * 2.0)),
        "projected_forward": [float(value) for value in proj_forward],
        "projected_alternate": [float(value) for value in proj_alternate],
        "projected_reverse": [float(value) for value in proj_reverse],
    }


def build_silicon_lattice_calibration(
    config: SimulationConfig,
    nist: dict[str, float],
    quartet: dict[str, float],
    deviation_ops: dict[str, Any],
    baseline_frequency_norm: float,
    amplitude_cap: float,
    temporal_persistence: float,
    prediction_confidence: float,
    residual_norm: float,
    coupling_norm: float,
    mean_phase_lock_matrix: np.ndarray,
    amplitude_history: np.ndarray,
    shared_history: np.ndarray,
    coherence_history: np.ndarray,
    curvature_history: np.ndarray,
    step_dominant_freqs: np.ndarray,
    packet_classes: list[dict[str, Any]],
    tensor_gradient_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    lattice_spacing_norm = clamp01(float(nist.get("lattice_constant_m", 5.431020511e-10)) / 6.0e-10)
    excitation_norm = clamp01(float(nist.get("mean_excitation_energy_ev", 173.0)) / 220.0)
    density_norm = clamp01(float(nist.get("density_g_cm3", 2.33)) / 4.0)
    phase_lock_flux = clamp01(matrix_offdiag_mean(mean_phase_lock_matrix))
    amplitude_flux = clamp01(float(np.mean(amplitude_history[-1])) / max(float(config.max_amplitude), 1.0e-9))
    shared_flux = clamp01(float(np.mean(shared_history)))
    coherence_flux = clamp01(float(np.mean(coherence_history)))
    curvature_scale = max(1.0, float(np.max(np.abs(curvature_history))))
    curvature_flux = clamp01(float(np.mean(np.abs(curvature_history))) / curvature_scale)
    dominant_freq_vector = np.mean(step_dominant_freqs[-1], axis=0)
    dominant_freq_norm = [
        clamp01(abs(float(value)) / max(float(config.bin_count), 1.0))
        for value in dominant_freq_vector
    ]
    max_freq_vector = np.max(np.abs(step_dominant_freqs[-1]), axis=0)
    max_freq_norm = [
        clamp01(abs(float(value)) / max(float(config.bin_count), 1.0))
        for value in max_freq_vector
    ]
    if step_dominant_freqs.shape[0] > 1:
        last_frequency_step_interval = clamp01(
            float(np.mean(np.abs(step_dominant_freqs[-1] - step_dominant_freqs[-2])))
            / max(float(config.bin_count), 1.0)
        )
    else:
        last_frequency_step_interval = clamp01(float(np.mean(max_freq_norm)) * 0.25)
    if amplitude_history.shape[0] > 1:
        last_amplitude_step_interval = clamp01(
            float(np.mean(np.abs(amplitude_history[-1] - amplitude_history[-2])))
            / max(float(config.max_amplitude), 1.0e-9)
        )
    else:
        last_amplitude_step_interval = clamp01(amplitude_cap * 0.20)
    kernel_lateral_step = clamp01(
        matrix_offdiag_mean(mean_phase_lock_matrix)
        * (0.42 + 0.58 * last_frequency_step_interval)
    )
    unison_step_interval = clamp01(
        0.62 * last_frequency_step_interval
        + 0.38 * last_amplitude_step_interval
    )
    cascade_interval = clamp01(
        0.58 * unison_step_interval
        + 0.42 * kernel_lateral_step
    )

    shared_count = int(sum(1 for row in packet_classes if str(row.get("classification", "")) == "shared"))
    total_packets = max(1, len(packet_classes))
    shared_fraction = float(shared_count / total_packets)
    individual_fraction = 1.0 - shared_fraction

    phase_gradients: list[float] = []
    amplitude_gradients: list[float] = []
    tensor_norms: list[float] = []
    temporal_inertias: list[float] = []
    oam_norms: list[float] = []
    for item in tensor_gradient_samples:
        phase_gradients.append(
            float(np.mean(np.abs(np.array(item.get("phase_gradient", [0.0, 0.0, 0.0]), dtype=np.float64))))
        )
        amplitude_gradients.append(
            float(np.mean(np.abs(np.array(item.get("amplitude_gradient", [0.0, 0.0, 0.0]), dtype=np.float64))))
        )
        tensor_norms.append(
            float(np.mean(np.abs(np.array(item.get("tensor", np.zeros((3, 3)).tolist()), dtype=np.float64))))
        )
        temporal_inertias.append(abs(float(item.get("temporal_inertia", 0.0))))
        oam_norms.append(abs(float(item.get("oam_twist", 0.0))))

    phase_gradient_norm = clamp01(float(np.mean(phase_gradients) if phase_gradients else 0.0) / 0.25)
    amplitude_gradient_norm = clamp01(float(np.mean(amplitude_gradients) if amplitude_gradients else 0.0) / 0.08)
    tensor_norm = clamp01(float(np.mean(tensor_norms) if tensor_norms else 0.0) / 0.35)
    temporal_inertia_norm = clamp01(float(np.mean(temporal_inertias) if temporal_inertias else 0.0) / 1.5)
    oam_norm = clamp01(float(np.mean(oam_norms) if oam_norms else 0.0) / 1.5)
    tensor_gradient_flux = clamp01(
        0.46 * phase_gradient_norm
        + 0.26 * amplitude_gradient_norm
        + 0.18 * tensor_norm
        + 0.10 * oam_norm
    )
    larger_field_exposure = clamp01(
        0.32 * tensor_gradient_flux
        + 0.22 * phase_lock_flux
        + 0.16 * shared_flux
        + 0.15 * coherence_flux
        + 0.15 * coupling_norm
    )

    charge_field = clamp01(
        0.30 * phase_lock_flux
        + 0.22 * shared_flux
        + 0.18 * density_norm
        + 0.16 * prediction_confidence
        + 0.14 * dominant_freq_norm[0]
    )
    lattice_field = clamp01(
        0.28 * tensor_gradient_flux
        + 0.22 * lattice_spacing_norm
        + 0.18 * density_norm
        + 0.16 * curvature_flux
        + 0.16 * dominant_freq_norm[1]
    )
    coherence_field = clamp01(
        0.28 * coherence_flux
        + 0.22 * temporal_persistence
        + 0.18 * excitation_norm
        + 0.16 * shared_fraction
        + 0.16 * dominant_freq_norm[2]
    )
    vacancy_field = clamp01(
        0.28 * individual_fraction
        + 0.22 * amplitude_flux
        + 0.18 * residual_norm
        + 0.16 * temporal_inertia_norm
        + 0.16 * (1.0 - shared_flux)
    )

    axis_wave_norm = {
        "F": clamp01(
            0.46 * max_freq_norm[0]
            + 0.24 * baseline_frequency_norm
            + 0.18 * phase_lock_flux
            + 0.12 * clamp01(float(quartet.get("f_code", 0.0)))
        ),
        "A": clamp01(
            0.38 * amplitude_flux
            + 0.24 * amplitude_gradient_norm
            + 0.20 * amplitude_cap
            + 0.18 * clamp01(float(quartet.get("a_code", 0.0)))
        ),
        "I": clamp01(
            0.36 * coherence_flux
            + 0.26 * phase_gradient_norm
            + 0.22 * temporal_persistence
            + 0.16 * clamp01(float(quartet.get("i_code", 0.0)))
        ),
        "V": clamp01(
            0.34 * shared_flux
            + 0.24 * prediction_confidence
            + 0.22 * coupling_norm
            + 0.20 * clamp01(float(quartet.get("v_code", 0.0)))
        ),
    }
    axis_shortest_wave_frequency = {
        axis: float(1.0 + float(config.bin_count) * (0.08 + 0.92 * wave_norm))
        for axis, wave_norm in axis_wave_norm.items()
    }
    axis_step_interval = {
        "F": clamp01(unison_step_interval * (0.72 + 0.28 * axis_wave_norm["F"])),
        "A": clamp01(unison_step_interval * (0.70 + 0.30 * axis_wave_norm["A"])),
        "I": clamp01(kernel_lateral_step * (0.66 + 0.34 * axis_wave_norm["I"])),
        "V": clamp01(kernel_lateral_step * (0.64 + 0.36 * axis_wave_norm["V"])),
    }
    score_metric = dict(deviation_ops.get("score", {}) or {})
    coherence_metric = dict(deviation_ops.get("coherence", {}) or {})
    curvature_metric = dict(deviation_ops.get("curvature", {}) or {})
    trap_metric = dict(deviation_ops.get("trap", {}) or {})
    pair_context = {
        "F_A": clamp01(0.46 * amplitude_gradient_norm + 0.34 * phase_lock_flux + 0.20 * amplitude_flux),
        "F_V": clamp01(0.40 * prediction_confidence + 0.34 * phase_lock_flux + 0.26 * coupling_norm),
        "F_I": clamp01(0.44 * coherence_flux + 0.34 * phase_gradient_norm + 0.22 * temporal_persistence),
        "A_V": clamp01(0.42 * amplitude_flux + 0.32 * prediction_confidence + 0.26 * density_norm),
        "A_I": clamp01(0.42 * amplitude_gradient_norm + 0.34 * coherence_flux + 0.24 * temporal_persistence),
        "I_V": clamp01(0.44 * coherence_flux + 0.32 * shared_flux + 0.24 * coupling_norm),
    }
    coupling_gradient_field: dict[str, float] = {}
    for lhs, rhs in (("F", "A"), ("F", "V"), ("F", "I"), ("A", "V"), ("A", "I"), ("I", "V")):
        pair_key = f"{lhs}_{rhs}"
        raw_cross = (
            abs(metric_cross_term(score_metric, lhs, rhs)) * 18.0
            + abs(metric_cross_term(coherence_metric, lhs, rhs)) * 14.0
            + abs(metric_cross_term(curvature_metric, lhs, rhs)) * 0.06
            + abs(metric_cross_term(trap_metric, lhs, rhs)) * 18.0
        )
        raw_norm = clamp01(math.tanh(raw_cross))
        wave_span = clamp01(
            0.55 * min(axis_wave_norm[lhs], axis_wave_norm[rhs])
            + 0.45 * max(axis_wave_norm[lhs], axis_wave_norm[rhs])
        )
        coupling_gradient_field[pair_key] = clamp01(
            0.38 * wave_span
            + 0.28 * raw_norm
            + 0.22 * float(pair_context.get(pair_key, 0.0))
            + 0.12 * larger_field_exposure
        )
    coupling_gradient_labels = {
        "frequency_amplitude": float(coupling_gradient_field.get("F_A", 0.0)),
        "frequency_voltage": float(coupling_gradient_field.get("F_V", 0.0)),
        "frequency_amperage": float(coupling_gradient_field.get("F_I", 0.0)),
        "amplitude_voltage": float(coupling_gradient_field.get("A_V", 0.0)),
        "amplitude_amperage": float(coupling_gradient_field.get("A_I", 0.0)),
        "voltage_amperage": float(coupling_gradient_field.get("I_V", 0.0)),
    }

    photonic_basins = [
        {
            "basin_id": "electron_basin",
            "field": "charge_field",
            "particle": "electron",
            "packet_affinity": "shared",
            "depth": float(clamp01(0.44 * charge_field + 0.22 * shared_fraction + 0.18 * dominant_freq_norm[0] + 0.16 * phase_lock_flux)),
            "occupancy": float(clamp01(0.56 * shared_fraction + 0.24 * phase_lock_flux + 0.20 * prediction_confidence)),
            "tensor_alignment": float(clamp01(0.48 * phase_gradient_norm + 0.28 * tensor_gradient_flux + 0.24 * charge_field)),
            "frequency_anchor": float(dominant_freq_norm[0]),
        },
        {
            "basin_id": "phonon_basin",
            "field": "lattice_field",
            "particle": "phonon",
            "packet_affinity": "shared",
            "depth": float(clamp01(0.42 * lattice_field + 0.24 * tensor_gradient_flux + 0.18 * density_norm + 0.16 * lattice_spacing_norm)),
            "occupancy": float(clamp01(0.36 * shared_fraction + 0.34 * tensor_gradient_flux + 0.30 * density_norm)),
            "tensor_alignment": float(clamp01(0.52 * tensor_norm + 0.24 * amplitude_gradient_norm + 0.24 * lattice_field)),
            "frequency_anchor": float(dominant_freq_norm[1]),
        },
        {
            "basin_id": "exciton_basin",
            "field": "coherence_field",
            "particle": "exciton",
            "packet_affinity": "shared",
            "depth": float(clamp01(0.44 * coherence_field + 0.22 * excitation_norm + 0.18 * temporal_persistence + 0.16 * larger_field_exposure)),
            "occupancy": float(clamp01(0.48 * coherence_flux + 0.28 * shared_fraction + 0.24 * temporal_persistence)),
            "tensor_alignment": float(clamp01(0.46 * phase_gradient_norm + 0.28 * oam_norm + 0.26 * coherence_field)),
            "frequency_anchor": float(dominant_freq_norm[2]),
        },
        {
            "basin_id": "hole_basin",
            "field": "vacancy_field",
            "particle": "hole",
            "packet_affinity": "individual",
            "depth": float(clamp01(0.42 * vacancy_field + 0.22 * individual_fraction + 0.18 * amplitude_flux + 0.18 * residual_norm)),
            "occupancy": float(clamp01(0.58 * individual_fraction + 0.22 * vacancy_field + 0.20 * temporal_inertia_norm)),
            "tensor_alignment": float(clamp01(0.42 * amplitude_gradient_norm + 0.30 * temporal_inertia_norm + 0.28 * vacancy_field)),
            "frequency_anchor": float(float(np.mean(dominant_freq_norm))),
        },
    ]
    dominant_basin = max(
        photonic_basins,
        key=lambda basin: float(basin.get("depth", 0.0)) * float(basin.get("occupancy", 0.0)),
    )

    calibration_vector = np.array(
        [
            charge_field - vacancy_field * 0.55 + (dominant_freq_norm[0] - 0.5) * 0.45 + (axis_wave_norm["F"] - 0.5) * 0.22,
            lattice_field - amplitude_flux * 0.20 + (dominant_freq_norm[1] - 0.5) * 0.35 + (axis_wave_norm["A"] - 0.5) * 0.22,
            coherence_field - vacancy_field * 0.18 + (dominant_freq_norm[2] - 0.5) * 0.35 + (axis_wave_norm["I"] - 0.5) * 0.22,
            temporal_persistence + larger_field_exposure * 0.35 - 0.5 + (axis_wave_norm["V"] - 0.5) * 0.22,
        ],
        dtype=np.float64,
    )
    calibration_vector = np.tanh(calibration_vector)

    environment_tensor = np.array(
        [
            [0.82 + 0.10 * charge_field, -0.08 + 0.18 * phase_lock_flux, 0.09 + 0.15 * coherence_field, 0.06 + 0.14 * larger_field_exposure],
            [-0.08 + 0.18 * phase_lock_flux, 0.80 + 0.10 * lattice_field, 0.07 + 0.16 * tensor_gradient_flux, 0.08 + 0.12 * temporal_inertia_norm],
            [0.09 + 0.15 * coherence_field, 0.07 + 0.16 * tensor_gradient_flux, 0.81 + 0.10 * coherence_field, 0.10 + 0.14 * temporal_persistence],
            [0.06 + 0.14 * larger_field_exposure, 0.08 + 0.12 * temporal_inertia_norm, 0.10 + 0.14 * temporal_persistence, 0.83 + 0.10 * larger_field_exposure],
        ],
        dtype=np.float64,
    )
    environment_tensor[0, 1] += coupling_gradient_field.get("F_A", 0.0) * 0.12
    environment_tensor[0, 2] += coupling_gradient_field.get("F_I", 0.0) * 0.12
    environment_tensor[0, 3] += coupling_gradient_field.get("F_V", 0.0) * 0.12
    environment_tensor[1, 2] += coupling_gradient_field.get("A_I", 0.0) * 0.12
    environment_tensor[1, 3] += coupling_gradient_field.get("A_V", 0.0) * 0.12
    environment_tensor[2, 3] += coupling_gradient_field.get("I_V", 0.0) * 0.12
    environment_tensor = 0.5 * (environment_tensor + environment_tensor.T)

    field_pressure = clamp01(
        0.44 * float(np.mean([float(basin["depth"]) for basin in photonic_basins]))
        + 0.34 * larger_field_exposure
        + 0.22 * phase_lock_flux
    )
    basin_alignment = clamp01(
        0.58 * float(dominant_basin.get("depth", 0.0))
        + 0.42 * float(dominant_basin.get("occupancy", 0.0))
    )
    amplitude_guard = clamp01(
        amplitude_cap * (0.74 + 0.24 * field_pressure)
        + 0.10 * coherence_field
        + 0.06 * tensor_gradient_flux
        + 0.08 * float(np.mean(list(coupling_gradient_field.values()) or [0.0]))
    )
    temporal_gain = clamp01(
        0.42
        + 0.26 * coherence_field
        + 0.18 * temporal_persistence
        + 0.14 * larger_field_exposure
    )
    frequency_bias = clamp01(
        0.36 * baseline_frequency_norm
        + 0.24 * float(dominant_basin.get("frequency_anchor", 0.0))
        + 0.22 * charge_field
        + 0.18 * larger_field_exposure
        + 0.10 * axis_wave_norm["F"]
    )
    field_environment = {
        "charge_field": float(charge_field),
        "lattice_field": float(lattice_field),
        "coherence_field": float(coherence_field),
        "vacancy_field": float(vacancy_field),
    }
    wave_step_field = {
        "frequency_interval": float(last_frequency_step_interval),
        "amplitude_interval": float(last_amplitude_step_interval),
        "lateral_interval": float(kernel_lateral_step),
        "unison_interval": float(unison_step_interval),
        "cascade_interval": float(cascade_interval),
    }
    gpu_pulse_dof_basis = build_gpu_pulse_dof_basis(
        axis_wave_norm=axis_wave_norm,
        axis_step_interval=axis_step_interval,
        coupling_gradient_field=coupling_gradient_field,
        field_environment=field_environment,
        wave_step_field=wave_step_field,
        field_pressure=field_pressure,
        larger_field_exposure=larger_field_exposure,
    )
    return {
        "lattice_constant_m": float(nist.get("lattice_constant_m", 5.431020511e-10)),
        "density_g_cm3": float(nist.get("density_g_cm3", 2.33)),
        "mean_excitation_energy_ev": float(nist.get("mean_excitation_energy_ev", 173.0)),
        "lattice_spacing_norm": float(lattice_spacing_norm),
        "density_norm": float(density_norm),
        "excitation_norm": float(excitation_norm),
        "field_pressure": float(field_pressure),
        "larger_field_exposure": float(larger_field_exposure),
        "basin_alignment": float(basin_alignment),
        "amplitude_guard": float(amplitude_guard),
        "temporal_gain": float(temporal_gain),
        "frequency_bias": float(frequency_bias),
        "field_environment": field_environment,
        "wave_step_field": wave_step_field,
        "axis_shortest_wave_frequency": axis_shortest_wave_frequency,
        "axis_wave_norm": {axis: float(value) for axis, value in axis_wave_norm.items()},
        "axis_step_interval": axis_step_interval,
        "coupling_gradient_field": {key: float(value) for key, value in coupling_gradient_field.items()},
        "coupling_gradient_labels": coupling_gradient_labels,
        "gpu_pulse_dof_labels": list(gpu_pulse_dof_basis.get("gpu_pulse_dof_labels", [])),
        "gpu_pulse_dof_vector": list(gpu_pulse_dof_basis.get("gpu_pulse_dof_vector", [])),
        "gpu_pulse_dof_tensor": list(gpu_pulse_dof_basis.get("gpu_pulse_dof_tensor", [])),
        "gpu_pulse_projection_tensor": list(gpu_pulse_dof_basis.get("gpu_pulse_projection_tensor", [])),
        "gpu_pulse_axis_projection": list(gpu_pulse_dof_basis.get("gpu_pulse_axis_projection", [])),
        "gpu_pulse_axis_tensor": list(gpu_pulse_dof_basis.get("gpu_pulse_axis_tensor", [])),
        "tensor_metrics": {
            "phase_gradient_norm": float(phase_gradient_norm),
            "amplitude_gradient_norm": float(amplitude_gradient_norm),
            "tensor_norm": float(tensor_norm),
            "temporal_inertia_norm": float(temporal_inertia_norm),
            "oam_norm": float(oam_norm),
            "phase_lock_flux": float(phase_lock_flux),
            "shared_flux": float(shared_flux),
            "coherence_flux": float(coherence_flux),
            "amplitude_flux": float(amplitude_flux),
            "curvature_flux": float(curvature_flux),
            "tensor_gradient_flux": float(tensor_gradient_flux),
        },
        "dominant_frequency_norm": [float(value) for value in dominant_freq_norm],
        "photonic_basins": photonic_basins,
        "dominant_basin": dict(dominant_basin),
        "calibration_vector": [float(value) for value in calibration_vector],
        "environment_tensor": environment_tensor.tolist(),
        "summary": str(
            f"{dominant_basin.get('basin_id', 'unknown')} | field={field_pressure:.3f} | exposure={larger_field_exposure:.3f}"
        ),
    }


def build_fourier_kernel_fanout(
    config: SimulationConfig,
    step_dominant_freqs: np.ndarray,
    mean_phase_lock_matrix: np.ndarray,
    theta_history: np.ndarray,
    amplitude_history: np.ndarray,
    shared_history: np.ndarray,
    coherence_history: np.ndarray,
) -> dict[str, Any]:
    freq_snapshot = np.array(step_dominant_freqs[-1], dtype=np.float64)
    packet_count = int(freq_snapshot.shape[0]) if freq_snapshot.ndim == 2 else 0
    if packet_count <= 0:
        return {
            "fanout_vector": [0.0, 0.0, 0.0, 0.0],
            "activation_density": 0.0,
            "fanout_span": 0.0,
            "kernel_points": [],
            "activation_rows": [],
            "motif_clusters": [],
            "motif_signature": [0.0, 0.0, 0.0, 0.0],
            "motif_energy": 0.0,
            "motif_stability": 0.0,
            "motif_spread": 0.0,
            "activation_row_count": 0,
        }
    amplitude_norm = np.array(
        [
            clamp01(float(value) / max(float(config.max_amplitude), 1.0e-9))
            for value in np.array(amplitude_history[-1], dtype=np.float64)
        ],
        dtype=np.float64,
    )
    shared_norm = np.array(np.mean(shared_history, axis=0), dtype=np.float64)
    coherence_norm = np.array(np.mean(coherence_history, axis=0), dtype=np.float64)
    phase_norm = np.array(
        [clamp01(0.5 + 0.5 * math.sin(float(value))) for value in np.array(theta_history[-1], dtype=np.float64)],
        dtype=np.float64,
    )
    fanout_accum = np.zeros(4, dtype=np.float64)
    total_weight = 0.0
    kernel_points: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    motif_clusters: dict[tuple[int, ...], dict[str, Any]] = {}
    for packet_idx in range(packet_count):
        row = np.array(mean_phase_lock_matrix[packet_idx], dtype=np.float64).copy()
        if packet_idx < len(row):
            row[packet_idx] = 0.0
        positive = np.maximum(row, 0.0)
        if float(np.max(positive)) <= 1.0e-9:
            continue
        row_mean = float(np.mean(positive[positive > 0.0])) if np.any(positive > 0.0) else 0.0
        row_std = float(np.std(positive[positive > 0.0])) if np.any(positive > 0.0) else 0.0
        strength_threshold = max(0.12, row_mean + row_std * 0.35, float(np.max(positive)) * 0.62)
        strong_neighbors = [int(idx) for idx in np.where(positive >= strength_threshold)[0].tolist()]
        if len(strong_neighbors) < 4:
            strong_neighbors = [
                int(idx) for idx in np.argsort(positive)[-min(6, max(packet_count - 1, 1)) :].tolist()
                if float(positive[int(idx)]) > 0.0
            ]
        if not strong_neighbors:
            continue
        row_vector_accum = np.zeros(4, dtype=np.float64)
        row_total_weight = 0.0
        row_phase_delta: list[float] = []
        row_coherence_accum = 0.0
        row_neighbor_preview: list[dict[str, Any]] = []
        source_freq = np.array(freq_snapshot[packet_idx], dtype=np.float64)
        source_norm = np.array(
            [
                clamp01(abs(float(source_freq[0])) / max(float(config.bin_count), 1.0)),
                clamp01(abs(float(source_freq[1])) / max(float(config.bin_count), 1.0)),
                clamp01(abs(float(source_freq[2])) / max(float(config.bin_count), 1.0)),
            ],
            dtype=np.float64,
        )
        for neighbor in strong_neighbors:
            edge_strength = float(max(positive[int(neighbor)], 0.0))
            if edge_strength <= 0.0:
                continue
            target_freq = np.array(freq_snapshot[int(neighbor)], dtype=np.float64)
            target_norm = np.array(
                [
                    clamp01(abs(float(target_freq[0])) / max(float(config.bin_count), 1.0)),
                    clamp01(abs(float(target_freq[1])) / max(float(config.bin_count), 1.0)),
                    clamp01(abs(float(target_freq[2])) / max(float(config.bin_count), 1.0)),
                ],
                dtype=np.float64,
            )
            amplitude_pair = 0.5 * float(amplitude_norm[packet_idx]) + 0.5 * float(amplitude_norm[int(neighbor)])
            coherence_pair = 0.5 * float(coherence_norm[packet_idx]) + 0.5 * float(coherence_norm[int(neighbor)])
            shared_pair = 0.5 * float(shared_norm[packet_idx]) + 0.5 * float(shared_norm[int(neighbor)])
            phase_pair = 0.5 * float(phase_norm[packet_idx]) + 0.5 * float(phase_norm[int(neighbor)])
            phase_delta = abs(float(theta_history[-1, packet_idx]) - float(theta_history[-1, int(neighbor)]))
            phase_delta_norm = clamp01(phase_delta / math.pi)
            freq_delta_norm = clamp01(
                float(np.linalg.norm(target_freq - source_freq))
                / max(math.sqrt(3.0) * float(config.bin_count), 1.0e-9)
            )
            weight = clamp01(
                0.34 * edge_strength
                + 0.16 * amplitude_pair
                + 0.16 * coherence_pair
                + 0.14 * shared_pair
                + 0.12 * phase_pair
                + 0.08 * (1.0 - phase_delta_norm)
            )
            if weight <= 0.0:
                continue
            kernel_vector = np.array(
                [
                    clamp01(
                        0.26 * source_norm[0]
                        + 0.14 * target_norm[0]
                        + 0.16 * coherence_pair
                        + 0.14 * edge_strength
                        + 0.16 * amplitude_pair
                        + 0.14 * freq_delta_norm
                    ),
                    clamp01(
                        0.24 * source_norm[1]
                        + 0.14 * target_norm[1]
                        + 0.16 * shared_pair
                        + 0.16 * phase_pair
                        + 0.14 * amplitude_pair
                        + 0.16 * (1.0 - phase_delta_norm)
                    ),
                    clamp01(
                        0.20 * source_norm[2]
                        + 0.12 * target_norm[2]
                        + 0.16 * freq_delta_norm
                        + 0.18 * coherence_pair
                        + 0.16 * phase_pair
                        + 0.18 * shared_pair
                    ),
                    clamp01(
                        0.24 * phase_pair
                        + 0.18 * coherence_pair
                        + 0.18 * shared_pair
                        + 0.14 * edge_strength
                        + 0.12 * (1.0 - phase_delta_norm)
                        + 0.14 * freq_delta_norm
                    ),
                ],
                dtype=np.float64,
            )
            row_vector_accum += kernel_vector * weight
            row_total_weight += weight
            row_coherence_accum += coherence_pair * weight
            row_phase_delta.append(phase_delta_norm)
            kernel_points.append(
                {
                    "source": int(packet_idx),
                    "target": int(neighbor),
                    "weight": float(weight),
                    "vector": [float(value) for value in kernel_vector],
                    "phase_delta": float(phase_delta),
                    "phase_delta_norm": float(phase_delta_norm),
                    "edge_strength": float(edge_strength),
                }
            )
            row_neighbor_preview.append(
                {
                    "target": int(neighbor),
                    "weight": float(weight),
                    "edge_strength": float(edge_strength),
                }
            )
        if row_total_weight <= 1.0e-9:
            continue
        row_vector = row_vector_accum / row_total_weight
        row_activation = clamp01(row_total_weight / max(float(len(strong_neighbors)), 1.0))
        row_span = clamp01(
            float(np.std(row_vector)) * 2.8
            + (float(np.std(row_phase_delta)) if row_phase_delta else 0.0) * 0.55
        )
        row_coherence = clamp01(row_coherence_accum / row_total_weight)
        phase_partner = int(
            row_neighbor_preview[0]["target"]
            if row_neighbor_preview
            else packet_idx
        )
        row_neighbor_preview.sort(key=lambda item: (item["weight"], item["edge_strength"]), reverse=True)
        activation_rows.append(
            {
                "source": int(packet_idx),
                "row_vector": [float(value) for value in row_vector],
                "activation": float(row_activation),
                "span": float(row_span),
                "row_coherence": float(row_coherence),
                "neighbor_count": int(len(strong_neighbors)),
                "phase_partner": int(phase_partner),
                "neighbors": row_neighbor_preview[:8],
            }
        )
        fanout_accum += row_vector * row_activation
        total_weight += row_activation
        cluster_key = quantize_vector_key(row_vector, bucket_count=5)
        cluster = motif_clusters.setdefault(
            cluster_key,
            {
                "key": "-".join(str(part) for part in cluster_key),
                "vector_accum": np.zeros(4, dtype=np.float64),
                "activation_accum": 0.0,
                "stability_accum": 0.0,
                "coherence_accum": 0.0,
                "member_count": 0,
                "sources": [],
            },
        )
        cluster["vector_accum"] += row_vector * row_activation
        cluster["activation_accum"] += row_activation
        cluster["stability_accum"] += (1.0 - row_span) * row_activation
        cluster["coherence_accum"] += row_coherence * row_activation
        cluster["member_count"] += 1
        cluster["sources"].append(int(packet_idx))
    if total_weight > 1.0e-9:
        fanout_vector = fanout_accum / total_weight
    else:
        fanout_vector = np.zeros(4, dtype=np.float64)
    activation_rows.sort(
        key=lambda item: (
            item["activation"],
            item["row_coherence"],
            1.0 - item["span"],
        ),
        reverse=True,
    )
    motif_cluster_rows: list[dict[str, Any]] = []
    motif_accum = np.zeros(4, dtype=np.float64)
    motif_total_weight = 0.0
    motif_energy = 0.0
    motif_stability = 0.0
    for cluster in motif_clusters.values():
        activation_accum = float(cluster["activation_accum"])
        if activation_accum <= 1.0e-9:
            continue
        centroid = cluster["vector_accum"] / activation_accum
        stability = clamp01(float(cluster["stability_accum"]) / activation_accum)
        coherence = clamp01(float(cluster["coherence_accum"]) / activation_accum)
        cluster_weight = clamp01(
            0.54 * activation_accum / max(float(cluster["member_count"]), 1.0)
            + 0.26 * stability
            + 0.20 * coherence
        )
        motif_cluster_rows.append(
            {
                "cluster_id": str(cluster["key"]),
                "vector": [float(value) for value in centroid],
                "activation": float(cluster_weight),
                "stability": float(stability),
                "coherence": float(coherence),
                "member_count": int(cluster["member_count"]),
                "sources": list(cluster["sources"])[:8],
            }
        )
        motif_accum += centroid * cluster_weight
        motif_total_weight += cluster_weight
        motif_energy += cluster_weight * coherence
        motif_stability += cluster_weight * stability
    motif_cluster_rows.sort(
        key=lambda item: (
            item["activation"],
            item["stability"],
            item["coherence"],
        ),
        reverse=True,
    )
    motif_signature = motif_accum / motif_total_weight if motif_total_weight > 1.0e-9 else fanout_vector.copy()
    fanout_span = clamp01(
        float(np.std(fanout_vector)) * 1.8
        + (
            float(np.mean(np.std(np.array([row["row_vector"] for row in activation_rows], dtype=np.float64), axis=0)))
            if activation_rows
            else 0.0
        )
        * 1.8
    )
    activation_density = clamp01(
        float(np.mean([float(row["activation"]) for row in activation_rows])) if activation_rows else 0.0
    )
    motif_spread = clamp01(
        float(
            np.mean(
                np.std(
                    np.array([cluster["vector"] for cluster in motif_cluster_rows], dtype=np.float64),
                    axis=0,
                )
            )
        )
        * 2.4
        if motif_cluster_rows
        else 0.0
    )
    motif_energy = clamp01(motif_energy / max(motif_total_weight, 1.0e-9))
    motif_stability = clamp01(motif_stability / max(motif_total_weight, 1.0e-9))
    kernel_points.sort(
        key=lambda item: (
            item["weight"],
            item["edge_strength"],
            1.0 - item["phase_delta_norm"],
        ),
        reverse=True,
    )
    return {
        "fanout_vector": [float(value) for value in fanout_vector],
        "activation_density": float(activation_density),
        "fanout_span": float(fanout_span),
        "kernel_points": kernel_points[:64],
        "activation_rows": activation_rows[:32],
        "motif_clusters": motif_cluster_rows[:8],
        "motif_signature": [float(value) for value in motif_signature],
        "motif_energy": float(motif_energy),
        "motif_stability": float(motif_stability),
        "motif_spread": float(motif_spread),
        "activation_row_count": int(len(activation_rows)),
    }


def detect_interference_vector_field(
    config: SimulationConfig,
    step_dominant_freqs: np.ndarray,
    theta_history: np.ndarray,
    mean_phase_lock_matrix: np.ndarray,
    amplitude_history: np.ndarray,
    shared_history: np.ndarray,
    coherence_history: np.ndarray,
    packet_classes: list[dict[str, Any]],
    tensor_gradient_samples: list[dict[str, Any]],
    lattice_calibration: dict[str, Any],
    fourier_kernel_fanout: dict[str, Any],
    simulation_field_state: dict[str, Any],
    temporal_manifold: dict[str, Any],
    effective_vector: dict[str, Any],
    target_profile: dict[str, Any],
) -> dict[str, Any]:
    packet_class_index = {int(row.get("packet_id", -1)): row for row in packet_classes}
    tensor_gradient_index = {int(row.get("packet_id", -1)): row for row in tensor_gradient_samples}
    environment_tensor = np.array(
        lattice_calibration.get("environment_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if environment_tensor.shape != (4, 4):
        environment_tensor = np.zeros((4, 4), dtype=np.float64)
    calibration_vector = np.array(
        list(lattice_calibration.get("calibration_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if calibration_vector.shape[0] != 4:
        calibration_vector = np.zeros(4, dtype=np.float64)
    interval_windows = list(target_profile.get("interval_windows", []) or [0.5])
    phase_windows = list(target_profile.get("phase_windows", []) or interval_windows)
    network_algorithm = dict(target_profile.get("network_algorithm", {}) or {})
    btc_force_vector = np.array(
        list(network_algorithm.get("force_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if btc_force_vector.shape[0] != 4:
        btc_force_vector = np.zeros(4, dtype=np.float64)
    btc_force_tensor = np.array(
        network_algorithm.get("force_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if btc_force_tensor.shape != (4, 4):
        btc_force_tensor = np.zeros((4, 4), dtype=np.float64)
    btc_phase_turns = [
        wrap_turns(float(value))
        for value in list(network_algorithm.get("phase_turns", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])
    ]
    if len(btc_phase_turns) < 4:
        btc_phase_turns = [0.0, 0.0, 0.0, 0.0]
    btc_phase_pressure = clamp01(float(network_algorithm.get("phase_pressure", 0.0)))
    btc_amplitude_pressure = clamp01(float(network_algorithm.get("amplitude_pressure", 0.0)))
    btc_algorithm_bias = clamp01(float(network_algorithm.get("algorithm_bias", 0.0)))
    target_difficulty_window = clamp01(float(target_profile.get("difficulty_window", 0.5)))
    field_pressure = float(lattice_calibration.get("field_pressure", 0.0))
    larger_field_exposure = float(lattice_calibration.get("larger_field_exposure", 0.0))
    amplitude_guard = float(lattice_calibration.get("amplitude_guard", 0.5))
    field_resonance = float(temporal_manifold.get("field_resonance", 0.0))
    dominant_basin = dict(lattice_calibration.get("dominant_basin", {}) or {})
    dominant_affinity = str(dominant_basin.get("packet_affinity", "shared"))
    axis_wave_norm = dict(lattice_calibration.get("axis_wave_norm", {}) or {})
    axis_step_interval = dict(lattice_calibration.get("axis_step_interval", {}) or {})
    wave_step_field = dict(lattice_calibration.get("wave_step_field", {}) or {})
    coupling_gradient_field = dict(lattice_calibration.get("coupling_gradient_field", {}) or {})
    unison_interval = float(wave_step_field.get("unison_interval", 0.0))
    lateral_interval = float(wave_step_field.get("lateral_interval", 0.0))
    cascade_interval = float(wave_step_field.get("cascade_interval", 0.0))
    coupling_mean = float(np.mean(list(coupling_gradient_field.values()) or [0.0]))
    activation_rows = list(fourier_kernel_fanout.get("activation_rows", []) or [])
    activation_row_index = {
        int(row.get("source", -1)): row
        for row in activation_rows
    }
    motif_signature = np.array(
        list(
            fourier_kernel_fanout.get(
                "motif_signature",
                simulation_field_state.get("motif_signature", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if motif_signature.shape[0] != 4:
        motif_signature = np.zeros(4, dtype=np.float64)
    motif_energy = float(
        simulation_field_state.get(
            "motif_energy",
            fourier_kernel_fanout.get("motif_energy", 0.0),
        )
    )
    motif_consistency = float(simulation_field_state.get("motif_consistency", 0.0))
    motif_repeat_count = int(simulation_field_state.get("motif_repeat_count", 0))
    motif_spread = float(
        simulation_field_state.get(
            "motif_spread",
            fourier_kernel_fanout.get("motif_spread", 0.0),
        )
    )
    simulation_field_vector = np.array(
        list(simulation_field_state.get("simulation_field_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if simulation_field_vector.shape[0] != 4:
        simulation_field_vector = np.zeros(4, dtype=np.float64)
    trace_state = dict(simulation_field_state.get("substrate_trace_state", {}) or {})
    trace_vector = np.array(
        list(trace_state.get("trace_vector", simulation_field_vector.tolist()) or simulation_field_vector.tolist()),
        dtype=np.float64,
    )
    if trace_vector.shape[0] != 4:
        trace_vector = np.array(simulation_field_vector, dtype=np.float64)
    trace_axis_vector = np.array(
        list(trace_state.get("trace_axis_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if trace_axis_vector.shape[0] != 4:
        trace_axis_vector = np.zeros(4, dtype=np.float64)
    trace_resonance = clamp01(float(trace_state.get("trace_resonance", 0.0)))
    trace_alignment = clamp01(float(trace_state.get("trace_alignment", 0.0)))
    trace_support = clamp01(float(trace_state.get("trace_support", 0.0)))
    field_frequency_bias = float(simulation_field_state.get("field_frequency_bias", 0.0))
    field_amplitude_bias = float(simulation_field_state.get("field_amplitude_bias", 0.0))
    calibration_readiness = float(simulation_field_state.get("calibration_readiness", 0.0))
    effective_anchor = np.array(
        [
            float(effective_vector.get("x", 0.0)),
            float(effective_vector.get("y", 0.0)),
            float(effective_vector.get("z", 0.0)),
            float(effective_vector.get("t_eff", 0.0)),
        ],
        dtype=np.float64,
    )
    packet_vectors: list[dict[str, Any]] = []
    latent_vectors: list[np.ndarray] = []
    projected_vectors: list[np.ndarray] = []

    for packet_idx in range(step_dominant_freqs.shape[1]):
        freq_vector = np.array(step_dominant_freqs[-1, packet_idx], dtype=np.float64)
        freq_norm = np.array(
            [
                clamp01(abs(float(freq_vector[0])) / max(float(config.bin_count), 1.0)),
                clamp01(abs(float(freq_vector[1])) / max(float(config.bin_count), 1.0)),
                clamp01(abs(float(freq_vector[2])) / max(float(config.bin_count), 1.0)),
            ],
            dtype=np.float64,
        )
        phase_row = np.array(mean_phase_lock_matrix[packet_idx], dtype=np.float64).copy()
        if packet_idx < len(phase_row):
            phase_row[packet_idx] = 0.0
        crosstalk = clamp01(float(np.max(phase_row)) if phase_row.size else 0.0)
        if phase_row.size:
            neighbor_index = np.argsort(phase_row)[-4:]
            neighbor_weights = np.array(
                [float(max(phase_row[idx], 0.0)) for idx in neighbor_index],
                dtype=np.float64,
            )
            if float(np.sum(neighbor_weights)) > 1.0e-9:
                neighbor_freq = np.average(step_dominant_freqs[-1, neighbor_index], axis=0, weights=neighbor_weights)
            else:
                neighbor_freq = freq_vector
        else:
            neighbor_index = np.array([packet_idx], dtype=np.int64)
            neighbor_freq = freq_vector
        neighbor_norm = np.array(
            [
                clamp01(abs(float(neighbor_freq[0])) / max(float(config.bin_count), 1.0)),
                clamp01(abs(float(neighbor_freq[1])) / max(float(config.bin_count), 1.0)),
                clamp01(abs(float(neighbor_freq[2])) / max(float(config.bin_count), 1.0)),
            ],
            dtype=np.float64,
        )
        theta = float(theta_history[-1, packet_idx])
        coherence_score = float(np.mean(coherence_history[:, packet_idx]))
        shared_score = float(np.mean(shared_history[:, packet_idx]))
        amplitude_norm = clamp01(float(amplitude_history[-1, packet_idx]) / max(float(config.max_amplitude), 1.0e-9))
        activation_row = dict(activation_row_index.get(packet_idx, {}) or {})
        row_vector = np.array(
            list(activation_row.get("row_vector", motif_signature.tolist()) or motif_signature.tolist()),
            dtype=np.float64,
        )
        if row_vector.shape[0] != 4:
            row_vector = np.array(motif_signature, dtype=np.float64)
        row_activation = clamp01(float(activation_row.get("activation", 0.0)))
        row_span = clamp01(float(activation_row.get("span", motif_spread)))
        row_coherence = clamp01(float(activation_row.get("row_coherence", 0.0)))
        neighbor_count = int(activation_row.get("neighbor_count", len(neighbor_index)))
        phase_partner = int(
            activation_row.get(
                "phase_partner",
                int(neighbor_index[-1]) if len(neighbor_index) else int(packet_idx),
            )
        )
        motif_alignment = vector_similarity(row_vector, motif_signature)
        class_row = packet_class_index.get(packet_idx, {})
        tensor_row = tensor_gradient_index.get(packet_idx, {})
        phase_gradient = np.abs(
            np.array(tensor_row.get("phase_gradient", [0.0, 0.0, 0.0]), dtype=np.float64)
        )
        amplitude_gradient = np.abs(
            np.array(tensor_row.get("amplitude_gradient", [0.0, 0.0, 0.0]), dtype=np.float64)
        )
        phase_gradient_norm = np.array(
            [clamp01(float(value) / 0.25) for value in phase_gradient],
            dtype=np.float64,
        )
        amplitude_gradient_norm = np.array(
            [clamp01(float(value) / 0.08) for value in amplitude_gradient],
            dtype=np.float64,
        )
        target_interval = float(interval_windows[packet_idx % len(interval_windows)])
        target_phase = float(phase_windows[packet_idx % len(phase_windows)])
        class_match = 1.0 if str(class_row.get("classification", "shared")) == dominant_affinity else 0.72
        cascade_phase = (
            theta
            + packet_idx * cascade_interval * math.pi
            + float(neighbor_index[-1] if len(neighbor_index) else packet_idx) * lateral_interval * math.pi * 0.5
        )
        cascade_activation = clamp01(0.5 + 0.5 * math.sin(cascade_phase))
        latent_seed = np.array(
            [
                0.24 * freq_norm[0]
                + 0.14 * neighbor_norm[0]
                + 0.10 * phase_gradient_norm[0]
                + 0.10 * crosstalk
                + 0.10 * target_interval
                + 0.06 * float(axis_wave_norm.get("F", 0.0))
                + 0.06 * float(coupling_gradient_field.get("F_A", 0.0))
                + 0.08 * class_match
                + 0.07 * row_vector[0]
                + 0.05 * row_activation,
                0.22 * freq_norm[1]
                + 0.14 * neighbor_norm[1]
                + 0.12 * amplitude_gradient_norm[1]
                + 0.08 * crosstalk
                + 0.10 * field_pressure
                + 0.06 * float(axis_wave_norm.get("A", 0.0))
                + 0.06 * float(coupling_gradient_field.get("A_V", 0.0))
                + 0.08 * class_match
                + 0.07 * row_vector[1]
                + 0.07 * row_coherence,
                0.20 * freq_norm[2]
                + 0.14 * neighbor_norm[2]
                + 0.10 * phase_gradient_norm[2]
                + 0.10 * coherence_score
                + 0.10 * target_phase
                + 0.06 * float(axis_wave_norm.get("I", 0.0))
                + 0.06 * float(coupling_gradient_field.get("F_I", 0.0))
                + 0.06 * larger_field_exposure
                + 0.08 * row_vector[2]
                + 0.04 * motif_alignment
                + 0.06 * motif_energy,
                0.20 * coherence_score
                + 0.16 * shared_score
                + 0.12 * amplitude_norm
                + 0.08 * crosstalk
                + 0.10 * target_phase
                + 0.06 * float(axis_wave_norm.get("V", 0.0))
                + 0.06 * float(coupling_gradient_field.get("I_V", 0.0))
                + 0.08 * target_difficulty_window
                + 0.08 * row_vector[3]
                + 0.06 * row_activation
                + 0.04 * motif_consistency,
            ],
            dtype=np.float64,
        )
        latent_seed += btc_force_vector * (
            0.18 + 0.14 * btc_phase_pressure + 0.10 * btc_algorithm_bias
        )
        phase_drive = np.array(
            [
                math.cos(theta),
                math.sin(theta),
                math.cos(theta + target_phase * math.pi + btc_phase_turns[packet_idx % 4] * math.pi),
                math.sin(theta + target_interval * math.pi + btc_phase_turns[(packet_idx + 1) % 4] * math.pi) * 0.5 + 0.5,
            ],
            dtype=np.float64,
        )
        cascade_drive = np.array(
            [
                math.cos(cascade_phase),
                math.sin(cascade_phase),
                math.cos(cascade_phase + target_phase * math.pi),
                cascade_activation,
            ],
            dtype=np.float64,
        )
        transformed = (environment_tensor @ latent_seed) * (
            0.66 + 0.18 * field_pressure + 0.16 * larger_field_exposure
        )
        transformed += calibration_vector * (0.22 + 0.12 * class_match)
        transformed += simulation_field_vector * (0.18 + 0.16 * calibration_readiness)
        transformed += trace_vector * (0.12 + 0.14 * trace_support)
        transformed += trace_axis_vector * np.array([0.08, 0.08, 0.08, 0.10], dtype=np.float64) * (
            0.64 + 0.36 * trace_alignment
        )
        transformed += row_vector * np.array([0.16, 0.16, 0.16, 0.18], dtype=np.float64) * (
            0.72 + 0.28 * row_activation
        )
        transformed += motif_signature * np.array([0.10, 0.10, 0.10, 0.14], dtype=np.float64) * (
            0.70 + 0.30 * motif_consistency
        )
        transformed += effective_anchor * np.array([0.18, 0.18, 0.18, 0.22], dtype=np.float64)
        transformed += phase_drive * np.array([0.12, 0.12, 0.10, 0.08], dtype=np.float64)
        transformed += cascade_drive * np.array([0.14, 0.14, 0.12, 0.10], dtype=np.float64) * (0.80 + 0.20 * coupling_mean)
        transformed += btc_force_vector * np.array([0.16, 0.18, 0.16, 0.18], dtype=np.float64) * (
            0.70 + 0.18 * btc_phase_pressure + 0.12 * btc_amplitude_pressure
        )
        transformed += (btc_force_tensor @ row_vector) * (0.04 + 0.06 * btc_algorithm_bias)
        transformed += np.array(
            [
                float(axis_step_interval.get("F", 0.0)),
                float(axis_step_interval.get("A", 0.0)),
                float(axis_step_interval.get("I", 0.0)),
                float(axis_step_interval.get("V", 0.0)),
            ],
            dtype=np.float64,
        ) * 0.12
        transformed = clamp_vector_norm(
            transformed,
            max_norm=max(0.75, 2.45 * amplitude_guard),
        )
        projected = project_spatial_components(transformed)
        trace_vector_alignment = clamp01(vector_similarity(projected, trace_vector))
        vector_alignment = clamp01(
            0.68 * (1.0 - float(np.mean(np.abs(projected[:3] - effective_anchor[:3]))))
            + 0.32 * trace_vector_alignment
        )
        btc_force_alignment = clamp01(vector_similarity(projected, btc_force_vector))
        target_alignment = clamp01(
            1.0 - abs(float(projected[3]) - clamp01(0.34 + 0.66 * target_phase))
            + 0.12 * btc_force_alignment
            + 0.08 * btc_phase_pressure
            + 0.08 * trace_alignment
        )
        resonance = clamp01(
            0.22 * vector_alignment
            + 0.16 * target_alignment
            + 0.10 * crosstalk
            + 0.10 * coherence_score
            + 0.06 * shared_score
            + 0.08 * field_resonance
            + 0.08 * cascade_activation
            + 0.05 * field_frequency_bias
            + 0.05 * field_amplitude_bias
            + 0.06 * row_activation
            + 0.05 * row_coherence
            + 0.05 * motif_alignment
            + 0.04 * motif_consistency
            + 0.03 * clamp01(1.0 - row_span)
            + 0.03 * clamp01(float(neighbor_count) / 6.0)
            + 0.02 * clamp01(float(motif_repeat_count) / 3.0)
            + 0.06 * btc_force_alignment
            + 0.04 * btc_amplitude_pressure
            + 0.08 * trace_resonance
            + 0.06 * trace_vector_alignment
        )
        carrier_bias = int(
            round(
                (float(projected[0]) + float(row_vector[0]) + 2.0) * 104729.0
                + (float(projected[1]) + float(row_vector[1]) + 2.0) * 130363.0
                + (float(projected[2]) + float(motif_signature[2]) + 2.0) * 65535.0
                + (float(projected[3]) + float(row_vector[3])) * 32749.0
                + row_activation * 8191.0
                + btc_force_alignment * 65521.0
                + btc_phase_pressure * 32719.0
            )
        ) & 0xFFFFFFFF
        packet_vectors.append(
            {
                "packet_id": int(packet_idx),
                "vector": [float(value) for value in projected],
                "latent_vector": [float(value) for value in transformed],
                "vector_alignment": float(vector_alignment),
                "target_alignment": float(target_alignment),
                "resonance": float(resonance),
                "phase_signature": float(clamp01(0.5 + 0.5 * math.sin(theta))),
                "frequency_signature": [float(value) for value in freq_norm],
                "target_interval": float(target_interval),
                "crosstalk": float(crosstalk),
                "carrier_bias": int(carrier_bias),
                "cascade_phase": float(cascade_phase),
                "cascade_activation": float(cascade_activation),
                "phase_partner": int(phase_partner),
                "row_activation": float(row_activation),
                "row_span": float(row_span),
                "row_coherence": float(row_coherence),
                "neighbor_count": int(neighbor_count),
                "motif_alignment": float(motif_alignment),
                "trace_alignment": float(trace_vector_alignment),
                "btc_force_alignment": float(btc_force_alignment),
                "btc_phase_pressure": float(btc_phase_pressure),
                "btc_amplitude_pressure": float(btc_amplitude_pressure),
            }
        )
        latent_vectors.append(transformed)
        projected_vectors.append(projected)

    packet_vectors.sort(
        key=lambda item: (
            item["resonance"],
            item["target_alignment"],
            item.get("btc_force_alignment", 0.0),
            item["row_activation"],
            item["vector_alignment"],
            item["crosstalk"],
        ),
        reverse=True,
    )
    top_vectors = packet_vectors[: max(1, min(8, len(packet_vectors)))]
    if top_vectors:
        dominant_latent = np.mean(
            np.array([item.get("latent_vector", [0.0, 0.0, 0.0, 0.0]) for item in top_vectors], dtype=np.float64),
            axis=0,
        )
        dominant_projected = np.mean(
            np.array([item.get("vector", [0.0, 0.0, 0.0, 0.0]) for item in top_vectors], dtype=np.float64),
            axis=0,
        )
    else:
        dominant_latent = np.zeros(4, dtype=np.float64)
        dominant_projected = np.zeros(4, dtype=np.float64)
    dominant_vector = {
        "vector": [float(value) for value in dominant_projected],
        "latent_vector": [float(value) for value in dominant_latent],
        "resonance": float(np.mean([float(item.get("resonance", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "target_alignment": float(np.mean([float(item.get("target_alignment", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "row_activation": float(np.mean([float(item.get("row_activation", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "motif_alignment": float(np.mean([float(item.get("motif_alignment", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "trace_alignment": float(np.mean([float(item.get("trace_alignment", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "btc_force_alignment": float(np.mean([float(item.get("btc_force_alignment", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "btc_phase_pressure": float(np.mean([float(item.get("btc_phase_pressure", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
        "btc_amplitude_pressure": float(np.mean([float(item.get("btc_amplitude_pressure", 0.0)) for item in top_vectors]) if top_vectors else 0.0),
    }
    return {
        "packet_vectors": packet_vectors,
        "dominant_vector": dominant_vector,
        "field_resonance": float(dominant_vector.get("resonance", 0.0)),
        "vector_count": int(len(packet_vectors)),
        "top_vectors": top_vectors,
        "motif_signature": [float(value) for value in motif_signature],
        "motif_energy": float(motif_energy),
        "motif_consistency": float(motif_consistency),
    }


def build_kernel_temporal_control_surface(
    fourier_kernel_fanout: dict[str, Any],
    target_profile: dict[str, Any],
    temporal_sequence_accounting: dict[str, Any],
    pulse_index: int,
    previous_field_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_field_state = dict(previous_field_state or {})
    interval_windows = list(target_profile.get("interval_windows", []) or [0.5])
    phase_windows = list(target_profile.get("phase_windows", []) or interval_windows)
    activation_rows = list(fourier_kernel_fanout.get("activation_rows", []) or [])
    motif_clusters = list(fourier_kernel_fanout.get("motif_clusters", []) or [])
    motif_signature = np.array(
        list(fourier_kernel_fanout.get("motif_signature", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if motif_signature.shape[0] != 4:
        motif_signature = np.zeros(4, dtype=np.float64)
    gpu_pulse_feedback = dict(temporal_sequence_accounting.get("gpu_pulse_feedback", {}) or {})
    feedback_axis_vector = np.array(
        list(
            temporal_sequence_accounting.get(
                "feedback_axis_vector",
                gpu_pulse_feedback.get("feedback_axis_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if feedback_axis_vector.shape[0] != 4:
        feedback_axis_vector = np.zeros(4, dtype=np.float64)
    feedback_dof_vector = np.array(
        list(gpu_pulse_feedback.get("feedback_dof_vector", []) or [0.0] * len(GPU_PULSE_DOF_LABELS)),
        dtype=np.float64,
    )
    if feedback_dof_vector.shape[0] != len(GPU_PULSE_DOF_LABELS):
        feedback_dof_vector = np.zeros(len(GPU_PULSE_DOF_LABELS), dtype=np.float64)
    feedback_phase_anchor_turns = wrap_turns(float(gpu_pulse_feedback.get("phase_anchor_turns", 0.0)))
    feedback_phase_alignment = clamp01(float(gpu_pulse_feedback.get("phase_alignment", 0.0)))
    feedback_memory_proxy = clamp01(float(gpu_pulse_feedback.get("memory_proxy", 0.0)))
    feedback_flux_proxy = clamp01(float(gpu_pulse_feedback.get("flux_proxy", 0.0)))
    feedback_temporal_drive = clamp01(
        float(temporal_sequence_accounting.get("feedback_temporal_drive", gpu_pulse_feedback.get("temporal_drive", 0.0)))
    )
    feedback_delta_target_vector = np.array(
        list(temporal_sequence_accounting.get("feedback_delta_target_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if feedback_delta_target_vector.shape[0] != 4:
        feedback_delta_target_vector = np.zeros(4, dtype=np.float64)
    feedback_delta_phase_shift_turns = float(
        temporal_sequence_accounting.get("feedback_delta_phase_shift_turns", 0.0)
    )
    feedback_delta_phase_retention = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_phase_retention", 0.0))
    )
    feedback_delta_response_gate = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_response_gate", 0.0))
    )
    feedback_delta_response_energy = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_response_energy", 0.0))
    )
    feedback_delta_memory_retention = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_memory_retention", 0.0))
    )
    sequence_persistence_score = clamp01(
        float(temporal_sequence_accounting.get("sequence_persistence_score", 0.0))
    )
    temporal_index_overlap = clamp01(
        float(temporal_sequence_accounting.get("temporal_index_overlap", 0.0))
    )
    previous_controls = {
        int(item.get("kernel_id", -1)): dict(item)
        for item in list(previous_field_state.get("kernel_temporal_controls", []) or [])
        if int(item.get("kernel_id", -1)) >= 0
    }
    previous_ancilla_particles = {
        int(item.get("kernel_id", -1)): dict(item)
        for item in list(previous_field_state.get("kernel_ancilla_particles", []) or [])
        if int(item.get("kernel_id", -1)) >= 0
    }
    previous_delta_feedback = dict(previous_field_state.get("gpu_pulse_delta_feedback", {}) or {})
    previous_response_gate = clamp01(float(previous_delta_feedback.get("response_gate", 0.0)))
    previous_memory_retention = clamp01(float(previous_delta_feedback.get("memory_retention", 0.0)))
    previous_phase_shift_turns = float(previous_delta_feedback.get("phase_shift_turns", 0.0))
    previous_phase_retention = clamp01(float(previous_delta_feedback.get("phase_retention", 0.0)))
    previous_response_energy = clamp01(float(previous_delta_feedback.get("response_energy", 0.0)))
    previous_delta_target_vector = np.array(
        list(previous_delta_feedback.get("delta_target_vector", []) or feedback_delta_target_vector.tolist()),
        dtype=np.float64,
    )
    if previous_delta_target_vector.shape[0] != 4:
        previous_delta_target_vector = np.array(feedback_delta_target_vector, dtype=np.float64)
    previous_axis_delta_vector = np.array(
        list(previous_delta_feedback.get("axis_delta_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if previous_axis_delta_vector.shape[0] != 4:
        previous_axis_delta_vector = np.zeros(4, dtype=np.float64)
    previous_dof_delta_vector = np.array(
        list(previous_delta_feedback.get("dof_delta_vector", []) or [0.0] * len(GPU_PULSE_DOF_LABELS)),
        dtype=np.float64,
    )
    if previous_dof_delta_vector.shape[0] != len(GPU_PULSE_DOF_LABELS):
        previous_dof_delta_vector = np.zeros(len(GPU_PULSE_DOF_LABELS), dtype=np.float64)
    ancilla_commit_ratio_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_commit_ratio_previous", 0.0))
    )
    ancilla_convergence_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_convergence_previous", 0.0))
    )
    ancilla_flux_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_flux_previous", 0.0))
    )
    ancilla_phase_alignment_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_phase_alignment_previous", 0.0))
    )
    ancilla_current_norm_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_current_norm_previous", 0.0))
    )
    ancilla_tension_headroom_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_tension_headroom_previous", 0.0))
    )
    ancilla_gradient_headroom_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_gradient_headroom_previous", 0.0))
    )
    ancilla_temporal_persistence_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_temporal_persistence_previous", 0.0))
    )
    ancilla_activation_gate_previous = clamp01(
        float(temporal_sequence_accounting.get("ancilla_activation_gate_previous", 0.0))
    )
    ancilla_activation_baseline = clamp01(
        0.26 * float(previous_field_state.get("calibration_readiness", 0.0))
        + 0.18 * float(previous_field_state.get("target_gate", 0.0))
        + 0.16 * sequence_persistence_score
        + 0.14 * temporal_index_overlap
        + 0.10 * ancilla_commit_ratio_previous
        + 0.08 * ancilla_convergence_previous
        + 0.08 * ancilla_temporal_persistence_previous
        + 0.08 * ancilla_activation_gate_previous
        + 0.06 * float(previous_field_state.get("entry_trigger", False))
    )
    if not activation_rows and motif_clusters:
        for row_idx, cluster in enumerate(motif_clusters[: min(8, len(motif_clusters))]):
            activation_rows.append(
                {
                    "source": int(row_idx),
                    "row_vector": list(cluster.get("vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
                    "activation": float(cluster.get("activation", 0.0)),
                    "span": 1.0 - float(cluster.get("stability", 0.0)),
                    "row_coherence": float(cluster.get("coherence", 0.0)),
                    "neighbor_count": int(cluster.get("member_count", 0)),
                    "phase_partner": int(row_idx),
                    "neighbors": [],
                }
            )
    kernel_temporal_controls: list[dict[str, Any]] = []
    kernel_ancilla_particles: list[dict[str, Any]] = []
    control_vector_accum = np.zeros(4, dtype=np.float64)
    control_total_weight = 0.0
    kernel_balance_mean = 0.0
    harmonic_resonance_mean = 0.0
    retro_temporal_gain_mean = 0.0
    resonance_alignment_mean = 0.0
    resonance_bandwidth_mean = 0.0
    kernel_drive_mean = 0.0
    kernel_delta_gate_mean = 0.0
    kernel_delta_memory_mean = 0.0
    kernel_delta_flux_mean = 0.0
    kernel_delta_phase_alignment_mean = 0.0
    kernel_retro_target_mean = 0.0
    ancilla_commit_total = 0.0
    ancilla_commit_gate_mean = 0.0
    ancilla_convergence_mean = 0.0
    ancilla_flux_mean = 0.0
    ancilla_phase_alignment_mean = 0.0
    ancilla_current_norm_mean = 0.0
    ancilla_tension_headroom_mean = 0.0
    ancilla_gradient_headroom_mean = 0.0
    ancilla_temporal_persistence_mean = 0.0
    ancilla_activation_gate_mean = 0.0
    ancilla_phase_turns_mean = 0.0
    control_gate_total = 0.0
    control_count = min(max(len(activation_rows), 1), 32)
    bits_per_lane = max(1, 64 // len(GPU_PULSE_DOF_LABELS))
    lane_mask = (1 << bits_per_lane) - 1
    for row_idx, row in enumerate(activation_rows[:control_count]):
        kernel_id = int(row.get("source", row_idx))
        previous_control = dict(previous_controls.get(kernel_id, {}) or {})
        previous_ancilla = dict(previous_ancilla_particles.get(kernel_id, {}) or {})
        row_vector = np.array(list(row.get("row_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        if row_vector.shape[0] != 4:
            row_vector = np.zeros(4, dtype=np.float64)
        row_activation = clamp01(float(row.get("activation", 0.0)))
        row_span = clamp01(float(row.get("span", 1.0)))
        row_coherence = clamp01(float(row.get("row_coherence", 0.0)))
        previous_ancilla_current = clamp01(
            float(
                previous_ancilla.get(
                    "current_norm",
                    decode_q32_32(previous_ancilla.get("current_mA_q32_32", 0)),
                )
            )
        )
        previous_ancilla_delta = float(
            previous_ancilla.get(
                "delta_current_norm",
                decode_q32_32(previous_ancilla.get("delta_I_mA_q32_32", 0)),
            )
        )
        previous_ancilla_delta_prev = float(
            previous_ancilla.get(
                "delta_current_prev_norm",
                decode_q32_32(previous_ancilla.get("delta_I_prev_mA_q32_32", 0)),
            )
        )
        previous_ancilla_phase_turns = wrap_turns(
            float(
                previous_ancilla.get(
                    "phase_turns",
                    phase_u64_to_turns(previous_ancilla.get("phase_offset_u64", 0)),
                )
            )
        )
        previous_ancilla_convergence = clamp01(
            float(
                previous_ancilla.get(
                    "convergence",
                    decode_q32_32(previous_ancilla.get("convergence_metric_q32_32", 0)),
                )
            )
        )
        previous_ancilla_flux = clamp01(float(previous_ancilla.get("flux_norm", ancilla_flux_previous)))
        previous_ancilla_phase_alignment = clamp01(
            float(previous_ancilla.get("phase_alignment", ancilla_phase_alignment_previous))
        )
        previous_ancilla_temporal_persistence = clamp01(
            float(previous_ancilla.get("temporal_persistence", ancilla_temporal_persistence_previous))
        )
        previous_ancilla_commit_gate = clamp01(
            float(previous_ancilla.get("commit_gate", ancilla_commit_ratio_previous))
        )
        phase_partner = int(row.get("phase_partner", kernel_id))
        target_phase = float(phase_windows[(kernel_id + pulse_index) % len(phase_windows)])
        target_interval = float(interval_windows[(phase_partner + pulse_index) % len(interval_windows)])
        seed_material = (
            f"{pulse_index}:{kernel_id}:{phase_partner}:{feedback_phase_anchor_turns:.9f}:"
            f"{target_phase:.9f}:{target_interval:.9f}:{','.join(f'{float(v):.9f}' for v in row_vector)}"
        )
        seed_u64 = int.from_bytes(hashlib.sha256(seed_material.encode("ascii")).digest()[:8], byteorder="little", signed=False)
        harmonic_orders: list[int] = []
        harmonic_weights: list[float] = []
        weighted_projection = 0.0
        weight_sum = 0.0
        for dim_idx in range(len(GPU_PULSE_DOF_LABELS)):
            shift = (dim_idx * bits_per_lane) % 60
            lane_bits = (seed_u64 >> shift) & lane_mask
            base_signal = clamp01(
                float(feedback_dof_vector[dim_idx])
                + row_activation * 0.18
                + row_coherence * 0.14
                + sequence_persistence_score * 0.10
            )
            harmonic_order = 1 + int((int(lane_bits) + int(round(base_signal * 8.0))) % 9)
            harmonic_weight = 1.0 / float(harmonic_order)
            harmonic_orders.append(int(harmonic_order))
            harmonic_weights.append(float(harmonic_weight))
            weighted_projection += clamp01(float(feedback_dof_vector[dim_idx])) * harmonic_weight
            weight_sum += harmonic_weight
        harmonic_projection = clamp01(weighted_projection / max(weight_sum, 1.0e-9))
        delta_projection_weighted = 0.0
        delta_projection_weight_sum = 0.0
        for dim_idx, harmonic_weight in enumerate(harmonic_weights):
            delta_projection_weighted += (
                clamp01(abs(float(previous_dof_delta_vector[dim_idx])) * 8.0) * float(harmonic_weight)
            )
            delta_projection_weight_sum += float(harmonic_weight)
        kernel_delta_projection = clamp01(delta_projection_weighted / max(delta_projection_weight_sum, 1.0e-9))
        if float(np.linalg.norm(previous_axis_delta_vector)) > 1.0e-9:
            kernel_delta_axis_alignment = vector_similarity(row_vector, previous_axis_delta_vector)
        else:
            kernel_delta_axis_alignment = clamp01(0.54 * row_activation + 0.46 * row_coherence)
        order_sum = sum(harmonic_orders)
        denom = len(harmonic_orders) + (order_sum % (len(harmonic_orders) * len(harmonic_orders))) + 1
        resonance_center_turns = wrap_turns(
            0.45 * feedback_phase_anchor_turns
            + 0.30 * target_phase
            + 0.15 * clamp01(float(row_vector[0]))
            + 0.10 * row_activation
        )
        resonance_bandwidth = clamp01(float(len(harmonic_orders) + 1) / float(denom))
        resonance_alignment = turn_alignment(resonance_center_turns, target_phase)
        delta_phase_turns = wrap_turns(
            resonance_center_turns
            + previous_phase_shift_turns
            * (0.38 + 0.26 * kernel_delta_projection + 0.20 * kernel_delta_axis_alignment)
            + float(previous_delta_target_vector[1]) * 0.09
        )
        kernel_delta_phase_alignment = clamp01(
            0.56 * turn_alignment(delta_phase_turns, target_phase)
            + 0.44 * turn_alignment(delta_phase_turns, resonance_center_turns)
        )
        kernel_delta_memory = clamp01(
            0.24 * previous_memory_retention
            + 0.18 * float(previous_delta_target_vector[0])
            + 0.14 * float(previous_delta_target_vector[1])
            + 0.10 * kernel_delta_projection
            + 0.10 * kernel_delta_axis_alignment
            + 0.10 * feedback_memory_proxy
            + 0.08 * feedback_temporal_drive
            + 0.06 * row_activation
            + 0.04 * previous_ancilla_convergence
            + 0.04 * previous_ancilla_commit_gate
        )
        kernel_delta_flux = clamp01(
            0.20 * previous_response_energy
            + 0.18 * float(previous_delta_target_vector[2])
            + 0.16 * float(previous_delta_target_vector[3])
            + 0.12 * kernel_delta_projection
            + 0.10 * kernel_delta_axis_alignment
            + 0.10 * feedback_flux_proxy
            + 0.08 * target_interval
            + 0.06 * row_coherence
            + 0.04 * previous_ancilla_flux
            + 0.04 * previous_ancilla_current
        )
        kernel_delta_gate = clamp01(
            0.24 * previous_response_gate
            + 0.18 * previous_phase_retention
            + 0.16 * kernel_delta_phase_alignment
            + 0.12 * kernel_delta_memory
            + 0.10 * kernel_delta_flux
            + 0.10 * kernel_delta_axis_alignment
            + 0.10 * kernel_delta_projection
            + 0.04 * previous_ancilla_phase_alignment
            + 0.03 * previous_ancilla_commit_gate
        )
        retro_temporal_target = clamp01(
            0.24 * kernel_delta_gate
            + 0.20 * kernel_delta_memory
            + 0.18 * kernel_delta_phase_alignment
            + 0.14 * previous_phase_retention
            + 0.12 * feedback_temporal_drive
            + 0.12 * target_interval
            + 0.04 * previous_ancilla_temporal_persistence
            + 0.03 * previous_ancilla_phase_alignment
        )
        previous_state = clamp01(float(previous_control.get("control_state", 0.0)))
        previous_retro = clamp01(float(previous_control.get("retro_gain", 0.0)))
        previous_harmonic_resonance = clamp01(float(previous_control.get("harmonic_resonance", 0.0)))
        retro_gain = clamp01(
            0.18 * previous_retro
            + 0.14 * previous_state
            + 0.12 * previous_response_gate
            + 0.10 * previous_memory_retention
            + 0.10 * feedback_temporal_drive
            + 0.08 * row_activation
            + 0.08 * (1.0 - row_span)
            + 0.10 * kernel_delta_gate
            + 0.10 * kernel_delta_memory
            + 0.04 * previous_ancilla_commit_gate
            + 0.03 * previous_ancilla_convergence
        )
        harmonic_resonance = clamp01(
            0.20 * harmonic_projection
            + 0.16 * resonance_alignment
            + 0.12 * row_activation
            + 0.10 * row_coherence
            + 0.10 * feedback_phase_alignment
            + 0.06 * target_interval
            + 0.06 * previous_harmonic_resonance
            + 0.10 * kernel_delta_projection
            + 0.06 * kernel_delta_phase_alignment
            + 0.04 * kernel_delta_flux
            + 0.04 * previous_ancilla_phase_alignment
            + 0.03 * previous_ancilla_flux
        )
        balance_target = clamp01(
            0.18 * row_activation
            + 0.14 * row_coherence
            + 0.14 * resonance_alignment
            + 0.12 * retro_gain
            + 0.10 * feedback_memory_proxy
            + 0.08 * (1.0 - row_span)
            + 0.06 * harmonic_projection
            + 0.10 * kernel_delta_memory
            + 0.05 * kernel_delta_flux
            + 0.03 * kernel_delta_phase_alignment
            + 0.04 * previous_ancilla_convergence
            + 0.03 * previous_ancilla_current
        )
        control_state = clamp01(0.28 * previous_state + 0.44 * balance_target + 0.28 * kernel_delta_gate)
        ancilla_phase_target = wrap_turns(
            0.34 * delta_phase_turns
            + 0.26 * resonance_center_turns
            + 0.18 * target_phase
            + 0.12 * feedback_phase_anchor_turns
            + 0.10 * kernel_delta_phase_alignment
        )
        ancilla_phase_prime = wrap_turns(
            ancilla_phase_target
            + signed_turn_delta(delta_phase_turns, previous_ancilla_phase_turns)
            * (0.20 + 0.16 * kernel_delta_phase_alignment + 0.12 * feedback_delta_phase_retention)
            + feedback_delta_phase_shift_turns * (0.08 + 0.12 * feedback_delta_phase_retention)
            + (harmonic_projection - 0.5) * (0.04 + 0.06 * harmonic_resonance)
        )
        ancilla_phase_delta = signed_turn_delta(ancilla_phase_prime, previous_ancilla_phase_turns)
        ancilla_phase_alignment = clamp01(
            0.46 * turn_alignment(ancilla_phase_prime, target_phase)
            + 0.28 * turn_alignment(ancilla_phase_prime, resonance_center_turns)
            + 0.26 * turn_alignment(ancilla_phase_prime, delta_phase_turns)
        )
        ancilla_activation_gate = clamp01(
            0.28 * ancilla_activation_baseline
            + 0.18 * row_activation
            + 0.14 * row_coherence
            + 0.12 * kernel_delta_gate
            + 0.10 * kernel_delta_phase_alignment
            + 0.08 * feedback_temporal_drive
            + 0.06 * previous_ancilla_temporal_persistence
            + 0.04 * previous_ancilla_commit_gate
        )
        ancilla_current_target = clamp01(
            0.18 * control_state
            + 0.16 * kernel_delta_flux
            + 0.14 * kernel_delta_memory
            + 0.12 * kernel_delta_gate
            + 0.10 * harmonic_resonance
            + 0.10 * retro_gain
            + 0.08 * feedback_flux_proxy
            + 0.06 * target_interval
            + 0.06 * row_activation
            + 0.06 * previous_ancilla_current
        )
        ancilla_delta_current_target = (
            ancilla_current_target - previous_ancilla_current
        ) * (
            0.24
            + 0.22 * ancilla_activation_gate
            + 0.18 * kernel_delta_gate
            + 0.14 * kernel_delta_flux
            + 0.12 * kernel_delta_phase_alignment
            + 0.10 * resonance_alignment
        )
        ancilla_current_limit = clamp01(
            0.16
            + 0.28 * control_state
            + 0.16 * retro_gain
            + 0.12 * harmonic_resonance
            + 0.10 * target_interval
            + 0.10 * feedback_memory_proxy
            + 0.08 * ancilla_activation_gate
        )
        ancilla_tension_headroom = clamp01(
            1.0 - abs(ancilla_current_target - ancilla_current_limit) / max(ancilla_current_limit + 0.20, 0.20)
        )
        ancilla_gradient_window = (
            0.08
            + 0.12 * resonance_bandwidth
            + 0.10 * kernel_delta_phase_alignment
            + 0.08 * sequence_persistence_score
            + 0.08 * temporal_index_overlap
        )
        ancilla_gradient_headroom = clamp01(
            1.0 - abs(ancilla_delta_current_target - previous_ancilla_delta) / max(ancilla_gradient_window, 1.0e-6)
        )
        ancilla_flux_norm = clamp01(
            0.22 * kernel_delta_flux
            + 0.18 * feedback_flux_proxy
            + 0.14 * clamp01(abs(ancilla_delta_current_target) * 4.0)
            + 0.12 * target_interval
            + 0.10 * row_coherence
            + 0.10 * previous_ancilla_flux
            + 0.08 * harmonic_projection
            + 0.06 * feedback_temporal_drive
        )
        ancilla_convergence_candidate = clamp01(
            0.22 * ancilla_phase_alignment
            + 0.18 * ancilla_tension_headroom
            + 0.16 * ancilla_gradient_headroom
            + 0.12 * kernel_delta_memory
            + 0.10 * kernel_delta_gate
            + 0.08 * harmonic_resonance
            + 0.06 * retro_gain
            + 0.08 * previous_ancilla_convergence
        )
        ancilla_commit_gate = clamp01(
            0.24 * ancilla_convergence_candidate
            + 0.20 * ancilla_tension_headroom
            + 0.18 * ancilla_gradient_headroom
            + 0.14 * ancilla_activation_gate
            + 0.10 * row_activation
            + 0.08 * kernel_delta_phase_alignment
            + 0.06 * row_coherence
        )
        ancilla_commit = bool(ancilla_commit_gate >= 0.58 and ancilla_activation_gate >= 0.42)
        ancilla_delta_current = ancilla_delta_current_target * (1.0 if ancilla_commit else 0.28)
        ancilla_current = clamp01(previous_ancilla_current + ancilla_delta_current)
        ancilla_phase_turns = wrap_turns(
            previous_ancilla_phase_turns + ancilla_phase_delta * (1.0 if ancilla_commit else 0.35)
        )
        ancilla_convergence = clamp01(
            previous_ancilla_convergence * (0.48 if ancilla_commit else 0.72)
            + ancilla_convergence_candidate * (0.52 if ancilla_commit else 0.28)
        )
        ancilla_temporal_persistence = clamp01(
            0.24 * ancilla_convergence
            + 0.20 * ancilla_commit_gate
            + 0.18 * ancilla_phase_alignment
            + 0.14 * ancilla_activation_gate
            + 0.12 * sequence_persistence_score
            + 0.12 * previous_ancilla_temporal_persistence
        )
        kernel_delta_memory = clamp01(
            0.82 * kernel_delta_memory
            + 0.10 * ancilla_convergence
            + 0.08 * ancilla_temporal_persistence
        )
        kernel_delta_flux = clamp01(
            0.80 * kernel_delta_flux
            + 0.12 * ancilla_flux_norm
            + 0.08 * ancilla_current
        )
        kernel_delta_phase_alignment = clamp01(
            0.80 * kernel_delta_phase_alignment
            + 0.12 * ancilla_phase_alignment
            + 0.08 * ancilla_temporal_persistence
        )
        kernel_delta_gate = clamp01(
            0.78 * kernel_delta_gate
            + 0.12 * ancilla_commit_gate
            + 0.10 * ancilla_activation_gate
        )
        retro_temporal_target = clamp01(
            0.78 * retro_temporal_target
            + 0.12 * ancilla_temporal_persistence
            + 0.10 * ancilla_phase_alignment
        )
        retro_gain = clamp01(
            0.80 * retro_gain
            + 0.12 * ancilla_commit_gate
            + 0.08 * ancilla_temporal_persistence
        )
        harmonic_resonance = clamp01(
            0.80 * harmonic_resonance
            + 0.10 * ancilla_phase_alignment
            + 0.10 * ancilla_flux_norm
        )
        balance_target = clamp01(
            0.76 * balance_target
            + 0.10 * ancilla_convergence
            + 0.08 * ancilla_tension_headroom
            + 0.06 * ancilla_activation_gate
        )
        control_state = clamp01(
            0.24 * previous_state
            + 0.42 * balance_target
            + 0.20 * kernel_delta_gate
            + 0.14 * ancilla_commit_gate
        )
        temporal_phase_delta = signed_turn_delta(
            resonance_center_turns,
            float(previous_control.get("resonance_center_turns", resonance_center_turns)),
        )
        kernel_drive = clamp01(
            0.20 * control_state
            + 0.16 * retro_gain
            + 0.14 * harmonic_resonance
            + 0.12 * feedback_temporal_drive
            + 0.10 * feedback_flux_proxy
            + 0.08 * (1.0 - row_span)
            + 0.08 * resonance_alignment
            + 0.07 * kernel_delta_flux
            + 0.05 * kernel_delta_gate
        )
        control_vector = clamp_vector_norm(
            row_vector * (0.62 + 0.18 * control_state)
            + motif_signature * (0.24 + 0.14 * harmonic_resonance)
            + feedback_axis_vector * (0.22 + 0.16 * retro_gain)
            + previous_axis_delta_vector * (0.12 + 0.10 * kernel_delta_gate),
            max_norm=2.75,
        )
        ancilla_phase_vector = np.array(
            [
                math.cos(ancilla_phase_turns * math.tau),
                math.sin(ancilla_phase_turns * math.tau),
                math.cos((ancilla_phase_turns + target_phase) * math.tau),
                math.sin((ancilla_phase_turns + target_interval) * math.tau),
            ],
            dtype=np.float64,
        )
        control_vector = clamp_vector_norm(
            control_vector
            + ancilla_phase_vector * (0.08 + 0.10 * ancilla_commit_gate)
            + row_vector * (0.06 + 0.08 * ancilla_current)
            + feedback_axis_vector * (0.04 + 0.06 * ancilla_temporal_persistence),
            max_norm=2.95,
        )
        control_weight = clamp01(
            0.36 * control_state
            + 0.22 * harmonic_resonance
            + 0.16 * retro_gain
            + 0.14 * row_activation
            + 0.12 * kernel_delta_gate
            + 0.08 * ancilla_commit_gate
        )
        kernel_ancilla_particles.append(
            {
                "kernel_id": int(kernel_id),
                "current_mA_q32_32": int(encode_q32_32(ancilla_current)),
                "delta_I_mA_q32_32": int(encode_q32_32(ancilla_delta_current)),
                "delta_I_prev_mA_q32_32": int(encode_q32_32(previous_ancilla_delta)),
                "phase_offset_u64": int(phase_turns_to_u64(ancilla_phase_turns)),
                "convergence_metric_q32_32": int(encode_q32_32(ancilla_convergence)),
                "commit": bool(ancilla_commit),
                "commit_gate": float(ancilla_commit_gate),
                "activation_gate": float(ancilla_activation_gate),
                "current_norm": float(ancilla_current),
                "delta_current_norm": float(ancilla_delta_current),
                "delta_current_prev_norm": float(previous_ancilla_delta),
                "phase_turns": float(ancilla_phase_turns),
                "phase_delta_turns": float(ancilla_phase_delta),
                "phase_alignment": float(ancilla_phase_alignment),
                "flux_norm": float(ancilla_flux_norm),
                "convergence": float(ancilla_convergence),
                "tension_headroom": float(ancilla_tension_headroom),
                "gradient_headroom": float(ancilla_gradient_headroom),
                "temporal_persistence": float(ancilla_temporal_persistence),
            }
        )
        kernel_temporal_controls.append(
            {
                "kernel_id": int(kernel_id),
                "phase_partner": int(phase_partner),
                "target_phase_window": float(target_phase),
                "target_interval_window": float(target_interval),
                "control_state": float(control_state),
                "balance_target": float(balance_target),
                "retro_gain": float(retro_gain),
                "delta_projection": float(kernel_delta_projection),
                "delta_axis_alignment": float(kernel_delta_axis_alignment),
                "delta_gate": float(kernel_delta_gate),
                "delta_memory_target": float(kernel_delta_memory),
                "delta_flux_target": float(kernel_delta_flux),
                "delta_phase_alignment": float(kernel_delta_phase_alignment),
                "delta_phase_turns": float(delta_phase_turns),
                "retro_temporal_target": float(retro_temporal_target),
                "harmonic_resonance": float(harmonic_resonance),
                "harmonic_projection": float(harmonic_projection),
                "harmonic_orders": harmonic_orders,
                "harmonic_weights": harmonic_weights,
                "resonance_center_turns": float(resonance_center_turns),
                "resonance_bandwidth": float(resonance_bandwidth),
                "resonance_alignment": float(resonance_alignment),
                "temporal_phase_delta": float(temporal_phase_delta),
                "kernel_drive": float(kernel_drive),
                "activation": float(row_activation),
                "row_coherence": float(row_coherence),
                "row_span": float(row_span),
                "neighbor_count": int(row.get("neighbor_count", 0)),
                "ancilla_commit": bool(ancilla_commit),
                "ancilla_commit_gate": float(ancilla_commit_gate),
                "ancilla_activation_gate": float(ancilla_activation_gate),
                "ancilla_current_norm": float(ancilla_current),
                "ancilla_flux_norm": float(ancilla_flux_norm),
                "ancilla_phase_alignment": float(ancilla_phase_alignment),
                "ancilla_convergence": float(ancilla_convergence),
                "ancilla_temporal_persistence": float(ancilla_temporal_persistence),
                "ancilla_tension_headroom": float(ancilla_tension_headroom),
                "ancilla_gradient_headroom": float(ancilla_gradient_headroom),
                "ancilla_phase_turns": float(ancilla_phase_turns),
                "control_vector": [float(value) for value in control_vector],
            }
        )
        control_vector_accum += control_vector * control_weight
        control_total_weight += control_weight
        kernel_balance_mean += control_state
        harmonic_resonance_mean += harmonic_resonance
        retro_temporal_gain_mean += retro_gain
        resonance_alignment_mean += resonance_alignment
        resonance_bandwidth_mean += resonance_bandwidth
        kernel_drive_mean += kernel_drive
        kernel_delta_gate_mean += kernel_delta_gate
        kernel_delta_memory_mean += kernel_delta_memory
        kernel_delta_flux_mean += kernel_delta_flux
        kernel_delta_phase_alignment_mean += kernel_delta_phase_alignment
        kernel_retro_target_mean += retro_temporal_target
        ancilla_commit_total += 1.0 if ancilla_commit else 0.0
        ancilla_commit_gate_mean += ancilla_commit_gate
        ancilla_convergence_mean += ancilla_convergence
        ancilla_flux_mean += ancilla_flux_norm
        ancilla_phase_alignment_mean += ancilla_phase_alignment
        ancilla_current_norm_mean += ancilla_current
        ancilla_tension_headroom_mean += ancilla_tension_headroom
        ancilla_gradient_headroom_mean += ancilla_gradient_headroom
        ancilla_temporal_persistence_mean += ancilla_temporal_persistence
        ancilla_activation_gate_mean += ancilla_activation_gate
        ancilla_phase_turns_mean += ancilla_phase_turns
        control_gate_total += clamp01(
            0.38 * control_state
            + 0.24 * harmonic_resonance
            + 0.18 * retro_gain
            + 0.20 * kernel_delta_gate
        )
    control_count = max(len(kernel_temporal_controls), 1)
    if control_total_weight > 1.0e-9:
        control_signature = control_vector_accum / control_total_weight
    else:
        control_signature = np.zeros(4, dtype=np.float64)
    kernel_ancilla_summary = {
        "commit_ratio": float(ancilla_commit_total / float(control_count)),
        "commit_gate_mean": float(ancilla_commit_gate_mean / float(control_count)),
        "convergence_mean": float(ancilla_convergence_mean / float(control_count)),
        "flux_mean": float(ancilla_flux_mean / float(control_count)),
        "phase_alignment_mean": float(ancilla_phase_alignment_mean / float(control_count)),
        "current_norm_mean": float(ancilla_current_norm_mean / float(control_count)),
        "tension_headroom_mean": float(ancilla_tension_headroom_mean / float(control_count)),
        "gradient_headroom_mean": float(ancilla_gradient_headroom_mean / float(control_count)),
        "temporal_persistence_mean": float(ancilla_temporal_persistence_mean / float(control_count)),
        "activation_gate_mean": float(ancilla_activation_gate_mean / float(control_count)),
        "phase_turns_mean": float(ancilla_phase_turns_mean / float(control_count)),
    }
    return {
        "kernel_temporal_controls": kernel_temporal_controls,
        "kernel_ancilla_particles": kernel_ancilla_particles,
        "kernel_ancilla_summary": kernel_ancilla_summary,
        "kernel_balance_mean": float(kernel_balance_mean / float(control_count)),
        "kernel_harmonic_resonance_mean": float(harmonic_resonance_mean / float(control_count)),
        "kernel_retro_gain_mean": float(retro_temporal_gain_mean / float(control_count)),
        "kernel_resonance_alignment_mean": float(resonance_alignment_mean / float(control_count)),
        "kernel_resonance_bandwidth_mean": float(resonance_bandwidth_mean / float(control_count)),
        "kernel_drive_mean": float(kernel_drive_mean / float(control_count)),
        "kernel_control_gate": float(control_gate_total / float(control_count)),
        "kernel_delta_gate_mean": float(kernel_delta_gate_mean / float(control_count)),
        "kernel_delta_memory_mean": float(kernel_delta_memory_mean / float(control_count)),
        "kernel_delta_flux_mean": float(kernel_delta_flux_mean / float(control_count)),
        "kernel_delta_phase_alignment_mean": float(
            kernel_delta_phase_alignment_mean / float(control_count)
        ),
        "kernel_retro_target_mean": float(kernel_retro_target_mean / float(control_count)),
        "kernel_control_signature": [float(value) for value in control_signature],
        "kernel_control_count": int(len(kernel_temporal_controls)),
    }


def build_simulation_field_entry_state(
    lattice_calibration: dict[str, Any],
    fourier_kernel_fanout: dict[str, Any],
    target_profile: dict[str, Any],
    pulse_sweep: dict[str, Any],
    pulse_index: int,
    previous_field_state: dict[str, Any] | None = None,
    encoded_event_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_field_state = dict(previous_field_state or {})
    calibration_vector = np.array(
        list(lattice_calibration.get("calibration_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if calibration_vector.shape[0] != 4:
        calibration_vector = np.zeros(4, dtype=np.float64)
    environment_tensor = np.array(
        lattice_calibration.get("environment_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        dtype=np.float64,
    )
    if environment_tensor.shape != (4, 4):
        environment_tensor = np.zeros((4, 4), dtype=np.float64)
    field_environment = dict(lattice_calibration.get("field_environment", {}) or {})
    wave_step_field = dict(lattice_calibration.get("wave_step_field", {}) or {})
    axis_wave_norm = dict(lattice_calibration.get("axis_wave_norm", {}) or {})
    axis_step_interval = dict(lattice_calibration.get("axis_step_interval", {}) or {})
    coupling_gradient_labels = dict(lattice_calibration.get("coupling_gradient_labels", {}) or {})
    dominant_frequency_norm = list(lattice_calibration.get("dominant_frequency_norm", []) or [0.0, 0.0, 0.0])
    if len(dominant_frequency_norm) < 3:
        dominant_frequency_norm = [0.0, 0.0, 0.0]
    field_pressure = float(lattice_calibration.get("field_pressure", 0.0))
    larger_field_exposure = float(lattice_calibration.get("larger_field_exposure", 0.0))
    target_window = clamp01(float(target_profile.get("difficulty_window", 0.5)))
    interval_windows = list(target_profile.get("interval_windows", []) or [target_window])
    phase_windows = list(target_profile.get("phase_windows", []) or interval_windows)
    target_interval = float(interval_windows[pulse_index % len(interval_windows)])
    target_phase = float(phase_windows[pulse_index % len(phase_windows)])
    sweep_quality = clamp01(
        float(pulse_sweep.get("coherence", 0.0))
        - float(pulse_sweep.get("deviation", 0.0)) * 0.10
    )
    frequency_interval = float(wave_step_field.get("frequency_interval", 0.0))
    amplitude_interval = float(wave_step_field.get("amplitude_interval", 0.0))
    lateral_interval = float(wave_step_field.get("lateral_interval", 0.0))
    cascade_interval = float(wave_step_field.get("cascade_interval", 0.0))
    unison_interval = float(wave_step_field.get("unison_interval", 0.0))
    coupling_mean = float(np.mean(list(coupling_gradient_labels.values()) or [0.0]))
    fanout_vector = np.array(
        list(fourier_kernel_fanout.get("fanout_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if fanout_vector.shape[0] != 4:
        fanout_vector = np.zeros(4, dtype=np.float64)
    activation_density = float(fourier_kernel_fanout.get("activation_density", 0.0))
    fanout_span = float(fourier_kernel_fanout.get("fanout_span", 0.0))
    motif_signature = np.array(
        list(fourier_kernel_fanout.get("motif_signature", fanout_vector.tolist()) or fanout_vector.tolist()),
        dtype=np.float64,
    )
    if motif_signature.shape[0] != 4:
        motif_signature = np.array(fanout_vector, dtype=np.float64)
    motif_energy = clamp01(float(fourier_kernel_fanout.get("motif_energy", 0.0)))
    motif_stability = clamp01(float(fourier_kernel_fanout.get("motif_stability", 0.0)))
    motif_spread = clamp01(float(fourier_kernel_fanout.get("motif_spread", fanout_span)))
    activation_rows = list(fourier_kernel_fanout.get("activation_rows", []) or [])
    motif_clusters = list(fourier_kernel_fanout.get("motif_clusters", []) or [])
    activation_row_count = int(fourier_kernel_fanout.get("activation_row_count", len(activation_rows)))
    previous_vector = np.array(
        list(previous_field_state.get("simulation_field_vector", calibration_vector.tolist()) or calibration_vector.tolist()),
        dtype=np.float64,
    )
    if previous_vector.shape[0] != 4:
        previous_vector = np.array(calibration_vector, dtype=np.float64)
    previous_readiness = clamp01(float(previous_field_state.get("calibration_readiness", 0.0)))
    previous_motif_signature = np.array(
        list(previous_field_state.get("motif_signature", motif_signature.tolist()) or motif_signature.tolist()),
        dtype=np.float64,
    )
    if previous_motif_signature.shape[0] != 4:
        previous_motif_signature = np.array(motif_signature, dtype=np.float64)
    previous_motif_consistency = clamp01(float(previous_field_state.get("motif_consistency", 0.0)))
    previous_repeat_count = int(previous_field_state.get("motif_repeat_count", 0))
    motif_consistency = vector_similarity(motif_signature, previous_motif_signature)
    motif_cluster_support = clamp01(float(len(motif_clusters)) / 4.0)
    stable_rows = sum(
        1
        for row in activation_rows
        if clamp01(float(row.get("activation", 0.0))) >= 0.42
        and clamp01(float(row.get("span", 1.0))) <= 0.42
    )
    stable_row_density = clamp01(float(stable_rows) / max(float(max(activation_row_count, 1)), 1.0))
    if motif_consistency >= 0.82 and motif_energy >= 0.34 and stable_row_density >= 0.28:
        motif_repeat_count = previous_repeat_count + 1
    elif motif_consistency >= 0.74 and motif_energy >= 0.28 and stable_row_density >= 0.22:
        motif_repeat_count = max(1, previous_repeat_count)
    else:
        motif_repeat_count = 0
    motif_repeat_norm = clamp01(float(motif_repeat_count) / 3.0)
    temporal_sequence_accounting = build_temporal_sequence_accounting(
        target_profile=target_profile,
        pulse_index=pulse_index,
        lattice_calibration=lattice_calibration,
        pulse_sweep=pulse_sweep,
        previous_field_state=previous_field_state,
        encoded_event_model=encoded_event_model,
    )
    persistent_temporal_vector = np.array(
        list(temporal_sequence_accounting.get("persistent_temporal_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if persistent_temporal_vector.shape[0] != 4:
        persistent_temporal_vector = np.zeros(4, dtype=np.float64)
    projected_temporal_dof_vector = np.array(
        list(temporal_sequence_accounting.get("projected_temporal_dof_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if projected_temporal_dof_vector.shape[0] != 4:
        projected_temporal_dof_vector = np.zeros(4, dtype=np.float64)
    voltage_frequency_correlation_tensor = np.array(
        temporal_sequence_accounting.get(
            "voltage_frequency_correlation_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if voltage_frequency_correlation_tensor.shape != (4, 4):
        voltage_frequency_correlation_tensor = np.zeros((4, 4), dtype=np.float64)
    voltage_frequency_dof_axis_tensor = np.array(
        temporal_sequence_accounting.get(
            "voltage_frequency_dof_axis_tensor",
            np.zeros((4, 4), dtype=np.float64).tolist(),
        ),
        dtype=np.float64,
    )
    if voltage_frequency_dof_axis_tensor.shape != (4, 4):
        voltage_frequency_dof_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    btc_network_force_vector = np.array(
        list(
            temporal_sequence_accounting.get(
                "btc_network_force_vector",
                target_profile.get("network_algorithm", {}).get("force_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if btc_network_force_vector.shape[0] != 4:
        btc_network_force_vector = np.zeros(4, dtype=np.float64)
    btc_network_force_tensor = np.array(
        temporal_sequence_accounting.get(
            "btc_network_force_tensor",
            target_profile.get("network_algorithm", {}).get(
                "force_tensor",
                np.zeros((4, 4), dtype=np.float64).tolist(),
            ),
        ),
        dtype=np.float64,
    )
    if btc_network_force_tensor.shape != (4, 4):
        btc_network_force_tensor = np.zeros((4, 4), dtype=np.float64)
    btc_network_phase_turns = [
        wrap_turns(float(value))
        for value in list(
            temporal_sequence_accounting.get(
                "btc_network_phase_turns",
                target_profile.get("network_algorithm", {}).get("phase_turns", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        )
    ]
    if len(btc_network_phase_turns) < 4:
        btc_network_phase_turns = [0.0, 0.0, 0.0, 0.0]
    btc_network_phase_pressure = clamp01(
        float(
            temporal_sequence_accounting.get(
                "btc_network_phase_pressure",
                target_profile.get("network_algorithm", {}).get("phase_pressure", 0.0),
            )
        )
    )
    btc_network_amplitude_pressure = clamp01(
        float(
            temporal_sequence_accounting.get(
                "btc_network_amplitude_pressure",
                target_profile.get("network_algorithm", {}).get("amplitude_pressure", 0.0),
            )
        )
    )
    btc_network_algorithm_bias = clamp01(
        float(
            temporal_sequence_accounting.get(
                "btc_network_algorithm_bias",
                target_profile.get("network_algorithm", {}).get("algorithm_bias", 0.0),
            )
        )
    )
    gpu_pulse_feedback = dict(temporal_sequence_accounting.get("gpu_pulse_feedback", {}) or {})
    feedback_axis_vector = np.array(
        list(
            temporal_sequence_accounting.get(
                "feedback_axis_vector",
                gpu_pulse_feedback.get("feedback_axis_vector", [0.0, 0.0, 0.0, 0.0]),
            )
            or [0.0, 0.0, 0.0, 0.0]
        ),
        dtype=np.float64,
    )
    if feedback_axis_vector.shape[0] != 4:
        feedback_axis_vector = np.zeros(4, dtype=np.float64)
    feedback_axis_tensor = np.array(
        temporal_sequence_accounting.get(
            "feedback_axis_tensor",
            gpu_pulse_feedback.get("feedback_axis_tensor", np.zeros((4, 4), dtype=np.float64).tolist()),
        ),
        dtype=np.float64,
    )
    if feedback_axis_tensor.shape != (4, 4):
        feedback_axis_tensor = np.zeros((4, 4), dtype=np.float64)
    feedback_frequency_axis = clamp01(float(feedback_axis_vector[0]))
    feedback_amplitude_axis = clamp01(float(feedback_axis_vector[1]))
    feedback_current_axis = clamp01(float(feedback_axis_vector[2]))
    feedback_voltage_axis = clamp01(float(feedback_axis_vector[3]))
    feedback_phase_anchor_turns = wrap_turns(float(gpu_pulse_feedback.get("phase_anchor_turns", 0.0)))
    feedback_phase_alignment = clamp01(float(gpu_pulse_feedback.get("phase_alignment", 0.0)))
    feedback_memory_proxy = clamp01(float(gpu_pulse_feedback.get("memory_proxy", 0.0)))
    feedback_flux_proxy = clamp01(float(gpu_pulse_feedback.get("flux_proxy", 0.0)))
    feedback_stability_proxy = clamp01(float(gpu_pulse_feedback.get("stability_proxy", 0.0)))
    feedback_temporal_drive = clamp01(
        float(temporal_sequence_accounting.get("feedback_temporal_drive", gpu_pulse_feedback.get("temporal_drive", 0.0)))
    )
    feedback_delta_target_vector = np.array(
        list(temporal_sequence_accounting.get("feedback_delta_target_vector", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if feedback_delta_target_vector.shape[0] != 4:
        feedback_delta_target_vector = np.zeros(4, dtype=np.float64)
    feedback_delta_phase_shift_turns = float(
        temporal_sequence_accounting.get("feedback_delta_phase_shift_turns", 0.0)
    )
    feedback_delta_phase_retention = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_phase_retention", 0.0))
    )
    feedback_delta_response_gate = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_response_gate", 0.0))
    )
    feedback_delta_response_energy = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_response_energy", 0.0))
    )
    feedback_delta_memory_retention = clamp01(
        float(temporal_sequence_accounting.get("feedback_delta_memory_retention", 0.0))
    )
    kernel_control_surface = build_kernel_temporal_control_surface(
        fourier_kernel_fanout=fourier_kernel_fanout,
        target_profile=target_profile,
        temporal_sequence_accounting=temporal_sequence_accounting,
        pulse_index=pulse_index,
        previous_field_state=previous_field_state,
    )
    kernel_temporal_controls = list(kernel_control_surface.get("kernel_temporal_controls", []) or [])
    kernel_ancilla_particles = list(kernel_control_surface.get("kernel_ancilla_particles", []) or [])
    kernel_ancilla_summary = dict(kernel_control_surface.get("kernel_ancilla_summary", {}) or {})
    kernel_balance_mean = clamp01(float(kernel_control_surface.get("kernel_balance_mean", 0.0)))
    kernel_harmonic_resonance_mean = clamp01(
        float(kernel_control_surface.get("kernel_harmonic_resonance_mean", 0.0))
    )
    kernel_retro_gain_mean = clamp01(float(kernel_control_surface.get("kernel_retro_gain_mean", 0.0)))
    kernel_resonance_alignment_mean = clamp01(
        float(kernel_control_surface.get("kernel_resonance_alignment_mean", 0.0))
    )
    kernel_resonance_bandwidth_mean = clamp01(
        float(kernel_control_surface.get("kernel_resonance_bandwidth_mean", 0.0))
    )
    kernel_drive_mean = clamp01(float(kernel_control_surface.get("kernel_drive_mean", 0.0)))
    kernel_control_gate = clamp01(float(kernel_control_surface.get("kernel_control_gate", 0.0)))
    kernel_delta_gate_mean = clamp01(float(kernel_control_surface.get("kernel_delta_gate_mean", 0.0)))
    kernel_delta_memory_mean = clamp01(float(kernel_control_surface.get("kernel_delta_memory_mean", 0.0)))
    kernel_delta_flux_mean = clamp01(float(kernel_control_surface.get("kernel_delta_flux_mean", 0.0)))
    kernel_delta_phase_alignment_mean = clamp01(
        float(kernel_control_surface.get("kernel_delta_phase_alignment_mean", 0.0))
    )
    kernel_retro_target_mean = clamp01(float(kernel_control_surface.get("kernel_retro_target_mean", 0.0)))
    kernel_control_signature = np.array(
        list(kernel_control_surface.get("kernel_control_signature", []) or [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if kernel_control_signature.shape[0] != 4:
        kernel_control_signature = np.zeros(4, dtype=np.float64)
    ancilla_commit_ratio = clamp01(float(kernel_ancilla_summary.get("commit_ratio", 0.0)))
    ancilla_commit_gate = clamp01(float(kernel_ancilla_summary.get("commit_gate_mean", ancilla_commit_ratio)))
    ancilla_convergence = clamp01(float(kernel_ancilla_summary.get("convergence_mean", 0.0)))
    ancilla_flux = clamp01(float(kernel_ancilla_summary.get("flux_mean", 0.0)))
    ancilla_phase_alignment = clamp01(float(kernel_ancilla_summary.get("phase_alignment_mean", 0.0)))
    ancilla_current_norm = clamp01(float(kernel_ancilla_summary.get("current_norm_mean", 0.0)))
    ancilla_tension_headroom = clamp01(float(kernel_ancilla_summary.get("tension_headroom_mean", 0.0)))
    ancilla_gradient_headroom = clamp01(float(kernel_ancilla_summary.get("gradient_headroom_mean", 0.0)))
    ancilla_temporal_persistence = clamp01(
        float(kernel_ancilla_summary.get("temporal_persistence_mean", 0.0))
    )
    ancilla_activation_gate = clamp01(float(kernel_ancilla_summary.get("activation_gate_mean", 0.0)))
    ancilla_phase_turns_mean = wrap_turns(float(kernel_ancilla_summary.get("phase_turns_mean", 0.0)))
    if float(np.linalg.norm(feedback_delta_target_vector)) <= 1.0e-9:
        feedback_delta_target_vector = np.array(
            [
                kernel_delta_gate_mean,
                kernel_delta_memory_mean,
                kernel_delta_flux_mean,
                kernel_delta_phase_alignment_mean,
            ],
            dtype=np.float64,
        )
    else:
        feedback_delta_target_vector = np.array(
            [
                clamp01(0.72 * float(feedback_delta_target_vector[0]) + 0.28 * kernel_delta_gate_mean),
                clamp01(0.72 * float(feedback_delta_target_vector[1]) + 0.28 * kernel_delta_memory_mean),
                clamp01(0.72 * float(feedback_delta_target_vector[2]) + 0.28 * kernel_delta_flux_mean),
                clamp01(0.72 * float(feedback_delta_target_vector[3]) + 0.28 * kernel_delta_phase_alignment_mean),
            ],
            dtype=np.float64,
        )
    feedback_delta_phase_retention = clamp01(
        max(feedback_delta_phase_retention, 0.72 * kernel_delta_phase_alignment_mean)
    )
    feedback_delta_response_gate = clamp01(
        max(feedback_delta_response_gate, 0.68 * kernel_delta_gate_mean + 0.32 * kernel_retro_target_mean)
    )
    feedback_delta_response_energy = clamp01(
        max(feedback_delta_response_energy, 0.58 * kernel_delta_flux_mean + 0.42 * kernel_control_gate)
    )
    feedback_delta_memory_retention = clamp01(
        max(feedback_delta_memory_retention, 0.66 * kernel_delta_memory_mean + 0.34 * feedback_memory_proxy)
    )
    sequence_persistence_score = clamp01(
        float(temporal_sequence_accounting.get("sequence_persistence_score", 0.0))
    )
    temporal_index_overlap = clamp01(
        float(temporal_sequence_accounting.get("temporal_index_overlap", 0.0))
    )
    voltage_frequency_flux = clamp01(
        float(temporal_sequence_accounting.get("voltage_frequency_flux", 0.0))
    )
    frequency_voltage_flux = clamp01(
        float(temporal_sequence_accounting.get("frequency_voltage_flux", 0.0))
    )
    interference_accounting = clamp01(
        0.16 * field_pressure
        + 0.12 * larger_field_exposure
        + 0.12 * coupling_mean
        + 0.10 * target_interval
        + 0.08 * target_phase
        + 0.08 * activation_density
        + 0.05 * fanout_span
        + 0.08 * unison_interval
        + 0.06 * lateral_interval
        + 0.07 * motif_energy
        + 0.04 * motif_stability
        + 0.04 * motif_consistency
        + 0.06 * sequence_persistence_score
        + 0.05 * voltage_frequency_flux
        + 0.04 * temporal_index_overlap
        + 0.04 * clamp01(float(np.mean(projected_temporal_dof_vector)))
        + 0.07 * feedback_memory_proxy
        + 0.06 * feedback_flux_proxy
        + 0.05 * feedback_phase_alignment
        + 0.04 * feedback_temporal_drive
        + 0.06 * kernel_balance_mean
        + 0.05 * kernel_harmonic_resonance_mean
        + 0.05 * kernel_retro_gain_mean
        + 0.04 * kernel_control_gate
        + 0.05 * feedback_delta_response_gate
        + 0.04 * feedback_delta_phase_retention
        + 0.05 * kernel_delta_gate_mean
        + 0.04 * kernel_delta_flux_mean
        + 0.06 * ancilla_commit_ratio
        + 0.05 * ancilla_flux
        + 0.04 * ancilla_phase_alignment
        + 0.04 * ancilla_activation_gate
        + 0.08 * btc_network_phase_pressure
        + 0.06 * btc_network_amplitude_pressure
        + 0.05 * btc_network_algorithm_bias
    )
    calibration_seed = np.array(
        [
            0.34 * float(axis_wave_norm.get("F", dominant_frequency_norm[0]))
            + 0.22 * float(dominant_frequency_norm[0])
            + 0.14 * float(coupling_gradient_labels.get("frequency_amplitude", 0.0))
            + 0.10 * target_interval
            + 0.10 * unison_interval,
            0.32 * float(axis_wave_norm.get("A", 0.0))
            + 0.22 * amplitude_interval
            + 0.14 * float(coupling_gradient_labels.get("amplitude_voltage", 0.0))
            + 0.12 * target_phase
            + 0.10 * unison_interval,
            0.30 * float(axis_wave_norm.get("I", 0.0))
            + 0.24 * lateral_interval
            + 0.14 * float(coupling_gradient_labels.get("frequency_amperage", 0.0))
            + 0.12 * target_interval
            + 0.10 * cascade_interval,
            0.28 * float(axis_wave_norm.get("V", 0.0))
            + 0.24 * float(axis_step_interval.get("V", 0.0))
            + 0.16 * float(coupling_gradient_labels.get("voltage_amperage", 0.0))
            + 0.12 * target_phase
            + 0.10 * cascade_interval,
        ],
        dtype=np.float64,
    )
    calibration_seed += np.array(
        [
            0.18 * feedback_frequency_axis + 0.10 * feedback_phase_alignment,
            0.18 * feedback_amplitude_axis + 0.10 * feedback_memory_proxy,
            0.18 * feedback_current_axis + 0.10 * feedback_flux_proxy,
            0.18 * feedback_voltage_axis + 0.10 * feedback_temporal_drive,
        ],
        dtype=np.float64,
    )
    calibration_seed += btc_network_force_vector * (
        0.18 + 0.16 * btc_network_phase_pressure + 0.12 * btc_network_algorithm_bias
    )
    cascade_phase = pulse_index * cascade_interval * math.pi
    cascade_drive = np.array(
        [
            math.cos(cascade_phase),
            math.sin(cascade_phase),
            math.cos(cascade_phase + target_phase * math.pi),
            0.5 + 0.5 * math.sin(cascade_phase + target_interval * math.pi),
        ],
        dtype=np.float64,
    )
    field_environment_vector = np.array(
        [
            float(field_environment.get("charge_field", 0.0)),
            float(field_environment.get("lattice_field", 0.0)),
            float(field_environment.get("coherence_field", 0.0)),
            float(field_environment.get("vacancy_field", 0.0)),
        ],
        dtype=np.float64,
    )
    ancilla_phase_vector = np.array(
        [
            math.cos(ancilla_phase_turns_mean * math.tau),
            math.sin(ancilla_phase_turns_mean * math.tau),
            math.cos((ancilla_phase_turns_mean + target_phase) * math.tau),
            math.sin((ancilla_phase_turns_mean + target_interval) * math.tau),
        ],
        dtype=np.float64,
    )
    motif_anchor = clamp_vector_norm(
        fanout_vector * (0.56 + 0.20 * activation_density)
        + motif_signature * (0.44 + 0.18 * motif_energy)
        + feedback_axis_vector * (0.18 + 0.14 * feedback_temporal_drive),
        max_norm=2.45,
    )
    motif_anchor = clamp_vector_norm(
        motif_anchor
        + kernel_control_signature * (0.16 + 0.16 * kernel_harmonic_resonance_mean)
        + feedback_delta_target_vector * (0.10 + 0.12 * feedback_delta_response_gate),
        max_norm=2.55,
    )
    motif_anchor = clamp_vector_norm(
        motif_anchor
        + btc_network_force_vector * (0.12 + 0.14 * btc_network_amplitude_pressure)
        + (btc_network_force_tensor @ motif_signature) * (0.04 + 0.06 * btc_network_phase_pressure),
        max_norm=2.72,
    )
    motif_anchor = clamp_vector_norm(
        motif_anchor
        + ancilla_phase_vector * (0.08 + 0.12 * ancilla_commit_gate)
        + kernel_control_signature * (0.04 + 0.06 * ancilla_temporal_persistence),
        max_norm=2.86,
    )
    temporal_sequence_drive = clamp_vector_norm(
        persistent_temporal_vector
        + (voltage_frequency_correlation_tensor @ motif_signature) * (0.18 + 0.22 * sequence_persistence_score)
        + (feedback_axis_tensor @ motif_signature) * (0.06 + 0.08 * feedback_flux_proxy),
        max_norm=2.55,
    )
    temporal_sequence_drive = clamp_vector_norm(
        temporal_sequence_drive
        + projected_temporal_dof_vector * (0.12 + 0.12 * sequence_persistence_score)
        + (voltage_frequency_dof_axis_tensor @ motif_signature) * (0.06 + 0.08 * temporal_index_overlap)
        + feedback_axis_vector * (0.06 + 0.08 * feedback_temporal_drive),
        max_norm=2.55,
    )
    temporal_sequence_drive = clamp_vector_norm(
        temporal_sequence_drive
        + kernel_control_signature * (0.10 + 0.12 * kernel_retro_gain_mean)
        + motif_signature * (0.04 + 0.06 * kernel_control_gate),
        max_norm=2.65,
    )
    temporal_sequence_drive = clamp_vector_norm(
        temporal_sequence_drive
        + feedback_delta_target_vector * (0.10 + 0.10 * feedback_delta_phase_retention)
        + kernel_control_signature * (0.06 + 0.08 * kernel_delta_gate_mean),
        max_norm=2.72,
    )
    temporal_sequence_drive = clamp_vector_norm(
        temporal_sequence_drive
        + btc_network_force_vector * (0.10 + 0.12 * btc_network_algorithm_bias)
        + (btc_network_force_tensor @ temporal_sequence_drive) * (0.02 + 0.04 * btc_network_phase_pressure),
        max_norm=2.86,
    )
    temporal_sequence_drive = clamp_vector_norm(
        temporal_sequence_drive
        + ancilla_phase_vector * (0.08 + 0.10 * ancilla_phase_alignment)
        + feedback_axis_vector * (0.04 + 0.06 * ancilla_temporal_persistence),
        max_norm=2.96,
    )
    balance_target = clamp_vector_norm(
        calibration_vector * (0.48 + 0.20 * interference_accounting)
        + calibration_seed * (0.20 + 0.16 * sweep_quality)
        + field_environment_vector * (0.16 + 0.14 * field_pressure)
        + fanout_vector * (0.10 + 0.10 * activation_density)
        + motif_anchor * (0.18 + 0.14 * motif_energy)
        + temporal_sequence_drive * (0.14 + 0.14 * sequence_persistence_score)
        + projected_temporal_dof_vector * (0.10 + 0.10 * temporal_index_overlap)
        + cascade_drive * (0.12 + 0.12 * interference_accounting)
        + feedback_axis_vector * (0.14 + 0.14 * feedback_memory_proxy),
        max_norm=2.85,
    )
    balance_target = clamp_vector_norm(
        balance_target + kernel_control_signature * (0.12 + 0.14 * kernel_balance_mean),
        max_norm=2.95,
    )
    balance_target = clamp_vector_norm(
        balance_target
        + ancilla_phase_vector * (0.08 + 0.10 * ancilla_convergence)
        + kernel_control_signature * (0.06 + 0.08 * ancilla_activation_gate),
        max_norm=3.02,
    )
    balance_target = clamp_vector_norm(
        balance_target + feedback_delta_target_vector * (0.08 + 0.10 * kernel_delta_gate_mean),
        max_norm=3.05,
    )
    balance_target = clamp_vector_norm(
        balance_target
        + btc_network_force_vector * (0.12 + 0.12 * btc_network_phase_pressure)
        + (btc_network_force_tensor @ projected_temporal_dof_vector) * (0.03 + 0.04 * btc_network_amplitude_pressure),
        max_norm=3.18,
    )
    lattice_feedback = clamp_vector_norm(
        environment_tensor @ previous_vector,
        max_norm=2.85,
    )
    balance_center = clamp_vector_norm(
        0.62 * balance_target
        + 0.24 * lattice_feedback
        + 0.14 * field_environment_vector
        + 0.10 * motif_anchor
        + 0.10 * feedback_axis_vector,
        max_norm=2.85,
    )
    balance_center = clamp_vector_norm(
        balance_center + kernel_control_signature * (0.08 + 0.10 * kernel_control_gate),
        max_norm=2.95,
    )
    balance_center = clamp_vector_norm(
        balance_center + feedback_delta_target_vector * (0.06 + 0.08 * feedback_delta_response_gate),
        max_norm=3.05,
    )
    balance_center = clamp_vector_norm(
        balance_center + btc_network_force_vector * (0.08 + 0.10 * btc_network_algorithm_bias),
        max_norm=3.12,
    )
    balance_center = clamp_vector_norm(
        balance_center
        + ancilla_phase_vector * (0.06 + 0.08 * ancilla_commit_ratio)
        + feedback_axis_vector * (0.04 + 0.06 * ancilla_temporal_persistence),
        max_norm=3.18,
    )
    nudge_rate = clamp01(
        0.05
        + 0.08 * target_window
        + 0.06 * sweep_quality
        + 0.06 * coupling_mean
        + 0.05 * unison_interval
        + 0.05 * lateral_interval
        + 0.06 * motif_stability
        + 0.04 * motif_consistency
        + 0.04 * sequence_persistence_score
        + 0.03 * temporal_index_overlap
        + 0.05 * feedback_memory_proxy
        + 0.04 * feedback_phase_alignment
        + 0.03 * feedback_stability_proxy
        + 0.04 * kernel_control_gate
        + 0.04 * kernel_balance_mean
        + 0.03 * kernel_retro_gain_mean
        + 0.05 * feedback_delta_response_gate
        + 0.04 * kernel_delta_gate_mean
        + 0.03 * kernel_retro_target_mean
        + 0.04 * ancilla_commit_ratio
        + 0.04 * ancilla_temporal_persistence
        + 0.03 * ancilla_activation_gate
    )
    nudged_vector = previous_vector * (1.0 - nudge_rate) + balance_center * nudge_rate
    blended_vector = clamp_vector_norm(
        nudged_vector * 0.92 + balance_target * 0.08,
        max_norm=2.85,
    )
    balance_error = clamp01(
        float(np.mean(np.abs(balance_center - blended_vector))) / 0.55
    )
    field_alignment_score = clamp01(1.0 - balance_error)
    nudge_magnitude = clamp01(
        float(np.linalg.norm(blended_vector - previous_vector)) / max(float(np.linalg.norm(balance_center)), 1.0e-6)
    )
    projected_balance = np.array(
        [
            clamp01(0.5 + 0.5 * math.tanh(float(blended_vector[0]))),
            clamp01(0.5 + 0.5 * math.tanh(float(blended_vector[1]))),
            clamp01(0.5 + 0.5 * math.tanh(float(blended_vector[2]))),
            clamp01(0.5 + 0.5 * math.tanh(float(blended_vector[3]))),
        ],
        dtype=np.float64,
    )
    motif_alignment_score = vector_similarity(projected_balance, motif_signature)
    calibration_support = clamp01(
        0.18 * field_alignment_score
        + 0.12 * field_pressure
        + 0.10 * larger_field_exposure
        + 0.09 * interference_accounting
        + 0.08 * activation_density
        + 0.08 * sweep_quality
        + 0.08 * motif_energy
        + 0.08 * motif_stability
        + 0.08 * motif_consistency
        + 0.07 * motif_alignment_score
        + 0.04 * stable_row_density
        + 0.04 * motif_cluster_support
        + 0.04 * target_interval
        + 0.04 * target_phase
        + 0.04 * (1.0 - nudge_magnitude)
        + 0.06 * sequence_persistence_score
        + 0.05 * voltage_frequency_flux
        + 0.04 * temporal_index_overlap
        + 0.06 * feedback_memory_proxy
        + 0.05 * feedback_phase_alignment
        + 0.04 * feedback_stability_proxy
        + 0.05 * kernel_balance_mean
        + 0.04 * kernel_harmonic_resonance_mean
        + 0.04 * kernel_control_gate
        + 0.05 * feedback_delta_response_gate
        + 0.04 * feedback_delta_phase_retention
        + 0.05 * kernel_delta_gate_mean
        + 0.04 * kernel_delta_memory_mean
        + 0.03 * kernel_retro_target_mean
        + 0.04 * ancilla_convergence
        + 0.04 * ancilla_phase_alignment
        + 0.03 * ancilla_tension_headroom
        + 0.03 * ancilla_gradient_headroom
    )
    calibration_readiness = clamp01(
        0.38 * previous_readiness
        + 0.14 * previous_motif_consistency
        + 0.48 * calibration_support
    )
    field_frequency_bias = clamp01(
        0.28 * calibration_readiness
        + 0.20 * target_window
        + 0.16 * interference_accounting
        + 0.10 * field_alignment_score
        + 0.10 * motif_energy
        + 0.08 * motif_consistency
        + 0.14 * unison_interval
        + 0.08 * voltage_frequency_flux
        + 0.08 * feedback_frequency_axis
        + 0.06 * feedback_phase_alignment
        + 0.08 * kernel_harmonic_resonance_mean
        + 0.06 * kernel_resonance_alignment_mean
        + 0.06 * feedback_delta_response_gate
        + 0.05 * kernel_delta_flux_mean
        + 0.06 * ancilla_phase_alignment
        + 0.05 * ancilla_temporal_persistence
        + 0.04 * ancilla_activation_gate
        + 0.08 * float(btc_network_force_vector[0])
        + 0.06 * btc_network_phase_pressure
    )
    field_amplitude_bias = clamp01(
        0.28 * calibration_readiness
        + 0.20 * target_phase
        + 0.16 * larger_field_exposure
        + 0.10 * field_alignment_score
        + 0.10 * motif_stability
        + 0.08 * stable_row_density
        + 0.14 * cascade_interval
        + 0.06 * sequence_persistence_score
        + 0.08 * feedback_amplitude_axis
        + 0.06 * feedback_memory_proxy
        + 0.08 * kernel_balance_mean
        + 0.06 * kernel_retro_gain_mean
        + 0.06 * feedback_delta_memory_retention
        + 0.05 * kernel_delta_memory_mean
        + 0.06 * ancilla_convergence
        + 0.05 * ancilla_current_norm
        + 0.04 * ancilla_tension_headroom
        + 0.08 * btc_network_amplitude_pressure
        + 0.06 * float(btc_network_force_vector[1])
    )
    target_gate = clamp01(
        0.24 * calibration_readiness
        + 0.20 * field_alignment_score
        + 0.12 * target_phase
        + 0.10 * interference_accounting
        + 0.08 * (1.0 - balance_error)
        + 0.10 * target_window
        + 0.08 * motif_alignment_score
        + 0.08 * motif_consistency
        + 0.06 * motif_repeat_norm
        + 0.06 * stable_row_density
        + 0.06 * sequence_persistence_score
        + 0.04 * temporal_index_overlap
        + 0.06 * feedback_phase_alignment
        + 0.05 * feedback_memory_proxy
        + 0.06 * kernel_control_gate
        + 0.05 * kernel_balance_mean
        + 0.04 * kernel_harmonic_resonance_mean
        + 0.05 * feedback_delta_response_gate
        + 0.04 * kernel_delta_gate_mean
        + 0.04 * kernel_delta_phase_alignment_mean
        + 0.05 * ancilla_commit_ratio
        + 0.04 * ancilla_phase_alignment
        + 0.03 * ancilla_temporal_persistence
        + 0.06 * btc_network_phase_pressure
        + 0.06 * btc_network_amplitude_pressure
        + 0.04 * btc_network_algorithm_bias
    )
    entry_threshold = 0.66 - 0.04 * target_window
    entry_field_floor = max(
        0.72,
        0.76
        - 0.03 * btc_network_phase_pressure
        - 0.03 * btc_network_amplitude_pressure
        - 0.02 * btc_network_algorithm_bias,
    )
    entry_trigger = bool(
        calibration_readiness >= entry_threshold
        and field_alignment_score >= entry_field_floor
        and target_gate >= 0.62
        and motif_energy >= 0.34
        and motif_stability >= 0.40
        and motif_consistency >= 0.78
        and motif_repeat_count >= 2
        and sequence_persistence_score >= 0.42
        and temporal_index_overlap >= 0.40
        and kernel_control_gate >= 0.18
        and feedback_delta_response_gate >= 0.16
    )
    if entry_trigger:
        entry_reason = "stable_interference_motif"
    elif motif_repeat_count < 2:
        entry_reason = "stabilizing_interference_motif"
    elif motif_consistency < 0.78:
        entry_reason = "tracking_motif_alignment"
    elif target_gate < 0.62:
        entry_reason = "tightening_target_gate"
    else:
        entry_reason = "balancing_lattice_field"
    kernel_execution_event = {
        "event_type": "kernel_execution_event",
        "entry_trigger": entry_trigger,
        "entry_reason": entry_reason,
        "calibration_readiness": float(calibration_readiness),
        "target_gate": float(target_gate),
        "entry_field_floor": float(entry_field_floor),
        "field_frequency_bias": float(field_frequency_bias),
        "field_amplitude_bias": float(field_amplitude_bias),
        "field_alignment_score": float(field_alignment_score),
        "balance_error": float(balance_error),
        "nudge_rate": float(nudge_rate),
        "motif_consistency": float(motif_consistency),
        "motif_repeat_count": int(motif_repeat_count),
        "motif_energy": float(motif_energy),
        "motif_stability": float(motif_stability),
        "sequence_persistence_score": float(sequence_persistence_score),
        "temporal_index_overlap": float(temporal_index_overlap),
        "voltage_frequency_flux": float(voltage_frequency_flux),
        "frequency_voltage_flux": float(frequency_voltage_flux),
        "gpu_feedback_source": str(gpu_pulse_feedback.get("source", "fallback")),
        "feedback_phase_anchor_turns": float(feedback_phase_anchor_turns),
        "feedback_memory_proxy": float(feedback_memory_proxy),
        "feedback_flux_proxy": float(feedback_flux_proxy),
        "feedback_stability_proxy": float(feedback_stability_proxy),
        "feedback_temporal_drive": float(feedback_temporal_drive),
        "feedback_delta_phase_shift_turns": float(feedback_delta_phase_shift_turns),
        "feedback_delta_phase_retention": float(feedback_delta_phase_retention),
        "feedback_delta_response_gate": float(feedback_delta_response_gate),
        "feedback_delta_response_energy": float(feedback_delta_response_energy),
        "feedback_delta_memory_retention": float(feedback_delta_memory_retention),
        "btc_network_phase_pressure": float(btc_network_phase_pressure),
        "btc_network_amplitude_pressure": float(btc_network_amplitude_pressure),
        "btc_network_algorithm_bias": float(btc_network_algorithm_bias),
        "kernel_balance_mean": float(kernel_balance_mean),
        "kernel_harmonic_resonance_mean": float(kernel_harmonic_resonance_mean),
        "kernel_retro_gain_mean": float(kernel_retro_gain_mean),
        "kernel_resonance_alignment_mean": float(kernel_resonance_alignment_mean),
        "kernel_resonance_bandwidth_mean": float(kernel_resonance_bandwidth_mean),
        "kernel_drive_mean": float(kernel_drive_mean),
        "kernel_control_gate": float(kernel_control_gate),
        "kernel_delta_gate_mean": float(kernel_delta_gate_mean),
        "kernel_delta_memory_mean": float(kernel_delta_memory_mean),
        "kernel_delta_flux_mean": float(kernel_delta_flux_mean),
        "kernel_delta_phase_alignment_mean": float(kernel_delta_phase_alignment_mean),
        "kernel_retro_target_mean": float(kernel_retro_target_mean),
        "ancilla_commit_ratio": float(ancilla_commit_ratio),
        "ancilla_commit_gate": float(ancilla_commit_gate),
        "ancilla_convergence": float(ancilla_convergence),
        "ancilla_flux": float(ancilla_flux),
        "ancilla_phase_alignment": float(ancilla_phase_alignment),
        "ancilla_current_norm": float(ancilla_current_norm),
        "ancilla_tension_headroom": float(ancilla_tension_headroom),
        "ancilla_gradient_headroom": float(ancilla_gradient_headroom),
        "ancilla_temporal_persistence": float(ancilla_temporal_persistence),
        "ancilla_activation_gate": float(ancilla_activation_gate),
        "trace_support": float(
            dict(previous_field_state.get("substrate_trace_state", {}) or {}).get(
                "trace_support", 0.0
            )
        ),
        "trace_resonance": float(
            dict(previous_field_state.get("substrate_trace_state", {}) or {}).get(
                "trace_resonance", 0.0
            )
        ),
    }
    return {
        "simulation_field_vector": [float(value) for value in blended_vector],
        "field_frequency_bias": float(field_frequency_bias),
        "field_amplitude_bias": float(field_amplitude_bias),
        "target_gate": float(target_gate),
        "interference_accounting": float(interference_accounting),
        "fourier_fanout_vector": [float(value) for value in fanout_vector],
        "activation_density": float(activation_density),
        "fanout_span": float(fanout_span),
        "cascade_phase": float(cascade_phase),
        "field_alignment_score": float(field_alignment_score),
        "balance_error": float(balance_error),
        "nudge_rate": float(nudge_rate),
        "nudge_magnitude": float(nudge_magnitude),
        "motif_signature": [float(value) for value in motif_signature],
        "motif_consistency": float(motif_consistency),
        "motif_repeat_count": int(motif_repeat_count),
        "motif_alignment_score": float(motif_alignment_score),
        "motif_energy": float(motif_energy),
        "motif_stability": float(motif_stability),
        "motif_spread": float(motif_spread),
        "stable_row_density": float(stable_row_density),
        "activation_row_count": int(activation_row_count),
        "motif_cluster_count": int(len(motif_clusters)),
        "temporal_sequence_index": int(temporal_sequence_accounting.get("temporal_sequence_index", 0)),
        "temporal_sequence_length": int(temporal_sequence_accounting.get("temporal_sequence_length", 0)),
        "temporal_sequence_indexes": list(temporal_sequence_accounting.get("temporal_sequence_indexes", []) or []),
        "temporal_persistence_span": int(temporal_sequence_accounting.get("temporal_persistence_span", 0)),
        "sequence_stride": int(temporal_sequence_accounting.get("sequence_stride", 0)),
        "sequence_persistence_score": float(sequence_persistence_score),
        "temporal_index_overlap": float(temporal_index_overlap),
        "voltage_frequency_flux": float(voltage_frequency_flux),
        "frequency_voltage_flux": float(frequency_voltage_flux),
        "persistent_temporal_vector": [float(value) for value in persistent_temporal_vector],
        "persistent_temporal_dof_vector": list(
            temporal_sequence_accounting.get("persistent_temporal_dof_vector", []) or []
        ),
        "projected_temporal_dof_vector": [float(value) for value in projected_temporal_dof_vector],
        "temporal_sequence_signature": list(temporal_sequence_accounting.get("temporal_sequence_signature", []) or []),
        "voltage_frequency_correlation_tensor": voltage_frequency_correlation_tensor.tolist(),
        "voltage_frequency_dof_axis_tensor": voltage_frequency_dof_axis_tensor.tolist(),
        "voltage_frequency_dof_tensor": list(
            temporal_sequence_accounting.get("voltage_frequency_dof_tensor", []) or []
        ),
        "gpu_pulse_feedback": dict(gpu_pulse_feedback),
        "feedback_temperature_norm": float(
            temporal_sequence_accounting.get("feedback_temperature_norm", 0.0)
        ),
        "feedback_thermal_headroom": float(
            temporal_sequence_accounting.get("feedback_thermal_headroom", 0.0)
        ),
        "feedback_temperature_velocity": float(
            temporal_sequence_accounting.get("feedback_temperature_velocity", 0.0)
        ),
        "feedback_environment_pressure": float(
            temporal_sequence_accounting.get("feedback_environment_pressure", 0.0)
        ),
        "feedback_environment_stability": float(
            temporal_sequence_accounting.get("feedback_environment_stability", 0.0)
        ),
        "feedback_latency_norm": float(temporal_sequence_accounting.get("feedback_latency_norm", 0.0)),
        "feedback_latency_jitter": float(
            temporal_sequence_accounting.get("feedback_latency_jitter", 0.0)
        ),
        "feedback_latency_gate": float(temporal_sequence_accounting.get("feedback_latency_gate", 0.0)),
        "feedback_delta_target_vector": [float(value) for value in feedback_delta_target_vector],
        "feedback_delta_phase_shift_turns": float(feedback_delta_phase_shift_turns),
        "feedback_delta_phase_retention": float(feedback_delta_phase_retention),
        "feedback_delta_response_gate": float(feedback_delta_response_gate),
        "feedback_delta_response_energy": float(feedback_delta_response_energy),
        "feedback_delta_memory_retention": float(feedback_delta_memory_retention),
        "feedback_delta_latency_norm": float(
            temporal_sequence_accounting.get("feedback_delta_latency_norm", 0.0)
        ),
        "feedback_delta_latency_gate": float(
            temporal_sequence_accounting.get("feedback_delta_latency_gate", 0.0)
        ),
        "feedback_delta_window_span_norm": float(
            temporal_sequence_accounting.get("feedback_delta_window_span_norm", 0.0)
        ),
        "feedback_delta_window_density": float(
            temporal_sequence_accounting.get("feedback_delta_window_density", 0.0)
        ),
        "feedback_delta_window_latency_alignment": float(
            temporal_sequence_accounting.get("feedback_delta_window_latency_alignment", 0.0)
        ),
        "feedback_delta_window_steps": int(
            temporal_sequence_accounting.get("feedback_delta_window_steps", 1)
        ),
        "feedback_delta_environment_pressure": float(
            temporal_sequence_accounting.get("feedback_delta_environment_pressure", 0.0)
        ),
        "feedback_delta_thermal_headroom": float(
            temporal_sequence_accounting.get("feedback_delta_thermal_headroom", 0.0)
        ),
        "btc_network_force_vector": [float(value) for value in btc_network_force_vector],
        "btc_network_force_tensor": btc_network_force_tensor.tolist(),
        "btc_network_phase_turns": [float(value) for value in btc_network_phase_turns],
        "btc_network_phase_pressure": float(btc_network_phase_pressure),
        "btc_network_amplitude_pressure": float(btc_network_amplitude_pressure),
        "btc_network_algorithm_bias": float(btc_network_algorithm_bias),
        "kernel_temporal_controls": kernel_temporal_controls,
        "kernel_ancilla_particles": kernel_ancilla_particles,
        "kernel_ancilla_summary": kernel_ancilla_summary,
        "kernel_balance_mean": float(kernel_balance_mean),
        "kernel_harmonic_resonance_mean": float(kernel_harmonic_resonance_mean),
        "kernel_retro_gain_mean": float(kernel_retro_gain_mean),
        "kernel_resonance_alignment_mean": float(kernel_resonance_alignment_mean),
        "kernel_resonance_bandwidth_mean": float(kernel_resonance_bandwidth_mean),
        "kernel_drive_mean": float(kernel_drive_mean),
        "kernel_control_gate": float(kernel_control_gate),
        "kernel_delta_gate_mean": float(kernel_delta_gate_mean),
        "kernel_delta_memory_mean": float(kernel_delta_memory_mean),
        "kernel_delta_flux_mean": float(kernel_delta_flux_mean),
        "kernel_delta_phase_alignment_mean": float(kernel_delta_phase_alignment_mean),
        "kernel_retro_target_mean": float(kernel_retro_target_mean),
        "ancilla_commit_ratio": float(ancilla_commit_ratio),
        "ancilla_commit_gate": float(ancilla_commit_gate),
        "ancilla_convergence": float(ancilla_convergence),
        "ancilla_flux": float(ancilla_flux),
        "ancilla_phase_alignment": float(ancilla_phase_alignment),
        "ancilla_current_norm": float(ancilla_current_norm),
        "ancilla_tension_headroom": float(ancilla_tension_headroom),
        "ancilla_gradient_headroom": float(ancilla_gradient_headroom),
        "ancilla_temporal_persistence": float(ancilla_temporal_persistence),
        "ancilla_activation_gate": float(ancilla_activation_gate),
        "ancilla_phase_turns_mean": float(ancilla_phase_turns_mean),
        "kernel_control_signature": [float(value) for value in kernel_control_signature],
        "feedback_axis_vector": [float(value) for value in feedback_axis_vector],
        "feedback_axis_tensor": feedback_axis_tensor.tolist(),
        "substrate_trace_state": dict(previous_field_state.get("substrate_trace_state", {}) or {}),
        "substrate_trace_vram": dict(previous_field_state.get("substrate_trace_vram", {}) or {}),
        "substrate_material": "silicon_wafer",
        "silicon_reference_source": str(NIST_REFERENCE.name),
        "compute_regime": str(previous_field_state.get("compute_regime", "classical_calibration")),
        "vector_harmonic_gate": float(previous_field_state.get("vector_harmonic_gate", 0.0)),
        "harmonic_compute_weight": float(previous_field_state.get("harmonic_compute_weight", 0.0)),
        "gpu_pulse_dof_labels": list(
            temporal_sequence_accounting.get("gpu_pulse_dof_labels", []) or list(GPU_PULSE_DOF_LABELS)
        ),
        "calibration_readiness": float(calibration_readiness),
        "entry_trigger": entry_trigger,
        "entry_threshold": float(entry_threshold),
        "entry_reason": entry_reason,
        "kernel_execution_event": kernel_execution_event,
    }


def build_btc_header_from_seed(
    seed_bytes: bytes,
    pulse_index: int,
    difficulty_norm: float,
    quartet: dict[str, float],
) -> tuple[str, int, str]:
    version = (
        0x20000000
        | ((int(round(float(quartet["v_code"]) * 255.0)) & 0xFF) << 8)
        | (int(round(float(quartet["i_code"]) * 255.0)) & 0xFF)
    )
    timestamp = 1712275200 + pulse_index * 11 + int(round(float(quartet["f_code"]) * 1000.0))
    nbits = 0x1D00FFFF - int(round(float(difficulty_norm) * 0x00000FFF))
    nbits = max(0x1B0404CB, min(0x1D00FFFF, nbits))
    nbits_hex = f"{nbits:08x}"
    prev_hash = hashlib.sha256(seed_bytes + b"prev").digest()
    merkle_root = hashlib.sha256(seed_bytes + b"merkle").digest()
    header = b"".join(
        [
            struct.pack("<I", version),
            prev_hash[::-1],
            merkle_root[::-1],
            struct.pack("<I", timestamp),
            struct.pack("<I", nbits),
            struct.pack("<I", 0),
        ]
    )
    return header.hex(), timestamp, nbits_hex


def compute_ledger_delta(
    current_state: dict[str, Any],
    candidate_next_state: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    prev_coherence = float(current_state.get("coherence_peak", candidate_next_state.get("coherence_peak", 0.0)))
    next_coherence = float(candidate_next_state.get("coherence_peak", 0.0))
    prev_yield = int(current_state.get("yield_count", 0))
    next_yield = int(candidate_next_state.get("yield_count", 0))
    prev_residual = float(current_state.get("temporal_residual", candidate_next_state.get("temporal_residual", 0.0)))
    next_residual = float(candidate_next_state.get("temporal_residual", 0.0))
    return {
        "operator": str(ctx.get("ledger_delta_operator", "compute_ledger_delta")),
        "coherence_drop": max(0.0, prev_coherence - next_coherence),
        "yield_delta": next_yield - prev_yield,
        "temporal_delta": abs(next_residual - prev_residual),
    }


def accept_state(
    current_state: dict[str, Any],
    candidate_next_state: dict[str, Any],
    ledger_delta: dict[str, Any],
    ctx: dict[str, Any],
) -> bool:
    gate_coherence = float(ctx.get("gate_coherence", 0.0))
    amplitude_cap = float(ctx.get("amplitude_cap", 1.0))
    coherence_floor = max(0.30, gate_coherence * 0.28)
    clamp_ceiling = min(1.0, amplitude_cap + 0.45)
    return (
        int(candidate_next_state.get("yield_count", 0)) > 0
        and
        float(candidate_next_state.get("coherence_peak", 0.0)) >= coherence_floor
        and float(candidate_next_state.get("clamp_pressure", 0.0)) <= clamp_ceiling
        and float(ledger_delta.get("temporal_delta", 0.0)) <= 1.0
    )


def make_sink_state(current_state: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "sink",
        "pulse_id": int(current_state.get("pulse_id", -1)),
        "yield_count": 0,
        "coherence_peak": 0.0,
        "temporal_residual": float(current_state.get("temporal_residual", 0.0)),
        "clamp_pressure": min(1.0, float(ctx.get("amplitude_cap", 1.0)) + 0.15),
        "worker_count": int(current_state.get("worker_count", 0)),
        "worker_batches": [],
        "nonces": [],
        "nbits": str(current_state.get("nbits", "")),
        "target_hex": str(current_state.get("target_hex", "")),
        "network_target_hex": str(current_state.get("network_target_hex", "")),
        "effective_vector": {"x": 0.0, "y": 0.0, "z": 0.0, "t_eff": 0.0},
        "candidate_yield_count": 0,
        "prototype_valid_count": 0,
        "network_valid_count": 0,
        "interference_resonance": 0.0,
        "target_alignment": 0.0,
        "target_interval": 0.0,
        "cascade_activation": 0.0,
        "candidate_function_score": 0.0,
        "candidate_round_coupling": 0.0,
        "stable_target_phase_pressure": 0.0,
        "candidate_phase_length_pressure": 0.0,
        "candidate_phase_alignment_score": 0.0,
        "candidate_phase_confinement_cost": 0.0,
        "candidate_sequence_persistence": 0.0,
        "candidate_temporal_overlap": 0.0,
        "candidate_voltage_frequency_flux": 0.0,
        "candidate_gpu_feedback_delta_score": 0.0,
        "candidate_decode_integrity": 0.0,
        "candidate_decode_entropy": 0.0,
        "candidate_decode_phase_integrity": 0.0,
        "candidate_decode_phase_alignment": 0.0,
        "candidate_decode_target_ring_alignment": 0.0,
        "candidate_decode_target_prefix_lock": 0.0,
        "candidate_decode_target_prefix_vector_alignment": 0.0,
        "candidate_decode_target_prefix_vector_phase_pressure": 0.0,
        "candidate_decode_phase_orbital_alignment": 0.0,
        "candidate_decode_phase_orbital_resonance": 0.0,
        "candidate_decode_phase_orbital_stability": 0.0,
        "candidate_decode_prefix_asymptote": 1.0,
        "candidate_hash_target_phase_pressure": 0.0,
        "candidate_hash_target_phase_cost": 1.0,
        "candidate_hash_target_prefix_lock": 0.0,
        "candidate_hash_target_window_coverage": 0.0,
        "candidate_hash_target_band_alignment": 0.0,
        "candidate_hash_target_flux_alignment": 0.0,
        "probe_pool_size": 0,
        "probe_cluster_count": 0,
        "dominant_probe_cluster": "",
        "best_phase_length_pressure": 0.0,
        "best_phase_alignment_score": 0.0,
        "best_phase_confinement_cost": 0.0,
        "best_decode_integrity": 0.0,
        "best_decode_entropy": 0.0,
        "best_decode_phase_integrity": 0.0,
        "best_decode_phase_alignment": 0.0,
        "best_decode_target_ring_alignment": 0.0,
        "best_decode_target_prefix_lock": 0.0,
        "best_decode_target_prefix_vector_alignment": 0.0,
        "best_decode_target_prefix_vector_phase_pressure": 0.0,
        "best_decode_phase_orbital_alignment": 0.0,
        "best_decode_phase_orbital_resonance": 0.0,
        "best_decode_phase_orbital_stability": 0.0,
        "best_decode_prefix_asymptote": 1.0,
        "best_sequence_persistence": 0.0,
        "best_temporal_overlap": 0.0,
        "best_voltage_frequency_flux": 0.0,
        "best_gpu_feedback_delta_score": 0.0,
        "best_function_score": 0.0,
        "best_round_coupling": 0.0,
        "best_hash_target_phase_pressure": 0.0,
        "best_hash_target_phase_cost": 1.0,
        "best_hash_target_prefix_lock": 0.0,
        "best_hash_target_window_coverage": 0.0,
        "best_hash_target_band_alignment": 0.0,
        "best_hash_target_flux_alignment": 0.0,
        "submission_anchor_rate_per_second": 0.0,
        "submission_tick_duration_s": 0.0,
        "submission_jitter_fraction": 0.0,
        "cuda_kernel_telemetry": {},
        "verification_mode": "cpu_hash_check",
        "verification_consistency": 0.0,
        "production_unlock_ready": False,
        "calibration_readiness": 0.0,
        "entry_trigger": False,
        "interference_field": {},
        "kernel_execution_event": {},
        "motif_consistency": 0.0,
        "motif_repeat_count": 0,
        "motif_energy": 0.0,
        "field_pressure": float(current_state.get("field_pressure", 0.0)),
        "larger_field_exposure": float(current_state.get("larger_field_exposure", 0.0)),
        "dominant_basin": str(current_state.get("dominant_basin", "")),
        "path_equivalence_error": 1.0,
        "temporal_ordering_delta": 1.0,
        "basis_rotation_residual": 1.0,
        "trace_relative_spatial_field": [0.0, 0.0, 0.0, 0.0],
        "coherent_noise_axis_vector": [0.0, 0.0, 0.0, 0.0],
        "noise_resonance_nodes": [0.0, 0.0, 0.0, 0.0],
        "drift_compensation_vector": [0.0, 0.0, 0.0, 0.0],
        "noise_resonance_gate": 0.0,
        "environment_turbulence": 0.0,
    }


def persist_research_state(path: Path, prototype: dict[str, Any]) -> None:
    state: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = load_json(path)
            if isinstance(loaded, dict):
                state = loaded
        except Exception:
            state = {}

    pulses = list(prototype.get("pulse_batches", []) or [])
    latest = next(
        (
            pulse
            for pulse in reversed(pulses)
            if int(pulse.get("yield_count", 0)) > 0
            or int(pulse.get("prototype_valid_count", 0)) > 0
            or int(pulse.get("probe_pool_size", 0)) > 0
        ),
        pulses[-1] if pulses else {},
    )
    latest_field_state = dict(latest.get("simulation_field_state", {}) or {})
    latest_event = dict(latest.get("kernel_execution_event", {}) or {})
    runtime = dict(state.get("runtime", {}) or {})
    runtime["btc_miner_prototype"] = {
        "enabled": 1,
        "live_nonce_yield": int(latest.get("yield_count", 0)),
        "coherence_peak": float(latest.get("coherence_peak", 0.0)),
        "worker_count": int(latest.get("worker_count", 0)),
        "queue_depth": int(sum(int(worker.get("queue_depth", 0)) for worker in latest.get("worker_batches", []) or [])),
        "carrier_count": int(prototype.get("carrier_count", 70)),
        "max_depth": int(prototype.get("max_depth", 5)),
        "effective_vector": dict(latest.get("effective_vector", prototype.get("effective_vector", {})) or {}),
        "silicon_calibration": dict(prototype.get("silicon_calibration", {}) or {}),
        "field_pressure": float(latest.get("field_pressure", prototype.get("silicon_calibration", {}).get("field_pressure", 0.0))),
        "dominant_basin": str(latest.get("dominant_basin", dict(prototype.get("silicon_calibration", {}).get("dominant_basin", {}) or {}).get("basin_id", ""))),
        "interference_resonance": float(latest.get("interference_resonance", 0.0)),
        "entry_trigger": bool(latest_event.get("entry_trigger", latest.get("entry_trigger", False))),
        "kernel_execution_event": latest_event,
        "substrate_material": str(
            latest.get(
                "substrate_material",
                latest_event.get(
                    "substrate_material",
                    latest_field_state.get("substrate_material", "silicon_wafer"),
                ),
            )
        ),
        "silicon_reference_source": str(
            latest.get(
                "silicon_reference_source",
                latest_event.get(
                    "silicon_reference_source",
                    latest_field_state.get("silicon_reference_source", NIST_REFERENCE.name),
                ),
            )
        ),
        "gpu_feedback_source": str(
            latest.get(
                "gpu_feedback_source",
                latest_field_state.get("gpu_pulse_feedback_post_cuda", {}).get(
                    "source",
                    latest_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                ),
            )
        ),
        "gpu_injection_feedback_source": str(
            latest.get(
                "gpu_injection_feedback_source",
                latest_field_state.get("gpu_pulse_feedback_injection", {}).get(
                    "source",
                    latest_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                ),
            )
        ),
        "compute_regime": str(latest_event.get("compute_regime", latest_field_state.get("compute_regime", "classical_calibration"))),
        "vector_harmonic_gate": float(latest_event.get("vector_harmonic_gate", latest_field_state.get("vector_harmonic_gate", 0.0))),
        "harmonic_compute_weight": float(latest_event.get("harmonic_compute_weight", latest_field_state.get("harmonic_compute_weight", 0.0))),
        "motif_consistency": float(
            latest.get("motif_consistency", latest_field_state.get("motif_consistency", 0.0))
        ),
        "motif_repeat_count": int(
            latest.get("motif_repeat_count", latest_field_state.get("motif_repeat_count", 0))
        ),
        "motif_energy": float(latest.get("motif_energy", latest_field_state.get("motif_energy", 0.0))),
        "selection_mode": str(latest.get("selection_mode", "network_only")),
        "candidate_function_score": float(latest.get("candidate_function_score", 0.0)),
        "candidate_round_coupling": float(latest.get("candidate_round_coupling", 0.0)),
        "stable_target_phase_pressure": float(latest.get("stable_target_phase_pressure", 0.0)),
        "candidate_phase_length_pressure": float(latest.get("candidate_phase_length_pressure", 0.0)),
        "candidate_phase_alignment_score": float(latest.get("candidate_phase_alignment_score", 0.0)),
        "candidate_phase_confinement_cost": float(latest.get("candidate_phase_confinement_cost", 0.0)),
        "candidate_sequence_persistence": float(latest.get("candidate_sequence_persistence", 0.0)),
        "candidate_temporal_overlap": float(latest.get("candidate_temporal_overlap", 0.0)),
        "candidate_voltage_frequency_flux": float(latest.get("candidate_voltage_frequency_flux", 0.0)),
        "candidate_gpu_feedback_delta_score": float(latest.get("candidate_gpu_feedback_delta_score", 0.0)),
        "candidate_decode_integrity": float(latest.get("candidate_decode_integrity", 0.0)),
        "candidate_decode_entropy": float(latest.get("candidate_decode_entropy", 0.0)),
        "candidate_decode_phase_integrity": float(latest.get("candidate_decode_phase_integrity", 0.0)),
        "candidate_decode_phase_alignment": float(latest.get("candidate_decode_phase_alignment", 0.0)),
        "candidate_decode_target_ring_alignment": float(latest.get("candidate_decode_target_ring_alignment", 0.0)),
        "candidate_decode_target_prefix_lock": float(latest.get("candidate_decode_target_prefix_lock", 0.0)),
        "candidate_decode_target_prefix_vector_alignment": float(
            latest.get("candidate_decode_target_prefix_vector_alignment", 0.0)
        ),
        "candidate_decode_target_prefix_vector_phase_pressure": float(
            latest.get("candidate_decode_target_prefix_vector_phase_pressure", 0.0)
        ),
        "candidate_decode_phase_orbital_alignment": float(
            latest.get("candidate_decode_phase_orbital_alignment", 0.0)
        ),
        "candidate_decode_phase_orbital_resonance": float(
            latest.get("candidate_decode_phase_orbital_resonance", 0.0)
        ),
        "candidate_decode_phase_orbital_stability": float(
            latest.get("candidate_decode_phase_orbital_stability", 0.0)
        ),
        "candidate_decode_prefix_asymptote": float(latest.get("candidate_decode_prefix_asymptote", 1.0)),
        "candidate_hash_target_phase_pressure": float(latest.get("candidate_hash_target_phase_pressure", 0.0)),
        "candidate_hash_target_phase_cost": float(latest.get("candidate_hash_target_phase_cost", 1.0)),
        "candidate_hash_target_prefix_lock": float(latest.get("candidate_hash_target_prefix_lock", 0.0)),
        "candidate_hash_target_window_coverage": float(latest.get("candidate_hash_target_window_coverage", 0.0)),
        "candidate_hash_target_band_alignment": float(latest.get("candidate_hash_target_band_alignment", 0.0)),
        "candidate_hash_target_flux_alignment": float(latest.get("candidate_hash_target_flux_alignment", 0.0)),
        "prototype_valid_count": int(latest.get("prototype_valid_count", 0)),
        "network_valid_count": int(latest.get("network_valid_count", 0)),
        "probe_pool_size": int(latest.get("probe_pool_size", 0)),
        "probe_cluster_count": int(latest.get("probe_cluster_count", 0)),
        "dominant_probe_cluster": str(latest.get("dominant_probe_cluster", "")),
        "candidate_count": int(latest.get("cuda_kernel_telemetry", {}).get("candidate_count", 0)),
        "expanded_eval_count": int(latest.get("cuda_kernel_telemetry", {}).get("expanded_eval_count", 0)),
        "expanded_keep_count": int(latest.get("cuda_kernel_telemetry", {}).get("expanded_keep_count", 0)),
        "best_phase_length_pressure": float(latest.get("best_phase_length_pressure", 0.0)),
        "best_phase_alignment_score": float(latest.get("best_phase_alignment_score", 0.0)),
        "best_phase_confinement_cost": float(latest.get("best_phase_confinement_cost", 0.0)),
        "best_decode_integrity": float(latest.get("best_decode_integrity", 0.0)),
        "best_decode_entropy": float(latest.get("best_decode_entropy", 0.0)),
        "best_decode_phase_integrity": float(latest.get("best_decode_phase_integrity", 0.0)),
        "best_decode_phase_alignment": float(latest.get("best_decode_phase_alignment", 0.0)),
        "best_decode_target_ring_alignment": float(latest.get("best_decode_target_ring_alignment", 0.0)),
        "best_decode_target_prefix_lock": float(latest.get("best_decode_target_prefix_lock", 0.0)),
        "best_decode_target_prefix_vector_alignment": float(
            latest.get("best_decode_target_prefix_vector_alignment", 0.0)
        ),
        "best_decode_target_prefix_vector_phase_pressure": float(
            latest.get("best_decode_target_prefix_vector_phase_pressure", 0.0)
        ),
        "best_decode_phase_orbital_alignment": float(
            latest.get("best_decode_phase_orbital_alignment", 0.0)
        ),
        "best_decode_phase_orbital_resonance": float(
            latest.get("best_decode_phase_orbital_resonance", 0.0)
        ),
        "best_decode_phase_orbital_stability": float(
            latest.get("best_decode_phase_orbital_stability", 0.0)
        ),
        "best_decode_prefix_asymptote": float(latest.get("best_decode_prefix_asymptote", 1.0)),
        "best_sequence_persistence": float(latest.get("best_sequence_persistence", 0.0)),
        "best_temporal_overlap": float(latest.get("best_temporal_overlap", 0.0)),
        "best_voltage_frequency_flux": float(latest.get("best_voltage_frequency_flux", 0.0)),
        "best_gpu_feedback_delta_score": float(latest.get("best_gpu_feedback_delta_score", 0.0)),
        "best_function_score": float(latest.get("best_function_score", 0.0)),
        "best_round_coupling": float(latest.get("best_round_coupling", 0.0)),
        "best_hash_target_phase_pressure": float(latest.get("best_hash_target_phase_pressure", 0.0)),
        "best_hash_target_phase_cost": float(latest.get("best_hash_target_phase_cost", 1.0)),
        "best_hash_target_prefix_lock": float(latest.get("best_hash_target_prefix_lock", 0.0)),
        "best_hash_target_window_coverage": float(latest.get("best_hash_target_window_coverage", 0.0)),
        "best_hash_target_band_alignment": float(latest.get("best_hash_target_band_alignment", 0.0)),
        "best_hash_target_flux_alignment": float(latest.get("best_hash_target_flux_alignment", 0.0)),
        "submission_anchor_rate_per_second": float(latest.get("submission_anchor_rate_per_second", 0.0)),
        "submission_tick_duration_s": float(latest.get("submission_tick_duration_s", 0.0)),
        "submission_jitter_fraction": float(latest.get("submission_jitter_fraction", 0.0)),
        "injection_feedback_window_ms": float(
            latest.get("injection_feedback_window_ms", 0.0)
        ),
        "injection_observation_gap_ms": float(
            latest.get("injection_observation_gap_ms", 0.0)
        ),
        "injection_observation_freshness_gate": float(
            latest.get("injection_observation_freshness_gate", 0.0)
        ),
        "injection_response_gate": float(latest.get("injection_response_gate", 0.0)),
        "observation_gap_ms": float(latest.get("observation_gap_ms", 0.0)),
        "observation_freshness_gate": float(latest.get("observation_freshness_gate", 0.0)),
        "dispatch_feedback_ratio": float(latest.get("dispatch_feedback_ratio", 0.0)),
        "post_feedback_apply_gate": float(latest_event.get("post_feedback_apply_gate", 0.0)),
        "post_feedback_applied": bool(latest_event.get("post_feedback_applied", False)),
        "trace_support": float(latest_event.get("trace_support", latest_field_state.get("substrate_trace_state", {}).get("trace_support", 0.0))),
        "trace_resonance": float(latest_event.get("trace_resonance", latest_field_state.get("substrate_trace_state", {}).get("trace_resonance", 0.0))),
        "trace_alignment": float(latest_event.get("trace_alignment", latest_field_state.get("substrate_trace_state", {}).get("trace_alignment", 0.0))),
        "trace_vram_resident": bool(latest_event.get("trace_vram_resident", latest_field_state.get("substrate_trace_vram", {}).get("resident", False))),
        "trace_vram_updates": int(latest_event.get("trace_vram_updates", latest_field_state.get("substrate_trace_vram", {}).get("update_count", 0))),
        "ancilla_commit_ratio": float(
            latest.get("ancilla_commit_ratio", latest_field_state.get("ancilla_commit_ratio", 0.0))
        ),
        "ancilla_commit_gate": float(
            latest.get("ancilla_commit_gate", latest_field_state.get("ancilla_commit_gate", 0.0))
        ),
        "ancilla_convergence": float(
            latest.get("ancilla_convergence", latest_field_state.get("ancilla_convergence", 0.0))
        ),
        "ancilla_flux": float(latest.get("ancilla_flux", latest_field_state.get("ancilla_flux", 0.0))),
        "ancilla_phase_alignment": float(
            latest.get("ancilla_phase_alignment", latest_field_state.get("ancilla_phase_alignment", 0.0))
        ),
        "ancilla_current_norm": float(
            latest.get("ancilla_current_norm", latest_field_state.get("ancilla_current_norm", 0.0))
        ),
        "ancilla_tension_headroom": float(
            latest.get("ancilla_tension_headroom", latest_field_state.get("ancilla_tension_headroom", 0.0))
        ),
        "ancilla_gradient_headroom": float(
            latest.get("ancilla_gradient_headroom", latest_field_state.get("ancilla_gradient_headroom", 0.0))
        ),
        "ancilla_temporal_persistence": float(
            latest.get(
                "ancilla_temporal_persistence",
                latest_field_state.get("ancilla_temporal_persistence", 0.0),
            )
        ),
        "ancilla_activation_gate": float(
            latest.get("ancilla_activation_gate", latest_field_state.get("ancilla_activation_gate", 0.0))
        ),
        "cuda_kernel_telemetry": dict(latest.get("cuda_kernel_telemetry", {}) or {}),
        "verification_mode": str(latest.get("verification_mode", "cpu_hash_check")),
        "verification_consistency": float(latest.get("verification_consistency", 0.0)),
        "production_unlock_ready": bool(latest.get("production_unlock_ready", False)),
        "path_equivalence_error": float(latest.get("path_equivalence_error", 0.0)),
        "basis_rotation_residual": float(latest.get("basis_rotation_residual", 0.0)),
    }
    state["runtime"] = runtime
    state["btc_miner_prototype"] = {
        "mode": str(prototype.get("mode", "")),
        "quartet": dict(prototype.get("quartet", {}) or {}),
        "difficulty_norm": float(prototype.get("difficulty_norm", 0.0)),
        "amplitude_cap": float(prototype.get("amplitude_cap", 0.0)),
        "schematic": dict(prototype.get("schematic", {}) or {}),
        "pseudocode": list(prototype.get("pseudocode", []) or []),
        "silicon_calibration": dict(prototype.get("silicon_calibration", {}) or {}),
        "fourier_kernel_fanout": dict(prototype.get("fourier_kernel_fanout", {}) or {}),
        "interference_vector_field": dict(prototype.get("interference_vector_field", {}) or {}),
        "simulation_field_state": dict(prototype.get("simulation_field_state", {}) or {}),
        "psi_encode": dict(prototype.get("psi_encode", {}) or {}),
        "temporal_manifold": dict(prototype.get("temporal_manifold", {}) or {}),
        "effective_vector": dict(prototype.get("effective_vector", {}) or {}),
        "manifold_diagnostics": dict(prototype.get("manifold_diagnostics", {}) or {}),
        "yield_sequence": list(prototype.get("yield_sequence", []) or []),
        "coherence_sequence": list(prototype.get("coherence_sequence", []) or []),
        "interference_resonance_sequence": list(prototype.get("interference_resonance_sequence", []) or []),
        "calibration_readiness_sequence": list(prototype.get("calibration_readiness_sequence", []) or []),
        "vector_magnitude_sequence": list(prototype.get("vector_magnitude_sequence", []) or []),
        "temporal_projection_sequence": list(prototype.get("temporal_projection_sequence", []) or []),
        "latest_summary": dict(prototype.get("latest_summary", {}) or {}),
        "operator_pipeline": dict(prototype.get("operator_pipeline", {}) or {}),
        "pulse_batches": pulses,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")


def build_btc_miner_prototype(
    config: SimulationConfig,
    nist: dict[str, float],
    mean_phase_lock_matrix: np.ndarray,
    theta_history: np.ndarray,
    amplitude_history: np.ndarray,
    shared_history: np.ndarray,
    coherence_history: np.ndarray,
    curvature_history: np.ndarray,
    step_dominant_freqs: np.ndarray,
    packet_classes: list[dict[str, Any]],
    tensor_gradient_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    pulse_ops = load_json(PULSE_OPERATORS)
    temporal_schema = load_json(TEMPORAL_COUPLING)
    process_schema = load_json(PROCESS_SUBSTRATE)
    state = load_json(ROOT_STATE) if ROOT_STATE.exists() else {}

    pulse_codes = dict(temporal_schema.get("pulse_codes", {}) or {})
    encoded_event_model = dict(temporal_schema.get("encoded_event_model", {}) or {})
    pulse_center = dict(pulse_ops.get("pulse_center", {}) or {})
    quartet = {
        "f_code": float(pulse_codes.get("f_code", pulse_center.get("F", 0.245))),
        "a_code": float(pulse_codes.get("a_code", pulse_center.get("A", 0.18))),
        "i_code": float(pulse_codes.get("i_code", pulse_center.get("I", 0.33))),
        "v_code": float(pulse_codes.get("v_code", pulse_center.get("V", 0.33))),
    }
    normalized_window = dict(
        pulse_codes.get("normalized_window", {}) or pulse_ops.get("rtx2060_normalized_window", {})
    )
    deviation_ops = dict(
        temporal_schema.get("deviation_operators", {}) or pulse_ops.get("metrics", {})
    )
    collapse_gates = dict(temporal_schema.get("collapse_gates", {}) or {})
    process_ops = dict(process_schema.get("canonical_operator_pipeline", {}) or {})
    pulse_step = dict(pulse_ops.get("pulse_step", {}) or {})

    amplitude_cap = window_norm(quartet["a_code"], dict(normalized_window.get("amplitude", {}) or {}))
    baseline_frequency_norm = window_norm(
        quartet["f_code"], dict(normalized_window.get("frequency", {}) or {})
    )

    guidance = dict(state.get("guidance", {}) or {})
    gpu_calibration = dict(state.get("gpu_calibration", {}) or {})
    process_state = dict(state.get("process_substrate", {}) or {})
    subsystem_state = dict(state.get("subsystem_substrate", {}) or {})

    baseline_coherence = float(np.mean(coherence_history))
    baseline_shared = float(np.mean(shared_history))
    prediction_confidence = float(gpu_calibration.get("prediction_confidence_norm", baseline_coherence))
    temporal_persistence = float(guidance.get("temporal_coupling_norm", baseline_shared))
    residual_norm = clamp01(float(process_state.get("residual_norm_q15", 0)) / 32767.0)
    coupling_norm = clamp01(float(subsystem_state.get("aggregate_coupling_norm_q15", 0)) / 32767.0)
    difficulty_norm = clamp01(
        0.28
        + (1.0 - prediction_confidence) * 0.45
        + residual_norm * 0.20
        + (1.0 - temporal_persistence) * 0.12
        + coupling_norm * 0.10
    )
    silicon_calibration = build_silicon_lattice_calibration(
        config=config,
        nist=nist,
        quartet=quartet,
        deviation_ops=deviation_ops,
        baseline_frequency_norm=baseline_frequency_norm,
        amplitude_cap=amplitude_cap,
        temporal_persistence=temporal_persistence,
        prediction_confidence=prediction_confidence,
        residual_norm=residual_norm,
        coupling_norm=coupling_norm,
        mean_phase_lock_matrix=mean_phase_lock_matrix,
        amplitude_history=amplitude_history,
        shared_history=shared_history,
        coherence_history=coherence_history,
        curvature_history=curvature_history,
        step_dominant_freqs=step_dominant_freqs,
        packet_classes=packet_classes,
        tensor_gradient_samples=tensor_gradient_samples,
    )
    fourier_kernel_fanout = build_fourier_kernel_fanout(
        config=config,
        step_dominant_freqs=step_dominant_freqs,
        mean_phase_lock_matrix=mean_phase_lock_matrix,
        theta_history=theta_history,
        amplitude_history=amplitude_history,
        shared_history=shared_history,
        coherence_history=coherence_history,
    )

    base_carrier_count = min(128, max(84, 84 + int(round(difficulty_norm * 28.0))))
    carrier_count = base_carrier_count
    adaptive_depth = min(5, max(3, 3 + int(round(difficulty_norm * 2.0))))
    min_worker_count = 6
    max_worker_count = 14
    adaptive_workers = min(max_worker_count, max(min_worker_count, 6 + int(round(difficulty_norm * 6.0))))
    packet_class_index = {int(row.get("packet_id", -1)): row for row in packet_classes}
    tensor_gradient_index = {int(row.get("packet_id", -1)): row for row in tensor_gradient_samples}
    pulse_count = 10

    tensor_metrics = dict(silicon_calibration.get("tensor_metrics", {}) or {})
    phase_gradient_norm = float(tensor_metrics.get("phase_gradient_norm", 0.0))
    amplitude_gradient_norm = float(tensor_metrics.get("amplitude_gradient_norm", 0.0))
    field_pressure = float(silicon_calibration.get("field_pressure", 0.0))
    larger_field_exposure = float(silicon_calibration.get("larger_field_exposure", 0.0))
    dominant_basin = dict(silicon_calibration.get("dominant_basin", {}) or {})
    dominant_basin_id = str(dominant_basin.get("basin_id", "unknown_basin"))
    dominant_basin_affinity = str(dominant_basin.get("packet_affinity", "shared"))
    wave_step_field = dict(silicon_calibration.get("wave_step_field", {}) or {})
    axis_step_interval = dict(silicon_calibration.get("axis_step_interval", {}) or {})
    lattice_amplitude_cap = clamp01(
        0.75 * amplitude_cap + 0.25 * float(silicon_calibration.get("amplitude_guard", amplitude_cap))
    )

    sweep_entries: list[dict[str, Any]] = []
    unison_step_interval = max(
        float(pulse_step.get("F", 0.01)),
        float(wave_step_field.get("unison_interval", 0.0)),
    )
    lateral_step_interval = max(
        float(pulse_step.get("A", 0.02)) * 0.5,
        float(wave_step_field.get("lateral_interval", 0.0)),
    )
    cascade_step_interval = max(
        float(wave_step_field.get("cascade_interval", 0.0)),
        0.5 * (unison_step_interval + lateral_step_interval),
    )
    for freq_dir in (-1, 0, 1):
        for amp_dir in (-1, 0, 1):
            lateral_sign = float(freq_dir) if freq_dir != 0 else float(amp_dir)
            deltas = {
                "F": float(freq_dir) * max(unison_step_interval, float(axis_step_interval.get("F", unison_step_interval))),
                "A": float(amp_dir) * max(unison_step_interval, float(axis_step_interval.get("A", unison_step_interval))),
                "I": lateral_sign * max(lateral_step_interval, float(axis_step_interval.get("I", lateral_step_interval))),
                "V": float(amp_dir - freq_dir) * max(cascade_step_interval * 0.5, float(axis_step_interval.get("V", lateral_step_interval))),
            }
            score_value = pulse_packet_dev(dict(deviation_ops.get("score", {}) or {}), deltas)
            trap_value = pulse_packet_dev(dict(deviation_ops.get("trap", {}) or {}), deltas)
            coherence_value = pulse_packet_dev(dict(deviation_ops.get("coherence", {}) or {}), deltas)
            curvature_value = pulse_packet_dev(dict(deviation_ops.get("curvature", {}) or {}), deltas)
            score_center = float(dict(deviation_ops.get("score", {}) or {}).get("center_value", score_value))
            trap_center = float(dict(deviation_ops.get("trap", {}) or {}).get("center_value", trap_value))
            tensor_gradient = clamp01(
                abs(freq_dir) * phase_gradient_norm * 0.5
                + abs(amp_dir) * amplitude_gradient_norm * 0.5
            )
            deviation = abs(score_value - score_center) + abs(trap_value - trap_center)
            sweep_entries.append(
                {
                    "tag": f"F{freq_dir:+d}_A{amp_dir:+d}",
                    "deltas": deltas,
                    "score": score_value,
                    "trap": trap_value,
                    "coherence": coherence_value,
                    "curvature": curvature_value,
                    "tensor_gradient": tensor_gradient,
                    "deviation": deviation,
                    "cascade_interval": cascade_step_interval,
                }
            )
    sweep_entries.sort(
        key=lambda item: (
            item["coherence"] - item["trap"] - item["deviation"] * 0.05,
            item["tensor_gradient"],
        ),
        reverse=True,
    )
    best_sweep = sweep_entries[0] if sweep_entries else {
        "tag": "F+0_A+0",
        "deltas": {"F": 0.0, "A": 0.0, "I": 0.0, "V": 0.0},
        "score": 0.0,
        "trap": 0.0,
        "coherence": baseline_coherence,
        "curvature": 0.0,
        "tensor_gradient": 0.0,
        "deviation": 0.0,
    }

    global_theta = float(np.mean(theta_history[-1]))
    global_freq = float(np.mean(step_dominant_freqs[-1]))
    prev_state = {
        "yield_count": 0,
        "coherence_peak": baseline_coherence,
        "temporal_residual": 0.0,
        "clamp_pressure": amplitude_cap,
    }
    pulse_batches: list[dict[str, Any]] = []
    gate_coherence = float(collapse_gates.get("gate_coherence", 0.9972))
    gate_trap = float(collapse_gates.get("gate_trap", 0.157))

    for pulse_index in range(pulse_count):
        pulse_depth = adaptive_depth
        pulse_workers = adaptive_workers
        pulse_sweep = sweep_entries[pulse_index % max(len(sweep_entries), 1)]
        seed_material = (
            f"{quartet['f_code']:.6f}:{quartet['a_code']:.6f}:{quartet['i_code']:.6f}:"
            f"{quartet['v_code']:.6f}:{pulse_index}:{pulse_sweep['tag']}:{state.get('state_sig9', [])}"
        )
        seed_bytes = hashlib.sha256(seed_material.encode("ascii")).digest()
        header_hex, timestamp, nbits_hex = build_btc_header_from_seed(
            seed_bytes,
            pulse_index=pulse_index,
            difficulty_norm=difficulty_norm,
            quartet=quartet,
        )
        pulse_seed = int.from_bytes(seed_bytes[:4], byteorder="little", signed=False)
        target_profile = build_btc_target_profile(nbits_hex)
        target_hex = str(target_profile.get("target_hex", ""))
        target_profile["network_algorithm"] = build_btc_network_algorithm_profile(
            header_hex=header_hex,
            nbits_hex=nbits_hex,
            target_hex=target_hex,
        )
        network_target_hex = target_hex
        target_difficulty_window = clamp01(float(target_profile.get("difficulty_window", 0.5)))
        target_interval_windows = list(target_profile.get("interval_windows", []) or [target_difficulty_window])
        target_phase_windows = list(target_profile.get("phase_windows", []) or target_interval_windows)
        simulation_field_state = build_simulation_field_entry_state(
            lattice_calibration=silicon_calibration,
            fourier_kernel_fanout=fourier_kernel_fanout,
            target_profile=target_profile,
            pulse_sweep=pulse_sweep,
            pulse_index=pulse_index,
            previous_field_state=dict(prev_state.get("simulation_field_state", {}) or {}),
            encoded_event_model=encoded_event_model,
        )
        kernel_execution_event = dict(
            simulation_field_state.get("kernel_execution_event", {}) or {}
        )
        if not bool(simulation_field_state.get("entry_trigger", False)):
            pulse_state = make_sink_state(
                {
                    "pulse_id": pulse_index,
                    "worker_count": pulse_workers,
                    "nbits": nbits_hex,
                    "target_hex": target_hex,
                    "network_target_hex": network_target_hex,
                    "field_pressure": field_pressure,
                    "larger_field_exposure": larger_field_exposure,
                    "dominant_basin": dominant_basin_id,
                },
                {"amplitude_cap": amplitude_cap},
            )
            pulse_state["simulation_field_state"] = simulation_field_state
            pulse_state["kernel_execution_event"] = kernel_execution_event
            pulse_state["calibration_readiness"] = float(simulation_field_state.get("calibration_readiness", 0.0))
            pulse_state["entry_trigger"] = False
            pulse_state["interference_resonance"] = float(simulation_field_state.get("interference_accounting", 0.0))
            pulse_state["motif_consistency"] = float(simulation_field_state.get("motif_consistency", 0.0))
            pulse_state["motif_repeat_count"] = int(simulation_field_state.get("motif_repeat_count", 0))
            pulse_state["motif_energy"] = float(simulation_field_state.get("motif_energy", 0.0))
            pulse_batches.append(pulse_state)
            prev_state = pulse_state
            continue
        psi_encode = build_encoded_wave_state(
            quartet=quartet,
            pulse_sweep=pulse_sweep,
            lattice_calibration=silicon_calibration,
            simulation_field_state=simulation_field_state,
            deviation_ops=deviation_ops,
            baseline_frequency_norm=baseline_frequency_norm,
            amplitude_cap=lattice_amplitude_cap,
            temporal_persistence=temporal_persistence,
            prediction_confidence=prediction_confidence,
            residual_norm=residual_norm,
            coupling_norm=coupling_norm,
            pulse_index=pulse_index,
            bin_count=config.bin_count,
        )
        temporal_manifold = evolve_temporal_manifold_state(
            encoded_state=psi_encode,
            simulation_field_state=simulation_field_state,
            lattice_calibration=silicon_calibration,
            pulse_index=pulse_index,
            nesting_depth=pulse_depth,
            global_theta=global_theta,
            temporal_persistence=temporal_persistence,
            baseline_coherence=baseline_coherence,
            baseline_shared=baseline_shared,
            residual_norm=residual_norm,
            coupling_norm=coupling_norm,
        )
        effective_vector = project_effective_vector(temporal_manifold)
        manifold_diagnostics = evaluate_manifold_diagnostics(
            encoded_state=psi_encode,
            manifold_state=temporal_manifold,
            effective_vector=effective_vector,
            pulse_sweep=pulse_sweep,
            residual_norm=residual_norm,
            coupling_norm=coupling_norm,
        )
        feedback_seed = dict(simulation_field_state.get("gpu_pulse_feedback", {}) or {})
        injection_gpu_feedback = sample_gpu_pulse_feedback(
            pulse_index=pulse_index,
            previous_feedback=feedback_seed,
            feedback_context=build_vector_feedback_context(
                pulse_index=pulse_index,
                lattice_calibration=silicon_calibration,
                pulse_sweep=pulse_sweep,
                feedback_state=simulation_field_state,
                target_profile=target_profile,
                effective_vector=effective_vector,
                temporal_manifold=temporal_manifold,
                kernel_execution_event=kernel_execution_event,
                feedback_stage="injection",
            ),
        )
        injection_feedback_window_ms = float(
            injection_gpu_feedback.get("sampling_latency_ms", 0.0)
        )
        injection_gpu_feedback["pulse_roundtrip_latency_ms"] = float(
            injection_feedback_window_ms
        )
        injection_gpu_feedback["pulse_feedback_window_ms"] = float(
            injection_feedback_window_ms
        )
        injection_delta_feedback = build_gpu_pulse_delta_feedback(
            pulse_index=pulse_index,
            pre_feedback=feedback_seed,
            post_feedback=injection_gpu_feedback,
            cuda_kernel_telemetry={
                "dispatch_latency_ms": 0.0,
                "feedback_window_ms": float(injection_feedback_window_ms),
            },
        )
        simulation_field_state["gpu_pulse_feedback_injection"] = dict(
            injection_gpu_feedback
        )
        simulation_field_state["gpu_pulse_delta_feedback_injection"] = dict(
            injection_delta_feedback
        )
        integrate_gpu_feedback_into_field_state(
            simulation_field_state=simulation_field_state,
            gpu_feedback=injection_gpu_feedback,
            gpu_pulse_delta_feedback=injection_delta_feedback,
            blend=0.32,
        )
        kernel_execution_event["gpu_injection_feedback_source"] = str(
            injection_gpu_feedback.get("source", "fallback")
        )
        kernel_execution_event["gpu_injection_feedback_window_ms"] = float(
            injection_delta_feedback.get("feedback_window_ms", injection_feedback_window_ms)
        )
        kernel_execution_event["gpu_injection_response_gate"] = float(
            injection_delta_feedback.get("response_gate", 0.0)
        )
        kernel_execution_event["gpu_injection_observation_gap_ms"] = float(
            injection_delta_feedback.get("observation_gap_ms", 0.0)
        )
        kernel_execution_event["gpu_injection_observation_freshness_gate"] = float(
            injection_delta_feedback.get("observation_freshness_gate", 0.0)
        )
        vector_bias = float(effective_vector.get("coherence_bias", 0.0))
        temporal_confinement = float(effective_vector.get("t_eff", 0.0))
        baseline_frequency_hz = (
            1.0
            + (pulse_seed / float(0xFFFFFFFF)) * float(config.bin_count)
            + float(effective_vector.get("spatial_magnitude", 0.0)) * 2.75
            + float(silicon_calibration.get("frequency_bias", 0.0)) * 2.20
            + field_pressure * 1.35
        )
        interference_field = detect_interference_vector_field(
            config=config,
            step_dominant_freqs=step_dominant_freqs,
            theta_history=theta_history,
            mean_phase_lock_matrix=mean_phase_lock_matrix,
            amplitude_history=amplitude_history,
            shared_history=shared_history,
            coherence_history=coherence_history,
            packet_classes=packet_classes,
            tensor_gradient_samples=tensor_gradient_samples,
            lattice_calibration=silicon_calibration,
            fourier_kernel_fanout=fourier_kernel_fanout,
            simulation_field_state=simulation_field_state,
            temporal_manifold=temporal_manifold,
            effective_vector=effective_vector,
            target_profile=target_profile,
        )
        packet_vector_index = {
            int(item.get("packet_id", -1)): item
            for item in list(interference_field.get("packet_vectors", []) or [])
        }
        dominant_interference_vector = dict(interference_field.get("dominant_vector", {}) or {})
        field_vector_resonance = float(interference_field.get("field_resonance", 0.0))
        substrate_trace_state = update_substrate_trace_state(
            pulse_index=pulse_index,
            previous_trace_state=simulation_field_state.get("substrate_trace_state", {}),
            simulation_field_state=simulation_field_state,
            gpu_feedback=injection_gpu_feedback,
            gpu_pulse_delta_feedback=injection_delta_feedback,
            interference_field=interference_field,
            effective_vector=effective_vector,
            kernel_execution_event=kernel_execution_event,
            trace_label="injection_decode",
        )
        simulation_field_state["substrate_trace_state"] = dict(substrate_trace_state)
        substrate_trace_vram = sync_substrate_trace_state_to_vram(substrate_trace_state)
        simulation_field_state["substrate_trace_vram"] = dict(substrate_trace_vram)
        kernel_execution_event["trace_support"] = float(
            substrate_trace_state.get("trace_support", 0.0)
        )
        kernel_execution_event["trace_resonance"] = float(
            substrate_trace_state.get("trace_resonance", 0.0)
        )
        kernel_execution_event["trace_alignment"] = float(
            substrate_trace_state.get("trace_alignment", 0.0)
        )
        kernel_execution_event["trace_memory"] = float(
            substrate_trace_state.get("trace_memory", 0.0)
        )
        kernel_execution_event["trace_flux"] = float(
            substrate_trace_state.get("trace_flux", 0.0)
        )
        kernel_execution_event["trace_vram_resident"] = bool(
            substrate_trace_vram.get("resident", False)
        )
        kernel_execution_event["trace_vram_updates"] = int(
            substrate_trace_vram.get("update_count", 0)
        )
        update_compute_regime(
            simulation_field_state=simulation_field_state,
            kernel_execution_event=kernel_execution_event,
            freshness_gate=float(
                injection_delta_feedback.get("observation_freshness_gate", 0.0)
            ),
        )
        baseline_frequency_hz += float(simulation_field_state.get("field_frequency_bias", 0.0)) * 1.15
        kernel_temporal_controls = list(simulation_field_state.get("kernel_temporal_controls", []) or [])
        kernel_control_index = {
            int(item.get("kernel_id", -1)): dict(item)
            for item in kernel_temporal_controls
            if int(item.get("kernel_id", -1)) >= 0
        }
        phase_pressure_volume = clamp01(
            0.26 * field_pressure
            + 0.18 * lattice_amplitude_cap
            + 0.16 * float(simulation_field_state.get("field_amplitude_bias", 0.0))
            + 0.14 * float(simulation_field_state.get("target_gate", 0.0))
            + 0.14 * float(simulation_field_state.get("sequence_persistence_score", 0.0))
            + 0.12 * float(simulation_field_state.get("temporal_index_overlap", 0.0))
        )
        coupling_volume = clamp01(
            0.34 * float(simulation_field_state.get("voltage_frequency_flux", 0.0))
            + 0.34 * float(simulation_field_state.get("frequency_voltage_flux", 0.0))
            + 0.18 * float(simulation_field_state.get("kernel_delta_flux_mean", 0.0))
            + 0.14 * float(simulation_field_state.get("kernel_delta_gate_mean", 0.0))
        )
        kernel_volume = clamp01(
            0.40 * float(simulation_field_state.get("kernel_drive_mean", 0.0))
            + 0.30 * float(simulation_field_state.get("kernel_balance_mean", 0.0))
            + 0.30 * float(simulation_field_state.get("kernel_harmonic_resonance_mean", 0.0))
        )
        pulse_volume_gain = clamp01(
            0.42 * phase_pressure_volume
            + 0.36 * coupling_volume
            + 0.22 * kernel_volume
        )
        pulse_carrier_count = max(
            base_carrier_count,
            min(
                128,
                int(
                    round(
                        float(base_carrier_count)
                        + 18.0
                        + 16.0 * phase_pressure_volume
                        + 18.0 * coupling_volume
                        + 12.0 * kernel_volume
                    )
                ),
            ),
        )
        pulse_workers = max(
            adaptive_workers,
            min(
                max_worker_count,
                int(
                    round(
                        float(adaptive_workers)
                        + 1.0
                        + 2.0 * phase_pressure_volume
                        + 3.0 * coupling_volume
                        + 2.0 * kernel_volume
                    )
                ),
            ),
        )
        default_kernel_control = {
            "kernel_id": -1,
            "control_state": float(simulation_field_state.get("kernel_balance_mean", 0.0)),
            "harmonic_resonance": float(simulation_field_state.get("kernel_harmonic_resonance_mean", 0.0)),
            "retro_gain": float(simulation_field_state.get("kernel_retro_gain_mean", 0.0)),
            "delta_gate": float(simulation_field_state.get("kernel_delta_gate_mean", 0.0)),
            "delta_memory_target": float(simulation_field_state.get("kernel_delta_memory_mean", 0.0)),
            "delta_flux_target": float(simulation_field_state.get("kernel_delta_flux_mean", 0.0)),
            "delta_phase_alignment": float(simulation_field_state.get("kernel_delta_phase_alignment_mean", 0.0)),
            "retro_temporal_target": float(simulation_field_state.get("kernel_retro_target_mean", 0.0)),
            "resonance_alignment": float(simulation_field_state.get("kernel_resonance_alignment_mean", 0.0)),
            "resonance_center_turns": float(simulation_field_state.get("feedback_phase_anchor_turns", 0.0)),
            "resonance_bandwidth": float(simulation_field_state.get("kernel_resonance_bandwidth_mean", 0.0)),
            "kernel_drive": float(simulation_field_state.get("kernel_drive_mean", 0.0)),
            "harmonic_orders": [1] * len(GPU_PULSE_DOF_LABELS),
            "harmonic_weights": [1.0] * len(GPU_PULSE_DOF_LABELS),
            "ancilla_commit_gate": float(simulation_field_state.get("ancilla_commit_gate", 0.0)),
            "ancilla_activation_gate": float(simulation_field_state.get("ancilla_activation_gate", 0.0)),
            "ancilla_current_norm": float(simulation_field_state.get("ancilla_current_norm", 0.0)),
            "ancilla_flux_norm": float(simulation_field_state.get("ancilla_flux", 0.0)),
            "ancilla_phase_alignment": float(simulation_field_state.get("ancilla_phase_alignment", 0.0)),
            "ancilla_convergence": float(simulation_field_state.get("ancilla_convergence", 0.0)),
            "ancilla_temporal_persistence": float(
                simulation_field_state.get("ancilla_temporal_persistence", 0.0)
            ),
            "ancilla_tension_headroom": float(
                simulation_field_state.get("ancilla_tension_headroom", 0.0)
            ),
            "ancilla_gradient_headroom": float(
                simulation_field_state.get("ancilla_gradient_headroom", 0.0)
            ),
        }

        btc_network_force_vector = np.array(
            list(simulation_field_state.get("btc_network_force_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if btc_network_force_vector.shape[0] != 4:
            btc_network_force_vector = np.zeros(4, dtype=np.float64)
        btc_network_phase_turns = [
            wrap_turns(float(value))
            for value in list(
                simulation_field_state.get("btc_network_phase_turns", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0]
            )
        ]
        if len(btc_network_phase_turns) < 4:
            btc_network_phase_turns = [0.0, 0.0, 0.0, 0.0]
        btc_network_phase_pressure = clamp01(float(simulation_field_state.get("btc_network_phase_pressure", 0.0)))
        btc_network_amplitude_pressure = clamp01(
            float(simulation_field_state.get("btc_network_amplitude_pressure", 0.0))
        )
        btc_network_algorithm_bias = clamp01(float(simulation_field_state.get("btc_network_algorithm_bias", 0.0)))
        coherent_noise_axis_vector = [
            float(value)
            for value in list(substrate_trace_state.get("coherent_noise_axis_vector", simulation_field_state.get("coherent_noise_axis_vector", [0.0, 0.0, 0.0, 0.0])) or [0.0, 0.0, 0.0, 0.0])
        ]
        noise_resonance_nodes = [
            float(value)
            for value in list(substrate_trace_state.get("noise_resonance_nodes", simulation_field_state.get("noise_resonance_nodes", [0.0, 0.0, 0.0, 0.0])) or [0.0, 0.0, 0.0, 0.0])
        ]
        drift_compensation_vector = [
            float(value)
            for value in list(substrate_trace_state.get("drift_compensation_vector", simulation_field_state.get("drift_compensation_vector", [0.0, 0.0, 0.0, 0.0])) or [0.0, 0.0, 0.0, 0.0])
        ]
        relative_spatial_field = [
            float(value)
            for value in list(substrate_trace_state.get("trace_relative_spatial_field", simulation_field_state.get("relative_spatial_field", [0.0, 0.0, 0.0, 0.0])) or [0.0, 0.0, 0.0, 0.0])
        ]
        projected_temporal_dof_vector = [
            float(value)
            for value in list(simulation_field_state.get("projected_temporal_dof_vector", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])
        ]
        candidate_pool: list[dict[str, Any]] = []
        for packet_idx in range(step_dominant_freqs.shape[1]):
            freq_vector = step_dominant_freqs[-1, packet_idx]
            freq_norm = np.array(
                [
                    clamp01(abs(float(freq_vector[0])) / max(float(config.bin_count), 1.0)),
                    clamp01(abs(float(freq_vector[1])) / max(float(config.bin_count), 1.0)),
                    clamp01(abs(float(freq_vector[2])) / max(float(config.bin_count), 1.0)),
                ],
                dtype=np.float64,
            )
            coherence_score = float(np.mean(coherence_history[:, packet_idx]))
            shared_score = float(np.mean(shared_history[:, packet_idx]))
            curvature_score = float(np.mean(curvature_history[:, packet_idx]))
            amplitude_norm = clamp01(float(amplitude_history[-1, packet_idx]) / config.max_amplitude)
            theta = float(theta_history[-1, packet_idx])
            phase_row = mean_phase_lock_matrix[packet_idx].copy()
            if packet_idx < len(phase_row):
                phase_row[packet_idx] = 0.0
            crosstalk = float(np.max(phase_row))
            target_interval = float(target_interval_windows[packet_idx % len(target_interval_windows)])
            target_phase_window = float(target_phase_windows[packet_idx % len(target_phase_windows)])
            detected_vector = dict(packet_vector_index.get(packet_idx, {}) or {})
            interference_vector = np.array(
                list(
                    detected_vector.get(
                        "vector",
                        [
                            float(effective_vector.get("x", 0.0)),
                            float(effective_vector.get("y", 0.0)),
                            float(effective_vector.get("z", 0.0)),
                            float(effective_vector.get("t_eff", 0.0)),
                        ],
                    )
                ),
                dtype=np.float64,
            )
            if interference_vector.shape[0] != 4:
                interference_vector = np.array(
                    [
                        float(effective_vector.get("x", 0.0)),
                        float(effective_vector.get("y", 0.0)),
                        float(effective_vector.get("z", 0.0)),
                        float(effective_vector.get("t_eff", 0.0)),
                    ],
                    dtype=np.float64,
                )
            interference_resonance = clamp01(
                float(detected_vector.get("resonance", field_vector_resonance))
            )
            detected_target_alignment = clamp01(
                float(detected_vector.get("target_alignment", target_phase_window))
            )
            detected_phase_signature = clamp01(
                float(detected_vector.get("phase_signature", target_phase_window))
            )
            detected_cascade_phase = float(detected_vector.get("cascade_phase", theta))
            detected_cascade_activation = clamp01(
                float(detected_vector.get("cascade_activation", 0.0))
            )
            detected_row_activation = clamp01(
                float(detected_vector.get("row_activation", 0.0))
            )
            detected_row_coherence = clamp01(
                float(detected_vector.get("row_coherence", 0.0))
            )
            detected_motif_alignment = clamp01(
                float(detected_vector.get("motif_alignment", 0.0))
            )
            detected_trace_alignment = clamp01(
                float(detected_vector.get("trace_alignment", 0.0))
            )
            detected_btc_force_alignment = clamp01(
                float(detected_vector.get("btc_force_alignment", 0.0))
            )
            detected_btc_phase_pressure = clamp01(
                float(detected_vector.get("btc_phase_pressure", btc_network_phase_pressure))
            )
            detected_neighbor_count = int(detected_vector.get("neighbor_count", 0))
            interference_carrier_bias = int(detected_vector.get("carrier_bias", 0)) & 0xFFFFFFFF
            if packet_idx in kernel_control_index:
                kernel_control = dict(kernel_control_index[packet_idx])
            elif kernel_temporal_controls:
                kernel_control = dict(kernel_temporal_controls[packet_idx % len(kernel_temporal_controls)])
            else:
                kernel_control = dict(default_kernel_control)
            kernel_control_id = int(kernel_control.get("kernel_id", packet_idx))
            kernel_balance_score = clamp01(float(kernel_control.get("control_state", 0.0)))
            harmonic_resonance_score = clamp01(float(kernel_control.get("harmonic_resonance", 0.0)))
            retro_temporal_gain = clamp01(float(kernel_control.get("retro_gain", 0.0)))
            kernel_delta_gate = clamp01(float(kernel_control.get("delta_gate", 0.0)))
            kernel_delta_memory = clamp01(float(kernel_control.get("delta_memory_target", 0.0)))
            kernel_delta_flux = clamp01(float(kernel_control.get("delta_flux_target", 0.0)))
            kernel_delta_phase_alignment = clamp01(float(kernel_control.get("delta_phase_alignment", 0.0)))
            retro_temporal_target = clamp01(float(kernel_control.get("retro_temporal_target", 0.0)))
            retro_temporal_gain = clamp01(0.72 * retro_temporal_gain + 0.28 * retro_temporal_target)
            kernel_resonance_alignment = clamp01(float(kernel_control.get("resonance_alignment", 0.0)))
            kernel_resonance_center_turns = wrap_turns(float(kernel_control.get("resonance_center_turns", target_phase_window)))
            kernel_resonance_bandwidth = clamp01(float(kernel_control.get("resonance_bandwidth", 0.0)))
            kernel_drive = clamp01(float(kernel_control.get("kernel_drive", 0.0)))
            ancilla_commit_gate = clamp01(float(kernel_control.get("ancilla_commit_gate", 0.0)))
            ancilla_activation_gate = clamp01(float(kernel_control.get("ancilla_activation_gate", 0.0)))
            ancilla_current_norm = clamp01(float(kernel_control.get("ancilla_current_norm", 0.0)))
            ancilla_flux_norm = clamp01(float(kernel_control.get("ancilla_flux_norm", 0.0)))
            ancilla_phase_alignment = clamp01(float(kernel_control.get("ancilla_phase_alignment", 0.0)))
            ancilla_convergence = clamp01(float(kernel_control.get("ancilla_convergence", 0.0)))
            ancilla_temporal_persistence = clamp01(
                float(kernel_control.get("ancilla_temporal_persistence", 0.0))
            )
            ancilla_tension_headroom = clamp01(
                float(kernel_control.get("ancilla_tension_headroom", 0.0))
            )
            ancilla_gradient_headroom = clamp01(
                float(kernel_control.get("ancilla_gradient_headroom", 0.0))
            )
            harmonic_orders = [
                max(1, int(value))
                for value in list(kernel_control.get("harmonic_orders", []) or [1] * len(GPU_PULSE_DOF_LABELS))
            ]
            harmonic_weights = [
                float(value)
                for value in list(kernel_control.get("harmonic_weights", []) or [1.0] * len(harmonic_orders))
            ]
            delta_theta = abs(theta - global_theta)
            delta_freq = abs(float(np.mean(freq_vector)) - global_freq) / max(float(config.bin_count), 1.0)
            temporal_weight = clamp01(
                math.exp(-delta_theta)
                * math.exp(-delta_freq)
                * (0.5 + 0.5 * coherence_score)
                * (0.72 + 0.28 * temporal_confinement)
                * (0.60 + 0.40 * target_interval)
                * (0.64 + 0.36 * target_difficulty_window)
                * (0.72 + 0.28 * interference_resonance)
                * (0.70 + 0.30 * detected_cascade_activation)
                * (0.72 + 0.28 * float(simulation_field_state.get("target_gate", 0.0)))
                * (0.70 + 0.30 * detected_row_activation)
                * (0.72 + 0.28 * detected_motif_alignment)
                * (0.70 + 0.30 * kernel_balance_score)
                * (0.68 + 0.32 * harmonic_resonance_score)
                * (0.68 + 0.32 * retro_temporal_gain)
                * (0.70 + 0.30 * kernel_delta_gate)
                * (0.70 + 0.30 * kernel_delta_phase_alignment)
                * (0.68 + 0.32 * ancilla_convergence)
                * (0.68 + 0.32 * ancilla_phase_alignment)
                * (0.66 + 0.34 * ancilla_commit_gate)
            )
            curvature_center = float(
                dict(deviation_ops.get("curvature", {}) or {}).get("center_value", curvature_score)
            )
            trap_ratio = clamp01(
                abs(curvature_score - curvature_center) / max(abs(curvature_center), 1.0)
            )
            violation = clamp01(max(0.0, gate_trap - shared_score))
            class_row = packet_class_index.get(packet_idx, {})
            tensor_row = tensor_gradient_index.get(packet_idx, {})
            packet_phase_gradient = clamp01(
                float(
                    np.mean(
                        np.abs(np.array(tensor_row.get("phase_gradient", [0.0, 0.0, 0.0]), dtype=np.float64))
                    )
                )
                / 0.25
            )
            packet_amplitude_gradient = clamp01(
                float(
                    np.mean(
                        np.abs(np.array(tensor_row.get("amplitude_gradient", [0.0, 0.0, 0.0]), dtype=np.float64))
                    )
                )
                / 0.08
            )
            affinity_match = 1.0 if str(class_row.get("classification", "shared")) == dominant_basin_affinity else 0.68
            basin_alignment = clamp01(
                0.42 * float(dominant_basin.get("depth", 0.0))
                + 0.24 * affinity_match
                + 0.18 * packet_phase_gradient
                + 0.16 * packet_amplitude_gradient
            )
            target_amplitude_cap = clamp01(
                0.52 * lattice_amplitude_cap
                + 0.24 * amplitude_cap
                + 0.16 * target_interval
                + 0.08 * target_difficulty_window
                + 0.08 * interference_resonance
                + 0.06 * detected_target_alignment
                + 0.06 * detected_cascade_activation
                + 0.04 * detected_row_activation
                + 0.04 * detected_motif_alignment
                + 0.08 * float(simulation_field_state.get("field_amplitude_bias", 0.0))
                + 0.08 * kernel_balance_score
                + 0.06 * harmonic_resonance_score
                + 0.04 * retro_temporal_gain
                + 0.05 * kernel_delta_gate
                + 0.04 * kernel_delta_memory
                + 0.04 * ancilla_current_norm
                + 0.03 * ancilla_tension_headroom
            )
            clamp_pressure = clamp_band(amplitude_norm, target_amplitude_cap, trap_ratio, violation)
            if clamp_pressure > target_amplitude_cap + 0.16 + temporal_confinement * 0.08 + field_pressure * 0.06:
                continue

            effective_alignment = clamp01(
                1.0
                - (
                    abs(float(freq_norm[0]) - (float(effective_vector.get("x", 0.0)) * 0.5 + 0.5))
                    + abs(float(freq_norm[1]) - (float(effective_vector.get("y", 0.0)) * 0.5 + 0.5))
                    + abs(float(freq_norm[2]) - (float(effective_vector.get("z", 0.0)) * 0.5 + 0.5))
                )
                / 3.0
            )
            interference_freq_alignment = clamp01(
                1.0
                - (
                    abs(float(freq_norm[0]) - (float(interference_vector[0]) * 0.5 + 0.5))
                    + abs(float(freq_norm[1]) - (float(interference_vector[1]) * 0.5 + 0.5))
                    + abs(float(freq_norm[2]) - (float(interference_vector[2]) * 0.5 + 0.5))
                )
                / 3.0
            )
            vector_alignment = clamp01(
                0.42 * effective_alignment
                + 0.34 * interference_freq_alignment
                + 0.24 * clamp01(float(detected_vector.get("vector_alignment", 0.0)))
            )

            cluster_id = "%s-%02d-%02d" % (
                str(class_row.get("classification", "shared")),
                int(class_row.get("group_id", packet_idx)),
                int(round((0.5 * crosstalk + 0.5 * float(effective_vector.get("spatial_magnitude", 0.0))) * 10.0)),
            )
            cluster_root_id = (
                f"{cluster_id}-{dominant_basin_id}"
                f"-v{int(round(interference_resonance * 10.0)):02d}"
                f"-m{int(round(detected_motif_alignment * 10.0)):02d}"
                f"-k{int(kernel_control_id) & 0xFF:02d}"
            )
            for carrier_idx in range(pulse_carrier_count):
                harmonic_slot = (carrier_idx + packet_idx + pulse_index) % max(len(harmonic_orders), 1)
                harmonic_order = max(1, int(harmonic_orders[harmonic_slot % len(harmonic_orders)]))
                harmonic_weight = clamp01(float(harmonic_weights[harmonic_slot % len(harmonic_weights)]))
                harmonic_carrier_gain = clamp01(
                    0.64 * harmonic_weight
                    + 0.36 * harmonic_resonance_score
                )
                retro_carrier_gain = clamp01(
                    0.76 * retro_temporal_gain
                    + 0.24 * (1.0 / float(harmonic_order))
                )
                kernel_carrier_phase = wrap_turns(
                    kernel_resonance_center_turns
                    + float(harmonic_slot) / max(float(len(harmonic_orders)), 1.0)
                    * (0.12 + 0.22 * kernel_resonance_bandwidth)
                    + kernel_drive * 0.09
                )
                kernel_phase_alignment = clamp01(
                    0.58 * turn_alignment(kernel_carrier_phase, target_phase_window)
                    + 0.42 * kernel_resonance_alignment
                )
                target_carrier_window = float(
                    target_interval_windows[
                        (packet_idx + carrier_idx + pulse_index) % len(target_interval_windows)
                    ]
                )
                phase_length_event = build_phase_length_event_profile(
                    target_profile=target_profile,
                    pulse_index=pulse_index,
                    packet_idx=packet_idx,
                    carrier_idx=carrier_idx,
                    simulation_field_state=simulation_field_state,
                    target_phase_window=target_phase_window,
                    target_carrier_window=target_carrier_window,
                    detected_phase_signature=detected_phase_signature,
                    detected_row_activation=detected_row_activation,
                    detected_motif_alignment=detected_motif_alignment,
                    detected_cascade_activation=detected_cascade_activation,
                    field_amplitude_bias=float(simulation_field_state.get("field_amplitude_bias", 0.0)),
                    target_amplitude_cap=target_amplitude_cap,
                    kernel_balance_score=kernel_balance_score,
                    harmonic_resonance_score=harmonic_resonance_score,
                    retro_temporal_gain=retro_carrier_gain,
                    kernel_phase_alignment=kernel_phase_alignment,
                    kernel_delta_gate=kernel_delta_gate,
                    kernel_delta_memory=kernel_delta_memory,
                    kernel_delta_flux=kernel_delta_flux,
                    kernel_delta_phase_alignment=kernel_delta_phase_alignment,
                )
                phase_length_index = int(phase_length_event.get("phase_length_index", 0))
                phase_length_window = clamp01(float(phase_length_event.get("phase_length_window", target_phase_window)))
                phase_length_span = clamp01(float(phase_length_event.get("phase_length_span", 0.0)))
                phase_length_pressure = clamp01(float(phase_length_event.get("phase_length_pressure", 0.0)))
                phase_alignment_score = clamp01(
                    float(phase_length_event.get("phase_alignment_score", phase_length_pressure))
                )
                phase_confinement_cost = clamp01(
                    float(
                        phase_length_event.get(
                            "phase_confinement_cost", 1.0 - phase_alignment_score
                        )
                    )
                )
                phase_amplitude_cap = clamp01(float(phase_length_event.get("phase_amplitude_cap", target_amplitude_cap)))
                phase_amplitude_pressure = clamp01(
                    float(phase_length_event.get("phase_amplitude_pressure", phase_length_pressure))
                )
                amplitude_phase_pressure = clamp01(
                    float(phase_length_event.get("amplitude_phase_pressure", phase_amplitude_pressure))
                )
                btc_phase_pressure = clamp01(
                    float(phase_length_event.get("btc_phase_pressure", btc_network_phase_pressure))
                )
                btc_phase_alignment = clamp01(
                    float(phase_length_event.get("btc_phase_alignment", btc_phase_pressure))
                )
                btc_phase_cost = clamp01(
                    float(phase_length_event.get("btc_phase_cost", 1.0 - btc_phase_alignment))
                )
                btc_force_alignment = clamp01(
                    float(phase_length_event.get("btc_force_alignment", detected_btc_force_alignment))
                )
                temporal_sequence_index = int(phase_length_event.get("temporal_sequence_index", pulse_index))
                temporal_sequence_length = max(
                    1,
                    int(phase_length_event.get("temporal_sequence_length", simulation_field_state.get("temporal_sequence_length", 1))),
                )
                temporal_index_overlap = clamp01(float(phase_length_event.get("temporal_index_overlap", 0.0)))
                sequence_persistence_score = clamp01(
                    float(phase_length_event.get("sequence_persistence_score", 0.0))
                )
                candidate_voltage_frequency_flux = clamp01(
                    float(phase_length_event.get("voltage_frequency_flux", 0.0))
                )
                candidate_frequency_voltage_flux = clamp01(
                    float(
                        phase_length_event.get(
                            "frequency_voltage_flux",
                            simulation_field_state.get("frequency_voltage_flux", 0.0),
                        )
                    )
                )
                event_delta_gate = clamp01(float(phase_length_event.get("retro_delta_gate", kernel_delta_gate)))
                event_delta_memory = clamp01(float(phase_length_event.get("retro_delta_memory", kernel_delta_memory)))
                event_delta_flux = clamp01(float(phase_length_event.get("retro_delta_flux", kernel_delta_flux)))
                event_delta_phase_alignment = clamp01(
                    float(phase_length_event.get("retro_delta_phase_alignment", kernel_delta_phase_alignment))
                )
                target_phase_anchor = clamp01(
                    0.26 * target_phase_window
                    + 0.28 * detected_phase_signature
                    + 0.18 * detected_target_alignment
                    + 0.10 * clamp01(0.5 + 0.5 * math.sin(detected_cascade_phase + carrier_idx * cascade_step_interval * math.pi))
                    + 0.20
                    * float(
                        target_phase_windows[
                            (packet_idx + carrier_idx + pulse_index) % len(target_phase_windows)
                        ]
                    )
                    + 0.08 * phase_length_window
                    + 0.08 * detected_row_activation
                    + 0.06 * detected_motif_alignment
                    + 0.08 * phase_alignment_score
                    + 0.06 * phase_amplitude_pressure
                    + 0.04 * amplitude_phase_pressure
                    + 0.05 * sequence_persistence_score
                    + 0.04 * temporal_index_overlap
                    + 0.03 * candidate_voltage_frequency_flux
                    + 0.04 * detected_trace_alignment
                    + 0.04 * ancilla_phase_alignment
                    + 0.03 * ancilla_temporal_persistence
                    + 0.06 * btc_phase_alignment
                    + 0.04 * btc_force_alignment
                    + 0.02 * (1.0 - phase_confinement_cost)
                    + 0.02 * (1.0 - btc_phase_cost)
                    + 0.02 * detected_btc_phase_pressure
                    + 0.05 * kernel_phase_alignment
                    + 0.05 * harmonic_carrier_gain
                    + 0.04 * retro_carrier_gain
                    + 0.04 * event_delta_gate
                    + 0.03 * event_delta_phase_alignment
                    + 0.03 * float(btc_network_force_vector[carrier_idx % 4])
                    + 0.02 * turn_alignment(kernel_carrier_phase, btc_network_phase_turns[carrier_idx % 4])
                )
                phase_confinement_pressure = clamp_band(
                    amplitude_norm,
                    phase_amplitude_cap,
                    trap_ratio,
                    violation,
                )
                if phase_confinement_pressure > phase_amplitude_cap + 0.08 + amplitude_phase_pressure * 0.08:
                    continue
                candidate_phase_confinement_cost = clamp01(
                    0.46 * phase_confinement_cost
                    + 0.32 * phase_confinement_pressure
                    + 0.22 * clamp_pressure
                )
                candidate_btc_phase_cost = clamp01(
                    0.72 * btc_phase_cost
                    + 0.16 * candidate_phase_confinement_cost
                    + 0.12 * (1.0 - btc_force_alignment)
                )
                interference_component = float(interference_vector[carrier_idx % 4])
                nonce_state = (
                    pulse_seed
                    ^ (packet_idx * 0x9E3779B1)
                    ^ (carrier_idx * 0x85EBCA77)
                    ^ interference_carrier_bias
                    ^ int((detected_cascade_activation + cascade_step_interval) * 262139.0)
                    ^ int((interference_component + 1.0) * 131071.0)
                    ^ int((detected_row_activation + detected_motif_alignment) * 524287.0)
                    ^ int((phase_alignment_score + phase_length_window) * 262147.0)
                    ^ int((phase_amplitude_pressure + amplitude_phase_pressure) * 262171.0)
                    ^ int((temporal_sequence_index + 1) * 65537.0)
                    ^ ((int(kernel_control_id) + 1) * 0x27D4EB2D)
                    ^ int((kernel_balance_score + harmonic_resonance_score) * 524309.0)
                    ^ int((kernel_phase_alignment + harmonic_carrier_gain + retro_carrier_gain) * 262153.0)
                    ^ int((event_delta_gate + event_delta_memory + event_delta_flux) * 262139.0)
                    ^ int(
                        (
                            ancilla_convergence
                            + ancilla_phase_alignment
                            + ancilla_flux_norm
                            + ancilla_temporal_persistence
                        )
                        * 262147.0
                    )
                    ^ int(
                        (
                            candidate_voltage_frequency_flux
                            + candidate_frequency_voltage_flux
                            + sequence_persistence_score
                            + btc_phase_pressure
                            + btc_force_alignment
                        )
                        * 262151.0
                    )
                    ^ int((baseline_frequency_hz + float(freq_vector[carrier_idx % 3])) * 4096.0)
                    ^ int((float(btc_network_force_vector[carrier_idx % 4]) + btc_network_algorithm_bias) * 262147.0)
                ) & 0xFFFFFFFF
                for depth_idx in range(pulse_depth):
                    carrier_phase = int(
                        (
                            float(freq_vector[depth_idx % 3])
                            + baseline_frequency_hz
                            + detected_phase_signature
                            + interference_resonance
                            + detected_cascade_activation
                            + detected_motif_alignment
                            + phase_length_pressure
                            + sequence_persistence_score
                            + temporal_index_overlap
                            + candidate_voltage_frequency_flux
                            + phase_amplitude_pressure
                            + btc_phase_pressure
                            + unison_step_interval
                            + kernel_drive
                            + harmonic_carrier_gain
                            + retro_carrier_gain
                            + ancilla_phase_alignment
                            + ancilla_commit_gate
                            + float(btc_network_force_vector[depth_idx % 4])
                        )
                        * 65535.0
                    ) & 0xFFFFFFFF
                    nonce_state = (
                        nonce_state * 1664525
                        + 1013904223
                        + carrier_phase
                        + int(
                            (
                                temporal_weight
                                + crosstalk
                                + target_carrier_window
                                + interference_resonance
                                + detected_target_alignment
                                + lateral_step_interval
                                + detected_cascade_activation
                                + detected_row_activation
                                + detected_motif_alignment
                                + phase_length_pressure
                                + amplitude_phase_pressure
                                + sequence_persistence_score
                                + temporal_index_overlap
                                + candidate_frequency_voltage_flux
                                + btc_force_alignment
                                + kernel_balance_score
                                + harmonic_carrier_gain
                                + retro_carrier_gain
                                + event_delta_gate
                                + event_delta_memory
                                + ancilla_convergence
                                + ancilla_flux_norm
                            )
                            * 1000000.0
                        )
                    ) & 0xFFFFFFFF
                    nonce_state ^= (pulse_seed >> (depth_idx % 13))
                    nonce_state = (
                        nonce_state
                        + int(
                            (
                                coherence_score
                                + shared_score
                                + detected_target_alignment
                                + interference_resonance
                                + detected_cascade_activation
                                + detected_row_coherence
                                + phase_length_window
                                + candidate_voltage_frequency_flux
                                + btc_phase_pressure
                                + kernel_phase_alignment
                                + harmonic_resonance_score
                                + event_delta_phase_alignment
                                + event_delta_flux
                                + ancilla_temporal_persistence
                                + ancilla_activation_gate
                                + float(btc_network_force_vector[(depth_idx + carrier_idx) % 4])
                            )
                            * 65535.0
                        )
                        * (depth_idx + 1)
                    ) & 0xFFFFFFFF
                trajectory_phase = clamp01(
                    abs(
                        math.sin(
                            theta
                            + carrier_idx * 0.071
                            + pulse_index * 0.33
                            + temporal_sequence_index * 0.041
                            + kernel_carrier_phase * math.tau * 0.18
                        )
                    )
                )
                decoded_nonce = decode_temporal_nonce(
                    mixed_nonce_state=int(nonce_state),
                    pulse_seed=int(pulse_seed),
                    packet_idx=int(packet_idx),
                    carrier_idx=int(carrier_idx),
                    pulse_index=int(pulse_index),
                    harmonic_order=int(harmonic_order),
                    temporal_sequence_index=int(temporal_sequence_index),
                    temporal_sequence_length=int(temporal_sequence_length),
                    sequence_persistence_score=float(sequence_persistence_score),
                    temporal_index_overlap=float(temporal_index_overlap),
                    target_phase_pressure=float(btc_phase_pressure),
                    target_phase_alignment=float(btc_phase_alignment),
                    target_phase_cost=float(candidate_btc_phase_cost),
                    target_phase_anchor=float(target_phase_anchor),
                    target_flux_alignment=float(
                        clamp01(
                            0.58 * candidate_voltage_frequency_flux
                            + 0.42 * candidate_frequency_voltage_flux
                        )
                    ),
                    target_hex=str(network_target_hex),
                    electron_ring_stability=float(
                        clamp01(
                            0.32 * sequence_persistence_score
                            + 0.24 * temporal_index_overlap
                            + 0.16 * phase_alignment_score
                            + 0.14 * kernel_phase_alignment
                            + 0.14 * ancilla_phase_alignment
                        )
                    ),
                    electron_phase_anchor=float(kernel_carrier_phase),
                    decode_terms=[
                        float(target_phase_anchor),
                        float(trajectory_phase),
                        float(phase_alignment_score),
                        float(phase_length_pressure),
                        float(phase_length_window),
                        float(target_carrier_window),
                        float(phase_amplitude_pressure),
                        float(amplitude_phase_pressure),
                        float(candidate_voltage_frequency_flux),
                        float(candidate_frequency_voltage_flux),
                        float(sequence_persistence_score),
                        float(temporal_index_overlap),
                        float(detected_target_alignment),
                        float(detected_trace_alignment),
                        float(interference_resonance),
                        float(field_vector_resonance),
                        float(btc_force_alignment),
                        float(btc_phase_alignment),
                        float(kernel_phase_alignment),
                        float(kernel_balance_score),
                        float(harmonic_carrier_gain),
                        float(retro_carrier_gain),
                        float(event_delta_gate),
                        float(event_delta_memory),
                        float(event_delta_flux),
                        float(event_delta_phase_alignment),
                        float(ancilla_phase_alignment),
                        float(ancilla_temporal_persistence),
                        float(ancilla_flux_norm),
                        float(ancilla_convergence),
                    ],
                    noise_resonance_nodes=noise_resonance_nodes,
                    drift_compensation_vector=drift_compensation_vector,
                    relative_spatial_field=relative_spatial_field,
                    projected_temporal_dof_vector=projected_temporal_dof_vector,
                )
                nonce_state = int(decoded_nonce.get("nonce", nonce_state)) & 0xFFFFFFFF
                decode_phase_integrity = clamp01(
                    float(
                        decoded_nonce.get(
                            "decode_phase_integrity_score",
                            decoded_nonce.get("decode_integrity_score", 0.0),
                        )
                    )
                )
                decode_phase_alignment = clamp01(
                    float(decoded_nonce.get("decode_phase_delta_alignment", 0.0))
                )
                decode_target_ring_alignment = clamp01(
                    float(decoded_nonce.get("decode_target_ring_alignment", 0.0))
                )
                decode_target_prefix_lock = clamp01(
                    float(decoded_nonce.get("decode_target_prefix_lock", 0.0))
                )
                decode_target_prefix_vector_alignment = clamp01(
                    float(decoded_nonce.get("decode_target_prefix_vector_alignment", 0.0))
                )
                decode_target_prefix_vector_phase_pressure = clamp01(
                    float(decoded_nonce.get("decode_target_prefix_vector_phase_pressure", 0.0))
                )
                decode_prefix_asymptote_pressure = clamp01(
                    float(decoded_nonce.get("decode_prefix_asymptote_pressure", 1.0))
                )
                decode_phase_orbital_alignment = clamp01(
                    float(decoded_nonce.get("decode_phase_orbital_alignment", 0.0))
                )
                decode_phase_orbital_resonance = clamp01(
                    float(decoded_nonce.get("decode_phase_orbital_resonance", 0.0))
                )
                decode_phase_orbital_stability = clamp01(
                    float(decoded_nonce.get("decode_phase_orbital_stability", 0.0))
                )
                candidate_phase_confinement_cost = clamp01(
                    0.62 * candidate_phase_confinement_cost
                    + 0.16 * (1.0 - decode_phase_alignment)
                    + 0.12 * (1.0 - decode_target_ring_alignment)
                    + 0.08 * (1.0 - decode_target_prefix_vector_alignment)
                    + 0.06 * (1.0 - decode_target_prefix_vector_phase_pressure)
                    + 0.08 * (1.0 - decode_phase_orbital_alignment)
                    + 0.06 * (1.0 - decode_phase_orbital_stability)
                    + 0.10 * decode_prefix_asymptote_pressure
                )
                candidate_btc_phase_cost = clamp01(
                    0.70 * candidate_btc_phase_cost
                    + 0.14 * (1.0 - decode_phase_integrity)
                    + 0.08 * (1.0 - decode_target_ring_alignment)
                    + 0.06 * (1.0 - decode_target_prefix_vector_alignment)
                    + 0.04 * (1.0 - decode_target_prefix_vector_phase_pressure)
                    + 0.05 * (1.0 - decode_phase_orbital_resonance)
                    + 0.08 * decode_prefix_asymptote_pressure
                )
                if (
                    float(decoded_nonce.get("decode_integrity_score", 0.0)) < 0.18
                    and decode_phase_integrity < 0.22
                    and decode_phase_alignment < 0.18
                    and int(decoded_nonce.get("decode_unique_symbol_count", 0)) <= 2
                    and float(decoded_nonce.get("decode_longest_run_ratio", 1.0)) >= 0.50
                ):
                    continue
                if (
                    decode_prefix_asymptote_pressure > 0.92
                    and decode_target_prefix_lock < 0.08
                    and decode_target_prefix_vector_alignment < 0.12
                ):
                    continue
                target_alignment = clamp01(
                    0.46 * (1.0 - abs(trajectory_phase - target_phase_anchor))
                    + 0.34 * detected_target_alignment
                    + 0.08 * phase_alignment_score
                    + 0.04 * phase_amplitude_pressure
                    + 0.03 * amplitude_phase_pressure
                    + 0.05 * detected_trace_alignment
                    + 0.04 * temporal_index_overlap
                    + 0.03 * sequence_persistence_score
                    + 0.05 * btc_force_alignment
                    + 0.06 * btc_phase_alignment
                    + 0.05 * (1.0 - candidate_phase_confinement_cost)
                    + 0.04 * (1.0 - candidate_btc_phase_cost)
                    + 0.03 * detected_btc_force_alignment
                    + 0.07 * decode_phase_integrity
                    + 0.06 * decode_phase_alignment
                    + 0.06 * decode_target_ring_alignment
                    + 0.06 * decode_target_prefix_lock
                    + 0.05 * decode_target_prefix_vector_alignment
                    + 0.04 * decode_target_prefix_vector_phase_pressure
                    + 0.05 * decode_phase_orbital_alignment
                    + 0.04 * decode_phase_orbital_resonance
                    + 0.04 * decode_phase_orbital_stability
                    - 0.04 * decode_prefix_asymptote_pressure
                    + 0.04 * kernel_phase_alignment
                    + 0.04 * harmonic_carrier_gain
                    + 0.03 * retro_carrier_gain
                    + 0.03 * event_delta_gate
                    + 0.03 * event_delta_phase_alignment
                    + 0.03 * ancilla_phase_alignment
                    + 0.02 * ancilla_temporal_persistence
                    + 0.03 * clamp01(float(np.mean(coherent_noise_axis_vector[:4])))
                )
                coherence_peak = clamp01(
                    0.34 * coherence_score
                    + 0.16 * shared_score
                    + 0.13 * temporal_weight
                    + 0.06 * crosstalk
                    + 0.08 * interference_resonance
                    + 0.08 * target_alignment
                    + 0.06 * detected_cascade_activation
                    + 0.05 * detected_row_activation
                    + 0.05 * detected_motif_alignment
                    + 0.03 * detected_row_coherence
                    + 0.04 * detected_trace_alignment
                    + 0.05 * float(simulation_field_state.get("calibration_readiness", 0.0))
                    + 0.04 * target_carrier_window
                    + 0.04 * trajectory_phase
                    + 0.06 * vector_alignment
                    + 0.05 * basin_alignment
                    + 0.05 * vector_bias
                    + 0.04 * float(temporal_manifold.get("coherence_norm", 0.0))
                    + 0.03 * field_pressure
                    + 0.02 * larger_field_exposure
                    + 0.02 * clamp01(float(detected_neighbor_count) / 6.0)
                    + 0.04 * phase_length_pressure
                    + 0.04 * phase_amplitude_pressure
                    + 0.03 * amplitude_phase_pressure
                    + 0.03 * (1.0 - phase_length_span)
                    + 0.04 * sequence_persistence_score
                    + 0.04 * decode_phase_integrity
                    + 0.03 * decode_phase_alignment
                    + 0.03 * decode_target_ring_alignment
                    + 0.03 * decode_target_prefix_lock
                    + 0.03 * decode_target_prefix_vector_alignment
                    + 0.02 * decode_target_prefix_vector_phase_pressure
                    + 0.04 * decode_phase_orbital_alignment
                    + 0.03 * decode_phase_orbital_resonance
                    + 0.03 * decode_phase_orbital_stability
                    - 0.02 * decode_prefix_asymptote_pressure
                    + 0.03 * temporal_index_overlap
                    + 0.02 * candidate_voltage_frequency_flux
                    + 0.02 * candidate_frequency_voltage_flux
                    + 0.04 * kernel_balance_score
                    + 0.03 * harmonic_resonance_score
                    + 0.03 * retro_carrier_gain
                    + 0.03 * kernel_phase_alignment
                    + 0.03 * event_delta_gate
                    + 0.02 * event_delta_memory
                    + 0.02 * event_delta_flux
                    + 0.02 * event_delta_phase_alignment
                    + 0.04 * btc_force_alignment
                    + 0.03 * btc_phase_pressure
                    + 0.03 * ancilla_convergence
                    + 0.02 * ancilla_flux_norm
                    + 0.02 * ancilla_temporal_persistence
                    + 0.02 * ancilla_commit_gate
                )
                candidate_cluster_id = (
                    f"{cluster_root_id}"
                    f"-p{phase_length_index:02d}"
                    f"-q{int(round(phase_length_pressure * 10.0)):02d}"
                    f"-t{temporal_sequence_index:02d}"
                    f"-s{int(round(sequence_persistence_score * 10.0)):02d}"
                    f"-h{harmonic_order:02d}"
                )
                candidate_pool.append(
                    ensure_temporal_decode_metrics(
                        {
                        "nonce": int(nonce_state),
                        "packet_id": int(packet_idx),
                        "carrier_index": int(carrier_idx),
                        "cluster_id": candidate_cluster_id,
                        "coherence_peak": coherence_peak,
                        "shared_score": shared_score,
                        "crosstalk": crosstalk,
                        "temporal_weight": temporal_weight,
                        "clamp_pressure": clamp_pressure,
                        "phase_confinement_pressure": phase_confinement_pressure,
                        "target_alignment": target_alignment,
                        "target_interval": target_carrier_window,
                        "target_amplitude_cap": target_amplitude_cap,
                        "phase_amplitude_cap": phase_amplitude_cap,
                        "phase_amplitude_pressure": float(phase_amplitude_pressure),
                        "amplitude_phase_pressure": float(amplitude_phase_pressure),
                        "phase_alignment_score": float(phase_alignment_score),
                        "phase_confinement_cost": float(candidate_phase_confinement_cost),
                        "btc_force_alignment": float(btc_force_alignment),
                        "btc_phase_pressure": float(btc_phase_pressure),
                        "btc_phase_alignment": float(btc_phase_alignment),
                        "btc_phase_cost": float(candidate_btc_phase_cost),
                        "interference_resonance": interference_resonance,
                        "cascade_activation": detected_cascade_activation,
                        "vector_alignment": vector_alignment,
                        "basin_alignment": basin_alignment,
                        "row_activation": detected_row_activation,
                        "motif_alignment": detected_motif_alignment,
                        "trace_alignment": detected_trace_alignment,
                        "phase_length_index": int(phase_length_index),
                        "phase_length_window": float(phase_length_window),
                        "phase_length_span": float(phase_length_span),
                        "phase_length_pressure": float(phase_length_pressure),
                        "temporal_sequence_index": int(temporal_sequence_index),
                        "temporal_sequence_length": int(temporal_sequence_length),
                        "temporal_index_overlap": float(temporal_index_overlap),
                        "sequence_persistence_score": float(sequence_persistence_score),
                        "voltage_frequency_flux": float(candidate_voltage_frequency_flux),
                        "frequency_voltage_flux": float(candidate_frequency_voltage_flux),
                        "kernel_control_id": int(kernel_control_id),
                        "kernel_balance_score": float(kernel_balance_score),
                        "harmonic_resonance_score": float(harmonic_resonance_score),
                        "retro_temporal_gain": float(retro_carrier_gain),
                        "kernel_phase_alignment": float(kernel_phase_alignment),
                        "kernel_delta_gate": float(event_delta_gate),
                        "kernel_delta_memory": float(event_delta_memory),
                        "kernel_delta_flux": float(event_delta_flux),
                        "kernel_delta_phase_alignment": float(event_delta_phase_alignment),
                        "kernel_drive": float(kernel_drive),
                        "ancilla_commit_gate": float(ancilla_commit_gate),
                        "ancilla_activation_gate": float(ancilla_activation_gate),
                        "ancilla_current_norm": float(ancilla_current_norm),
                        "ancilla_flux_norm": float(ancilla_flux_norm),
                        "ancilla_phase_alignment": float(ancilla_phase_alignment),
                        "ancilla_convergence": float(ancilla_convergence),
                        "ancilla_temporal_persistence": float(ancilla_temporal_persistence),
                        "ancilla_tension_headroom": float(ancilla_tension_headroom),
                        "ancilla_gradient_headroom": float(ancilla_gradient_headroom),
                        "harmonic_order": int(harmonic_order),
                        "harmonic_weight": float(harmonic_weight),
                        "decoded_sequence_hex": str(decoded_nonce.get("decoded_sequence_hex", "")),
                        "decoded_sequence_utf8_preview": str(
                            decoded_nonce.get("decoded_sequence_utf8_preview", "")
                        ),
                        "decoded_raw_nonce": int(decoded_nonce.get("decoded_raw_nonce", 0)),
                        "mixed_nonce": int(decoded_nonce.get("mixed_nonce", nonce_state)),
                        "decode_sequence_positions": list(decoded_nonce.get("decode_sequence_positions", [])),
                        "decode_phase_ring_turns": list(decoded_nonce.get("decode_phase_ring_turns", [])),
                        "decode_phase_ring_deltas": list(decoded_nonce.get("decode_phase_ring_deltas", [])),
                        "decode_carrier_turns": list(decoded_nonce.get("decode_carrier_turns", [])),
                        "decode_temporal_coverage": float(decoded_nonce.get("decode_temporal_coverage", 0.0)),
                        "decode_temporal_span": float(decoded_nonce.get("decode_temporal_span", 0.0)),
                        "decode_integrity_score": float(decoded_nonce.get("decode_integrity_score", 0.0)),
                        "decode_symbol_entropy": float(decoded_nonce.get("decode_symbol_entropy", 0.0)),
                        "decode_transition_ratio": float(decoded_nonce.get("decode_transition_ratio", 0.0)),
                        "decode_longest_run_ratio": float(
                            decoded_nonce.get("decode_longest_run_ratio", 1.0)
                        ),
                        "decode_dominant_symbol_ratio": float(
                            decoded_nonce.get("decode_dominant_symbol_ratio", 1.0)
                        ),
                        "decode_unique_symbol_count": int(decoded_nonce.get("decode_unique_symbol_count", 0)),
                        "decode_unique_symbol_ratio": float(decoded_nonce.get("decode_unique_symbol_ratio", 0.0)),
                        "decode_phase_delta_alignment": float(
                            decoded_nonce.get("decode_phase_delta_alignment", 0.0)
                        ),
                        "decode_phase_delta_balance": float(
                            decoded_nonce.get("decode_phase_delta_balance", 0.0)
                        ),
                        "decode_phase_integrity_score": float(
                            decoded_nonce.get("decode_phase_integrity_score", 0.0)
                        ),
                        "decode_target_ring_alignment": float(
                            decoded_nonce.get("decode_target_ring_alignment", 0.0)
                        ),
                        "decode_target_phase_pressure": float(
                            decoded_nonce.get("decode_target_phase_pressure", 0.0)
                        ),
                        "decode_target_prefix_lock": float(
                            decoded_nonce.get("decode_target_prefix_lock", 0.0)
                        ),
                        "decode_target_prefix_phase_alignment": float(
                            decoded_nonce.get("decode_target_prefix_phase_alignment", 0.0)
                        ),
                        "decode_target_prefix_flux_alignment": float(
                            decoded_nonce.get("decode_target_prefix_flux_alignment", 0.0)
                        ),
                        "decode_target_prefix_vector": [
                            float(value)
                            for value in list(decoded_nonce.get("decode_target_prefix_vector", []) or [])
                        ],
                        "decode_target_prefix_vector_alignment": float(
                            decoded_nonce.get("decode_target_prefix_vector_alignment", 0.0)
                        ),
                        "decode_target_prefix_vector_flux_alignment": float(
                            decoded_nonce.get("decode_target_prefix_vector_flux_alignment", 0.0)
                        ),
                        "decode_target_prefix_vector_phase_pressure": float(
                            decoded_nonce.get("decode_target_prefix_vector_phase_pressure", 0.0)
                        ),
                        "decode_phase_orbital_trace_vector": [
                            float(value)
                            for value in list(decoded_nonce.get("decode_phase_orbital_trace_vector", []) or [])
                        ],
                        "decode_phase_orbital_vectors": list(
                            decoded_nonce.get("decode_phase_orbital_vectors", []) or []
                        ),
                        "decode_phase_orbital_alignment": float(
                            decoded_nonce.get("decode_phase_orbital_alignment", 0.0)
                        ),
                        "decode_phase_orbital_resonance": float(
                            decoded_nonce.get("decode_phase_orbital_resonance", 0.0)
                        ),
                        "decode_phase_orbital_stability": float(
                            decoded_nonce.get("decode_phase_orbital_stability", 0.0)
                        ),
                        "decode_phase_orbital_relative_field": [
                            float(value)
                            for value in list(decoded_nonce.get("decode_phase_orbital_relative_field", []) or [])
                        ],
                        "decode_prefix_asymptote_pressure": float(
                            decoded_nonce.get("decode_prefix_asymptote_pressure", 1.0)
                        ),
                        }
                    )
                )

        apply_gpu_feedback_delta_to_candidates(
            candidate_pool,
            injection_delta_feedback,
            simulation_field_state=simulation_field_state,
        )
        feedback_seed = dict(simulation_field_state.get("gpu_pulse_feedback", {}) or {})
        pre_cuda_gpu_feedback = sample_gpu_pulse_feedback(
            pulse_index=pulse_index,
            previous_feedback=feedback_seed,
            feedback_context=build_vector_feedback_context(
                pulse_index=pulse_index,
                lattice_calibration=silicon_calibration,
                pulse_sweep=pulse_sweep,
                feedback_state=simulation_field_state,
                target_profile=target_profile,
                candidate_pool=candidate_pool,
                effective_vector=effective_vector,
                temporal_manifold=temporal_manifold,
                kernel_execution_event=kernel_execution_event,
                feedback_stage="pre_cuda",
            ),
        )
        simulation_field_state["gpu_pulse_feedback_pre"] = dict(pre_cuda_gpu_feedback)
        cuda_dispatch_started = time.perf_counter()
        cuda_kernel_telemetry = run_cuda_temporal_candidate_stage(
            candidate_pool=candidate_pool,
            simulation_field_state=simulation_field_state,
            target_profile=target_profile,
        )
        cuda_dispatch_latency_ms = float((time.perf_counter() - cuda_dispatch_started) * 1000.0)
        cuda_kernel_telemetry["dispatch_latency_ms"] = float(cuda_dispatch_latency_ms)
        post_cuda_gpu_feedback = sample_gpu_pulse_feedback(
            pulse_index=pulse_index,
            previous_feedback=pre_cuda_gpu_feedback,
            feedback_context=build_vector_feedback_context(
                pulse_index=pulse_index,
                lattice_calibration=silicon_calibration,
                pulse_sweep=pulse_sweep,
                feedback_state=simulation_field_state,
                target_profile=target_profile,
                candidate_pool=candidate_pool,
                cuda_kernel_telemetry=cuda_kernel_telemetry,
                effective_vector=effective_vector,
                temporal_manifold=temporal_manifold,
                kernel_execution_event=kernel_execution_event,
                feedback_stage="post_cuda",
            ),
        )
        pre_sample_latency_ms = float(pre_cuda_gpu_feedback.get("sampling_latency_ms", 0.0))
        post_sample_latency_ms = float(post_cuda_gpu_feedback.get("sampling_latency_ms", 0.0))
        feedback_window_ms = float(pre_sample_latency_ms + post_sample_latency_ms)
        cuda_kernel_telemetry["pre_sample_latency_ms"] = float(pre_sample_latency_ms)
        cuda_kernel_telemetry["post_sample_latency_ms"] = float(post_sample_latency_ms)
        cuda_kernel_telemetry["feedback_window_ms"] = float(feedback_window_ms)
        post_cuda_gpu_feedback["dispatch_latency_ms"] = float(cuda_dispatch_latency_ms)
        post_cuda_gpu_feedback["pulse_roundtrip_latency_ms"] = float(feedback_window_ms)
        post_cuda_gpu_feedback["pulse_feedback_window_ms"] = float(feedback_window_ms)
        gpu_pulse_delta_feedback = build_gpu_pulse_delta_feedback(
            pulse_index=pulse_index,
            pre_feedback=pre_cuda_gpu_feedback,
            post_feedback=post_cuda_gpu_feedback,
            cuda_kernel_telemetry=cuda_kernel_telemetry,
        )
        simulation_field_state["gpu_pulse_feedback_post_cuda"] = dict(
            post_cuda_gpu_feedback
        )
        simulation_field_state["gpu_pulse_delta_feedback_post_cuda"] = dict(
            gpu_pulse_delta_feedback
        )
        integrate_gpu_feedback_into_field_state(
            simulation_field_state=simulation_field_state,
            gpu_feedback=post_cuda_gpu_feedback,
            gpu_pulse_delta_feedback=gpu_pulse_delta_feedback,
            blend=0.26,
        )
        substrate_trace_state = update_substrate_trace_state(
            pulse_index=pulse_index,
            previous_trace_state=simulation_field_state.get("substrate_trace_state", {}),
            simulation_field_state=simulation_field_state,
            gpu_feedback=post_cuda_gpu_feedback,
            gpu_pulse_delta_feedback=gpu_pulse_delta_feedback,
            interference_field=interference_field,
            effective_vector=effective_vector,
            kernel_execution_event=kernel_execution_event,
            trace_label="post_cuda",
        )
        simulation_field_state["substrate_trace_state"] = dict(substrate_trace_state)
        substrate_trace_vram = sync_substrate_trace_state_to_vram(substrate_trace_state)
        simulation_field_state["substrate_trace_vram"] = dict(substrate_trace_vram)
        kernel_execution_event["gpu_delta_response_energy"] = float(
            gpu_pulse_delta_feedback.get("response_energy", 0.0)
        )
        kernel_execution_event["gpu_delta_phase_shift_turns"] = float(
            gpu_pulse_delta_feedback.get("phase_shift_turns", 0.0)
        )
        kernel_execution_event["gpu_delta_response_gate"] = float(
            gpu_pulse_delta_feedback.get("response_gate", 0.0)
        )
        kernel_execution_event["gpu_delta_memory_retention"] = float(
            gpu_pulse_delta_feedback.get("memory_retention", 0.0)
        )
        kernel_execution_event["gpu_delta_latency_norm"] = float(
            gpu_pulse_delta_feedback.get("latency_norm", 0.0)
        )
        kernel_execution_event["gpu_delta_latency_gate"] = float(
            gpu_pulse_delta_feedback.get("latency_gate", 0.0)
        )
        kernel_execution_event["gpu_delta_environment_pressure"] = float(
            gpu_pulse_delta_feedback.get("environment_pressure_target", 0.0)
        )
        kernel_execution_event["gpu_feedback_temperature_norm"] = float(
            post_cuda_gpu_feedback.get("temperature_norm", 0.0)
        )
        kernel_execution_event["gpu_feedback_environment_pressure"] = float(
            post_cuda_gpu_feedback.get("environment_pressure", 0.0)
        )
        kernel_execution_event["gpu_feedback_roundtrip_latency_ms"] = float(
            gpu_pulse_delta_feedback.get("roundtrip_latency_ms", 0.0)
        )
        kernel_execution_event["gpu_observation_gap_ms"] = float(
            gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)
        )
        kernel_execution_event["gpu_observation_gap_norm"] = float(
            gpu_pulse_delta_feedback.get("observation_gap_norm", 0.0)
        )
        kernel_execution_event["gpu_observation_freshness_gate"] = float(
            gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)
        )
        kernel_execution_event["gpu_dispatch_feedback_ratio"] = float(
            gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0)
        )
        kernel_execution_event["trace_support"] = float(
            substrate_trace_state.get("trace_support", 0.0)
        )
        kernel_execution_event["trace_resonance"] = float(
            substrate_trace_state.get("trace_resonance", 0.0)
        )
        kernel_execution_event["trace_alignment"] = float(
            substrate_trace_state.get("trace_alignment", 0.0)
        )
        kernel_execution_event["trace_memory"] = float(
            substrate_trace_state.get("trace_memory", 0.0)
        )
        kernel_execution_event["trace_flux"] = float(
            substrate_trace_state.get("trace_flux", 0.0)
        )
        kernel_execution_event["trace_vram_resident"] = bool(
            substrate_trace_vram.get("resident", False)
        )
        kernel_execution_event["trace_vram_updates"] = int(
            substrate_trace_vram.get("update_count", 0)
        )
        update_compute_regime(
            simulation_field_state=simulation_field_state,
            kernel_execution_event=kernel_execution_event,
            freshness_gate=max(
                float(injection_delta_feedback.get("observation_freshness_gate", 0.0)),
                float(gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)),
            ),
        )
        kernel_execution_event["gpu_feedback_window_span_norm"] = float(
            gpu_pulse_delta_feedback.get("window_span_norm", 0.0)
        )
        kernel_execution_event["gpu_feedback_window_steps"] = int(
            gpu_pulse_delta_feedback.get("window_calibration_steps", 1)
        )
        kernel_execution_event["gpu_pulse_delta_feedback"] = {
            "response_energy": float(gpu_pulse_delta_feedback.get("response_energy", 0.0)),
            "phase_shift_turns": float(gpu_pulse_delta_feedback.get("phase_shift_turns", 0.0)),
            "sequence_persistence_target": float(
                gpu_pulse_delta_feedback.get("sequence_persistence_target", 0.0)
            ),
            "temporal_overlap_target": float(gpu_pulse_delta_feedback.get("temporal_overlap_target", 0.0)),
            "voltage_frequency_flux_target": float(
                gpu_pulse_delta_feedback.get("voltage_frequency_flux_target", 0.0)
            ),
            "latency_norm": float(gpu_pulse_delta_feedback.get("latency_norm", 0.0)),
            "latency_gate": float(gpu_pulse_delta_feedback.get("latency_gate", 0.0)),
            "environment_pressure_target": float(
                gpu_pulse_delta_feedback.get("environment_pressure_target", 0.0)
            ),
            "observation_gap_ms": float(gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)),
            "observation_freshness_gate": float(
                gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)
            ),
            "dispatch_feedback_ratio": float(
                gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0)
            ),
            "window_span_norm": float(gpu_pulse_delta_feedback.get("window_span_norm", 0.0)),
            "window_calibration_steps": int(
                gpu_pulse_delta_feedback.get("window_calibration_steps", 1)
            ),
        }
        cuda_kernel_telemetry["gpu_delta_response_energy"] = float(
            gpu_pulse_delta_feedback.get("response_energy", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_phase_shift_turns"] = float(
            gpu_pulse_delta_feedback.get("phase_shift_turns", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_response_gate"] = float(
            gpu_pulse_delta_feedback.get("response_gate", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_latency_norm"] = float(
            gpu_pulse_delta_feedback.get("latency_norm", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_latency_gate"] = float(
            gpu_pulse_delta_feedback.get("latency_gate", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_environment_pressure"] = float(
            gpu_pulse_delta_feedback.get("environment_pressure_target", 0.0)
        )
        cuda_kernel_telemetry["gpu_observation_gap_ms"] = float(
            gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)
        )
        cuda_kernel_telemetry["gpu_observation_gap_norm"] = float(
            gpu_pulse_delta_feedback.get("observation_gap_norm", 0.0)
        )
        cuda_kernel_telemetry["gpu_observation_freshness_gate"] = float(
            gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)
        )
        cuda_kernel_telemetry["gpu_dispatch_feedback_ratio"] = float(
            gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_window_span_norm"] = float(
            gpu_pulse_delta_feedback.get("window_span_norm", 0.0)
        )
        cuda_kernel_telemetry["gpu_delta_window_steps"] = int(
            gpu_pulse_delta_feedback.get("window_calibration_steps", 1)
        )
        post_feedback_apply_gate = clamp01(
            0.52 * float(gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0))
            + 0.28 * float(gpu_pulse_delta_feedback.get("latency_gate", 0.0))
            + 0.20
            * (
                1.0
                - clamp01(
                    math.log10(
                        1.0 + max(float(gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)), 0.0)
                    )
                    / 4.0
                )
            )
        )
        post_feedback_applied = True
        if (
            str(simulation_field_state.get("compute_regime", "classical_calibration"))
            == "vector_harmonic"
            and post_feedback_apply_gate < 0.42
        ):
            post_feedback_applied = False
        if post_feedback_applied:
            apply_gpu_feedback_delta_to_candidates(
                candidate_pool,
                gpu_pulse_delta_feedback,
                simulation_field_state=simulation_field_state,
            )
        kernel_execution_event["post_feedback_apply_gate"] = float(post_feedback_apply_gate)
        kernel_execution_event["post_feedback_applied"] = bool(post_feedback_applied)
        candidate_pool.sort(
            key=lambda item: (
                item.get("gpu_feedback_delta_score", 0.0),
                item.get("cuda_temporal_score", 0.0),
                item.get("decode_phase_integrity_score", 0.0),
                item.get("decode_phase_delta_alignment", 0.0),
                item.get("decode_target_ring_alignment", 0.0),
                item.get("decode_target_prefix_lock", 0.0),
                item.get("decode_target_prefix_vector_alignment", 0.0),
                item.get("decode_target_prefix_vector_phase_pressure", 0.0),
                item.get("decode_phase_orbital_alignment", 0.0),
                item.get("decode_phase_orbital_resonance", 0.0),
                item.get("decode_phase_orbital_stability", 0.0),
                1.0 - item.get("decode_prefix_asymptote_pressure", 1.0),
                item.get("decode_target_phase_pressure", 0.0),
                item["target_alignment"],
                item.get("btc_force_alignment", 0.0),
                item.get("decode_integrity_score", 0.0),
                item.get("decode_symbol_entropy", 0.0),
                item.get("decode_transition_ratio", 0.0),
                1.0 - item.get("decode_longest_run_ratio", 1.0),
                1.0 - item.get("phase_confinement_cost", 0.0),
                1.0 - item.get("phase_confinement_pressure", 0.0),
                1.0 - item.get("clamp_pressure", 0.0),
                1.0 - item.get("btc_phase_cost", 0.0),
                item.get("btc_phase_alignment", item.get("btc_phase_pressure", 0.0)),
                item.get("phase_alignment_score", item.get("phase_length_pressure", 0.0)),
                item["coherence_peak"],
                item["interference_resonance"],
                item["motif_alignment"],
                item["sequence_persistence_score"],
                item["temporal_index_overlap"],
                item.get("kernel_phase_alignment", 0.0),
                item.get("harmonic_resonance_score", 0.0),
                item.get("kernel_balance_score", 0.0),
                item.get("retro_temporal_gain", 0.0),
                item.get("kernel_delta_phase_alignment", 0.0),
                item.get("kernel_delta_gate", 0.0),
                item.get("kernel_delta_memory", 0.0),
                item.get("kernel_delta_flux", 0.0),
                item["phase_length_pressure"],
                item["row_activation"],
                item["cascade_activation"],
                item["shared_score"],
                item["basin_alignment"],
                item["vector_alignment"],
                item["target_interval"],
            ),
            reverse=True,
        )
        if not candidate_pool:
            pulse_state = make_sink_state(
                {
                    "pulse_id": pulse_index,
                    "worker_count": pulse_workers,
                    "nbits": nbits_hex,
                    "target_hex": target_hex,
                    "network_target_hex": network_target_hex,
                    "field_pressure": field_pressure,
                    "larger_field_exposure": larger_field_exposure,
                    "dominant_basin": dominant_basin_id,
                },
                {"amplitude_cap": amplitude_cap},
            )
            pulse_state["psi_encode"] = psi_encode
            pulse_state["temporal_manifold"] = temporal_manifold
            pulse_state["effective_vector"] = effective_vector
            pulse_state["manifold_diagnostics"] = manifold_diagnostics
            pulse_state["simulation_field_state"] = simulation_field_state
            pulse_state["interference_field"] = interference_field
            pulse_state["kernel_execution_event"] = kernel_execution_event
            pulse_state["interference_resonance"] = float(field_vector_resonance)
            pulse_state["calibration_readiness"] = float(simulation_field_state.get("calibration_readiness", 0.0))
            pulse_state["entry_trigger"] = bool(simulation_field_state.get("entry_trigger", False))
            pulse_state["motif_consistency"] = float(simulation_field_state.get("motif_consistency", 0.0))
            pulse_state["motif_repeat_count"] = int(simulation_field_state.get("motif_repeat_count", 0))
            pulse_state["motif_energy"] = float(simulation_field_state.get("motif_energy", 0.0))
            pulse_state["vector_magnitude"] = float(effective_vector.get("spatial_magnitude", 0.0))
            pulse_state["temporal_projection"] = temporal_confinement
            pulse_state["field_pressure"] = field_pressure
            pulse_state["larger_field_exposure"] = larger_field_exposure
            pulse_state["dominant_basin"] = dominant_basin_id
            pulse_state["substrate_material"] = str(
                kernel_execution_event.get(
                    "substrate_material",
                    simulation_field_state.get("substrate_material", "silicon_wafer"),
                )
            )
            pulse_state["silicon_reference_source"] = str(
                kernel_execution_event.get(
                    "silicon_reference_source",
                    simulation_field_state.get("silicon_reference_source", NIST_REFERENCE.name),
                )
            )
            pulse_state["compute_regime"] = str(
                kernel_execution_event.get(
                    "compute_regime",
                    simulation_field_state.get("compute_regime", "classical_calibration"),
                )
            )
            pulse_state["vector_harmonic_gate"] = float(
                kernel_execution_event.get(
                    "vector_harmonic_gate",
                    simulation_field_state.get("vector_harmonic_gate", 0.0),
                )
            )
            pulse_state["harmonic_compute_weight"] = float(
                kernel_execution_event.get(
                    "harmonic_compute_weight",
                    simulation_field_state.get("harmonic_compute_weight", 0.0),
                )
            )
            pulse_state["trace_support"] = float(
                kernel_execution_event.get(
                    "trace_support",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_support", 0.0),
                )
            )
            pulse_state["trace_resonance"] = float(
                kernel_execution_event.get(
                    "trace_resonance",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_resonance", 0.0),
                )
            )
            pulse_state["trace_alignment"] = float(
                kernel_execution_event.get(
                    "trace_alignment",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_alignment", 0.0),
                )
            )
            pulse_state["trace_vram_resident"] = bool(
                kernel_execution_event.get(
                    "trace_vram_resident",
                    simulation_field_state.get("substrate_trace_vram", {}).get("resident", False),
                )
            )
            pulse_state["trace_vram_updates"] = int(
                kernel_execution_event.get(
                    "trace_vram_updates",
                    simulation_field_state.get("substrate_trace_vram", {}).get("update_count", 0),
                )
            )
            pulse_state["gpu_feedback_source"] = str(
                simulation_field_state.get("gpu_pulse_feedback_post_cuda", {}).get(
                    "source",
                    simulation_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                )
            )
            pulse_state["gpu_injection_feedback_source"] = str(
                simulation_field_state.get("gpu_pulse_feedback_injection", {}).get(
                    "source",
                    simulation_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                )
            )
            pulse_state["path_equivalence_error"] = float(manifold_diagnostics.get("path_equivalence_error", 1.0))
            pulse_state["temporal_ordering_delta"] = float(manifold_diagnostics.get("temporal_ordering_delta", 1.0))
            pulse_state["basis_rotation_residual"] = float(manifold_diagnostics.get("basis_rotation_residual", 1.0))
            pulse_batches.append(pulse_state)
            prev_state = pulse_state
            continue

        rolling_head = candidate_pool[: max(24, min(96, len(candidate_pool)))]
        yield_count = max(
            20,
            min(
                70,
                int(
                    round(
                        20.0
                        + 50.0
                        * clamp01(
                            float(np.mean([item["coherence_peak"] for item in rolling_head]))
                        )
                    )
                ),
            ),
        )

        cluster_map: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidate_pool:
            ensure_temporal_decode_metrics(candidate)
            cluster_map.setdefault(str(candidate["cluster_id"]), []).append(candidate)

        previous_best_function_score = clamp01(float(prev_state.get("best_function_score", 0.0)))
        previous_function_bias = int(round(previous_best_function_score * 8.0))
        cuda_probe_bonus = 0
        cuda_search_volume_gain = max(1.0, float(cuda_kernel_telemetry.get("search_volume_gain", 1.0)))
        if bool(cuda_kernel_telemetry.get("enabled", False)):
            cuda_probe_bonus = min(
                288,
                40
                + int(round(float(cuda_kernel_telemetry.get("expanded_keep_count", 0)) / 64.0))
                + int(round((cuda_search_volume_gain - 1.0) * 72.0)),
            )
        probe_seed_budget = max(
            40,
            min(
                320,
                40
                + int(round((1.0 - previous_best_function_score) * 48.0))
                + int(cuda_probe_bonus),
            ),
        )
        probe_seed_budget = min(probe_seed_budget, len(candidate_pool))
        probe_seed_candidates, cluster_probe_profiles = build_cluster_probe_plan(
            cluster_map=cluster_map,
            seed_budget=probe_seed_budget,
        )
        if not probe_seed_candidates:
            cluster_probe_profiles = []
            probe_seed_candidates = candidate_pool[:probe_seed_budget]
        probed_candidates_by_nonce: dict[int, dict[str, Any]] = {}
        for seed_rank, candidate in enumerate(probe_seed_candidates):
            offsets = build_target_probe_offsets(
                candidate=candidate,
                target_profile=target_profile,
                pulse_index=pulse_index,
                cluster_rank=int(candidate.get("cluster_probe_rank", seed_rank)),
                cluster_weight=float(candidate.get("cluster_probe_weight", 0.0)),
                prefix_bias=previous_function_bias,
            )
            for offset_index, offset in enumerate(offsets):
                probe_nonce = (int(candidate["nonce"]) + int(offset)) & 0xFFFFFFFF
                share = sha256d_compute_share({"header_hex": header_hex}, probe_nonce)
                function_score = clamp01(float(share.get("sha256_function_score", 0.0)))
                round_coupling = clamp01(float(share.get("sha256_round_coupling", 0.0)))
                first_entropy = clamp01(float(share.get("sha256_first_entropy", 0.0)))
                final_entropy = clamp01(float(share.get("sha256_final_entropy", 0.0)))
                transition_ratio = clamp01(float(share.get("sha256_transition_ratio", 0.0)))
                bit_balance = clamp01(float(share.get("sha256_bit_balance", 0.0)))
                lane_balance = clamp01(float(share.get("sha256_lane_balance", 0.0)))
                hash_phase_metrics = hash_target_phase_metrics(
                    str(share.get("hash_hex", "")),
                    str(network_target_hex),
                )
                hash_target_phase_pressure = clamp01(
                    float(hash_phase_metrics.get("hash_target_phase_pressure", 0.0))
                )
                hash_target_phase_cost = clamp01(
                    float(hash_phase_metrics.get("hash_target_phase_cost", 1.0))
                )
                hash_target_prefix_lock = clamp01(
                    float(hash_phase_metrics.get("hash_target_prefix_lock", 0.0))
                )
                hash_target_window_coverage = clamp01(
                    float(hash_phase_metrics.get("hash_target_window_coverage", 0.0))
                )
                hash_target_band_alignment = clamp01(
                    float(hash_phase_metrics.get("hash_target_band_alignment", 0.0))
                )
                hash_target_flux_alignment = clamp01(
                    float(hash_phase_metrics.get("hash_target_flux_alignment", 0.0))
                )
                hash_target_frontier_slack = clamp01(
                    float(hash_phase_metrics.get("hash_target_frontier_slack", 0.0))
                )
                hash_target_stable_pressure = clamp01(
                    float(hash_phase_metrics.get("hash_target_stable_pressure", 0.0))
                )
                network_valid = bool(
                    pow_hex_leq_target(str(share.get("hash_hex", "")), str(network_target_hex))
                )
                probe_score = clamp01(
                    0.22 * function_score
                    + 0.10 * round_coupling
                    + 0.10 * hash_target_phase_pressure
                    + 0.08 * hash_target_stable_pressure
                    + 0.06 * hash_target_prefix_lock
                    + 0.05 * hash_target_window_coverage
                    + 0.05 * hash_target_flux_alignment
                    + 0.04 * hash_target_band_alignment
                    + 0.05 * (1.0 - hash_target_phase_cost)
                    + 0.08 * final_entropy
                    + 0.06 * first_entropy
                    + 0.06 * transition_ratio
                    + 0.06 * bit_balance
                    + 0.05 * lane_balance
                    + 0.05 * float(candidate.get("decode_integrity_score", 0.0))
                    + 0.04 * float(candidate.get("decode_phase_integrity_score", 0.0))
                    + 0.04 * float(candidate.get("decode_phase_delta_alignment", 0.0))
                    + 0.04 * float(candidate.get("decode_target_ring_alignment", 0.0))
                    + 0.05 * float(candidate.get("decode_target_prefix_lock", 0.0))
                    + 0.04 * float(candidate.get("decode_target_prefix_vector_alignment", 0.0))
                    + 0.03 * float(candidate.get("decode_target_prefix_vector_phase_pressure", 0.0))
                    + 0.04 * float(candidate.get("decode_phase_orbital_alignment", 0.0))
                    + 0.03 * float(candidate.get("decode_phase_orbital_resonance", 0.0))
                    + 0.03 * float(candidate.get("decode_phase_orbital_stability", 0.0))
                    + 0.04 * (
                        1.0 - float(candidate.get("decode_prefix_asymptote_pressure", 1.0))
                    )
                    + 0.03 * float(candidate.get("decode_target_phase_pressure", 0.0))
                    + 0.03 * float(candidate.get("decode_phase_delta_balance", 0.0))
                    + 0.03 * float(candidate.get("decode_symbol_entropy", 0.0))
                    + 0.03 * float(candidate.get("decode_transition_ratio", 0.0))
                    + 0.03 * (1.0 - float(candidate.get("decode_longest_run_ratio", 1.0)))
                    + 0.05 * float(candidate.get("btc_force_alignment", 0.0))
                    + 0.03 * float(candidate.get("phase_alignment_score", candidate.get("phase_length_pressure", 0.0)))
                    + 0.03 * float(candidate.get("btc_phase_alignment", candidate.get("btc_phase_pressure", 0.0)))
                    + 0.06 * (1.0 - float(candidate.get("phase_confinement_cost", 0.0)))
                    + 0.04 * (1.0 - float(candidate.get("btc_phase_cost", 0.0)))
                    + 0.05 * (1.0 - float(candidate.get("phase_confinement_pressure", 0.0)))
                    + 0.05 * (1.0 - float(candidate.get("clamp_pressure", 0.0)))
                    + 0.04 * float(candidate.get("coherence_peak", 0.0))
                    + 0.06 * float(candidate.get("motif_alignment", 0.0))
                    + 0.05 * float(candidate.get("cluster_probe_weight", 0.0))
                    + 0.04 * float(candidate.get("row_activation", 0.0))
                    + 0.04 * float(candidate.get("sequence_persistence_score", 0.0))
                    + 0.03 * float(candidate.get("temporal_index_overlap", 0.0))
                    + 0.02 * float(candidate.get("voltage_frequency_flux", 0.0))
                    + 0.02 * float(candidate.get("frequency_voltage_flux", 0.0))
                    + 0.05 * float(candidate.get("cuda_temporal_score", 0.0))
                    + 0.04 * float(candidate.get("gpu_feedback_delta_score", 0.0))
                    + 0.02 * float(gpu_pulse_delta_feedback.get("response_gate", 0.0))
                    + 0.01 * float(candidate.get("kernel_balance_score", 0.0))
                    + 0.01 * float(candidate.get("harmonic_resonance_score", 0.0))
                    + 0.01 * float(candidate.get("retro_temporal_gain", 0.0))
                    + 0.01 * float(candidate.get("kernel_phase_alignment", 0.0))
                    + 0.01 * float(candidate.get("kernel_delta_phase_alignment", 0.0))
                    + 0.01 * float(candidate.get("kernel_delta_gate", 0.0))
                    + 0.01 * float(candidate.get("kernel_delta_memory", 0.0))
                    + 0.01 * float(candidate.get("kernel_delta_flux", 0.0))
                    + 0.02 * float(candidate.get("ancilla_convergence", 0.0))
                    + 0.02 * float(candidate.get("ancilla_phase_alignment", 0.0))
                    + 0.01 * float(candidate.get("ancilla_flux_norm", 0.0))
                    + 0.01 * float(candidate.get("ancilla_temporal_persistence", 0.0))
                )
                probed_candidate = dict(candidate)
                probed_candidate["nonce"] = int(probe_nonce)
                probed_candidate["probe_seed_rank"] = int(seed_rank)
                probed_candidate["probe_offset"] = int(offset)
                probed_candidate["probe_offset_index"] = int(offset_index)
                probed_candidate["probe_first_hash_hex"] = str(share.get("first_hash_hex", ""))
                probed_candidate["probe_hash_hex"] = str(share.get("hash_hex", ""))
                probed_candidate["probe_header_hex"] = str(share.get("header", header_hex))
                probed_candidate["sha256_function_score"] = float(function_score)
                probed_candidate["sha256_round_coupling"] = float(round_coupling)
                probed_candidate["sha256_first_entropy"] = float(first_entropy)
                probed_candidate["sha256_final_entropy"] = float(final_entropy)
                probed_candidate["sha256_transition_ratio"] = float(transition_ratio)
                probed_candidate["sha256_bit_balance"] = float(bit_balance)
                probed_candidate["sha256_lane_balance"] = float(lane_balance)
                probed_candidate["hash_target_phase_pressure"] = float(hash_target_phase_pressure)
                probed_candidate["hash_target_phase_cost"] = float(hash_target_phase_cost)
                probed_candidate["hash_target_prefix_lock"] = float(hash_target_prefix_lock)
                probed_candidate["hash_target_window_coverage"] = float(hash_target_window_coverage)
                probed_candidate["hash_target_band_alignment"] = float(hash_target_band_alignment)
                probed_candidate["hash_target_flux_alignment"] = float(hash_target_flux_alignment)
                probed_candidate["hash_target_frontier_slack"] = float(hash_target_frontier_slack)
                probed_candidate["hash_target_stable_pressure"] = float(hash_target_stable_pressure)
                probed_candidate["probe_score"] = float(probe_score)
                probed_candidate["network_valid"] = bool(network_valid)
                ensure_temporal_decode_metrics(probed_candidate)
                existing = probed_candidates_by_nonce.get(probe_nonce)
                if existing is None or (
                    bool(probed_candidate["network_valid"]),
                    float(probed_candidate["probe_score"]),
                    float(probed_candidate["hash_target_stable_pressure"]),
                    float(probed_candidate["sha256_function_score"]),
                    float(probed_candidate["coherence_peak"]),
                ) > (
                    bool(existing.get("network_valid", False)),
                    float(existing.get("probe_score", 0.0)),
                    float(existing.get("hash_target_stable_pressure", 0.0)),
                    float(existing.get("sha256_function_score", 0.0)),
                    float(existing.get("coherence_peak", 0.0)),
                ):
                    probed_candidates_by_nonce[probe_nonce] = probed_candidate
        probed_candidates = list(probed_candidates_by_nonce.values())
        try:
            network_target_value = int(normalize_hex_64(network_target_hex), 16)
        except Exception:
            network_target_value = 1
        network_target_probability = min(
            1.0,
            float(network_target_value + 1) / float(1 << 256),
        )
        prototype_share_target = {
            "prototype_target_hex": str(network_target_hex),
            "sample_budget": int(max(len(probed_candidates), 1)),
            "desired_valid_count": 0,
            "target_multiplier": 1.0,
            "target_probability": float(network_target_probability),
        }
        prototype_target_hex = str(network_target_hex)
        for candidate in probed_candidates:
            candidate["prototype_valid"] = bool(candidate.get("network_valid", False))
        probed_candidates.sort(
            key=lambda item: (
                bool(item.get("network_valid", False)),
                float(item.get("probe_score", 0.0)),
                float(item.get("hash_target_stable_pressure", 0.0)),
                float(item.get("hash_target_phase_pressure", 0.0)),
                1.0 - float(item.get("hash_target_phase_cost", 1.0)),
                float(item.get("hash_target_prefix_lock", 0.0)),
                float(item.get("hash_target_flux_alignment", 0.0)),
                float(item.get("hash_target_window_coverage", 0.0)),
                float(item.get("sha256_function_score", 0.0)),
                float(item.get("sha256_round_coupling", 0.0)),
                float(item.get("sha256_final_entropy", 0.0)),
                float(item.get("sha256_transition_ratio", 0.0)),
                float(item.get("decode_integrity_score", 0.0)),
                float(item.get("decode_phase_integrity_score", 0.0)),
                float(item.get("decode_phase_delta_alignment", 0.0)),
                float(item.get("decode_target_ring_alignment", 0.0)),
                float(item.get("decode_target_prefix_lock", 0.0)),
                float(item.get("decode_target_prefix_vector_alignment", 0.0)),
                float(item.get("decode_target_prefix_vector_phase_pressure", 0.0)),
                float(item.get("decode_phase_orbital_alignment", 0.0)),
                float(item.get("decode_phase_orbital_resonance", 0.0)),
                float(item.get("decode_phase_orbital_stability", 0.0)),
                1.0 - float(item.get("decode_prefix_asymptote_pressure", 1.0)),
                float(item.get("decode_target_phase_pressure", 0.0)),
                float(item.get("decode_symbol_entropy", 0.0)),
                float(item.get("decode_transition_ratio", 0.0)),
                1.0 - float(item.get("decode_longest_run_ratio", 1.0)),
                float(item.get("btc_force_alignment", 0.0)),
                1.0 - float(item.get("phase_confinement_cost", 0.0)),
                1.0 - float(item.get("phase_confinement_pressure", 0.0)),
                1.0 - float(item.get("clamp_pressure", 0.0)),
                1.0 - float(item.get("btc_phase_cost", 0.0)),
                float(item.get("phase_alignment_score", item.get("phase_length_pressure", 0.0))),
                float(item.get("btc_phase_alignment", item.get("btc_phase_pressure", 0.0))),
                float(item.get("gpu_feedback_delta_score", 0.0)),
                float(item.get("cuda_temporal_score", 0.0)),
                float(item.get("sequence_persistence_score", 0.0)),
                float(item.get("temporal_index_overlap", 0.0)),
                float(item.get("coherence_peak", 0.0)),
                float(item.get("motif_alignment", 0.0)),
                float(item.get("kernel_phase_alignment", 0.0)),
                float(item.get("harmonic_resonance_score", 0.0)),
                float(item.get("kernel_balance_score", 0.0)),
                float(item.get("retro_temporal_gain", 0.0)),
                float(item.get("kernel_delta_phase_alignment", 0.0)),
                float(item.get("kernel_delta_gate", 0.0)),
                float(item.get("kernel_delta_memory", 0.0)),
                float(item.get("kernel_delta_flux", 0.0)),
            ),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        network_valid_candidates = [
            candidate for candidate in probed_candidates if bool(candidate.get("network_valid", False))
        ]
        selection_source = (
            network_valid_candidates
            if network_valid_candidates
            else (probed_candidates if probed_candidates else candidate_pool)
        )
        selection_cluster_map: dict[str, list[dict[str, Any]]] = {}
        for candidate in selection_source:
            selection_cluster_map.setdefault(str(candidate["cluster_id"]), []).append(candidate)
        active_clusters = list(selection_cluster_map.keys())
        while len(selected) < yield_count and active_clusters:
            next_clusters: list[str] = []
            for cluster_id in active_clusters:
                cluster_items = selection_cluster_map.get(cluster_id, [])
                if cluster_items:
                    candidate = cluster_items.pop(0)
                    if not any(prev["nonce"] == candidate["nonce"] for prev in selected):
                        selected.append(candidate)
                    if len(selected) >= yield_count:
                        break
                if cluster_items:
                    next_clusters.append(cluster_id)
            active_clusters = next_clusters
        if len(selected) < yield_count:
            for candidate in selection_source:
                if len(selected) >= yield_count:
                    break
                if any(prev["nonce"] == candidate["nonce"] for prev in selected):
                    continue
                selected.append(candidate)
        for candidate in selected:
            ensure_temporal_decode_metrics(candidate)

        candidate_yield_count = len(selected)
        candidate_coherence_peak = max(item["coherence_peak"] for item in selected)
        candidate_temporal_residual = float(
            np.mean([abs(item["temporal_weight"] - temporal_persistence) for item in selected])
        )
        candidate_clamp_pressure = float(np.mean([item["clamp_pressure"] for item in selected]))
        candidate_target_alignment = float(np.mean([item["target_alignment"] for item in selected]))
        candidate_target_interval = float(np.mean([item["target_interval"] for item in selected]))
        candidate_interference_resonance = float(np.mean([item["interference_resonance"] for item in selected]))
        candidate_cascade_activation = float(np.mean([item["cascade_activation"] for item in selected]))
        candidate_phase_length_pressure = float(
            np.mean([float(item.get("phase_length_pressure", 0.0)) for item in selected])
        )
        candidate_sequence_persistence = float(
            np.mean([float(item.get("sequence_persistence_score", 0.0)) for item in selected])
        )
        candidate_temporal_overlap = float(
            np.mean([float(item.get("temporal_index_overlap", 0.0)) for item in selected])
        )
        candidate_voltage_frequency_flux = float(
            np.mean([float(item.get("voltage_frequency_flux", 0.0)) for item in selected])
        )
        candidate_phase_alignment_score = float(
            np.mean(
                [
                    float(item.get("phase_alignment_score", item.get("phase_length_pressure", 0.0)))
                    for item in selected
                ]
            )
        )
        candidate_phase_confinement_cost = float(
            np.mean(
                [
                    float(item.get("phase_confinement_cost", 0.0))
                    for item in selected
                ]
            )
        )
        candidate_decode_integrity = float(
            np.mean([float(item.get("decode_integrity_score", 0.0)) for item in selected])
        )
        candidate_decode_entropy = float(
            np.mean([float(item.get("decode_symbol_entropy", 0.0)) for item in selected])
        )
        candidate_decode_phase_integrity = float(
            np.mean([float(item.get("decode_phase_integrity_score", 0.0)) for item in selected])
        )
        candidate_decode_phase_alignment = float(
            np.mean([float(item.get("decode_phase_delta_alignment", 0.0)) for item in selected])
        )
        candidate_decode_target_ring_alignment = float(
            np.mean([float(item.get("decode_target_ring_alignment", 0.0)) for item in selected])
        )
        candidate_decode_target_prefix_lock = float(
            np.mean([float(item.get("decode_target_prefix_lock", 0.0)) for item in selected])
        )
        candidate_decode_target_prefix_vector_alignment = float(
            np.mean(
                [
                    float(item.get("decode_target_prefix_vector_alignment", 0.0))
                    for item in selected
                ]
            )
        )
        candidate_decode_target_prefix_vector_phase_pressure = float(
            np.mean(
                [
                    float(item.get("decode_target_prefix_vector_phase_pressure", 0.0))
                    for item in selected
                ]
            )
        )
        candidate_decode_phase_orbital_alignment = float(
            np.mean([float(item.get("decode_phase_orbital_alignment", 0.0)) for item in selected])
        )
        candidate_decode_phase_orbital_resonance = float(
            np.mean([float(item.get("decode_phase_orbital_resonance", 0.0)) for item in selected])
        )
        candidate_decode_phase_orbital_stability = float(
            np.mean([float(item.get("decode_phase_orbital_stability", 0.0)) for item in selected])
        )
        candidate_decode_prefix_asymptote = float(
            np.mean([float(item.get("decode_prefix_asymptote_pressure", 1.0)) for item in selected])
        )
        candidate_gpu_feedback_delta_score = float(
            np.mean([float(item.get("gpu_feedback_delta_score", 0.0)) for item in selected])
        )
        candidate_function_score = float(
            np.mean([float(item.get("sha256_function_score", 0.0)) for item in selected])
        )
        candidate_round_coupling = float(
            np.mean([float(item.get("sha256_round_coupling", 0.0)) for item in selected])
        )
        candidate_hash_target_phase_pressure = float(
            np.mean([float(item.get("hash_target_phase_pressure", 0.0)) for item in selected])
        )
        candidate_hash_target_phase_cost = float(
            np.mean([float(item.get("hash_target_phase_cost", 1.0)) for item in selected])
        )
        candidate_hash_target_prefix_lock = float(
            np.mean([float(item.get("hash_target_prefix_lock", 0.0)) for item in selected])
        )
        candidate_hash_target_window_coverage = float(
            np.mean([float(item.get("hash_target_window_coverage", 0.0)) for item in selected])
        )
        candidate_hash_target_band_alignment = float(
            np.mean([float(item.get("hash_target_band_alignment", 0.0)) for item in selected])
        )
        candidate_hash_target_flux_alignment = float(
            np.mean([float(item.get("hash_target_flux_alignment", 0.0)) for item in selected])
        )
        best_probe = dict(probed_candidates[0]) if probed_candidates else {}
        probe_pool_size = int(len(probed_candidates))
        probe_cluster_count = int(len(cluster_probe_profiles))
        dominant_probe_cluster = str(
            cluster_probe_profiles[0]["cluster_id"] if cluster_probe_profiles else ""
        )
        best_phase_length_pressure = float(best_probe.get("phase_length_pressure", 0.0))
        best_phase_alignment_score = float(
            best_probe.get("phase_alignment_score", best_phase_length_pressure)
        )
        best_phase_confinement_cost = float(
            best_probe.get("phase_confinement_cost", 0.0)
        )
        best_decode_integrity = float(best_probe.get("decode_integrity_score", 0.0))
        best_decode_entropy = float(best_probe.get("decode_symbol_entropy", 0.0))
        best_decode_phase_integrity = float(best_probe.get("decode_phase_integrity_score", 0.0))
        best_decode_phase_alignment = float(best_probe.get("decode_phase_delta_alignment", 0.0))
        best_decode_target_ring_alignment = float(best_probe.get("decode_target_ring_alignment", 0.0))
        best_decode_target_prefix_lock = float(best_probe.get("decode_target_prefix_lock", 0.0))
        best_decode_target_prefix_vector_alignment = float(
            best_probe.get("decode_target_prefix_vector_alignment", 0.0)
        )
        best_decode_target_prefix_vector_phase_pressure = float(
            best_probe.get("decode_target_prefix_vector_phase_pressure", 0.0)
        )
        best_decode_phase_orbital_alignment = float(
            best_probe.get("decode_phase_orbital_alignment", 0.0)
        )
        best_decode_phase_orbital_resonance = float(
            best_probe.get("decode_phase_orbital_resonance", 0.0)
        )
        best_decode_phase_orbital_stability = float(
            best_probe.get("decode_phase_orbital_stability", 0.0)
        )
        best_decode_prefix_asymptote = float(best_probe.get("decode_prefix_asymptote_pressure", 1.0))
        best_sequence_persistence = float(best_probe.get("sequence_persistence_score", 0.0))
        best_temporal_overlap = float(best_probe.get("temporal_index_overlap", 0.0))
        best_voltage_frequency_flux = float(best_probe.get("voltage_frequency_flux", 0.0))
        best_gpu_feedback_delta_score = float(best_probe.get("gpu_feedback_delta_score", 0.0))
        best_function_score = float(best_probe.get("sha256_function_score", 0.0))
        best_round_coupling = float(best_probe.get("sha256_round_coupling", 0.0))
        best_hash_target_phase_pressure = float(best_probe.get("hash_target_phase_pressure", 0.0))
        best_hash_target_phase_cost = float(best_probe.get("hash_target_phase_cost", 1.0))
        best_hash_target_prefix_lock = float(best_probe.get("hash_target_prefix_lock", 0.0))
        best_hash_target_window_coverage = float(best_probe.get("hash_target_window_coverage", 0.0))
        best_hash_target_band_alignment = float(best_probe.get("hash_target_band_alignment", 0.0))
        best_hash_target_flux_alignment = float(best_probe.get("hash_target_flux_alignment", 0.0))
        observation_freshness_gate = clamp01(
            float(gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0))
        )
        observation_gap_ms = float(gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0))
        dispatch_feedback_ratio = float(gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0))
        injection_observation_gap_ms = float(
            injection_delta_feedback.get("observation_gap_ms", 0.0)
        )
        injection_observation_freshness_gate = clamp01(
            float(injection_delta_feedback.get("observation_freshness_gate", 0.0))
        )
        injection_feedback_window_ms = float(
            injection_delta_feedback.get("feedback_window_ms", 0.0)
        )
        injection_response_gate = clamp01(
            float(injection_delta_feedback.get("response_gate", 0.0))
        )
        stable_target_phase_pressure = clamp01(
            0.34 * candidate_hash_target_phase_pressure
            + 0.18 * best_hash_target_phase_pressure
            + 0.14 * candidate_hash_target_prefix_lock
            + 0.08 * best_hash_target_prefix_lock
            + 0.10 * candidate_hash_target_flux_alignment
            + 0.06 * best_hash_target_flux_alignment
            + 0.06 * candidate_hash_target_window_coverage
            + 0.04 * best_hash_target_window_coverage
            + 0.06 * candidate_hash_target_band_alignment
            + 0.04 * best_hash_target_band_alignment
            + 0.06 * (1.0 - candidate_hash_target_phase_cost)
            + 0.04 * (1.0 - best_hash_target_phase_cost)
        )
        submission_anchor = load_submission_rate_anchor("BTC")
        verification_mode = "cpu_hash_check"
        verification_consistency = clamp01(
            (
                0.24 * candidate_function_score
            + 0.20 * best_function_score
            + 0.10 * candidate_round_coupling
            + 0.06 * best_round_coupling
            + 0.14 * candidate_sequence_persistence
            + 0.12 * candidate_temporal_overlap
            + 0.10 * candidate_decode_integrity
            + 0.06 * best_decode_integrity
            + 0.04 * candidate_decode_entropy
            + 0.06 * candidate_decode_phase_integrity
            + 0.04 * best_decode_phase_integrity
            + 0.04 * candidate_decode_phase_alignment
            + 0.04 * candidate_decode_target_prefix_lock
            + 0.03 * candidate_decode_target_prefix_vector_alignment
            + 0.02 * candidate_decode_target_prefix_vector_phase_pressure
            + 0.03 * candidate_decode_phase_orbital_alignment
            + 0.02 * candidate_decode_phase_orbital_resonance
            + 0.02 * candidate_decode_phase_orbital_stability
            + 0.03 * (1.0 - candidate_decode_prefix_asymptote)
            + 0.06 * candidate_hash_target_phase_pressure
            + 0.04 * best_hash_target_phase_pressure
            + 0.04 * candidate_hash_target_prefix_lock
            + 0.10 * candidate_gpu_feedback_delta_score
            + 0.10 * observation_freshness_gate
        )
            * (0.68 + 0.32 * observation_freshness_gate)
        )
        production_unlock_ready = False
        shares: list[dict[str, Any]] = []
        valid_shares: list[dict[str, Any]] = []
        prototype_valid_count = 0
        network_valid_count = 0
        valid_selected: list[dict[str, Any]] = []
        for order, candidate in enumerate(selected):
            if str(candidate.get("probe_hash_hex", "")):
                share = {
                    "header": str(candidate.get("probe_header_hex", header_hex)),
                    "first_hash_hex": str(candidate.get("probe_first_hash_hex", "")),
                    "hash_hex": str(candidate.get("probe_hash_hex", "")),
                }
            else:
                share = sha256d_compute_share({"header_hex": header_hex}, int(candidate["nonce"]))
            share_hash_metrics = hash_target_phase_metrics(
                str(share.get("hash_hex", "")),
                str(network_target_hex),
            )
            share_payload = {
                "job_id": f"pulse-{pulse_index:02d}",
                "extranonce2": f"{(pulse_seed + order) & 0xFFFFFFFF:08x}",
                "ntime": f"{timestamp:08x}",
                "nonce": f"{int(candidate['nonce']) & 0xFFFFFFFF:08x}",
                "header_hex": str(share.get("header", header_hex)),
                "first_hash_hex": str(share.get("first_hash_hex", "")),
                "hash_hex": str(share.get("hash_hex", "")),
                "target_hex": network_target_hex,
                "prototype_target_hex": prototype_target_hex,
                "network_target_hex": network_target_hex,
                "packet_id": int(candidate["packet_id"]),
                "cluster_id": str(candidate["cluster_id"]),
                "basin_id": extract_basin_id(
                    str(candidate.get("cluster_id", "")),
                    default=dominant_basin_id,
                ),
                "coherence_peak": float(candidate["coherence_peak"]),
                "target_alignment": float(candidate["target_alignment"]),
                "target_interval": float(candidate["target_interval"]),
                "phase_length_pressure": float(candidate.get("phase_length_pressure", 0.0)),
                "phase_alignment_score": float(
                    candidate.get("phase_alignment_score", candidate.get("phase_length_pressure", 0.0))
                ),
                "phase_confinement_cost": float(candidate.get("phase_confinement_cost", 0.0)),
                "stable_target_phase_pressure": float(stable_target_phase_pressure),
                "hash_target_phase_pressure": float(
                    share_hash_metrics.get("hash_target_phase_pressure", candidate.get("hash_target_phase_pressure", 0.0))
                ),
                "hash_target_phase_cost": float(
                    share_hash_metrics.get("hash_target_phase_cost", candidate.get("hash_target_phase_cost", 1.0))
                ),
                "hash_target_prefix_lock": float(
                    share_hash_metrics.get("hash_target_prefix_lock", candidate.get("hash_target_prefix_lock", 0.0))
                ),
                "hash_target_window_coverage": float(
                    share_hash_metrics.get(
                        "hash_target_window_coverage",
                        candidate.get("hash_target_window_coverage", 0.0),
                    )
                ),
                "hash_target_band_alignment": float(
                    share_hash_metrics.get(
                        "hash_target_band_alignment",
                        candidate.get("hash_target_band_alignment", 0.0),
                    )
                ),
                "hash_target_flux_alignment": float(
                    share_hash_metrics.get(
                        "hash_target_flux_alignment",
                        candidate.get("hash_target_flux_alignment", 0.0),
                    )
                ),
                "hash_target_frontier_slack": float(
                    share_hash_metrics.get(
                        "hash_target_frontier_slack",
                        candidate.get("hash_target_frontier_slack", 0.0),
                    )
                ),
                "btc_phase_alignment": float(
                    candidate.get("btc_phase_alignment", candidate.get("btc_phase_pressure", 0.0))
                ),
                "btc_phase_cost": float(candidate.get("btc_phase_cost", 0.0)),
                "phase_length_index": int(candidate.get("phase_length_index", 0)),
                "temporal_sequence_index": int(candidate.get("temporal_sequence_index", 0)),
                "sequence_persistence_score": float(candidate.get("sequence_persistence_score", 0.0)),
                "temporal_index_overlap": float(candidate.get("temporal_index_overlap", 0.0)),
                "voltage_frequency_flux": float(candidate.get("voltage_frequency_flux", 0.0)),
                "decoded_sequence_hex": str(candidate.get("decoded_sequence_hex", "")),
                "decoded_sequence_utf8_preview": str(
                    candidate.get("decoded_sequence_utf8_preview", "")
                ),
                "decode_integrity_score": float(candidate.get("decode_integrity_score", 0.0)),
                "decode_symbol_entropy": float(candidate.get("decode_symbol_entropy", 0.0)),
                "decode_transition_ratio": float(candidate.get("decode_transition_ratio", 0.0)),
                "decode_longest_run_ratio": float(candidate.get("decode_longest_run_ratio", 1.0)),
                "decode_temporal_coverage": float(candidate.get("decode_temporal_coverage", 0.0)),
                "decode_phase_delta_alignment": float(
                    candidate.get("decode_phase_delta_alignment", 0.0)
                ),
                "decode_phase_delta_balance": float(
                    candidate.get("decode_phase_delta_balance", 0.0)
                ),
                "decode_phase_integrity_score": float(
                    candidate.get("decode_phase_integrity_score", 0.0)
                ),
                "decode_phase_orbital_trace_vector": [
                    float(value)
                    for value in list(candidate.get("decode_phase_orbital_trace_vector", []) or [])
                ],
                "decode_phase_orbital_alignment": float(
                    candidate.get("decode_phase_orbital_alignment", 0.0)
                ),
                "decode_phase_orbital_resonance": float(
                    candidate.get("decode_phase_orbital_resonance", 0.0)
                ),
                "decode_phase_orbital_stability": float(
                    candidate.get("decode_phase_orbital_stability", 0.0)
                ),
                "sha256_function_score": float(
                    share.get("sha256_function_score", candidate.get("sha256_function_score", 0.0))
                ),
                "sha256_round_coupling": float(
                    share.get("sha256_round_coupling", candidate.get("sha256_round_coupling", 0.0))
                ),
                "sha256_first_entropy": float(
                    share.get("sha256_first_entropy", candidate.get("sha256_first_entropy", 0.0))
                ),
                "sha256_final_entropy": float(
                    share.get("sha256_final_entropy", candidate.get("sha256_final_entropy", 0.0))
                ),
                "sha256_transition_ratio": float(
                    share.get("sha256_transition_ratio", candidate.get("sha256_transition_ratio", 0.0))
                ),
                "sha256_bit_balance": float(
                    share.get("sha256_bit_balance", candidate.get("sha256_bit_balance", 0.0))
                ),
                "sha256_lane_balance": float(
                    share.get("sha256_lane_balance", candidate.get("sha256_lane_balance", 0.0))
                ),
                "observation_freshness_gate": float(
                    candidate.get("gpu_feedback_observation_freshness", observation_freshness_gate)
                ),
            }
            share_payload["prototype_valid"] = bool(
                pow_hex_leq_target(str(share_payload["hash_hex"]), str(network_target_hex))
            )
            share_payload["network_valid"] = bool(
                pow_hex_leq_target(str(share_payload["hash_hex"]), str(network_target_hex))
            )
            shares.append(share_payload)
            if bool(share_payload["network_valid"]):
                network_valid_count += 1
                prototype_valid_count += 1
                valid_shares.append(share_payload)
                valid_selected.append(candidate)
        if valid_shares:
            deduped_valid_shares: list[dict[str, Any]] = []
            deduped_valid_selected: list[dict[str, Any]] = []
            seen_share_keys: set[str] = set()
            for candidate, payload in zip(valid_selected, valid_shares):
                share_key = f"{str(payload.get('header_hex', ''))}:{str(payload.get('nonce', ''))}"
                if share_key in seen_share_keys:
                    continue
                seen_share_keys.add(share_key)
                deduped_valid_shares.append(payload)
                deduped_valid_selected.append(candidate)
            valid_shares = deduped_valid_shares
            valid_selected = deduped_valid_selected
            network_valid_count = len(valid_shares)
            prototype_valid_count = network_valid_count
        production_unlock_ready = bool(
            network_valid_count > 0
            and verification_consistency >= 0.82
            and candidate_function_score >= 0.56
            and best_function_score >= 0.60
            and observation_freshness_gate >= 0.72
        )

        pulse_state = {
            "yield_count": len(valid_shares),
            "candidate_yield_count": candidate_yield_count,
            "prototype_valid_count": prototype_valid_count,
            "network_valid_count": network_valid_count,
            "coherence_peak": max((item["coherence_peak"] for item in valid_selected), default=0.0),
            "candidate_coherence_peak": candidate_coherence_peak,
            "target_alignment": candidate_target_alignment,
            "target_interval": candidate_target_interval,
            "interference_resonance": candidate_interference_resonance,
            "cascade_activation": candidate_cascade_activation,
            "calibration_readiness": float(simulation_field_state.get("calibration_readiness", 0.0)),
            "entry_trigger": bool(simulation_field_state.get("entry_trigger", False)),
            "temporal_residual": candidate_temporal_residual,
            "clamp_pressure": candidate_clamp_pressure,
            "candidate_phase_length_pressure": candidate_phase_length_pressure,
            "candidate_phase_alignment_score": candidate_phase_alignment_score,
            "candidate_phase_confinement_cost": candidate_phase_confinement_cost,
            "stable_target_phase_pressure": stable_target_phase_pressure,
            "candidate_sequence_persistence": candidate_sequence_persistence,
            "candidate_temporal_overlap": candidate_temporal_overlap,
            "candidate_voltage_frequency_flux": candidate_voltage_frequency_flux,
            "candidate_gpu_feedback_delta_score": candidate_gpu_feedback_delta_score,
            "candidate_decode_integrity": candidate_decode_integrity,
            "candidate_decode_entropy": candidate_decode_entropy,
            "candidate_decode_phase_integrity": candidate_decode_phase_integrity,
            "candidate_decode_phase_alignment": candidate_decode_phase_alignment,
            "candidate_decode_target_ring_alignment": candidate_decode_target_ring_alignment,
            "candidate_decode_target_prefix_lock": candidate_decode_target_prefix_lock,
            "candidate_decode_target_prefix_vector_alignment": candidate_decode_target_prefix_vector_alignment,
            "candidate_decode_target_prefix_vector_phase_pressure": candidate_decode_target_prefix_vector_phase_pressure,
            "candidate_decode_phase_orbital_alignment": candidate_decode_phase_orbital_alignment,
            "candidate_decode_phase_orbital_resonance": candidate_decode_phase_orbital_resonance,
            "candidate_decode_phase_orbital_stability": candidate_decode_phase_orbital_stability,
            "candidate_decode_prefix_asymptote": candidate_decode_prefix_asymptote,
            "candidate_hash_target_phase_pressure": candidate_hash_target_phase_pressure,
            "candidate_hash_target_phase_cost": candidate_hash_target_phase_cost,
            "candidate_hash_target_prefix_lock": candidate_hash_target_prefix_lock,
            "candidate_hash_target_window_coverage": candidate_hash_target_window_coverage,
            "candidate_hash_target_band_alignment": candidate_hash_target_band_alignment,
            "candidate_hash_target_flux_alignment": candidate_hash_target_flux_alignment,
            "candidate_function_score": candidate_function_score,
            "candidate_round_coupling": candidate_round_coupling,
            "probe_pool_size": probe_pool_size,
            "probe_cluster_count": probe_cluster_count,
            "dominant_probe_cluster": dominant_probe_cluster,
            "best_phase_length_pressure": best_phase_length_pressure,
            "best_phase_alignment_score": best_phase_alignment_score,
            "best_phase_confinement_cost": best_phase_confinement_cost,
            "best_decode_integrity": best_decode_integrity,
            "best_decode_entropy": best_decode_entropy,
            "best_decode_phase_integrity": best_decode_phase_integrity,
            "best_decode_phase_alignment": best_decode_phase_alignment,
            "best_decode_target_ring_alignment": best_decode_target_ring_alignment,
            "best_decode_target_prefix_lock": best_decode_target_prefix_lock,
            "best_decode_target_prefix_vector_alignment": best_decode_target_prefix_vector_alignment,
            "best_decode_target_prefix_vector_phase_pressure": best_decode_target_prefix_vector_phase_pressure,
            "best_decode_phase_orbital_alignment": best_decode_phase_orbital_alignment,
            "best_decode_phase_orbital_resonance": best_decode_phase_orbital_resonance,
            "best_decode_phase_orbital_stability": best_decode_phase_orbital_stability,
            "best_decode_prefix_asymptote": best_decode_prefix_asymptote,
            "best_sequence_persistence": best_sequence_persistence,
            "best_temporal_overlap": best_temporal_overlap,
            "best_voltage_frequency_flux": best_voltage_frequency_flux,
            "best_gpu_feedback_delta_score": best_gpu_feedback_delta_score,
            "best_function_score": best_function_score,
            "best_round_coupling": best_round_coupling,
            "best_hash_target_phase_pressure": best_hash_target_phase_pressure,
            "best_hash_target_phase_cost": best_hash_target_phase_cost,
            "best_hash_target_prefix_lock": best_hash_target_prefix_lock,
            "best_hash_target_window_coverage": best_hash_target_window_coverage,
            "best_hash_target_band_alignment": best_hash_target_band_alignment,
            "best_hash_target_flux_alignment": best_hash_target_flux_alignment,
            "injection_feedback_window_ms": injection_feedback_window_ms,
            "injection_observation_gap_ms": injection_observation_gap_ms,
            "injection_observation_freshness_gate": injection_observation_freshness_gate,
            "injection_response_gate": injection_response_gate,
            "observation_gap_ms": observation_gap_ms,
            "observation_freshness_gate": observation_freshness_gate,
            "dispatch_feedback_ratio": dispatch_feedback_ratio,
            "cuda_kernel_telemetry": cuda_kernel_telemetry,
            "verification_mode": verification_mode,
            "verification_consistency": verification_consistency,
            "production_unlock_ready": production_unlock_ready,
            "vector_magnitude": float(effective_vector.get("spatial_magnitude", 0.0)),
            "temporal_projection": temporal_confinement,
            "field_pressure": field_pressure,
            "larger_field_exposure": larger_field_exposure,
            "dominant_basin": dominant_basin_id,
            "substrate_material": str(
                kernel_execution_event.get(
                    "substrate_material",
                    simulation_field_state.get("substrate_material", "silicon_wafer"),
                )
            ),
            "silicon_reference_source": str(
                kernel_execution_event.get(
                    "silicon_reference_source",
                    simulation_field_state.get("silicon_reference_source", NIST_REFERENCE.name),
                )
            ),
            "compute_regime": str(
                kernel_execution_event.get(
                    "compute_regime",
                    simulation_field_state.get("compute_regime", "classical_calibration"),
                )
            ),
            "vector_harmonic_gate": float(
                kernel_execution_event.get(
                    "vector_harmonic_gate",
                    simulation_field_state.get("vector_harmonic_gate", 0.0),
                )
            ),
            "harmonic_compute_weight": float(
                kernel_execution_event.get(
                    "harmonic_compute_weight",
                    simulation_field_state.get("harmonic_compute_weight", 0.0),
                )
            ),
            "trace_support": float(
                kernel_execution_event.get(
                    "trace_support",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_support", 0.0),
                )
            ),
            "trace_resonance": float(
                kernel_execution_event.get(
                    "trace_resonance",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_resonance", 0.0),
                )
            ),
            "trace_alignment": float(
                kernel_execution_event.get(
                    "trace_alignment",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_alignment", 0.0),
                )
            ),
            "trace_relative_spatial_field": [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "trace_relative_spatial_field", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "coherent_noise_axis_vector": [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "coherent_noise_axis_vector", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "noise_resonance_nodes": [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "noise_resonance_nodes", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "drift_compensation_vector": [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "drift_compensation_vector", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "noise_resonance_gate": float(
                simulation_field_state.get("substrate_trace_state", {}).get(
                    "noise_resonance_gate", 0.0
                )
            ),
            "environment_turbulence": float(
                simulation_field_state.get("substrate_trace_state", {}).get(
                    "environment_turbulence", 0.0
                )
            ),
            "trace_vram_resident": bool(
                kernel_execution_event.get(
                    "trace_vram_resident",
                    simulation_field_state.get("substrate_trace_vram", {}).get("resident", False),
                )
            ),
            "trace_vram_updates": int(
                kernel_execution_event.get(
                    "trace_vram_updates",
                    simulation_field_state.get("substrate_trace_vram", {}).get("update_count", 0),
                )
            ),
            "gpu_feedback_source": str(
                post_cuda_gpu_feedback.get(
                    "source",
                    simulation_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                )
            ),
            "gpu_injection_feedback_source": str(
                injection_gpu_feedback.get(
                    "source",
                    simulation_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                )
            ),
            "path_equivalence_error": float(manifold_diagnostics.get("path_equivalence_error", 0.0)),
            "temporal_ordering_delta": float(manifold_diagnostics.get("temporal_ordering_delta", 0.0)),
            "basis_rotation_residual": float(manifold_diagnostics.get("basis_rotation_residual", 0.0)),
        }
        ledger_delta = compute_ledger_delta(
            prev_state,
            pulse_state,
            {
                "ledger_delta_operator": process_ops.get(
                    "ledger_delta_operator", "compute_ledger_delta"
                )
            },
        )
        if ledger_delta["coherence_drop"] > 0.03 and adaptive_depth > 3:
            adaptive_depth -= 1
        elif ledger_delta["coherence_drop"] < 0.005 and adaptive_depth < 5 and difficulty_norm > 0.55:
            adaptive_depth += 1
        if ledger_delta["coherence_drop"] > 0.02 and adaptive_workers < max_worker_count:
            adaptive_workers += 1
        elif pulse_state["yield_count"] > 40 and adaptive_workers > min_worker_count:
            adaptive_workers -= 1

        if not accept_state(
            prev_state,
            pulse_state,
            ledger_delta,
            {"gate_coherence": gate_coherence, "amplitude_cap": amplitude_cap},
        ):
            sink_state = make_sink_state(
                {
                    "pulse_id": pulse_index,
                    "worker_count": pulse_workers,
                    "nbits": nbits_hex,
                    "target_hex": target_hex,
                    "network_target_hex": network_target_hex,
                    "field_pressure": field_pressure,
                    "larger_field_exposure": larger_field_exposure,
                    "dominant_basin": dominant_basin_id,
                },
                {"amplitude_cap": amplitude_cap},
            )
            sink_state["psi_encode"] = psi_encode
            sink_state["temporal_manifold"] = temporal_manifold
            sink_state["effective_vector"] = effective_vector
            sink_state["manifold_diagnostics"] = manifold_diagnostics
            sink_state["vector_magnitude"] = float(effective_vector.get("spatial_magnitude", 0.0))
            sink_state["temporal_projection"] = temporal_confinement
            sink_state["field_pressure"] = field_pressure
            sink_state["larger_field_exposure"] = larger_field_exposure
            sink_state["dominant_basin"] = dominant_basin_id
            sink_state["substrate_material"] = str(
                kernel_execution_event.get(
                    "substrate_material",
                    simulation_field_state.get("substrate_material", "silicon_wafer"),
                )
            )
            sink_state["silicon_reference_source"] = str(
                kernel_execution_event.get(
                    "silicon_reference_source",
                    simulation_field_state.get("silicon_reference_source", NIST_REFERENCE.name),
                )
            )
            sink_state["compute_regime"] = str(
                kernel_execution_event.get(
                    "compute_regime",
                    simulation_field_state.get("compute_regime", "classical_calibration"),
                )
            )
            sink_state["vector_harmonic_gate"] = float(
                kernel_execution_event.get(
                    "vector_harmonic_gate",
                    simulation_field_state.get("vector_harmonic_gate", 0.0),
                )
            )
            sink_state["harmonic_compute_weight"] = float(
                kernel_execution_event.get(
                    "harmonic_compute_weight",
                    simulation_field_state.get("harmonic_compute_weight", 0.0),
                )
            )
            sink_state["trace_support"] = float(
                kernel_execution_event.get(
                    "trace_support",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_support", 0.0),
                )
            )
            sink_state["trace_resonance"] = float(
                kernel_execution_event.get(
                    "trace_resonance",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_resonance", 0.0),
                )
            )
            sink_state["trace_alignment"] = float(
                kernel_execution_event.get(
                    "trace_alignment",
                    simulation_field_state.get("substrate_trace_state", {}).get("trace_alignment", 0.0),
                )
            )
            sink_state["trace_relative_spatial_field"] = [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "trace_relative_spatial_field", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ]
            sink_state["coherent_noise_axis_vector"] = [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "coherent_noise_axis_vector", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ]
            sink_state["noise_resonance_nodes"] = [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "noise_resonance_nodes", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ]
            sink_state["drift_compensation_vector"] = [
                float(value)
                for value in list(
                    simulation_field_state.get("substrate_trace_state", {}).get(
                        "drift_compensation_vector", [0.0, 0.0, 0.0, 0.0]
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ]
            sink_state["noise_resonance_gate"] = float(
                simulation_field_state.get("substrate_trace_state", {}).get(
                    "noise_resonance_gate", 0.0
                )
            )
            sink_state["environment_turbulence"] = float(
                simulation_field_state.get("substrate_trace_state", {}).get(
                    "environment_turbulence", 0.0
                )
            )
            sink_state["trace_vram_resident"] = bool(
                kernel_execution_event.get(
                    "trace_vram_resident",
                    simulation_field_state.get("substrate_trace_vram", {}).get("resident", False),
                )
            )
            sink_state["trace_vram_updates"] = int(
                kernel_execution_event.get(
                    "trace_vram_updates",
                    simulation_field_state.get("substrate_trace_vram", {}).get("update_count", 0),
                )
            )
            sink_state["gpu_feedback_source"] = str(
                post_cuda_gpu_feedback.get(
                    "source",
                    simulation_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                )
            )
            sink_state["gpu_injection_feedback_source"] = str(
                injection_gpu_feedback.get(
                    "source",
                    simulation_field_state.get("gpu_pulse_feedback", {}).get(
                        "source", "vector_runtime_feedback"
                    ),
                )
            )
            sink_state["candidate_yield_count"] = candidate_yield_count
            sink_state["prototype_valid_count"] = prototype_valid_count
            sink_state["network_valid_count"] = network_valid_count
            sink_state["candidate_coherence_peak"] = candidate_coherence_peak
            sink_state["target_alignment"] = candidate_target_alignment
            sink_state["target_interval"] = candidate_target_interval
            sink_state["interference_resonance"] = candidate_interference_resonance
            sink_state["cascade_activation"] = candidate_cascade_activation
            sink_state["candidate_function_score"] = candidate_function_score
            sink_state["candidate_round_coupling"] = candidate_round_coupling
            sink_state["candidate_phase_length_pressure"] = candidate_phase_length_pressure
            sink_state["candidate_phase_alignment_score"] = candidate_phase_alignment_score
            sink_state["candidate_phase_confinement_cost"] = candidate_phase_confinement_cost
            sink_state["stable_target_phase_pressure"] = stable_target_phase_pressure
            sink_state["candidate_sequence_persistence"] = candidate_sequence_persistence
            sink_state["candidate_temporal_overlap"] = candidate_temporal_overlap
            sink_state["candidate_voltage_frequency_flux"] = candidate_voltage_frequency_flux
            sink_state["candidate_gpu_feedback_delta_score"] = candidate_gpu_feedback_delta_score
            sink_state["candidate_decode_integrity"] = candidate_decode_integrity
            sink_state["candidate_decode_entropy"] = candidate_decode_entropy
            sink_state["candidate_decode_phase_integrity"] = candidate_decode_phase_integrity
            sink_state["candidate_decode_phase_alignment"] = candidate_decode_phase_alignment
            sink_state["candidate_decode_target_ring_alignment"] = candidate_decode_target_ring_alignment
            sink_state["candidate_decode_target_prefix_lock"] = candidate_decode_target_prefix_lock
            sink_state["candidate_decode_target_prefix_vector_alignment"] = (
                candidate_decode_target_prefix_vector_alignment
            )
            sink_state["candidate_decode_target_prefix_vector_phase_pressure"] = (
                candidate_decode_target_prefix_vector_phase_pressure
            )
            sink_state["candidate_decode_phase_orbital_alignment"] = (
                candidate_decode_phase_orbital_alignment
            )
            sink_state["candidate_decode_phase_orbital_resonance"] = (
                candidate_decode_phase_orbital_resonance
            )
            sink_state["candidate_decode_phase_orbital_stability"] = (
                candidate_decode_phase_orbital_stability
            )
            sink_state["candidate_decode_prefix_asymptote"] = candidate_decode_prefix_asymptote
            sink_state["candidate_hash_target_phase_pressure"] = candidate_hash_target_phase_pressure
            sink_state["candidate_hash_target_phase_cost"] = candidate_hash_target_phase_cost
            sink_state["candidate_hash_target_prefix_lock"] = candidate_hash_target_prefix_lock
            sink_state["candidate_hash_target_window_coverage"] = candidate_hash_target_window_coverage
            sink_state["candidate_hash_target_band_alignment"] = candidate_hash_target_band_alignment
            sink_state["candidate_hash_target_flux_alignment"] = candidate_hash_target_flux_alignment
            sink_state["submission_anchor_rate_per_second"] = float(
                submission_anchor.get("allowed_rate_per_second", 0.0)
            )
            sink_state["submission_tick_duration_s"] = float(
                submission_anchor.get("tick_duration_s", 0.5)
            )
            sink_state["submission_jitter_fraction"] = float(
                submission_anchor.get("jitter_fraction", 0.18)
            )
            sink_state["prototype_target_hex"] = prototype_target_hex
            sink_state["prototype_target_multiplier"] = float(
                prototype_share_target.get("target_multiplier", 1.0)
            )
            sink_state["prototype_target_probability"] = float(
                prototype_share_target.get("target_probability", 0.0)
            )
            sink_state["prototype_target_valid_goal"] = int(
                prototype_share_target.get("desired_valid_count", 0)
            )
            sink_state["selection_mode"] = "network_only"
            sink_state["feedback_temperature_norm"] = float(post_cuda_gpu_feedback.get("temperature_norm", 0.0))
            sink_state["feedback_environment_pressure"] = float(
                post_cuda_gpu_feedback.get("environment_pressure", 0.0)
            )
            sink_state["feedback_roundtrip_latency_ms"] = float(
                gpu_pulse_delta_feedback.get("roundtrip_latency_ms", 0.0)
            )
            sink_state["injection_feedback_window_ms"] = float(
                injection_delta_feedback.get("feedback_window_ms", 0.0)
            )
            sink_state["injection_observation_gap_ms"] = float(
                injection_delta_feedback.get("observation_gap_ms", 0.0)
            )
            sink_state["injection_observation_freshness_gate"] = float(
                injection_delta_feedback.get("observation_freshness_gate", 0.0)
            )
            sink_state["injection_response_gate"] = float(
                injection_delta_feedback.get("response_gate", 0.0)
            )
            sink_state["observation_gap_ms"] = float(
                gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)
            )
            sink_state["observation_freshness_gate"] = float(
                gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)
            )
            sink_state["dispatch_feedback_ratio"] = float(
                gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0)
            )
            sink_state["feedback_latency_gate"] = float(gpu_pulse_delta_feedback.get("latency_gate", 0.0))
            sink_state["probe_pool_size"] = probe_pool_size
            sink_state["probe_cluster_count"] = probe_cluster_count
            sink_state["dominant_probe_cluster"] = dominant_probe_cluster
            sink_state["best_phase_length_pressure"] = best_phase_length_pressure
            sink_state["best_phase_alignment_score"] = best_phase_alignment_score
            sink_state["best_phase_confinement_cost"] = best_phase_confinement_cost
            sink_state["best_decode_integrity"] = best_decode_integrity
            sink_state["best_decode_entropy"] = best_decode_entropy
            sink_state["best_decode_phase_integrity"] = best_decode_phase_integrity
            sink_state["best_decode_phase_alignment"] = best_decode_phase_alignment
            sink_state["best_decode_target_ring_alignment"] = best_decode_target_ring_alignment
            sink_state["best_decode_target_prefix_lock"] = best_decode_target_prefix_lock
            sink_state["best_decode_target_prefix_vector_alignment"] = (
                best_decode_target_prefix_vector_alignment
            )
            sink_state["best_decode_target_prefix_vector_phase_pressure"] = (
                best_decode_target_prefix_vector_phase_pressure
            )
            sink_state["best_decode_phase_orbital_alignment"] = (
                best_decode_phase_orbital_alignment
            )
            sink_state["best_decode_phase_orbital_resonance"] = (
                best_decode_phase_orbital_resonance
            )
            sink_state["best_decode_phase_orbital_stability"] = (
                best_decode_phase_orbital_stability
            )
            sink_state["best_decode_prefix_asymptote"] = best_decode_prefix_asymptote
            sink_state["best_sequence_persistence"] = best_sequence_persistence
            sink_state["best_temporal_overlap"] = best_temporal_overlap
            sink_state["best_voltage_frequency_flux"] = best_voltage_frequency_flux
            sink_state["best_gpu_feedback_delta_score"] = best_gpu_feedback_delta_score
            sink_state["best_function_score"] = best_function_score
            sink_state["best_round_coupling"] = best_round_coupling
            sink_state["best_hash_target_phase_pressure"] = best_hash_target_phase_pressure
            sink_state["best_hash_target_phase_cost"] = best_hash_target_phase_cost
            sink_state["best_hash_target_prefix_lock"] = best_hash_target_prefix_lock
            sink_state["best_hash_target_window_coverage"] = best_hash_target_window_coverage
            sink_state["best_hash_target_band_alignment"] = best_hash_target_band_alignment
            sink_state["best_hash_target_flux_alignment"] = best_hash_target_flux_alignment
            sink_state["cuda_kernel_telemetry"] = cuda_kernel_telemetry
            sink_state["verification_mode"] = verification_mode
            sink_state["verification_consistency"] = verification_consistency
            sink_state["production_unlock_ready"] = production_unlock_ready
            sink_state["calibration_readiness"] = float(simulation_field_state.get("calibration_readiness", 0.0))
            sink_state["entry_trigger"] = bool(simulation_field_state.get("entry_trigger", False))
            sink_state["simulation_field_state"] = simulation_field_state
            sink_state["interference_field"] = interference_field
            sink_state["kernel_execution_event"] = kernel_execution_event
            sink_state["motif_consistency"] = float(simulation_field_state.get("motif_consistency", 0.0))
            sink_state["motif_repeat_count"] = int(simulation_field_state.get("motif_repeat_count", 0))
            sink_state["motif_energy"] = float(simulation_field_state.get("motif_energy", 0.0))
            sink_state["path_equivalence_error"] = float(manifold_diagnostics.get("path_equivalence_error", 1.0))
            sink_state["temporal_ordering_delta"] = float(manifold_diagnostics.get("temporal_ordering_delta", 1.0))
            sink_state["basis_rotation_residual"] = float(manifold_diagnostics.get("basis_rotation_residual", 1.0))
            pulse_batches.append(sink_state)
            prev_state = sink_state
            continue

        deduped_valid_shares: list[dict[str, Any]] = []
        seen_share_keys: set[str] = set()
        for payload in valid_shares:
            share_key = f"{str(payload.get('header_hex', ''))}:{str(payload.get('nonce', ''))}"
            if share_key in seen_share_keys:
                continue
            seen_share_keys.add(share_key)
            deduped_valid_shares.append(payload)
        valid_shares = deduped_valid_shares
        basin_ids = sorted(
            {
                str(payload.get("basin_id", dominant_basin_id or "electron_basin"))
                for payload in valid_shares
            }
            or {str(dominant_basin_id or "electron_basin")}
        )
        worker_batches = [
            {
                "worker_id": f"substrate-{worker_index + 1:02d}",
                "queue_key": f"research_confinement/btc_queue/substrate-{worker_index + 1:02d}",
                "basin_id": str(basin_ids[worker_index % len(basin_ids)]),
                "submit_payloads": [],
            }
            for worker_index in range(pulse_workers)
        ]
        for payload in valid_shares:
            payload_basin = str(payload.get("basin_id", dominant_basin_id or "electron_basin"))
            eligible_workers = [
                worker for worker in worker_batches if str(worker.get("basin_id", "")) == payload_basin
            ]
            if not eligible_workers:
                eligible_workers = worker_batches
            target_worker = min(
                eligible_workers,
                key=lambda worker: (
                    len(list(worker.get("submit_payloads", []) or [])),
                    str(worker.get("worker_id", "")),
                ),
            )
            queue_index = int(len(list(target_worker.get("submit_payloads", []) or [])))
            tick_duration_s = float(submission_anchor.get("tick_duration_s", 0.5))
            jitter_fraction = float(submission_anchor.get("jitter_fraction", 0.18))
            decode_phase = wrap_turns(
                0.24 * float(payload.get("decode_temporal_coverage", 0.0))
                + 0.20 * float(payload.get("decode_transition_ratio", 0.0))
                + 0.18 * float(payload.get("sequence_persistence_score", 0.0))
                + 0.16 * float(payload.get("temporal_index_overlap", 0.0))
                + 0.12 * float(payload.get("sha256_round_coupling", 0.0))
                + 0.10 * float(payload.get("hash_target_phase_pressure", 0.0))
                + 0.08 * float(payload.get("hash_target_flux_alignment", 0.0))
                + 0.06 * float(payload.get("stable_target_phase_pressure", stable_target_phase_pressure))
            )
            jitter_window_s = tick_duration_s * jitter_fraction
            submit_delay_s = max(
                0.0,
                queue_index * tick_duration_s
                + jitter_window_s * (0.5 + 0.5 * math.sin(decode_phase * math.tau)),
            )
            payload["submit_delay_s"] = float(submit_delay_s)
            payload["submit_tick_offset_s"] = float(tick_duration_s)
            payload["submission_anchor_rate_per_second"] = float(
                submission_anchor.get("allowed_rate_per_second", 0.0)
            )
            payload["submission_jitter_fraction"] = float(jitter_fraction)
            payload["stale_after_job_id"] = str(payload.get("job_id", ""))
            target_worker["submit_payloads"].append(payload)
        for worker in worker_batches:
            queue_depth = len(worker["submit_payloads"])
            worker["batch_size"] = queue_depth
            worker["queue_depth"] = queue_depth
            worker["submit_preview"] = [payload["nonce"] for payload in worker["submit_payloads"][:4]]
            worker["frequency_patterns"] = [
                str(
                    payload.get(
                        "decoded_sequence_utf8_preview",
                        payload.get("decoded_sequence_hex", ""),
                    )
                )
                for payload in worker["submit_payloads"][:4]
            ]
            worker["schedule_preview_s"] = [
                float(payload.get("submit_delay_s", 0.0)) for payload in worker["submit_payloads"][:4]
            ]
            worker["submission_anchor_rate_per_second"] = float(
                submission_anchor.get("allowed_rate_per_second", 0.0)
            )
            worker["tick_duration_s"] = float(submission_anchor.get("tick_duration_s", 0.5))
            worker["submission_jitter_fraction"] = float(
                submission_anchor.get("jitter_fraction", 0.18)
            )
        deduped_share_count = int(len(valid_shares))

        pulse_batches.append(
            {
                "pulse_id": pulse_index,
                "status": "accepted",
                "seed_nonce": f"{pulse_seed:08x}",
                "header_hex": header_hex,
                "nbits": nbits_hex,
                "target_hex": target_hex,
                "network_target_hex": network_target_hex,
                "baseline_frequency": baseline_frequency_hz,
                "amplitude_cap": amplitude_cap,
                "carrier_count": pulse_carrier_count,
                "nesting_depth": pulse_depth,
                "state_capacity": pulse_carrier_count ** pulse_depth,
                "coherence_peak": float(pulse_state["coherence_peak"]),
                "candidate_coherence_peak": float(candidate_coherence_peak),
                "target_alignment": candidate_target_alignment,
                "target_interval": candidate_target_interval,
                "interference_resonance": candidate_interference_resonance,
                "cascade_activation": candidate_cascade_activation,
                "coherence_drop": float(ledger_delta["coherence_drop"]),
                "yield_count": len(valid_shares),
                "candidate_yield_count": candidate_yield_count,
                "prototype_valid_count": prototype_valid_count,
                "network_valid_count": network_valid_count,
                "candidate_function_score": candidate_function_score,
                "candidate_round_coupling": candidate_round_coupling,
                "deduped_share_count": deduped_share_count,
                "submission_anchor_rate_per_second": float(
                    submission_anchor.get("allowed_rate_per_second", 0.0)
                ),
                "submission_tick_duration_s": float(
                    submission_anchor.get("tick_duration_s", 0.5)
                ),
                "submission_jitter_fraction": float(
                    submission_anchor.get("jitter_fraction", 0.18)
                ),
                "candidate_phase_length_pressure": candidate_phase_length_pressure,
                "candidate_phase_alignment_score": candidate_phase_alignment_score,
                "candidate_phase_confinement_cost": candidate_phase_confinement_cost,
                "stable_target_phase_pressure": stable_target_phase_pressure,
                "candidate_sequence_persistence": candidate_sequence_persistence,
                "candidate_temporal_overlap": candidate_temporal_overlap,
                "candidate_voltage_frequency_flux": candidate_voltage_frequency_flux,
                "candidate_gpu_feedback_delta_score": candidate_gpu_feedback_delta_score,
                "candidate_decode_integrity": candidate_decode_integrity,
                "candidate_decode_entropy": candidate_decode_entropy,
                "candidate_hash_target_phase_pressure": candidate_hash_target_phase_pressure,
                "candidate_hash_target_phase_cost": candidate_hash_target_phase_cost,
                "candidate_hash_target_prefix_lock": candidate_hash_target_prefix_lock,
                "candidate_hash_target_window_coverage": candidate_hash_target_window_coverage,
                "candidate_hash_target_band_alignment": candidate_hash_target_band_alignment,
                "candidate_hash_target_flux_alignment": candidate_hash_target_flux_alignment,
                "prototype_target_hex": prototype_target_hex,
                "prototype_target_multiplier": float(
                    prototype_share_target.get("target_multiplier", 1.0)
                ),
                "prototype_target_probability": float(
                    prototype_share_target.get("target_probability", 0.0)
                ),
                "prototype_target_valid_goal": int(
                    prototype_share_target.get("desired_valid_count", 0)
                ),
                "selection_mode": "network_only",
                "feedback_temperature_norm": float(post_cuda_gpu_feedback.get("temperature_norm", 0.0)),
                "feedback_environment_pressure": float(
                    post_cuda_gpu_feedback.get("environment_pressure", 0.0)
                ),
                "feedback_roundtrip_latency_ms": float(
                    gpu_pulse_delta_feedback.get("roundtrip_latency_ms", 0.0)
                ),
                "injection_feedback_window_ms": float(
                    injection_delta_feedback.get("feedback_window_ms", 0.0)
                ),
                "injection_observation_gap_ms": float(
                    injection_delta_feedback.get("observation_gap_ms", 0.0)
                ),
                "injection_observation_freshness_gate": float(
                    injection_delta_feedback.get("observation_freshness_gate", 0.0)
                ),
                "injection_response_gate": float(
                    injection_delta_feedback.get("response_gate", 0.0)
                ),
                "observation_gap_ms": float(gpu_pulse_delta_feedback.get("observation_gap_ms", 0.0)),
                "observation_freshness_gate": float(
                    gpu_pulse_delta_feedback.get("observation_freshness_gate", 0.0)
                ),
                "dispatch_feedback_ratio": float(
                    gpu_pulse_delta_feedback.get("dispatch_feedback_ratio", 0.0)
                ),
                "feedback_latency_gate": float(gpu_pulse_delta_feedback.get("latency_gate", 0.0)),
                "pulse_phase_pressure_volume": phase_pressure_volume,
                "pulse_coupling_volume": coupling_volume,
                "pulse_kernel_volume": kernel_volume,
                "pulse_volume_gain": pulse_volume_gain,
                "probe_pool_size": probe_pool_size,
                "probe_cluster_count": probe_cluster_count,
                "dominant_probe_cluster": dominant_probe_cluster,
                "best_phase_length_pressure": best_phase_length_pressure,
                "best_phase_alignment_score": best_phase_alignment_score,
                "best_phase_confinement_cost": best_phase_confinement_cost,
                "best_decode_integrity": best_decode_integrity,
                "best_decode_entropy": best_decode_entropy,
                "best_sequence_persistence": best_sequence_persistence,
                "best_temporal_overlap": best_temporal_overlap,
                "best_voltage_frequency_flux": best_voltage_frequency_flux,
                "best_gpu_feedback_delta_score": best_gpu_feedback_delta_score,
                "best_function_score": best_function_score,
                "best_round_coupling": best_round_coupling,
                "best_hash_target_phase_pressure": best_hash_target_phase_pressure,
                "best_hash_target_phase_cost": best_hash_target_phase_cost,
                "best_hash_target_prefix_lock": best_hash_target_prefix_lock,
                "best_hash_target_window_coverage": best_hash_target_window_coverage,
                "best_hash_target_band_alignment": best_hash_target_band_alignment,
                "best_hash_target_flux_alignment": best_hash_target_flux_alignment,
                "cuda_kernel_telemetry": cuda_kernel_telemetry,
                "verification_mode": verification_mode,
                "verification_consistency": verification_consistency,
                "production_unlock_ready": production_unlock_ready,
                "difficulty_norm": difficulty_norm,
                "worker_count": pulse_workers,
                "worker_batches": worker_batches,
                "nonces": [payload["nonce"] for payload in valid_shares],
                "cluster_preview": [item["cluster_id"] for item in valid_selected[:8]],
                "interference_field": interference_field,
                "simulation_field_state": simulation_field_state,
                "kernel_execution_event": kernel_execution_event,
                "motif_consistency": float(simulation_field_state.get("motif_consistency", 0.0)),
                "motif_repeat_count": int(simulation_field_state.get("motif_repeat_count", 0)),
                "motif_energy": float(simulation_field_state.get("motif_energy", 0.0)),
                "psi_encode": psi_encode,
                "temporal_manifold": temporal_manifold,
                "effective_vector": effective_vector,
                "manifold_diagnostics": manifold_diagnostics,
                "vector_magnitude": float(effective_vector.get("spatial_magnitude", 0.0)),
                "temporal_projection": temporal_confinement,
                "field_pressure": field_pressure,
                "larger_field_exposure": larger_field_exposure,
                "dominant_basin": dominant_basin_id,
                "path_equivalence_error": float(manifold_diagnostics.get("path_equivalence_error", 0.0)),
                "temporal_ordering_delta": float(manifold_diagnostics.get("temporal_ordering_delta", 0.0)),
                "basis_rotation_residual": float(manifold_diagnostics.get("basis_rotation_residual", 0.0)),
            }
        )
        prev_state = pulse_state

    latest_pulse = next(
        (
            pulse
            for pulse in reversed(pulse_batches)
            if int(pulse.get("yield_count", 0)) > 0
            or int(pulse.get("prototype_valid_count", 0)) > 0
            or int(pulse.get("probe_pool_size", 0)) > 0
        ),
        pulse_batches[-1] if pulse_batches else {},
    )
    prototype_carrier_count = max(
        [int(pulse.get("carrier_count", carrier_count)) for pulse in pulse_batches] or [carrier_count]
    )
    latest_psi_encode = dict(latest_pulse.get("psi_encode", {}) or {})
    latest_temporal_manifold = dict(latest_pulse.get("temporal_manifold", {}) or {})
    latest_effective_vector = dict(latest_pulse.get("effective_vector", {}) or {})
    latest_manifold_diagnostics = dict(latest_pulse.get("manifold_diagnostics", {}) or {})
    latest_interference_field = dict(latest_pulse.get("interference_field", {}) or {})
    latest_simulation_field = dict(latest_pulse.get("simulation_field_state", {}) or {})
    latest_kernel_event = dict(latest_pulse.get("kernel_execution_event", {}) or {})
    psi_state_norm = clamp01(float(latest_psi_encode.get("state_norm", 0.0)) / max(amplitude_cap * 2.4, 1.0e-6))
    detect_weight = float(latest_interference_field.get("field_resonance", 0.0))
    if detect_weight <= 0.0:
        detect_weight = max(
            float(latest_simulation_field.get("motif_energy", 0.0)),
            float(latest_simulation_field.get("interference_accounting", 0.0)),
        )
    schematic = {
        "nodes": [
            {"id": "header", "label": "Header Pulse", "x": 0.10, "y": 0.55, "weight": baseline_frequency_norm},
            {"id": "lattice", "label": "Lattice Cal", "x": 0.22, "y": 0.34, "weight": field_pressure},
            {"id": "detect", "label": "Vector Detect", "x": 0.32, "y": 0.58, "weight": detect_weight},
            {"id": "encode", "label": "Psi Encode", "x": 0.40, "y": 0.36, "weight": psi_state_norm},
            {"id": "clamp", "label": "Clamp Band", "x": 0.48, "y": 0.24, "weight": amplitude_cap},
            {"id": "manifold", "label": "M_t Fold", "x": 0.62, "y": 0.56, "weight": float(latest_temporal_manifold.get("coherence_norm", temporal_persistence))},
            {"id": "project", "label": "R_eff Projection", "x": 0.76, "y": 0.32, "weight": float(latest_effective_vector.get("spatial_magnitude", 0.0))},
            {"id": "entry", "label": "Kernel Entry", "x": 0.86, "y": 0.20, "weight": float(latest_simulation_field.get("calibration_readiness", 0.0))},
            {"id": "crosstalk", "label": "Crosstalk Cluster", "x": 0.87, "y": 0.58, "weight": baseline_shared},
            {"id": "queue", "label": "Stratum Queue", "x": 0.96, "y": 0.38, "weight": float(latest_pulse.get("worker_count", adaptive_workers)) / 10.0},
        ],
        "edges": [
            {"source": "header", "target": "lattice", "flow": baseline_frequency_norm},
            {"source": "lattice", "target": "detect", "flow": field_pressure},
            {"source": "detect", "target": "encode", "flow": float(latest_interference_field.get("field_resonance", 0.0))},
            {"source": "lattice", "target": "encode", "flow": field_pressure},
            {"source": "encode", "target": "clamp", "flow": psi_state_norm},
            {"source": "encode", "target": "manifold", "flow": abs(float(latest_psi_encode.get("cross_terms", {}).get("F_I", 0.0)))},
            {"source": "clamp", "target": "manifold", "flow": amplitude_cap},
            {"source": "manifold", "target": "project", "flow": float(latest_effective_vector.get("t_eff", temporal_persistence))},
            {"source": "project", "target": "entry", "flow": float(latest_simulation_field.get("target_gate", 0.0))},
            {"source": "entry", "target": "queue", "flow": float(latest_simulation_field.get("calibration_readiness", 0.0))},
            {"source": "project", "target": "crosstalk", "flow": float(latest_effective_vector.get("coherence_bias", baseline_shared))},
            {"source": "project", "target": "queue", "flow": float(latest_effective_vector.get("spatial_magnitude", prediction_confidence))},
            {"source": "crosstalk", "target": "queue", "flow": clamp01(float(latest_pulse.get("yield_count", 0)) / 70.0)},
        ],
    }
    pseudocode = [
        "lattice_cal = calibrate_silicon_lattice(nist, tensor_gradients, packet_classes, photonic_basins)",
        "Psi_encode = encode_basis(F, A, I, V, lattice_cal, pulse_packet_dev, mixed_cross_terms)",
        "Psi_t = M_t(Psi_encode, lattice_cal.environment_tensor, temporal_fold, crosstalk=shared_phase_lock, depth<=5)",
        "R_eff = project_effective_vector(Psi_t) -> [X, Y, Z, T_eff]",
        "diag = evaluate_vector_consistency(R_eff, path_equiv, ordering_delta, basis_rotation)",
        "header_freq = pulse_packet_dev(block_header, quartet, photon_pulse)",
        "target_cap = clamp_band(a_code, btc_target_interval, trap_ratio, viol_band)",
        "phase_event = indexed_phase_length_event(target_phase_windows, carrier_idx, pulse_idx) -> amplitude_pressure_confinement",
        "interference_field = detect_vectors_from_lattice(kernels, target_windows, cascade_interval, coupling_gradients)",
        "simulation_field = calibrate_fields_in_flight(interference_field, target_windows, kernel_execution_event)",
        "coherent_vectors = bias_nonce_trajectories(R_eff, target_cap, btc_target_windows, interference_field, carriers=70)",
        "hash_probe = expand_target_probe_neighborhood(coherent_vectors, btc_target_hex, offsets=cluster_weighted_target_offsets)",
        "candidate_next_state = evolve_state(current_state, coherent_vectors, ctx)",
        "ledger_delta = compute_ledger_delta(current_state, candidate_next_state, ctx)",
        "nonce_batch = harvest_coherent_nonces(coherent_vectors, yield_cap=20..70)",
        "worker_queue = package_stratum_batches(nonce_batch, workers=6..14)",
    ]

    return {
        "mode": "quantum_inspired_btc_pulse_harvest",
        "description": "Research-only BTC nonce harvest prototype driven by existing photonic pulse and temporal-coupling operators.",
        "pulse_operator_source": PULSE_OPERATORS.name,
        "temporal_schema_source": TEMPORAL_COUPLING.name,
        "process_schema_source": PROCESS_SUBSTRATE.name,
        "quartet": quartet,
        "amplitude_cap": amplitude_cap,
        "baseline_frequency_norm": baseline_frequency_norm,
        "difficulty_norm": difficulty_norm,
        "carrier_count": prototype_carrier_count,
        "max_depth": 5,
        "silicon_calibration": silicon_calibration,
        "fourier_kernel_fanout": fourier_kernel_fanout,
        "calibration_sweep": sweep_entries,
        "best_sweep": best_sweep,
        "interference_vector_field": latest_interference_field,
        "simulation_field_state": latest_simulation_field,
        "kernel_execution_event": latest_kernel_event,
        "psi_encode": latest_psi_encode,
        "temporal_manifold": latest_temporal_manifold,
        "effective_vector": latest_effective_vector,
        "manifold_diagnostics": latest_manifold_diagnostics,
        "schematic": schematic,
        "pseudocode": pseudocode,
        "pulse_batches": pulse_batches,
        "yield_sequence": [int(pulse.get("yield_count", 0)) for pulse in pulse_batches],
        "coherence_sequence": [float(pulse.get("coherence_peak", 0.0)) for pulse in pulse_batches],
        "interference_resonance_sequence": [float(pulse.get("interference_resonance", 0.0)) for pulse in pulse_batches],
        "calibration_readiness_sequence": [float(pulse.get("calibration_readiness", 0.0)) for pulse in pulse_batches],
        "sequence_persistence_sequence": [float(pulse.get("candidate_sequence_persistence", 0.0)) for pulse in pulse_batches],
        "temporal_overlap_sequence": [float(pulse.get("candidate_temporal_overlap", 0.0)) for pulse in pulse_batches],
        "voltage_frequency_flux_sequence": [float(pulse.get("candidate_voltage_frequency_flux", 0.0)) for pulse in pulse_batches],
        "gpu_feedback_delta_score_sequence": [
            float(pulse.get("candidate_gpu_feedback_delta_score", 0.0)) for pulse in pulse_batches
        ],
        "vector_magnitude_sequence": [float(pulse.get("vector_magnitude", 0.0)) for pulse in pulse_batches],
        "temporal_projection_sequence": [float(pulse.get("temporal_projection", 0.0)) for pulse in pulse_batches],
        "queue_depth_sequence": [
            int(sum(int(worker.get("queue_depth", 0)) for worker in pulse.get("worker_batches", []) or []))
            for pulse in pulse_batches
        ],
        "latest_summary": {
            "yield_count": int(latest_pulse.get("yield_count", 0)),
            "network_valid_count": int(latest_pulse.get("network_valid_count", 0)),
            "prototype_valid_count": int(latest_pulse.get("prototype_valid_count", 0)),
            "worker_count": int(latest_pulse.get("worker_count", adaptive_workers)),
            "carrier_count": int(latest_pulse.get("carrier_count", prototype_carrier_count)),
            "coherence_peak": float(latest_pulse.get("coherence_peak", 0.0)),
            "nesting_depth": int(latest_pulse.get("nesting_depth", adaptive_depth)),
            "state_capacity": int(latest_pulse.get("state_capacity", prototype_carrier_count ** adaptive_depth)),
            "field_pressure": float(latest_pulse.get("field_pressure", field_pressure)),
            "larger_field_exposure": float(latest_pulse.get("larger_field_exposure", larger_field_exposure)),
            "dominant_basin": str(latest_pulse.get("dominant_basin", dominant_basin_id)),
            "effective_vector": latest_effective_vector,
            "temporal_projection": float(latest_pulse.get("temporal_projection", 0.0)),
            "interference_resonance": float(latest_pulse.get("interference_resonance", 0.0)),
            "calibration_readiness": float(latest_pulse.get("calibration_readiness", 0.0)),
            "activation_density": float(latest_simulation_field.get("activation_density", 0.0)),
            "fanout_span": float(latest_simulation_field.get("fanout_span", 0.0)),
            "motif_consistency": float(latest_simulation_field.get("motif_consistency", 0.0)),
            "motif_repeat_count": int(latest_simulation_field.get("motif_repeat_count", 0)),
            "motif_energy": float(latest_simulation_field.get("motif_energy", 0.0)),
            "selection_mode": str(latest_pulse.get("selection_mode", "network_only")),
            "substrate_material": str(
                latest_pulse.get(
                    "substrate_material",
                    latest_kernel_event.get(
                        "substrate_material",
                        latest_simulation_field.get("substrate_material", "silicon_wafer"),
                    ),
                )
            ),
            "silicon_reference_source": str(
                latest_pulse.get(
                    "silicon_reference_source",
                    latest_kernel_event.get(
                        "silicon_reference_source",
                        latest_simulation_field.get(
                            "silicon_reference_source", NIST_REFERENCE.name
                        ),
                    ),
                )
            ),
            "gpu_feedback_source": str(
                latest_pulse.get(
                    "gpu_feedback_source",
                    latest_simulation_field.get("gpu_pulse_feedback_post_cuda", {}).get(
                        "source",
                        latest_simulation_field.get("gpu_pulse_feedback", {}).get(
                            "source", "vector_runtime_feedback"
                        ),
                    ),
                )
            ),
            "gpu_injection_feedback_source": str(
                latest_pulse.get(
                    "gpu_injection_feedback_source",
                    latest_simulation_field.get("gpu_pulse_feedback_injection", {}).get(
                        "source",
                        latest_simulation_field.get("gpu_pulse_feedback", {}).get(
                            "source", "vector_runtime_feedback"
                        ),
                    ),
                )
            ),
            "compute_regime": str(latest_kernel_event.get("compute_regime", latest_simulation_field.get("compute_regime", "classical_calibration"))),
            "vector_harmonic_gate": float(latest_kernel_event.get("vector_harmonic_gate", latest_simulation_field.get("vector_harmonic_gate", 0.0))),
            "harmonic_compute_weight": float(latest_kernel_event.get("harmonic_compute_weight", latest_simulation_field.get("harmonic_compute_weight", 0.0))),
            "candidate_function_score": float(latest_pulse.get("candidate_function_score", 0.0)),
            "candidate_round_coupling": float(latest_pulse.get("candidate_round_coupling", 0.0)),
            "stable_target_phase_pressure": float(latest_pulse.get("stable_target_phase_pressure", 0.0)),
            "candidate_hash_target_phase_pressure": float(
                latest_pulse.get("candidate_hash_target_phase_pressure", 0.0)
            ),
            "candidate_hash_target_phase_cost": float(
                latest_pulse.get("candidate_hash_target_phase_cost", 1.0)
            ),
            "candidate_hash_target_prefix_lock": float(
                latest_pulse.get("candidate_hash_target_prefix_lock", 0.0)
            ),
            "candidate_hash_target_window_coverage": float(
                latest_pulse.get("candidate_hash_target_window_coverage", 0.0)
            ),
            "candidate_hash_target_band_alignment": float(
                latest_pulse.get("candidate_hash_target_band_alignment", 0.0)
            ),
            "candidate_hash_target_flux_alignment": float(
                latest_pulse.get("candidate_hash_target_flux_alignment", 0.0)
            ),
            "candidate_phase_length_pressure": float(latest_pulse.get("candidate_phase_length_pressure", 0.0)),
            "candidate_phase_alignment_score": float(latest_pulse.get("candidate_phase_alignment_score", 0.0)),
            "candidate_phase_confinement_cost": float(latest_pulse.get("candidate_phase_confinement_cost", 0.0)),
            "candidate_sequence_persistence": float(latest_pulse.get("candidate_sequence_persistence", 0.0)),
            "candidate_temporal_overlap": float(latest_pulse.get("candidate_temporal_overlap", 0.0)),
            "candidate_voltage_frequency_flux": float(latest_pulse.get("candidate_voltage_frequency_flux", 0.0)),
            "candidate_gpu_feedback_delta_score": float(
                latest_pulse.get("candidate_gpu_feedback_delta_score", 0.0)
            ),
            "candidate_decode_phase_integrity": float(
                latest_pulse.get("candidate_decode_phase_integrity", 0.0)
            ),
            "candidate_decode_phase_alignment": float(
                latest_pulse.get("candidate_decode_phase_alignment", 0.0)
            ),
            "candidate_decode_target_ring_alignment": float(
                latest_pulse.get("candidate_decode_target_ring_alignment", 0.0)
            ),
            "candidate_decode_target_prefix_lock": float(
                latest_pulse.get("candidate_decode_target_prefix_lock", 0.0)
            ),
            "candidate_decode_target_prefix_vector_alignment": float(
                latest_pulse.get("candidate_decode_target_prefix_vector_alignment", 0.0)
            ),
            "candidate_decode_target_prefix_vector_phase_pressure": float(
                latest_pulse.get("candidate_decode_target_prefix_vector_phase_pressure", 0.0)
            ),
            "candidate_decode_phase_orbital_alignment": float(
                latest_pulse.get("candidate_decode_phase_orbital_alignment", 0.0)
            ),
            "candidate_decode_phase_orbital_resonance": float(
                latest_pulse.get("candidate_decode_phase_orbital_resonance", 0.0)
            ),
            "candidate_decode_phase_orbital_stability": float(
                latest_pulse.get("candidate_decode_phase_orbital_stability", 0.0)
            ),
            "candidate_decode_prefix_asymptote": float(
                latest_pulse.get("candidate_decode_prefix_asymptote", 1.0)
            ),
            "prototype_target_hex": str(latest_pulse.get("prototype_target_hex", "")),
            "prototype_target_multiplier": float(
                latest_pulse.get("prototype_target_multiplier", 1.0)
            ),
            "prototype_target_probability": float(
                latest_pulse.get("prototype_target_probability", 0.0)
            ),
            "prototype_target_valid_goal": int(
                latest_pulse.get("prototype_target_valid_goal", 0)
            ),
            "feedback_temperature_norm": float(latest_pulse.get("feedback_temperature_norm", 0.0)),
            "feedback_environment_pressure": float(
                latest_pulse.get("feedback_environment_pressure", 0.0)
            ),
            "feedback_roundtrip_latency_ms": float(
                latest_pulse.get("feedback_roundtrip_latency_ms", 0.0)
            ),
            "injection_feedback_window_ms": float(
                latest_pulse.get("injection_feedback_window_ms", 0.0)
            ),
            "injection_observation_gap_ms": float(
                latest_pulse.get("injection_observation_gap_ms", 0.0)
            ),
            "injection_observation_freshness_gate": float(
                latest_pulse.get("injection_observation_freshness_gate", 0.0)
            ),
            "injection_response_gate": float(
                latest_pulse.get("injection_response_gate", 0.0)
            ),
            "observation_gap_ms": float(latest_pulse.get("observation_gap_ms", 0.0)),
            "observation_freshness_gate": float(
                latest_pulse.get("observation_freshness_gate", 0.0)
            ),
            "dispatch_feedback_ratio": float(latest_pulse.get("dispatch_feedback_ratio", 0.0)),
            "feedback_latency_gate": float(latest_pulse.get("feedback_latency_gate", 0.0)),
            "post_feedback_apply_gate": float(
                latest_kernel_event.get("post_feedback_apply_gate", 0.0)
            ),
            "post_feedback_applied": bool(
                latest_kernel_event.get("post_feedback_applied", False)
            ),
            "trace_support": float(latest_kernel_event.get("trace_support", latest_simulation_field.get("substrate_trace_state", {}).get("trace_support", 0.0))),
            "trace_resonance": float(latest_kernel_event.get("trace_resonance", latest_simulation_field.get("substrate_trace_state", {}).get("trace_resonance", 0.0))),
            "trace_alignment": float(latest_kernel_event.get("trace_alignment", latest_simulation_field.get("substrate_trace_state", {}).get("trace_alignment", 0.0))),
            "trace_relative_spatial_field": [
                float(value)
                for value in list(
                    latest_pulse.get(
                        "trace_relative_spatial_field",
                        latest_simulation_field.get("substrate_trace_state", {}).get(
                            "trace_relative_spatial_field", [0.0, 0.0, 0.0, 0.0]
                        ),
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "coherent_noise_axis_vector": [
                float(value)
                for value in list(
                    latest_pulse.get(
                        "coherent_noise_axis_vector",
                        latest_simulation_field.get("substrate_trace_state", {}).get(
                            "coherent_noise_axis_vector", [0.0, 0.0, 0.0, 0.0]
                        ),
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "noise_resonance_nodes": [
                float(value)
                for value in list(
                    latest_pulse.get(
                        "noise_resonance_nodes",
                        latest_simulation_field.get("substrate_trace_state", {}).get(
                            "noise_resonance_nodes", [0.0, 0.0, 0.0, 0.0]
                        ),
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "drift_compensation_vector": [
                float(value)
                for value in list(
                    latest_pulse.get(
                        "drift_compensation_vector",
                        latest_simulation_field.get("substrate_trace_state", {}).get(
                            "drift_compensation_vector", [0.0, 0.0, 0.0, 0.0]
                        ),
                    )
                    or [0.0, 0.0, 0.0, 0.0]
                )
            ],
            "noise_resonance_gate": float(
                latest_pulse.get(
                    "noise_resonance_gate",
                    latest_simulation_field.get("substrate_trace_state", {}).get(
                        "noise_resonance_gate", 0.0
                    ),
                )
            ),
            "environment_turbulence": float(
                latest_pulse.get(
                    "environment_turbulence",
                    latest_simulation_field.get("substrate_trace_state", {}).get(
                        "environment_turbulence", 0.0
                    ),
                )
            ),
            "trace_vram_resident": bool(latest_kernel_event.get("trace_vram_resident", latest_simulation_field.get("substrate_trace_vram", {}).get("resident", False))),
            "trace_vram_updates": int(latest_kernel_event.get("trace_vram_updates", latest_simulation_field.get("substrate_trace_vram", {}).get("update_count", 0))),
            "probe_pool_size": int(latest_pulse.get("probe_pool_size", 0)),
            "probe_cluster_count": int(latest_pulse.get("probe_cluster_count", 0)),
            "dominant_probe_cluster": str(latest_pulse.get("dominant_probe_cluster", "")),
            "candidate_count": int(latest_pulse.get("cuda_kernel_telemetry", {}).get("candidate_count", 0)),
            "expanded_eval_count": int(latest_pulse.get("cuda_kernel_telemetry", {}).get("expanded_eval_count", 0)),
            "expanded_keep_count": int(latest_pulse.get("cuda_kernel_telemetry", {}).get("expanded_keep_count", 0)),
            "best_phase_length_pressure": float(latest_pulse.get("best_phase_length_pressure", 0.0)),
            "best_phase_alignment_score": float(latest_pulse.get("best_phase_alignment_score", 0.0)),
            "best_phase_confinement_cost": float(latest_pulse.get("best_phase_confinement_cost", 0.0)),
            "best_sequence_persistence": float(latest_pulse.get("best_sequence_persistence", 0.0)),
            "best_temporal_overlap": float(latest_pulse.get("best_temporal_overlap", 0.0)),
            "best_voltage_frequency_flux": float(latest_pulse.get("best_voltage_frequency_flux", 0.0)),
            "best_gpu_feedback_delta_score": float(
                latest_pulse.get("best_gpu_feedback_delta_score", 0.0)
            ),
            "best_function_score": float(latest_pulse.get("best_function_score", 0.0)),
            "best_round_coupling": float(latest_pulse.get("best_round_coupling", 0.0)),
            "best_decode_phase_integrity": float(
                latest_pulse.get("best_decode_phase_integrity", 0.0)
            ),
            "best_decode_phase_alignment": float(
                latest_pulse.get("best_decode_phase_alignment", 0.0)
            ),
            "best_decode_target_ring_alignment": float(
                latest_pulse.get("best_decode_target_ring_alignment", 0.0)
            ),
            "best_decode_target_prefix_lock": float(
                latest_pulse.get("best_decode_target_prefix_lock", 0.0)
            ),
            "best_decode_target_prefix_vector_alignment": float(
                latest_pulse.get("best_decode_target_prefix_vector_alignment", 0.0)
            ),
            "best_decode_target_prefix_vector_phase_pressure": float(
                latest_pulse.get("best_decode_target_prefix_vector_phase_pressure", 0.0)
            ),
            "best_decode_phase_orbital_alignment": float(
                latest_pulse.get("best_decode_phase_orbital_alignment", 0.0)
            ),
            "best_decode_phase_orbital_resonance": float(
                latest_pulse.get("best_decode_phase_orbital_resonance", 0.0)
            ),
            "best_decode_phase_orbital_stability": float(
                latest_pulse.get("best_decode_phase_orbital_stability", 0.0)
            ),
            "best_decode_prefix_asymptote": float(
                latest_pulse.get("best_decode_prefix_asymptote", 1.0)
            ),
            "best_hash_target_phase_pressure": float(
                latest_pulse.get("best_hash_target_phase_pressure", 0.0)
            ),
            "best_hash_target_phase_cost": float(
                latest_pulse.get("best_hash_target_phase_cost", 1.0)
            ),
            "best_hash_target_prefix_lock": float(
                latest_pulse.get("best_hash_target_prefix_lock", 0.0)
            ),
            "best_hash_target_window_coverage": float(
                latest_pulse.get("best_hash_target_window_coverage", 0.0)
            ),
            "best_hash_target_band_alignment": float(
                latest_pulse.get("best_hash_target_band_alignment", 0.0)
            ),
            "best_hash_target_flux_alignment": float(
                latest_pulse.get("best_hash_target_flux_alignment", 0.0)
            ),
            "submission_anchor_rate_per_second": float(latest_pulse.get("submission_anchor_rate_per_second", 0.0)),
            "submission_tick_duration_s": float(latest_pulse.get("submission_tick_duration_s", 0.0)),
            "submission_jitter_fraction": float(latest_pulse.get("submission_jitter_fraction", 0.0)),
            "cuda_kernel_telemetry": dict(latest_pulse.get("cuda_kernel_telemetry", {}) or {}),
            "verification_mode": str(latest_pulse.get("verification_mode", "cpu_hash_check")),
            "verification_consistency": float(latest_pulse.get("verification_consistency", 0.0)),
            "production_unlock_ready": bool(latest_pulse.get("production_unlock_ready", False)),
            "entry_trigger": bool(latest_pulse.get("entry_trigger", False)),
            "kernel_execution_event": latest_kernel_event,
            "path_equivalence_error": float(latest_pulse.get("path_equivalence_error", 0.0)),
            "basis_rotation_residual": float(latest_pulse.get("basis_rotation_residual", 0.0)),
        },
        "operator_pipeline": {
            "candidate_next_state_operator": str(
                process_ops.get("candidate_next_state_operator", "evolve_state(current_state, inputs, ctx)")
            ),
            "ledger_delta_operator": str(
                process_ops.get("ledger_delta_operator", "compute_ledger_delta(current_state, candidate_next_state, ctx)")
            ),
            "accept_operator": str(process_ops.get("accept_operator", "accept_state")),
            "sink_operator": str(process_ops.get("sink_operator", "make_sink_state")),
            "commit_operator": str(process_ops.get("commit_operator", "commit_state")),
        },
    }


def make_packet_spectrum(
    packet_id: int,
    packet_count: int,
    bin_count: int,
    cohort: CohortSeed,
    cohort_index: int,
    nist: dict[str, float],
) -> PacketState:
    bin_idx = np.arange(bin_count, dtype=np.float64)
    bin_norm = bin_idx / max(bin_count - 1, 1)
    packet_theta = (2.0 * math.pi * packet_id) / max(packet_count, 1)
    charge_cycle = (-2, -1, 1, 2)
    topological_charge = charge_cycle[packet_id % len(charge_cycle)]

    base_spacing = 1.0 + 0.35 * cohort.freq_norm + 0.04 * cohort_index
    packet_shift = (packet_id / max(packet_count - 1, 1)) * (bin_count * 0.18)
    center_base = bin_count * (0.12 + 0.58 * cohort.freq_norm) + packet_shift

    centers = np.array(
        [
            center_base,
            center_base + bin_count * 0.06 * math.sin(packet_theta),
            center_base + bin_count * 0.05 * math.cos(packet_theta),
        ],
        dtype=np.float64,
    )
    widths = np.array(
        [
            2.0 + 4.5 * (1.0 - cohort.curr_norm),
            2.4 + 3.8 * (1.0 - cohort.curr_norm),
            2.8 + 3.5 * (1.0 - cohort.curr_norm),
        ],
        dtype=np.float64,
    )

    amplitude_drive = 0.55 + 1.35 * cohort.amp_norm
    frequency_drive = base_spacing + 0.08 * topological_charge
    voltage_drive = 0.30 + 1.50 * cohort.volt_norm
    amperage_drive = 0.25 + 1.30 * cohort.curr_norm
    retro_gain = 0.04 + 0.03 * (packet_id % 5)

    amplitudes = np.zeros((3, bin_count), dtype=np.float64)
    phases = np.zeros((3, bin_count), dtype=np.float64)
    harmonic_offsets = (0.0, 7.0, 13.0)
    harmonic_scales = (1.0, 0.42, 0.24)
    excitation_scale = 0.004 * nist.get("mean_excitation_energy_ev", 173.0)

    for axis_idx in range(3):
        for harmonic_idx, harmonic_scale in enumerate(harmonic_scales):
            harmonic_center = min(
                bin_count - 1,
                centers[axis_idx]
                + harmonic_offsets[harmonic_idx] * (1.0 + 0.25 * cohort.freq_norm),
            )
            amplitudes[axis_idx] += (
                amplitude_drive
                * harmonic_scale
                * gaussian(bin_idx, harmonic_center, widths[axis_idx] + harmonic_idx)
            )
        amplitudes[axis_idx] += excitation_scale * np.exp(-bin_norm * (2.5 + axis_idx))

        helical_phase = topological_charge * packet_theta
        voltage_tilt = voltage_drive * bin_norm * math.pi
        harmonic_twist = frequency_drive * (axis_idx + 1) * bin_norm * 2.0 * math.pi
        axis_offset = axis_idx * (math.pi / 2.0)
        phases[axis_idx] = helical_phase + axis_offset + harmonic_twist + voltage_tilt

    compressed = blur_along_bins(amplitudes[None, :, :], sigma=1.0 + 1.25 * amperage_drive)[0]
    high_bin_falloff = np.exp(-bin_norm * (0.60 + 0.45 * amperage_drive))
    compressed *= high_bin_falloff[None, :]

    spectrum = compressed * np.exp(1j * phases)
    return PacketState(
        packet_id=packet_id,
        cohort=cohort.name,
        cohort_index=cohort_index,
        topological_charge=topological_charge,
        amplitude_drive=amplitude_drive,
        frequency_drive=frequency_drive,
        voltage_drive=voltage_drive,
        amperage_drive=amperage_drive,
        retro_gain=retro_gain,
        spectrum=spectrum.astype(np.complex128),
    )


def build_packet_bank(config: SimulationConfig, nist: dict[str, float]) -> list[PacketState]:
    cohorts = load_cohort_seeds()
    out: list[PacketState] = []
    for packet_id in range(config.packet_count):
        cohort_index = packet_id % len(cohorts)
        out.append(
            make_packet_spectrum(
                packet_id=packet_id,
                packet_count=config.packet_count,
                bin_count=config.bin_count,
                cohort=cohorts[cohort_index],
                cohort_index=cohort_index,
                nist=nist,
            )
        )
    return out


def dominant_axis_frequencies(
    spectrum: np.ndarray, freq_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    magnitudes = np.abs(spectrum)
    dominant_bins = np.argmax(magnitudes, axis=-1)
    freqs = np.take_along_axis(freq_axis[None, None, :], dominant_bins[..., None], axis=-1)[
        ..., 0
    ]
    return dominant_bins, freqs


def reconstruct_packet_path(
    spectrum: np.ndarray, recon_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals = np.fft.ifft(spectrum, n=recon_samples, axis=-1)
    pos = np.real(signals).T
    vel = np.gradient(pos, axis=0)
    acc = np.gradient(vel, axis=0)
    return pos, vel, acc


def mean_phase_lock(lhs: np.ndarray, rhs: np.ndarray) -> float:
    numerator = np.vdot(lhs.reshape(-1), rhs.reshape(-1))
    denom = np.linalg.norm(lhs) * np.linalg.norm(rhs) + 1.0e-9
    return float(abs(numerator) / denom)


def packet_color(packet: PacketState) -> tuple[float, float, float]:
    charge_norm = min(abs(packet.topological_charge) / 2.0, 1.0)
    cohort_bias = packet.cohort_index / 2.0
    return (
        0.25 + 0.60 * charge_norm,
        0.20 + 0.65 * (1.0 - cohort_bias * 0.5),
        0.30 + 0.55 * cohort_bias,
    )


def update_packet_bank(
    packets: list[PacketState],
    config: SimulationConfig,
    freq_axis: np.ndarray,
    step_index: int,
) -> dict[str, Any]:
    packet_count = len(packets)
    spectrum = np.stack([p.spectrum for p in packets], axis=0)
    bin_norm = np.linspace(0.0, 1.0, config.bin_count, dtype=np.float64)

    amplitudes = np.abs(spectrum)
    phases = np.unwrap(np.angle(spectrum), axis=-1)
    dln_a = np.gradient(np.log(amplitudes + 1.0e-9), axis=-1)

    voltage_ramp = np.array([p.voltage_drive for p in packets], dtype=np.float64)[:, None, None]
    frequency_drive = np.array([p.frequency_drive for p in packets], dtype=np.float64)[:, None, None]
    dln_f = np.gradient(
        np.log(freq_axis[None, None, :] * (1.0 + 0.12 * frequency_drive) + 1.0e-6),
        axis=-1,
    )
    voltage_phase = voltage_ramp * bin_norm[None, None, :] * (
        0.50 + 0.35 * math.sin(step_index * 0.15)
    )

    updated = spectrum * np.exp(
        config.kappa_a * dln_a + 1j * (config.kappa_f * dln_f + voltage_phase)
    )

    dominant_bins, dominant_freqs = dominant_axis_frequencies(updated, freq_axis)
    packet_center_freq = np.mean(dominant_freqs, axis=1)
    beat_delta = np.abs(packet_center_freq[:, None] - packet_center_freq[None, :])
    beat_weight = np.exp(-beat_delta / max(freq_axis[-1] * 0.18, 1.0e-6))

    phase_lock = np.zeros((packet_count, packet_count), dtype=np.float64)
    for lhs_idx in range(packet_count):
        for rhs_idx in range(packet_count):
            if lhs_idx == rhs_idx:
                continue
            phase_lock[lhs_idx, rhs_idx] = mean_phase_lock(updated[lhs_idx], updated[rhs_idx])

    coupling_term = np.zeros_like(updated)
    leakage_term = np.zeros_like(updated)
    shared_scores = np.zeros(packet_count, dtype=np.float64)
    for lhs_idx in range(packet_count):
        accumulator = np.zeros_like(updated[lhs_idx])
        leak_accumulator = np.zeros_like(updated[lhs_idx])
        weight_accumulator = 1.0e-9
        for rhs_idx in range(packet_count):
            if lhs_idx == rhs_idx:
                continue
            weight = beat_weight[lhs_idx, rhs_idx] * phase_lock[lhs_idx, rhs_idx]
            weight_accumulator += weight
            accumulator += weight * (updated[rhs_idx] - updated[lhs_idx])
            leak_accumulator += weight * (np.abs(updated[rhs_idx]) - np.abs(updated[lhs_idx]))
        coupling_term[lhs_idx] = accumulator / weight_accumulator
        leakage_term[lhs_idx] = leak_accumulator / weight_accumulator
        shared_scores[lhs_idx] = float(np.max(phase_lock[lhs_idx]))

    low_envelope = np.mean(np.abs(updated[:, :, : config.low_bin_count]), axis=-1, keepdims=True)
    trap_profile = np.exp(-bin_norm * 5.0)[None, None, :]
    high_profile = np.clip(bin_norm - 0.25, 0.0, 1.0)[None, None, :]

    retro_future = np.mean(updated[:, :, -4:], axis=-1, keepdims=True)
    retro_gain = np.array([p.retro_gain for p in packets], dtype=np.float64)[:, None, None]
    retro_term = retro_gain * retro_future * np.exp(-bin_norm * 8.0)[None, None, :]

    updated += config.kappa_couple * coupling_term
    updated *= 1.0 + config.kappa_leak * leakage_term
    updated += config.kappa_retro * retro_term
    updated += (
        config.kappa_trap
        * low_envelope
        * trap_profile
        * np.exp(1j * np.angle(updated))
    )
    updated[:, :, config.low_bin_count :] *= np.clip(
        1.0 - 0.11 * low_envelope * high_profile[:, :, config.low_bin_count :],
        0.65,
        1.0,
    )

    blur_sigma = 0.85 + np.mean([p.amperage_drive for p in packets]) * 0.90
    blurred = blur_along_bins(np.abs(updated), sigma=blur_sigma)
    updated = np.clip(blurred, 0.0, config.max_amplitude) * np.exp(1j * np.angle(updated))

    for packet_idx, packet in enumerate(packets):
        packet.spectrum = updated[packet_idx]

    return {
        "shared_scores": shared_scores,
        "beat_weight": beat_weight,
        "phase_lock": phase_lock,
        "dominant_bins": dominant_bins,
        "dominant_freqs": dominant_freqs,
    }


def build_debug_html(
    path: Path,
    final_paths: dict[int, np.ndarray],
    inspector_rows: list[dict[str, Any]],
    packet_classes: list[dict[str, Any]],
    btc_prototype: dict[str, Any] | None = None,
) -> None:
    packet_index = {row["packet_id"]: row for row in inspector_rows}
    class_index = {row["packet_id"]: row for row in packet_classes}

    final_points = []
    for packet_id, points in final_paths.items():
        final_points.append((packet_id, float(points[-1, 0]), float(points[-1, 1])))
    if not final_points:
        return

    xs = [p[1] for p in final_points]
    ys = [p[2] for p in final_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0e-6)
    span_y = max(max_y - min_y, 1.0e-6)

    def project_x(value: float) -> float:
        return 80.0 + 840.0 * ((value - min_x) / span_x)

    def project_y(value: float) -> float:
        return 620.0 - 500.0 * ((value - min_y) / span_y)

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'><title>Photon Frequency Domain Debug</title>")
    parts.append(
        "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px;}"
        "svg{background:#111827;border:1px solid #30363d;border-radius:12px;}"
        "table{border-collapse:collapse;margin-top:20px;width:100%;}"
        "th,td{border:1px solid #30363d;padding:8px;text-align:left;font-size:13px;}"
        "th{background:#161b22;} .note{color:#8b949e;}"
        ".cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0;}"
        ".card{background:#111827;border:1px solid #30363d;border-radius:12px;padding:12px;}"
        ".card h3{margin:0 0 6px 0;font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em;}"
        ".card strong{font-size:24px;}"
        ".proto-svg{background:#0f172a;border:1px solid #30363d;border-radius:12px;}"
        ".proto-node{fill:#132238;stroke:#7ee787;stroke-width:1.6;opacity:0.92;}"
        ".proto-ring{fill:none;stroke:#58a6ff;stroke-width:1.2;opacity:0.55;animation:pulse-ring 2.4s ease-in-out infinite;}"
        ".proto-edge{stroke:#3b82f6;stroke-width:1.4;stroke-dasharray:8 8;animation:edge-flow 1.2s linear infinite;}"
        ".proto-play{fill:#f8fafc;opacity:0.92;}"
        ".proto-label{fill:#e6edf3;font-size:12px;font-weight:600;}"
        "pre{background:#111827;border:1px solid #30363d;border-radius:12px;padding:14px;overflow:auto;}"
        "@keyframes pulse-ring{0%{opacity:0.35;}50%{opacity:0.95;}100%{opacity:0.35;}}"
        "@keyframes edge-flow{0%{stroke-dashoffset:16;}100%{stroke-dashoffset:0;}}</style></head><body>"
    )
    parts.append("<h1>Photon Frequency-Domain Debug View</h1>")
    parts.append("<p class='note'>Hover a packet marker to inspect representative bin state and emergent vector curl.</p>")
    parts.append("<svg width='1000' height='700' viewBox='0 0 1000 700'>")
    parts.append("<line x1='80' y1='620' x2='920' y2='620' stroke='#30363d' stroke-width='1' />")
    parts.append("<line x1='80' y1='620' x2='80' y2='120' stroke='#30363d' stroke-width='1' />")

    for packet_id, x, y in final_points:
        debug = packet_index[packet_id]
        packet_class = class_index[packet_id]
        title = (
            f"Packet {packet_id} ({debug['cohort']})\n"
            f"Classification: {packet_class['classification']}\n"
            f"Dominant bin: {debug['dominant_bin']}\n"
            f"Bin 10: mag {debug['bin10_mag']:.3f}, phase {debug['bin10_phase_rad']:.3f} rad "
            f"-> vector curl {debug['vector_curl_deg']:.2f} deg\n"
            f"Shared score: {packet_class['phase_lock_score']:.3f}"
        )
        parts.append(
            f"<circle cx='{project_x(x):.2f}' cy='{project_y(y):.2f}' r='7' fill='{debug['color_hex']}' stroke='#f8fafc' stroke-width='1.0'>"
            f"<title>{title}</title></circle>"
        )

    parts.append("</svg>")
    parts.append("<table><thead><tr><th>Packet</th><th>Cohort</th><th>Top bins</th><th>Bin 10</th><th>Vector curl</th></tr></thead><tbody>")
    for row in inspector_rows:
        top_bins = ", ".join(
            f"{item['bin']}:{item['mag']:.2f}@{item['phase_rad']:.2f}"
            for item in row["top_bins"]
        )
        parts.append(
            "<tr>"
            f"<td>{row['packet_id']}</td>"
            f"<td>{row['cohort']}</td>"
            f"<td>{top_bins}</td>"
            f"<td>mag {row['bin10_mag']:.3f}, phase {row['bin10_phase_rad']:.3f} rad</td>"
            f"<td>{row['vector_curl_deg']:.2f} deg</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    if btc_prototype:
        latest = dict(btc_prototype.get("latest_summary", {}) or {})
        schematic = dict(btc_prototype.get("schematic", {}) or {})
        nodes = list(schematic.get("nodes", []) or [])
        edges = list(schematic.get("edges", []) or [])
        node_index = {str(node.get("id", "")): node for node in nodes}
        yield_sequence = list(btc_prototype.get("yield_sequence", []) or [])
        coherence_sequence = list(btc_prototype.get("coherence_sequence", []) or [])
        vector_sequence = list(btc_prototype.get("vector_magnitude_sequence", []) or [])
        temporal_sequence = list(btc_prototype.get("temporal_projection_sequence", []) or [])
        pulse_batches = list(btc_prototype.get("pulse_batches", []) or [])
        latest_pulse = next(
            (
                pulse
                for pulse in reversed(pulse_batches)
                if int(pulse.get("yield_count", 0)) > 0
                or int(pulse.get("prototype_valid_count", 0)) > 0
                or int(pulse.get("probe_pool_size", 0)) > 0
            ),
            pulse_batches[-1] if pulse_batches else {},
        )
        effective_vector = dict(btc_prototype.get("effective_vector", {}) or latest.get("effective_vector", {}) or {})
        silicon_calibration = dict(btc_prototype.get("silicon_calibration", {}) or {})
        manifold_diagnostics = dict(btc_prototype.get("manifold_diagnostics", {}) or {})
        psi_encode = dict(btc_prototype.get("psi_encode", {}) or {})
        temporal_manifold = dict(btc_prototype.get("temporal_manifold", {}) or {})
        dominant_basin = dict(silicon_calibration.get("dominant_basin", {}) or {})
        field_environment = dict(silicon_calibration.get("field_environment", {}) or {})
        tensor_metrics = dict(silicon_calibration.get("tensor_metrics", {}) or {})
        path_equivalence_error = float(
            manifold_diagnostics.get(
                "path_equivalence_error",
                latest.get("path_equivalence_error", latest_pulse.get("path_equivalence_error", 0.0)),
            )
        )
        temporal_ordering_delta = float(
            manifold_diagnostics.get(
                "temporal_ordering_delta",
                latest_pulse.get("temporal_ordering_delta", 0.0),
            )
        )
        basis_rotation_residual = float(
            manifold_diagnostics.get(
                "basis_rotation_residual",
                latest.get("basis_rotation_residual", latest_pulse.get("basis_rotation_residual", 0.0)),
            )
        )
        temporal_projection = float(
            latest.get(
                "temporal_projection",
                latest_pulse.get("temporal_projection", effective_vector.get("t_eff", 0.0)),
            )
        )
        field_pressure = float(
            latest.get(
                "field_pressure",
                latest_pulse.get("field_pressure", silicon_calibration.get("field_pressure", 0.0)),
            )
        )
        larger_field_exposure = float(
            latest.get(
                "larger_field_exposure",
                latest_pulse.get("larger_field_exposure", silicon_calibration.get("larger_field_exposure", 0.0)),
            )
        )

        parts.append("<h2>Quantum-Inspired BTC Miner Prototype</h2>")
        parts.append("<p class='note'>Research-only pulse harvest path wired into ResearchConfinement. The lattice calibration stage simulates a silicon field envelope and mapped photonic basins before manifold folding. Production miner submit governor remains untouched.</p>")
        parts.append("<div class='cards'>")
        parts.append(
            "<div class='card'><h3>Yield</h3>"
            f"<strong>{int(latest.get('yield_count', 0))}</strong><div>nonces per pulse</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Coherence</h3>"
            f"<strong>{float(latest.get('coherence_peak', 0.0)):.3f}</strong><div>peak acceptance band</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Nesting</h3>"
            f"<strong>{int(latest.get('nesting_depth', 0))}</strong>"
            f"<div>depth over {int(latest.get('carrier_count', 0))} carriers</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Workers</h3>"
            f"<strong>{int(latest.get('worker_count', 0))}</strong><div>substrate-resident queue lanes</div></div>"
        )
        parts.append(
            "<div class='card'><h3>R_eff</h3>"
            f"<strong>{float(effective_vector.get('spatial_magnitude', 0.0)):.3f}</strong>"
            f"<div>X {float(effective_vector.get('x', 0.0)):.3f} | Y {float(effective_vector.get('y', 0.0)):.3f} | Z {float(effective_vector.get('z', 0.0)):.3f}</div></div>"
        )
        parts.append(
            "<div class='card'><h3>T_eff</h3>"
            f"<strong>{temporal_projection:.3f}</strong><div>coherence bias {float(effective_vector.get('coherence_bias', 0.0)):.3f}</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Path Eq</h3>"
            f"<strong>{path_equivalence_error:.3f}</strong><div>trajectory closure error</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Basis Drift</h3>"
            f"<strong>{basis_rotation_residual:.3f}</strong><div>ordering {temporal_ordering_delta:.3f}</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Lattice</h3>"
            f"<strong>{field_pressure:.3f}</strong><div>{str(dominant_basin.get('basin_id', 'unknown_basin'))}</div></div>"
        )
        parts.append(
            "<div class='card'><h3>Field Env</h3>"
            f"<strong>{larger_field_exposure:.3f}</strong><div>tensor flux {float(tensor_metrics.get('tensor_gradient_flux', 0.0)):.3f}</div></div>"
        )
        parts.append("</div>")

        parts.append("<svg class='proto-svg' width='1000' height='340' viewBox='0 0 1000 340'>")
        for edge in edges:
            src = node_index.get(str(edge.get("source", "")), {})
            dst = node_index.get(str(edge.get("target", "")), {})
            if not src or not dst:
                continue
            x1 = 80.0 + float(src.get("x", 0.0)) * 840.0
            y1 = 40.0 + float(src.get("y", 0.0)) * 220.0
            x2 = 80.0 + float(dst.get("x", 0.0)) * 840.0
            y2 = 40.0 + float(dst.get("y", 0.0)) * 220.0
            parts.append(
                f"<line class='proto-edge' x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' />"
            )
        for node in nodes:
            cx = 80.0 + float(node.get("x", 0.0)) * 840.0
            cy = 40.0 + float(node.get("y", 0.0)) * 220.0
            radius = 18.0 + 18.0 * clamp01(float(node.get("weight", 0.5)))
            title = f"{node.get('label', node.get('id', 'node'))} weight={float(node.get('weight', 0.0)):.3f}"
            parts.append(
                f"<circle class='proto-ring' cx='{cx:.2f}' cy='{cy:.2f}' r='{radius + 8.0:.2f}'><title>{title}</title></circle>"
            )
            parts.append(
                f"<circle class='proto-node' cx='{cx:.2f}' cy='{cy:.2f}' r='{radius:.2f}'><title>{title}</title></circle>"
            )
            parts.append(
                f"<polygon class='proto-play' points='{cx - radius * 0.22:.2f},{cy - radius * 0.28:.2f} {cx - radius * 0.22:.2f},{cy + radius * 0.28:.2f} {cx + radius * 0.36:.2f},{cy:.2f}' />"
            )
            parts.append(
                f"<text class='proto-label' x='{cx:.2f}' y='{cy + radius + 18.0:.2f}' text-anchor='middle'>{node.get('label', node.get('id', 'node'))}</text>"
            )
        parts.append("</svg>")

        parts.append("<h3>Pulse Injection Loop</h3>")
        parts.append("<pre>")
        for line in list(btc_prototype.get("pseudocode", []) or []):
            parts.append(str(line))
            parts.append("\n")
        parts.append("</pre>")

        parts.append("<h3>Encoded Manifold State</h3>")
        parts.append("<table><thead><tr><th>Layer</th><th>Values</th><th>Notes</th></tr></thead><tbody>")
        parts.append(
            "<tr>"
            "<td>Lattice Cal</td>"
            f"<td>field={field_pressure:.3f}, exposure={larger_field_exposure:.3f}, basin={str(dominant_basin.get('basin_id', 'unknown_basin'))}</td>"
            "<td>NIST silicon anchor, tensor gradients, and packet classes collapsed into photonic basin pressures.</td>"
            "</tr>"
        )
        parts.append(
            "<tr>"
            "<td>Psi_encode</td>"
            f"<td>state_norm={float(psi_encode.get('state_norm', 0.0)):.3f}, "
            f"F_I={float(dict(psi_encode.get('cross_terms', {}) or {}).get('F_I', 0.0)):.3f}, "
            f"A_V={float(dict(psi_encode.get('cross_terms', {}) or {}).get('A_V', 0.0)):.3f}</td>"
            "<td>Control quartet encoded into the latent excitation basis.</td>"
            "</tr>"
        )
        parts.append(
            "<tr>"
            "<td>M_t(Psi)</td>"
            f"<td>coherence={float(temporal_manifold.get('coherence_norm', 0.0)):.3f}, "
            f"field_resonance={float(temporal_manifold.get('field_resonance', 0.0)):.3f}, "
            f"transport_norm={float(temporal_manifold.get('transport_norm', 0.0)):.3f}</td>"
            "<td>Temporal fold and crosstalk operator applied without full collapse.</td>"
            "</tr>"
        )
        parts.append(
            "<tr>"
            "<td>R_eff</td>"
            f"<td>X={float(effective_vector.get('x', 0.0)):.3f}, "
            f"Y={float(effective_vector.get('y', 0.0)):.3f}, "
            f"Z={float(effective_vector.get('z', 0.0)):.3f}, "
            f"T_eff={float(effective_vector.get('t_eff', temporal_projection)):.3f}</td>"
            "<td>Projected effective vector used to bias nonce trajectories.</td>"
            "</tr>"
        )
        parts.append(
            "<tr>"
            "<td>Diagnostics</td>"
            f"<td>path={path_equivalence_error:.3f}, order={temporal_ordering_delta:.3f}, basis={basis_rotation_residual:.3f}</td>"
            "<td>Consistency checks for path equivalence, ordering, and basis transforms.</td>"
            "</tr>"
        )
        parts.append("</tbody></table>")

        parts.append("<h3>Silicon Lattice Calibration</h3>")
        parts.append("<table><thead><tr><th>Basin</th><th>Field</th><th>Depth</th><th>Occupancy</th><th>Tensor Align</th></tr></thead><tbody>")
        for basin in list(silicon_calibration.get("photonic_basins", []) or []):
            parts.append(
                "<tr>"
                f"<td>{str(basin.get('basin_id', ''))}</td>"
                f"<td>{str(basin.get('field', ''))} / {str(basin.get('particle', ''))}</td>"
                f"<td>{float(basin.get('depth', 0.0)):.3f}</td>"
                f"<td>{float(basin.get('occupancy', 0.0)):.3f}</td>"
                f"<td>{float(basin.get('tensor_alignment', 0.0)):.3f}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
        parts.append(
            "<p class='note'>charge="
            f"{float(field_environment.get('charge_field', 0.0)):.3f}"
            + " | lattice="
            + f"{float(field_environment.get('lattice_field', 0.0)):.3f}"
            + " | coherence="
            + f"{float(field_environment.get('coherence_field', 0.0)):.3f}"
            + " | vacancy="
            + f"{float(field_environment.get('vacancy_field', 0.0)):.3f}"
            + "</p>"
        )

        parts.append("<h3>Live Pulse History</h3>")
        parts.append("<table><thead><tr><th>Pulse</th><th>Yield</th><th>Coherence</th><th>|R_eff|</th><th>T_eff</th><th>Field</th><th>Depth</th><th>Workers</th><th>Preview</th></tr></thead><tbody>")
        for pulse in pulse_batches:
            preview = ", ".join(list(pulse.get("nonces", []) or [])[:6])
            parts.append(
                "<tr>"
                f"<td>{int(pulse.get('pulse_id', -1))}</td>"
                f"<td>{int(pulse.get('yield_count', 0))}</td>"
                f"<td>{float(pulse.get('coherence_peak', 0.0)):.3f}</td>"
                f"<td>{float(pulse.get('vector_magnitude', 0.0)):.3f}</td>"
                f"<td>{float(pulse.get('temporal_projection', 0.0)):.3f}</td>"
                f"<td>{float(pulse.get('field_pressure', 0.0)):.3f}</td>"
                f"<td>{int(pulse.get('nesting_depth', 0))}</td>"
                f"<td>{int(pulse.get('worker_count', 0))}</td>"
                f"<td>{preview}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

        parts.append("<h3>Yield Sequence</h3>")
        parts.append(
            "<p class='note'>Yield="
            + ", ".join(str(int(value)) for value in yield_sequence)
            + " | Coherence="
            + ", ".join(f"{float(value):.3f}" for value in coherence_sequence)
            + " | |R_eff|="
            + ", ".join(f"{float(value):.3f}" for value in vector_sequence)
            + " | T_eff="
            + ", ".join(f"{float(value):.3f}" for value in temporal_sequence)
            + "</p>"
        )

        if latest_pulse:
            parts.append("<h3>Latest Worker Batches</h3>")
            parts.append("<table><thead><tr><th>Worker</th><th>Batch Size</th><th>Queue Depth</th><th>Preview</th></tr></thead><tbody>")
            for worker in list(latest_pulse.get("worker_batches", []) or []):
                preview = ", ".join(list(worker.get("submit_preview", []) or [])[:4])
                parts.append(
                    "<tr>"
                    f"<td>{worker.get('worker_id', '')}</td>"
                    f"<td>{int(worker.get('batch_size', 0))}</td>"
                    f"<td>{int(worker.get('queue_depth', 0))}</td>"
                    f"<td>{preview}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")

    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def save_plots(
    output_dir: Path,
    packets: list[PacketState],
    aggregate_amplitude: np.ndarray,
    final_paths: dict[int, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axis_names = ("f_x", "f_y", "f_z")
    colors = ("#ff6b6b", "#ffd166", "#4dabf7")
    for axis_idx, axis in enumerate(axes):
        axis.plot(aggregate_amplitude[axis_idx], color=colors[axis_idx], linewidth=1.6)
        axis.set_ylabel(f"{axis_names[axis_idx]} mag")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Bin index")
    fig.suptitle("Frequency-Domain Photon Packet Spectrum")
    fig.tight_layout()
    fig.savefig(output_dir / "spectrum_plot.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 10))
    axis = fig.add_subplot(111, projection="3d")
    for packet in packets:
        path = final_paths[packet.packet_id]
        axis.plot(path[:, 0], path[:, 1], path[:, 2], color=packet_color(packet), linewidth=1.2, alpha=0.9)
    axis.set_title("Reconstructed Packet Vectors From Inverse FFT")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(output_dir / "reconstructed_vectors.png", dpi=180)
    plt.close(fig)


def packet_spin_from_path(vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
    spin = np.mean(np.cross(vel, acc), axis=0)
    norm = np.linalg.norm(spin)
    if norm <= 1.0e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return spin / norm


def pack_color_hex(rgb: tuple[float, float, float]) -> str:
    clipped = [int(np.clip(channel, 0.0, 1.0) * 255.0) for channel in rgb]
    return "#{:02x}{:02x}{:02x}".format(*clipped)


def write_frequency_outputs(
    output_dir: Path,
    payloads: dict[str, Any],
) -> None:
    json_names = {
        "trajectory_json": "photon_packet_trajectory_sample.json",
        "path_classification_json": "photon_packet_path_classification_sample.json",
        "tensor6d_json": "photon_lattice_tensor6d_sample.json",
        "tensor_gradient_json": "photon_tensor_gradient_sample.json",
        "vector_excitation_json": "photon_vector_excitation_sample.json",
        "tensor_glyph_json": "photon_tensor_gradient_glyph_sample.json",
        "shader_texture_json": "photon_shader_texture_sample.json",
        "audio_waveform_json": "photon_audio_waveform_sample.json",
        "packet_debug_json": "packet_frequency_debug.json",
        "run_summary_json": "frequency_domain_run_summary.json",
        "reconstructed_paths_json": "reconstructed_packet_paths.json",
    }
    for key, filename in json_names.items():
        write_json_with_comment(output_dir / filename, payloads[key])

    write_csv_with_comment(
        output_dir / "photon_packet_trajectory_sample.csv",
        [
            "packet_id",
            "timestep",
            "x",
            "y",
            "z",
            "theta",
            "amplitude",
            "freq_x",
            "freq_y",
            "freq_z",
            "phase_coupling",
            "temporal_inertia",
            "curvature",
            "coherence",
            "flux",
        ],
        payloads["trajectory_csv_rows"],
    )
    write_csv_with_comment(
        output_dir / "photon_packet_path_classification_sample.csv",
        [
            "packet_id",
            "classification",
            "group_id",
            "phase_lock_score",
            "curvature_depth",
            "coherence_score",
        ],
        payloads["path_classification_csv_rows"],
    )


def append_u32_le(out: bytearray, value: int) -> None:
    out.extend(int(value).to_bytes(4, byteorder="little", signed=False))


def append_u64_le(out: bytearray, value: int) -> None:
    out.extend(int(value).to_bytes(8, byteorder="little", signed=False))


def append_blob(out: bytearray, data: bytes) -> None:
    append_u32_le(out, len(data))
    out.extend(data)


def write_virtual_state_drive(path: Path, records: list[dict[str, Any]]) -> None:
    blob = bytearray()
    append_u32_le(blob, 0x31545356)  # 'VST1'
    append_u32_le(blob, 1)
    append_u32_le(blob, len(records))
    for record in records:
        key_bytes = record["key"].encode("utf-8")
        type_bytes = record["type"].encode("utf-8")
        meta_bytes = record["meta"].encode("utf-8")
        value_bytes = record["value"]
        append_blob(blob, key_bytes)
        append_blob(blob, type_bytes)
        append_blob(blob, meta_bytes)
        append_blob(blob, value_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def pack_u16(value: int) -> bytes:
    return int(value).to_bytes(2, byteorder="little", signed=False)


def pack_i16(value: int) -> bytes:
    return int(value).to_bytes(2, byteorder="little", signed=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        packet_count=args.packet_count,
        bin_count=args.bin_count,
        steps=args.steps,
        recon_samples=args.recon_samples,
        equivalent_grid_linear=args.equivalent_grid_linear,
        seed=args.seed,
    )

    np.random.seed(config.seed)
    nist = load_nist_reference()
    packets = build_packet_bank(config, nist)
    freq_axis = np.linspace(1.0, float(config.bin_count), config.bin_count, dtype=np.float64)

    trajectory_records: list[dict[str, Any]] = []
    trajectory_csv_rows: list[list[Any]] = []
    vector_excitations: list[dict[str, Any]] = []
    tensor6d_cells: list[dict[str, Any]] = []
    tensor_gradient_samples: list[dict[str, Any]] = []
    tensor_glyphs: list[dict[str, Any]] = []
    shader_texture: list[dict[str, Any]] = []
    audio_waveform: list[dict[str, Any]] = []
    inspector_rows: list[dict[str, Any]] = []
    reconstructed_paths: list[dict[str, Any]] = []
    final_paths: dict[int, np.ndarray] = {}
    phase_lock_history = np.zeros((config.packet_count, config.packet_count), dtype=np.float64)
    shared_score_history = np.zeros((config.packet_count,), dtype=np.float64)

    step_dominant_bins = np.zeros((config.steps, config.packet_count, 3), dtype=np.int32)
    step_dominant_freqs = np.zeros((config.steps, config.packet_count, 3), dtype=np.float64)
    theta_history = np.zeros((config.steps, config.packet_count), dtype=np.float64)
    amplitude_history = np.zeros((config.steps, config.packet_count), dtype=np.float64)
    shared_history = np.zeros((config.steps, config.packet_count), dtype=np.float64)
    coherence_history = np.zeros((config.steps, config.packet_count), dtype=np.float64)
    curvature_history = np.zeros((config.steps, config.packet_count), dtype=np.float64)
    flux_history = np.zeros((config.steps, config.packet_count), dtype=np.float64)

    for step_index in range(config.steps):
        update_meta = update_packet_bank(packets, config, freq_axis, step_index)
        phase_lock_history += update_meta["phase_lock"]
        shared_score_history += update_meta["shared_scores"]

        for packet_idx, packet in enumerate(packets):
            pos, vel, acc = reconstruct_packet_path(packet.spectrum, config.recon_samples)
            sample_idx = min(
                config.recon_samples - 1,
                int(round(step_index * (config.recon_samples - 1) / max(config.steps - 1, 1))),
            )
            point = pos[sample_idx]
            velocity = vel[sample_idx]
            accel = acc[sample_idx]
            theta = float(np.mean(np.angle(packet.spectrum)))
            amplitude = float(np.mean(np.abs(packet.spectrum)))
            freq_x, freq_y, freq_z = [float(value) for value in update_meta["dominant_freqs"][packet_idx]]
            phase_coupling = float(update_meta["shared_scores"][packet_idx])
            temporal_inertia = float(np.linalg.norm(accel))
            curvature = float(np.linalg.norm(np.cross(velocity, accel)) / (np.linalg.norm(velocity) ** 3 + 1.0e-9))
            coherence = float(mean_phase_lock(packet.spectrum, np.mean(np.stack([p.spectrum for p in packets], axis=0), axis=0)))
            flux = float(np.linalg.norm(velocity) * amplitude)

            step_dominant_bins[step_index, packet_idx] = update_meta["dominant_bins"][packet_idx]
            step_dominant_freqs[step_index, packet_idx] = update_meta["dominant_freqs"][packet_idx]
            theta_history[step_index, packet_idx] = theta
            amplitude_history[step_index, packet_idx] = amplitude
            shared_history[step_index, packet_idx] = phase_coupling
            coherence_history[step_index, packet_idx] = coherence
            curvature_history[step_index, packet_idx] = curvature
            flux_history[step_index, packet_idx] = flux

            record = {
                "packet_id": packet.packet_id,
                "timestep": step_index,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "theta": theta,
                "amplitude": amplitude,
                "freq_x": freq_x,
                "freq_y": freq_y,
                "freq_z": freq_z,
                "phase_coupling": phase_coupling,
                "temporal_inertia": temporal_inertia,
                "curvature": curvature,
                "coherence": coherence,
                "flux": flux,
            }
            trajectory_records.append(record)
            trajectory_csv_rows.append(
                [
                    packet.packet_id,
                    step_index,
                    record["x"],
                    record["y"],
                    record["z"],
                    theta,
                    amplitude,
                    freq_x,
                    freq_y,
                    freq_z,
                    phase_coupling,
                    temporal_inertia,
                    curvature,
                    coherence,
                    flux,
                ]
            )

    mean_phase_lock_matrix = phase_lock_history / max(config.steps, 1)
    mean_shared_scores = shared_score_history / max(config.steps, 1)
    mean_curvature_scores = np.mean(curvature_history, axis=0)
    mean_coherence_scores = np.mean(coherence_history, axis=0)

    packet_classes: list[dict[str, Any]] = []
    path_classification_csv_rows: list[list[Any]] = []
    aggregate_amplitude = np.mean(
        np.stack([np.abs(packet.spectrum) for packet in packets], axis=0),
        axis=0,
    )

    phase_norm = (mean_shared_scores - np.min(mean_shared_scores)) / (
        np.ptp(mean_shared_scores) + 1.0e-9
    )
    coherence_norm = (mean_coherence_scores - np.min(mean_coherence_scores)) / (
        np.ptp(mean_coherence_scores) + 1.0e-9
    )
    curvature_norm = (mean_curvature_scores - np.min(mean_curvature_scores)) / (
        np.ptp(mean_curvature_scores) + 1.0e-9
    )
    shared_index = 0.40 * phase_norm + 0.40 * coherence_norm + 0.20 * (1.0 - curvature_norm)
    shared_threshold = float(np.quantile(shared_index, 0.55))

    for packet_idx, packet in enumerate(packets):
        pos, vel, acc = reconstruct_packet_path(packet.spectrum, config.recon_samples)
        final_paths[packet.packet_id] = pos
        spin = packet_spin_from_path(vel, acc)
        final_theta = theta_history[-1, packet_idx]
        final_amp = amplitude_history[-1, packet_idx]
        final_freq = step_dominant_freqs[-1, packet_idx]
        dtheta_dt = float(np.gradient(theta_history[:, packet_idx])[-1])
        d2theta_dt2 = float(np.gradient(np.gradient(theta_history[:, packet_idx]))[-1])
        inertia = float(np.mean(np.linalg.norm(acc, axis=1)))
        curvature_depth = float(np.mean(curvature_history[:, packet_idx]))
        coherence_score = float(np.mean(coherence_history[:, packet_idx]))
        flux_score = float(np.mean(flux_history[:, packet_idx]))
        oam_twist = float(np.dot(spin, np.mean(vel, axis=0)) * (packet.topological_charge / 2.0))
        higgs_inertia = float(max(0.0, inertia * coherence_score - np.std(np.linalg.norm(vel, axis=1))))

        partner_idx = int(np.argmax(mean_phase_lock_matrix[packet_idx]))
        if partner_idx == packet_idx:
            off_diag = mean_phase_lock_matrix[packet_idx].copy()
            off_diag[packet_idx] = -1.0
            partner_idx = int(np.argmax(off_diag))
        phase_lock_score = float(mean_shared_scores[packet_idx])
        classification = "shared" if shared_index[packet_idx] >= shared_threshold else "individual"
        group_id = (
            min(packet.packet_id, packets[partner_idx].packet_id)
            if classification == "shared"
            else packet.packet_id
        )

        packet_class_row = {
            "packet_id": packet.packet_id,
            "classification": classification,
            "group_id": group_id,
            "phase_lock_score": phase_lock_score,
            "curvature_depth": curvature_depth,
            "coherence_score": coherence_score,
        }
        packet_classes.append(packet_class_row)
        path_classification_csv_rows.append(
            [
                packet.packet_id,
                classification,
                group_id,
                phase_lock_score,
                curvature_depth,
                coherence_score,
            ]
        )

        dominant_bins = step_dominant_bins[-1, packet_idx]
        magnitudes = np.abs(packet.spectrum)
        top_bin_indices = np.argsort(magnitudes.mean(axis=0))[-3:][::-1]
        vector_curl_deg = float(np.degrees(np.arctan2(np.linalg.norm(np.mean(np.cross(vel, acc), axis=0)), np.linalg.norm(np.mean(vel, axis=0)) + 1.0e-9)))
        color_rgb = packet_color(packet)
        inspector_rows.append(
            {
                "packet_id": packet.packet_id,
                "cohort": packet.cohort,
                "dominant_bin": int(int(np.mean(dominant_bins))),
                "bin10_mag": float(np.mean(magnitudes[:, min(10, config.bin_count - 1)])),
                "bin10_phase_rad": float(np.mean(np.angle(packet.spectrum[:, min(10, config.bin_count - 1)]))),
                "vector_curl_deg": vector_curl_deg,
                "color_hex": pack_color_hex(color_rgb),
                "top_bins": [
                    {
                        "bin": int(bin_idx),
                        "mag": float(magnitudes.mean(axis=0)[bin_idx]),
                        "phase_rad": float(np.mean(np.angle(packet.spectrum[:, bin_idx]))),
                    }
                    for bin_idx in top_bin_indices
                ],
            }
        )

        reconstructed_paths.append(
            {
                "packet_id": packet.packet_id,
                "cohort": packet.cohort,
                "points": [
                    {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
                    for point in pos
                ],
            }
        )

        packet_bin_center = int(np.mean(dominant_bins))
        for sample_idx in np.linspace(0, config.recon_samples - 1, 4, dtype=np.int32):
            sample_point = pos[int(sample_idx)]
            sample_vel = vel[int(sample_idx)]
            vector_excitations.append(
                {
                    "x": float(sample_point[0]),
                    "y": float(sample_point[1]),
                    "z": float(sample_point[2]),
                    "vec_x": float(sample_vel[0]),
                    "vec_y": float(sample_vel[1]),
                    "vec_z": float(sample_vel[2]),
                    "spin": [float(spin[0]), float(spin[1]), float(spin[2])],
                    "oam_twist": oam_twist,
                }
            )

        tensor_matrix = np.array(
            [
                [float(final_freq[0] / config.bin_count), float(np.mean(vel[:, 0])), float(np.mean(acc[:, 0]))],
                [float(np.mean(vel[:, 1])), float(final_freq[1] / config.bin_count), float(np.mean(acc[:, 1]))],
                [float(np.mean(acc[:, 2])), float(np.mean(vel[:, 2])), float(final_freq[2] / config.bin_count)],
            ],
            dtype=np.float64,
        )
        tensor6d_cells.append(
            {
                "x": int(packet.packet_id),
                "y": int(dominant_bins[0]),
                "z": int(dominant_bins[1]),
                "phase_coherence": coherence_score,
                "curvature": curvature_depth,
                "flux": flux_score,
                "inertia": inertia,
                "freq_x": float(final_freq[0]),
                "freq_y": float(final_freq[1]),
                "freq_z": float(final_freq[2]),
                "dtheta_dt": dtheta_dt,
                "d2theta_dt2": d2theta_dt2,
                "oam_twist": oam_twist,
                "spin_vector": [float(spin[0]), float(spin[1]), float(spin[2])],
                "higgs_inertia": higgs_inertia,
            }
        )
        tensor_gradient_samples.append(
            {
                "packet_id": packet.packet_id,
                "bin_center": packet_bin_center,
                "tensor": tensor_matrix.tolist(),
                "phase_gradient": [
                    float(np.mean(np.gradient(np.unwrap(np.angle(packet.spectrum[0]))))),
                    float(np.mean(np.gradient(np.unwrap(np.angle(packet.spectrum[1]))))),
                    float(np.mean(np.gradient(np.unwrap(np.angle(packet.spectrum[2]))))),
                ],
                "amplitude_gradient": [
                    float(np.mean(np.gradient(np.abs(packet.spectrum[0])))),
                    float(np.mean(np.gradient(np.abs(packet.spectrum[1])))),
                    float(np.mean(np.gradient(np.abs(packet.spectrum[2])))),
                ],
                "oam_twist": oam_twist,
                "temporal_inertia": inertia,
            }
        )
        tensor_glyphs.append(
            {
                "x": int(packet.packet_id),
                "y": int(dominant_bins[0]),
                "z": int(dominant_bins[2]),
                "tensor": tensor_matrix.tolist(),
                "color": [
                    float(np.clip(0.20 + 0.70 * coherence_score, 0.0, 1.0)),
                    float(np.clip(0.15 + 0.65 * abs(oam_twist), 0.0, 1.0)),
                    float(np.clip(0.20 + 0.50 * higgs_inertia, 0.0, 1.0)),
                ],
            }
        )
        shader_texture.append(
            {
                "x": int(packet.packet_id),
                "y": int(dominant_bins[0]),
                "z": int(dominant_bins[1]),
                "rgb": [
                    float(np.clip(final_amp / config.max_amplitude, 0.0, 1.0)),
                    float(np.clip((math.sin(final_theta) + 1.0) * 0.5, 0.0, 1.0)),
                    float(np.clip(abs(oam_twist) * 4.0, 0.0, 1.0)),
                ],
            }
        )

    vsd_records = [
        {
            "key": "photon/volume/edge",
            "type": "u64",
            "meta": "expanded lattice edge hint",
            "value": int(config.equivalent_grid_linear).to_bytes(8, byteorder="little", signed=False),
        },
        {
            "key": "photon/volume/scale",
            "type": "f64",
            "meta": "normalized volume scale",
            "value": np.array([1.35], dtype=np.float64).tobytes(),
        },
    ]
    for packet_idx, packet in enumerate(packets):
        freq = step_dominant_freqs[-1, packet_idx]
        coherence = float(np.mean(coherence_history[:, packet_idx]))
        shared = float(np.mean(shared_history[:, packet_idx]))
        curvature = float(np.mean(curvature_history[:, packet_idx]))
        amp = float(amplitude_history[-1, packet_idx])
        theta = float(theta_history[-1, packet_idx])
        payload = bytearray()
        payload.extend(pack_i16(int(round((freq[0] / config.bin_count) * 16384.0))))
        payload.extend(pack_i16(int(round((freq[1] / config.bin_count) * 16384.0))))
        payload.extend(pack_i16(int(round((freq[2] / config.bin_count) * 16384.0))))
        payload.extend(pack_u16(int(round(np.clip(amp / config.max_amplitude, 0.0, 1.0) * 65535.0))))
        payload.extend(pack_u16(int(round(np.clip(coherence, 0.0, 1.0) * 65535.0))))
        payload.extend(pack_u16(int(round(np.clip(shared, 0.0, 1.0) * 65535.0))))
        payload.extend(pack_u16(int(round(np.clip(curvature / 64.0, 0.0, 1.0) * 65535.0))))
        payload.extend(pack_i16(int(round(np.sin(theta) * 16384.0))))
        payload.extend(pack_i16(int(round(np.cos(theta) * 16384.0))))
        meta = (
            f"packet={packet.packet_id};cohort={packet.cohort};charge={packet.topological_charge};"
            f"amp={amp:.6f};theta={theta:.6f};coherence={coherence:.6f};shared={shared:.6f};curvature={curvature:.6f}"
        )
        vsd_records.append(
            {
                "key": f"photon/tensor/{packet.packet_id:04d}",
                "type": "freq_tensor_q16",
                "meta": meta,
                "value": bytes(payload),
            }
        )

    time_axis = np.linspace(0.0, 1.0 / 60.0, config.recon_samples, endpoint=False)
    aggregate_signal = np.mean(
        np.stack([np.fft.ifft(packet.spectrum, n=config.recon_samples, axis=-1) for packet in packets], axis=0),
        axis=0,
    )
    oam_mean = np.mean([entry["oam_twist"] for entry in vector_excitations]) if vector_excitations else 0.0
    for sample_idx, time_s in enumerate(time_axis):
        sample = aggregate_signal[:, sample_idx]
        mix = np.array(
            [
                np.real(sample[0]),
                np.real(sample[1]) + 0.25 * np.imag(sample[2]),
                np.real(sample[2]) + 0.20 * oam_mean,
                0.50 * (np.imag(sample[0]) + np.imag(sample[1])),
            ],
            dtype=np.float64,
        )
        peak = max(np.max(np.abs(mix)), 1.0e-6)
        channels = np.clip(mix / max(peak, 1.0), -1.0, 1.0)
        audio_waveform.append(
            {
                "time": float(time_s),
                "channels": [float(value) for value in channels],
            }
        )

    btc_prototype = build_btc_miner_prototype(
        config=config,
        nist=nist,
        mean_phase_lock_matrix=mean_phase_lock_matrix,
        theta_history=theta_history,
        amplitude_history=amplitude_history,
        shared_history=shared_history,
        coherence_history=coherence_history,
        curvature_history=curvature_history,
        step_dominant_freqs=step_dominant_freqs,
        packet_classes=packet_classes,
        tensor_gradient_samples=tensor_gradient_samples,
    )
    persist_research_state(ROOT_STATE, btc_prototype)

    run_summary = {
        "mode": "frequency_domain_packets_only",
        "description": "Pure spectral photon confinement simulation with inverse-FFT emergent vectors.",
        "packet_count": config.packet_count,
        "bin_count": config.bin_count,
        "steps": config.steps,
        "recon_samples": config.recon_samples,
        "equivalent_grid_linear": config.equivalent_grid_linear,
        "voxel_grid_materialized": False,
        "fft_backend": "numpy.fft",
        "nist_reference": {
            "lattice_constant_m": nist.get("lattice_constant_m"),
            "mean_excitation_energy_ev": nist.get("mean_excitation_energy_ev"),
            "density_g_cm3": nist.get("density_g_cm3"),
        },
        "aggregate_metrics": {
            "mean_shared_score": float(np.mean(mean_shared_scores)),
            "max_phase_lock": float(np.max(mean_phase_lock_matrix)),
            "mean_amplitude": float(np.mean(aggregate_amplitude)),
            "mean_audio_abs": float(np.mean(np.abs([channel for row in audio_waveform for channel in row["channels"]]))),
            "packet_class_counts": {
                "shared": int(sum(1 for row in packet_classes if row["classification"] == "shared")),
                "individual": int(sum(1 for row in packet_classes if row["classification"] == "individual")),
            },
        },
        "artifacts": {
            "spectrum_plot": "spectrum_plot.png",
            "reconstructed_vectors": "reconstructed_vectors.png",
            "debug_view": "debug_view.html",
        },
        "btc_miner_prototype": btc_prototype,
    }

    payloads = {
        "trajectory_json": trajectory_records,
        "path_classification_json": packet_classes,
        "tensor6d_json": tensor6d_cells,
        "tensor_gradient_json": tensor_gradient_samples,
        "vector_excitation_json": vector_excitations,
        "tensor_glyph_json": tensor_glyphs,
        "shader_texture_json": shader_texture,
        "audio_waveform_json": audio_waveform,
        "packet_debug_json": inspector_rows,
        "run_summary_json": run_summary,
        "reconstructed_paths_json": reconstructed_paths,
        "trajectory_csv_rows": trajectory_csv_rows,
        "path_classification_csv_rows": path_classification_csv_rows,
    }

    write_frequency_outputs(output_dir, payloads)
    save_plots(output_dir, packets, aggregate_amplitude, final_paths)
    build_debug_html(
        output_dir / "debug_view.html",
        final_paths,
        inspector_rows,
        packet_classes,
        btc_prototype=btc_prototype,
    )
    write_virtual_state_drive(output_dir / "photon_volume_expansion.gevsd", vsd_records)

    if args.write_root_samples:
        root_payloads = {
            "trajectory_json": trajectory_records,
            "path_classification_json": packet_classes,
            "tensor6d_json": tensor6d_cells,
            "tensor_gradient_json": tensor_gradient_samples,
            "vector_excitation_json": vector_excitations,
            "tensor_glyph_json": tensor_glyphs,
            "shader_texture_json": shader_texture,
            "audio_waveform_json": audio_waveform,
            "packet_debug_json": inspector_rows,
            "run_summary_json": run_summary,
            "reconstructed_paths_json": reconstructed_paths,
            "trajectory_csv_rows": trajectory_csv_rows,
            "path_classification_csv_rows": path_classification_csv_rows,
        }
        write_frequency_outputs(ROOT, root_payloads)
        build_debug_html(
            ROOT / "debug_view.html",
            final_paths,
            inspector_rows,
            packet_classes,
            btc_prototype=btc_prototype,
        )
        write_virtual_state_drive(ROOT / "photon_volume_expansion.gevsd", vsd_records)

    print(f"frequency-domain run complete: {output_dir}")
    print(
        "shared packets="
        f"{sum(1 for row in packet_classes if row['classification'] == 'shared')}/"
        f"{len(packet_classes)}, mean shared score={np.mean(mean_shared_scores):.4f}"
    )
    print(
        "artifacts: spectrum_plot.png, reconstructed_vectors.png, debug_view.html, "
        "photon_packet_trajectory_sample.json"
    )


if __name__ == "__main__":
    main()
