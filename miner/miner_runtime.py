# ============================================================================
# VirtualMiner / MINER
# ASCII-ONLY SOURCE FILE
# File: miner_runtime.py
# Version: v7.0.0 Hybrid Extended (B + D)
# ============================================================================
"""
Purpose
-------
Full Mining Runtime for VirtualMiner, combining:

  * Option B: Full Expansion Runtime (Patch Matrix Style)
  * Option D: Extended Parallelism Runtime (Tiers, Clusters, Kernels)

This runtime is the top-level orchestration layer that handles:

  - BIOS gate / VSD boot barrier
  - Environment normalization
  - ComputeManager integration (tier/cluster/lane orchestration)
  - MinerEngine lifecycle control
  - Extended tier/cluster/kernels hooks
  - Hardened logging and error surfaces
  - EventBus announcements
  - Config & runtime parameter unification
  - Full patch-matrix annotations for maintainability

This file is MINING SUBSYSTEM ONLY:
  - No prediction engine calls
  - No trading engine hooks
  - No AI subsystem interference
  - No market utilities
  - No cross-subsystem telemetry bleed

This runtime remains the controlling entrypoint for:
  - MinerEngine
  - ComputeManager (generic compute fabric)
  - ShareAllocator (mining-only)
  - Submitter (pool-only)
  - FailsafeGovernor (mining-only envelope)

All logic here is ASCII-safe. No Unicode permitted.
"""

# ============================================================================
# SECTION 0: IMPORTS & SAFETY WRAPPERS
# ============================================================================

from __future__ import annotations
from typing import Dict, Any, Callable, Optional
import time
import json
import threading
import logging
import traceback
from pathlib import Path

from bios.event_bus import get_event_bus

ConfigManager = dict  # simple alias to document CFG as dict-like

from VHW.vsd_manager import VSDManager

from core.utils import get, store, delete

# ----------------------------------------------------------------------------
# Miner subsystem imports
# ----------------------------------------------------------------------------
from VHW.compute_manager import ComputeManager
from VHW.share_allocator import ShareAllocator
from miner.failsafe import FailsafeGovernor
from miner.miner_engine import MinerEngine
from miner.submitter import Submitter
# handler_miner no longer required; ComputeManager integrates NonceMath directly

# ============================================================================
# SECTION 1: CONFIG LOADING AND LOGGING SETUP
# ============================================================================

def _load_json_config(name: str) -> Dict[str, Any]:
    try:
        p = Path(__file__).parent / name
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

CFG_DATA = _load_json_config("miner_runtime_config.json")
CFG = CFG_DATA  # dict

# Helper for transposed coins structure
def _get_coin_info(coins_dict: Dict[str, Any], coin: str) -> Dict[str, Any]:
    if not isinstance(coins_dict, dict):
        return {}
    # Check if transposed (keys are variables like "wallet")
    if "wallet" in coins_dict and isinstance(coins_dict["wallet"], dict):
        return {var: coins_dict[var].get(coin, "") for var in coins_dict if isinstance(coins_dict[var], dict) and coin in coins_dict[var]}
    else:
        # Old structure
        return coins_dict.get(coin, {})

# ----------------------------------------------------------------------------
# LOGGING: ADA v4.8.7 Hybrid Standard
# ----------------------------------------------------------------------------

def _mk_logger() -> logging.Logger:
    lg = logging.getLogger("Miner.Runtime")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg

LOG = _mk_logger()
BUS = get_event_bus()

# ============================================================================
# SECTION 2: BIOS GATE / SYSTEM SANITY
# ============================================================================

def _bios_ready(vsd) -> bool:
    try:
        return bool(vsd.get("system/bios_boot_ok", False))
    except Exception:
        return False

def _wait_for_bios(vsd, delay: float = 0.25) -> None:
    LOG.info("Waiting for BIOS boot flag (system/bios_boot_ok)...")
    while not _bios_ready(vsd):
        time.sleep(delay)
    LOG.info("BIOS boot flag detected")

# ============================================================================
# SECTION 3: EXTENDED HARDWARE NORMALIZATION
# ============================================================================

def _normalize_hardware(env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides a consistent hardware spec for ComputeManager.
    Extended for future tier/cluster expansion:
      - vram_mb
      - mem_bw_gbps
      - cpu_ghz
      - num_clusters (future extension)
      - num_kernels (future extension)
    """
    raw = env.get("hardware", {})

    # defaults
    default = {
        "vram_mb": 8192,
        "mem_bw_gbps": 448.0,
        "cpu_ghz": 4.0,
        "num_clusters": 2,
        "num_kernels": 4,
    }

    hw = {}
    for k, v in default.items():
        hw[k] = raw.get(k, v)

    LOG.info("Hardware normalized: %s", hw)
    return hw

# ============================================================================
# SECTION 4: EXTENDED TIER/CLUSTER/KERNEL DEFINITIONS
# ============================================================================

def _build_extended_tiers() -> Any:
    """
    Build tier definitions.

    Extended:
      - Each tier supports multiple clusters
      - Tier includes kernel slots for future compute kernels
    """
    cfg_tiers = CFG.get("tiers")
    if isinstance(cfg_tiers, list) and cfg_tiers:
        tiers = []
        for t in cfg_tiers:
            tiers.append({
                "tier_id": int(t.get("tier_id", 0)),
                "vqram_mb": int(t.get("vqram_mb", 256)),
                "cluster_count": int(t.get("cluster_count", 1)),
                "kernel_slots": int(t.get("kernel_slots", 2))
            })
        return tiers

    # default: 4 tiers, 1 cluster each, 2 kernel slots
    return [
        {"tier_id": i, "vqram_mb": 256, "cluster_count": 1, "kernel_slots": 2}
        for i in range(4)
    ]


DEFAULT_BLOCK_TIME_S = {
    "BTC": 600.0,
    "ETC": 13.0,
    "RVN": 60.0,
    "LTC": 150.0,
    "ERG": 120.0,
}

DIFF1_TARGET = int(
    "00000000FFFF0000000000000000000000000000000000000000000000000000",
    16,
)

SUBMISSION_RATE_DEFAULTS = {
    "fallback_allowed_rate_per_second": 2.0,
    "target_acceptance_rate": 0.92,
    "expected_weight": 0.70,
    "observed_weight": 0.30,
    "smoothing": 0.60,
    "network_target_fraction": 0.05,
    "pool_submit_ceiling_per_second": 2.0,
    "network_submit_ceiling_per_second": 2.0,
    "pool_submit_guard_fraction": 0.985,
    "share_rate_taper_start_ratio": 0.80,
    "share_rate_taper_floor_fraction": 0.85,
    "share_rate_taper_power": 1.35,
    "min_tick_duration_s": 0.05,
    "jitter_fraction": 0.18,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _positive_min(values: list[float], default: float) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    if positives:
        return min(positives)
    return float(default)


def _submission_rate_cfg() -> Dict[str, float]:
    raw = CFG.get("submission_rate", {}) if isinstance(CFG, dict) else {}
    cfg = dict(raw) if isinstance(raw, dict) else {}
    out = dict(SUBMISSION_RATE_DEFAULTS)
    out.update(cfg)
    return {str(k): _safe_float(v, out.get(str(k), 0.0)) for k, v in out.items()}


def _known_submission_networks(raw_capacity: Any, coin_cfg: Dict[str, Any]) -> list[str]:
    names = set()
    if isinstance(coin_cfg, dict):
        if "wallet" in coin_cfg and isinstance(coin_cfg["wallet"], dict):
            # Transposed structure
            names.update(coin_cfg["wallet"].keys())
        else:
            # Old structure
            names.update(coin_cfg.keys())
    if isinstance(raw_capacity, dict):
        for key, value in raw_capacity.items():
            if isinstance(value, dict):
                names.add(str(key).upper())
    return sorted(name for name in names if name)


def _capacity_view_for_network(raw_capacity: Any, network: str) -> Dict[str, Any]:
    if not isinstance(raw_capacity, dict):
        return {}
    net_key = str(network).upper()
    nested = raw_capacity.get(net_key)
    if isinstance(nested, dict):
        return dict(nested)
    if any(
        key in raw_capacity
        for key in (
            "network_hashrate",
            "network_hashrate_hs",
            "hashrate",
            "difficulty",
            "network_difficulty",
            "block_time",
            "block_time_s",
        )
    ):
        return dict(raw_capacity)
    return {}


def _difficulty_from_target(target_value: Any) -> float:
    try:
        target_hex = str(target_value or "").strip()
        if target_hex.startswith("0x"):
            target_hex = target_hex[2:]
        target = int(target_hex or "0", 16)
    except Exception:
        return 0.0
    if target <= 0:
        return 0.0
    return float(DIFF1_TARGET / float(target))


def _share_difficulty(vsd: Any, network: str) -> float:
    try:
        rec = dict(vsd.get("miner/difficulty/%s" % str(network).upper(), {}) or {})
    except Exception:
        return 0.0
    method = str(rec.get("method", ""))
    params = rec.get("params", [])
    if method == "mining.set_difficulty" and isinstance(params, list) and params:
        return max(0.0, _safe_float(params[0], 0.0))
    if method == "mining.set_target" and isinstance(params, list) and params:
        return max(0.0, _difficulty_from_target(params[0]))
    return 0.0


def _share_hashrate_hs(share_rate_per_second: float, share_difficulty: float, hashes_per_diff: float) -> float:
    rate = max(0.0, _safe_float(share_rate_per_second, 0.0))
    difficulty = max(0.0, _safe_float(share_difficulty, 0.0))
    diff1_hashes = max(1.0, _safe_float(hashes_per_diff, float(2 ** 32)))
    return rate * difficulty * diff1_hashes


def _submit_rate_ceiling_per_second(coin_cfg: Dict[str, Any], rate_cfg: Dict[str, float], fallback: float) -> float:
    explicit = [
        _safe_float(coin_cfg.get("pool_submit_ceiling_per_second", 0.0), 0.0),
        _safe_float(coin_cfg.get("network_submit_ceiling_per_second", 0.0), 0.0),
    ]
    if any(value > 0.0 for value in explicit):
        return max(1.0e-6, _positive_min(explicit, fallback))
    defaults = [
        _safe_float(rate_cfg.get("pool_submit_ceiling_per_second", 0.0), 0.0),
        _safe_float(rate_cfg.get("network_submit_ceiling_per_second", 0.0), 0.0),
    ]
    return max(1.0e-6, _positive_min(defaults, fallback))


def _submit_guard_fraction(coin_cfg: Dict[str, Any], rate_cfg: Dict[str, float]) -> float:
    return _clamp(
        _safe_float(
            coin_cfg.get("pool_submit_guard_fraction", rate_cfg.get("pool_submit_guard_fraction", 0.985)),
            rate_cfg.get("pool_submit_guard_fraction", 0.985),
        ),
        0.50,
        1.0,
    )


def _tapered_submit_rate(guarded_submit_ceiling: float, ratio_progress: float, rate_cfg: Dict[str, float]) -> tuple[float, float, float]:
    ceiling = max(1.0e-6, _safe_float(guarded_submit_ceiling, 1.0e-6))
    start_ratio = _clamp(rate_cfg.get("share_rate_taper_start_ratio", 0.80), 0.0, 0.999)
    floor_fraction = _clamp(rate_cfg.get("share_rate_taper_floor_fraction", 0.85), 0.50, 1.0)
    taper_power = max(0.10, _safe_float(rate_cfg.get("share_rate_taper_power", 1.35), 1.35))
    progress = _clamp(_safe_float(ratio_progress, 0.0), 0.0, 2.0)
    if progress <= start_ratio:
        return ceiling, 1.0, progress
    norm = _clamp((progress - start_ratio) / max(1.0e-6, 1.0 - start_ratio), 0.0, 1.0)
    taper_fraction = 1.0 - ((1.0 - floor_fraction) * (norm ** taper_power))
    return max(1.0e-6, ceiling * taper_fraction), float(taper_fraction), progress


def _hashes_per_diff1(coin_cfg: Dict[str, Any]) -> float:
    override = _safe_float(coin_cfg.get("hashes_per_diff1", 0.0), 0.0)
    if override > 0.0:
        return override
    return float(2 ** 32)


def _local_hashrate_hs(vsd: Any, network: str, coin_cfg: Dict[str, Any]) -> float:
    key = "telemetry/mine/%s/accepted_hashrate_hs" % str(network).upper()
    try:
        live = _safe_float(vsd.get(key, 0.0), 0.0)
        if live > 0.0:
            return live
    except Exception:
        pass
    return max(0.0, _safe_float(coin_cfg.get("hashrate_sim_hs", 0.0), 0.0))


def _network_submission_state(
    vsd: Any,
    network: str,
    coin_cfg: Dict[str, Any],
    capacity_view: Dict[str, Any],
    rate_cfg: Dict[str, float],
) -> Dict[str, Any]:
    network_u = str(network).upper()
    metrics_key = "telemetry/metrics/%s/current" % network_u
    try:
        metrics = dict(vsd.get(metrics_key, {}) or {})
    except Exception:
        metrics = {}

    observed_valid = max(0.0, _safe_float(metrics.get("accepted_hs", 0.0), 0.0))
    observed_submitted = max(0.0, _safe_float(metrics.get("hashes_submitted_hs", 0.0), 0.0))
    acceptance_rate = max(0.0, _safe_float(metrics.get("acceptance_rate", 0.0), 0.0))
    if acceptance_rate <= 0.0 and observed_submitted > 1e-9:
        acceptance_rate = observed_valid / observed_submitted

    local_hs = _local_hashrate_hs(vsd, network_u, coin_cfg)
    share_difficulty = max(0.0, _share_difficulty(vsd, network_u))
    network_hashrate = max(
        0.0,
        _safe_float(
            capacity_view.get("network_hashrate_hs", capacity_view.get("network_hashrate", capacity_view.get("hashrate", 0.0))),
            0.0,
        ),
    )
    block_time = max(
        0.0,
        _safe_float(
            capacity_view.get("block_time_s", capacity_view.get("block_time", DEFAULT_BLOCK_TIME_S.get(network_u, 60.0))),
            DEFAULT_BLOCK_TIME_S.get(network_u, 60.0),
        ),
    )
    network_difficulty = max(
        0.0,
        _safe_float(
            capacity_view.get("network_difficulty", capacity_view.get("difficulty", 0.0)),
            0.0,
        ),
    )

    expected_valid = 0.0
    expected_source = "unavailable"
    hashes_per_diff = _hashes_per_diff1(coin_cfg)
    if local_hs > 0.0 and share_difficulty > 0.0 and hashes_per_diff > 0.0:
        expected_valid = local_hs / max(1.0, share_difficulty * hashes_per_diff)
        expected_source = "share_difficulty"

    if local_hs > 0.0 and network_hashrate > 0.0 and block_time > 0.0:
        block_rate = (local_hs / network_hashrate) / block_time
        if network_difficulty > 0.0 and share_difficulty > 0.0:
            expected_valid = block_rate * max(1e-9, network_difficulty / share_difficulty)
            expected_source = "network_vs_share_difficulty"
        elif expected_valid <= 0.0:
            expected_valid = block_rate
            expected_source = "network_hashrate_block_time"

    expected_weight = _clamp(rate_cfg.get("expected_weight", 0.70), 0.0, 1.0)
    observed_weight = _clamp(rate_cfg.get("observed_weight", 0.30), 0.0, 1.0)
    weight_sum = expected_weight + observed_weight
    if weight_sum <= 0.0:
        expected_weight = 1.0
        observed_weight = 0.0
    else:
        expected_weight /= weight_sum
        observed_weight /= weight_sum

    blended_valid = (expected_valid * expected_weight) + (observed_valid * observed_weight)
    if blended_valid <= 0.0:
        blended_valid = max(expected_valid, observed_valid)

    target_acceptance = _clamp(rate_cfg.get("target_acceptance_rate", 0.92), 0.50, 0.99)
    quality_factor = 1.0
    if acceptance_rate > 0.0:
        quality_factor = _clamp(acceptance_rate / target_acceptance, 0.50, 1.05)

    allowed_submit_rate = 0.0
    if blended_valid > 0.0:
        allowed_submit_rate = (blended_valid / target_acceptance) * quality_factor

    target_fraction = _clamp(rate_cfg.get("network_target_fraction", 0.05), 0.0, 1.0)
    target_hashrate_hs = max(0.0, network_hashrate * target_fraction)
    preferred_valid_rate = max(
        1.0e-6,
        _safe_float(
            coin_cfg.get(
                "preferred_valid_share_rate_per_second",
                coin_cfg.get(
                    "moderate_valid_share_rate_per_second",
                    rate_cfg.get(
                        "preferred_valid_share_rate_per_second",
                        rate_cfg.get("moderate_valid_share_rate_per_second", 2.0),
                    ),
                ),
            ),
            rate_cfg.get(
                "preferred_valid_share_rate_per_second",
                rate_cfg.get("moderate_valid_share_rate_per_second", 2.0),
            ),
        ),
    )
    observed_target_fraction = 0.0
    if network_hashrate > 0.0 and local_hs > 0.0:
        observed_target_fraction = local_hs / network_hashrate
    observed_submit_hashrate_hs = _share_hashrate_hs(observed_submitted, share_difficulty, hashes_per_diff)
    observed_valid_hashrate_hs = _share_hashrate_hs(observed_valid, share_difficulty, hashes_per_diff)
    ratio_progress = 0.0
    if target_hashrate_hs > 0.0:
        ratio_progress = max(
            observed_target_fraction / max(target_fraction, 1.0e-9) if target_fraction > 0.0 else 0.0,
            observed_submit_hashrate_hs / max(target_hashrate_hs, 1.0e-9),
            observed_valid_hashrate_hs / max(target_hashrate_hs, 1.0e-9),
        )
    submit_ceiling = _submit_rate_ceiling_per_second(
        coin_cfg,
        rate_cfg,
        max(rate_cfg.get("fallback_allowed_rate_per_second", 2.0), allowed_submit_rate, 1.0),
    )
    submit_guard = _submit_guard_fraction(coin_cfg, rate_cfg)
    guarded_submit_ceiling = max(1.0e-6, submit_ceiling * submit_guard)
    preferred_submit_rate = min(
        guarded_submit_ceiling,
        max(1.0e-6, preferred_valid_rate / max(target_acceptance, 1.0e-6)),
    )
    target_share_difficulty = 1.0
    if target_hashrate_hs > 0.0 and preferred_submit_rate > 0.0 and target_acceptance > 0.0:
        target_share_difficulty = max(
            1.0,
            target_hashrate_hs / (preferred_submit_rate * target_acceptance * hashes_per_diff),
        )
    assigned_share_difficulty = max(1.0, share_difficulty) if share_difficulty > 0.0 else float(target_share_difficulty)
    share_difficulty_gap_ratio = max(1.0e-6, target_share_difficulty / max(assigned_share_difficulty, 1.0e-6))
    assigned_required_valid_rate = 0.0
    assigned_required_submit_rate = 0.0
    if target_hashrate_hs > 0.0 and assigned_share_difficulty > 0.0:
        assigned_required_valid_rate = target_hashrate_hs / (assigned_share_difficulty * hashes_per_diff)
        assigned_required_submit_rate = assigned_required_valid_rate / max(target_acceptance, 1.0e-6)
    desired_submit_rate = min(
        guarded_submit_ceiling,
        max(allowed_submit_rate, preferred_submit_rate, assigned_required_submit_rate),
    )
    tapered_allowed, taper_fraction, ratio_progress = _tapered_submit_rate(
        desired_submit_rate,
        ratio_progress,
        rate_cfg,
    )
    allowed_submit_rate = min(guarded_submit_ceiling, max(0.0, allowed_submit_rate, tapered_allowed))

    active = any(
        value > 0.0
        for value in (
            local_hs,
            share_difficulty,
            network_hashrate,
            network_difficulty,
            observed_valid,
            observed_submitted,
        )
    )
    expected_interval_s = 0.0
    if expected_valid > 1e-9:
        expected_interval_s = 1.0 / expected_valid

    return {
        "network": network_u,
        "active": bool(active),
        "local_hashrate_hs": float(local_hs),
        "network_hashrate_hs": float(network_hashrate),
        "network_difficulty": float(network_difficulty),
        "share_difficulty": float(share_difficulty),
        "block_time_s": float(block_time),
        "expected_valid_share_rate": float(expected_valid),
        "observed_valid_share_rate": float(observed_valid),
        "observed_submit_rate": float(observed_submitted),
        "acceptance_rate": float(acceptance_rate),
        "allowed_submit_rate": float(max(0.0, allowed_submit_rate)),
        "target_fraction": float(target_fraction),
        "target_hashrate_hs": float(target_hashrate_hs),
        "target_share_difficulty": float(target_share_difficulty),
        "preferred_valid_share_rate_per_second": float(preferred_valid_rate),
        "preferred_submit_rate_per_second": float(preferred_submit_rate),
        "desired_submit_rate_per_second": float(desired_submit_rate),
        "assigned_share_difficulty": float(assigned_share_difficulty),
        "share_difficulty_gap_ratio": float(share_difficulty_gap_ratio),
        "assigned_required_valid_share_rate_per_second": float(assigned_required_valid_rate),
        "assigned_required_submit_rate_per_second": float(assigned_required_submit_rate),
        "observed_target_fraction": float(observed_target_fraction),
        "observed_submit_hashrate_hs": float(observed_submit_hashrate_hs),
        "observed_valid_hashrate_hs": float(observed_valid_hashrate_hs),
        "share_rate_taper_fraction": float(taper_fraction),
        "ratio_progress": float(ratio_progress),
        "submit_rate_ceiling_per_second": float(submit_ceiling),
        "guarded_submit_rate_ceiling_per_second": float(guarded_submit_ceiling),
        "expected_interval_s": float(expected_interval_s),
        "expected_source": expected_source,
        "quality_factor": float(quality_factor),
    }


def compute_submission_rate_state(
    vsd: Any,
    raw_capacity: Any,
    coin_cfg: Dict[str, Any],
    base_tick_s: float,
) -> Dict[str, Any]:
    rate_cfg = _submission_rate_cfg()
    per_network: Dict[str, Any] = {}
    total_allowed = 0.0

    for network in _known_submission_networks(raw_capacity, coin_cfg):
        coin_info = _get_coin_info(coin_cfg, network)
        cap_view = _capacity_view_for_network(raw_capacity, network)
        state = _network_submission_state(vsd, network, coin_info, cap_view, rate_cfg)
        per_network[network] = state
        total_allowed += _safe_float(state.get("allowed_submit_rate", 0.0), 0.0)

    if total_allowed <= 0.0:
        total_allowed = max(1.0, rate_cfg.get("fallback_allowed_rate_per_second", 2.0))

    prev_allowed = 0.0
    try:
        prev_allowed = _safe_float(dict(vsd.get("miner/runtime/submission_rate", {}) or {}).get("allowed_rate_per_second", 0.0), 0.0)
    except Exception:
        prev_allowed = 0.0

    smoothing = _clamp(rate_cfg.get("smoothing", 0.60), 0.0, 0.95)
    if prev_allowed > 0.0 and smoothing > 0.0:
        total_allowed = (prev_allowed * smoothing) + (total_allowed * (1.0 - smoothing))

    total_allowed = max(1.0, total_allowed)
    base_tick = max(0.01, _safe_float(base_tick_s, 0.25))
    min_tick = _clamp(rate_cfg.get("min_tick_duration_s", 0.05), 0.01, base_tick)
    target_interval_s = 1.0 / total_allowed
    tick_duration = min(base_tick, max(min_tick, target_interval_s * 0.5))
    jitter_fraction = _clamp(rate_cfg.get("jitter_fraction", 0.18), 0.0, 0.45)
    jitter_window_s = min(tick_duration * jitter_fraction, target_interval_s * jitter_fraction)

    for network, state in per_network.items():
        net_allowed = _safe_float(state.get("allowed_submit_rate", 0.0), 0.0)
        net_interval = (1.0 / net_allowed) if net_allowed > 1e-9 else 0.0
        state["tick_duration_s"] = float(tick_duration)
        state["jitter_window_s"] = float(
            min(jitter_window_s, net_interval * jitter_fraction) if net_interval > 0.0 else 0.0
        )

    return {
        "ts": time.time(),
        "model": "expected_hashrate_difficulty_valid_share",
        "allowed_rate_per_second": float(total_allowed),
        "tick_duration": float(tick_duration),
        "jitter_window_s": float(jitter_window_s),
        "target_acceptance_rate": float(rate_cfg.get("target_acceptance_rate", 0.92)),
        "per_network": per_network,
    }

# ============================================================================
# SECTION 5: BUILD PIPELINE (CORE)
# ============================================================================

def build(env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the full miner stack.

    Required:
      - telemetry_reader: () -> dict
      - network_capacity_fn: () -> dict

    This function wires:
      - ComputeManager (generic compute fabric)
      - Parallel Tiers + Clusters hooks
      - ShareAllocator
      - Submitter
      - FailsafeGovernor
      - MinerEngine (wrapper-driven)
    """

    LOG.info("build(): starting stack synthesis")

    # BIOS-level components
    vsd = env.get("vsd") or VSDManager()
    read_tel = env.get("telemetry_reader")
    capacity_fn = env.get("network_capacity_fn")

    if not callable(read_tel):
        raise RuntimeError("build(): telemetry_reader is missing or not callable")

    if not callable(capacity_fn):
        raise RuntimeError("build(): network_capacity_fn is missing or not callable")

    # Hardware normalization
    hw = _normalize_hardware(env)

    # Extended tiers
    tiers = _build_extended_tiers()

    # ---------------------------------------------------------------------
    # ComputeManager Integration
    # ---------------------------------------------------------------------
    cm = ComputeManager({
        "tiers": tiers,
        "hardware": hw,
        "vsd": vsd
    })

    LOG.info("ComputeManager initialized with extended tiers")

    # ---------------------------------------------------------------------
    # ShareAllocator
    # ---------------------------------------------------------------------
    share_alloc = ShareAllocator(
        read_telemetry=read_tel,
        list_active_lanes=lambda: cm.status()
    )

    LOG.info("ShareAllocator initialized")

    # ---------------------------------------------------------------------
    # Submitter
    # ---------------------------------------------------------------------
    submitter = Submitter(
        read_telemetry=read_tel,
        network_capacity_fn=capacity_fn,
        vsd=vsd,
        tick_duration_s=float(CFG.get("tick_interval_s", 0.25)),
    )

    LOG.info("Submitter initialized")

    # ---------------------------------------------------------------------
    # FailsafeGovernor (mining-only)
    # ---------------------------------------------------------------------
    failsafe = FailsafeGovernor(
        list_lanes=lambda: [
            {"lane_id": lid, **cm.status().get(lid, {})}
            for lid in cm.status().keys()
        ],
        disable_lane=lambda lid: cm.lanes.get(lid) and setattr(cm.lanes[lid], "active", False),
        difficulty_of=lambda net: 1.0  # extended runtime uses static difficulty here
    )

    LOG.info("FailsafeGovernor initialized")

    # ---------------------------------------------------------------------
    # MinerEngine (ComputeManager + handler_miner)
    # ---------------------------------------------------------------------
    # Optional miner nonce configuration (defaults + per-network overrides)
    nonce_cfg = CFG.get("miner_nonce", {}) if isinstance(CFG, dict) else {}

    engine = MinerEngine(
        compute_manager=cm,
        share_allocator=share_alloc,
        submitter=submitter,
        failsafe=failsafe,
        read_telemetry=read_tel,
        vsd=vsd,
        handler_miner=None,
        tick_interval_s=float(CFG.get("tick_interval_s", 0.25)),
        network_cfg=capacity_fn(),
        nonce_cfg=nonce_cfg,
    )

    LOG.info("MinerEngine constructed")

    return {
        "cm": cm,
        "share_allocator": share_alloc,
        "submitter": submitter,
        "failsafe": failsafe,
        "engine": engine,
        "read_tel": read_tel,
        "capacity_fn": capacity_fn,
        "vsd": vsd
    }

# ============================================================================
# SECTION 6: EXTENDED PARALLELISM CONTROLLER
# ============================================================================

def _spawn_parallelism(cm: ComputeManager, hw: Dict[str, Any], tiers: Any) -> None:
    """
    Extended parallelism controller.

    Responsibilities:
      - Pre-warm compute lanes
      - Expand clusters per tier
      - Initialize kernel slots for each cluster
      - ComputeManager handles the actual lane instantiation

    This function does NOT:
      - allocate mining logic
      - handle nonces
      - involve prediction engine
    """

    LOG.info("Spawning extended parallelism...")

    # For each tier, spawn cluster_count * initial_lanes
    for t in tiers:
        tid = t["tier_id"]
        clusters = t.get("cluster_count", 1)
        kernels = t.get("kernel_slots", 2)

        for c in range(clusters):
            # initial lanes per cluster
            init_lanes = kernels * 2

            for _ in range(init_lanes):
                try:
                    cm.allocate_lane(tid)
                except Exception:
                    LOG.error("Lane spawn failed", exc_info=True)

    LOG.info("Extended parallelism spawn complete")

# ============================================================================
# SECTION 7: MAIN RUNTIME LOOP
# ============================================================================

def MinerRuntime(stack: Dict[str, Any]) -> Dict[str, Any]:
    vsd = stack["vsd"]
    engine = stack["engine"]
    cm = stack["cm"]

    # BIOS gate
    _wait_for_bios(vsd)

    # Announce start
    try:
        BUS.publish("miner.runtime.start", {"ts": time.time()})
    except Exception:
        pass

    LOG.info("MinerRuntime(): invoking extended parallelism warm-up")

    # Warm-up extended parallelism
    try:
        # Need tiers from ComputeManager cfg
        tiers = CFG.get("tiers") or _build_extended_tiers()
        hw = stack.get("hardware", {})
        _spawn_parallelism(cm, hw, tiers)
    except Exception:
        LOG.error("Parallel warm-up failed", exc_info=True)

    LOG.info("MinerRuntime(): starting MinerEngine thread")

    engine.start()

    LOG.info("MinerRuntime(): live loop active")

    # Hard loop (indefinite)
    # Publish submission rate telemetry every tick
    tick_dur = float(CFG.get("tick_interval_s", 0.25))
    last_sec = int(time.time())
    try:
        while True:
            # Refresh the governed base rate using network difficulty and valid-share telemetry.
            try:
                existing_rate_state = dict(stack["vsd"].get("miner/runtime/submission_rate", {}) or {})
            except Exception:
                existing_rate_state = {}

            try:
                existing_model = str(existing_rate_state.get("model", ""))
                existing_networks = dict(existing_rate_state.get("per_network", {}) or {})
                existing_ts = _safe_float(existing_rate_state.get("ts", 0.0), 0.0)
                has_pool_governor = bool(existing_networks) and existing_model.startswith("network_fraction_exact_band")
                if has_pool_governor and (time.time() - existing_ts) <= max(5.0, tick_dur * 40.0):
                    rate_state = existing_rate_state
                else:
                    cap = stack.get("capacity_fn", lambda: {})() or {}
                    rate_state = compute_submission_rate_state(
                        vsd=stack["vsd"],
                        raw_capacity=cap,
                        coin_cfg=dict(CFG.get("coins", {})),
                        base_tick_s=tick_dur,
                    )
            except Exception:
                rate_state = {
                    "ts": time.time(),
                    "model": "expected_hashrate_difficulty_valid_share",
                    "allowed_rate_per_second": 2.0,
                    "tick_duration": tick_dur,
                    "jitter_window_s": 0.0,
                    "per_network": {},
                }

            # Write runtime submission rate telemetry
            try:
                stack["vsd"].store("miner/runtime/submission_rate", rate_state)
            except Exception:
                pass

            # Publish a minimal rolling stat per second
            try:
                now_sec = int(time.time())
                if now_sec != last_sec:
                    last_sec = now_sec
                    # Mirror submitter snapshot to runtime path for observability
                    if hasattr(stack.get("submitter"), "governor_snapshot"):
                        snap = stack["submitter"].governor_snapshot()
                        stack["vsd"].store("miner/runtime/submission_rate_snapshot", snap)
            except Exception:
                pass

            time.sleep(max(0.05, tick_dur))
    except KeyboardInterrupt:
        LOG.info("MinerRuntime(): interrupted by keyboard")
    except Exception as exc:
        LOG.error("MinerRuntime(): unexpected error %s", exc, exc_info=True)
    finally:
        LOG.info("MinerRuntime(): stopping MinerEngine")
        try:
            BUS.publish("miner.runtime.stop", {"ts": time.time()})
        except Exception:
            pass

        try:
            engine.stop()
        except Exception:
            LOG.error("MinerEngine stop failed", exc_info=True)

        LOG.info("MinerRuntime(): terminated")

    return {"ok": True}

# ============================================================================
# SECTION 8: BIOS ENTRYPOINT
# ============================================================================

def main(env: Dict[str, Any]) -> Dict[str, Any]:
    LOG.info("miner_runtime.main(): initializing miner environment")
    stack = build(env)
    return MinerRuntime(stack)
