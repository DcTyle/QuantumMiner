# ============================================================================
# Quantum Application / miner
# ASCII-ONLY SOURCE FILE
# File: stratum_adapter.py
# Version: v1.0.0 (Stratum-to-Engine Bridge, Multi-Coin)
# ----------------------------------------------------------------------------
# Purpose
# -------
# Bridge real Stratum pool jobs into MinerEngine by:
#   - Reading miner_runtime_config.json (coins, endpoints, wallets)
#   - Spinning up a StratumClient per enabled coin
#   - Subscribing to Stratum bus events (mining.notify, difficulty)
#   - Converting jobs into the engine's job format
#   - Writing per-lane job mappings into VSD (payload consumed by MinerEngine)
#   - Providing a client_resolver for Submitter to submit shares to pools
#
# Design
# ------
# - EventBus-based, VSD-backed, ASCII-only
# - Multi-coin simultaneous feed with lane segregation (round-robin)
# - No circular imports: only touches public Engine API & VSD
# - ADA v4.7 compliant logging pattern
# - Neuralis-friendly: clear separation of responsibilities
# ============================================================================

from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import json
import time
import threading
import logging
import sys
from miner.common_types import QMJob, qmjob_from_dict
from dataclasses import asdict as _asdict
from miner.coin_switcher import SupportedCoin, coin_config_entry, configured_coins, resolve_coin_profile
from miner.pool_stats import PoolStats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_def_fmt = logging.Formatter(
    fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.Formatter.converter = time.gmtime  # UTC
_logger = logging.getLogger("miner.stratum_adapter")
if not _logger.handlers:
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(_def_fmt)
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)

DEFAULT_HASHES_PER_DIFF1 = float(2 ** 32)
DIFF1_TARGET = int(
    "00000000FFFF0000000000000000000000000000000000000000000000000000",
    16,
)
DEFAULT_SUBMISSION_POLICY = {
    "network_target_fraction": 0.05,
    "network_target_fraction_floor": 0.05,
    "network_target_fraction_ceiling": 0.05002,
    "network_target_fraction_nominal": 0.05001,
    "moderate_valid_share_rate_per_second": 2.0,
    "preferred_valid_share_rate_per_second": 2.0,
    "difficulty_request_interval_s": 60.0,
    "difficulty_request_change_fraction": 0.05,
    "target_acceptance_rate": 0.92,
    "fallback_allowed_rate_per_second": 2.0,
    "pool_submit_ceiling_per_second": 2.0,
    "network_submit_ceiling_per_second": 2.0,
    "pool_submit_guard_fraction": 0.985,
    "share_rate_taper_start_ratio": 0.80,
    "share_rate_taper_floor_fraction": 0.85,
    "share_rate_taper_power": 1.35,
    "difficulty_request_low_rate_fraction": 0.90,
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


def _submission_policy(cfg: Dict[str, Any]) -> Dict[str, float]:
    raw = dict(cfg.get("submission_rate", {})) if isinstance(cfg.get("submission_rate", {}), dict) else {}
    out = dict(DEFAULT_SUBMISSION_POLICY)
    out.update(raw)
    return {str(k): _safe_float(v, out.get(str(k), 0.0)) for k, v in out.items()}


def _hashes_per_diff1(coin_cfg: Dict[str, Any]) -> float:
    val = _safe_float(coin_cfg.get("hashes_per_diff1", DEFAULT_HASHES_PER_DIFF1), DEFAULT_HASHES_PER_DIFF1)
    return max(1.0, val)


def _positive_min(values: list[float], default: float) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    if positives:
        return min(positives)
    return float(default)


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


def _share_difficulty_from_record(record: Dict[str, Any]) -> float:
    if not isinstance(record, dict):
        return 0.0
    method = str(record.get("method", ""))
    params = record.get("params", [])
    if method == "mining.set_difficulty" and isinstance(params, list) and params:
        return max(0.0, _safe_float(params[0], 0.0))
    if method == "mining.set_target" and isinstance(params, list) and params:
        return max(0.0, _difficulty_from_target(params[0]))
    return 0.0


def _share_difficulty_from_vsd(vsd: Any, network: str) -> float:
    try:
        rec = dict(vsd.get("miner/difficulty/%s" % str(network).upper(), {}) or {})
    except Exception:
        rec = {}
    return _share_difficulty_from_record(rec)


def _target_hex_from_difficulty(difficulty: float) -> str:
    diff = max(1.0, _safe_float(difficulty, 1.0))
    target = max(1, int(DIFF1_TARGET / diff))
    return ("%064x" % target)


def _normalize_target_hex(value: Any) -> str:
    try:
        text = str(value or "").strip().lower()
    except Exception:
        text = ""
    if text.startswith("0x"):
        text = text[2:]
    text = "".join(ch for ch in text if ch in "0123456789abcdef")
    if not text:
        return ""
    return text[-64:].rjust(64, "0")


def _share_target_hex_from_record(record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    method = str(record.get("method", ""))
    params = record.get("params", [])
    if method == "mining.set_target" and isinstance(params, list) and params:
        return _normalize_target_hex(params[0])
    if method == "mining.set_difficulty" and isinstance(params, list) and params:
        diff = max(0.0, _safe_float(params[0], 0.0))
        if diff > 0.0:
            return _target_hex_from_difficulty(diff)
    return ""


def _share_target_hex_from_vsd(vsd: Any, network: str) -> str:
    try:
        rec = dict(vsd.get("miner/difficulty/%s" % str(network).upper(), {}) or {})
    except Exception:
        rec = {}
    return _share_target_hex_from_record(rec)


def _share_difficulty_from_job(job_payload: Dict[str, Any]) -> float:
    if not isinstance(job_payload, dict):
        return 0.0
    share_target = _normalize_target_hex(job_payload.get("share_target", job_payload.get("active_target", "")))
    if not share_target:
        return 0.0
    return max(0.0, _difficulty_from_target(share_target))


def _share_hashrate_hs(share_rate_per_second: float, share_difficulty: float, hashes_per_diff: float) -> float:
    rate = max(0.0, _safe_float(share_rate_per_second, 0.0))
    difficulty = max(0.0, _safe_float(share_difficulty, 0.0))
    diff1_hashes = max(1.0, _safe_float(hashes_per_diff, DEFAULT_HASHES_PER_DIFF1))
    return rate * difficulty * diff1_hashes


def _tapered_submit_rate(
    guarded_submit_ceiling: float,
    ratio_progress: float,
    policy: Dict[str, float],
) -> tuple[float, float, float]:
    ceiling = max(1.0e-6, _safe_float(guarded_submit_ceiling, 1.0e-6))
    start_ratio = _clamp(policy.get("share_rate_taper_start_ratio", 0.80), 0.0, 0.999)
    floor_fraction = _clamp(policy.get("share_rate_taper_floor_fraction", 0.85), 0.50, 1.0)
    taper_power = max(0.10, _safe_float(policy.get("share_rate_taper_power", 1.35), 1.35))
    progress = _clamp(_safe_float(ratio_progress, 0.0), 0.0, 2.0)
    if progress <= start_ratio:
        return ceiling, 1.0, progress
    norm = _clamp((progress - start_ratio) / max(1.0e-6, 1.0 - start_ratio), 0.0, 1.0)
    taper_fraction = 1.0 - ((1.0 - floor_fraction) * (norm ** taper_power))
    return max(1.0e-6, ceiling * taper_fraction), float(taper_fraction), progress


def _observed_hashrate_hs(metrics: Dict[str, Any]) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    for key in ("local_hashrate_hs", "accepted_hashrate_hs", "hashrate_hs", "measured_hashrate_hs"):
        value = _safe_float(metrics.get(key, 0.0), 0.0)
        if value > 0.0:
            return value
    return 0.0


def _target_fraction_bounds(policy: Dict[str, float], coin_cfg: Dict[str, Any]) -> tuple[float, float, float]:
    fallback = _clamp(policy.get("network_target_fraction", 0.05), 0.0, 1.0)
    floor = _clamp(
        _safe_float(coin_cfg.get("network_target_fraction_floor", policy.get("network_target_fraction_floor", fallback)), fallback),
        0.0,
        1.0,
    )
    ceiling = _clamp(
        _safe_float(coin_cfg.get("network_target_fraction_ceiling", policy.get("network_target_fraction_ceiling", max(floor, fallback))), max(floor, fallback)),
        floor,
        1.0,
    )
    nominal_default = (floor + ceiling) * 0.5
    nominal = _clamp(
        _safe_float(coin_cfg.get("network_target_fraction_nominal", policy.get("network_target_fraction_nominal", nominal_default)), nominal_default),
        floor,
        ceiling,
    )
    return floor, ceiling, nominal


def _submit_rate_ceiling_per_second(
    coin_cfg: Dict[str, Any],
    network_stats: Dict[str, Any],
    policy: Dict[str, float],
) -> float:
    fallback = max(1.0e-6, _safe_float(policy.get("fallback_allowed_rate_per_second", 2.0), 2.0))
    explicit = [
        _safe_float(coin_cfg.get("pool_submit_ceiling_per_second", 0.0), 0.0),
        _safe_float(coin_cfg.get("network_submit_ceiling_per_second", 0.0), 0.0),
        _safe_float(network_stats.get("pool_submit_ceiling_per_second", 0.0), 0.0),
        _safe_float(network_stats.get("network_submit_ceiling_per_second", 0.0), 0.0),
    ]
    if any(value > 0.0 for value in explicit):
        return max(1.0e-6, _positive_min(explicit, fallback))
    defaults = [
        _safe_float(policy.get("pool_submit_ceiling_per_second", 0.0), 0.0),
        _safe_float(policy.get("network_submit_ceiling_per_second", 0.0), 0.0),
    ]
    return max(1.0e-6, _positive_min(defaults, fallback))


def _submit_guard_fraction(coin_cfg: Dict[str, Any], policy: Dict[str, float]) -> float:
    return _clamp(
        _safe_float(
            coin_cfg.get("pool_submit_guard_fraction", policy.get("pool_submit_guard_fraction", 0.985)),
            policy.get("pool_submit_guard_fraction", 0.985),
        ),
        0.50,
        1.0,
    )


def _control_target_fraction(
    observed_fraction: float,
    floor: float,
    ceiling: float,
    nominal: float,
) -> tuple[float, str]:
    if observed_fraction > ceiling + 1.0e-12:
        return floor, "pull_down"
    if observed_fraction > 0.0 and observed_fraction < floor - 1.0e-12:
        return ceiling, "pull_up"
    if observed_fraction <= 0.0:
        return nominal, "bootstrap"
    return nominal, "hold"


def _build_network_target_snapshot(
    network: str,
    coin_cfg: Dict[str, Any],
    network_stats: Dict[str, Any],
    policy: Dict[str, float],
    metrics: Optional[Dict[str, Any]] = None,
    current_share_difficulty: float = 0.0,
) -> Dict[str, Any]:
    network_u = str(network).upper()
    network_hashrate_hs = max(0.0, _safe_float(network_stats.get("network_hashrate_hs", 0.0), 0.0))
    block_time_s = max(0.0, _safe_float(network_stats.get("block_time_s", 0.0), 0.0))
    network_difficulty = max(0.0, _safe_float(network_stats.get("difficulty", 0.0), 0.0))
    metrics = dict(metrics or {})
    target_fraction_floor, target_fraction_ceiling, target_fraction_nominal = _target_fraction_bounds(policy, coin_cfg)
    observed_hashrate_hs = _observed_hashrate_hs(metrics)
    observed_submit_rate = max(0.0, _safe_float(metrics.get("hashes_submitted_hs", metrics.get("submitted_hs", 0.0)), 0.0))
    observed_valid_rate = max(0.0, _safe_float(metrics.get("accepted_hs", metrics.get("hashes_found_hs", 0.0)), 0.0))
    observed_target_fraction = 0.0
    if network_hashrate_hs > 0.0 and observed_hashrate_hs > 0.0:
        observed_target_fraction = observed_hashrate_hs / network_hashrate_hs
    control_target_fraction, control_action = _control_target_fraction(
        observed_target_fraction,
        target_fraction_floor,
        target_fraction_ceiling,
        target_fraction_nominal,
    )

    acceptance_target = _clamp(
        _safe_float(coin_cfg.get("target_acceptance_rate", policy.get("target_acceptance_rate", 0.92)), policy.get("target_acceptance_rate", 0.92)),
        0.50,
        0.99,
    )
    preferred_valid_rate = max(
        1.0e-6,
        _safe_float(
            coin_cfg.get(
                "preferred_valid_share_rate_per_second",
                coin_cfg.get(
                    "moderate_valid_share_rate_per_second",
                    policy.get(
                        "preferred_valid_share_rate_per_second",
                        policy.get("moderate_valid_share_rate_per_second", 2.0),
                    ),
                ),
            ),
            policy.get("preferred_valid_share_rate_per_second", policy.get("moderate_valid_share_rate_per_second", 2.0)),
        ),
    )
    submit_ceiling = _submit_rate_ceiling_per_second(coin_cfg, network_stats, policy)
    submit_guard = _submit_guard_fraction(coin_cfg, policy)
    guarded_submit_ceiling = max(1.0e-6, submit_ceiling * submit_guard)
    preferred_submit_rate = min(
        guarded_submit_ceiling,
        max(1.0e-6, preferred_valid_rate / max(acceptance_target, 1.0e-6)),
    )

    target_hashrate_floor_hs = max(0.0, network_hashrate_hs * target_fraction_floor)
    target_hashrate_ceiling_hs = max(0.0, network_hashrate_hs * target_fraction_ceiling)
    target_hashrate_hs = max(0.0, network_hashrate_hs * control_target_fraction)
    hashes_per_diff = _hashes_per_diff1(coin_cfg)
    target_share_difficulty = 1.0
    if target_hashrate_hs > 0.0 and preferred_submit_rate > 0.0 and acceptance_target > 0.0:
        target_share_difficulty = max(
            1.0,
            target_hashrate_hs / (preferred_submit_rate * acceptance_target * hashes_per_diff),
        )
    current_share_difficulty = max(0.0, _safe_float(current_share_difficulty, 0.0))
    assigned_share_difficulty = max(1.0, current_share_difficulty) if current_share_difficulty > 0.0 else float(target_share_difficulty)
    share_difficulty_gap_ratio = max(1.0e-6, target_share_difficulty / max(assigned_share_difficulty, 1.0e-6))
    assigned_required_valid_rate = 0.0
    assigned_required_submit_rate = 0.0
    if target_hashrate_hs > 0.0 and assigned_share_difficulty > 0.0:
        assigned_required_valid_rate = target_hashrate_hs / (assigned_share_difficulty * hashes_per_diff)
        assigned_required_submit_rate = assigned_required_valid_rate / max(acceptance_target, 1.0e-6)
    observed_submit_hashrate_hs = _share_hashrate_hs(observed_submit_rate, current_share_difficulty, hashes_per_diff)
    observed_valid_hashrate_hs = _share_hashrate_hs(observed_valid_rate, current_share_difficulty, hashes_per_diff)
    target_progress = 0.0
    if control_target_fraction > 0.0:
        target_progress = observed_target_fraction / max(control_target_fraction, 1.0e-9)
    share_progress = 0.0
    if target_hashrate_hs > 0.0:
        share_progress = max(
            observed_submit_hashrate_hs / max(target_hashrate_hs, 1.0e-9),
            observed_valid_hashrate_hs / max(target_hashrate_hs, 1.0e-9),
        )
    ratio_progress = max(target_progress, share_progress)
    desired_submit_rate = min(
        guarded_submit_ceiling,
        max(preferred_submit_rate, assigned_required_submit_rate),
    )
    allowed_submit_rate, taper_fraction, ratio_progress = _tapered_submit_rate(
        desired_submit_rate,
        ratio_progress,
        policy,
    )
    target_valid_rate = max(1.0e-6, allowed_submit_rate * acceptance_target)
    expected_valid_rate = 0.0
    planned_hashrate_hs = 0.0
    if target_hashrate_hs > 0.0 and target_share_difficulty > 0.0:
        expected_valid_rate = target_hashrate_hs / (target_share_difficulty * hashes_per_diff)
        planned_hashrate_hs = expected_valid_rate * target_share_difficulty * hashes_per_diff
    return {
        "network": network_u,
        "network_hashrate_hs": float(network_hashrate_hs),
        "block_time_s": float(block_time_s),
        "network_difficulty": float(network_difficulty),
        "target_fraction": float(control_target_fraction),
        "target_fraction_floor": float(target_fraction_floor),
        "target_fraction_ceiling": float(target_fraction_ceiling),
        "target_fraction_nominal": float(target_fraction_nominal),
        "control_target_fraction": float(control_target_fraction),
        "control_action": control_action,
        "target_hashrate_floor_hs": float(target_hashrate_floor_hs),
        "target_hashrate_ceiling_hs": float(target_hashrate_ceiling_hs),
        "target_hashrate_hs": float(target_hashrate_hs),
        "planned_hashrate_hs": float(planned_hashrate_hs),
        "observed_hashrate_hs": float(observed_hashrate_hs),
        "observed_target_fraction": float(observed_target_fraction),
        "observed_submit_rate_per_second": float(observed_submit_rate),
        "observed_valid_share_rate_per_second": float(observed_valid_rate),
        "current_share_difficulty": float(current_share_difficulty),
        "observed_submit_hashrate_hs": float(observed_submit_hashrate_hs),
        "observed_valid_hashrate_hs": float(observed_valid_hashrate_hs),
        "ratio_progress": float(ratio_progress),
        "share_rate_taper_fraction": float(taper_fraction),
        "preferred_submit_rate_per_second": float(preferred_submit_rate),
        "desired_submit_rate_per_second": float(desired_submit_rate),
        "assigned_share_difficulty": float(assigned_share_difficulty),
        "share_difficulty_gap_ratio": float(share_difficulty_gap_ratio),
        "assigned_required_valid_share_rate_per_second": float(assigned_required_valid_rate),
        "assigned_required_submit_rate_per_second": float(assigned_required_submit_rate),
        "target_share_difficulty": float(target_share_difficulty),
        "target_valid_share_rate_per_second": float(target_valid_rate),
        "expected_valid_share_rate_per_second": float(expected_valid_rate),
        "preferred_valid_share_rate_per_second": float(preferred_valid_rate),
        "submit_rate_ceiling_per_second": float(submit_ceiling),
        "guarded_submit_rate_ceiling_per_second": float(guarded_submit_ceiling),
        "allowed_submit_rate": float(allowed_submit_rate),
        "allowed_submit_rate_per_second": float(allowed_submit_rate),
        "target_acceptance_rate": float(acceptance_target),
        "hashes_per_diff1": float(hashes_per_diff),
        "active": bool(network_hashrate_hs > 0.0),
    }


def _build_runtime_submission_state(
    targets: Dict[str, Dict[str, Any]],
    policy: Dict[str, float],
    tick_duration_s: float,
    worker_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    target_acceptance = _clamp(policy.get("target_acceptance_rate", 0.92), 0.50, 0.99)
    min_tick = _clamp(policy.get("min_tick_duration_s", 0.05), 0.01, max(0.01, tick_duration_s))
    jitter_fraction = _clamp(policy.get("jitter_fraction", 0.18), 0.0, 0.45)
    total_allowed = 0.0
    per_network: Dict[str, Any] = {}
    worker_map = dict(worker_map or {})

    for network, target in targets.items():
        if not bool(target.get("active")):
            continue
        allowed = max(0.0, _safe_float(target.get("allowed_submit_rate_per_second", target.get("allowed_submit_rate", 0.0)), 0.0))
        snap = dict(target)
        snap["allowed_submit_rate"] = float(allowed)
        network_u = str(network).upper()
        lane_workers = []
        for lane_id, worker in sorted(worker_map.items()):
            worker_coin = str(dict(worker or {}).get("coin", "")).upper()
            if worker_coin != network_u:
                continue
            lane_workers.append((str(lane_id), dict(worker or {})))
        worker_count = len(lane_workers)
        workers = []
        if worker_count > 0:
            per_worker_allowed = allowed / float(worker_count)
            per_worker_valid = _safe_float(snap.get("target_valid_share_rate_per_second", 0.0), 0.0) / float(worker_count)
            per_worker_hashrate = _safe_float(snap.get("target_hashrate_hs", 0.0), 0.0) / float(worker_count)
            for lane_id, worker in lane_workers:
                share_spacing_s = (1.0 / per_worker_allowed) if per_worker_allowed > 1.0e-9 else 0.0
                workers.append({
                    "lane_id": lane_id,
                    "coin": network_u,
                    "worker_index": int(worker.get("worker_index", 1) or 1),
                    "username": str(worker.get("username", "")),
                    "allowed_submit_rate": float(per_worker_allowed),
                    "target_valid_share_rate_per_second": float(per_worker_valid),
                    "target_hashrate_hs": float(per_worker_hashrate),
                    "share_spacing_s": float(share_spacing_s),
                })
        snap["worker_count"] = int(worker_count)
        snap["workers"] = workers
        per_network[network_u] = snap
        total_allowed += allowed

    if total_allowed <= 0.0:
        total_allowed = max(1.0e-6, _safe_float(policy.get("fallback_allowed_rate_per_second", 2.0), 2.0))

    base_tick = max(0.01, _safe_float(tick_duration_s, 0.25),)
    target_interval_s = 1.0 / total_allowed
    tick_duration = min(base_tick, max(min_tick, target_interval_s * 0.5))
    jitter_window_s = min(tick_duration * jitter_fraction, target_interval_s * jitter_fraction)

    for snap in per_network.values():
        allowed = max(0.0, _safe_float(snap.get("allowed_submit_rate", 0.0), 0.0))
        interval_s = (1.0 / allowed) if allowed > 1e-9 else 0.0
        snap["tick_duration_s"] = float(tick_duration)
        snap["jitter_window_s"] = float(
            min(jitter_window_s, interval_s * jitter_fraction) if interval_s > 0.0 else 0.0
        )
        for worker in list(snap.get("workers", []) or []):
            worker_allowed = max(0.0, _safe_float(worker.get("allowed_submit_rate", 0.0), 0.0))
            worker_interval_s = (1.0 / worker_allowed) if worker_allowed > 1.0e-9 else 0.0
            worker["tick_duration_s"] = float(tick_duration)
            worker["jitter_window_s"] = float(
                min(jitter_window_s, worker_interval_s * jitter_fraction) if worker_interval_s > 0.0 else 0.0
            )

    return {
        "ts": time.time(),
        "model": "network_fraction_exact_band_v2",
        "allowed_rate_per_second": float(total_allowed),
        "tick_duration": float(tick_duration),
        "jitter_window_s": float(jitter_window_s),
        "target_acceptance_rate": float(target_acceptance),
        "per_network": per_network,
    }

from bios.event_bus import get_event_bus

from miner.stratum_client import StratumClient

# Unified schema + packet types
from neural_object import (
    neural_objectPacket,
    variable_format_enum,
    ComputeNetwork,
    neural_objectSchema,
    network_to_packet_type,
    name_to_network,
)

# NetSpec registry (normalize raw jobs per coin)
from miner.pool_clients import get_netspec

# Optional coin-specific pool clients for Submitter resolver
from miner.pool_clients.btc_client import BTCPoolClient
from miner.pool_clients.ltc_client import LTCPoolClient
from miner.pool_clients.etc_client import ETCPoolClient
from miner.pool_clients.rvn_client import RVNPoolClient


def _coin_adapter_for(network: str):
    coin_u = str(network).upper()
    if coin_u == "BTC":
        from miner.stratum_adapters import BTCAdapter
        return BTCAdapter()
    if coin_u == "LTC":
        from miner.stratum_adapters import LTCAdapter
        return LTCAdapter()
    if coin_u == "ETC":
        from miner.stratum_adapters import ETCAdapter
        return ETCAdapter()
    if coin_u == "RVN":
        from miner.stratum_adapters import RVNAdapter
        return RVNAdapter()
    return None

# ---------------------------------------------------------------------------
# Stratum Adapter
# ---------------------------------------------------------------------------
class StratumAdapter:
    def __init__(self,
                 engine: Any,
                 vsd: Any,
                 submitter: Any,
                 config_path: str = "miner/miner_runtime_config.json",
                 bus: Optional[Any] = None) -> None:
        self.engine = engine
        self.vsd = vsd
        self.submitter = submitter
        self.config_path = config_path
        self.bus = bus or get_event_bus()

        self.cfg: Dict[str, Any] = {}
        self.clients: Dict[str, Any] = {}
        self.coin_adapters: Dict[str, Any] = {}
        # schema-driven; no NetSpec handlers
        self._lock = threading.RLock()
        self._lane_map: Dict[str, str] = {}  # lane_id -> COIN
        self._lane_worker_index: Dict[str, int] = {}  # lane_id -> per-coin worker number
        self._lane_session_id: Dict[str, str] = {}  # lane_id -> worker session id
        self._worker_map: Dict[str, Dict[str, Any]] = {}
        self._last_job_packet: Dict[str, Any] = {}  # lane_id -> neural_objectPacket
        self._submit_clients: Dict[tuple[str, str, str], Any] = {}
        self._control_client_by_coin: Dict[str, Any] = {}
        self._stop = False
        self._thr: Optional[threading.Thread] = None
        self.connected_flags: Dict[str, bool] = {}
        self._submission_policy: Dict[str, float] = dict(DEFAULT_SUBMISSION_POLICY)
        self._pool_stats = PoolStats(ttl_s=int(self._submission_policy["difficulty_request_interval_s"]))
        self._last_target_refresh_ts = 0.0
        self._last_requested_difficulty: Dict[str, float] = {}
        self._last_requested_ts: Dict[str, float] = {}
        self._failure_counts: Dict[str, int] = {}  # COIN -> consecutive failures
        self._disabled_until: Dict[str, float] = {}  # COIN -> timestamp when to re-enable
        self._paused = False
        self._pause_note = ""

        # Prepare Submitter resolver
        try:
            self.submitter.client_resolver = self._client_resolver
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assign_lanes(self):
        """Public method to (re-)assign lanes after engine startup."""
        self._assign_lanes_round_robin()

    def start(self) -> None:
        if self._thr:
            return
        self._load_config()
        self._assign_lanes_round_robin()
        self._wire_bus()
        self._init_clients()
        self._thr = threading.Thread(target=self._sync_loop, daemon=True, name="stratum_adapter")
        self._thr.start()
        _logger.info("StratumAdapter started for worker sessions: %s", sorted(self.clients.keys()))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop = True
        try:
            for c in list(self.clients.values()):
                try:
                    c.stop()
                except Exception:
                    pass
        except Exception:
            pass
        if self._thr:
            try:
                self._thr.join(timeout=timeout)
            except Exception:
                pass
        self._thr = None

    def is_paused(self) -> bool:
        with self._lock:
            return bool(self._paused)

    def pause(self, note: str = "", source: str = "control_center") -> None:
        with self._lock:
            self._paused = True
            self._pause_note = str(note or "")
        try:
            self.vsd.store("miner/control/stratum_adapter", {
                "ts": time.time(),
                "paused": True,
                "note": self._pause_note,
                "source": str(source or "control_center"),
            })
        except Exception:
            pass
        _logger.info("StratumAdapter paused: %s", self._pause_note or "pause requested")

    def resume(self, note: str = "", source: str = "control_center") -> None:
        with self._lock:
            self._paused = False
            self._pause_note = str(note or "")
        try:
            self.vsd.store("miner/control/stratum_adapter", {
                "ts": time.time(),
                "paused": False,
                "note": self._pause_note,
                "source": str(source or "control_center"),
            })
        except Exception:
            pass
        _logger.info("StratumAdapter resumed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        try:
            with open(self.config_path, "r") as f:
                self.cfg = json.load(f)
            self._submission_policy = _submission_policy(self.cfg)
            ttl_s = int(max(10.0, self._submission_policy.get("difficulty_request_interval_s", 60.0)))
            self._pool_stats = PoolStats(ttl_s=ttl_s)
            _logger.info("Loaded miner config from %s", self.config_path)
        except Exception as exc:
            _logger.error("Failed to load %s: %s", self.config_path, exc)
            self.cfg = {"coins": {}}
            self._submission_policy = dict(DEFAULT_SUBMISSION_POLICY)
            self._pool_stats = PoolStats(ttl_s=int(self._submission_policy["difficulty_request_interval_s"]))

    def _init_clients(self) -> None:
        for lane_id, worker in sorted(dict(self._worker_map or {}).items()):
            coin_u = str(dict(worker or {}).get("coin", "")).upper()
            worker_index = int(dict(worker or {}).get("worker_index", 1) or 1)
            session_id = str(dict(worker or {}).get("session_id", "") or self._lane_session_id.get(str(lane_id), ""))
            if not coin_u or not session_id:
                continue
            now = time.time()
            if session_id in self._disabled_until and now < self._disabled_until[session_id]:
                _logger.warning("Session %s disabled until %s due to repeated failures", session_id, time.ctime(self._disabled_until[session_id]))
                continue
            try:
                cli = StratumClient.from_json(
                    coin_u,
                    self.cfg,
                    self.bus,
                    worker_index=worker_index,
                    lane_id=str(lane_id),
                    session_id=session_id,
                )
                cli.start()
                self.clients[session_id] = cli
                adapter = _coin_adapter_for(coin_u)
                if adapter:
                    self.coin_adapters[session_id] = adapter
                if coin_u not in self._control_client_by_coin or worker_index == 1:
                    self._control_client_by_coin[coin_u] = cli
                self.connected_flags[session_id] = False
            except Exception as exc:
                _logger.error("Failed to start StratumClient for %s lane=%s: %s", coin_u, lane_id, exc)

    def _wire_bus(self) -> None:
        try:
            self.bus.subscribe("stratum.job", self._on_stratum_job)
            self.bus.subscribe("stratum.difficulty", self._on_difficulty)
            self.bus.subscribe("stratum.response", self._on_response)
            self.bus.subscribe("stratum.error", self._on_stratum_error)
            # connection lifecycle events (for guarded ERG print)
            self.bus.subscribe("stratum.connected", self._on_connected)
            self.bus.subscribe("stratum.disconnected", self._on_disconnected)
        except Exception as exc:
            _logger.error("EventBus subscribe failed: %s", exc)

    def _assign_lanes_round_robin(self) -> None:
        try:
            lanes = list(getattr(self.engine.cm, "lanes", {}).keys())
            coins = sorted(configured_coins(dict(self.cfg.get("coins", {}))))
            if not lanes or not coins:
                return
            i = 0
            per_coin_counts: Dict[str, int] = {}
            worker_map: Dict[str, Dict[str, Any]] = {}
            self._lane_map = {}
            self._lane_worker_index = {}
            self._lane_session_id = {}
            for lid in lanes:
                coin = coins[i % len(coins)]
                self._lane_map[lid] = coin
                per_coin_counts[coin] = per_coin_counts.get(coin, 0) + 1
                self._lane_worker_index[lid] = per_coin_counts[coin]
                try:
                    profile = resolve_coin_profile(self.cfg.get("coins", {}), coin, self._lane_worker_index[lid])
                    session_id = "%s:%s:%s" % (coin, str(lid), profile.username)
                    self._lane_session_id[lid] = session_id
                    worker_map[lid] = {
                        "coin": coin,
                        "worker_index": self._lane_worker_index[lid],
                        "username": profile.username,
                        "session_id": session_id,
                    }
                except Exception:
                    session_id = "%s:%s:%s" % (coin, str(lid), self._lane_worker_index[lid])
                    self._lane_session_id[lid] = session_id
                    worker_map[lid] = {
                        "coin": coin,
                        "worker_index": self._lane_worker_index[lid],
                        "username": "",
                        "session_id": session_id,
                    }
                i += 1
            # Store mapping in VSD for visibility
            self._worker_map = dict(worker_map)
            self.vsd.store("miner/lanes/coin_map", dict(self._lane_map))
            self.vsd.store("miner/lanes/worker_map", worker_map)
            self.vsd.store("miner/lanes/session_map", dict(self._lane_session_id))
            _logger.info("Lane->coin mapping: %s", self._lane_map)
        except Exception as exc:
            _logger.error("Lane assignment failed: %s", exc)

    def _lane_ids_for_coin(self, coin: str) -> list[str]:
        coin_u = str(coin).upper()
        return [str(lane_id) for lane_id, lane_coin in sorted(self._lane_map.items()) if str(lane_coin).upper() == coin_u]

    def _difficulty_key(self, coin: str, lane_id: str = "") -> str:
        coin_u = str(coin).upper()
        if lane_id:
            return "miner/difficulty/%s/workers/%s" % (coin_u, str(lane_id))
        return "miner/difficulty/%s" % coin_u

    def _runtime_pulse_state(self, coin: str, lane_id: str = "") -> Dict[str, Any]:
        coin_u = str(coin).upper()
        try:
            metrics = dict(self.vsd.get("telemetry/metrics/%s/current" % coin_u, {}) or {})
        except Exception:
            metrics = {}
        share_difficulty = self._current_share_difficulty(coin_u, lane_id=lane_id)
        share_target = self._current_share_target(coin_u, lane_id=lane_id)
        out: Dict[str, Any] = {}
        for key in (
            "network_hashrate_hs",
            "target_hashrate_hs",
            "allowed_submit_rate_per_second",
            "control_target_fraction",
            "ratio_progress",
            "share_rate_taper_fraction",
            "observed_submit_rate_per_second",
            "observed_valid_share_rate_per_second",
        ):
            if key in metrics:
                out[key] = metrics.get(key)
        if share_difficulty > 0.0:
            out["current_share_difficulty"] = float(share_difficulty)
        if share_target:
            out["share_target"] = share_target
        return out

    def _current_share_target(self, coin: str, lane_id: str = "") -> str:
        coin_u = str(coin).upper()
        share_target = ""
        if lane_id:
            share_target = _share_target_hex_from_record(dict(self.vsd.get(self._difficulty_key(coin_u, lane_id), {}) or {}))
        if not share_target:
            share_target = _share_target_hex_from_vsd(self.vsd, coin_u)
        if share_target:
            return share_target
        packet = None
        if lane_id:
            try:
                packet = self._last_job_packet.get(str(lane_id))
            except Exception:
                packet = None
        if packet is None:
            for coin_lane_id in self._lane_ids_for_coin(coin_u):
                try:
                    packet = self._last_job_packet.get(coin_lane_id)
                except Exception:
                    packet = None
                if packet is not None:
                    break
        if packet is None:
            return ""
        raw = dict(packet.raw_payload or {})
        return _normalize_target_hex(raw.get("share_target", raw.get("active_target", "")))

    def _current_share_difficulty(self, coin: str, lane_id: str = "") -> float:
        coin_u = str(coin).upper()
        if lane_id:
            share_difficulty = _share_difficulty_from_record(dict(self.vsd.get(self._difficulty_key(coin_u, lane_id), {}) or {}))
            if share_difficulty > 0.0:
                return share_difficulty
        share_difficulty = _share_difficulty_from_vsd(self.vsd, coin_u)
        if share_difficulty > 0.0:
            return share_difficulty
        if lane_id:
            try:
                packet = self._last_job_packet.get(str(lane_id))
            except Exception:
                packet = None
            if packet is not None:
                share_difficulty = _share_difficulty_from_job(dict(packet.raw_payload or {}))
                if share_difficulty > 0.0:
                    return share_difficulty
        difficulties: list[float] = []
        for coin_lane_id in self._lane_ids_for_coin(coin_u):
            try:
                packet = self._last_job_packet.get(coin_lane_id)
            except Exception:
                packet = None
            if packet is None:
                continue
            lane_difficulty = _share_difficulty_from_job(dict(packet.raw_payload or {}))
            if lane_difficulty > 0.0:
                difficulties.append(lane_difficulty)
        if difficulties:
            return float(sum(difficulties) / float(len(difficulties)))
        return 0.0

    def _apply_runtime_state_to_job(self, coin: str, job_payload: Dict[str, Any], lane_id: str = "") -> tuple[Dict[str, Any], Dict[str, Any]]:
        job = dict(job_payload or {})
        runtime_state = self._runtime_pulse_state(coin, lane_id=lane_id)
        raw_target = _normalize_target_hex(job.get("target", ""))
        if raw_target:
            job["target"] = raw_target
        share_target = _normalize_target_hex(runtime_state.get("share_target", job.get("share_target", "")))
        if not share_target and str(coin).upper() == "ETC":
            share_target = raw_target
        if share_target:
            if raw_target and raw_target != share_target and "block_target" not in job:
                job["block_target"] = raw_target
            job["share_target"] = share_target
            job["active_target"] = share_target
        elif raw_target:
            job["active_target"] = raw_target
        current_share_difficulty = _safe_float(runtime_state.get("current_share_difficulty", 0.0), 0.0)
        if current_share_difficulty <= 0.0:
            current_share_difficulty = _share_difficulty_from_job(job)
            if current_share_difficulty > 0.0:
                runtime_state["current_share_difficulty"] = float(current_share_difficulty)
        if current_share_difficulty > 0.0:
            job["current_share_difficulty"] = float(current_share_difficulty)
        for key, value in runtime_state.items():
            if key not in job:
                job[key] = value
        return job, runtime_state

    def _publish_jobs_map(self) -> None:
        try:
            telem = self.vsd.get("telemetry/global", {})
            if not isinstance(telem, dict):
                telem = {}
            jobs_map = {}
            for lane_id, lane_coin in self._lane_map.items():
                packet = self._last_job_packet.get(str(lane_id))
                if packet is not None:
                    jobs_map[str(lane_id)] = packet
            telem["jobs_map"] = jobs_map
            self.vsd.store("telemetry/global", telem)
            _logger.info("Updated telemetry/global with jobs_map: %s", list(jobs_map.keys()))
        except Exception as exc:
            _logger.error("Telemetry update failed: %s", exc)

    def _refresh_packet_runtime(self, coin: str, lane_id: str = "") -> None:
        coin_u = str(coin).upper()
        lane_ids = [str(lane_id)] if lane_id else self._lane_ids_for_coin(coin_u)
        refreshed_any = False
        for lane_key in lane_ids:
            try:
                packet = self._last_job_packet.get(lane_key)
            except Exception:
                packet = None
            if packet is None:
                continue
            try:
                raw_payload, runtime_state = self._apply_runtime_state_to_job(coin_u, dict(packet.raw_payload or {}), lane_id=lane_key)
                system_payload = dict(packet.system_payload or {})
                system_payload.update(runtime_state)
                system_payload["job"] = dict(raw_payload)
                system_payload["lane"] = lane_key
                refreshed = neural_objectPacket(
                    packet_type=packet.packet_type,
                    network=packet.network,
                    raw_payload=raw_payload,
                    system_payload=system_payload,
                    metadata=dict(packet.metadata or {}),
                    derived_state=dict(packet.derived_state or {}),
                )
                with self._lock:
                    self._last_job_packet[lane_key] = refreshed
                refreshed_any = True
            except Exception as exc:
                _logger.error("Packet runtime refresh failed coin=%s lane=%s err=%s", coin_u, lane_key, exc)
        if refreshed_any:
            self._publish_jobs_map()

    def _client_for_lane(self, lane_id: str) -> Any:
        session_id = str(self._lane_session_id.get(str(lane_id), ""))
        if not session_id:
            return None
        return self.clients.get(session_id)

    def _store_worker_runtime_records(self, runtime_state: Dict[str, Any], targets: Dict[str, Dict[str, Any]]) -> None:
        now = time.time()
        per_network = dict(runtime_state.get("per_network", {})) if isinstance(runtime_state, dict) else {}
        for coin_u, network_state in per_network.items():
            net_state = dict(network_state or {})
            target = dict(targets.get(str(coin_u).upper(), {}) or {})
            workers = list(net_state.get("workers", []) or [])
            for worker in workers:
                worker_state = dict(worker or {})
                lane_id = str(worker_state.get("lane_id", "") or "")
                if not lane_id:
                    continue
                sr_key = "miner/runtime/submission_rate/workers/%s" % lane_id
                existing = dict(self.vsd.get(sr_key, {}) or {})
                boosted_until = _safe_float(existing.get("boosted_until", 0.0), 0.0)
                allowed_rate = _safe_float(worker_state.get("allowed_submit_rate", existing.get("allowed_rate_per_second", 0.0)), 0.0)
                if boosted_until > now:
                    allowed_rate = max(allowed_rate, _safe_float(existing.get("allowed_rate_per_second", 0.0), 0.0))
                else:
                    existing.pop("boosted_until", None)
                runtime_record = {
                    "coin": str(coin_u).upper(),
                    "lane_id": lane_id,
                    "worker_index": int(worker_state.get("worker_index", self._lane_worker_index.get(lane_id, 1)) or 1),
                    "username": str(worker_state.get("username", dict(self._worker_map.get(lane_id, {}) or {}).get("username", ""))),
                    "session_id": str(self._lane_session_id.get(lane_id, "")),
                    "allowed_rate_per_second": float(allowed_rate),
                    "tick_duration": float(worker_state.get("tick_duration_s", runtime_state.get("tick_duration", 0.25)) or runtime_state.get("tick_duration", 0.25)),
                    "jitter_window_s": float(worker_state.get("jitter_window_s", runtime_state.get("jitter_window_s", 0.0)) or runtime_state.get("jitter_window_s", 0.0)),
                    "target_hashrate_hs": float(worker_state.get("target_hashrate_hs", 0.0) or 0.0),
                    "target_valid_share_rate_per_second": float(worker_state.get("target_valid_share_rate_per_second", 0.0) or 0.0),
                    "share_spacing_s": float(worker_state.get("share_spacing_s", 0.0) or 0.0),
                    "target_share_difficulty": float(target.get("target_share_difficulty", 0.0) or 0.0),
                    "assigned_share_difficulty": float(self._current_share_difficulty(str(coin_u), lane_id=lane_id)),
                    "share_target": str(self._current_share_target(str(coin_u), lane_id=lane_id)),
                    "desired_submit_rate_per_second": float(target.get("desired_submit_rate_per_second", 0.0) or 0.0),
                    "guarded_submit_rate_ceiling_per_second": float(target.get("guarded_submit_rate_ceiling_per_second", 0.0) or 0.0),
                }
                if boosted_until > now:
                    runtime_record["boosted_until"] = float(boosted_until)
                self.vsd.store(sr_key, dict(existing | runtime_record))

    def _store_worker_session_status(self, lane_id: str, payload: Dict[str, Any]) -> None:
        lane_key = str(lane_id or "")
        if not lane_key:
            return
        key = "miner/stratum/workers/%s/session" % lane_key
        state = dict(self.vsd.get(key, {}) or {})
        state.update(dict(payload or {}))
        self.vsd.store(key, state)

    def _on_stratum_job(self, payload: Dict[str, Any]) -> None:
        try:
            if self.is_paused():
                try:
                    self.vsd.store("miner/control/stratum_adapter/last_dropped_job", {
                        "ts": time.time(),
                        "reason": "paused",
                        "coin": str(payload.get("coin", "")).upper(),
                    })
                except Exception:
                    pass
                return
            coin = str(payload.get("coin", "")).upper()
            session_id = str(payload.get("session_id", "") or "")
            lane_id = str(payload.get("lane_id", "") or "")
            if not lane_id and session_id:
                for mapped_lane_id, mapped_session_id in self._lane_session_id.items():
                    if str(mapped_session_id) == session_id:
                        lane_id = str(mapped_lane_id)
                        break
            if not lane_id:
                lane_ids = self._lane_ids_for_coin(coin)
                lane_id = str(lane_ids[0]) if lane_ids else ""
            if not lane_id:
                _logger.error("Job handling failed: missing lane for coin=%s session=%s", coin, session_id)
                return
            job_raw = payload.get("job", {})
            # 1) Normalize incoming raw job via stateful coin adapter (preferred) or fallback to NetSpec
            job_norm = None
            adapter = self.coin_adapters.get(session_id or self._lane_session_id.get(lane_id, ""))
            if adapter:
                # Use stateful adapter for live job normalization
                try:
                    job_norm = adapter.convert_job(job_raw)
                except Exception as exc:
                    _logger.error("Stateful adapter normalization failed coin=%s lane=%s err=%s", coin, lane_id, exc)
            if job_norm is None:
                # Fallback to stateless NetSpec if adapter unavailable
                if get_netspec is None:
                    _logger.error("NetSpec registry unavailable; dropping job coin=%s", coin)
                    return
                try:
                    netspec = get_netspec(coin)
                    job_norm = netspec.network_to_system_fn(job_raw)
                except Exception as exc:
                    _logger.error("NetSpec normalization failed coin=%s err=%s", coin, exc)
                    return

            # 2) Build neural_objectPacket representation
            try:
                pkt_type = network_to_packet_type(coin)
                net_enum = name_to_network(coin)
            except Exception:
                pkt_type = None
                net_enum = None

            # Build packet only; drop job if schema construction fails
            try:
                # Ensure dict form for packet payloads
                if isinstance(job_norm, QMJob):
                    job_norm_dict = _asdict(job_norm)
                else:
                    job_norm_dict = dict(job_norm or {})
                job_norm_dict, runtime_state = self._apply_runtime_state_to_job(coin, job_norm_dict, lane_id=lane_id)
                worker_meta = dict(self._worker_map.get(lane_id, {}) or {})
                packet_job = neural_objectPacket(
                    packet_type=pkt_type,  # type: ignore[arg-type]
                    network=net_enum,      # type: ignore[arg-type]
                    raw_payload=job_norm_dict,
                    system_payload=dict({
                        "job": job_norm_dict,
                        "lane": lane_id,
                        "worker_index": int(worker_meta.get("worker_index", 1) or 1),
                        "username": str(worker_meta.get("username", payload.get("user", ""))),
                        "session_id": str(session_id or self._lane_session_id.get(lane_id, "")),
                    } | runtime_state),
                    metadata={},
                    derived_state={},
                )
            except Exception as exc:
                _logger.error("Packet construction failed coin=%s err=%s", coin, exc)
                return

            # 3) Store BOTH packet and native
            with self._lock:
                self._last_job_packet[lane_id] = packet_job

            self._store_worker_session_status(lane_id, {
                "ts": time.time(),
                "coin": coin,
                "lane_id": lane_id,
                "session_id": str(session_id or self._lane_session_id.get(lane_id, "")),
                "username": str(worker_meta.get("username", payload.get("user", ""))),
                "last_job_id": str(job_norm_dict.get("job_id", "")),
                "share_target": str(job_norm_dict.get("share_target", job_norm_dict.get("active_target", ""))),
                "assigned_share_difficulty": float(job_norm_dict.get("current_share_difficulty", 0.0) or 0.0),
            })

            _logger.info("Job packet stored for coin=%s lane=%s", coin, lane_id)

            # 4) Update telemetry/global with jobs_map for miner engine
            self._publish_jobs_map()
        except Exception as exc:
            _logger.error("Job handling failed: %s", exc)

    def _on_difficulty(self, payload: Dict[str, Any]) -> None:
        try:
            coin = str(payload.get("coin", "")).upper()
            lane_id = str(payload.get("lane_id", "") or "")
            method = str(payload.get("method", ""))
            params = payload.get("params", [])
            record = {
                "method": method,
                "params": params,
                "ts": time.time()
            }
            self.vsd.store(self._difficulty_key(coin), record)
            if lane_id:
                self.vsd.store(self._difficulty_key(coin, lane_id), record)
                self._store_worker_session_status(lane_id, {
                    "ts": record["ts"],
                    "coin": coin,
                    "lane_id": lane_id,
                    "last_difficulty_method": method,
                    "assigned_share_difficulty": float(self._current_share_difficulty(coin, lane_id=lane_id)),
                    "share_target": str(self._current_share_target(coin, lane_id=lane_id)),
                })
            self._update_submission_rate_for_difficulty(coin, lane_id=lane_id)
            self._refresh_packet_runtime(coin, lane_id=lane_id)
        except Exception as exc:
            _logger.error("Difficulty handling failed: %s", exc)

    def _update_submission_rate_for_difficulty(self, coin: str, lane_id: str = "") -> None:
        try:
            coin_u = str(coin).upper()
            lane_ids = [str(lane_id)] if lane_id else self._lane_ids_for_coin(coin_u)
            if not lane_ids:
                lane_ids = [""]
            # Get network hashrate
            network_hashrate_hs = 0.0
            try:
                market_key = "miner/market/%s" % coin_u
                market = dict(self.vsd.get(market_key, {}) or {})
                network_hashrate_hs = float(market.get("network_hashrate_hs", 0.0))
            except Exception:
                pass
            if network_hashrate_hs <= 0.0:
                return
            # Simulated hashrate = network * target_fraction
            target_fraction = float(self._submission_policy.get("network_target_fraction", 0.05))
            simulated_hashrate_hs = network_hashrate_hs * target_fraction
            num_workers = max(1, len(self._lane_ids_for_coin(coin_u)))
            if num_workers <= 0:
                num_workers = 1
            # Per worker simulated hashrate
            per_worker_hashrate_hs = simulated_hashrate_hs / num_workers
            # Hashes per diff1
            coin_cfg = coin_config_entry(self.cfg.get("coins", {}), coin_u)
            hashes_per_diff1 = _hashes_per_diff1(coin_cfg)
            max_rate = _submit_rate_ceiling_per_second(dict(coin_cfg or {}), {}, self._submission_policy)
            tick_duration = float(self._submission_policy.get("min_tick_duration_s", 0.05))
            total_allowed = 0.0
            for worker_lane_id in lane_ids:
                difficulty = self._current_share_difficulty(coin_u, lane_id=worker_lane_id)
                if difficulty <= 0.0:
                    continue
                allowed_rate = min(max_rate, per_worker_hashrate_hs / (difficulty * hashes_per_diff1))
                total_allowed += allowed_rate
                worker_sr_key = "miner/runtime/submission_rate/workers/%s" % str(worker_lane_id)
                worker_sr = dict(self.vsd.get(worker_sr_key, {}) or {})
                worker_sr.update({
                    "coin": coin_u,
                    "lane_id": str(worker_lane_id),
                    "allowed_rate_per_second": float(allowed_rate),
                    "tick_duration": tick_duration,
                    "worker_index": int(self._lane_worker_index.get(str(worker_lane_id), 1) or 1),
                    "username": str(dict(self._worker_map.get(str(worker_lane_id), {}) or {}).get("username", "")),
                })
                self.vsd.store(worker_sr_key, worker_sr)
            sr_key = "miner/runtime/submission_rate/%s" % coin_u
            sr = dict(self.vsd.get(sr_key, {}) or {})
            sr["allowed_rate_per_second"] = float(total_allowed)
            sr["tick_duration"] = tick_duration
            sr["worker_count"] = int(num_workers)
            self.vsd.store(sr_key, sr)
        except Exception as exc:
            _logger.error("Submission rate update failed coin=%s err=%s", coin, exc)

    def _on_response(self, payload: Dict[str, Any]) -> None:
        try:
            coin = str(payload.get("coin", "")).upper()
            session_id = str(payload.get("session_id", "") or "")
            id_val = payload.get("id")
            result = payload.get("result")
            error = payload.get("error")
            _logger.info("Stratum response coin=%s id=%s result=%s error=%s", coin, id_val, result, error)
            adapter = self.coin_adapters.get(session_id)
            if adapter and isinstance(payload, dict):
                adapter.on_response(payload)
        except Exception as exc:
            _logger.error("Response handling failed: %s", exc)

    def _on_connected(self, payload: Dict[str, Any]) -> None:
        try:
            coin = str(payload.get("coin", "")).upper()
            session_id = str(payload.get("session_id", "") or coin)
            lane_id = str(payload.get("lane_id", "") or "")
            self._failure_counts[session_id] = 0
            self._store_worker_session_status(lane_id, {
                "ts": time.time(),
                "coin": coin,
                "lane_id": lane_id,
                "session_id": session_id,
                "username": str(payload.get("user", "")),
                "connected": True,
                "host": str(payload.get("host", "")),
                "port": int(payload.get("port", 0) or 0),
            })
            if coin != "ERG":
                return
            host = str(payload.get("host", ""))
            port = int(payload.get("port", 0))
            cli = self.clients.get(session_id)
            wallet = getattr(cli, "user", "worker") if cli else "worker"
            # Guard: print once per TCP session
            if not self.connected_flags.get(session_id, False):
                print(f"[Stratum] Connected to {host}:{port} as {wallet}")
                self.connected_flags[session_id] = True
        except Exception:
            pass

    def _on_disconnected(self, payload: Dict[str, Any]) -> None:
        try:
            coin = str(payload.get("coin", "")).upper()
            session_id = str(payload.get("session_id", "") or coin)
            lane_id = str(payload.get("lane_id", "") or "")
            self._store_worker_session_status(lane_id, {
                "ts": time.time(),
                "coin": coin,
                "lane_id": lane_id,
                "session_id": session_id,
                "connected": False,
            })
            if coin == "ERG":
                # Reset guard so next TCP connect prints again
                self.connected_flags[session_id] = False
        except Exception:
            pass

    def _on_stratum_error(self, payload: Dict[str, Any]) -> None:
        try:
            coin = str(payload.get("coin", "")).upper()
            session_id = str(payload.get("session_id", "") or coin)
            self._failure_counts[session_id] = self._failure_counts.get(session_id, 0) + 1
            if self._failure_counts[session_id] >= 5:
                # Disable for 5 minutes
                self._disabled_until[session_id] = time.time() + 300.0
                _logger.warning("Disabling session %s for 5 minutes due to repeated failures", session_id)
                cli = self.clients.get(session_id)
                if cli:
                    cli.stop()
                    del self.clients[session_id]
                self.coin_adapters.pop(session_id, None)
                self.connected_flags.pop(session_id, None)
                if self._control_client_by_coin.get(coin) is cli:
                    self._control_client_by_coin.pop(coin, None)
        except Exception as exc:
            _logger.error("Error handling stratum error: %s", exc)

    def _sync_loop(self) -> None:
        # Periodically sync jobs_map into the global telemetry so MinerEngine picks it up
        while not self._stop:
            try:
                if self.is_paused():
                    self.bus.publish("miner.stratum_adapter.heartbeat", {
                        "ts": time.time(),
                        "lane_map_size": len(self._lane_map),
                        "coins": sorted(list(self._last_job_packet.keys())),
                        "paused": True,
                    })
                    time.sleep(0.5)
                    continue
                jobs_map: Dict[str, Any] = {}
                with self._lock:
                    for lid, coin in self._lane_map.items():
                        pkt = self._last_job_packet.get(str(lid))
                        if pkt:
                            jobs_map[str(lid)] = pkt
                # Merge into telemetry/global frame for BIOS monitor; packet-only
                snap = self.vsd.get("telemetry/global", {})
                if not isinstance(snap, dict):
                    snap = {}
                snap["jobs_map"] = jobs_map
                self.vsd.store("telemetry/global", snap)
                # Emit an adapter heartbeat
                self.bus.publish("miner.stratum_adapter.heartbeat", {
                    "ts": time.time(),
                    "lane_map_size": len(self._lane_map),
                    "coins": sorted(list({str(coin).upper() for coin in self._lane_map.values()})),
                    "paused": False,
                })
                now = time.time()
                if now - self._last_target_refresh_ts >= 5.0:
                    self._refresh_network_targets()
                    self._last_target_refresh_ts = now
            except Exception:
                # keep adapter resilient
                pass
            time.sleep(0.5)

    # (legacy _convert_job removed; normalization handled via NetSpec)

    # ------------------------------------------------------------------
    # Submitter resolver (returns coin-specific pool client)
    # ------------------------------------------------------------------
    def _client_resolver(self, network: str, lane_id: str = ""):
        coin = str(network).upper()
        worker_index = self._lane_worker_index.get(str(lane_id), 1)
        try:
            profile = resolve_coin_profile(self.cfg.get("coins", {}), coin, worker_index=worker_index)
        except Exception:
            return None
        cache_key = (profile.coin.value, profile.endpoint, profile.username)
        with self._lock:
            client = self._submit_clients.get(cache_key)
            if client is not None:
                return client
        if not profile.endpoint:
            return None
        try:
            if profile.coin == SupportedCoin.BTC and BTCPoolClient:
                client = BTCPoolClient(profile.endpoint, username=profile.username, password=profile.password)
            elif profile.coin == SupportedCoin.LTC and LTCPoolClient:
                client = LTCPoolClient(profile.endpoint, username=profile.username, password=profile.password)
            elif profile.coin == SupportedCoin.ETC and ETCPoolClient:
                client = ETCPoolClient(profile.endpoint)
            elif profile.coin == SupportedCoin.RVN and RVNPoolClient:
                client = RVNPoolClient(profile.endpoint, username=profile.username, password=profile.password)
            else:
                client = None
        except Exception:
            client = None
        if client is not None:
            with self._lock:
                self._submit_clients[cache_key] = client
        return client

    def _refresh_network_targets(self) -> None:
        coins = dict(self.cfg.get("coins", {}))
        if not coins:
            return
        try:
            stats = self._pool_stats.get_all(coins)
        except Exception as exc:
            _logger.error("PoolStats refresh failed: %s", exc)
            return

        targets: Dict[str, Dict[str, Any]] = {}
        for coin_u in configured_coins(coins):
            coin_cfg = coin_config_entry(coins, coin_u)
            snap = dict(stats.get(coin_u, {}))
            if not snap:
                continue
            metrics_key = "telemetry/metrics/%s/current" % coin_u
            try:
                metrics = dict(self.vsd.get(metrics_key, {}) or {})
            except Exception:
                metrics = {}
            current_share_difficulty = self._current_share_difficulty(coin_u)
            target = _build_network_target_snapshot(
                network=coin_u,
                coin_cfg=dict(coin_cfg or {}),
                network_stats=snap,
                policy=self._submission_policy,
                metrics=metrics,
                current_share_difficulty=current_share_difficulty,
            )
            self._store_network_target(target)
            self._maybe_request_share_difficulty(coin_u, target)
            targets[coin_u] = target

        tick_dur = float(self.cfg.get("tick_interval_s", 0.25))
        runtime_state = _build_runtime_submission_state(targets, self._submission_policy, tick_dur, worker_map=self._worker_map)
        try:
            self.vsd.store("miner/runtime/submission_rate", runtime_state)
        except Exception:
            pass
        try:
            self._store_worker_runtime_records(runtime_state, targets)
        except Exception as exc:
            _logger.error("Worker runtime state update failed: %s", exc)

    def _store_network_target(self, target: Dict[str, Any]) -> None:
        coin = str(target.get("network", "")).upper()
        if not coin:
            return
        try:
            self.vsd.store("miner/network_stats/%s" % coin, dict(target))
        except Exception:
            pass
        try:
            metrics_key = "telemetry/metrics/%s/current" % coin
            metrics = dict(self.vsd.get(metrics_key, {}) or {})
            metrics["network_hashrate_hs"] = float(target.get("network_hashrate_hs", 0.0))
            metrics["block_time_s"] = float(target.get("block_time_s", 0.0))
            metrics["network_difficulty"] = float(target.get("network_difficulty", 0.0))
            metrics["target_hashrate_hs"] = float(target.get("target_hashrate_hs", 0.0))
            metrics["target_hashrate_floor_hs"] = float(target.get("target_hashrate_floor_hs", 0.0))
            metrics["target_hashrate_ceiling_hs"] = float(target.get("target_hashrate_ceiling_hs", 0.0))
            metrics["target_share_difficulty"] = float(target.get("target_share_difficulty", 0.0))
            metrics["target_valid_share_rate_per_second"] = float(target.get("target_valid_share_rate_per_second", 0.0))
            metrics["allowed_submit_rate_per_second"] = float(target.get("allowed_submit_rate_per_second", 0.0))
            metrics["observed_target_fraction"] = float(target.get("observed_target_fraction", 0.0))
            metrics["control_target_fraction"] = float(target.get("control_target_fraction", 0.0))
            metrics["target_fraction_floor"] = float(target.get("target_fraction_floor", 0.0))
            metrics["target_fraction_ceiling"] = float(target.get("target_fraction_ceiling", 0.0))
            metrics["observed_submit_rate_per_second"] = float(target.get("observed_submit_rate_per_second", 0.0))
            metrics["observed_valid_share_rate_per_second"] = float(target.get("observed_valid_share_rate_per_second", 0.0))
            metrics["observed_submit_hashrate_hs"] = float(target.get("observed_submit_hashrate_hs", 0.0))
            metrics["observed_valid_hashrate_hs"] = float(target.get("observed_valid_hashrate_hs", 0.0))
            metrics["current_share_difficulty"] = float(target.get("current_share_difficulty", 0.0))
            metrics["assigned_share_difficulty"] = float(target.get("assigned_share_difficulty", 0.0))
            metrics["share_difficulty_gap_ratio"] = float(target.get("share_difficulty_gap_ratio", 0.0))
            metrics["preferred_submit_rate_per_second"] = float(target.get("preferred_submit_rate_per_second", 0.0))
            metrics["desired_submit_rate_per_second"] = float(target.get("desired_submit_rate_per_second", 0.0))
            metrics["assigned_required_valid_share_rate_per_second"] = float(target.get("assigned_required_valid_share_rate_per_second", 0.0))
            metrics["assigned_required_submit_rate_per_second"] = float(target.get("assigned_required_submit_rate_per_second", 0.0))
            metrics["share_rate_taper_fraction"] = float(target.get("share_rate_taper_fraction", 0.0))
            metrics["ratio_progress"] = float(target.get("ratio_progress", 0.0))
            metrics["submit_rate_ceiling_per_second"] = float(target.get("submit_rate_ceiling_per_second", 0.0))
            metrics["guarded_submit_rate_ceiling_per_second"] = float(target.get("guarded_submit_rate_ceiling_per_second", 0.0))
            metrics["control_action"] = str(target.get("control_action", "hold"))
            self.vsd.store(metrics_key, metrics)
        except Exception:
            pass

    def _maybe_request_share_difficulty(self, coin: str, target: Dict[str, Any]) -> None:
        coin_cfg = coin_config_entry(self.cfg.get("coins", {}), coin)
        control_method = str(dict(coin_cfg or {}).get("difficulty_control_method", "suggest_difficulty") or "suggest_difficulty").strip().lower()
        if control_method in ("pool_assigned", "none", "disabled", "passive"):
            return
        coin_u = str(coin).upper()
        target_diff = max(1.0, _safe_float(target.get("target_share_difficulty", 0.0), 0.0))
        if target_diff <= 0.0:
            return

        request_results = []
        lane_ids = self._lane_ids_for_coin(coin_u)
        for lane_id in lane_ids:
            client = self._client_for_lane(lane_id)
            if client is None or not hasattr(client, "suggest_difficulty"):
                continue
            try:
                if hasattr(client, "is_connected") and not bool(client.is_connected()):
                    continue
            except Exception:
                continue

            session_id = str(self._lane_session_id.get(lane_id, lane_id))
            request_key = session_id or lane_id or coin_u
            worker_record = dict(self.vsd.get("miner/runtime/submission_rate/workers/%s" % lane_id, {}) or {})
            lane_metrics = dict(self.vsd.get("telemetry/metrics/%s/shares/lanes/%s" % (coin_u, lane_id), {}) or {})
            now = time.time()
            last_diff = _safe_float(self._last_requested_difficulty.get(request_key, 0.0), 0.0)
            last_ts = _safe_float(self._last_requested_ts.get(request_key, 0.0), 0.0)
            interval_s = max(1.0, _safe_float(self._submission_policy.get("difficulty_request_interval_s", 60.0), 60.0))
            change_fraction = _clamp(self._submission_policy.get("difficulty_request_change_fraction", 0.05), 0.0, 1.0)
            low_rate_fraction = _clamp(self._submission_policy.get("difficulty_request_low_rate_fraction", 0.90), 0.0, 1.0)
            observed_submit_rate = max(0.0, _safe_float(lane_metrics.get("submitted_hs", lane_metrics.get("accepted_hs", 0.0)), 0.0))
            allowed_submit_rate = max(0.0, _safe_float(worker_record.get("allowed_rate_per_second", 0.0), 0.0))
            low_share_pressure = allowed_submit_rate > 0.0 and observed_submit_rate < (allowed_submit_rate * low_rate_fraction)
            changed = (last_diff <= 0.0) or (abs(target_diff - last_diff) / max(1.0, last_diff) >= change_fraction)
            if (now - last_ts) < interval_s and not changed and str(target.get("control_action", "hold")) == "hold" and not low_share_pressure:
                continue

            request_method = "mining.suggest_difficulty"
            suggested_target = _normalize_target_hex(_target_hex_from_difficulty(target_diff))
            if control_method == "suggest_target" and hasattr(client, "suggest_target"):
                ok = bool(client.suggest_target(suggested_target))
                request_method = "mining.suggest_target"
            else:
                ok = bool(client.suggest_difficulty(target_diff))
                if not ok and hasattr(client, "suggest_target"):
                    ok = bool(client.suggest_target(suggested_target))
                    request_method = "mining.suggest_target" if ok else "mining.suggest_difficulty|mining.suggest_target"

            request_record = {
                "ts": now,
                "requested": bool(ok),
                "coin": coin_u,
                "lane_id": str(lane_id),
                "session_id": session_id,
                "username": str(worker_record.get("username", dict(self._worker_map.get(lane_id, {}) or {}).get("username", ""))),
                "target_share_difficulty": float(target_diff),
                "assigned_share_difficulty": float(self._current_share_difficulty(coin_u, lane_id=lane_id)),
                "target_hashrate_hs": float(worker_record.get("target_hashrate_hs", target.get("target_hashrate_hs", 0.0))),
                "network_hashrate_hs": float(target.get("network_hashrate_hs", 0.0)),
                "target_fraction": float(target.get("target_fraction", 0.0)),
                "target_fraction_floor": float(target.get("target_fraction_floor", 0.0)),
                "target_fraction_ceiling": float(target.get("target_fraction_ceiling", 0.0)),
                "observed_submit_rate_per_second": float(observed_submit_rate),
                "allowed_submit_rate_per_second": float(allowed_submit_rate),
                "low_share_pressure": bool(low_share_pressure),
                "control_action": str(target.get("control_action", "hold")),
                "method": request_method,
                "suggest_target": suggested_target,
            }
            try:
                self.vsd.store("miner/difficulty_request/%s/workers/%s" % (coin_u, lane_id), request_record)
            except Exception:
                pass

            if ok and target_diff > last_diff:
                try:
                    sr_key = "miner/runtime/submission_rate/workers/%s" % lane_id
                    sr = dict(self.vsd.get(sr_key, {}) or {})
                    boosted_rate = min(
                        max(_safe_float(sr.get("allowed_rate_per_second", allowed_submit_rate), allowed_submit_rate), allowed_submit_rate) * 2.0,
                        float(_submit_rate_ceiling_per_second(dict(coin_cfg or {}), {}, self._submission_policy)),
                    )
                    sr["allowed_rate_per_second"] = float(boosted_rate)
                    sr["boosted_until"] = now + 60.0
                    self.vsd.store(sr_key, sr)
                    _logger.info("Boosted submission rate for coin=%s lane=%s to %.2f/sec for difficulty request", coin_u, lane_id, boosted_rate)
                except Exception as exc:
                    _logger.error("Rate boost failed coin=%s lane=%s err=%s", coin_u, lane_id, exc)

            if ok:
                self._last_requested_difficulty[request_key] = float(target_diff)
                self._last_requested_ts[request_key] = now
            request_results.append(request_record)

        if request_results:
            try:
                self.vsd.store("miner/difficulty_request/%s" % coin_u, {
                    "ts": max(float(item.get("ts", 0.0)) for item in request_results),
                    "coin": coin_u,
                    "worker_requests": request_results,
                    "requested_workers": int(sum(1 for item in request_results if bool(item.get("requested")))),
                    "worker_count": int(len(request_results)),
                })
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience starter
# ---------------------------------------------------------------------------

def start_stratum_adapter(engine: Any, submitter: Any, vsd: Any, bus: Optional[Any] = None,
                          config_path: str = "miner/miner_runtime_config.json") -> StratumAdapter:
    adapter = StratumAdapter(engine=engine, vsd=vsd, submitter=submitter, config_path=config_path, bus=bus)
    adapter.start()
    return adapter
