# =====================================================================
# Quantum Application / miner
# File: miner_runtime_config.py
# Version: v4.8.2 (Static Python Config)
# ASCII-ONLY SOURCE FILE
# Jarvis ADA v4.8.2 Hybrid Ready
# =====================================================================
"""
Purpose
-------
Static configuration converted from miner_runtime_config.json.
Provides all runtime, wallet, pool, and system parameters
as native Python dictionaries.
"""

from __future__ import annotations

wallets = {
    "BTC": "3FbC1wKBXRx3yUYaLXXyEjKxPLpehnLogB",
    "ETC": "0x5119bF3205dbA7b22E58A7e9FD934c342fDD8Ab3",
    "RVN": "RSYEUmpvdErLArh2SSqPfDPyfka1VpDrfc",
    "ERG": "0x5119bF3205dbA7b22E58A7e9FD934c342fDD8Ab3",
}

pools = {
    "BTC": "stratum+tcp://btc.2miners.com:2020",
    "ETC": "stratum+tcp://etc.2miners.com:1010",
    "RVN": "stratum+tcp://rvn.2miners.com:6060",
    "ERG": "stratum+tcp://erg.2miners.com:8888",
}

algorithms = {
    "BTC": "sha256d",
    "ETC": "etchash",
    "RVN": "kawpow",
    "ERG": "autolykos",
}

runtime = {
    "total_phs": 0.3,
    "max_fraction_per_network": 0.2,
    "btc_overflow_sink": True,
}

network = {
    "auto_refresh_interval_sec": 60,
    "retry_interval_sec": 10,
}

submission_rate = {
    "fallback_allowed_rate_per_second": 2.0,
    "target_acceptance_rate": 0.92,
    "expected_weight": 0.7,
    "observed_weight": 0.3,
    "smoothing": 0.6,
    "min_tick_duration_s": 0.05,
    "jitter_fraction": 0.18,
    "network_target_fraction": 0.05,
    "network_target_fraction_floor": 0.05,
    "network_target_fraction_ceiling": 0.05002,
    "network_target_fraction_nominal": 0.05001,
    "moderate_valid_share_rate_per_second": 2.0,
    "preferred_valid_share_rate_per_second": 2.0,
    "pool_submit_ceiling_per_second": 2.0,
    "network_submit_ceiling_per_second": 2.0,
    "pool_submit_guard_fraction": 0.985,
    "difficulty_request_interval_s": 10.0,
    "difficulty_request_change_fraction": 0.002,
}

system = {
    "mode": "ADA_v3.3_HYBRID",
    "vqram_policy": "vram_hot_ram_cold",
    "self_healing": True,
    "tier": 10,
    "mining_period_s": 0.02,
    "prediction_period_s": 2.0,
    "multiplexer_flush_interval_s": 5.0,
}

coins = {
    "BTC": {
        "mode": "stratum",
        "algorithm": "sha256d",
        "hashrate_sim_hs": 100000000.0,
        "resource_share": 1.0,
        "avg_latency_ms": 100,
        "wallet": "dax97625.rig",
        "wallet_address": wallets["BTC"],
        "pool_url": "stratum+tcp://btc.f2pool.com:1314",
        "port": 1314,
        "difficulty_control_method": "pool_assigned",
        "pool_submit_ceiling_per_second": 3.0,
        "network_submit_ceiling_per_second": 3.0,
        "pool_submit_guard_fraction": 0.90,
        "stratum": {"host": "btc.f2pool.com", "port": 1314},
    },
    "ETC": {
        "mode": "stratum",
        "algorithm": "etchash",
        "hashrate_sim_hs": 96000000.0,
        "resource_share": 1.0,
        "avg_latency_ms": 120,
        "wallet_address": wallets["ETC"],
        "pool_url": "stratum+tcp://etc.2miners.com:1010",
        "port": 1010,
        "authorize_method": "mining.authorize",
        "authorize_with_password": True,
        "difficulty_control_method": "pool_assigned",
        "pool_submit_ceiling_per_second": 10.0,
        "network_submit_ceiling_per_second": 10.0,
        "pool_submit_guard_fraction": 0.92,
        "stratum": {"host": "etc.2miners.com", "port": 1010},
    },
    "RVN": {
        "mode": "stratum",
        "algorithm": "kawpow",
        "hashrate_sim_hs": 15000000.0,
        "resource_share": 0.8,
        "avg_latency_ms": 110,
        "wallet_address": wallets["RVN"],
        "pool_url": "stratum+tcp://rvn.2miners.com:6060",
        "port": 6060,
        "difficulty_control_method": "pool_assigned",
        "pool_submit_ceiling_per_second": 12.0,
        "network_submit_ceiling_per_second": 12.0,
        "pool_submit_guard_fraction": 0.92,
        "stratum": {"host": "rvn.2miners.com", "port": 6060},
    },
    "ERG": {
        "mode": "stratum",
        "algorithm": "autolykos",
        "hashrate_sim_hs": 8000000.0,
        "resource_share": 0.4,
        "avg_latency_ms": 90,
        "wallet_address": wallets["ERG"],
        "stratum": {"host": "erg.2miners.com", "port": 8888},
    },
}

miner_nonce = {
    "default_mode": "gpu_vectorized",
    "default_batch": 64,
    "per_network": {
        "BTC": {
            "mode": "gpu_vectorized",
            "batch": 64,
            "sequence_scan": 256,
            "gpu_batch_size": 2048,
            "amplitude_floor": 0.10,
            "amplitude_ceiling": 0.82,
            "phase_stride_scale": 0.32,
            "substrate_feedback_weight": 0.40,
            "substrate_scan_boost": 96,
            "mining_resonance_weight": 0.28,
            "process_resonance_weight": 0.18,
            "temporal_relativity_weight": 0.16,
            "zero_point_line_weight": 0.14,
            "field_interference_weight": 0.12,
            "collapse_readiness_weight": 0.12,
            "pulse_hash_audit_mode": False,
            "pulse_hash_audit_count": 2000,
        },
        "ETC": {
            "mode": "gpu_vectorized",
            "batch": 64,
            "sequence_scan": 224,
            "gpu_batch_size": 512,
            "amplitude_floor": 0.12,
            "amplitude_ceiling": 0.84,
            "phase_stride_scale": 0.34,
            "substrate_feedback_weight": 0.40,
            "substrate_scan_boost": 96,
        },
        "RVN": {
            "mode": "gpu_vectorized",
            "batch": 64,
            "sequence_scan": 192,
            "gpu_batch_size": 384,
            "amplitude_floor": 0.14,
            "amplitude_ceiling": 0.86,
            "phase_stride_scale": 0.36,
            "substrate_feedback_weight": 0.40,
            "substrate_scan_boost": 96,
        },
        "ERG": {
            "mode": "derivative",
            "batch": 64,
        },
    },
}

vhw_gpu_initiator = {
    "sustain_pct": 0.05,
    "program_id": "gpu_initiator_photonic",
    "profile_name": "photonic_actuation",
    "telemetry_mode": "live_startup",
    "actuation_backend": "vulkan_calibration",
    "capture_sleep": False,
    "telemetry_sample_period_s": 0.05,
    "loop_period_s": 0.25,
    "history_size": 8,
    "horizon_frames": 2,
    "pulse_cycles": 1,
    "memory_basin_count": 4,
    "scheduler_zone_count": 4,
    "active_zone_limit": 2,
    "actuation_settle_s": 0.004,
    "vulkan_element_count": 4096,
    "vulkan_iterations": 12,
    "vulkan_frequency": 0.245,
    "vulkan_amplitude": 0.18,
    "vulkan_voltage": 0.33,
    "vulkan_current": 0.33,
}

# ---------------------------------------------------------------------
# Access helpers
# ---------------------------------------------------------------------
def get_coin(symbol: str):
    """Return coin configuration."""
    return coins.get(symbol.upper(), {})

def get_wallet(symbol: str) -> str:
    return wallets.get(symbol.upper(), "")

def get_pool(symbol: str) -> str:
    return pools.get(symbol.upper(), "")

def get_algorithm(symbol: str) -> str:
    return algorithms.get(symbol.upper(), "")

# ---------------------------------------------------------------------
# Diagnostic view
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Miner Runtime Config (Static Python) ===")
    print(f"Runtime total PH/s: {runtime['total_phs']}")
    for sym, info in coins.items():
        print(f"{sym}: {info['algorithm']} @ {info['hashrate_sim_hs']} H/s")
