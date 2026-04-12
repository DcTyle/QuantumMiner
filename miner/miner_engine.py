# ============================================================================
# Quantum Application / miner
# ASCII-ONLY SOURCE FILE
# File: miner_engine.py
# Version: v5.0.1 Hybrid (ComputeManager + Strict Subsystem Isolation)
# ============================================================================
"""
Purpose
-------
Pure mining engine for VirtualMiner.

This engine:
  - Builds mining ComputeWrappers
  - Dispatches them to ComputeManager
  - Processes lane compute outputs STRICTLY for mining
  - Sends shares to ShareAllocator and Submitter ONLY
  - Updates VSD miner telemetry
  - Applies mining failsafe logic

Subsystem Isolation (Guaranteed)
-------------------------------
This engine DOES NOT:
  - Communicate with prediction engine
  - Communicate with market utils
  - Communicate with trading engine
  - Communicate with Neuralis AI
  - Communicate with Jarvis AI
  - Provide compute results to any subsystem except mining

All subsystem interactions outside mining are PROHIBITED.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import threading
import time
import os
import logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _init_logger() -> logging.Logger:
    logger = logging.getLogger("miner.engine")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = _init_logger()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from VHW.compute_manager import (
    ComputeManager,
    ComputeWrapper,
    Subsystem,
    ComputeMode
)
from VHW.share_allocator import ShareAllocator
from VHW.failsafe import FailsafeGovernor
from core.utils import append_telemetry

# BIOS EventBus
try:
    from bios.event_bus import get_event_bus
except Exception:
    def get_event_bus(): return None

BOOT_COMPLETE_TOPIC = "boot.complete"

# ============================================================================
# MINER ENGINE
# ============================================================================
class MinerEngine:
    def __init__(self,
                 compute_manager: ComputeManager,
                 share_allocator: ShareAllocator,
                 submitter: Any,
                 failsafe: Optional[FailsafeGovernor],
                 read_telemetry,
                 vsd,
                 handler_miner: Optional[Any] = None,
                 tick_interval_s: float = 0.25,
                 network_cfg: Optional[Dict[str, Any]] = None,
                 nonce_cfg: Optional[Dict[str, Any]] = None) -> None:
        # core components
        self.cm = compute_manager
        self.share_allocator = share_allocator
        self.submitter = submitter
        self.failsafe = failsafe
        self.read_telemetry = read_telemetry
        self.vsd = vsd
        self.handler_miner = handler_miner
        # configuration
        self.network_cfg = network_cfg or {}
        self.nonce_cfg = nonce_cfg or {}
        self.tick_interval_s = float(tick_interval_s)
        # thread state
        self._stop = False
        self._thr: Optional[threading.Thread] = None
        self.iterations = 0
        self.last_snapshot_path = ""
        self._pause_lock = threading.RLock()
        self._paused = False
        self._pause_note = ""
        # error tracking
        self._error_count = 0
        self._max_errors = 10

    # ---------------------------------------------------------------------
    def start(self) -> None:
        if self._thr:
            logger.info("MinerEngine already running")
            return
        logger.info("MinerEngine starting loop")
        self._stop = False
        self._thr = threading.Thread(target=self._loop, daemon=True, name="miner_engine")
        self._thr.start()

    # ---------------------------------------------------------------------
    def stop(self, timeout: float = 2.0) -> None:
        if not self._thr:
            logger.info("MinerEngine.stop called with no active thread")
            return

        logger.info("MinerEngine stopping")
        self._stop = True
        self._thr.join(timeout=timeout)
        self._thr = None

        # Snapshot via Neuralis_AI is not permitted by boundary policy; no-op here

    def is_paused(self) -> bool:
        with self._pause_lock:
            return bool(self._paused)

    def pause(self, note: str = "", source: str = "control_center") -> None:
        with self._pause_lock:
            self._paused = True
            self._pause_note = str(note or "")
        try:
            self.vsd.store("miner/control/engine", {
                "ts": time.time(),
                "paused": True,
                "note": self._pause_note,
                "source": str(source or "control_center"),
            })
        except Exception:
            pass
        logger.info("MinerEngine paused: %s", self._pause_note or "pause requested")

    def resume(self, note: str = "", source: str = "control_center") -> None:
        with self._pause_lock:
            self._paused = False
            self._pause_note = str(note or "")
        try:
            self.vsd.store("miner/control/engine", {
                "ts": time.time(),
                "paused": False,
                "note": self._pause_note,
                "source": str(source or "control_center"),
            })
        except Exception:
            pass
        logger.info("MinerEngine resumed")

    # ---------------------------------------------------------------------
    def _loop(self) -> None:
        logger.info("MinerEngine loop started")
        while not self._stop:
            if self.is_paused():
                time.sleep(self.tick_interval_s)
                continue
            try:
                telem = self.read_telemetry() or {}
                append_telemetry("miner_engine_tick", telem)

                # Build mining wrapper ONLY. Per-network nonce selection is
                # resolved in ComputeLane once the lane-specific packet is known.
                wrapper_params = {
                    "handler": self.handler_miner,
                    "network_cfg": self.network_cfg,
                    "submitter": self.submitter,
                    "nonce_cfg": dict(self.nonce_cfg or {}),
                }

                wrapper = ComputeWrapper(
                    subsys=Subsystem.MINER,
                    mode=ComputeMode.BATCH,
                    payload=telem,
                    params=wrapper_params,
                )

                # Dispatch to ComputeManager
                lane_results = self.cm.dispatch(wrapper)

                # Handle ONLY mining results
                self._handle_lane_results(lane_results)

                # Failsafe checks
                if self.failsafe:
                    try:
                        self.failsafe.check_health(telem)
                    except Exception:
                        logger.error("Failsafe health check failed", exc_info=True)

                self.iterations += 1
                self._error_count = 0  # Reset error count on success

            except Exception as exc:
                self._error_count += 1
                logger.error("MinerEngine loop error: %s (error %d/%d)", exc, self._error_count, self._max_errors, exc_info=True)
                if self._error_count >= self._max_errors:
                    logger.critical("MinerEngine encountered too many errors (%d). Stopping.", self._error_count)
                    self._stop = True
                    break
            time.sleep(self.tick_interval_s)
        logger.info("MinerEngine loop exited")

    # ---------------------------------------------------------------------
    def _handle_lane_results(self, lane_results: Dict[str, Any]):
        """
        Processes lane compute outputs STRICTLY for mining.
        handler_miner must return:
            {
                "shares": [...],
                "meta": {...},
                "telemetry": {...}
            }
        """
        for lane_id, result in lane_results.items():
            if not result:
                continue

            # 1. Submit shares ONLY to mining pipeline (Governor v2 aware)
            shares = result.get("shares", [])
            for share in shares:
                try:
                    # Submit to pool via Submitter (uses client_resolver)
                    net = str(share.get("network", self.network_cfg.get("network", "ETC")))
                    can_accept = True
                    try:
                        if hasattr(self.submitter, "can_accept"):
                            can_accept = bool(self.submitter.can_accept(net, lane_id))
                    except Exception:
                        can_accept = True
                    if not can_accept:
                        # Emit throttle telemetry
                        try:
                            self.vsd.store("miner/engine/throttle", {
                                "ts": time.time(),
                                "lane": lane_id,
                                "network": net,
                                "reason": "rate_limit"
                            })
                        except Exception:
                            pass
                        continue
                    if hasattr(self.submitter, "submit_share"):
                        self.submitter.submit_share(lane_id, net, share)
                    elif hasattr(self.submitter, "submit"):
                        self.submitter.submit(share)

                    # Telemetry allocation if allocator signature supports it
                    # VHW ShareAllocator strict signature: (lane_id, network, nonce_hex, hash_hex, target_hex, is_valid, extra)
                    try:
                        net = str(share.get("network", self.network_cfg.get("network", "ETC")))
                        nonce_hex = ("%x" % int(share.get("nonce", 0)))
                        hash_hex = str(share.get("hash_hex", ""))
                        target_hex = str(share.get("target", ""))
                        is_valid = False
                        try:
                            if hash_hex and target_hex:
                                is_valid = int(hash_hex, 16) <= int(target_hex, 16)
                        except Exception:
                            is_valid = False
                        self.share_allocator.submit_share(lane_id, net, nonce_hex, hash_hex, target_hex, bool(is_valid), {})
                    except Exception:
                        logger.error("ShareAllocator submission failed", exc_info=True)

                except Exception:
                    logger.error("Share processing failed", exc_info=True)

            # 2. Update miner-only telemetry
            miner_tel = result.get("telemetry", {}) or {}
            meta = result.get("meta", {}) or {}

            # Enrich telemetry with NonceMath lane snapshot if present
            try:
                nm = meta.get("nonce_math", {}) if isinstance(meta, dict) else {}
                if isinstance(nm, dict) and nm:
                    miner_tel = dict(miner_tel)
                    miner_tel.update({
                        "entropy_score": float(nm.get("entropy_score", 0.0)),
                        "psi": float(nm.get("psi", 0.0)),
                        "flux": float(nm.get("flux", 0.0)),
                        "harmonic": float(nm.get("harmonic", 0.0)),
                        "phase": float(nm.get("phase", 0.0)),
                        "last_nonce": int(nm.get("last_nonce", 0)),
                        "amplitude_cap": float(nm.get("amplitude_cap", 0.0)),
                        "coherence_peak": float(nm.get("coherence_peak", 0.0)),
                        "target_interval": int(nm.get("target_interval", 0)),
                        "candidate_count": int(nm.get("candidate_count", 0)),
                        "validated_candidate_count": int(nm.get("validated_candidate_count", 0)),
                        "valid_share_count": int(nm.get("valid_share_count", 0)),
                        "valid_ratio": float(nm.get("valid_ratio", 0.0)),
                        "atomic_vector_x": float(nm.get("atomic_vector_x", 0.0)),
                        "atomic_vector_y": float(nm.get("atomic_vector_y", 0.0)),
                        "atomic_vector_z": float(nm.get("atomic_vector_z", 0.0)),
                        "vector_path_score": float(nm.get("vector_path_score", 0.0)),
                        "search_backend": str(nm.get("search_backend", "")),
                        "decoded_phase_turns": float(nm.get("decoded_phase_turns", 0.0)),
                        "decoded_target_word": int(nm.get("decoded_target_word", 0)),
                        "decoded_vector_word": int(nm.get("decoded_vector_word", 0)),
                        "decoded_nonce_mix_word": int(nm.get("decoded_nonce_mix_word", 0)),
                        "decoded_nonce_hex": str(nm.get("decoded_nonce_hex", "")),
                        "decoded_phase_word": int(nm.get("decoded_phase_word", 0)),
                        "decoded_trace_word": int(nm.get("decoded_trace_word", 0)),
                        "decoded_window": float(nm.get("decoded_window", 0.0)),
                        "decoded_interval": int(nm.get("decoded_interval", 0)),
                        "decoded_trace_gate": float(nm.get("decoded_trace_gate", 0.0)),
                        "pow_test_enabled": bool(nm.get("pow_test_enabled", False)),
                        "pow_test_requested_count": int(nm.get("pow_test_requested_count", 0)),
                        "pow_test_audit_count": int(nm.get("pow_test_audit_count", 0)),
                        "pow_test_valid_count": int(nm.get("pow_test_valid_count", 0)),
                        "pow_test_valid_ratio": float(nm.get("pow_test_valid_ratio", 0.0)),
                        "pow_test_best_nonce": int(nm.get("pow_test_best_nonce", 0)),
                        "pow_test_best_zero_nibbles": int(nm.get("pow_test_best_zero_nibbles", 0)),
                        "pow_test_best_distance_score": float(nm.get("pow_test_best_distance_score", 0.0)),
                        "pow_test_best_hash_prefix": str(nm.get("pow_test_best_hash_prefix", "")),
                    })
            except Exception:
                logger.error("Failed to merge NonceMath telemetry", exc_info=True)

            try:
                substrate_trace = meta.get("substrate_trace", {}) if isinstance(meta, dict) else {}
                trace_state = dict(substrate_trace.get("trace_state", {}) or {}) if isinstance(substrate_trace, dict) else {}
                trace_vram = dict(substrate_trace.get("trace_vram", {}) or {}) if isinstance(substrate_trace, dict) else {}
                if trace_state:
                    miner_tel = dict(miner_tel)
                    miner_tel.update({
                        "trace_support": float(trace_state.get("trace_support", 0.0)),
                        "trace_resonance": float(trace_state.get("trace_resonance", 0.0)),
                        "trace_alignment": float(trace_state.get("trace_alignment", 0.0)),
                        "trace_memory": float(trace_state.get("trace_memory", 0.0)),
                        "trace_flux": float(trace_state.get("trace_flux", 0.0)),
                        "trace_temporal_persistence": float(trace_state.get("trace_temporal_persistence", 0.0)),
                        "trace_temporal_overlap": float(trace_state.get("trace_temporal_overlap", 0.0)),
                        "trace_voltage_frequency_flux": float(trace_state.get("trace_voltage_frequency_flux", 0.0)),
                        "trace_frequency_voltage_flux": float(trace_state.get("trace_frequency_voltage_flux", 0.0)),
                        "trace_vram_resident": bool(trace_vram.get("resident", False)),
                        "trace_vram_updates": int(trace_vram.get("update_count", 0)),
                    })
            except Exception:
                logger.error("Failed to merge substrate trace telemetry", exc_info=True)

            if miner_tel:
                try:
                    append_telemetry("miner_lane_output", miner_tel)
                except Exception:
                    logger.error("Telemetry update failed", exc_info=True)

    # ---------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        try:
            cm_status = self.cm.status()
        except Exception:
            cm_status = {}

        try:
            share_stats = self.share_allocator.counters()
        except Exception:
            share_stats = {}

        try:
            last_snap = self.vsd.get("snapshots/last_engine_state", "")
        except Exception:
            last_snap = ""

        return {
            "iterations": self.iterations,
            "paused": bool(self.is_paused()),
            "compute_manager": cm_status,
            "share_allocator": share_stats,
            "last_snapshot": last_snap
        }


# ============================================================================
# AUTO-START HOOK
# ============================================================================
def register_miner_autostart(engine: MinerEngine) -> None:
    bus = get_event_bus()
    if bus is None:
        logger.warning("EventBus unavailable; autostart disabled")
        return

    def _on_boot(payload: Dict[str, Any]) -> None:
        logger.info("boot.complete received; starting MinerEngine")
        engine.start()

    try:
        bus.subscribe(BOOT_COMPLETE_TOPIC, _on_boot, once=True, priority=0)
        logger.info("MinerEngine autostart wired")
    except Exception as exc:
        logger.error("Autostart wiring failed: %s", exc, exc_info=True)
