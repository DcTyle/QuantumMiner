# ============================================================================
# VirtualMiner / VHW
# ASCII-ONLY SOURCE FILE
# File: compute_manager.py
# Version: v1.0.0 (Generic Compute Fabric)
# ----------------------------------------------------------------------------
# Purpose:
#   This module replaces lane_allocator.py and provides a pure compute fabric
#   used by all subsystems: miner, prediction, AI, Jarvis, and market modules.
#
#   The ComputeManager manages tiers, clusters, and lane spawning.
#   The ComputeLane executes work dispatched through ComputeWrapper.
#
#   All domain-specific logic (mining, prediction, AI, etc.) now belongs in
#   the engines associated with those modules. This file contains no mining
#   logic, no difficulty math, no share accounting, no pruning logic, and
#   no statevector encoding.
#
# Logging Standard:
#   ASCII-only, UTC timestamps, no unicode.
# ============================================================================

from __future__ import annotations
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
import threading
import time
import logging
import random
from bios.event_bus import get_event_bus
from VHW.gpu_pulse_runtime import build_substrate_trace_runtime
from VHW.system_utils import TierOscillator, system_headroom

# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
def _mk_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        logging.Formatter.converter = time.gmtime
        h.setFormatter(fmt)
        lg.addHandler(h)
        lg.propagate = False
    return lg

LOG = _mk_logger("VHW.ComputeManager")

# ----------------------------------------------------------------------------
# SAFE FALLBACKS
# ----------------------------------------------------------------------------
try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def __init__(self):
            self._kv = {}
        def get(self, k, d=None): return self._kv.get(k, d)
        def store(self, k, v): self._kv[k] = v

class ComputeMode:
    BATCH = "BATCH"
    SINGLE = "SINGLE"
    VECTOR = "VECTOR"
    CUSTOM = "CUSTOM"


class Subsystem:
    MINER = "MINER"
    PREDICTION = "PREDICTION"
    AI = "AI"
    JARVIS = "JARVIS"
    MARKET = "MARKET"
    SYSTEM = "SYSTEM"

@dataclass
class ComputeWrapper:
    subsys: str
    mode: str
    payload: Dict[str, Any]
    params: Dict[str, Any]

# ----------------------------------------------------------------------------
# DATA STRUCTURES
# ----------------------------------------------------------------------------
@dataclass
class TierSpec:
    tier_id: int
    vqram_mb: int
    lane_count: int

@dataclass
class ComputeLane:
    lane_id: str
    tier_id: int
    cluster_id: int = 0
    active: bool = True
    tick: int = 0
    _lock: threading.RLock = threading.RLock()

    def compute(self, wrapper: ComputeWrapper) -> Any:
        with self._lock:
            self.tick += 1
            # Mining orchestration via NonceMath
            try:
                if wrapper.subsys == Subsystem.MINER:
                    from miner.nonce_math import NonceMath
                    from neural_object import (
                        neural_objectPacket,
                    )
                    submitter = wrapper.params.get("submitter")

                    payload = wrapper.payload or {}
                    try:
                        prior_trace_state = dict(getattr(self, "_gpu_pulse_trace_state", {}))
                    except Exception:
                        prior_trace_state = {}
                    packet = None
                    # Prefer per-lane packet from jobs_map if provided
                    try:
                        jm = payload.get("jobs_map", {})
                        if isinstance(jm, dict):
                            entry = jm.get(self.lane_id)
                            if isinstance(entry, neural_objectPacket):
                                packet = entry
                    except Exception:
                        packet = None
                    # Or a direct packet in payload
                    if packet is None and isinstance(payload, dict):
                        pkt_in = payload.get("packet")
                        if isinstance(pkt_in, neural_objectPacket):
                            packet = pkt_in
                    # If no packet available, nothing to do this tick
                    if packet is None:
                        return {"shares": [], "telemetry": {"lane_id": self.lane_id, "ticks": self.tick}, "meta": {}}

                    try:
                        prior_trace_profile = dict(getattr(self, "_gpu_pulse_trace_profile", {}))
                    except Exception:
                        prior_trace_profile = {}
                    try:
                        prior_trace_frames = [dict(item or {}) for item in list(getattr(self, "_gpu_pulse_frame_history", []))]
                    except Exception:
                        prior_trace_frames = []
                    try:
                        prior_memory_basin_state = dict(getattr(self, "_gpu_pulse_memory_basin_state", {}))
                    except Exception:
                        prior_memory_basin_state = {}
                    try:
                        prior_scheduler_state = dict(getattr(self, "_gpu_pulse_scheduler_state", {}))
                    except Exception:
                        prior_scheduler_state = {}
                    try:
                        prior_process_state = dict(getattr(self, "_gpu_pulse_process_state", {}))
                    except Exception:
                        prior_process_state = {}
                    try:
                        pre_nm_snapshot = NonceMath.snapshot_lane_state(self)
                    except Exception:
                        pre_nm_snapshot = {}

                    # Feed live telemetry into the packet so NonceMath can shape
                    # its vector field from real GPU and system utilization.
                    merged_payload = dict(packet.system_payload or {})
                    if isinstance(payload, dict):
                        for key in (
                            "global_util",
                            "gpu_util",
                            "mem_bw_util",
                            "cpu_util",
                            "alpha_flux",
                            "flux_coeff",
                            "drift_coeff",
                            "phase_step",
                            "telemetry_mode",
                            "telemetry_sample_period_s",
                            "actuation_backend",
                        ):
                            if key in payload:
                                merged_payload[key] = payload.get(key)

                    # Build the live runtime-owned substrate trace before nonce
                    # generation so the same-pulse telemetry/actuation window
                    # can shape candidate formation for this tick.
                    try:
                        substrate_trace = build_substrate_trace_runtime(
                            lane_id=self.lane_id,
                            tick=self.tick,
                            system_payload=dict(merged_payload or {}),
                            nonce_snapshot=dict(pre_nm_snapshot or {}),
                            previous_trace_state=prior_trace_state,
                            packet=packet,
                            runtime_payload=dict(payload or {}) if isinstance(payload, dict) else {},
                            previous_profile=prior_trace_profile,
                            frame_history=prior_trace_frames,
                            previous_memory_basin_state=prior_memory_basin_state,
                            previous_scheduler_state=prior_scheduler_state,
                            previous_process_state=prior_process_state,
                            live_cycle=True,
                        )
                    except Exception:
                        substrate_trace = {
                            "active": False,
                            "error": "trace_runtime_failed",
                            "path": "trace_runtime_failed",
                            "trace_state": prior_trace_state,
                            "trace_vram": {"resident": False, "reason": "trace_runtime_failed", "update_count": 0},
                            "profile": prior_trace_profile,
                            "frames": prior_trace_frames,
                            "memory_basin_state": prior_memory_basin_state,
                            "scheduler_state": prior_scheduler_state,
                            "process_state": prior_process_state,
                            "actuation_summary": {"applied": False, "call_count": 0, "applied_count": 0, "mode": "error"},
                        }
                    active_trace_state = dict(substrate_trace.get("trace_state", {}) or prior_trace_state)
                    active_memory_basin_state = dict(substrate_trace.get("memory_basin_state", {}) or prior_memory_basin_state)
                    active_scheduler_state = dict(substrate_trace.get("scheduler_state", {}) or prior_scheduler_state)
                    active_process_state = dict(substrate_trace.get("process_state", {}) or prior_process_state)
                    if active_trace_state:
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
                            "trace_trajectory_spectral_id",
                            "trace_predicted_trajectory_spectral_id",
                            "trace_trajectory_conservation_alignment",
                            "trace_trajectory_prediction_alignment",
                            "trace_trajectory_expansion_term",
                            "trace_gradient_spectral_id",
                            "trace_gpu_pulse_phase_effect",
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
                            "trace_material",
                        ):
                            if key in active_trace_state:
                                merged_payload[key] = active_trace_state.get(key)
                        for key in (
                            "trace_vector",
                            "trace_axis_vector",
                            "trace_dof_vector",
                            "trace_trajectory_9d",
                            "trace_predicted_trajectory_9d",
                            "trace_frequency_gradient_9d",
                        ):
                            if key in active_trace_state:
                                merged_payload[key] = list(active_trace_state.get(key) or [])
                    for key in (
                        "active_basin_id",
                        "active_basin_name",
                        "basin_coherence",
                        "retention_strength",
                        "temporal_relativity_norm",
                        "zero_point_line_distance",
                        "field_interference_norm",
                        "resonant_interception_inertia",
                    ):
                        if key in active_memory_basin_state:
                            merged_payload[key] = active_memory_basin_state.get(key)
                    for key in (
                        "scheduling_mode",
                        "active_zone_id",
                        "active_zone_name",
                        "process_transport_gate",
                        "isolation_wall_strength",
                        "resonance_delta",
                        "temporal_relativity_norm",
                        "zero_point_line_distance",
                        "field_interference_norm",
                        "resonant_interception_inertia",
                    ):
                        if key in active_scheduler_state:
                            merged_payload["scheduler_mode" if key == "scheduling_mode" else key] = active_scheduler_state.get(key)
                    for key in (
                        "process_mode",
                        "process_resonance",
                        "mining_resonance_gate",
                        "collapse_readiness",
                        "temporal_relativity_norm",
                        "zero_point_line_distance",
                        "field_interference_norm",
                        "resonant_interception_inertia",
                    ):
                        if key in active_process_state:
                            merged_payload[key] = active_process_state.get(key)
                    packet = neural_objectPacket(
                        packet_type=packet.packet_type,
                        network=packet.network,
                        raw_payload=dict(packet.raw_payload or {}),
                        system_payload=merged_payload,
                        metadata=dict(packet.metadata or {}),
                        derived_state=dict(packet.derived_state or {}),
                    )
                    try:
                        setattr(self, "_gpu_pulse_trace_state", dict(active_trace_state))
                        setattr(self, "_gpu_pulse_trace_profile", dict(substrate_trace.get("profile", {}) or {}))
                        setattr(self, "_gpu_pulse_frame_history", [dict(item or {}) for item in list(substrate_trace.get("frames", []) or [])])
                        setattr(self, "_gpu_pulse_memory_basin_state", dict(active_memory_basin_state))
                        setattr(self, "_gpu_pulse_scheduler_state", dict(active_scheduler_state))
                        setattr(self, "_gpu_pulse_process_state", dict(active_process_state))
                    except Exception:
                        pass

                    # Compute using NonceMath; return a list of neural_objectPacket.
                    # Resolve per-network nonce config once the lane-specific
                    # packet is known so the active BIOS runtime can mix coins.
                    params_dict = dict(wrapper.params or {}) if isinstance(wrapper.params, dict) else {}
                    mode = str(params_dict.get("mode", "derivative"))
                    count = None
                    nonce_params = {}
                    nonce_cfg = {}
                    try:
                        if "count" in params_dict and params_dict.get("count") is not None:
                            count = int(params_dict.get("count"))
                    except Exception:
                        count = None
                    try:
                        nonce_params = dict(params_dict.get("nonce_params", {}) or {})
                    except Exception:
                        nonce_params = {}
                    try:
                        nonce_cfg = dict(params_dict.get("nonce_cfg", {}) or {})
                    except Exception:
                        nonce_cfg = {}
                    try:
                        packet_network = str(getattr(packet.network, "name", packet.network)).upper()
                    except Exception:
                        packet_network = ""
                    if nonce_cfg:
                        default_mode = str(nonce_cfg.get("default_mode", mode or "derivative"))
                        default_batch = 0
                        try:
                            default_batch = int(nonce_cfg.get("default_batch", count or 0) or 0)
                        except Exception:
                            default_batch = 0
                        per_network_cfg = nonce_cfg.get("per_network", {}) if isinstance(nonce_cfg.get("per_network", {}), dict) else {}
                        net_cfg = dict(per_network_cfg.get(packet_network, {}) or {}) if packet_network else {}
                        resolved_mode = str(net_cfg.get("mode", default_mode or "derivative"))
                        resolved_count = 0
                        try:
                            resolved_count = int(net_cfg.get("batch", default_batch) or 0)
                        except Exception:
                            resolved_count = 0
                        derived_nonce_params = dict(net_cfg)
                        derived_nonce_params.pop("mode", None)
                        derived_nonce_params.pop("batch", None)
                        merged_nonce_params = dict(derived_nonce_params)
                        merged_nonce_params.update(nonce_params)
                        nonce_params = merged_nonce_params
                        if "mode" not in params_dict or not str(params_dict.get("mode", "")).strip():
                            mode = resolved_mode
                        if ("count" not in params_dict or params_dict.get("count") is None) and resolved_count > 0:
                            count = resolved_count
                    if active_trace_state and "substrate_trace_state" not in nonce_params:
                        nonce_params["substrate_trace_state"] = dict(active_trace_state)
                    # Persist per-lane state across ticks by passing lane_state=self
                    results = NonceMath.compute(
                        packet,
                        lane_state=self,
                        mode=mode,
                        count=count,
                        params=nonce_params,
                    )

                    # Optional NonceMath lane snapshot for telemetry/meta
                    try:
                        nm_snapshot = NonceMath.snapshot_lane_state(self)
                    except Exception:
                        nm_snapshot = {}

                    # Forward neural_objectPacket for pool submission
                    valid_count = 0
                    if submitter is not None:
                        for rp in results:
                            try:
                                valid = bool(getattr(rp, "system_payload", {}).get("valid", True))
                                if not valid:
                                    continue
                                valid_count += 1
                                # Submit packet to pool submission flow
                                if hasattr(submitter, "submit_packet"):
                                    submitter.submit_packet(self.lane_id, rp)
                            except Exception:
                                LOG.error("submit forward failed", exc_info=True)

                    telem = {
                        "lane_id": self.lane_id,
                        "ticks": self.tick,
                        "candidate_count": len(results),
                        "valid_count": valid_count,
                        "validated_candidate_count": int(dict(nm_snapshot or {}).get("validated_candidate_count", 0)),
                        "valid_share_count": int(dict(nm_snapshot or {}).get("valid_share_count", valid_count)),
                        "trace_active": bool(substrate_trace.get("active", False)),
                        "trace_support": float(dict(substrate_trace.get("trace_state", {})).get("trace_support", 0.0)),
                        "trace_resonance": float(dict(substrate_trace.get("trace_state", {})).get("trace_resonance", 0.0)),
                        "trace_alignment": float(dict(substrate_trace.get("trace_state", {})).get("trace_alignment", 0.0)),
                        "trace_path": str(substrate_trace.get("path", "")),
                        "trace_phase_ring_strength": float(dict(substrate_trace.get("trace_state", {})).get("trace_phase_ring_strength", 0.0)),
                        "trace_shared_vector_phase_lock": float(dict(substrate_trace.get("trace_state", {})).get("trace_shared_vector_phase_lock", 0.0)),
                        "trace_zero_point_crossover": float(dict(substrate_trace.get("trace_state", {})).get("trace_zero_point_crossover", 0.0)),
                        "trace_actuation_applied": bool(dict(substrate_trace.get("actuation_summary", {})).get("applied", False)),
                        "trace_vram_resident": bool(dict(substrate_trace.get("trace_vram", {})).get("resident", False)),
                        "trace_vram_updates": int(dict(substrate_trace.get("trace_vram", {})).get("update_count", 0)),
                    }
                    meta = {}
                    if isinstance(nm_snapshot, dict) and nm_snapshot:
                        meta["nonce_math"] = nm_snapshot
                    meta["substrate_trace"] = substrate_trace

                    return {"shares": [], "telemetry": telem, "meta": meta}
            except Exception:
                LOG.error("lane compute failed (miner path)", exc_info=True)
                return None

            # Default path for non-mining subsystems
            # Unified path for non-mining: accept neural_objectPacket and evolve
            try:
                from neural_object import neural_objectPacket as _Pkt, neural_object
                pkt = wrapper.payload.get("packet") if isinstance(wrapper.payload, dict) else None
                if isinstance(pkt, _Pkt):
                    lane = neural_object(self.lane_id)
                    return lane.evolve(pkt)
            except Exception:
                pass

            handler = wrapper.params.get("handler")
            if handler is None:
                return None
            try:
                return handler(self, wrapper.payload, wrapper.params)
            except Exception:
                LOG.error("lane compute failed", exc_info=True)
                return None

# ----------------------------------------------------------------------------
# COMPUTE MANAGER
# ----------------------------------------------------------------------------
class ComputeManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.vsd = cfg.get("vsd") or VSDManager()
        self.bus = get_event_bus()

        self.tiers: Dict[int, TierSpec] = {}
        self.lanes: Dict[str, ComputeLane] = {}
        self._lane_counter = 0
        self._lock = threading.RLock()
        self._tier_osc: TierOscillator | None = None

        self._init_tiers()
        self._subscribe()

        LOG.info("ComputeManager initialized")

    # ---------------------------------------------------------------------
    # BIOS GATE
    # ---------------------------------------------------------------------
    def _bios_ready(self) -> bool:
        try:
            return bool(self.vsd.get("system/bios_boot_ok", False))
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # EVENT BUS
    # ---------------------------------------------------------------------
    def _subscribe(self):
        try:
            self.bus.subscribe("boot.complete", lambda e=None: LOG.info("boot complete"))
        except Exception:
            LOG.error("eventbus subscribe failed", exc_info=True)

    # ---------------------------------------------------------------------
    # TIERS
    # ---------------------------------------------------------------------
    def _init_tiers(self):
        tiers_cfg = self.cfg.get("tiers")
        if not tiers_cfg:
            tiers_cfg = [{"tier_id": i, "vqram_mb": 256} for i in range(4)]

        ids: List[int] = []
        for t in tiers_cfg:
            tid = int(t["tier_id"])
            vq = int(t.get("vqram_mb", 256))
            self.tiers[tid] = TierSpec(tier_id=tid, vqram_mb=vq, lane_count=0)
            ids.append(tid)

        try:
            self._tier_osc = TierOscillator(ids)
        except Exception:
            self._tier_osc = None

        # Allocate default lanes: 1 per tier
        for tid in ids:
            self.allocate_lane(tid)
            LOG.info("Allocated default lane for tier %d", tid)

    # ---------------------------------------------------------------------
    # LANE SPAWNING
    # ---------------------------------------------------------------------
    def _new_lane_id(self, tid: int) -> str:
        self._lane_counter += 1
        r = hex(random.getrandbits(32))[2:10]
        return "lane_%d_%s" % (tid, r)

    def allocate_lane(self, tier_id: int) -> ComputeLane:
        lane_id = self._new_lane_id(tier_id)
        lane = ComputeLane(lane_id=lane_id, tier_id=tier_id)
        self.lanes[lane_id] = lane
        self.tiers[tier_id].lane_count += 1
        return lane

    # ---------------------------------------------------------------------
    # DISPATCH WRAPPER TO ALL LANES
    # ---------------------------------------------------------------------
    def dispatch(self, wrapper: ComputeWrapper) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if not self._bios_ready():
            return out

        # Optional backpressure hook: slow compute if submit queue is deep
        throttle_delay = 0.0
        try:
            fb = self.vsd.get("miner/submitter/queue_depth", 0)
            qd = int(fb)
            # Simple linear backoff: every 256 queued adds 1ms delay per dispatch
            throttle_delay = min(0.050, (qd // 256) * 0.001)
        except Exception:
            throttle_delay = 0.0

        weights: Dict[int, float] = {}
        if self._tier_osc is not None and len(self.tiers) > 1:
            try:
                sh = system_headroom()
                weights = self._tier_osc.update(float(sh.get("headroom", 0.12)))
            except BaseException:
                weights = {}

        with self._lock:
            for lid, lane in self.lanes.items():
                if not lane.active:
                    continue

                if weights:
                    w = weights.get(lane.tier_id, 0.0)
                    if w <= 0.0:
                        continue

                res = lane.compute(wrapper)
                if res is not None:
                    out[lid] = res

                if throttle_delay > 0.0:
                    try:
                        time.sleep(throttle_delay)
                    except Exception:
                        pass

        return out

    # ---------------------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                lid: {
                    "tier": lane.tier_id,
                    "active": lane.active,
                    "ticks": lane.tick
                }
                for lid, lane in self.lanes.items()
            }
