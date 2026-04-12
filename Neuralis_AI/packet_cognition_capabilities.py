"""
ASCII-ONLY
Neuralis Cognition Capabilities Packet (VSD-hosted)

Purpose
-------
Build and persist a structured packet that describes current cognition
capabilities, observed buses/keys, and proposed routing for missing features.

Storage
-------
VSD path: neuralis/packets/cognition_capabilities/v1/current

Minimal, modular, no external runtime dependencies beyond VSDManager.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import hashlib

try:
    # Authoritative VSD
    from VHW.vsd_manager import VSDManager
except Exception:  # Fallback (should not be used in production)
    class VSDManager:  # type: ignore
        def __init__(self) -> None:
            self._m: Dict[str, Any] = {}
        def get(self, k: str, d: Any = None) -> Any:
            return self._m.get(str(k), d)
        def store(self, k: str, v: Any) -> None:
            self._m[str(k)] = v


VSD_PATH_CURRENT = "neuralis/packets/cognition_capabilities/v1/current"


def _sha256_ascii(s: str) -> str:
    try:
        return hashlib.sha256(s.encode("ascii", "ignore")).hexdigest()
    except Exception:
        return "0" * 64


def _canonicalize_json(obj: Dict[str, Any]) -> str:
    # Minimal JSON canonicalization (no json import to keep deps tiny)
    # We rely on deterministic key order by sorting manually.
    # For large packets, consider switching to json.dumps with sort_keys=True.
    def _encode(o: Any) -> str:
        if isinstance(o, dict):
            items = []
            for k in sorted(o.keys()):
                items.append(f"\"{str(k)}\":{_encode(o[k])}")
            return "{" + ",".join(items) + "}"
        if isinstance(o, list):
            return "[" + ",".join(_encode(x) for x in o) + "]"
        if isinstance(o, str):
            # ensure ASCII-only
            s = o.encode("ascii", "ignore").decode("ascii")
            return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        if isinstance(o, (int, float)):
            return str(o)
        if isinstance(o, bool):
            return "true" if o else "false"
        if o is None:
            return "null"
        # Fallback to string
        s = str(o)
        s = s.encode("ascii", "ignore").decode("ascii")
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

    return _encode(obj)


def build_packet() -> Dict[str, Any]:
    now = time.time()
    packet: Dict[str, Any] = {
        "header": {
            "type": "neuralis.cognition.capabilities.v1",
            "version": 1,
            "ts": now,
            "ascii_only": True,
            "source": "Neuralis_AI.packet_cognition_capabilities",
        },
        "probe": {
            "bus_topics": {
                "vhw": [
                    "vsd.write",
                    "vsd.flush.prediction",
                    "vhw.core.ready",
                    "telemetry.perf",
                    "telemetry.vqram",
                    "telemetry.qpu",
                    "market.alert",
                    "lane.update",
                    "telemetry.bridge.share",
                    "prediction.data_dirty",
                ],
                "bios": [
                    "bios.init",
                    "boot.complete",
                    "scheduler.task.start",
                    "scheduler.task.rollup",
                    "prediction.archive_complete",
                    "bios.diagnostic.report",
                    "ci.cycle.start",
                    "ci.events.published",
                ],
                "neuralis": [
                    "neuralis.telemetry.update",
                    "ai_processor.pack_complete",
                    "ai_processor.rehydrate_complete",
                    "ai.rehydrated",
                    "pe.rehydrated",
                    "control_center.ui.action",
                ],
            },
            "vsd_keys": {
                "system": [
                    "system/bios_boot_ok",
                    "system/device_snapshot",
                ],
                "fullstate": [
                    "ai/fullstate/path",
                    "ai/fullstate/meta",
                    "pe/fullstate/path",
                    "pe/fullstate/meta",
                    "neuralis/fullstate/path",
                    "neuralis/fullstate/meta",
                    "ai_processor/last_pack_path",
                    "ai_processor/last_load_path",
                ],
                "prediction": [
                    "telemetry/predictions/latest",
                    "telemetry/predictions/{symbol}",
                    "telemetry/metrics/index",
                    "telemetry/metrics/{NET}/current",
                ],
                "neuralis": [
                    "neuralis/last_class",
                    "neuralis/training_log",
                    "telemetry/ai/cognition_summary",
                    "telemetry/ai/governor_status",
                    "telemetry/ai/learning_knobs",
                ],
            },
        },
        "gaps": [
            {
                "id": "parallel_reasoning",
                "title": "Parallel reasoning layer",
                "status": "missing",
                "proposed_vsd_keys": [
                    "neuralis/lanes/index",
                    "neuralis/lanes/{lane}/status",
                    "neuralis/lanes/{lane}/weights",
                    "neuralis/consensus/latest",
                ],
                "proposed_bus_topics": [
                    "neuralis.lane.tick",
                    "neuralis.lanes.update",
                    "neuralis.lanes.consensus",
                ],
            },
            {
                "id": "boundary_cognition",
                "title": "Boundary cognition",
                "status": "missing",
                "proposed_vsd_keys": [
                    "neuralis/boundary/report",
                    "neuralis/boundary/drift_flags",
                    "neuralis/boundary/subsystems_touch",
                ],
                "proposed_bus_topics": [
                    "neuralis.boundary.ready",
                    "neuralis.boundary.alert",
                ],
            },
            {
                "id": "impact_map",
                "title": "Impact-map generation",
                "status": "missing",
                "proposed_vsd_keys": [
                    "neuralis/impact_map/latest",
                    "neuralis/impact_map/history/{ts}",
                ],
                "proposed_bus_topics": [
                    "neuralis.impact.ready",
                ],
            },
            {
                "id": "packet_classifier",
                "title": "Packet classifier",
                "status": "missing",
                "proposed_metadata_fields": [
                    "subsystem",
                    "arch_tags",
                    "lane_affinity",
                    "risk_level",
                    "impact_seeds",
                ],
                "proposed_vsd_keys": [
                    "neuralis/classifier/last_stats",
                ],
                "proposed_bus_topics": [
                    "neuralis.classify.done",
                ],
            },
            {
                "id": "jarvis_pre_auth",
                "title": "Jarvis pre-authorization",
                "status": "missing",
                "proposed_vsd_keys": [
                    "neuralis/auth/pending/{task_id}",
                    "neuralis/auth/approved/{task_id}",
                    "neuralis/auth/denied/{task_id}",
                ],
                "proposed_bus_topics": [
                    "jarvis.auth.request",
                    "jarvis.auth.decision",
                ],
            },
            {
                "id": "scheduler",
                "title": "Cognition scheduler",
                "status": "missing",
                "proposed_bus_topics": [
                    "neuralis.cognition.tick",
                    "scheduler.task.rollup",
                ],
            },
            {
                "id": "unified_bus",
                "title": "Unified event-bus selection",
                "status": "missing",
                "decision": "use VHW.vsd_manager.get_event_bus via Neuralis shim",
            },
        ],
        "ci_checks": [
            "python3 scripts/check_import_cycles.py --strict --focus bios,VHW,miner,core",
            "python3 scripts/check_import_cycles.py --forbid-import Neuralis_AI --strict",
            "python3 scripts/check_import_cycles.py --forbid-import miner --strict --focus prediction_engine",
        ],
        "references": {
            "paths": [
                "VHW/vsd_manager.py",
                "VHW/system_utils.py",
                "bios/event_bus.py",
                "Neuralis_AI/cognition_layer.py",
                "Neuralis_AI/learning_bridge.py",
                "Neuralis_AI/AI_processor.py",
                "Neuralis_AI/ai_core.py",
                "Neuralis_AI/telemetry_adapter.py",
                "neural_object.py",
            ]
        },
    }

    # Compute digest over canonical form
    canon = _canonicalize_json(packet)
    packet["meta"] = {
        "digest": _sha256_ascii(canon),
        "size_bytes": len(canon.encode("ascii", "ignore")),
    }
    return packet


def write_packet(vsd: Optional[Any] = None, path: str = VSD_PATH_CURRENT) -> Dict[str, Any]:
    """
    Build and persist the packet to VSD. Returns the stored packet.
    """
    pkt = build_packet()
    mgr = vsd if vsd is not None else VSDManager()
    try:
        mgr.store(str(path), pkt)
    except Exception:
        # Best-effort: do not raise to avoid breaking callers
        pass
    return pkt


def read_packet(vsd: Optional[Any] = None, path: str = VSD_PATH_CURRENT) -> Dict[str, Any]:
    mgr = vsd if vsd is not None else VSDManager()
    try:
        data = mgr.get(str(path), {})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


__all__ = [
    "build_packet",
    "write_packet",
    "read_packet",
    "VSD_PATH_CURRENT",
]
