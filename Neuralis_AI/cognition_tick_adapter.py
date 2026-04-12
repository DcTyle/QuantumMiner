"""
ASCII-ONLY
Neuralis Cognition Tick Adapter (Neuralis-scoped)

Purpose
-------
Provide a minimal, modular bridge that listens to BIOS scheduler ticks and
updates Neuralis VSD-hosted cognition packets using an enum-based domain map.

Scope
-----
- Neuralis-only module; no imports from miner or prediction_engine.
- Subscribes to BIOS scheduler topics via bios.event_bus when available.
- Falls back to VHW bus if BIOS bus import fails (no-op safe).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time
from enum import Enum

try:
    from bios.event_bus import get_event_bus  # BIOS scheduler emits here
except Exception:
    try:
        from VHW.vsd_manager import get_event_bus  # fallback to unified bus
    except Exception:
        def get_event_bus():  # type: ignore
            class _NoBus:
                def subscribe(self, *a, **k): return None
                def publish(self, *a, **k): return None
            return _NoBus()

try:
    from VHW.vsd_manager import VSD as _GLOBAL_VSD
    from VHW.vsd_manager import VSDManager
except Exception:
    _GLOBAL_VSD = None  # type: ignore
    class VSDManager:  # type: ignore
        def __init__(self): self._m: Dict[str, Any] = {}
        def get(self, k: str, d: Any=None): return self._m.get(str(k), d)
        def store(self, k: str, v: Any): self._m[str(k)] = v

from Neuralis_AI.packet_cognition_capabilities import (
    write_packet,
    read_packet,
    VSD_PATH_CURRENT,
)
from Neuralis_AI.cognition_summary import build_and_store_summary
from Neuralis_AI.domain_specs import get_domain_spec
from Neuralis_AI.cognition_history import append_history


class CognitionDomain(Enum):
    SYSTEM_COGNITION = "systemcognition"
    NEURAL_NETWORK = "neural network"  # requested literal value


# Domain specs centralized in Neuralis_AI.domain_specs


def _ensure_domain_meta(pkt: Dict[str, Any], domain: CognitionDomain) -> None:
    hdr = pkt.setdefault("header", {})
    hdr["domain"] = domain.value
    meta = pkt.setdefault("meta", {})
    dkey = "domain_meta"
    dm = meta.get(dkey)
    if not isinstance(dm, dict):
        meta[dkey] = {}
        dm = meta[dkey]
    if domain.value not in dm:
        spec = get_domain_spec(domain.value)
        base = spec.defaults if spec else {"last_tick_ts": 0.0}
        dm[domain.value] = {
            "last_tick_ts": float(base.get("last_tick_ts", 0.0)),
            "tick_source": str(base.get("tick_source", "")),
            "notes": str(base.get("notes", "")),
        }


def _update_domain_meta(vsd: VSDManager, domain: CognitionDomain, payload: Dict[str, Any]) -> None:
    pkt = read_packet(vsd)
    if not isinstance(pkt, dict) or not pkt:
        pkt = write_packet(vsd)
    _ensure_domain_meta(pkt, domain)
    try:
        dm = pkt["meta"]["domain_meta"][domain.value]
        now_ts = time.time()
        dm["last_tick_ts"] = now_ts
        spec = get_domain_spec(domain.value)
        if spec and isinstance(payload, dict):
            normalized = spec.normalize(payload)
            dm.update(normalized)
            try:
                append_history(vsd, domain.value, {"ts": now_ts, **normalized})
            except Exception:
                pass
    except Exception:
        pass
    # Persist back to VSD
    try:
        vsd.store(VSD_PATH_CURRENT, pkt)
    except Exception:
        pass
    # Maintain a rolled-up cognition summary packet alongside capabilities.
    try:
        build_and_store_summary(vsd)
    except Exception:
        pass


def register(domain: CognitionDomain = CognitionDomain.NEURAL_NETWORK) -> bool:
    """
    Subscribe to BIOS scheduler tick topics and update the VSD packet.
    Returns True if subscription succeeded.
    """
    try:
        bus = get_event_bus()
        vsd = _GLOBAL_VSD if _GLOBAL_VSD is not None else VSDManager()

        def _handler(payload: Dict[str, Any]) -> None:
            try:
                _update_domain_meta(vsd, domain, dict(payload or {}))
            except Exception:
                pass

        # Common BIOS scheduler topics observed in repo
        topics = [
            "scheduler.task.rollup",
            "scheduler.task.start",
        ]
        for t in topics:
            try:
                bus.subscribe(t, _handler, priority=0, once=False)
            except TypeError:
                # Fallback for bus implementations without named args
                try:
                    bus.subscribe(t, _handler)
                except Exception:
                    pass
        return True
    except Exception:
        return False


__all__ = [
    "CognitionDomain",
    "register",
]
