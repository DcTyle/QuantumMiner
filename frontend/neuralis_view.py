from __future__ import annotations

from typing import Any, Dict, List

from VHW.vsd_manager import VSDManager
from Neuralis_AI.context_map import build_context_graph
from Neuralis_AI.packet_cognition_capabilities import read_packet
from Neuralis_AI.cognition_summary import build_summary
from Neuralis_AI.cognition_history import get_history


def _get_vsd(vsd: Any | None = None) -> VSDManager | None:
    if vsd is not None:
        return vsd  # type: ignore[return-value]
    try:
        return VSDManager.global_instance()  # type: ignore[attr-defined]
    except Exception:
        return None


def get_context_graph(vsd: Any | None = None) -> Dict[str, Any]:
    """Return the static Neuralis context graph.

    Pure view-model function; callers decide how to render the result.
    """
    return build_context_graph()


def get_cognition_capabilities(vsd: Any | None = None) -> Dict[str, Any]:
    vsd_obj = _get_vsd(vsd)
    if vsd_obj is None:
        return {}
    try:
        pkt = read_packet(vsd_obj) or {}
    except Exception:
        pkt = {}
    if not isinstance(pkt, dict):
        return {}
    return pkt


def get_cognition_summary(vsd: Any | None = None) -> Dict[str, Any]:
    vsd_obj = _get_vsd(vsd)
    if vsd_obj is None:
        return {}
    try:
        summary = build_summary(vsd_obj) or {}
    except Exception:
        summary = {}
    if not isinstance(summary, dict):
        return {}
    return summary


def get_all_domain_history(vsd: Any | None = None, limit: int = 16) -> Dict[str, List[Dict[str, Any]]]:
    vsd_obj = _get_vsd(vsd)
    if vsd_obj is None:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        pkt = read_packet(vsd_obj) or {}
        meta = (pkt.get("meta", {}) or {}).get("domain_meta", {}) or {}
        if not isinstance(meta, dict):
            return {}
        for domain in meta.keys():
            try:
                hist = get_history(vsd_obj, domain, limit=limit)
            except Exception:
                hist = []
            if isinstance(hist, list):
                out[str(domain)] = hist
    except Exception:
        return {}
    return out
