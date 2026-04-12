from typing import Any, Dict, List

from Neuralis_AI.packet_cognition_capabilities import VSD_PATH_CURRENT
from Neuralis_AI.cognition_history import get_history


def describe_cognition_vsd(vsd: Any, domains: List[str] = None, history_limit: int = 8) -> Dict[str, Any]:
    """Return a summary of cognition-related VSD state.

    This function is read-only and boundary-safe. It assumes `vsd` exposes
    `load(path)` and does not mutate any state.
    """
    summary: Dict[str, Any] = {"paths": {}, "history": {}}
    try:
        pkt = vsd.load(VSD_PATH_CURRENT)  # type: ignore[attr-defined]
    except Exception:
        pkt = None
    summary["paths"][VSD_PATH_CURRENT] = bool(pkt)

    meta = {}
    try:
        if isinstance(pkt, dict):
            meta = pkt.get("meta", {}) or {}
    except Exception:
        meta = {}

    dm = {}
    try:
        dm = meta.get("domain_meta", {}) or {}
    except Exception:
        dm = {}

    if domains is None:
        domains = sorted(dm.keys())

    for name in domains:
        try:
            hist = get_history(vsd, name, limit=history_limit)
        except Exception:
            hist = []
        summary["history"][name] = hist

    return summary
