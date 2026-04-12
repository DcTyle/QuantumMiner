from typing import Any, Dict

from Neuralis_AI.packet_cognition_capabilities import (
    VSD_PATH_CURRENT,
    read_packet,
)
from Neuralis_AI.cognition_history import get_history
from Neuralis_AI.classifier_boundary import classify_boundary_signal

SUMMARY_PATH = "neuralis/packets/cognition_summary/v1/current"


def build_summary(vsd: Any) -> Dict[str, Any]:
    """Build an in-memory cognition summary from capabilities + history.

    This is read-only with respect to external subsystems; it only interacts
    with VSD and Neuralis modules.
    """
    pkt = read_packet(vsd)
    meta = ((pkt or {}).get("meta", {}) or {})
    dm = meta.get("domain_meta", {}) or {}

    summary: Dict[str, Any] = {
        "domains": {},
        "boundary": {},
    }

    for domain, dmeta in dm.items():
        try:
            history = get_history(vsd, domain, limit=8)
        except Exception:
            history = []
        summary["domains"][domain] = {
            "meta": dict(dmeta) if isinstance(dmeta, dict) else {},
            "recent": history,
        }

    # Boundary classifier can use VSD if it needs to later
    try:
        boundary = classify_boundary_signal(vsd, {})
    except Exception:
        boundary = {"decision": "allow", "reason": "classifier error"}
    summary["boundary"] = boundary

    return summary


def build_and_store_summary(vsd: Any) -> Dict[str, Any]:
    summary = build_summary(vsd)
    try:
        vsd.store(SUMMARY_PATH, summary)
    except Exception:
        pass
    return summary
