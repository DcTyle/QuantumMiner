from typing import Any, Dict

# Boundary cognition classifier stub.
# Does NOT import miner or prediction_engine; reads signals indirectly via VSD
# if needed in the future.

DEFAULT_POLICY = "strict"


def classify_boundary_signal(vsd: Any, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = dict(overrides or {})
    policy = str(cfg.get("policy", DEFAULT_POLICY))
    decision = "allow"
    reason = "no violations detected"
    # Placeholder deterministic rule: advisory never blocks; strict may block
    # on a synthetic guard_trip flag supplied via overrides.
    if policy == "strict":
        guard_trip = bool(cfg.get("guard_trip", False))
        if guard_trip:
            decision = "deny"
            reason = "strict policy guard fired"
    return {
        "policy": policy,
        "decision": decision,
        "reason": reason,
    }
