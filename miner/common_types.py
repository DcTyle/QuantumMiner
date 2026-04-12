# ASCII-ONLY FILE
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class QMJob:
    """
    Canonical mining job passed through the miner pipeline.

    Fields
    - network: upper-case network name (e.g., ETC, RVN, LTC, BTC)
    - header: hex string header or header_hash depending on algo (no spaces)
    - seed: hex seed/seed_hash when applicable (Etchash/Autolykos); else ""
    - target: hex difficulty target (0x prefix optional)
    - height: integer block height or epoch; -1 if unknown
    - extra: algorithm-specific extras (dict)
    - raw_payload: original raw job payload as received from network/client
    """

    network: str
    header: str
    seed: str
    target: str
    height: int
    extra: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)


def _hex(s: str) -> str:
    try:
        ss = str(s or "").strip()
        return ss[2:] if ss.startswith("0x") else ss
    except Exception:
        return ""


def qmjob_from_dict(network: str, d: Dict[str, Any]) -> QMJob:
    """Best-effort conversion from a loose dict to QMJob."""
    net = str(network or d.get("network", "")).upper()
    header = _hex(str(d.get("header_hex", d.get("header_hash", d.get("header", "")))))
    seed = _hex(str(d.get("seed_hash", d.get("seed", ""))))
    target = str(d.get("target", ""))
    try:
        h = int(d.get("height", d.get("epoch", -1)) or -1)
    except Exception:
        h = -1
    extra = {}
    try:
        for k in ("job_id", "extranonce2", "ntime", "epoch"):
            if k in d:
                extra[k] = d.get(k)
    except Exception:
        pass
    return QMJob(network=net, header=header, seed=seed, target=target, height=int(h), extra=extra, raw_payload=dict(d))
