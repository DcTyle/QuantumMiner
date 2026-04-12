# ASCII-ONLY
# RVN KawPoW helpers (best-effort)
from __future__ import annotations
from typing import Dict, Any, Tuple

# Optional backend candidates:
# - pykawpow or kawpow python binding (if available)

try:
    import kawpow  # type: ignore
except Exception:
    kawpow = None  # type: ignore

# Try alternative module name commonly used by some forks
if kawpow is None:
    try:
        import pykawpow as kawpow  # type: ignore
    except Exception:
        kawpow = None  # type: ignore

import binascii

# KawPoW epoch length (blocks). Ravencoin commonly uses 7500 blocks per epoch.
EPOCH_BLOCKS = 7500

def _to_bytes(hexstr: str) -> bytes:
    s = str(hexstr).lower().replace("0x", "").strip()
    if len(s) % 2:
        s = "0" + s
    try:
        return binascii.unhexlify(s)
    except Exception:
        return b""


def _to_hex(b: bytes) -> str:
    try:
        return "0x" + binascii.hexlify(b).decode("ascii")
    except Exception:
        return "0x"


def kawpow_mix_digest_with_epoch(epoch_number: int, header_hex: str, nonce_int: int) -> Tuple[str, str]:
    """
    Compute (mix_digest_hex, result_hash_hex) for KawPoW if backend is present.
    Returns ("0x0", "0x0") when unavailable.
    """
    if kawpow is None:
        return ("0x0", "0x0")
    try:
        header = _to_bytes(header_hex)
        nonce = int(nonce_int) & 0xFFFFFFFF
        res = kawpow.hash(int(epoch_number), header, nonce)
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            mix_b, res_b = res[0], res[1]
            try:
                return (_to_hex(bytes(mix_b)), _to_hex(bytes(res_b)))
            except Exception:
                pass
        return ("0x0", "0x0")
    except Exception:
        return ("0x0", "0x0")


def kawpow_compute_share(job: Dict[str, Any], nonce_int: int) -> Dict[str, Any]:
    # Prefer explicit header_hash if present; else fall back to header_hex/header
    header_hex = str(job.get("header_hash", "") or job.get("header_hex", "") or job.get("header", ""))
    # Best-effort epoch: prefer job.epoch, else derive from height
    epoch = 0
    try:
        if "epoch" in job and str(job["epoch"]).isdigit():
            epoch = int(job["epoch"])  # already epoch
        elif "height" in job and str(job["height"]).isdigit():
            h = int(job["height"]) if int(job["height"]) >= 0 else 0
            epoch = h // EPOCH_BLOCKS
    except Exception:
        epoch = 0
    mix_hex, result_hex = kawpow_mix_digest_with_epoch(epoch, header_hex, nonce_int)
    nonce_hex = "0x%08x" % (nonce_int & 0xFFFFFFFF)
    return {
        "nonce": nonce_hex,
        "header": header_hex,
        "mix_hash": mix_hex,
        "hash_hex": result_hex or job.get("hash_hex", ""),
    }
