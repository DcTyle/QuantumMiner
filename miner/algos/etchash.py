# ASCII-ONLY
# ETC Etchash helpers (best-effort)
from __future__ import annotations
from typing import Dict, Any, Tuple

# We try multiple backends; at least one must be available
# 1) pyethash (classic ethash)
# 2) ethash (python binding)

try:
    import pyethash  # type: ignore
except Exception:
    pyethash = None  # type: ignore

try:
    import ethash  # type: ignore
except Exception:
    ethash = None  # type: ignore

import binascii


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


def _match_epoch_from_seed(seed_hex: str) -> int:
    """
    Derive epoch number from a seed hash by comparing against pyethash.get_seedhash
    for both ETH (30000) and ETC (60000) epoch sizes. Returns -1 if unknown.
    """
    if pyethash is None:
        return -1
    target = _to_bytes(seed_hex)
    if not target:
        return -1
    ETH_EPOCH = 30000
    ETC_EPOCH = 60000
    MAX_EPOCH = 4096
    for epoch in range(MAX_EPOCH):
        try:
            if pyethash.get_seedhash(epoch * ETH_EPOCH) == target:
                return epoch
        except Exception:
            break
    for epoch in range(MAX_EPOCH):
        try:
            if pyethash.get_seedhash(epoch * ETC_EPOCH) == target:
                return epoch
        except Exception:
            break
    return -1


def etchash_mix_digest(header_hash_hex: str, nonce_hex: str, seed_hash_hex: str) -> Tuple[str, str]:
    """
    Compute (mix_digest_hex, result_hash_hex) for Etchash using any available backend.
    Returns (mix_digest_hex, result_hash_hex). If no backend, returns ("0x0", "0x0").
    """
    header = _to_bytes(header_hash_hex)
    nonce = _to_bytes(nonce_hex)
    if len(nonce) == 0:
        raise ValueError("etchash_mix_digest: nonce is required")

    # Prefer ethash binding if available and we can infer epoch
    if ethash is not None:
        try:
            epoch = _match_epoch_from_seed(seed_hash_hex)
            if epoch >= 0:
                mix, result = ethash.hash(epoch, header, int(nonce_hex, 16))
                return (_to_hex(mix), _to_hex(result))
        except Exception:
            pass

    # pyethash path if we can approximate a block height from epoch
    if pyethash is not None:
        try:
            epoch = _match_epoch_from_seed(seed_hash_hex)
            if epoch >= 0:
                # Use ETH epoch length for cache sizing; hashimoto_light uses block number
                block_number = epoch * 30000
                out = pyethash.hashimoto_light(block_number, header, int(nonce_hex, 16))
                mix = out.get('mix digest') or out.get('mix_digest') or b""
                res = out.get('result') or b""
                return (_to_hex(mix if isinstance(mix, (bytes, bytearray)) else b""),
                        _to_hex(res if isinstance(res, (bytes, bytearray)) else b""))
        except Exception:
            pass

    # No backend available
    raise RuntimeError("etchash_mix_digest: no available backend (ethash/pyethash)")


def etchash_compute_share(job: Dict[str, Any], nonce_int: int) -> Dict[str, Any]:
    """
    Build ETC share fields from job and nonce using etchash backend if available.
    Ensures fields: nonce (0x...), header_hash, mix_hash.
    """
    header_hash = str(job.get("header_hash", job.get("header_hex", "")))
    seed_hash = str(job.get("seed_hash", ""))
    nonce_hex = "0x%08x" % (nonce_int & 0xFFFFFFFF)
    mix_hex, result_hex = etchash_mix_digest(header_hash, nonce_hex, seed_hash)
    return {
        "nonce": nonce_hex,
        "header_hash": header_hash,
        "mix_hash": mix_hex,
        "hash_hex": result_hex or job.get("hash_hex", ""),
    }
