# ASCII-ONLY
# Bitcoin SHA256d PoW helpers (best-effort)
from __future__ import annotations
from typing import Dict, Any
import struct
import hashlib
import binascii


def _to_bytes(hexstr: str) -> bytes:
    s = str(hexstr).lower().replace("0x", "").strip()
    if len(s) % 2:
        s = "0" + s
    try:
        return binascii.unhexlify(s)
    except Exception:
        return b""


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def sha256d_compute_share(job: Dict[str, Any], nonce_int: int) -> Dict[str, Any]:
    """
    Build BTC share using header with nonce injected (little-endian) and compute sha256d.
    Expects job['header_hex'] to be 80-byte header with nonce placeholder at the end.
    """
    header_hex = str(job.get("header_hex", job.get("header", "")))
    header = _to_bytes(header_hex)
    if len(header) < 80:
        # invalid header
        digest = b"\x00" * 32
        return {
            "nonce": "0x%08x" % (nonce_int & 0xFFFFFFFF),
            "header": header_hex,
            "hash_hex": digest.hex(),
        }
    hdr = bytearray(header)
    # write nonce into bytes 76..80 (LE)
    struct.pack_into("<I", hdr, 76, nonce_int & 0xFFFFFFFF)
    digest = sha256d(bytes(hdr))
    return {
        "nonce": "0x%08x" % (nonce_int & 0xFFFFFFFF),
        "header": bytes(hdr).hex(),
        "hash_hex": digest.hex(),
    }
