# ASCII-ONLY
# Litecoin Scrypt PoW helpers (best-effort)
from __future__ import annotations
from typing import Dict, Any
import struct
import binascii

try:
    import scrypt  # type: ignore
except Exception:
    scrypt = None  # type: ignore


def _to_bytes(hexstr: str) -> bytes:
    s = str(hexstr).lower().replace("0x", "").strip()
    if len(s) % 2:
        s = "0" + s
    try:
        return binascii.unhexlify(s)
    except Exception:
        return b""


def scrypt_hash_1024_1_1_256(data: bytes) -> bytes:
    """
    Compute scrypt_1024_1_1_256(data) -> 32-byte hash.
    Uses python 'scrypt' module as a KDF approximation when available.
    Note: This is a best-effort local validity estimation; pool is authoritative.
    """
    if scrypt is None:
        return b"\x00" * 32
    try:
        # Many bindings expose scrypt.hash(password, salt, N, r, p, buflen)
        # We reuse data as both password and salt here to derive a 32-byte digest.
        return scrypt.hash(data, data, N=1024, r=1, p=1, buflen=32)
    except Exception:
        return b"\x00" * 32


def scrypt_compute_share(job: Dict[str, Any], nonce_int: int) -> Dict[str, Any]:
    """
    Build LTC share fields using scrypt PoW when available.
    Expects job['header_hex'] to be the 80-byte header with nonce placeholder (0).
    """
    header_hex = str(job.get("header_hex", job.get("header", "")))
    header = _to_bytes(header_hex)
    if len(header) < 80:
        # invalid header; return placeholder
        return {
            "nonce": "0x%08x" % (nonce_int & 0xFFFFFFFF),
            "header": header_hex,
            "hash_hex": "0x0",
        }
    # write nonce into last 4 bytes (little-endian)
    hdr = bytearray(header)
    struct.pack_into("<I", hdr, 76, nonce_int & 0xFFFFFFFF)
    digest = scrypt_hash_1024_1_1_256(bytes(hdr))
    return {
        "nonce": "0x%08x" % (nonce_int & 0xFFFFFFFF),
        "header": bytes(hdr).hex(),
        "hash_hex": "0x" + digest.hex(),
    }
