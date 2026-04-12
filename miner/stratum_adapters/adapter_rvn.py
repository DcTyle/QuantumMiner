# ASCII-ONLY
# RVN (KawPoW) job adapter
from __future__ import annotations
from typing import Dict, Any, List, Optional
from .adapter_base import CoinAdapter
import os, hashlib, struct, random


def _hex_to_bytes(x: str) -> bytes:
    try:
        s = str(x).replace("0x", "").strip()
        if len(s) % 2:
            s = "0" + s
        return bytes.fromhex(s)
    except Exception:
        return b""


def _dbl_sha256(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def _bits_to_target_hex(nbits_hex: str) -> str:
    """Convert compact bits (nBits) to full target hex."""
    try:
        n = int(nbits_hex, 16)
        exp = (n >> 24) & 0xFF
        mant = n & 0xFFFFFF
        target = mant * (1 << (8 * (exp - 3)))
        s = f"{target:064x}"
        return s
    except Exception:
        return "00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"


def _compute_merkle_root(coinbase_hash_le: bytes, branches_hex: List[str]) -> bytes:
    h = coinbase_hash_le
    for br in branches_hex:
        try:
            b = _hex_to_bytes(br)[::-1]  # branch as little-endian
            h = _dbl_sha256(h + b)
        except Exception:
            continue
    return h

class RVNAdapter(CoinAdapter):
    coin = "RVN"
    extranonce1: Optional[str] = None
    extranonce2_size: int = 4

    def on_response(self, msg: Dict[str, Any]) -> None:
        try:
            res = msg.get("result")
            # mining.subscribe response usually: [ subscriptions, extranonce1, extranonce2_size ]
            if isinstance(res, list) and len(res) >= 3 and isinstance(res[1], str):
                self.extranonce1 = str(res[1])
                try:
                    self.extranonce2_size = int(res[2]) if int(res[2]) > 0 else 4
                except Exception:
                    self.extranonce2_size = 4
        except Exception:
            pass

    def convert_job(self, job_raw: Dict[str, Any]) -> Dict[str, Any]:
        params = job_raw.get("params", []) if isinstance(job_raw, dict) else []
        job_id = params[0] if len(params) > 0 else ""
        # RVN notify typically: [job_id, prevhash, coinb1, coinb2, merkle, version, nbits, ntime, clean_jobs]
        prevhash_hex = params[1] if len(params) > 1 else ""
        coinb1 = params[2] if len(params) > 2 else ""
        coinb2 = params[3] if len(params) > 3 else ""
        merkle_branch = params[4] if len(params) > 4 else []
        version_hex = params[5] if len(params) > 5 else "20000000"
        nbits = params[6] if len(params) > 6 else "1d00ffff"
        ntime = params[7] if len(params) > 7 else "00000000"
        # Some pools include height as an extra param or dict entry
        height = None
        try:
            # Common extension: params[9] or dict with 'height'/'blockheight'
            if len(params) > 9:
                cand = params[9]
                if isinstance(cand, (int,)):
                    height = int(cand)
                elif isinstance(cand, str) and cand.isdigit():
                    height = int(cand)
            if height is None:
                for p in params:
                    if isinstance(p, dict):
                        if "height" in p and str(p["height"]).isdigit():
                            height = int(p["height"])
                            break
                        if "blockheight" in p and str(p["blockheight"]).isdigit():
                            height = int(p["blockheight"])
                            break
        except Exception:
            height = None
        # Derive epoch from height if available (KawPoW epoch ~ every 7500 blocks)
        epoch = None
        try:
            if isinstance(height, int) and height >= 0:
                epoch = height // 7500
        except Exception:
            epoch = None
        # Build extranonce2
        en1 = self.extranonce1 or ""
        en2_size = max(1, int(self.extranonce2_size or 4))
        en2_val = random.getrandbits(en2_size * 8)
        en2 = f"{en2_val:0{en2_size*2}x}"

        # Assemble coinbase
        coinbase_hex = f"{coinb1}{en1}{en2}{coinb2}"
        coinbase_bytes = _hex_to_bytes(coinbase_hex)
        cb_hash = _dbl_sha256(coinbase_bytes)
        cb_hash_le = cb_hash[::-1]

        # Merkle root
        branches = merkle_branch if isinstance(merkle_branch, list) else []
        mr_le = _compute_merkle_root(cb_hash_le, [str(x) for x in branches])
        mr_be = mr_le[::-1]

        # Block header (80 bytes) for telemetry; PoW uses kawpow header hash
        try:
            ver = int(str(version_hex), 16)
        except Exception:
            ver = 0x20000000
        header = b"".join([
            struct.pack("<I", ver),
            _hex_to_bytes(prevhash_hex)[::-1],
            mr_be[::-1],  # ensure correct endian in header (little-endian on wire)
            bytes.fromhex(ntime)[::-1] if isinstance(ntime, str) else struct.pack("<I", int(ntime)),
            bytes.fromhex(nbits)[::-1] if isinstance(nbits, str) else struct.pack("<I", int(nbits)),
            struct.pack("<I", 0),  # nonce placeholder
        ])

        # Derive a 32-byte header_hash for kawpow backend. Use SHA3-256 as a fallback.
        try:
            import hashlib as _hh
            if hasattr(_hh, "sha3_256"):
                header_hash = _hh.sha3_256(header).digest()
            else:
                header_hash = _dbl_sha256(header)
        except Exception:
            header_hash = _dbl_sha256(header)

        out = super().convert_job(job_raw)
        out.update({
            "job_id": str(job_id or out.get("job_id", "RVN_job")),
            "nbits": str(nbits),
            "ntime": str(ntime),
            "header_hex": header.hex(),
            "header_hash": header_hash.hex(),
            "prevhash": str(prevhash_hex),
            "merkle_root": mr_be.hex(),
            "coinbase": coinbase_hex,
            "extranonce1": en1,
            "extranonce2_size": en2_size,
            "extranonce2": en2,
            "merkle_branch": branches,
            "target": _bits_to_target_hex(str(nbits)),
        })
        if height is not None:
            out["height"] = height
        if epoch is not None:
            out["epoch"] = epoch
        return out
