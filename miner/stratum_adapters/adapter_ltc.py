# ASCII-ONLY
# LTC Stratum job adapter (Stratum V1 Scrypt)
from __future__ import annotations
from typing import Dict, Any, List, Optional
from .adapter_base import CoinAdapter
import hashlib, struct, random


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
    try:
        n = int(nbits_hex, 16)
        exp = (n >> 24) & 0xFF
        mant = n & 0xFFFFFF
        target = mant * (1 << (8 * (exp - 3)))
        return f"{target:064x}"
    except Exception:
        return "00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"


def _compute_merkle_root(coinbase_hash_le: bytes, branches_hex: List[str]) -> bytes:
    h = coinbase_hash_le
    for br in branches_hex:
        try:
            b = _hex_to_bytes(br)[::-1]
            h = _dbl_sha256(h + b)
        except Exception:
            continue
    return h

class LTCAdapter(CoinAdapter):
    coin = "LTC"

    def __init__(self) -> None:
        self.extranonce1: Optional[str] = None
        self.extranonce2_size: int = 4

    def on_response(self, msg: Dict[str, Any]) -> None:
        try:
            res = msg.get("result")
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
        prevhash_hex = params[1] if len(params) > 1 else ""
        coinb1 = params[2] if len(params) > 2 else ""
        coinb2 = params[3] if len(params) > 3 else ""
        merkle_branch = params[4] if len(params) > 4 else []
        version_hex = params[5] if len(params) > 5 else "20000000"
        nbits = params[6] if len(params) > 6 else "1d00ffff"
        ntime = params[7] if len(params) > 7 else "00000000"

        # Capture extranonce via subscribe in on_response (handled in BTC adapter pattern)
        # For LTC, we construct coinbase and header similarly to BTC (Scrypt PoW)
        # Generate extranonce2
        en1 = getattr(self, "extranonce1", "")
        en2_size = int(getattr(self, "extranonce2_size", 4) or 4)
        en2_size = max(1, en2_size)
        en2_val = random.getrandbits(en2_size * 8)
        en2 = f"{en2_val:0{en2_size*2}x}"

        coinbase_hex = f"{coinb1}{en1}{en2}{coinb2}"
        cb_hash = _dbl_sha256(_hex_to_bytes(coinbase_hex))
        mr_le = _compute_merkle_root(cb_hash[::-1], [str(x) for x in (merkle_branch or [])])
        mr_be = mr_le[::-1]

        try:
            ver = int(str(version_hex), 16)
        except Exception:
            ver = 0x20000000
        header = b"".join([
            struct.pack("<I", ver),
            _hex_to_bytes(prevhash_hex)[::-1],
            mr_be[::-1],
            bytes.fromhex(ntime)[::-1] if isinstance(ntime, str) else struct.pack("<I", int(ntime)),
            bytes.fromhex(nbits)[::-1] if isinstance(nbits, str) else struct.pack("<I", int(nbits)),
            struct.pack("<I", 0),
        ])

        out = super().convert_job(job_raw)
        out.update({
            "job_id": str(job_id or out.get("job_id", "LTC_job")),
            "nbits": str(nbits),
            "ntime": str(ntime),
            "extranonce1": en1,
            "extranonce2_size": en2_size,
            "extranonce2": en2,
            "coinbase": coinbase_hex,
            "merkle_branch": merkle_branch or [],
            "merkle_root": mr_be.hex(),
            "header_hex": header.hex(),
            "prevhash": str(prevhash_hex),
            "target": _bits_to_target_hex(str(nbits)),
        })
        return out
