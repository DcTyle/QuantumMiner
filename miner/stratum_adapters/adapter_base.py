# ASCII-ONLY
# Base interface for coin-specific job conversions
from __future__ import annotations
from typing import Dict, Any
import time

class CoinAdapter:
    coin: str = "GENERIC"

    def on_response(self, msg: Dict[str, Any]) -> None:
        """
        Optional hook for handling stratum 'response' objects (e.g., subscribe/authorize
        results to capture extranonce sizes, etc.). Default no-op.
        """
        return

    def convert_job(self, job_raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a generic stratum job_raw into the engine's job dict.
        Default: minimal fields with conservative defaults.
        """
        params = job_raw.get("params", []) if isinstance(job_raw, dict) else []
        job_id = ""
        header_hex = ""
        target_hex = job_raw.get("target", "")
        try:
            if params:
                job_id = str(params[0])
            for p in params:
                if isinstance(p, str) and len(p) >= 64 and all(c in "0123456789abcdefABCDEF" for c in p):
                    header_hex = p
                    break
        except Exception:
            pass
        out: Dict[str, Any] = {
            "network": self.coin,
            "job_id": job_id or (self.coin + "_job"),
            "target": target_hex or "00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
            "header_hex": header_hex,
            "received_at": float(job_raw.get("received_at", time.time()))
        }
        try:
            if header_hex:
                out["header"] = bytes.fromhex(header_hex)
        except Exception:
            out["header"] = b""
        return out
