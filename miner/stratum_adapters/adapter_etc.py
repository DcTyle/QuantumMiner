# ASCII-ONLY
# ETC (Etchash) job adapter
from __future__ import annotations
from typing import Dict, Any
from .adapter_base import CoinAdapter

class ETCAdapter(CoinAdapter):
    coin = "ETC"

    def convert_job(self, job_raw: Dict[str, Any]) -> Dict[str, Any]:
        params = job_raw.get("params", []) if isinstance(job_raw, dict) else []
        job_id = params[0] if len(params) > 0 else ""
        header_hash = params[1] if len(params) > 1 else ""
        seed_hash = params[2] if len(params) > 2 else ""
        target_hex = params[3] if len(params) > 3 else job_raw.get("target", "")
        out = super().convert_job(job_raw)
        out.update({
            "job_id": str(job_id or out.get("job_id", "ETC_job")),
            "header_hash": str(header_hash),
            "seed_hash": str(seed_hash),
            "target": str(target_hex or out.get("target", "")),
        })
        return out
