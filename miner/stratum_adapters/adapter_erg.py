# ASCII-ONLY
# ERG (Autolykos) job adapter (telemetry-focused placeholder)
from __future__ import annotations
from typing import Dict, Any
from .adapter_base import CoinAdapter


class ERGAdapter(CoinAdapter):
    coin = "ERG"

    def convert_job(self, job_raw: Dict[str, Any]) -> Dict[str, Any]:
        params = job_raw.get("params", []) if isinstance(job_raw, dict) else []
        job_id = params[0] if len(params) > 0 else ""
        # Autolykos has distinct fields; we capture minimally for telemetry
        out = super().convert_job(job_raw)
        out.update({
            "job_id": str(job_id or out.get("job_id", "ERG_job")),
        })
        return out
