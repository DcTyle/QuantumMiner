# ASCII-ONLY FILE
# Ethereum Classic ETChash Stratum Client (JSON-RPC)
import socket
import json
import time
import threading
import math
from collections import deque
from typing import Any, Dict
from miner.common_types import QMJob, qmjob_from_dict

from . import NetSpecBase, NetID
from miner.algos.etchash import etchash_compute_share as _etchash

class ETCPoolClient:
    def __init__(self, endpoint, timeout=5):
        self.endpoint = endpoint
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()

    def _connect(self):
        host, port = self.endpoint.split(":")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((host, int(port)))

    def _send(self, method, params):
        msg = {
            "id": int(time.time()),
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        raw = (json.dumps(msg) + "\n").encode("ascii")
        with self.lock:
            self.sock.sendall(raw)
            resp_raw = self.sock.recv(4096).decode("ascii")
        try:
            return json.loads(resp_raw)
        except Exception:
            return {"error": "invalid_json"}

    def _ensure(self):
        if self.sock:
            return True
        try:
            self._connect()
            return True
        except Exception:
            return False

    def submit_share(self, network, share):
        if not self._ensure():
            return False

        # ETChash work submission fields
        nonce = share.get("nonce")
        header = share.get("header_hash")
        mix = share.get("mix_hash")

        params = [nonce, header, mix]
        resp = self._send("eth_submitWork", params)

        return bool(resp.get("result", False))


class NetSpec_ETC(NetSpecBase):
    network = "ETC"
    enum_id = NetID.ETC
    _share_lat_ms = deque(maxlen=10)
    _job_intervals = deque(maxlen=10)
    _last_job_key = ""
    _last_change_ts = 0.0

    def hash_fn(self, job: Any, nonce: int) -> Dict[str, str]:
        # Normalize to dict for backend helper
        if isinstance(job, QMJob):
            j = {
                "header_hash": job.header,
                "seed_hash": job.seed,
                "target": job.target,
            }
        else:
            j = dict(job or {})
        extra = _etchash(j, nonce)
        return {
            "powhash": str(extra.get("hash_hex", "")),
            "mixhash": str(extra.get("mix_hash", "")),
            "header": str(j.get("header_hash", j.get("header", "")))
        }

    def target_fn(self, job: Any, powhash_hex: str) -> bool:
        try:
            th = str(job.target if isinstance(job, QMJob) else job.get("target", ""))
            if th.startswith("0x"):
                th = th[2:]
            return int(powhash_hex or "0", 16) <= int(th or "0", 16)
        except Exception:
            return False

    def network_to_system_fn(self, raw: Dict[str, Any]) -> QMJob:
        params = []
        try:
            params = list(raw.get("params", [])) if isinstance(raw, dict) else []
        except Exception:
            params = []
        header_hex = ""
        seed_hex = ""
        target_hex = ""
        height = -1
        # Best-effort scan
        try:
            for p in params:
                if isinstance(p, str):
                    s = p.lower().lstrip("0x")
                    if len(s) == 64 and all(c in "0123456789abcdef" for c in s):
                        if not header_hex:
                            header_hex = s
                        elif not seed_hex:
                            seed_hex = s
                elif isinstance(p, dict):
                    if not target_hex and "target" in p:
                        target_hex = str(p.get("target", ""))
                    if height < 0 and "height" in p:
                        height = int(p.get("height", -1) or -1)
        except Exception:
            pass
        d = {"header": header_hex, "seed": seed_hex, "target": target_hex, "height": height}
        return qmjob_from_dict("ETC", d | {"raw": raw})

    def system_to_network_fn(self, qmshare: Any) -> Dict[str, Any]:
        sp = qmshare.system_payload
        job = sp.get("job")
        return {
            "network": "ETC",
            "nonce": "%#x" % int(qmshare.nonce),
            "header": (job.header if isinstance(job, QMJob) else (job or {}).get("header_hash")),
            "mix": qmshare.mixhash or ({} if not isinstance(job, dict) else job).get("mix_hash")
        }

    def compute_batch_size(self, job: Any) -> int:
        base = 128
        now = time.time()

        key = ""
        try:
            if isinstance(job, QMJob):
                key = job.header or str(job.height)
            else:
                key = str(job.get("job_id") or job.get("header_hash") or job.get("header") or job.get("height") or job.get("epoch") or "")
        except Exception:
            key = ""
        if key != self._last_job_key:
            if self._last_change_ts > 0:
                self._job_intervals.append(max(0.001, now - self._last_change_ts))
            self._last_job_key = key
            self._last_change_ts = now

        try:
            sp = job.get("system_payload", {}) if isinstance(job, dict) else {}
            lat = float(sp.get("last_share_latency_ms", sp.get("share_latency_ms", job.get("last_share_latency_ms", 0))))
            if lat > 0:
                self._share_lat_ms.append(lat)
        except Exception:
            pass

        diff_factor = 0.5
        try:
            t_hex = str(job.get("target", ""))
            if t_hex.startswith("0x"):
                t_hex = t_hex[2:]
            t_val = int(t_hex or "0", 16)
            max256 = (1 << 256) - 1
            t_val = min(max256, max(1, t_val))
            diff_factor = 1.0 - (t_val / float(max256))
        except Exception:
            diff_factor = 0.5

        if self._job_intervals:
            avg_int = sum(self._job_intervals) / len(self._job_intervals)
            if avg_int <= 1.0:
                stability = 0.6
            elif avg_int <= 3.0:
                stability = 0.8
            else:
                stability = 1.0
        else:
            stability = 1.0

        if self._share_lat_ms:
            avg_ms = sum(self._share_lat_ms) / len(self._share_lat_ms)
            if avg_ms <= 200.0:
                lat_factor = 1.25
            elif avg_ms <= 800.0:
                lat_factor = 1.0
            else:
                lat_factor = 0.75
        else:
            lat_factor = 1.0

        scale = (0.75 + 0.75 * max(0.0, min(1.0, diff_factor))) * stability * lat_factor
        batch = int(base * scale)
        batch = max(16, min(batch, min(512, base * 4)))
        return batch
