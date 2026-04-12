# ASCII-ONLY FILE
# Ravencoin KawPoW Stratum Client
import socket
import json
import time
import threading
import math
from collections import deque
from typing import Any, Dict
from miner.common_types import QMJob, qmjob_from_dict

from . import NetSpecBase, NetID
from miner.algos.kawpow import kawpow_compute_share as _kawpow

class RVNPoolClient:
    def __init__(self, endpoint, username="worker", password="x", timeout=5):
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.timeout = timeout

        self.sock = None
        self.lock = threading.Lock()
        self.subscribed = False
        self.authorized = False

    def _connect(self):
        host, port = self.endpoint.split(":")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((host, int(port)))

    def _send(self, method, params):
        msg = {
            "id": int(time.time()),
            "method": method,
            "params": params
        }
        raw = (json.dumps(msg) + "\n").encode("ascii")
        with self.lock:
            self.sock.sendall(raw)
            resp = self.sock.recv(4096).decode("ascii")
        try:
            return json.loads(resp)
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

    def _subscribe(self):
        resp = self._send("mining.subscribe", [])
        if "result" in resp:
            self.subscribed = True
        return self.subscribed

    def _authorize(self):
        resp = self._send("mining.authorize", [self.username, self.password])
        if resp.get("result") is True:
            self.authorized = True
        return self.authorized

    def submit_share(self, network, share):
        if not self._ensure():
            return False
        if not self.subscribed:
            if not self._subscribe():
                return False
        if not self.authorized:
            if not self._authorize():
                return False

        job_id = share.get("job_id")
        nonce = share.get("nonce")
        header = share.get("header")
        mix = share.get("mix_hash")

        params = [
            self.username,
            job_id,
            nonce,
            header,
            mix
        ]

        resp = self._send("mining.submit", params)
        return bool(resp.get("result", False))


class NetSpec_RVN(NetSpecBase):
    network = "RVN"
    enum_id = NetID.RVN
    _share_lat_ms = deque(maxlen=10)
    _job_intervals = deque(maxlen=10)
    _last_job_key = ""
    _last_change_ts = 0.0

    def hash_fn(self, job: Any, nonce: int) -> Dict[str, str]:
        j = {"header_hex": job.header, "epoch": job.height} if isinstance(job, QMJob) else dict(job or {})
        extra = _kawpow(j, nonce)
        return {
            "powhash": str(extra.get("hash_hex", "")),
            "mixhash": str(extra.get("mix_hash", "")),
            "header": (job.header if isinstance(job, QMJob) else str(j.get("header", j.get("header_hex", ""))))
        }

    def target_fn(self, job: Any, powhash_hex: str) -> bool:
        try:
            th = str(job.target if isinstance(job, QMJob) else job.get("target", ""))
            if th.startswith("0x"):
                th = th[2:]
            return int(powhash_hex or "", 16) <= int(th or "0", 16)
        except Exception:
            return False

    def network_to_system_fn(self, raw: Dict[str, Any]) -> QMJob:
        return qmjob_from_dict("RVN", {"raw": raw})

    def system_to_network_fn(self, qmshare: Any) -> Dict[str, Any]:
        sp = qmshare.system_payload
        job = sp.get("job")
        return {
            "network": "RVN",
            "job_id": (job.extra.get("job_id") if isinstance(job, QMJob) else (job or {}).get("job_id")),
            "nonce": "%#x" % int(qmshare.nonce),
            "header": (job.header if isinstance(job, QMJob) else (job or {}).get("header")),
            "mix_hash": qmshare.mixhash or ((job or {}).get("mix_hash") if isinstance(job, dict) else None)
        }

    def compute_batch_size(self, job: Any) -> int:
        base = 64
        now = time.time()

        try:
            key = (job.header if isinstance(job, QMJob) else str(job.get("job_id") or job.get("header") or job.get("header_hex") or job.get("height") or job.get("epoch") or ""))
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
