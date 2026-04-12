# ASCII-ONLY FILE
# Bitcoin SHA256 Stratum V1 Client
import socket
import json
import time
import threading
import hashlib
import math
from collections import deque
from typing import Any, Dict
from miner.common_types import QMJob, qmjob_from_dict

from . import NetSpecBase, NetID

class BTCPoolClient:
    def __init__(self, endpoint, username="worker", password="x", timeout=5):
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.timeout = timeout

        self.sock = None
        self.lock = threading.Lock()

        self.subscribed = False
        self.authorized = False
        self.extranonce1 = ""
        self.extranonce2_size = 0

    def _connect(self):
        host, port = self.endpoint.split(":")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((host, int(port)))

    def _send_msg(self, method, params):
        msg = {
            "id": int(time.time()),
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

    def _ensure_connected(self):
        if self.sock:
            return True
        try:
            self._connect()
            return True
        except Exception:
            return False

    def _subscribe(self):
        if not self._ensure_connected():
            return False
        resp = self._send_msg("mining.subscribe", [])
        try:
            result = resp.get("result", [])
            self.extranonce1 = result[1]
            self.extranonce2_size = result[2]
            self.subscribed = True
            return True
        except Exception:
            return False

    def _authorize(self):
        if not self.subscribed:
            if not self._subscribe():
                return False
        resp = self._send_msg("mining.authorize", [self.username, self.password])
        if resp.get("result") is True:
            self.authorized = True
            return True
        return False

    def submit_share(self, network, share):
        if not self._ensure_connected():
            return False
        if not self.subscribed:
            if not self._subscribe():
                return False
        if not self.authorized:
            if not self._authorize():
                return False

        job_id = share.get("job_id")
        extranonce2 = share.get("extranonce2")
        ntime = share.get("ntime")
        nonce = share.get("nonce")

        params = [
            self.username,
            job_id,
            extranonce2,
            ntime,
            nonce
        ]

        resp = self._send_msg("mining.submit", params)
        return bool(resp.get("result", False))


class NetSpec_BTC(NetSpecBase):
    network = "BTC"
    enum_id = NetID.BTC
    # rolling stats (class-wide)
    _share_lat_ms = deque(maxlen=10)
    _job_intervals = deque(maxlen=10)
    _last_job_key = ""
    _last_change_ts = 0.0

    def hash_fn(self, job: Any, nonce: int) -> Dict[str, str]:
        header_hex = str(job.header if isinstance(job, QMJob) else job.get("header", job.get("header_hex", "")))
        if header_hex.startswith("0x"):
            header_hex = header_hex[2:]
        header = bytes.fromhex(header_hex) if header_hex else b""
        n_bytes = nonce.to_bytes(4, byteorder="little", signed=False)
        preimage = header + n_bytes
        h = hashlib.sha256(hashlib.sha256(preimage).digest()).digest()
        return {
            "powhash": h.hex(),
            "mixhash": "",
            "header": header_hex or header.hex()
        }

    def target_fn(self, job: Any, powhash_hex: str) -> bool:
        target_hex = str(job.target if isinstance(job, QMJob) else job.get("target", ""))
        if target_hex.startswith("0x"):
            target_hex = target_hex[2:]
        try:
            return int(powhash_hex or "0", 16) <= int(target_hex or "0", 16)
        except Exception:
            return False

    def network_to_system_fn(self, raw: Dict[str, Any]) -> QMJob:
        # BTC Stratum V1 notify does not include a full header; we record target/height best-effort.
        params = raw.get("params", []) if isinstance(raw, dict) else []
        target_hex = ""
        height = -1
        # nbits sometimes appears in params or derived; leave as raw target if provided elsewhere.
        d = {"target": target_hex, "height": height}
        return qmjob_from_dict("BTC", d | {"raw": raw})

    def system_to_network_fn(self, qmshare: Any) -> Dict[str, Any]:
        sp = qmshare.system_payload
        job = sp.get("job")
        return {
            "network": "BTC",
            "job_id": (job.extra.get("job_id") if isinstance(job, QMJob) else (job or {}).get("job_id")),
            "extranonce2": (job.extra.get("extranonce2") if isinstance(job, QMJob) else (job or {}).get("extranonce2")),
            "ntime": (job.extra.get("ntime") if isinstance(job, QMJob) else (job or {}).get("ntime")),
            "nonce": "%08x" % int(qmshare.nonce)
        }

    def compute_batch_size(self, job: Any) -> int:
        base = 64
        now = time.time()

        # job change tracking
        try:
            key = (job.header if isinstance(job, QMJob) else str(job.get("header") or job.get("header_hex") or job.get("job_id") or job.get("height") or job.get("epoch") or ""))
        except Exception:
            key = ""
        if key != self._last_job_key:
            if self._last_change_ts > 0:
                self._job_intervals.append(max(0.001, now - self._last_change_ts))
            self._last_job_key = key
            self._last_change_ts = now

        # latency buffer (optional inputs)
        try:
            sp = job.get("system_payload", {}) if isinstance(job, dict) else {}
            lat = float(sp.get("last_share_latency_ms", sp.get("share_latency_ms", job.get("last_share_latency_ms", 0))))
            if lat > 0:
                self._share_lat_ms.append(lat)
        except Exception:
            pass

        # difficulty factor from target
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

        # stability factor from job change intervals
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

        # latency factor from share latencies
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
