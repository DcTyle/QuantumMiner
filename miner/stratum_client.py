# ================================================================
# Path: mining/stratum_client.py
# Description:
#   Stratum Client (patched for BIOS auto-connect compatibility)
#   + JSON runtime config support for multi-coin setup
# ================================================================

from __future__ import annotations
import socket, ssl, threading, json, time
from typing import Optional, Dict, Any, List

from miner.coin_switcher import configured_coins, resolve_coin_profile

EventBusType = Any  # duck-typed event bus

# Global registry to avoid duplicate connections to the same endpoint
_ACTIVE_ENDPOINTS: set = set()
_ACTIVE_LOCK = threading.RLock()


class StratumClient:
    def __init__(self,
                 coin: Optional[str] = None,
                 host: str = "",
                 port: int = 0,
                 user: Optional[str] = None,
                 password: str = "x",
                 use_tls: bool = False,
                 bus: Optional[EventBusType] = None,
                 wallet: Optional[str] = None,
                 worker_name: Optional[str] = None,
                 reconnect_base_sec: float = 1.0,
                 reconnect_max_sec: float = 60.0,
                 recv_buffer: int = 65536,
                 pool_url: str = "",
                 subscribe_method: str = "mining.subscribe",
                 subscribe_params: Optional[List[Any]] = None,
                 authorize_method: str = "mining.authorize",
                 authorize_with_password: bool = True,
                 worker_index: int = 1,
                 lane_id: str = "",
                 session_id: str = ""):
        self.coin = (coin or "").upper() or "ETC"
        self.host = host
        self.port = int(port)
        self.user = user or wallet or worker_name or "anonymous"
        self.password = password
        self.use_tls = bool(use_tls)
        self.bus = bus
        self.pool_url = pool_url
        self.subscribe_method = str(subscribe_method or "mining.subscribe")
        self.subscribe_params = list(subscribe_params or [])
        self.authorize_method = str(authorize_method or "mining.authorize")
        self.authorize_with_password = bool(authorize_with_password)
        self.worker_index = max(1, int(worker_index or 1))
        self.lane_id = str(lane_id or "")
        self.session_id = str(session_id or ("%s:%s" % (self.coin, self.user)))

        self._sock: Optional[socket.socket] = None
        self._ssl_ctx: Optional[ssl.SSLContext] = None
        self._reader: Optional[threading.Thread] = None
        self._running = False
        self._recv_buf_sz = int(recv_buffer)
        self._id = 0
        self._lock = threading.RLock()
        self._last_job: Optional[Dict[str, Any]] = None
        self._reconnect_base = float(reconnect_base_sec)
        self._reconnect_max = float(reconnect_max_sec)
        self._connected_announced = False
        self._last_connect_announce_ts = 0.0

    # ------------------------------------------------------------
    # Factory helpers for JSON-based configuration
    # ------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        coin: str,
        cfg: Dict[str, Any],
        bus: Optional[EventBusType] = None,
        worker_index: int = 1,
        lane_id: str = "",
        session_id: str = "",
    ):
        """Build a StratumClient directly from JSON coin entry."""
        profile = resolve_coin_profile(cfg.get("coins", {}), coin, worker_index=worker_index)
        return cls(
            coin=profile.coin.value,
            host=profile.host,
            port=profile.port,
            user=profile.username,
            password=profile.password,
            use_tls=profile.use_tls,
            bus=bus,
            wallet=profile.wallet,
            worker_name=profile.worker_name,
            pool_url=profile.pool_url,
            subscribe_method=profile.subscribe_method,
            subscribe_params=list(profile.subscribe_params),
            authorize_method=profile.authorize_method,
            authorize_with_password=profile.authorize_with_password,
            worker_index=worker_index,
            lane_id=lane_id,
            session_id=session_id,
        )

    @staticmethod
    def connect_all_from_config(cfg: Dict[str, Any], bus: Optional[EventBusType] = None) -> Dict[str, "StratumClient"]:
        """Build one StratumClient per coin from JSON config."""
        clients: Dict[str, StratumClient] = {}
        for coin in configured_coins(cfg.get("coins", {})):
            try:
                cli = StratumClient.from_json(coin, cfg, bus)
                cli.start()
                clients[coin.upper()] = cli
            except Exception as e:
                if bus:
                    bus.publish("stratum.error", {"coin": coin, "error": str(e)})
        return clients

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            key = (self.host, self.port, self.user)
            with _ACTIVE_LOCK:
                if key in _ACTIVE_ENDPOINTS:
                    # Duplicate connection attempt detected; skip
                    self._pub("stratum.start_skipped", {"coin": self.coin, "host": self.host, "port": self.port, "user": self.user})
                    return
                _ACTIVE_ENDPOINTS.add(key)
            self._running = True
            self._reader = threading.Thread(target=self._loop, daemon=True)
            self._reader.start()
            self._pub("stratum.start", {"coin": self.coin, "host": self.host, "port": self.port})

    def stop(self) -> None:
        self._running = False
        try:
            if self._sock:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._connected_announced = False
        self._pub("stratum.disconnected", {"coin": self.coin})
        # release endpoint reservation
        try:
            key = (self.host, self.port, self.user)
            with _ACTIVE_LOCK:
                _ACTIVE_ENDPOINTS.discard(key)
        except Exception:
            pass

    def get_current_job(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._last_job is None else dict(self._last_job)

    def is_connected(self) -> bool:
        with self._lock:
            return bool(self._running and self._sock is not None)

    def call(self, method: str, params: List[Any]) -> bool:
        payload = {"id": self._next_id(), "method": str(method), "params": list(params or [])}
        try:
            self._send_json(payload)
            return True
        except Exception as e:
            self._pub("stratum.send_error", {"coin": self.coin, "error": str(e), "method": str(method)})
            return False

    def suggest_difficulty(self, difficulty: float) -> bool:
        try:
            diff = max(1.0, float(difficulty))
        except Exception:
            return False
        return self.call("mining.suggest_difficulty", [diff])

    def suggest_target(self, target_hex: str) -> bool:
        target = str(target_hex or "").strip()
        if not target:
            return False
        return self.call("mining.suggest_target", [target])

    def submit(self, params: List[Any]) -> None:
        self._send_rpc("mining.submit", params)

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _loop(self) -> None:
        backoff = self._reconnect_base
        while self._running:
            try:
                self._sock = self._connect()
                self._pub("stratum.connected", {"coin": self.coin, "host": self.host, "port": self.port})
                now = time.time()
                if self.coin == "ERG":
                    # ERG: suppress client-side repeated prints; StratumAdapter handles a single guarded print
                    if not self._connected_announced:
                        self._connected_announced = True
                        self._last_connect_announce_ts = now
                else:
                    if (not self._connected_announced) or (now - self._last_connect_announce_ts > 5.0):
                        print(f"[Stratum] Connected to {self.host}:{self.port} as {self.user}")
                        self._connected_announced = True
                        self._last_connect_announce_ts = now
                self._subscribe_and_authorize()
                backoff = self._reconnect_base
                self._read_loop()
            except Exception as e:
                self._pub("stratum.error", {"coin": self.coin, "error": str(e)})
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                self._connected_announced = False
                if not self._running:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self._reconnect_max)
        self._pub("stratum.loop_exit", {"coin": self.coin})
        # ensure endpoint reservation cleared on exit
        try:
            key = (self.host, self.port, self.user)
            with _ACTIVE_LOCK:
                _ACTIVE_ENDPOINTS.discard(key)
        except Exception:
            pass

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self.host, self.port), timeout=10)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(30.0)  # Read timeout to prevent hanging
        if self.use_tls:
            if self._ssl_ctx is None:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                self._ssl_ctx = ctx
            s = self._ssl_ctx.wrap_socket(s, server_hostname=self.host)
        return s

    def _subscribe_and_authorize(self) -> None:
        self._send_json({"id": self._next_id(), "method": self.subscribe_method, "params": list(self.subscribe_params)})
        time.sleep(0.1)
        auth_params = [self.user, self.password] if self.authorize_with_password else [self.user]
        self._send_json({"id": self._next_id(), "method": self.authorize_method, "params": auth_params})

    def _read_loop(self) -> None:
        if self._sock is None:
            return
        buf = b""
        while self._running and self._sock:
            try:
                data = self._sock.recv(self._recv_buf_sz)
                if not data:
                    raise ConnectionError("stratum socket closed")
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        self._handle_line(line)
            except socket.timeout:
                # Timeout on read, check if still running
                continue
            except Exception as e:
                raise e

    def _handle_line(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception as e:
            self._pub("stratum.recv_error", {"coin": self.coin, "error": f"json_parse: {e}"})
            return

        if isinstance(msg, dict) and "method" in msg:
            method = msg.get("method")
            params = msg.get("params", [])
            if method == "mining.notify":
                job = {"method": method, "params": params, "received_at": time.time()}
                with self._lock:
                    self._last_job = job
                self._pub("stratum.job", {"coin": self.coin, "job": job})
            elif method in ("mining.set_difficulty", "mining.set_target"):
                self._pub("stratum.difficulty", {"coin": self.coin, "method": method, "params": params})
            elif method == "client.show_message":
                self._pub("stratum.server_message", {"coin": self.coin, "message": params})
            else:
                self._pub("stratum.unknown", {"coin": self.coin, "method": method, "params": params})
        else:
            self._pub("stratum.response", {
                "coin": self.coin,
                "id": msg.get("id"),
                "result": msg.get("result"),
                "error": msg.get("error"),
            })

    # ------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------
    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _send_json(self, obj: Dict[str, Any]) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self._send_bytes(data)

    def _send_rpc(self, method: str, params: List[Any]) -> None:
        payload = {"id": self._next_id(), "method": method, "params": params}
        try:
            self._send_json(payload)
        except Exception as e:
            self._pub("stratum.send_error", {"coin": self.coin, "error": str(e)})

    def _send_bytes(self, data: bytes) -> None:
        with self._lock:
            if self._sock is None:
                raise ConnectionError("stratum not connected")
            self._sock.sendall(data)

    # ------------------------------------------------------------
    def _pub(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.bus:
            try:
                meta = {
                    "coin": self.coin,
                    "user": self.user,
                    "worker_index": int(self.worker_index),
                    "lane_id": self.lane_id,
                    "session_id": self.session_id,
                    "host": self.host,
                    "port": int(self.port),
                }
                meta.update(dict(payload or {}))
                self.bus.publish(event_type, meta)
            except Exception:
                pass
