from __future__ import annotations
import threading
import queue
# ============================================================================
# VirtualMiner / Miner
# File: submitter.py
# ASCII-ONLY SOURCE FILE
# Jarvis ADA v4.7 Hybrid Ready
# ----------------------------------------------------------------------------
# Purpose
# -------
# Live per-lane share submission with a lightweight thread pool, optional
# client resolver, and VSD-backed counters. This preserves the older
# flush-pending behavior as a fallback path but now favors immediate
# per-lane submission.
#
# API
# ---
# class Submitter:
#   Submitter(vsd=None, client_resolver=None, max_workers=8)
#   .submit_share(lane_id: str, network: str, share: dict) -> None
#   # no legacy flush_pending path retained
#   .stop(timeout: float = 2.0) -> None
#
# VSD Paths (aggregate)
# ---------------------
# /telemetry/metrics/<NET>/shares/aggregate -> {
#   "submitted": int, "accepted": int, "found": int,
#   "last_ts": float
# }
# /telemetry/metrics/<NET>/shares/_rate_snapshot -> {
#   "submitted": int, "accepted": int, "found": int, "ts": float
# }
#
# VSD Paths (per-lane)
# --------------------
# /telemetry/metrics/<NET>/shares/lanes/<LANE_ID> -> {
#   "submitted": int, "accepted": int, "found": int, "last_ts": float
# }
#
# Console Consumption
# -------------------
# Engine or submitter writes derived rates back into the network metrics
# block stored at:
#   /telemetry/metrics/<NET>/current
# Fields updated:
#   "hashes_submitted_hs", "hashes_found_hs", "accepted_hs" (optional)
#
# Notes
# -----
# - ASCII only, no Unicode.
# - Client resolver must return an object with .submit_share(network, share).
#   If missing, submissions are recorded as submitted but not accepted (no-op).
# ============================================================================

from typing import Dict, Any, Optional, Callable, Tuple
import time

# ---------------------------------------------------------------------------
# Layered imports
# ---------------------------------------------------------------------------
from core.utils import append_telemetry
import logging, sys
_def_fmt = logging.Formatter(
    fmt="%(asctime)sZ | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.Formatter.converter = time.gmtime  # UTC
_logger = logging.getLogger("miner.submitter")
if not _logger.handlers:
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(_def_fmt)
    _logger.addHandler(h)
_logger.setLevel(logging.INFO)


class Submitter:
    def __init__(
        self,
        vsd: Optional[Any] = None,
        client_resolver: Optional[Callable[[str], Any]] = None,
        max_workers: int = 8,
        handler_resolver: Optional[Callable[[str], Any]] = None,
        # Governor v2 sources (optional)
        read_telemetry: Optional[Callable[[], Dict[str, Any]]] = None,
        network_capacity_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        # Queue and batching
        max_queue_depth: int = 1024,
        micro_batch_size: int = 8,
        tick_duration_s: float = 0.25,
        market_data: Optional[Any] = None,
        network_symbol: Optional[str] = None,
        hashrate_band_poll_interval: float = 15.0,
    ):
        self.vsd = vsd
        self.client_resolver = client_resolver
        self.handler_resolver = handler_resolver
        self.q = queue.PriorityQueue()
        self._stop = False
        self._workers = []
        self._queue_seq = 0
        self._max_workers = max(0, int(max_workers))
        self._max_queue_depth = int(max_queue_depth)
        self._micro_batch_size = int(max(1, micro_batch_size))
        self._read_tel = read_telemetry
        self._cap_fn = network_capacity_fn
        self._market_data = market_data
        self._network_symbol = network_symbol
        self._hashrate_band_poll_interval = hashrate_band_poll_interval
        self._banded_hashrate_thread = None
        self._banded_hashrate_stop = threading.Event()
        self._paused = False
        self._pause_note = ""

        # Governor v2: token buckets and RTT backpressure
        self._base_allowed_per_sec: float = 2.0
        self._tick_duration: float = float(tick_duration_s)
        self._max_per_tick: float = max(1.0e-6, self._base_allowed_per_sec * self._tick_duration)
        self._sec_tokens: float = min(max(self._base_allowed_per_sec, 0.0), 1.0)
        self._tick_tokens: float = 1.0
        self._last_sec_refill: float = self._now()
        self._last_tick_refill: float = self._now()
        self._baseline_rtt_ms: float = 50.0
        self._last_avg_rtt_ms: float = 50.0
        self._last_jitter_delay_s: float = 0.0
        self._submissions_this_second: int = 0
        self._submissions_this_tick: int = 0
        self._throttle_events: int = 0
        self._burst_events: int = 0
        self._worker_governors: Dict[str, Dict[str, Any]] = {}

        # Per-network block tracking so we never submit stale shares
        # from a previous block once a new block is active.
        self._current_block_by_net: Dict[str, Any] = {}


        # Try to seed from VSD
        self._refresh_governor_from_vsd()

        # Start banded hashrate controller thread if market_data and network_symbol are provided
        if self._market_data is not None and self._network_symbol is not None:
            self._banded_hashrate_thread = threading.Thread(target=self._banded_hashrate_controller, daemon=True)
            self._banded_hashrate_thread.start()
        self._start_workers()

    def _start_workers(self) -> None:
        if self._workers or self._max_workers <= 0:
            return
        for i in range(self._max_workers):
            t = threading.Thread(target=self._worker, daemon=True, name="submitter_worker_%d" % i)
            t.start()
            self._workers.append(t)

    def is_paused(self) -> bool:
        return bool(self._paused)

    def pause(
        self,
        note: str = "",
        source: str = "control_center",
        drain_timeout_s: float = 1.0,
    ) -> bool:
        deadline = self._now() + max(0.0, float(drain_timeout_s))
        while self.q.qsize() > 0 and self._now() < deadline:
            time.sleep(0.01)
        self._paused = True
        self._pause_note = str(note or "")
        queue_depth = int(self.q.qsize())
        drained = queue_depth == 0
        self._store("miner/control/submitter", {
            "ts": self._now(),
            "paused": True,
            "note": self._pause_note,
            "source": str(source or "control_center"),
            "queue_depth": queue_depth,
            "queue_drained": bool(drained),
        })
        return drained

    def resume(self, note: str = "", source: str = "control_center") -> None:
        self._paused = False
        self._pause_note = str(note or "")
        self._store("miner/control/submitter", {
            "ts": self._now(),
            "paused": False,
            "note": self._pause_note,
            "source": str(source or "control_center"),
            "queue_depth": int(self.q.qsize()),
            "queue_drained": bool(self.q.qsize() == 0),
        })

    def stop(self, timeout: float = 2.0) -> None:
        self._stop = True
        self._banded_hashrate_stop.set()
        try:
            for _ in self._workers:
                self.q.put(self._queue_item("", "", {}, "stop"))
        except Exception:
            pass
        for t in self._workers:
            try:
                t.join(timeout=timeout)
            except Exception:
                pass
        self._workers = []
        if self._banded_hashrate_thread:
            self._banded_hashrate_thread.join(timeout=timeout)
            self._banded_hashrate_thread = None

    def _banded_hashrate_controller(self):
        """
        Background thread: keeps allowed submission rate so miner hashrate stays in [4.8%, 5%] of network.
        """
        while not self._banded_hashrate_stop.is_set():
            try:
                stats = self._market_data.get_all_prices().get(self._network_symbol, {})
                net_hashrate = float(stats.get("network_hashrate_hs", 0.0))
                net_diff = float(stats.get("difficulty", 0.0))
                if net_hashrate <= 0 or net_diff <= 0:
                    self._banded_hashrate_stop.wait(self._hashrate_band_poll_interval)
                    continue
                # Target band
                min_frac = 0.048
                max_frac = 0.05
                min_hashrate = net_hashrate * min_frac
                max_hashrate = net_hashrate * max_frac
                # Estimate our current hashrate
                metrics = self.vsd.get(f"telemetry/metrics/{self._network_symbol}/current", {}) if self.vsd else {}
                our_hashrate = float(metrics.get("hashes_found_hs", 0.0))
                # Compute new allowed rate
                # shares/sec = target_hashrate / (diff * 2**32)
                # If under min, lower diff (more shares/sec); if over max, raise diff (fewer shares/sec)
                target_hashrate = our_hashrate
                if our_hashrate < min_hashrate:
                    target_hashrate = min_hashrate
                elif our_hashrate > max_hashrate:
                    target_hashrate = max_hashrate
                allowed_rate = target_hashrate / (net_diff * 2**32)
                allowed_rate = max(0.1, allowed_rate)
                # Write allowed rate to VSD for governor
                if self.vsd:
                    self.vsd.store(f"miner/runtime/submission_rate", {"allowed_rate_per_second": allowed_rate, "tick_duration": self._tick_duration})
            except Exception:
                pass
            self._banded_hashrate_stop.wait(self._hashrate_band_poll_interval)

        # no legacy buffers; strict immediate submission

    def _queue_item(self, lane_id: str, network: str, share: Dict[str, Any], op: str) -> tuple[int, int, str, str, str, Dict[str, Any]]:
        try:
            self._queue_seq += 1
        except Exception:
            self._queue_seq = 1
        seq = int(self._queue_seq)
        if op == "stop":
            return (10 ** 9, seq, op, str(lane_id), str(network), dict(share or {}))

        bucket_priority = 0.0
        trace_alignment = 0.0
        target_interval = 0
        sequence_index = 0
        try:
            bucket_priority = max(0.0, min(1.0, float(share.get("bucket_priority", share.get("coherence", 0.0)))))
        except Exception:
            bucket_priority = 0.0
        try:
            trace_alignment = max(0.0, min(1.0, float(share.get("trace_alignment", 0.0))))
        except Exception:
            trace_alignment = 0.0
        try:
            target_interval = max(0, int(share.get("target_interval", 0) or 0))
        except Exception:
            target_interval = 0
        try:
            sequence_index = max(0, int(share.get("sequence_index", 0) or 0))
        except Exception:
            sequence_index = 0

        interval_alignment = 1.0 / float(1 + target_interval)
        sequence_alignment = 1.0 / float(1 + (sequence_index % 16))
        lane_bias = self._stable_fraction(str(network).upper(), lane_id, share.get("worker_bucket", share.get("bucket_id", "")))
        priority_score = max(
            0.0,
            min(
                1.0,
                0.62 * bucket_priority
                + 0.18 * trace_alignment
                + 0.10 * interval_alignment
                + 0.05 * sequence_alignment
                + 0.05 * lane_bias,
            ),
        )
        rank = int(round((1.0 - priority_score) * 1000.0))
        if op == "defer":
            rank += 100
        return (rank, seq, op, str(lane_id), str(network), dict(share or {}))

    # -------------------------
    # Public submission methods
    # -------------------------

    # submit_qmshare removed; only neural_objectPacket path is supported via submit_packet

    # Governor v2 API
    def can_accept(self, network: Optional[str] = None, lane_id: str = "") -> bool:
        """
        Quick pre-check based on queue depth and token availability.
        """
        if self.is_paused():
            return False
        try:
            if self.q.qsize() >= self._max_queue_depth:
                return False
        except Exception:
            pass
        # Refill tokens opportunistically
        self._refill_tokens()
        if lane_id:
            self._refill_worker_tokens(network, lane_id)
        # Check tokens (allow small epsilon)
        if not ((self._sec_tokens >= 1.0) and (self._tick_tokens >= 1.0)):
            return False
        if lane_id:
            state = self._worker_governors.get(self._worker_governor_key(network, lane_id), {})
            return float(state.get("sec_tokens", 1.0)) >= 1.0 and float(state.get("tick_tokens", 1.0)) >= 1.0
        return True

    def governor_snapshot(self) -> Dict[str, Any]:
        self._refill_tokens()
        return {
            "allowed_rate_per_second": float(self._effective_allowed_per_sec()),
            "base_allowed_per_second": float(self._base_allowed_per_sec),
            "max_submissions_per_tick": float(self._max_per_tick),
            "tick_duration": float(self._tick_duration),
            "tokens_second": float(self._sec_tokens),
            "tokens_tick": float(self._tick_tokens),
            "queue_depth": int(self.q.qsize() if hasattr(self.q, 'qsize') else 0),
            "last_rtt_ms": float(self._last_avg_rtt_ms),
            "baseline_rtt_ms": float(self._baseline_rtt_ms),
            "jitter_window_s": float(self._configured_jitter_window_s()),
            "last_jitter_delay_ms": float(self._last_jitter_delay_s * 1000.0),
            "throttle_events": int(self._throttle_events),
            "burst_events": int(self._burst_events),
            "submissions_this_second": int(self._submissions_this_second),
            "submissions_this_tick": int(self._submissions_this_tick),
            "paused": bool(self.is_paused()),
        }

    def submit_packet(self, lane_id: str, packet: Any) -> None:
        """
        Primary send path: accepts a neural_objectPacket, converts with
        schema["convert_outgoing"], and enqueues for pool submission.
        """
        try:
            from neural_object import neural_objectPacket as _Pkt, neural_objectSchema
        except Exception:
            return
        if not isinstance(packet, _Pkt):
            return
        try:
            if not bool((packet.system_payload or {}).get("valid", True)):
                return
        except Exception:
            return
        schema = neural_objectSchema.get(packet.packet_type)
        if not schema:
            return
        try:
            payload = schema["convert_outgoing"](packet)
            if not isinstance(payload, dict):
                return
        except Exception:
            return
        network = packet.network.name
        self.submit_share(lane_id, network, payload)

    def submit_share(self, lane_id: str, network: str, share: Dict[str, Any]) -> None:
        """
        Live, per-lane submission. Enqueues a job for the thread pool.
        Records lane-level and aggregate counters in VSD.
        """
        try:
            lane_id = str(lane_id)
            network = str(network).upper()
        except Exception:
            return

        if self.is_paused():
            self._store("miner/engine/throttle", {
                "ts": self._now(),
                "reason": "paused",
                "network": network,
                "lane": lane_id,
            })
            return

        # Resolve block identifier (job_id or height) and record per-network
        block_id: Any = share.get("job_id")
        if block_id is None:
            # Fallback to block height if present
            block_id = share.get("height")
        if block_id is not None:
            try:
                # normalize for equality comparisons
                block_id = int(block_id)
            except Exception:
                block_id = str(block_id)
            self._current_block_by_net[network] = block_id
        # Governor v2 pre-check and backpressure
        self._refresh_governor_from_vsd()
        if not self.can_accept(network):
            # Throttled: record and write telemetry, then either drop or defer
            self._throttle_events += 1
            self._write_submitter_telemetry(network, throttled=True)
            # Defer if queue has room; else drop
            try:
                if self.q.qsize() < self._max_queue_depth:
                    self.q.put_nowait(self._queue_item(lane_id, network, dict(share or {}), "defer"))
                else:
                    return
            except Exception:
                return
            return

        # update "found" counter immediately; acceptance depends on pool response
        self._bump_found(network, lane_id)
        # enqueue submission (micro-batch friendly)
        try:
            self.q.put_nowait(self._queue_item(lane_id, network, dict(share or {}), "submit"))
        except Exception:
            _logger.error("submit queue full; dropping share lane=%s net=%s", lane_id, network)

    # flush_pending removed; no legacy path maintained

    # -------------------------
    # Internal worker and calls
    # -------------------------

    def _worker(self) -> None:
        while not self._stop:
            try:
                _rank, _seq, op, lane_id, network, share = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if op == "stop":
                break
            if op in ("submit", "defer"):
                # Normalize network key and drop stale-block shares
                try:
                    net_key = str(network).upper()
                except Exception:
                    net_key = str(network)

                # Determine share block id for comparison
                blk = share.get("job_id")
                if blk is None:
                    blk = share.get("height")
                if blk is not None:
                    try:
                        blk = int(blk)
                    except Exception:
                        blk = str(blk)
                current_blk = self._current_block_by_net.get(net_key)
                if current_blk is not None and blk is not None and blk != current_blk:
                    # Stale share from previous block; drop it
                    self.q.task_done()
                    continue

                # Micro-batch: process up to N items or until tokens exhausted
                self._refill_tokens()
                processed = 0
                max_batch = max(1, self._micro_batch_size)
                while processed < max_batch:
                    # If throttled mid-batch, break
                    if not self._consume_tokens_if_available(net_key, lane_id):
                        self._throttle_events += 1
                        break
                    self._maybe_apply_submission_jitter(net_key, lane_id, share)
                    ok = self._do_submit(lane_id, net_key, share)
                    processed += 1
                    # Peek next if same op available
                    try:
                        nxt = self.q.get_nowait()
                        _n_rank, _n_seq, n_op, n_lane, n_net, n_share = nxt
                        if n_op not in ("submit", "defer"):
                            # push back and stop
                            self.q.put_nowait(nxt)
                            break
                        lane_id, net_key, share = n_lane, str(n_net).upper(), n_share
                        # Re-check block freshness for the new item
                        blk = share.get("job_id")
                        if blk is None:
                            blk = share.get("height")
                        if blk is not None:
                            try:
                                blk = int(blk)
                            except Exception:
                                blk = str(blk)
                        current_blk = self._current_block_by_net.get(net_key)
                        if current_blk is not None and blk is not None and blk != current_blk:
                            # Stale share; do not process further for this item
                            self.q.task_done()
                            break
                    except queue.Empty:
                        break
            # no ack needed
            self.q.task_done()

    def _do_submit(self, lane_id: str, network: str, share: Dict[str, Any]) -> bool:
        """
        Perform the actual submission via client resolver. Update VSD counters
        for submitted and accepted. Returns True on acceptance.
        """
        self._bump_submitted(network, lane_id)
        self._submissions_this_second += 1
        self._submissions_this_tick += 1
        client = None
        if callable(self.client_resolver):
            try:
                client = self.client_resolver(network, lane_id)
            except TypeError:
                try:
                    client = self.client_resolver(network)
                except Exception:
                    client = None
            except Exception:
                client = None

        accepted = False
        try:
            if client is not None:
                # strict: require .submit_share(network, share)
                if hasattr(client, "submit_share"):
                    _logger.info("submit_share -> network=%s job=%s nonce=%s", network, share.get("job_id"), share.get("nonce"))
                    accepted = bool(client.submit_share(network, share))
                else:
                    accepted = False
            else:
                # no client available; treat as submitted but not accepted
                accepted = False
        except Exception:
            accepted = False

        if accepted:
            _logger.info("accepted -> network=%s job=%s nonce=%s", network, share.get("job_id"), share.get("nonce"))
            self._bump_accepted(network, lane_id)
        else:
            _logger.info("rejected -> network=%s job=%s nonce=%s", network, share.get("job_id"), share.get("nonce"))

        # Best-effort latency from share payload
        try:
            recv_at = float(share.get("received_at", 0.0))
            if recv_at > 0.0:
                lat_ms = max(0.0, (self._now() - recv_at) * 1000.0)
                self._update_latency(network, lat_ms)
                # Update moving average and baseline
                self._last_avg_rtt_ms = float(self._fetch("telemetry/metrics/%s/latency/avg_ms" % network, 0.0))
                if self._last_avg_rtt_ms > 0.0:
                    if self._baseline_rtt_ms <= 0.0:
                        self._baseline_rtt_ms = self._last_avg_rtt_ms
                    else:
                        self._baseline_rtt_ms = min(self._baseline_rtt_ms, self._last_avg_rtt_ms)
        except Exception:
            pass

        # after each submit, attempt to update rolling rates into metrics (network and lane)
        self._update_rates_into_metrics(network)
        self._update_lane_rates(network, lane_id)

        # Telemetry snapshot
        self._write_submitter_telemetry(network, throttled=False)
        return accepted

    # -------------------------
    # VSD counter helpers
    # -------------------------

    def _now(self) -> float:
        return float(time.time())

    def _fetch(self, key: str, default: Any) -> Any:
        if not self.vsd:
            return default
        try:
            val = self.vsd.get(key, default)
            return default if val is None else val
        except Exception:
            return default

    def _store(self, key: str, value: Any) -> None:
        if not self.vsd:
            return
        try:
            self.vsd.store(key, value)
        except Exception:
            pass

    def _bump_lane(self, network: str, lane_id: str, field: str) -> None:
        key = "telemetry/metrics/%s/shares/lanes/%s" % (network, lane_id)
        lane = dict(self._fetch(key, {}))
        lane["submitted"] = int(lane.get("submitted", 0))
        lane["accepted"] = int(lane.get("accepted", 0))
        lane["found"] = int(lane.get("found", 0))
        lane[field] = int(lane.get(field, 0)) + 1
        lane["last_ts"] = self._now()
        self._store(key, lane)
        # maintain lane index for discovery
        try:
            idx_key = "telemetry/metrics/%s/shares/lanes/_index" % network
            idx = list(self._fetch(idx_key, []))
            if lane_id not in idx:
                idx.append(lane_id)
                self._store(idx_key, idx)
        except Exception:
            pass

    def _bump_agg(self, network: str, field: str) -> None:
        key = "telemetry/metrics/%s/shares/aggregate" % network
        agg = dict(self._fetch(key, {}))
        agg["submitted"] = int(agg.get("submitted", 0))
        agg["accepted"] = int(agg.get("accepted", 0))
        agg["found"] = int(agg.get("found", 0))
        agg[field] = int(agg.get(field, 0)) + 1
        agg["last_ts"] = self._now()
        self._store(key, agg)

    def _bump_found(self, network: str, lane_id: str) -> None:
        self._bump_lane(network, lane_id, "found")
        self._bump_agg(network, "found")

    def _bump_submitted(self, network: str, lane_id: str) -> None:
        self._bump_lane(network, lane_id, "submitted")
        self._bump_agg(network, "submitted")

    def _bump_accepted(self, network: str, lane_id: str) -> None:
        self._bump_lane(network, lane_id, "accepted")
        self._bump_agg(network, "accepted")

    # -------------------------
    # Rate derivation into metrics
    # -------------------------

    def _update_rates_into_metrics(self, network: str) -> None:
        """
        Compute simple per-second rates from aggregate deltas and write them
        into the network metrics block so the telemetry console can show:
          - hashes_submitted_hs
          - hashes_found_hs
          - accepted_hs (optional)
        """
        agg_key = "telemetry/metrics/%s/shares/aggregate" % network
        snap_key = "telemetry/metrics/%s/shares/_rate_snapshot" % network
        cur = dict(self._fetch(agg_key, {}))
        snap = dict(self._fetch(snap_key, {}))
        now = self._now()
        prev_ts = float(snap.get("ts", now))
        dt = max(1e-6, now - prev_ts)

        # count deltas
        sub_cur = int(cur.get("submitted", 0))
        acc_cur = int(cur.get("accepted", 0))
        fnd_cur = int(cur.get("found", 0))

        sub_prev = int(snap.get("submitted", 0))
        acc_prev = int(snap.get("accepted", 0))
        fnd_prev = int(snap.get("found", 0))

        sub_rate = max(0.0, (sub_cur - sub_prev) / dt)
        acc_rate = max(0.0, (acc_cur - acc_prev) / dt)
        fnd_rate = max(0.0, (fnd_cur - fnd_prev) / dt)

        # write back into network metrics block
        metrics_key = "telemetry/metrics/%s/current" % network
        metrics = dict(self._fetch(metrics_key, {}))
        metrics["hashes_submitted_hs"] = float(sub_rate)
        metrics["hashes_found_hs"] = float(fnd_rate)
        metrics["accepted_hs"] = float(acc_rate)
        # rolling acceptance rate (accepted per submitted over this window)
        metrics["acceptance_rate"] = float(acc_rate / sub_rate) if sub_rate > 1e-9 else 0.0
        self._store(metrics_key, metrics)

        # update snapshot
        self._store(snap_key, {
            "submitted": sub_cur,
            "accepted": acc_cur,
            "found": fnd_cur,
            "ts": now
        })

        # Maintain simple index of networks with live metrics
        try:
            idx_key = "telemetry/metrics/index"
            idx = list(self._fetch(idx_key, []))
            if network not in idx:
                idx.append(network)
                self._store(idx_key, idx)
        except Exception:
            pass

    # -------------------------
    # Lane-level rate metrics
    # -------------------------
    def _update_lane_rates(self, network: str, lane_id: str) -> None:
        lane_key = "telemetry/metrics/%s/shares/lanes/%s" % (network, lane_id)
        snap_key = lane_key + "/_rate_snapshot"
        lane = dict(self._fetch(lane_key, {}))
        snap = dict(self._fetch(snap_key, {}))
        now = self._now()
        prev_ts = float(snap.get("ts", now))
        dt = max(1e-6, now - prev_ts)

        sub_cur = int(lane.get("submitted", 0))
        acc_cur = int(lane.get("accepted", 0))
        sub_prev = int(snap.get("submitted", 0))
        acc_prev = int(snap.get("accepted", 0))

        sub_rate = max(0.0, (sub_cur - sub_prev) / dt)
        acc_rate = max(0.0, (acc_cur - acc_prev) / dt)
        accept_ratio = float(acc_rate / sub_rate) if sub_rate > 1e-9 else 0.0

        lane["submitted_hs"] = float(sub_rate)
        lane["accepted_hs"] = float(acc_rate)
        lane["acceptance_rate"] = float(accept_ratio)
        # simple static ratio for context
        denom = float(sub_cur) if sub_cur > 0 else 1.0
        lane["acceptance_ratio_total"] = float(acc_cur / denom)
        self._store(lane_key, lane)

        self._store(snap_key, {
            "submitted": sub_cur,
            "accepted": acc_cur,
            "ts": now
        })

    # -------------------------
    # Latency tracking (per-network)
    # -------------------------
    def _update_latency(self, network: str, latency_ms: float, maxlen: int = 32) -> None:
        base = "telemetry/metrics/%s/latency" % network
        arr_key = base + "/recent_ms"
        avg_key = base + "/avg_ms"
        arr = list(self._fetch(arr_key, []))
        arr.append(float(latency_ms))
        if len(arr) > maxlen:
            arr = arr[-maxlen:]
        self._store(arr_key, arr)
        try:
            avg = sum(arr) / float(len(arr)) if arr else 0.0
        except Exception:
            avg = 0.0
        self._store(avg_key, float(avg))

    # -------------------------
    # Governor helpers
    # -------------------------
    def _worker_governor_key(self, network: Optional[str], lane_id: str) -> str:
        return "%s::%s" % (str(network or "").upper(), str(lane_id))

    def _worker_rate_record(self, network: Optional[str], lane_id: str) -> Dict[str, Any]:
        lane_key = str(lane_id or "")
        if not lane_key:
            return {}
        record: Dict[str, Any] = {}
        worker_key = "miner/runtime/submission_rate/workers/%s" % lane_key
        worker_sr = dict(self._fetch(worker_key, {}))
        if worker_sr:
            boosted_until = float(worker_sr.get("boosted_until", 0.0) or 0.0)
            if boosted_until > 0.0 and self._now() > boosted_until:
                worker_sr.pop("boosted_until", None)
                self._store(worker_key, worker_sr)
            record.update(worker_sr)

        sr = dict(self._fetch("miner/runtime/submission_rate", {}))
        network_u = str(network or record.get("coin", "")).upper()
        if network_u:
            per_network = dict(sr.get("per_network", {}))
            net_cfg = dict(per_network.get(network_u, {}))
            workers = list(net_cfg.get("workers", []) or [])
            for worker in workers:
                worker_dict = dict(worker or {})
                if str(worker_dict.get("lane_id", "")) != lane_key:
                    continue
                if "allowed_submit_rate" in worker_dict and "allowed_rate_per_second" not in record:
                    record["allowed_rate_per_second"] = float(worker_dict.get("allowed_submit_rate", 0.0) or 0.0)
                if "tick_duration_s" in worker_dict and "tick_duration" not in record:
                    record["tick_duration"] = float(worker_dict.get("tick_duration_s", 0.0) or 0.0)
                if "jitter_window_s" in worker_dict and "jitter_window_s" not in record:
                    record["jitter_window_s"] = float(worker_dict.get("jitter_window_s", 0.0) or 0.0)
                if "username" in worker_dict and "username" not in record:
                    record["username"] = str(worker_dict.get("username", ""))
                break
            if "allowed_rate_per_second" not in record and "allowed_submit_rate" in net_cfg:
                record["allowed_rate_per_second"] = float(net_cfg.get("allowed_submit_rate", 0.0) or 0.0)
            if "tick_duration" not in record and "tick_duration_s" in net_cfg:
                record["tick_duration"] = float(net_cfg.get("tick_duration_s", 0.0) or 0.0)
            if "jitter_window_s" not in record and "jitter_window_s" in net_cfg:
                record["jitter_window_s"] = float(net_cfg.get("jitter_window_s", 0.0) or 0.0)

        if network_u:
            coin_sr = dict(self._fetch("miner/runtime/submission_rate/%s" % network_u, {}))
            if "allowed_rate_per_second" not in record and "allowed_rate_per_second" in coin_sr:
                record["allowed_rate_per_second"] = float(coin_sr.get("allowed_rate_per_second", 0.0) or 0.0)
            if "tick_duration" not in record and "tick_duration" in coin_sr:
                record["tick_duration"] = float(coin_sr.get("tick_duration", 0.0) or 0.0)

        if "allowed_rate_per_second" not in record and "allowed_rate_per_second" in sr:
            record["allowed_rate_per_second"] = float(sr.get("allowed_rate_per_second", 0.0) or 0.0)
        if "tick_duration" not in record and "tick_duration" in sr:
            record["tick_duration"] = float(sr.get("tick_duration", 0.0) or 0.0)
        if "jitter_window_s" not in record and "jitter_window_s" in sr:
            record["jitter_window_s"] = float(sr.get("jitter_window_s", 0.0) or 0.0)
        return record

    def _refresh_governor_from_vsd(self) -> None:
        # Pull allowed rate and tick duration published by miner_runtime
        sr_key = "miner/runtime/submission_rate"
        sr = dict(self._fetch(sr_key, {}))
        base_rate = float(sr.get("allowed_rate_per_second", self._base_allowed_per_sec))
        tick_dur = float(sr.get("tick_duration", self._tick_duration))
        now = self._now()
        boosted_until = float(sr.get("boosted_until", 0.0))
        if boosted_until > 0.0 and now > boosted_until:
            # Boost expired, revert to calculated rate
            sr.pop("boosted_until", None)
            self._store(sr_key, sr)
        if base_rate > 0:
            self._base_allowed_per_sec = base_rate
        if tick_dur > 0:
            self._tick_duration = tick_dur
        self._max_per_tick = max(1.0e-6, self._base_allowed_per_sec * self._tick_duration)
        self._sec_tokens = min(max(0.0, self._sec_tokens), max(1.0, self._base_allowed_per_sec * 2.0))
        self._tick_tokens = min(max(0.0, self._tick_tokens), max(1.0, self._base_allowed_per_sec * max(self._tick_duration, 0.05)))

    def _refill_worker_tokens(self, network: Optional[str], lane_id: str) -> None:
        lane_key = str(lane_id or "")
        if not lane_key:
            return
        governor_key = self._worker_governor_key(network, lane_key)
        record = self._worker_rate_record(network, lane_key)
        now = self._now()
        state = self._worker_governors.get(governor_key)
        if state is None:
            base_allowed = float(record.get("allowed_rate_per_second", self._base_allowed_per_sec) or self._base_allowed_per_sec)
            tick_duration = float(record.get("tick_duration", self._tick_duration) or self._tick_duration)
            state = {
                "base_allowed_per_sec": max(1.0e-6, base_allowed),
                "tick_duration": max(0.01, tick_duration),
                "sec_tokens": min(max(base_allowed, 0.0), 1.0),
                "tick_tokens": 1.0,
                "last_sec_refill": now,
                "last_tick_refill": now,
                "jitter_window_s": float(record.get("jitter_window_s", 0.0) or 0.0),
            }
            self._worker_governors[governor_key] = state

        state["base_allowed_per_sec"] = max(1.0e-6, float(record.get("allowed_rate_per_second", state.get("base_allowed_per_sec", self._base_allowed_per_sec)) or state.get("base_allowed_per_sec", self._base_allowed_per_sec)))
        state["tick_duration"] = max(0.01, float(record.get("tick_duration", state.get("tick_duration", self._tick_duration)) or state.get("tick_duration", self._tick_duration)))
        state["jitter_window_s"] = max(0.0, float(record.get("jitter_window_s", state.get("jitter_window_s", 0.0)) or state.get("jitter_window_s", 0.0)))

        eff = float(state["base_allowed_per_sec"])
        curr = float(self._last_avg_rtt_ms)
        if curr > 0.0:
            baseline = float(self._baseline_rtt_ms) if self._baseline_rtt_ms > 0 else curr
            eff *= max(0.1, min(1.0, baseline / curr))
        try:
            fb = dict(self._fetch("miner/compute_feedback/submission_rate_throttle", {}))
            factor_fb = float(fb.get("factor", 1.0))
            eff = max(1.0e-6, eff * max(0.1, min(1.0, factor_fb)))
        except Exception:
            eff = max(1.0e-6, eff)

        sec_elapsed = max(0.0, now - float(state.get("last_sec_refill", now)))
        if sec_elapsed > 0.0:
            sec_capacity = max(1.0, eff * 2.0)
            state["sec_tokens"] = min(sec_capacity, float(state.get("sec_tokens", 0.0)) + (sec_elapsed * eff))
            state["last_sec_refill"] = now

        tick_elapsed = max(0.0, now - float(state.get("last_tick_refill", now)))
        if tick_elapsed > 0.0:
            tick_duration = max(0.01, float(state.get("tick_duration", self._tick_duration)))
            tick_capacity = max(1.0, eff * max(tick_duration, 0.05))
            state["tick_tokens"] = min(tick_capacity, float(state.get("tick_tokens", 0.0)) + (tick_elapsed * eff))
            state["last_tick_refill"] = now

    def _effective_allowed_per_sec(self) -> float:
        # RTT backpressure: new_rate = base * (baseline / current)
        curr = float(self._last_avg_rtt_ms)
        base = float(self._base_allowed_per_sec)
        if curr <= 0.0:
            eff = base
        else:
            b = float(self._baseline_rtt_ms) if self._baseline_rtt_ms > 0 else curr
            factor = max(0.1, min(1.0, b / curr))
            eff = max(1.0e-6, base * factor)

        # Honor compute_feedback/submission_rate_throttle factor if present
        try:
            fb = dict(self._fetch("miner/compute_feedback/submission_rate_throttle", {}))
            factor_fb = float(fb.get("factor", 1.0))
            eff = max(1.0e-6, eff * max(0.1, min(1.0, factor_fb)))
        except Exception:
            pass

        return eff

    def _refill_tokens(self) -> None:
        # Recompute targets from latest vsd
        self._refresh_governor_from_vsd()
        now = self._now()
        eff = max(1.0e-6, self._effective_allowed_per_sec())
        sec_elapsed = max(0.0, now - self._last_sec_refill)
        if sec_elapsed > 0.0:
            sec_capacity = max(1.0, eff * 2.0)
            self._sec_tokens = min(sec_capacity, self._sec_tokens + (sec_elapsed * eff))
            if sec_elapsed >= 1.0:
                self._submissions_this_second = 0
            self._last_sec_refill = now
        tick_elapsed = max(0.0, now - self._last_tick_refill)
        if tick_elapsed > 0.0:
            tick_capacity = max(1.0, eff * max(self._tick_duration, 0.05))
            self._tick_tokens = min(tick_capacity, self._tick_tokens + (tick_elapsed * eff))
            if tick_elapsed >= self._tick_duration:
                self._submissions_this_tick = 0
            self._last_tick_refill = now

    def _consume_tokens_if_available(self, network: Optional[str] = None, lane_id: str = "") -> bool:
        # Attempt to consume one token from both buckets
        self._refill_tokens()
        if not (self._sec_tokens >= 1.0 and self._tick_tokens >= 1.0):
            return False
        if lane_id:
            self._refill_worker_tokens(network, lane_id)
            state = self._worker_governors.get(self._worker_governor_key(network, lane_id), {})
            if float(state.get("sec_tokens", 0.0)) < 1.0 or float(state.get("tick_tokens", 0.0)) < 1.0:
                return False
            state["sec_tokens"] = float(state.get("sec_tokens", 0.0)) - 1.0
            state["tick_tokens"] = float(state.get("tick_tokens", 0.0)) - 1.0
        self._sec_tokens -= 1.0
        self._tick_tokens -= 1.0
        return True

    def _configured_jitter_window_s(self, network: Optional[str] = None, lane_id: str = "") -> float:
        jitter_window = 0.0
        if lane_id:
            worker_record = self._worker_rate_record(network, lane_id)
            jitter_window = float(worker_record.get("jitter_window_s", 0.0) or 0.0)
            if jitter_window > 0.0:
                return max(0.0, jitter_window)
        try:
            sr = dict(self._fetch("miner/runtime/submission_rate", {}))
            jitter_window = float(sr.get("jitter_window_s", 0.0))
            if network:
                per_network = dict(sr.get("per_network", {}))
                net_cfg = dict(per_network.get(str(network).upper(), {}))
                jitter_window = float(net_cfg.get("jitter_window_s", jitter_window))
        except Exception:
            jitter_window = 0.0
        return max(0.0, jitter_window)

    def _stable_fraction(self, *parts: Any) -> float:
        text = "|".join(str(part) for part in parts)
        acc = 2166136261
        for ch in text:
            acc ^= ord(ch)
            acc = (acc * 16777619) & 0xFFFFFFFF
        return float(acc % 1000) / 999.0

    def _submission_jitter_delay(self, network: str, lane_id: str, share: Dict[str, Any]) -> float:
        try:
            if self.q.qsize() > self._micro_batch_size:
                return 0.0
        except Exception:
            pass

        jitter_window = self._configured_jitter_window_s(network, lane_id=lane_id)
        if jitter_window <= 0.0:
            return 0.0

        allowed = 0.0
        if lane_id:
            worker_record = self._worker_rate_record(network, lane_id)
            allowed = float(worker_record.get("allowed_rate_per_second", 0.0) or 0.0)
        try:
            if allowed <= 0.0:
                sr = dict(self._fetch("miner/runtime/submission_rate", {}))
                per_network = dict(sr.get("per_network", {}))
                net_cfg = dict(per_network.get(str(network).upper(), {}))
                allowed = float(net_cfg.get("allowed_submit_rate", 0.0))
                if allowed <= 0.0:
                    allowed = float(sr.get("allowed_rate_per_second", 0.0))
        except Exception:
            allowed = 0.0
        if allowed <= 0.0:
            return 0.0

        spacing_s = 1.0 / max(1.0e-6, allowed)
        jitter_cap = min(jitter_window, spacing_s * 0.5, self._tick_duration * 0.5)
        if jitter_cap <= 0.0:
            return 0.0

        frac = self._stable_fraction(
            str(network).upper(),
            lane_id,
            share.get("job_id", share.get("height", "")),
            share.get("nonce", ""),
        )
        return jitter_cap * frac

    def _maybe_apply_submission_jitter(self, network: str, lane_id: str, share: Dict[str, Any]) -> None:
        delay_s = self._submission_jitter_delay(network, lane_id, share)
        self._last_jitter_delay_s = delay_s
        if delay_s > 0.0:
            time.sleep(delay_s)

    def _write_submitter_telemetry(self, network: str, throttled: bool) -> None:
        snap = self.governor_snapshot()
        base = "miner/submitter"
        self._store(base + "/paused", bool(snap["paused"]))
        self._store(base + "/queue_depth", int(snap["queue_depth"]))
        self._store(base + "/allowed_rate_per_second", float(snap["allowed_rate_per_second"]))
        self._store(base + "/max_submissions_per_tick", float(snap["max_submissions_per_tick"]))
        self._store(base + "/last_rtt_ms", float(snap["last_rtt_ms"]))
        self._store(base + "/jitter_window_s", float(snap["jitter_window_s"]))
        self._store(base + "/last_jitter_delay_ms", float(snap["last_jitter_delay_ms"]))
        self._store(base + "/throttle_events", int(snap["throttle_events"]))
        self._store(base + "/burst_events", int(snap["burst_events"]))
        self._store(base + "/submissions_this_second", int(snap["submissions_this_second"]))
        self._store(base + "/submissions_this_tick", int(snap["submissions_this_tick"]))
        if throttled:
            self._store("miner/engine/throttle", {"ts": self._now(), "reason": "rate_limit", "queue_depth": snap["queue_depth"]})

    # -------------------------
    # Entropy tracking (per-lane, per-network)
    # -------------------------
    def _update_entropy(self, network: str, lane_id: str, sample_x: int) -> None:
        """
        Welford online variance over 64-bit truncated powhash sample.
        entropy_score approximated by normalized variance in [0,1].
        """
        est_key = "telemetry/metrics/%s/shares/lanes/%s/_entropy_state" % (network, lane_id)
        st = dict(self._fetch(est_key, {}))
        n = int(st.get("n", 0)) + 1
        mean = float(st.get("mean", 0.0))
        M2 = float(st.get("M2", 0.0))
        x = float(sample_x & 0xFFFFFFFFFFFFFFFF)
        if n == 1:
            mean = x
            M2 = 0.0
        else:
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
        st["n"] = n
        st["mean"] = mean
        st["M2"] = M2
        self._store(est_key, st)

        var = float(M2 / (n - 1)) if n > 1 else 0.0
        # Normalize by ideal uniform variance over [0, 2^64)
        try:
            uniform_var = (2.0 ** 64) ** 2 / 12.0
            score = max(0.0, min(1.0, var / uniform_var))
        except Exception:
            score = 0.0
        lane_key = "telemetry/metrics/%s/shares/lanes/%s" % (network, lane_id)
        lane = dict(self._fetch(lane_key, {}))
        lane["entropy_score"] = float(score)
        self._store(lane_key, lane)
