"""
VHW.lane_allocator has been removed.

This module is kept as a stub to prevent import-time failures in legacy
environments. All lane allocation and compute orchestration are handled by
`VHW.compute_manager.ComputeManager`.

Any attempt to instantiate or use LaneAllocator will raise an ImportError.
"""

from __future__ import annotations

class LaneAllocatorRemoved(Exception):
    pass

def __getattr__(name):
    raise ImportError(
        "VHW.lane_allocator is removed; use VHW.compute_manager.ComputeManager"
    )

    # BIOS gate --------------------------------------------------------------
    def _bios_ready(self) -> bool:
        try:
            return bool(self._vsd.get("system/bios_boot_ok", False))
        except Exception as e:
            _LOG.error("_bios_ready check failed: %s", e, exc_info=True)
            return False

    # EventBus wiring --------------------------------------------------------
    def _wire_eventbus(self) -> None:
        try:
            def _on_boot_complete(event=None):
                _LOG.info("boot.complete observed; bios_ready=%s", str(self._bios_ready()))
            self._bus.subscribe("boot.complete", _on_boot_complete)
        except Exception as e:
            _LOG.error("Failed to subscribe to boot.complete: %s", e, exc_info=True)

    def _publish_lane_update(self, lane_id: str, action: str) -> None:
        try:
            self._bus.publish("lane.update", {"ts": _now(), "lane_id": lane_id, "action": action})
        except Exception as e:
            _LOG.error("Failed to publish lane.update: %s", e, exc_info=True)

    # Tier initialization -----------------------------------------------------
    def _init_tiers(self):
        try:
            tiers_cfg = self.cfg.get("tiers") or self._default_tiers()
            base = self.util_cap - DEFAULT_GROUP_HEADROOM
            for t in tiers_cfg:
                tier = TierSpec(
                    tier_id=int(t["tier_id"]),
                    vqram_mb=int(t.get("vqram_mb", 256)),
                    lane_count=0,
                    target_util=_clamp(base, 0.10, 0.95),
                    oscillation_phase=0.0,
                    bandwidth_weight=1.0,
                    latency_budget_ms=_clamp(5.0 + float(t["tier_id"]) * 0.5, 2.0, 25.0)
                )
                self.tiers[tier.tier_id] = tier
        except Exception as e:
            _LOG.error("_init_tiers failed: %s", e, exc_info=True)

    def _default_tiers(self):
        return [{"tier_id": i, "vqram_mb": 256} for i in range(4)]

    # Lane list --------------------------------------------------------------
    def list_active(self) -> List[Dict[str, Any]]:
        out = []
        try:
            for L in self.lanes.values():
                if L.active:
                    out.append({
                        "lane_id": L.lane_id,
                        "tier_id": L.tier_id,
                        "network": L.network,
                        "hashrate_est": L.hashrate_est,
                        "util_est": L.util_est
                    })
        except Exception as e:
            _LOG.error("list_active failed: %s", e, exc_info=True)
        return out

    # Allocation --------------------------------------------------------------
    def allocate_next(self, network: str) -> Optional[Lane]:
        with self._lock:
            try:
                if not self._bios_ready() and not self.initialization_bypass:
                    _LOG.warning("allocate_next blocked: BIOS not ready")
                    return None

                if not self._can_expand(network):
                    return None

                tier = self._choose_tier_for_expand(network)
                if tier is None:
                    return None

                lane_id = self._next_lane_id(network, tier.tier_id)
                lane = Lane(
                    lane_id=lane_id,
                    tier_id=tier.tier_id,
                    network=str(network).upper(),
                    hashrate_est=self._seed_hashrate(network, tier),
                    util_est=self._seed_util(network, tier),
                    active=True,
                    last_update_ts=_now()
                )

                self.lanes[lane.lane_id] = lane
                self.tiers[tier.tier_id].lane_count += 1
                self._publish_lane_update(lane.lane_id, "alloc")
                return lane
            except Exception as e:
                _LOG.error("allocate_next failed: %s", e, exc_info=True)
                return None

    # Rebalance ---------------------------------------------------------------
    def rebalance(self):
        with self._lock:
            try:
                now = _now()
                for tid, t in self._tiers_sorted():
                    t.oscillation_phase = (t.oscillation_phase + 0.017) % 1.0
                    base = self.util_cap - DEFAULT_GROUP_HEADROOM
                    envelope = 0.04
                    t.target_util = _clamp(
                        base + math.sin(2 * math.pi * t.oscillation_phase) * envelope,
                        0.10, 0.95
                    )

                changed = []
                for lane in self.lanes.values():
                    if not lane.active:
                        continue
                    tier = self.tiers.get(lane.tier_id)
                    if not tier:
                        continue
                    err = tier.target_util - lane.util_est
                    new_util = _clamp(lane.util_est + 0.15 * err, 0.05, 0.98)
                    if abs(new_util - lane.util_est) > 1e-6:
                        lane.util_est = new_util
                        changed.append(lane.lane_id)
                    lane.last_update_ts = now

                for lid in changed:
                    self._publish_lane_update(lid, "rebalance")
            except Exception as e:
                _LOG.error("rebalance failed: %s", e, exc_info=True)

    # Pruning -----------------------------------------------------------------
    def recommend_prune_candidates(self, limit: int = 8) -> List[PruneHint]:
        hints = []
        try:
            now = _now()
            for lane in self.lanes.values():
                if not lane.active:
                    continue
                diff = self._effective_difficulty(lane.network, lane)
                uptime = max(1.0, now - lane.last_update_ts)
                health = self._health_score(lane)
                reason = self._reason_for_prune(lane, diff, health)
                hints.append(PruneHint(
                    network=lane.network,
                    lane_id=lane.lane_id,
                    tier_id=lane.tier_id,
                    effective_difficulty=diff,
                    uptime_s=uptime,
                    health_score=health,
                    reason=reason
                ))
            hints.sort(key=lambda h: (-h.effective_difficulty, h.health_score))
            return hints[:limit]
        except Exception as e:
            _LOG.error("recommend_prune_candidates failed: %s", e, exc_info=True)
            return []

    # Network capacity --------------------------------------------------------
    def network_capacity(self) -> Dict[str, float]:
        try:
            if callable(self.network_capacity_fn):
                out = self.network_capacity_fn() or {}
                return {str(k).upper(): float(v) for k, v in out.items()}
        except Exception as e:
            _LOG.error("network_capacity_fn failed: %s", e, exc_info=True)
        try:
            tel = self._read_telemetry()
            caps = tel.get("network_capacity", {})
            return {str(k).upper(): float(v) for k, v in caps.items()}
        except Exception as e:
            _LOG.error("network capacity from telemetry failed: %s", e, exc_info=True)
            return {}

    def _can_expand(self, network: str) -> bool:
        try:
            util = float(self._read_telemetry().get("global_util", 0.0))
            if not (util <= 0.0 and self.initialization_bypass):
                if util >= self.util_cap:
                    return False

            caps = self.network_capacity()
            net = str(network).upper()
            net_cap = float(caps.get(net, 0.0))
            if net_cap > 0.0:
                in_use = self._network_allocated_ratio(net, net_cap)
                if in_use >= self.per_network_cap_ratio:
                    return False

            if len(self.lanes) >= DEFAULT_MAX_LANES:
                return False

            return True
        except Exception as e:
            _LOG.error("_can_expand failed: %s", e, exc_info=True)
            return False

    def _choose_tier_for_expand(self, network: str) -> Optional[TierSpec]:
        try:
            if not self.tiers:
                return None
            items = list(self._tiers_sorted())
            mean = math.sqrt(max(1, sum(t.lane_count for _, t in items)))
            scored = []
            for _, t in items:
                score = ((t.lane_count + 1) / (mean + 1.0)) + \
                        0.25 * t.oscillation_phase + \
                        0.10 / max(0.1, t.bandwidth_weight)
                scored.append((score, t))
            scored.sort(key=lambda x: x[0])
            return scored[0][1] if scored else None
        except Exception as e:
            _LOG.error("_choose_tier_for_expand failed: %s", e, exc_info=True)
            return None

    # Telemetry ---------------------------------------------------------------
    def _read_telemetry(self) -> Dict[str, Any]:
        try:
            if callable(self.telemetry_reader):
                out = self.telemetry_reader() or {}
                self.telemetry = out
                return out
        except Exception as e:
            _LOG.error("_read_telemetry failed: %s", e, exc_info=True)
        return dict(self.telemetry or {})

    def _seed_hashrate(self, network: str, tier: TierSpec) -> float:
        try:
            base = 1.0 + 0.05 * tier.tier_id
            weight = 1.0 + 0.2 * (hash(network) % 3)
            return base * weight
        except Exception as e:
            _LOG.error("_seed_hashrate failed: %s", e, exc_info=True)
            return 1.0

    def _seed_util(self, network: str, tier: TierSpec) -> float:
        try:
            tel = self._read_telemetry()
            util = float(tel.get("global_util", 0.0))
            if util <= 0.0 and self.initialization_bypass:
                vram = float(self.hardware.get("vram_mb", 8192))
                bw = float(self.hardware.get("mem_bw_gbps", 448))
                cpu = float(self.hardware.get("cpu_ghz", 4.0))
                heuristic = math.tanh(
                    (vram / 8192.0) * 0.5 +
                    (bw / 448.0) * 0.35 +
                    (cpu / 4.0) * 0.15
                )
                return _clamp(0.75 * heuristic, 0.10, 0.85)
            return _clamp(util * (0.90 - 0.03 * tier.tier_id), 0.05, 0.92)
        except Exception as e:
            _LOG.error("_seed_util failed: %s", e, exc_info=True)
            return 0.25

    # Difficulty / Health -----------------------------------------------------
    def _tiers_sorted(self):
        return sorted(self.tiers.items(), key=lambda kv: kv[0])

    def _effective_difficulty(self, network: str, lane: Lane) -> float:
        try:
            base = float(self.difficulty_table.get(str(network).upper(), 1.0))
            tier = self.tiers.get(lane.tier_id)
            latency = tier.latency_budget_ms if tier else 10.0
            return base * (1.0 + 0.5 * lane.util_est) * (1.0 + 0.01 * latency)
        except Exception as e:
            _LOG.error("_effective_difficulty failed: %s", e, exc_info=True)
            return 1e6

    def _health_score(self, lane: Lane) -> float:
        try:
            age = max(1.0, _now() - lane.last_update_ts)
            age_penalty = _clamp(math.log10(age + 10.0) / 6.0, 0.0, 0.25)
            util_penalty = _clamp(abs(0.70 - lane.util_est), 0.0, 0.40)
            return _clamp(1.0 - (age_penalty + 0.6 * util_penalty), 0.0, 1.0)
        except Exception as e:
            _LOG.error("_health_score failed: %s", e, exc_info=True)
            return 0.0

    def _reason_for_prune(self, lane: Lane, diff: float, health: float) -> str:
        try:
            rsn = []
            if health < 0.5:
                rsn.append("poor_health")
            if lane.util_est > 0.90:
                rsn.append("over_util")
            if diff > 1e13:
                rsn.append("very_high_difficulty")
            return ",".join(rsn) or "balance"
        except Exception as e:
            _LOG.error("_reason_for_prune failed: %s", e, exc_info=True)
            return "balance"

    def _network_allocated_ratio(self, network: str, network_capacity: float) -> float:
        try:
            s = sum(
                l.hashrate_est for l in self.lanes.values()
                if l.network == network and l.active
            )
            return _clamp(s / max(1e-6, network_capacity), 0.0, 1.0)
        except Exception as e:
            _LOG.error("_network_allocated_ratio failed: %s", e, exc_info=True)
            return 0.0

    def _next_lane_id(self, network: str, tier_id: int) -> str:
        try:
            self._lane_counter += 1
            suffix = _sha256(f"{self.lane_seed}:{network}:{tier_id}:{self._lane_counter}")[:8]
            return f"lane_{tier_id}_{suffix}"
        except Exception as e:
            _LOG.error("_next_lane_id failed: %s", e, exc_info=True)
            return f"lane_err_{int(random.random()*1e6)}"

    # ------------------------------------------------------------------------
    # STATEVECTOR ENCODING
    # ------------------------------------------------------------------------
    def snapshot_statevector(self) -> Dict[str, Any]:
        try:
            vec: List[float] = []
            for _, t in self._tiers_sorted():
                vec.extend([
                    float(t.tier_id) / 64.0 * 2.0 - 1.0,
                    _clamp(t.target_util, 0.0, 1.0) * 2.0 - 1.0,
                    _clamp(t.oscillation_phase, 0.0, 1.0) * 2.0 - 1.0,
                    _clamp(t.bandwidth_weight / 8.0, 0.0, 1.0) * 2.0 - 1.0,
                    _clamp(t.latency_budget_ms / 25.0, 0.0, 1.0) * 2.0 - 1.0
                ])

            for lid in sorted(self.lanes.keys()):
                L = self.lanes[lid]
                vec.extend([
                    _clamp(L.util_est, 0.0, 1.0) * 2.0 - 1.0,
                    _clamp(L.hashrate_est / 1000.0, 0.0, 1.0) * 2.0 - 1.0,
                    (1.0 if L.active else 0.0) * 2.0 - 1.0,
                    _clamp(((_now() - L.last_update_ts) / 60.0), 0.0, 1.0) * 2.0 - 1.0
                ])

            ascii_vec = _ascii_from_vector(vec)
            collapsed, token_table = _collapse_ascii_with_tokens(ascii_vec)

            meta = {
                "type": "json_statevector_v2",
                "encoding": "ascii_floatmap_v1",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lane_count": len(self.lanes),
                "tier_count": len(self.tiers),
                "hash": _sha256(collapsed)
            }

            return {"header": meta, "vector_ascii": collapsed, "token_table": token_table}
        except Exception as e:
            _LOG.error("snapshot_statevector failed: %s", e, exc_info=True)
            return {"header": {}, "vector_ascii": "", "token_table": {}}

    def restore_from_statevector(self, payload: Dict[str, Any]) -> bool:
        try:
            tok = str(payload.get("vector_ascii", ""))
            table = dict(payload.get("token_table", {}))
            _ = _expand_ascii_with_tokens(tok, table)
            return True
        except Exception as e:
            _LOG.error("restore_from_statevector failed: %s", e, exc_info=True)
            return False

    # ------------------------------------------------------------------------
    # SHARE ACCOUNTING
    # ------------------------------------------------------------------------
    def record_share_found(self, lane_id: str, nonces: int = 1) -> None:
        try:
            L = self.lanes.get(lane_id)
            if not L:
                return
            L._found_count = int(getattr(L, "_found_count", 0)) + max(0, int(nonces))
            L._last_found_ts = _now()
        except Exception as e:
            _LOG.error("record_share_found failed: %s", e, exc_info=True)

    def record_share_submitted(self, lane_id: str, nonces: int = 1) -> None:
        try:
            L = self.lanes.get(lane_id)
            if not L:
                return
            L._submitted_count = int(getattr(L, "_submitted_count", 0)) + max(0, int(nonces))
            L._last_submit_ts = _now()
        except Exception as e:
            _LOG.error("record_share_submitted failed: %s", e, exc_info=True)

    def export_hashrates_hs(self) -> Dict[str, Dict[str, float]]:
        now = _now()
        per_net: Dict[str, Dict[str, float]] = {}
        try:
            for L in self.lanes.values():
                if not getattr(L, "active", False):
                    continue
                net = str(getattr(L, "network", "")).upper()
                if not net:
                    continue

                found_tot = int(getattr(L, "_found_count", 0))
                subm_tot = int(getattr(L, "_submitted_count", 0))

                prev_f = int(getattr(L, "_found_prev", 0))
                prev_s = int(getattr(L, "_submitted_prev", 0))
                prev_t = float(getattr(L, "_rate_prev_ts", now))

                dt = max(1e-6, now - prev_t)
                rate_f = max(0.0, (found_tot - prev_f) / dt)
                rate_s = max(0.0, (subm_tot - prev_s) / dt)

                acc = per_net.setdefault(net, {"hashes_found_hs": 0.0, "hashes_submitted_hs": 0.0})
                acc["hashes_found_hs"] += rate_f
                acc["hashes_submitted_hs"] += rate_s

                L._found_prev = found_tot
                L._submitted_prev = subm_tot
                L._rate_prev_ts = now

            return per_net
        except Exception as e:
            _LOG.error("export_hashrates_hs failed: %s", e, exc_info=True)
            return {}

# End of File
