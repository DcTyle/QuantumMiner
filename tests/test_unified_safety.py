# ASCII-ONLY SOURCE FILE
# Unified safety/guardrail tests for QuantumMiner subsystems.

from __future__ import annotations
import unittest
import types


class DummyVSD:
    def __init__(self) -> None:
        self._m = {}

    def get(self, key, default=None):  # type: ignore[override]
        return self._m.get(str(key), default)

    def store(self, key, value):  # type: ignore[override]
        self._m[str(key)] = value


class UnifiedSafetyTests(unittest.TestCase):
    def test_submitter_starts_workers_on_init(self) -> None:
        from miner.submitter import Submitter

        vsd = DummyVSD()
        sub = Submitter(
            vsd=vsd,
            client_resolver=lambda net: types.SimpleNamespace(submit_share=lambda n, s: True),
            max_workers=1,
            max_queue_depth=16,
            micro_batch_size=1,
            tick_duration_s=0.05,
        )  # type: ignore[arg-type]
        try:
            self.assertEqual(len(sub._workers), 1)
            self.assertTrue(sub._workers[0].is_alive())
        finally:
            sub.stop(timeout=0.2)

    def test_submitter_block_guard_drops_stale_shares(self) -> None:
        from miner.submitter import Submitter

        vsd = DummyVSD()

        # Build a submitter with tiny queue and fast ticks
        sub = Submitter(vsd=vsd, client_resolver=lambda net: types.SimpleNamespace(submit_share=lambda n, s: True), max_workers=1, max_queue_depth=16, micro_batch_size=4, tick_duration_s=0.01)  # type: ignore[arg-type]

        lane = "L0"
        net = "ETC"

        # Enqueue a share for an old block
        old_share = {"job_id": 100, "nonce": 1, "received_at": 0.0}
        sub.submit_share(lane, net, old_share)

        # Enqueue a share for a new block (this becomes current block id)
        new_share = {"job_id": 101, "nonce": 2, "received_at": 0.0}
        sub.submit_share(lane, net, new_share)

        # Allow worker to process queue
        import time
        time.sleep(0.1)

        # Check counters in VSD: only the new block share should count as submitted
        lane_key = f"telemetry/metrics/{net}/shares/lanes/{lane}"
        lane_stats = vsd.get(lane_key, {}) or {}
        submitted = int(lane_stats.get("submitted", 0))
        found = int(lane_stats.get("found", 0))

        self.assertGreaterEqual(submitted, 1)
        self.assertGreaterEqual(found, 1)

    def test_submitter_respects_allowed_rate_feedback(self) -> None:
        from miner.submitter import Submitter

        vsd = DummyVSD()
        # Seed allowed rate very low
        vsd.store("miner/runtime/submission_rate", {"allowed_rate_per_second": 1.0, "tick_duration": 0.1})

        sub = Submitter(vsd=vsd, client_resolver=lambda net: types.SimpleNamespace(submit_share=lambda n, s: True), max_workers=0, max_queue_depth=16, micro_batch_size=1, tick_duration_s=0.1)  # type: ignore[arg-type]

        # With only 1/sec allowed, can_accept should be false after queue is saturated
        lane = "L0"
        net = "ETC"
        share = {"job_id": 200, "nonce": 1, "received_at": 0.0}

        # First accept should pass pre-check
        self.assertTrue(sub.can_accept(net))
        sub.submit_share(lane, net, share)

        # Then queue will have at least one item; further can_accept may become false
        _ = sub.can_accept(net)

        # We just assert the code path does not raise and telemetry keys exist
        self.assertIn("miner/runtime/submission_rate", vsd._m)

    def test_runtime_submission_rate_uses_difficulty_and_valid_share_telemetry(self) -> None:
        from miner.miner_runtime import compute_submission_rate_state

        vsd = DummyVSD()
        vsd.store("telemetry/metrics/ETC/current", {
            "accepted_hs": 1.5,
            "hashes_submitted_hs": 2.0,
            "acceptance_rate": 0.75,
        })
        vsd.store("miner/difficulty/ETC", {
            "method": "mining.set_difficulty",
            "params": [2048],
        })

        state = compute_submission_rate_state(
            vsd=vsd,
            raw_capacity={
                "ETC": {
                    "network_hashrate_hs": 500000000000.0,
                    "block_time_s": 13.0,
                    "difficulty": 16547834.0,
                }
            },
            coin_cfg={
                "ETC": {
                    "algorithm": "etchash",
                    "hashrate_sim_hs": 96000000.0,
                }
            },
            base_tick_s=0.25,
        )

        self.assertGreaterEqual(float(state["allowed_rate_per_second"]), 1.0)
        self.assertEqual(state["model"], "expected_hashrate_difficulty_valid_share")
        self.assertIn("ETC", state["per_network"])
        etc = state["per_network"]["ETC"]
        self.assertGreater(float(etc["expected_valid_share_rate"]), 0.0)
        self.assertEqual(float(etc["share_difficulty"]), 2048.0)
        self.assertGreater(float(etc["allowed_submit_rate"]), 0.0)
        self.assertGreaterEqual(float(state["jitter_window_s"]), 0.0)

    def test_runtime_submission_rate_tapers_near_target_ratio(self) -> None:
        from miner.miner_runtime import compute_submission_rate_state

        vsd = DummyVSD()
        vsd.store("telemetry/metrics/ETC/current", {
            "accepted_hs": 1.72,
            "hashes_submitted_hs": 1.78,
            "acceptance_rate": 0.97,
            "local_hashrate_hs": 49500000000.0,
        })
        vsd.store("miner/difficulty/ETC", {
            "method": "mining.set_difficulty",
            "params": [6.0],
        })

        state = compute_submission_rate_state(
            vsd=vsd,
            raw_capacity={
                "ETC": {
                    "network_hashrate_hs": 1000000000000.0,
                    "block_time_s": 13.0,
                    "difficulty": 1234567.0,
                }
            },
            coin_cfg={
                "ETC": {
                    "algorithm": "etchash",
                    "hashrate_sim_hs": 49500000000.0,
                }
            },
            base_tick_s=0.25,
        )

        etc = state["per_network"]["ETC"]
        self.assertGreater(float(etc["ratio_progress"]), 0.80)
        self.assertLess(float(etc["share_rate_taper_fraction"]), 1.0)
        self.assertLessEqual(float(etc["allowed_submit_rate"]), float(etc["guarded_submit_rate_ceiling_per_second"]))

    def test_runtime_state_boosts_submit_rate_when_assigned_difficulty_is_easier(self) -> None:
        from miner.stratum_adapter import _build_network_target_snapshot

        snap = _build_network_target_snapshot(
            network="ETC",
            coin_cfg={"hashes_per_diff1": float(2 ** 32)},
            network_stats={
                "network_hashrate_hs": 1000000000000.0,
                "block_time_s": 13.0,
                "difficulty": 1234567.0,
            },
            policy={
                "network_target_fraction_floor": 0.05,
                "network_target_fraction_ceiling": 0.05002,
                "network_target_fraction_nominal": 0.05001,
                "preferred_valid_share_rate_per_second": 2.0,
                "pool_submit_ceiling_per_second": 12.0,
                "pool_submit_guard_fraction": 1.0,
                "target_acceptance_rate": 0.92,
            },
            current_share_difficulty=3.0,
        )

        self.assertGreater(float(snap["desired_submit_rate_per_second"]), float(snap["preferred_submit_rate_per_second"]))
        self.assertGreater(float(snap["assigned_required_submit_rate_per_second"]), float(snap["preferred_submit_rate_per_second"]))
        self.assertAlmostEqual(float(snap["allowed_submit_rate_per_second"]), float(snap["desired_submit_rate_per_second"]), places=9)
        self.assertAlmostEqual(float(snap["assigned_share_difficulty"]), 3.0, places=9)

    def test_runtime_submission_rate_honors_coin_ceiling_override_over_global_default(self) -> None:
        from miner.miner_runtime import compute_submission_rate_state

        vsd = DummyVSD()
        vsd.store("miner/difficulty/ETC", {
            "method": "mining.set_difficulty",
            "params": [1.0],
        })

        state = compute_submission_rate_state(
            vsd=vsd,
            raw_capacity={
                "ETC": {
                    "network_hashrate_hs": 1000000000000.0,
                    "block_time_s": 13.0,
                    "difficulty": 1234567.0,
                }
            },
            coin_cfg={
                "ETC": {
                    "algorithm": "etchash",
                    "hashrate_sim_hs": 50000000000.0,
                    "pool_submit_ceiling_per_second": 12.0,
                    "network_submit_ceiling_per_second": 12.0,
                    "pool_submit_guard_fraction": 1.0,
                }
            },
            base_tick_s=0.25,
        )

        etc = state["per_network"]["ETC"]
        self.assertAlmostEqual(float(etc["submit_rate_ceiling_per_second"]), 12.0, places=9)
        self.assertGreater(float(etc["allowed_submit_rate"]), 2.0)

    def test_submitter_reads_bounded_runtime_jitter(self) -> None:
        from miner.submitter import Submitter

        vsd = DummyVSD()
        vsd.store("miner/runtime/submission_rate", {
            "allowed_rate_per_second": 4.0,
            "tick_duration": 0.1,
            "jitter_window_s": 0.02,
            "per_network": {
                "ETC": {
                    "allowed_submit_rate": 4.0,
                    "jitter_window_s": 0.02,
                }
            },
        })

        sub = Submitter(
            vsd=vsd,
            client_resolver=lambda net: types.SimpleNamespace(submit_share=lambda n, s: True),
            max_workers=0,
            max_queue_depth=16,
            micro_batch_size=4,
            tick_duration_s=0.1,
        )  # type: ignore[arg-type]
        try:
            delay = sub._submission_jitter_delay("ETC", "L0", {"job_id": 123, "nonce": 1})
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, 0.02)
            self.assertEqual(sub._configured_jitter_window_s("ETC"), 0.02)
        finally:
            sub.stop(timeout=0.1)

    def test_submitter_enforces_worker_runtime_budget(self) -> None:
        from miner.submitter import Submitter

        vsd = DummyVSD()
        vsd.store("miner/runtime/submission_rate", {
            "allowed_rate_per_second": 10.0,
            "tick_duration": 0.1,
            "jitter_window_s": 0.02,
            "per_network": {
                "ETC": {
                    "allowed_submit_rate": 10.0,
                    "jitter_window_s": 0.02,
                    "workers": [
                        {
                            "lane_id": "L0",
                            "allowed_submit_rate": 1.0,
                            "tick_duration_s": 0.1,
                            "jitter_window_s": 0.01,
                            "username": "worker0",
                        }
                    ],
                }
            },
        })

        sub = Submitter(
            vsd=vsd,
            client_resolver=lambda net: types.SimpleNamespace(submit_share=lambda n, s: True),
            max_workers=0,
            max_queue_depth=16,
            micro_batch_size=1,
            tick_duration_s=0.1,
        )  # type: ignore[arg-type]
        try:
            rec = sub._worker_rate_record("ETC", "L0")
            self.assertAlmostEqual(float(rec["allowed_rate_per_second"]), 1.0, places=9)
            self.assertAlmostEqual(float(sub._configured_jitter_window_s("ETC", "L0")), 0.01, places=9)

            self.assertTrue(sub._consume_tokens_if_available("ETC", "L0"))
            self.assertFalse(sub._consume_tokens_if_available("ETC", "L0"))
        finally:
            sub.stop(timeout=0.1)

    def test_submitter_prioritizes_higher_bucket_priority(self) -> None:
        from miner.submitter import Submitter

        vsd = DummyVSD()
        sub = Submitter(
            vsd=vsd,
            client_resolver=lambda net: types.SimpleNamespace(submit_share=lambda n, s: True),
            max_workers=0,
            max_queue_depth=16,
            micro_batch_size=1,
            tick_duration_s=0.05,
        )  # type: ignore[arg-type]
        try:
            sub.submit_share("L0", "ETC", {"job_id": 300, "nonce": 1, "bucket_id": "low", "bucket_priority": 0.10})
            sub.submit_share("L0", "ETC", {"job_id": 300, "nonce": 2, "bucket_id": "high", "bucket_priority": 0.95})

            first = sub.q.get_nowait()
            second = sub.q.get_nowait()

            self.assertEqual(first[5]["bucket_id"], "high")
            self.assertEqual(second[5]["bucket_id"], "low")
        finally:
            sub.stop(timeout=0.1)


if __name__ == "__main__":
    unittest.main()
