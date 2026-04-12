# ASCII-ONLY
import unittest


class StratumCalibrationTests(unittest.TestCase):
    def test_coin_switcher_builds_canonical_worker_profiles(self) -> None:
        from miner.coin_switcher import resolve_coin_profile

        coins = {
            "BTC": {
                "wallet": "dax97625.rig",
                "pool_url": "stratum+tcp://btc.example.com:4444",
                "mode": "stratum",
                "algorithm": "sha256d",
                "port": 4444,
            },
            "ETC": {
                "wallet": "0xabc",
                "pool_url": "stratum+tcp://etc.example.com:1010",
                "mode": "stratum",
                "algorithm": "etchash",
                "port": 1010,
            },
            "RVN": {
                "wallet": "RWallet",
                "pool_url": "stratum+tcp://rvn.example.com:6060",
                "mode": "stratum",
                "algorithm": "kawpow",
                "port": 6060,
            },
        }

        btc = resolve_coin_profile(coins, "BTC", worker_index=2)
        etc = resolve_coin_profile(coins, "ETC", worker_index=3)
        rvn = resolve_coin_profile(coins, "RVN", worker_index=4)

        self.assertEqual(btc.username, "dax97625.rig2")
        self.assertEqual(btc.authorize_method, "mining.authorize")
        self.assertEqual(etc.username, "0xabc.quantumRig3")
        self.assertEqual(etc.authorize_method, "eth_submitLogin")
        self.assertEqual(rvn.username, "RWallet.quantumRig4")
        self.assertEqual(rvn.authorize_method, "mining.authorize")

    def test_coin_switcher_honors_profile_overrides(self) -> None:
        from miner.coin_switcher import resolve_coin_profile

        profile = resolve_coin_profile(
            {
                "ETC": {
                    "wallet": "0xabc",
                    "pool_url": "stratum+tcp://etc.example.com:1010",
                    "mode": "stratum",
                    "algorithm": "etchash",
                    "port": 1010,
                    "authorize_method": "mining.authorize",
                    "authorize_with_password": True,
                    "password": "secret",
                }
            },
            "ETC",
            worker_index=1,
        )

        self.assertEqual(profile.authorize_method, "mining.authorize")
        self.assertTrue(profile.authorize_with_password)
        self.assertEqual(profile.password, "secret")

    def test_target_snapshot_uses_five_percent_network_hashrate(self) -> None:
        from miner.stratum_adapter import _build_network_target_snapshot

        policy = {
            "network_target_fraction_floor": 0.05,
            "network_target_fraction_ceiling": 0.05002,
            "network_target_fraction_nominal": 0.05001,
            "preferred_valid_share_rate_per_second": 2.0,
            "pool_submit_ceiling_per_second": 2.0,
            "pool_submit_guard_fraction": 0.985,
            "target_acceptance_rate": 0.92,
        }
        snap = _build_network_target_snapshot(
            network="ETC",
            coin_cfg={"hashes_per_diff1": float(2 ** 32)},
            network_stats={
                "network_hashrate_hs": 1000000000000.0,
                "block_time_s": 13.0,
                "difficulty": 1234567.0,
            },
            policy=policy,
        )

        self.assertEqual(snap["network"], "ETC")
        self.assertAlmostEqual(float(snap["target_fraction_floor"]), 0.05, places=9)
        self.assertAlmostEqual(float(snap["target_fraction_ceiling"]), 0.05002, places=9)
        self.assertAlmostEqual(float(snap["target_hashrate_hs"]), 50010000000.0, places=3)
        self.assertAlmostEqual(float(snap["target_hashrate_floor_hs"]), 50000000000.0, places=3)
        self.assertAlmostEqual(float(snap["target_hashrate_ceiling_hs"]), 50020000000.0, places=3)
        self.assertAlmostEqual(float(snap["allowed_submit_rate_per_second"]), 1.97, places=9)
        self.assertAlmostEqual(
            float(snap["target_share_difficulty"]),
            float(snap["target_hashrate_hs"]) / (float(snap["target_valid_share_rate_per_second"]) * float(2 ** 32)),
            places=9,
        )
        self.assertEqual(str(snap["control_action"]), "bootstrap")
        self.assertTrue(bool(snap["active"]))

    def test_target_snapshot_tapers_submit_rate_near_target_ratio(self) -> None:
        from miner.stratum_adapter import _build_network_target_snapshot

        policy = {
            "network_target_fraction_floor": 0.05,
            "network_target_fraction_ceiling": 0.05002,
            "network_target_fraction_nominal": 0.05001,
            "pool_submit_ceiling_per_second": 2.0,
            "pool_submit_guard_fraction": 0.985,
            "target_acceptance_rate": 0.92,
            "share_rate_taper_start_ratio": 0.80,
            "share_rate_taper_floor_fraction": 0.85,
            "share_rate_taper_power": 1.35,
        }
        snap = _build_network_target_snapshot(
            network="ETC",
            coin_cfg={"hashes_per_diff1": float(2 ** 32)},
            network_stats={
                "network_hashrate_hs": 1000000000000.0,
                "block_time_s": 13.0,
                "difficulty": 1234567.0,
            },
            policy=policy,
            metrics={
                "accepted_hs": 1.72,
                "hashes_submitted_hs": 1.78,
            },
            current_share_difficulty=6.0,
        )

        self.assertLess(float(snap["allowed_submit_rate_per_second"]), 1.97)
        self.assertGreater(float(snap["allowed_submit_rate_per_second"]), 1.65)
        self.assertGreater(float(snap["ratio_progress"]), 0.80)
        self.assertLess(float(snap["share_rate_taper_fraction"]), 1.0)

    def test_target_snapshot_increases_submit_rate_for_easier_assigned_difficulty(self) -> None:
        from miner.stratum_adapter import _build_network_target_snapshot

        policy = {
            "network_target_fraction_floor": 0.05,
            "network_target_fraction_ceiling": 0.05002,
            "network_target_fraction_nominal": 0.05001,
            "preferred_valid_share_rate_per_second": 2.0,
            "pool_submit_ceiling_per_second": 12.0,
            "pool_submit_guard_fraction": 1.0,
            "target_acceptance_rate": 0.92,
        }
        snap = _build_network_target_snapshot(
            network="ETC",
            coin_cfg={"hashes_per_diff1": float(2 ** 32)},
            network_stats={
                "network_hashrate_hs": 1000000000000.0,
                "block_time_s": 13.0,
                "difficulty": 1234567.0,
            },
            policy=policy,
            current_share_difficulty=3.0,
        )

        self.assertGreater(float(snap["share_difficulty_gap_ratio"]), 1.0)
        self.assertGreater(float(snap["assigned_required_submit_rate_per_second"]), float(snap["preferred_submit_rate_per_second"]))
        self.assertAlmostEqual(
            float(snap["allowed_submit_rate_per_second"]),
            float(snap["assigned_required_submit_rate_per_second"]),
            places=9,
        )
        self.assertGreater(float(snap["allowed_submit_rate_per_second"]), 1.97)

    def test_runtime_state_uses_target_valid_share_rate(self) -> None:
        from miner.stratum_adapter import _build_runtime_submission_state

        targets = {
            "ETC": {
                "network": "ETC",
                "active": True,
                "target_valid_share_rate_per_second": 1.8124,
                "target_share_difficulty": 6.0,
                "target_hashrate_hs": 50010000000.0,
                "allowed_submit_rate_per_second": 1.97,
            }
        }
        policy = {
            "target_acceptance_rate": 0.92,
            "fallback_allowed_rate_per_second": 2.0,
            "min_tick_duration_s": 0.05,
            "jitter_fraction": 0.18,
        }
        state = _build_runtime_submission_state(targets, policy, 0.25)

        self.assertEqual(state["model"], "network_fraction_exact_band_v2")
        self.assertAlmostEqual(float(state["allowed_rate_per_second"]), 1.97, places=9)
        self.assertIn("ETC", state["per_network"])
        self.assertGreaterEqual(float(state["jitter_window_s"]), 0.0)

    def test_runtime_state_distributes_worker_budgets_per_lane(self) -> None:
        from miner.stratum_adapter import _build_runtime_submission_state

        state = _build_runtime_submission_state(
            targets={
                "BTC": {
                    "network": "BTC",
                    "active": True,
                    "target_valid_share_rate_per_second": 1.8124,
                    "target_hashrate_hs": 50010000000.0,
                    "allowed_submit_rate_per_second": 1.97,
                }
            },
            policy={
                "target_acceptance_rate": 0.92,
                "fallback_allowed_rate_per_second": 2.0,
                "min_tick_duration_s": 0.05,
                "jitter_fraction": 0.18,
            },
            tick_duration_s=0.25,
            worker_map={
                "lane_btc_1": {"coin": "BTC", "worker_index": 1, "username": "dax97625.rig1"},
                "lane_btc_2": {"coin": "BTC", "worker_index": 2, "username": "dax97625.rig2"},
            },
        )

        btc = state["per_network"]["BTC"]
        self.assertEqual(int(btc["worker_count"]), 2)
        self.assertEqual(len(list(btc["workers"])), 2)
        self.assertAlmostEqual(float(btc["workers"][0]["allowed_submit_rate"]), 0.985, places=9)
        self.assertAlmostEqual(float(btc["workers"][0]["target_hashrate_hs"]), 25005000000.0, places=3)

    def test_outgoing_share_contract_preserves_mix_fields(self) -> None:
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork, neural_objectSchema

        etc_packet = neural_objectPacket(
            packet_type=variable_format_enum.ETC_EtchashTemplate,
            network=ComputeNetwork.ETC,
            raw_payload={},
            system_payload={
                "nonce": 7,
                "lane": "lane_etc",
                "received_at": 1.5,
                "mix_hash": "0xfeedbeef",
                "header_hash": "0xabc123",
                "bucket_id": "b1",
                "worker_bucket": "p01-t01-i01",
                "bucket_priority": 0.91,
                "job": {"job_id": "etc_job", "header_hash": "0xabc123"},
            },
            metadata={},
            derived_state={},
        )
        rvn_packet = neural_objectPacket(
            packet_type=variable_format_enum.RVN_KawpowJob,
            network=ComputeNetwork.RVN,
            raw_payload={},
            system_payload={
                "nonce": 9,
                "lane": "lane_rvn",
                "received_at": 2.5,
                "mix_hash": "0xdeadbeef",
                "header_hash": "0xrvnheader",
                "bucket_id": "b2",
                "worker_bucket": "p02-t02-i02",
                "bucket_priority": 0.88,
                "job": {"job_id": "rvn_job", "header": "0xrvnheader", "target": "0x01"},
            },
            metadata={},
            derived_state={},
        )

        etc_share = neural_objectSchema[variable_format_enum.ETC_EtchashTemplate]["convert_outgoing"](etc_packet)
        rvn_share = neural_objectSchema[variable_format_enum.RVN_KawpowJob]["convert_outgoing"](rvn_packet)

        self.assertEqual(etc_share["mix_hash"], "0xfeedbeef")
        self.assertEqual(etc_share["header_hash"], "0xabc123")
        self.assertEqual(rvn_share["mix_hash"], "0xdeadbeef")
        self.assertEqual(rvn_share["header"], "0xrvnheader")

    def test_stratum_adapter_uses_stateful_adapters_for_job_normalization(self) -> None:
        """Regression test: ensure stratum adapter uses stateful coin adapters for live job normalization."""
        from miner.stratum_adapter import StratumAdapter
        from bios.event_bus import get_event_bus

        # Mock VSD
        class MockVSD:
            def __init__(self):
                self.data = {}
            def get(self, key, default=None):
                return self.data.get(key, default)
            def store(self, key, value):
                self.data[key] = value

        # Mock Submitter
        class MockSubmitter:
            pass

        engine = type("Engine", (), {"cm": type("CM", (), {"lanes": {"lane0": object()}})()})()

        vsd = MockVSD()
        submitter = MockSubmitter()
        bus = get_event_bus()

        # Create adapter with BTC config
        cfg = {
            "coins": {
                "BTC": {
                    "stratum": {"host": "test.com", "port": 3333},
                    "wallet_address": "test_wallet"
                }
            }
        }

        adapter = StratumAdapter(
            engine=engine,
            vsd=vsd,
            submitter=submitter,
            config_path="",  # Will use provided cfg
            bus=bus
        )
        adapter.cfg = cfg  # Override loaded config
        adapter._assign_lanes_round_robin()

        # Initialize clients (this should create stateful adapters)
        adapter._init_clients()
        session_id = adapter._lane_session_id["lane0"]

        # Verify BTC adapter was created
        self.assertIn(session_id, adapter.coin_adapters)
        btc_adapter = adapter.coin_adapters[session_id]
        self.assertIsNotNone(btc_adapter)

        # Mock a stratum job event
        job_payload = {
            "coin": "BTC",
            "lane_id": "lane0",
            "session_id": session_id,
            "job": {
                "method": "mining.notify",
                "params": [
                    "test_job_123",
                    "0000000000000000000000000000000000000000000000000000000000000000",  # prevhash
                    "01000000",  # coinb1
                    "ffffffff",  # coinb2
                    [],  # merkle
                    "20000000",  # version
                    "1d00ffff",  # nbits
                    "00000000"   # ntime
                ]
            }
        }

        # Process the job (this should use the stateful adapter)
        adapter._on_stratum_job(job_payload)

        # Verify job was stored
        self.assertIn("lane0", adapter._last_job_packet)
        packet = adapter._last_job_packet["lane0"]
        self.assertIsNotNone(packet)

        # Verify the job normalization included adapter state (like extranonce)
        job_dict = packet.raw_payload
        self.assertIn("job_id", job_dict)
        self.assertEqual(job_dict["job_id"], "test_job_123")
        # Verify header was constructed (indicating adapter was used)
        self.assertIn("header_hex", job_dict)
        self.assertTrue(len(job_dict.get("header_hex", "")) > 0)

    def test_stratum_job_inherits_live_share_target_from_difficulty(self) -> None:
        from miner.stratum_adapter import StratumAdapter, _target_hex_from_difficulty
        from bios.event_bus import get_event_bus

        class MockVSD:
            def __init__(self):
                self.data = {
                    "miner/difficulty/BTC": {
                        "method": "mining.set_difficulty",
                        "params": [32.0],
                    }
                }

            def get(self, key, default=None):
                return self.data.get(key, default)

            def store(self, key, value):
                self.data[key] = value

        class MockSubmitter:
            client_resolver = None

        adapter = StratumAdapter(
            engine=type("Engine", (), {"cm": type("CM", (), {"lanes": {"lane0": object()}})()})(),
            vsd=MockVSD(),
            submitter=MockSubmitter(),
            config_path="",
            bus=get_event_bus(),
        )
        adapter.cfg = {
            "coins": {
                "BTC": {
                    "stratum": {"host": "test.com", "port": 3333},
                    "wallet_address": "test_wallet",
                }
            }
        }
        adapter._assign_lanes_round_robin()
        adapter._init_clients()
        session_id = adapter._lane_session_id["lane0"]

        adapter._on_stratum_job({
            "coin": "BTC",
            "lane_id": "lane0",
            "session_id": session_id,
            "job": {
                "method": "mining.notify",
                "params": [
                    "test_job_123",
                    "0000000000000000000000000000000000000000000000000000000000000000",
                    "01000000",
                    "ffffffff",
                    [],
                    "20000000",
                    "1d00ffff",
                    "00000000",
                ],
            },
        })

        packet = adapter._last_job_packet["lane0"]
        self.assertEqual(packet.raw_payload["share_target"], _target_hex_from_difficulty(32.0))
        self.assertEqual(float(packet.raw_payload["current_share_difficulty"]), 32.0)
        self.assertEqual(packet.system_payload["job"]["share_target"], _target_hex_from_difficulty(32.0))

    def test_etc_job_uses_job_target_as_current_share_difficulty(self) -> None:
        from miner.stratum_adapter import StratumAdapter, _difficulty_from_target
        from miner.stratum_adapters import ETCAdapter
        from bios.event_bus import get_event_bus

        class MockVSD:
            def __init__(self):
                self.data = {}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def store(self, key, value):
                self.data[key] = value

        class MockSubmitter:
            client_resolver = None

        adapter = StratumAdapter(
            engine=type("Engine", (), {"cm": type("CM", (), {"lanes": {"lane0": object()}})()})(),
            vsd=MockVSD(),
            submitter=MockSubmitter(),
            config_path="",
            bus=get_event_bus(),
        )
        adapter.cfg = {
            "coins": {
                "ETC": {
                    "pool_url": "stratum+tcp://etc.example.com:1010",
                    "wallet": "0xabc",
                }
            }
        }
        adapter._lane_map["lane0"] = "ETC"
        adapter._lane_worker_index["lane0"] = 1
        adapter._worker_map["lane0"] = {"coin": "ETC", "worker_index": 1, "username": "0xabc.quantumRig1", "session_id": "ETC:lane0:0xabc.quantumRig1"}
        adapter._lane_session_id["lane0"] = "ETC:lane0:0xabc.quantumRig1"
        adapter.coin_adapters["ETC:lane0:0xabc.quantumRig1"] = ETCAdapter()

        share_target = "00000000ffff0000000000000000000000000000000000000000000000000000"
        adapter._on_stratum_job({
            "coin": "ETC",
            "lane_id": "lane0",
            "session_id": "ETC:lane0:0xabc.quantumRig1",
            "job": {
                "params": ["etc_job", "0xheader", "0xseed", share_target],
            },
        })

        packet = adapter._last_job_packet["lane0"]
        self.assertEqual(packet.raw_payload["share_target"], share_target)
        self.assertAlmostEqual(
            float(packet.raw_payload["current_share_difficulty"]),
            float(_difficulty_from_target(share_target)),
            places=9,
        )

    def test_stratum_adapter_resolver_uses_pool_url_wallet_schema(self) -> None:
        from miner import stratum_adapter as adapter_module
        from miner.stratum_adapter import StratumAdapter
        from bios.event_bus import get_event_bus

        class MockVSD:
            def get(self, key, default=None):
                return default

            def store(self, key, value):
                return None

        class MockSubmitter:
            client_resolver = None

        class MockEngine:
            pass

        calls = {}

        class FakeBTCPoolClient:
            def __init__(self, endpoint, username, password):
                calls["endpoint"] = endpoint
                calls["username"] = username
                calls["password"] = password

        old_client = adapter_module.BTCPoolClient
        adapter_module.BTCPoolClient = FakeBTCPoolClient
        try:
            adapter = StratumAdapter(
                engine=MockEngine(),
                vsd=MockVSD(),
                submitter=MockSubmitter(),
                config_path="",
                bus=get_event_bus(),
            )
            adapter.cfg = {
                "coins": {
                    "BTC": {
                        "wallet": "dax97625.rig",
                        "pool_url": "stratum+tcp://btc.example.com:4444",
                        "mode": "stratum",
                        "algorithm": "sha256d",
                        "port": 4444,
                    }
                }
            }
            adapter._lane_worker_index["lane_btc"] = 2

            client = adapter._client_resolver("BTC", "lane_btc")

            self.assertIsInstance(client, FakeBTCPoolClient)
            self.assertEqual(calls["endpoint"], "btc.example.com:4444")
            self.assertEqual(calls["username"], "dax97625.rig2")
            self.assertEqual(calls["password"], "x")
        finally:
            adapter_module.BTCPoolClient = old_client

    def test_stratum_adapter_requests_difficulty_per_worker_session(self) -> None:
        from miner.stratum_adapter import StratumAdapter
        from bios.event_bus import get_event_bus

        class MockVSD:
            def __init__(self):
                self.data = {}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def store(self, key, value):
                self.data[key] = value

        class MockSubmitter:
            client_resolver = None

        class FakeClient:
            def __init__(self, session_id):
                self.session_id = session_id
                self.calls = []

            def is_connected(self):
                return True

            def suggest_difficulty(self, difficulty):
                self.calls.append(("mining.suggest_difficulty", float(difficulty)))
                return True

        vsd = MockVSD()
        adapter = StratumAdapter(
            engine=type("Engine", (), {"cm": type("CM", (), {"lanes": {"lane0": object(), "lane1": object()}})()})(),
            vsd=vsd,
            submitter=MockSubmitter(),
            config_path="",
            bus=get_event_bus(),
        )
        adapter.cfg = {
            "coins": {
                "BTC": {
                    "wallet": "dax97625.rig",
                    "pool_url": "stratum+tcp://btc.example.com:4444",
                    "mode": "stratum",
                    "algorithm": "sha256d",
                    "port": 4444,
                    "difficulty_control_method": "suggest_difficulty",
                }
            }
        }
        adapter._assign_lanes_round_robin()
        session0 = adapter._lane_session_id["lane0"]
        session1 = adapter._lane_session_id["lane1"]
        adapter.clients[session0] = FakeClient(session0)
        adapter.clients[session1] = FakeClient(session1)
        vsd.store("miner/runtime/submission_rate/workers/lane0", {"allowed_rate_per_second": 1.0, "username": "dax97625.rig1"})
        vsd.store("miner/runtime/submission_rate/workers/lane1", {"allowed_rate_per_second": 1.0, "username": "dax97625.rig2"})

        adapter._maybe_request_share_difficulty("BTC", {
            "target_share_difficulty": 64.0,
            "target_hashrate_hs": 1000.0,
            "network_hashrate_hs": 20000.0,
            "target_fraction": 0.05,
            "target_fraction_floor": 0.05,
            "target_fraction_ceiling": 0.05002,
            "allowed_submit_rate_per_second": 2.0,
            "control_action": "hold",
        })

        self.assertEqual(adapter.clients[session0].calls, [("mining.suggest_difficulty", 64.0)])
        self.assertEqual(adapter.clients[session1].calls, [("mining.suggest_difficulty", 64.0)])
        self.assertTrue(vsd.get("miner/difficulty_request/BTC/workers/lane0", {}).get("requested"))
        self.assertTrue(vsd.get("miner/difficulty_request/BTC/workers/lane1", {}).get("requested"))


if __name__ == "__main__":
    unittest.main()
