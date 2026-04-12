# ASCII-ONLY
from __future__ import annotations

import time
import types
import unittest


class _BusStub:
    def __init__(self) -> None:
        self.events = []

    def publish(self, topic, payload=None):
        self.events.append((str(topic), payload))

    def subscribe(self, topic, handler=None, once=False, priority=0, name=None):
        return None


class _VSDStub:
    def __init__(self) -> None:
        self._kv = {}

    def get(self, key, default=None):
        return self._kv.get(key, default)

    def store(self, key, value):
        self._kv[key] = value


class _ComputeManagerStub:
    def __init__(self) -> None:
        self.dispatch_count = 0

    def dispatch(self, wrapper):
        self.dispatch_count += 1
        return {}

    def status(self):
        return {}


class _ShareAllocatorStub:
    def submit_share(self, lane_id, net, nonce_hex, hash_hex, target_hex, is_valid, extra):
        return None

    def counters(self):
        return {}


class MinerPauseFlowTests(unittest.TestCase):
    def test_command_registry_pause_publishes_pause_event(self) -> None:
        from Control_Center.command_registry import CommandRegistry, Command

        bus = _BusStub()
        vsd = _VSDStub()
        registry = CommandRegistry(bus, vsd)

        registry.run_enum(Command.PAUSE_MINER, {"note": "Pause requested by test"})

        topics = [topic for topic, _payload in bus.events]
        self.assertIn("control_center.cmd.pause_miner", topics)
        payload = next(payload for topic, payload in bus.events if topic == "control_center.cmd.pause_miner")
        self.assertEqual(str(payload.get("note", "")), "Pause requested by test")

    def test_submitter_pause_blocks_new_submissions(self) -> None:
        from miner.submitter import Submitter

        vsd = _VSDStub()
        submitter = Submitter(
            vsd=vsd,
            client_resolver=lambda net, lane_id=None: types.SimpleNamespace(submit_share=lambda n, s: True),
            max_workers=0,
            max_queue_depth=16,
            micro_batch_size=1,
            tick_duration_s=0.05,
        )  # type: ignore[arg-type]
        try:
            self.assertTrue(submitter.pause(note="test pause", source="unit_test", drain_timeout_s=0.0))
            self.assertTrue(submitter.is_paused())
            self.assertFalse(submitter.can_accept("ETC"))
            submitter.submit_share("L0", "ETC", {"job_id": 1, "nonce": 99})
            self.assertEqual(submitter.q.qsize(), 0)
            state = dict(vsd.get("miner/control/submitter", {}) or {})
            self.assertTrue(bool(state.get("paused", False)))
        finally:
            submitter.stop(timeout=0.1)

    def test_stratum_adapter_drops_jobs_while_paused(self) -> None:
        from miner.stratum_adapter import StratumAdapter

        class _SubmitterStub:
            client_resolver = None

        adapter = StratumAdapter(
            engine=types.SimpleNamespace(cm=types.SimpleNamespace(lanes={})),
            vsd=_VSDStub(),
            submitter=_SubmitterStub(),
            config_path="",
            bus=_BusStub(),
        )
        adapter.pause(note="test pause", source="unit_test")
        adapter._on_stratum_job({"coin": "BTC", "job": {"method": "mining.notify", "params": []}})

        self.assertEqual(adapter._last_job_packet, {})

    def test_miner_engine_pause_stops_dispatch(self) -> None:
        from miner.miner_engine import MinerEngine

        cm = _ComputeManagerStub()
        engine = MinerEngine(
            compute_manager=cm,
            share_allocator=_ShareAllocatorStub(),
            submitter=types.SimpleNamespace(can_accept=lambda net: True, submit_share=lambda lane_id, net, share: None),
            failsafe=None,
            read_telemetry=lambda: {},
            vsd=_VSDStub(),
            tick_interval_s=0.02,
        )

        engine.start()
        try:
            time.sleep(0.10)
            self.assertGreater(cm.dispatch_count, 0)
            engine.pause(note="test pause", source="unit_test")
            time.sleep(0.06)
            paused_count = cm.dispatch_count
            time.sleep(0.06)
            self.assertEqual(cm.dispatch_count, paused_count)
            self.assertTrue(engine.is_paused())
        finally:
            engine.stop(timeout=0.2)


if __name__ == "__main__":
    unittest.main()