# ASCII-ONLY
import unittest


class _VSDStub:
    def __init__(self):
        self._kv = {"system/bios_boot_ok": True}
    def get(self, k, d=None):
        return self._kv.get(k, d)
    def store(self, k, v):
        self._kv[k] = v


class _SubmitterStub:
    def __init__(self):
        self.calls = []
    def submit_packet(self, lane_id, packet):  # mimics miner.submitter API
        self.calls.append((lane_id, packet))
        return True


class TestLaneSubmit(unittest.TestCase):
    def test_each_lane_submits_packet(self):
        from VHW.compute_manager import ComputeManager, ComputeWrapper, Subsystem, ComputeMode
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork

        vsd = _VSDStub()
        cm = ComputeManager({
            "tiers": [{"tier_id": 0, "vqram_mb": 256}],
            "vsd": vsd,
        })
        # allocate two lanes
        l1 = cm.allocate_lane(0).lane_id
        l2 = cm.allocate_lane(0).lane_id
        lane_ids = {l1, l2}

        # build a minimal packet type that requires no external adapters
        pkt = neural_objectPacket(
            packet_type=variable_format_enum.Miner_DerivativeNonce,
            network=ComputeNetwork.BTC,
            raw_payload={"job_id": "TESTJOB"},
            system_payload={},
            metadata={},
            derived_state={},
        )

        jobs_map = {lid: pkt for lid in lane_ids}
        submitter = _SubmitterStub()
        wrapper = ComputeWrapper(
            subsys=Subsystem.MINER,
            mode=ComputeMode.BATCH,
            payload={"jobs_map": jobs_map},
            params={"submitter": submitter},
        )

        res = cm.dispatch(wrapper)
        # Ensure compute returned entries for both lanes
        self.assertTrue(all(lid in res for lid in lane_ids))
        # Ensure submitter saw a packet per lane
        seen_lanes = {lid for (lid, _pkt) in submitter.calls}
        self.assertEqual(seen_lanes, lane_ids)
        self.assertEqual(len(submitter.calls), len(lane_ids))

    def test_invalid_network_packet_is_not_forwarded(self):
        from VHW.compute_manager import ComputeManager, ComputeWrapper, Subsystem, ComputeMode
        from neural_object import neural_objectPacket, variable_format_enum, ComputeNetwork

        vsd = _VSDStub()
        cm = ComputeManager({
            "tiers": [{"tier_id": 0, "vqram_mb": 256}],
            "vsd": vsd,
        })
        lane_id = cm.allocate_lane(0).lane_id
        submitter = _SubmitterStub()

        pkt = neural_objectPacket(
            packet_type=variable_format_enum.BTC_BlockTemplate,
            network=ComputeNetwork.BTC,
            raw_payload={
                "job_id": "btc_dead_target",
                "header_hex": (b"\x00" * 80).hex(),
                "target": "0" * 64,
                "ntime": "00000000",
                "extranonce2": "00000000",
            },
            system_payload={},
            metadata={},
            derived_state={},
        )

        wrapper = ComputeWrapper(
            subsys=Subsystem.MINER,
            mode=ComputeMode.BATCH,
            payload={"jobs_map": {lane_id: pkt}},
            params={
                "submitter": submitter,
                "mode": "phase_coherence",
                "count": 4,
            },
        )

        res = cm.dispatch(wrapper)
        self.assertIn(lane_id, res)
        self.assertEqual(submitter.calls, [])


if __name__ == "__main__":
    unittest.main()
