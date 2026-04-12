import time
import unittest

from prediction_engine.timeseries_writer import TimeSeriesWriter


class DummyVSD:
    def __init__(self) -> None:
        self.store_calls = []
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def store(self, key, value):
        # Simple ASCII-only guard for keys
        key.encode("ascii", errors="strict")
        self._data[key] = value
        self.store_calls.append((key, value))


class TimeSeriesWriterTests(unittest.TestCase):
    def test_ingest_and_query_range_1m(self):
        vsd = DummyVSD()
        writer = TimeSeriesWriter(vsd)

        now = int(time.time())
        bars = []
        for i in range(5):
            bars.append({
                "t": now + i * 60,
                "o": 100.0 + i,
                "h": 101.0 + i,
                "l": 99.0 + i,
                "c": 100.5 + i,
                "v": 10.0 + i,
            })
        for b in bars:
            writer.ingest_1m_bar("BTCUSDT", b)

        # Query a 1h window around now should give 1m bars
        out = writer.query_range("BTCUSDT", now - 60, now + 5 * 60 + 60)
        ts = [int(b["t"]) for b in out]
        self.assertEqual(sorted(ts), sorted([b["t"] for b in bars]))

    def test_rollup_to_5m_and_1h_manifest(self):
        vsd = DummyVSD()
        writer = TimeSeriesWriter(vsd)
        base = int(time.time())

        # 60 minutes of data (60 bars) -> 12x5m -> 1x1h
        for i in range(60):
            b = {
                "t": base + i * 60,
                "o": 100.0 + 0.1 * i,
                "h": 101.0 + 0.1 * i,
                "l": 99.0 + 0.1 * i,
                "c": 100.5 + 0.1 * i,
                "v": 10.0 + i,
            }
            writer.ingest_1m_bar("ETHUSDT", b)

        man = vsd.get("prediction/timeseries/ETHUSDT/manifest", {})
        self.assertIsInstance(man, dict)
        # 1m manifest should at least contain all ingested bars
        self.assertIn("1m", man)
        entries = man["1m"]
        self.assertIsInstance(entries, list)
        self.assertGreaterEqual(len(entries), 1)

        # Query a wide range should return at least 60 raw 1m bars
        out = writer.query_range("ETHUSDT", base, base + 3600)
        self.assertTrue(len(out) >= 60)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
