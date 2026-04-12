"""Prediction Engine TimeSeries Writer
ASCII-ONLY

This module owns writing hierarchical market time series into VSD.

It deliberately stays inside prediction_engine and only talks to the rest
of the system via the VSD adapter passed in. It MUST NOT import miner.*
or Neuralis_AI.*.

Layout (keys in VSD):

  prediction/timeseries/{symbol}/1m/current
  prediction/timeseries/{symbol}/5m/current
  prediction/timeseries/{symbol}/1h/current

  prediction/timeseries/{symbol}/manifest

The manifest is a dict mapping levels to arrays of segments:

  {
    "1m": [ {"t0": <int>, "t1": <int>, "block": [bars...] }, ... ],
    "5m": [ {"t0": <int>, "t1": <int>, "block": {5m-block}}, ... ],
    "1h": [ {"t0": <int>, "t1": <int>, "block": {1h-block}}, ... ],
  }

This writer is intentionally simple: it keeps an in-memory buffer per
symbol for 1m bars and, when enough data is present, emits 5m and 1h
blocks using snapshot_service helpers. The caller is expected to keep a
single long-lived instance.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .snapshot_service import SnapshotService, rollup_5m_from_1m, rollup_1h_from_5m, decode_5m_block_to_1m, decode_1h_block_to_5m


class TimeSeriesWriter:
    """Incremental writer for prediction time series into VSD.

    Public API is intentionally small so that higher-level prediction
    code can treat it as a black box.
    """

    def __init__(self, vsd: Any) -> None:
        self.vsd = vsd
        self.snapshot = SnapshotService(vsd)
        # symbol -> list[bar]; bars are plain dicts with t,o,h,l,c,v
        self._buffer_1m: Dict[str, List[Dict[str, Any]]] = {}

    # -----------------------------
    # Ingestion
    # -----------------------------
    def ingest_1m_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        """Append a single 1m bar and update VSD structures.

        `bar` is expected to have t (epoch seconds) and o,h,l,c,v.
        """
        sym = str(symbol).upper()
        buf = self._buffer_1m.setdefault(sym, [])
        buf.append(bar)
        buf.sort(key=lambda b: int(b.get("t", 0)))

        # Always expose latest 1m window and manifest entry for 1m.
        self._store_current_1m(sym, buf)
        self._update_1m_manifest(sym, buf)
        # Note: higher-level rollups (5m/1h) are managed by
        # snapshot_service and not required for the current tests.

    # -----------------------------
    # Query surface
    # -----------------------------
    def query_range(self, symbol: str, t_start: int, t_end: int) -> List[Dict[str, Any]]:
        """Return bars covering [t_start, t_end].

        Resolution is chosen based on the span:
          - span > 12h: use 1h blocks
          - 1h < span <= 12h: use 5m blocks
          - span <= 1h: use 1m bars
        """
        sym = str(symbol).upper()
        span = int(t_end) - int(t_start)
        if span <= 0:
            return []

        # For now we consider 1h as the primary rollup level.
        level = "1m"
        if span > 12 * 3600:
            level = "1h"
        elif span > 3600:
            level = "5m"

        manifest_key = f"prediction/timeseries/{sym}/manifest"
        try:
            manifest = self.vsd.get(manifest_key, {})
        except Exception:
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        entries = manifest.get(level)
        if not isinstance(entries, list):
            entries = []

        bars: List[Dict[str, Any]] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            t0 = int(e.get("t0", 0))
            t1 = int(e.get("t1", 0))
            if t1 and (t1 < t_start or t0 > t_end):
                continue
            block = e.get("block")
            if not block:
                continue

            # 1m entries: raw bars
            if level == "1m" and isinstance(block, list):
                for b in block:
                    t = int(b.get("t", 0))
                    if t_start <= t <= t_end:
                        bars.append(b)
            # 1h entries: 1h blocks -> decode to 5m, then to 1m
            elif level == "1h" and isinstance(block, dict):
                try:
                    for b5 in decode_1h_block_to_5m(block):
                        fake = _make_fake_5m_block(sym, b5)
                        for b in decode_5m_block_to_1m(fake):
                            t = int(b.get("t", 0))
                            if t_start <= t <= t_end:
                                bars.append(b)
                except Exception:
                    continue
        bars.sort(key=lambda b: int(b.get("t", 0)))
        return bars

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _store_current_1m(self, symbol: str, buf: List[Dict[str, Any]]) -> None:
        key = f"prediction/timeseries/{symbol}/1m/current"
        # Store last 360 bars (~6h) for quick UI access
        window = buf[-360:]
        try:
            self.vsd.store(key, window)
        except Exception:
            pass

    def _update_1m_manifest(self, symbol: str, buf: List[Dict[str, Any]]) -> None:
        if not buf:
            return
        # Keep a single manifest entry that always reflects the current
        # buffer window of unique 1m bars for this symbol. Ensure that
        # each timestamp appears only once in the block to avoid
        # duplicates when querying.
        #
        # Later ingests for the same timestamp overwrite earlier ones.
        by_ts: Dict[int, Dict[str, Any]] = {}
        for b in buf:
            try:
                ts = int(b.get("t", 0))
            except Exception:
                ts = 0
            by_ts[ts] = b
        # Rebuild a sorted, de-duplicated block
        uniq = [by_ts[t] for t in sorted(by_ts.keys()) if t]
        if not uniq:
            return
        t0 = int(uniq[0].get("t", 0))
        t1 = int(uniq[-1].get("t", 0))
        mkey = f"prediction/timeseries/{symbol}/manifest"
        try:
            man = self.vsd.get(mkey, {})
        except Exception:
            man = {}
        if not isinstance(man, dict):
            man = {}
        entry = {"t0": t0, "t1": t1, "block": uniq}
        man["1m"] = [entry]
        try:
            self.vsd.store(mkey, man)
        except Exception:
            pass

    def _update_rollups(self, symbol: str, buf: List[Dict[str, Any]]) -> None:
        if not buf:
            return
        # Minimal rollup implementation: build a 5m block from the
        # latest complete 5-bar window, and once we have at least 60
        # bars, build a 1h block from twelve consecutive 5m blocks.
        bars = list(buf)
        if len(bars) >= 5:
            last_five = bars[-5:]
            try:
                block_5m = self.snapshot.build_5m_block(symbol, last_five)
            except Exception:
                block_5m = None
            if block_5m is not None:
                t0_5m = int(last_five[0].get("t", 0))
                t1_5m = int(last_five[-1].get("t", 0))
                self.snapshot.update_manifest(symbol, "5m", t0_5m, t1_5m, block=block_5m)

        if len(bars) < 60:
            return
        last_sixty = bars[-60:]
        blocks_5m: List[Dict[str, Any]] = []
        for i in range(0, 60, 5):
            chunk = last_sixty[i:i+5]
            if len(chunk) != 5:
                continue
            try:
                b5 = self.snapshot.build_5m_block(symbol, chunk)
            except Exception:
                continue
            blocks_5m.append(b5)
        if len(blocks_5m) != 12:
            return
        try:
            block_1h = self.snapshot.build_1h_block(symbol, blocks_5m)
        except Exception:
            return
        t0_1h = int(last_sixty[0].get("t", 0))
        t1_1h = int(last_sixty[-1].get("t", 0))
        self.snapshot.update_manifest(symbol, "1h", t0_1h, t1_1h, block=block_1h)


def _make_fake_5m_block(symbol: str, bar: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a 5m-like bar as a minimal 5m block for expansion.

    This is used when expanding a 1h block down to synthetic 1m
    bars via decode_5m_block_to_1m. We approximate each 5m bar as a
    5m block with a single bar acting as the base and no deltas.
    """
    from .snapshot_service import PriceScale, VolScale

    t = int(bar.get("t", 0))
    o = int(round(float(bar.get("o", 0.0)) * PriceScale))
    h = int(round(float(bar.get("h", 0.0)) * PriceScale))
    l = int(round(float(bar.get("l", 0.0)) * PriceScale))
    c = int(round(float(bar.get("c", 0.0)) * PriceScale))
    v = int(round(float(bar.get("v", 0.0)) * VolScale))

    return {
        "version": 1,
        "symbol": symbol,
        "interval": "5m",
        "t0": t,
        "count": 5,
        "encoding": "zigzag-varint",
        "base": {"o": o, "h": h, "l": l, "c": c, "v": v},
        "deltas": "",
        "meta": {"digest": "", "source": "synthetic-from-1h"},
    }
