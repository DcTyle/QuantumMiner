"""
Prediction Engine Snapshot & Compaction Service
ASCII-ONLY

This module provides:
  - Minimal delta encoding utilities (zigzag + varint)
  - Canonical JSON + sha256 digest helpers
  - Rollup helpers from 1m -> 5m (L2) and 5m -> 1h (L3)
  - Extensible scaffolding for 1h -> 1d -> 1w -> 1mo -> 1y

It does NOT import miner.* and only relies on a provided `vsd` adapter that
exposes get(key, default=None) and store(key, value) methods.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import json
import hashlib
import base64


# -----------------------------
# Encoding utilities
# -----------------------------
def zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def zigzag_decode(n: int) -> int:
    return (n >> 1) ^ (-(n & 1))


def varint_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("varint_encode expects non-negative")
    out = bytearray()
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            out.append(0x80 | to_write)
        else:
            out.append(to_write)
            break
    return bytes(out)


def varint_decode(buf: bytes, offset: int = 0) -> Tuple[int, int]:
    shift = 0
    result = 0
    i = offset
    while i < len(buf):
        b = buf[i]
        result |= ((b & 0x7F) << shift)
        i += 1
        if not (b & 0x80):
            return result, i
        shift += 7
    raise ValueError("unterminated varint")


def encode_int_sequence(xs: List[int]) -> bytes:
    out = bytearray()
    for x in xs:
        out.extend(varint_encode(zigzag_encode(int(x))))
    return bytes(out)


def decode_int_sequence(buf: bytes) -> List[int]:
    i = 0
    out: List[int] = []
    while i < len(buf):
        v, i = varint_decode(buf, i)
        out.append(zigzag_decode(v))
    return out


# -----------------------------
# JSON helpers
# -----------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("ascii", errors="strict")).hexdigest()


# -----------------------------
# Rollup helpers
# -----------------------------
PriceScale = 10 ** 6  # scale floats to ints for stability
VolScale = 10 ** 3


def to_scaled_bar(bar: Dict[str, Any]) -> Dict[str, int]:
    return {
        "t": int(bar.get("t", 0)),
        "o": int(round(float(bar.get("o", 0.0)) * PriceScale)),
        "h": int(round(float(bar.get("h", 0.0)) * PriceScale)),
        "l": int(round(float(bar.get("l", 0.0)) * PriceScale)),
        "c": int(round(float(bar.get("c", 0.0)) * PriceScale)),
        "v": int(round(float(bar.get("v", 0.0)) * VolScale)),
    }


def rollup_5m_from_1m(symbol: str, bars_1m: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a compact 5m block (L2) from five 1m bars.
    Encoding: store a base (from the first bar) and deltas for the rest.
    """
    if len(bars_1m) != 5:
        raise ValueError("rollup_5m_from_1m requires exactly 5 bars")

    scaled = [to_scaled_bar(b) for b in bars_1m]
    scaled.sort(key=lambda b: b["t"])  # ensure time order
    base = scaled[0]
    t0 = base["t"]

    # deltas relative to base for remaining 4 bars: [dt, do, dh, dl, dc, dv] per bar
    seq: List[int] = []
    prev_t = t0
    for b in scaled[1:]:
        seq.extend([
            b["t"] - prev_t,  # delta-of-delta effect achieved across bars
            b["o"] - base["o"],
            b["h"] - base["h"],
            b["l"] - base["l"],
            b["c"] - base["c"],
            b["v"] - base["v"],
        ])
        prev_t = b["t"]

    encoded = encode_int_sequence(seq)
    blob = base64.b64encode(encoded).decode("ascii")

    block = {
        "version": 1,
        "symbol": symbol,
        "interval": "5m",
        "t0": t0,
        "count": 5,
        "encoding": "zigzag-varint",
        "base": {k: base[k] for k in ("o", "h", "l", "c", "v")},
        "deltas": blob,
        "meta": {"digest": "", "source": "rollup:1m"},
    }
    block["meta"]["digest"] = sha256_hex(canonical_json(block))
    return block


def rollup_1h_from_5m(symbol: str, blocks_5m: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a compact 1h block (L3) from twelve 5m blocks.
    We encode deltas for OHLCV bases of the 5m blocks relative to the first one.
    """
    if len(blocks_5m) != 12:
        raise ValueError("rollup_1h_from_5m requires exactly 12 blocks")

    blocks = sorted(blocks_5m, key=lambda b: int(b.get("t0", 0)))
    base_block = blocks[0]
    t0 = int(base_block["t0"])
    base = base_block["base"]

    seq: List[int] = []
    prev_t = t0
    for b in blocks[1:]:
        bb = b["base"]
        seq.extend([
            int(b["t0"]) - prev_t,
            int(bb["o"]) - int(base["o"]),
            int(bb["h"]) - int(base["h"]),
            int(bb["l"]) - int(base["l"]),
            int(bb["c"]) - int(base["c"]),
            int(bb["v"]) - int(base["v"]),
        ])
        prev_t = int(b["t0"]) 

    encoded = encode_int_sequence(seq)
    blob = base64.b64encode(encoded).decode("ascii")

    block = {
        "version": 1,
        "symbol": symbol,
        "interval": "1h",
        "t0": t0,
        "blocks": 12,
        "encoding": "zigzag-varint",
        "base": {k: int(base[k]) for k in ("o", "h", "l", "c", "v")},
        "deltas": blob,
        "meta": {"digest": "", "source": "rollup:5m"},
    }
    block["meta"]["digest"] = sha256_hex(canonical_json(block))
    return block


# -----------------------------
# Decoders for expansion
# -----------------------------
def decode_5m_block_to_1m(block_5m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a 5m block back to five 1m bars.
    Returns list of bars with t,o,h,l,c,v (scaled back to floats).
    """
    t0 = int(block_5m.get("t0", 0))
    base = block_5m.get("base", {})
    try:
        enc = base64.b64decode(str(block_5m.get("deltas", "")).encode("ascii"))
    except Exception:
        enc = b""
    ints = decode_int_sequence(enc)
    # ints is 4 bars * 6 ints per bar: [dt,do,dh,dl,dc,dv] x4
    bars: List[Dict[str, Any]] = []
    # first bar = base
    bars.append({
        "t": t0,
        "o": float(base.get("o", 0)) / PriceScale,
        "h": float(base.get("h", 0)) / PriceScale,
        "l": float(base.get("l", 0)) / PriceScale,
        "c": float(base.get("c", 0)) / PriceScale,
        "v": float(base.get("v", 0)) / VolScale,
    })
    prev_t = t0
    for i in range(0, len(ints), 6):
        if i + 5 >= len(ints):
            break
        dt, do, dh, dl, dc, dv = ints[i:i+6]
        prev_t = prev_t + int(dt)
        bars.append({
            "t": prev_t,
            "o": (int(base.get("o", 0)) + int(do)) / PriceScale,
            "h": (int(base.get("h", 0)) + int(dh)) / PriceScale,
            "l": (int(base.get("l", 0)) + int(dl)) / PriceScale,
            "c": (int(base.get("c", 0)) + int(dc)) / PriceScale,
            "v": (int(base.get("v", 0)) + int(dv)) / VolScale,
        })
    return bars


def decode_1h_block_to_5m(block_1h: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a 1h block back to twelve 5m base bars (approximation)."""
    t0 = int(block_1h.get("t0", 0))
    base = block_1h.get("base", {})
    try:
        enc = base64.b64decode(str(block_1h.get("deltas", "")).encode("ascii"))
    except Exception:
        enc = b""
    ints = decode_int_sequence(enc)
    bars: List[Dict[str, Any]] = []
    bars.append({
        "t": t0,
        "o": float(base.get("o", 0)) / PriceScale,
        "h": float(base.get("h", 0)) / PriceScale,
        "l": float(base.get("l", 0)) / PriceScale,
        "c": float(base.get("c", 0)) / PriceScale,
        "v": float(base.get("v", 0)) / VolScale,
    })
    prev_t = t0
    for i in range(0, len(ints), 6):
        if i + 5 >= len(ints):
            break
        dt, do, dh, dl, dc, dv = ints[i:i+6]
        prev_t = prev_t + int(dt)
        bars.append({
            "t": prev_t,
            "o": (int(base.get("o", 0)) + int(do)) / PriceScale,
            "h": (int(base.get("h", 0)) + int(dh)) / PriceScale,
            "l": (int(base.get("l", 0)) + int(dl)) / PriceScale,
            "c": (int(base.get("c", 0)) + int(dc)) / PriceScale,
            "v": (int(base.get("v", 0)) + int(dv)) / VolScale,
        })
    return bars

class SnapshotService:
    """Simple snapshot service scaffolding; callers provide a vsd adapter.
    Methods return data structures so higher-level code can decide persistence.
    """
    def __init__(self, vsd: Any | None = None) -> None:
        self.vsd = vsd

    def write_1m_jsonl(self, symbol: str, hour_epoch: int, bars_1m: List[Dict[str, Any]]) -> str:
        """Build JSONL content for a set of 1m bars belonging to an hour.
        Returns the ASCII text. Caller may store via VSD:
          key = f"prediction/timeseries/{symbol}/1m/{YYYY}/{MM}/{DD}/{HH}.jsonl"
        """
        lines = []
        for b in sorted(bars_1m, key=lambda x: int(x.get("t", 0))):
            lines.append(canonical_json(b))
        return "\n".join(lines) + ("\n" if lines else "")

    def build_5m_block(self, symbol: str, bars_1m: List[Dict[str, Any]]) -> Dict[str, Any]:
        return rollup_5m_from_1m(symbol, bars_1m)

    def build_1h_block(self, symbol: str, blocks_5m: List[Dict[str, Any]]) -> Dict[str, Any]:
        return rollup_1h_from_5m(symbol, blocks_5m)

    # Placeholders for higher rollups (1d, 1w, 1mo, 1y). Implement similarly.
    def build_1d_block(self, symbol: str, blocks_1h: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _rollup_generic(symbol, blocks_1h, "1d", 24, "rollup:1h")

    def build_1w_block(self, symbol: str, blocks_1d: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _rollup_generic(symbol, blocks_1d, "1w", 7, "rollup:1d")

    def build_1mo_block(self, symbol: str, blocks_1d: List[Dict[str, Any]], days: int = 30) -> Dict[str, Any]:
        return _rollup_generic(symbol, blocks_1d[:days], "1mo", days, "rollup:1d")

    def build_1y_block(self, symbol: str, blocks_1mo: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _rollup_generic(symbol, blocks_1mo, "1y", 12, "rollup:1mo")

    # Manifest writer helpers
    def update_manifest(self, symbol: str, level: str, t0: int, t1: int, key: str | None = None, block: Dict[str, Any] | None = None) -> None:
        mkey = f"prediction/timeseries/{symbol}/manifest"
        try:
            man = self.vsd.get(mkey, {}) if self.vsd else {}
            if not isinstance(man, dict):
                man = {}
        except Exception:
            man = {}
        arr = man.get(level)
        if not isinstance(arr, list):
            arr = []
        entry: Dict[str, Any] = {"t0": int(t0), "t1": int(t1)}
        if key:
            entry["key"] = str(key)
        if block is not None:
            entry["block"] = block
        arr.append(entry)
        man[level] = arr
        try:
            if self.vsd:
                self.vsd.store(mkey, man)
        except Exception:
            pass


# -----------------------------
# Higher-level generic rollup
# -----------------------------
def _rollup_generic(symbol: str, blocks: List[Dict[str, Any]], interval: str, expect: int, source: str) -> Dict[str, Any]:
    if len(blocks) != expect:
        raise ValueError(f"rollup {interval} expects {expect} blocks")
    bs = sorted(blocks, key=lambda b: int(b.get("t0", 0)))
    base_b = bs[0]
    t0 = int(base_b.get("t0", 0))
    base = base_b.get("base", {})
    seq: List[int] = []
    prev_t = t0
    for b in bs[1:]:
        bb = b.get("base", {})
        seq.extend([
            int(b.get("t0", 0)) - prev_t,
            int(bb.get("o", 0)) - int(base.get("o", 0)),
            int(bb.get("h", 0)) - int(base.get("h", 0)),
            int(bb.get("l", 0)) - int(base.get("l", 0)),
            int(bb.get("c", 0)) - int(base.get("c", 0)),
            int(bb.get("v", 0)) - int(base.get("v", 0)),
        ])
        prev_t = int(b.get("t0", 0))

    encoded = encode_int_sequence(seq)
    blob = base64.b64encode(encoded).decode("ascii")
    block = {
        "version": 1,
        "symbol": symbol,
        "interval": interval,
        "t0": t0,
        "blocks": expect,
        "encoding": "zigzag-varint",
        "base": {k: int(base.get(k, 0)) for k in ("o", "h", "l", "c", "v")},
        "deltas": blob,
        "meta": {"digest": "", "source": source},
    }
    block["meta"]["digest"] = sha256_hex(canonical_json(block))
    return block
