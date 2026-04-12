import base64
from prediction_engine.snapshot_service import (
    zigzag_encode, zigzag_decode,
    varint_encode, varint_decode,
    encode_int_sequence, decode_int_sequence,
    rollup_5m_from_1m, rollup_1h_from_5m,
    _rollup_generic,
)


def test_zigzag_roundtrip():
    for n in [0, 1, -1, 2, -2, 123456, -123456]:
        z = zigzag_encode(n)
        assert zigzag_decode(z) == n


def test_varint_roundtrip():
    for n in [0, 1, 127, 128, 255, 30000, 2**31-1]:
        b = varint_encode(n)
        v, idx = varint_decode(b, 0)
        assert v == n
        assert idx == len(b)


def test_sequence_roundtrip():
    xs = [0, 1, -1, 5000, -5000, 10, -10]
    enc = encode_int_sequence(xs)
    dec = decode_int_sequence(enc)
    assert dec == xs


def test_rollup_5m_and_1h():
    # Construct 5x1m bars
    t0 = 1731957600
    one_min = 60
    bars_1m = []
    p = 100.0
    for i in range(5):
        bars_1m.append({
            't': t0 + i*one_min,
            'o': p + i*0.1,
            'h': p + i*0.2,
            'l': p + i*0.05,
            'c': p + i*0.15,
            'v': 1.0 + i*0.01,
        })
    blk5 = rollup_5m_from_1m('BTCUSDT', bars_1m)
    assert blk5['interval'] == '5m'
    assert blk5['count'] == 5
    assert isinstance(blk5['deltas'], str)
    # 12x5m -> 1h
    blocks = []
    for j in range(12):
        # shift t0 by 5 minutes per block
        t = t0 + j*5*one_min
        # reuse the same shape but shifted
        b = rollup_5m_from_1m('BTCUSDT', [{**bars_1m[k], 't': t + k*one_min} for k in range(5)])
        blocks.append(b)
    blk1h = rollup_1h_from_5m('BTCUSDT', blocks)
    assert blk1h['interval'] == '1h'
    assert blk1h['blocks'] == 12
    assert isinstance(blk1h['deltas'], str)


def test_higher_rollups_generic():
    # build 24 hourly blocks to make a day
    def mk_1h(i: int) -> Dict[str, Any]:
        return {
            "version": 1,
            "symbol": "BTC",
            "interval": "1h",
            "t0": 1_000_000 + i * 3600,
            "blocks": 12,
            "encoding": "zigzag-varint",
            "base": {"o": 10000 + i, "h": 10100 + i, "l": 9900 + i, "c": 10000 + i, "v": 10 + i},
            "deltas": base64.b64encode(encode_int_sequence([5] * (6 * 11))).decode("ascii"),
            "meta": {"digest": "", "source": "rollup:5m"},
        }

    blocks_1h = [mk_1h(i) for i in range(24)]
    day = _rollup_generic("BTC", blocks_1h, "1d", 24, "rollup:1h")
    assert day["interval"] == "1d"
    assert day["blocks"] == 24
    assert day["base"]["o"] == blocks_1h[0]["base"]["o"]
