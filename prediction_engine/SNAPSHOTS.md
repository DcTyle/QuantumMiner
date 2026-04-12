# Prediction Engine Snapshots & Compaction
# ASCII-ONLY — PRODUCTION VERSION

============================================================
1. PURPOSE
============================================================
Define when and where snapshots occur (prediction_engine, not miner), the
storage layout, and the rolling compaction pipeline for long-term retention
of market data with delta-based compression.

Key principles:
- Miner does NOT snapshot. Miner emits lane telemetry and share outcomes only.
- Prediction engine owns ingestion, snapshots, and compaction.
- All artifacts are ASCII-only, versioned, and integrity-checked.

============================================================
2. OWNERSHIP & TRIGGERS
============================================================
Ownership:
- prediction_engine: ingest, snapshot, compaction, retention.
- miner: no snapshots; emits EventBus signals with minimal telemetry.

Event triggers (prediction_engine):
- Ingest: on each 1m bar or tick from crypto.com APIs.
- Snapshot: at end of each roll window (5m, 1h, 1d) or when N records buffered.
- Compaction: periodic tasks (e.g., every 5m and hourly) collapse lower-level data
  into higher-level deltas.

============================================================
3. HIERARCHY & COMPACTION MODEL
============================================================
Levels:
- L1 (origin): 1m bars (OHLCV + meta), directly from ingestion.
- L2 (delta): 5m blocks represented as deltas relative to reconstructed 5m base.
- L3 (delta): 1h blocks as deltas relative to hourly base reconstructed from L2.
- L4+ (optional): 1d, 1w, 1mo using same pattern as deltas of the nearest lower level.

Compression:
- Numeric deltas encoded via ZigZag + varint for integers.
- For floats: scaled integers (e.g., price * 1e6) before ZigZag + varint.
- Delta-of-delta for timestamps and monotonically increasing fields.
- XOR encoding for bit-pattern deltas on repeating values is optional.

Integrity:
- Each chunk includes meta.digest (sha256 of canonical JSON lines).
- Canonicalize JSON (sorted keys, no whitespace beyond single spaces).

============================================================
4. STORAGE LAYOUT (VSD PATHS)
============================================================
Per symbol and interval:
- L1 (1m):  `vsd://prediction/timeseries/{symbol}/1m/{YYYY}/{MM}/{DD}/{HH}.jsonl`
- L2 (5m):  `vsd://prediction/timeseries/{symbol}/5m/{YYYY}/{MM}/{DD}/{HH}.json`
- L3 (1h):  `vsd://prediction/timeseries/{symbol}/1h/{YYYY}/{MM}/{DD}.json`
- Manifests: `vsd://prediction/timeseries/{symbol}/manifest.json`

Notes:
- L1 uses JSONL (one record per line) for append-friendly ingestion.
- L2+ use compact JSON with varint-encoded byte strings base64’d in ASCII.
- All paths are stored via VSDManager using `vsd.store(key, ascii_text)`.

============================================================
5. SCHEMAS (SIMPLIFIED)
============================================================
L1 1m bar (JSONL line):
{
  "t": 1731957600,             // unix epoch sec (start of 1m)
  "o": 65321_000,              // price*1e3 or *1e6 as int
  "h": 65380_000,
  "l": 65290_000,
  "c": 65350_000,
  "v": 12_345_000,             // volume normalized (units*1e3)
  "sym": "BTCUSDT",
  "src": "crypto_com"
}

L2 5m block (JSON):
{
  "version": 1,
  "symbol": "BTCUSDT",
  "interval": "5m",
  "t0": 1731957600,           // start of 5-minute window (epoch)
  "count": 5,
  "encoding": "zigzag-varint",
  "base": { "o": 65321_000, "h": 0, "l": 0, "c": 0, "v": 0 },
  "deltas": "<base64 varint stream>",
  "meta": { "digest": "<sha256-hex>", "source": "rollup:1m" }
}

L3 1h block (JSON):
{
  "version": 1,
  "symbol": "BTCUSDT",
  "interval": "1h",
  "t0": 1731957600,
  "blocks": 12,
  "encoding": "zigzag-varint",
  "base": { "o": 65300_000, "h": 0, "l": 0, "c": 0, "v": 0 },
  "deltas": "<base64 varint stream>",
  "meta": { "digest": "<sha256-hex>", "source": "rollup:5m" }
}

============================================================
6. PIPELINE & TASKS
============================================================
Components (prediction_engine):
- Ingestor: reads crypto.com APIs, writes L1 JSONL lines per symbol.
- Snapshoter: finishes a window and seals the current chunk with digest.
- Roller5m: aggregates the last 5x L1 samples into one L2 delta block.
- Roller1h: aggregates last 12x L2 blocks into one L3 delta block.
- Retention: prunes or cold-archives older L1 after L2/L3 are confirmed.

Scheduling:
- Ingest: every minute (aligned to wall-clock minute).
- Roll 5m: every 5 minutes at minute % 5 == 0.
- Roll 1h: every hour at minute == 0.

============================================================
7. BOUNDARY & INTEGRATION
============================================================
- Miner → prediction_engine: no direct imports. Communication via VSD or EventBus.
- Miner may publish minimal telemetry events (e.g., hashrate, share stats) on EventBus.
- prediction_engine subscribes as needed and may snapshot telemetry separately.
- Snapshots for market data belong solely to prediction_engine.

============================================================
8. ERROR HANDLING & RECOVERABILITY
============================================================
- Chunks are idempotent by path; re-writing with the same content maintains digest.
- Partially written L1 files are recoverable by truncation to last full line.
- L2/L3 recomputation should rebuild digests based on canonicalized content.

============================================================
9. NEXT STEPS (OPTIONAL)
============================================================
- Implement `snapshot_service.py` in prediction_engine with:
  - SnapshotService (write/close for L1; seal chunks with digests)
  - Compactor5m and Compactor1h with delta encoding utilities
  - JSON canonicalizer and sha256 digest helper
- Add unit tests for encoding/decoding and rollup correctness.
- Wire CI to run these tests under `make verify`.

============================================================
END OF SNAPSHOTS.md
============================================================
