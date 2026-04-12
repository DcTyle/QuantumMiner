from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from VHW.vsd_manager import VSD


DIFF1_TARGET = int(
    "00000000FFFF0000000000000000000000000000000000000000000000000000",
    16,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_target_hex(value: Any) -> str:
    try:
        text = str(value or "").strip().lower()
    except Exception:
        text = ""
    if text.startswith("0x"):
        text = text[2:]
    text = "".join(ch for ch in text if ch in "0123456789abcdef")
    if not text:
        return ""
    return text[-64:].rjust(64, "0")


def _difficulty_from_target(target_value: Any) -> float:
    try:
        target_hex = _normalize_target_hex(target_value)
        target = int(target_hex or "0", 16)
    except Exception:
        return 0.0
    if target <= 0:
        return 0.0
    return float(DIFF1_TARGET / float(target))


def _difficulty_from_record(record: Dict[str, Any]) -> float:
    if not isinstance(record, dict):
        return 0.0
    method = str(record.get("method", ""))
    params = record.get("params", [])
    if method == "mining.set_difficulty" and isinstance(params, list) and params:
        return max(0.0, _safe_float(params[0], 0.0))
    if method == "mining.set_target" and isinstance(params, list) and params:
        return max(0.0, _difficulty_from_target(params[0]))
    return 0.0


def _packet_raw_payload(packet: Any) -> Dict[str, Any]:
    if hasattr(packet, "raw_payload"):
        try:
            return dict(getattr(packet, "raw_payload") or {})
        except Exception:
            return {}
    if isinstance(packet, dict):
        try:
            raw = packet.get("raw_payload", {})
            return dict(raw or {}) if isinstance(raw, dict) else {}
        except Exception:
            return {}
    return {}


def _truncate(text: Any, width: int) -> str:
    raw = str(text or "")
    if len(raw) <= width:
        return raw.ljust(width)
    if width <= 3:
        return raw[:width]
    return (raw[: width - 3] + "...")


def _collect_rows() -> List[Dict[str, Any]]:
    worker_map = dict(VSD.get("miner/lanes/worker_map", {}) or {})
    session_map = dict(VSD.get("miner/lanes/session_map", {}) or {})
    global_telem = dict(VSD.get("telemetry/global", {}) or {})
    jobs_map = dict(global_telem.get("jobs_map", {}) or {})

    rows: List[Dict[str, Any]] = []
    for lane_id in sorted(worker_map.keys()):
        worker = dict(worker_map.get(lane_id, {}) or {})
        coin = str(worker.get("coin", "")).upper()
        runtime = dict(VSD.get("miner/runtime/submission_rate/workers/%s" % lane_id, {}) or {})
        request = dict(VSD.get("miner/difficulty_request/%s/workers/%s" % (coin, lane_id), {}) or {})
        session = dict(VSD.get("miner/stratum/workers/%s/session" % lane_id, {}) or {})
        diff_record = dict(VSD.get("miner/difficulty/%s/workers/%s" % (coin, lane_id), {}) or {})
        if not diff_record:
            diff_record = dict(VSD.get("miner/difficulty/%s" % coin, {}) or {})
        packet_raw = _packet_raw_payload(jobs_map.get(lane_id))
        share_target = (
            str(runtime.get("share_target", ""))
            or str(session.get("share_target", ""))
            or str(packet_raw.get("share_target", packet_raw.get("active_target", "")))
        )
        assigned_diff = _safe_float(runtime.get("assigned_share_difficulty", 0.0), 0.0)
        if assigned_diff <= 0.0:
            assigned_diff = _safe_float(session.get("assigned_share_difficulty", 0.0), 0.0)
        if assigned_diff <= 0.0:
            assigned_diff = _difficulty_from_record(diff_record)
        if assigned_diff <= 0.0 and share_target:
            assigned_diff = _difficulty_from_target(share_target)

        rows.append({
            "lane_id": str(lane_id),
            "coin": coin,
            "worker_index": int(worker.get("worker_index", runtime.get("worker_index", 0)) or 0),
            "username": str(worker.get("username", runtime.get("username", session.get("username", "")))),
            "session_id": str(session_map.get(lane_id, runtime.get("session_id", session.get("session_id", "")))),
            "connected": bool(session.get("connected", False)),
            "assigned_share_difficulty": float(assigned_diff),
            "target_share_difficulty": float(runtime.get("target_share_difficulty", request.get("target_share_difficulty", 0.0)) or 0.0),
            "allowed_rate_per_second": float(runtime.get("allowed_rate_per_second", 0.0) or 0.0),
            "share_target": _normalize_target_hex(share_target),
            "last_job_id": str(session.get("last_job_id", packet_raw.get("job_id", ""))),
            "boosted_until": float(runtime.get("boosted_until", 0.0) or 0.0),
            "last_request_ts": float(request.get("ts", 0.0) or 0.0),
        })
    return rows


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        ("lane_id", 10),
        ("coin", 5),
        ("worker_index", 7),
        ("username", 24),
        ("connected", 9),
        ("assigned_diff", 14),
        ("target_diff", 12),
        ("allowed/s", 10),
        ("share_target", 18),
    ]
    header_line = " ".join(_truncate(label, width) for label, width in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = " ".join([
            _truncate(row.get("lane_id", ""), 10),
            _truncate(row.get("coin", ""), 5),
            _truncate(row.get("worker_index", 0), 7),
            _truncate(row.get("username", ""), 24),
            _truncate("yes" if row.get("connected") else "no", 9),
            _truncate("%.6f" % float(row.get("assigned_share_difficulty", 0.0)), 14),
            _truncate("%.6f" % float(row.get("target_share_difficulty", 0.0)), 12),
            _truncate("%.6f" % float(row.get("allowed_rate_per_second", 0.0)), 10),
            _truncate(row.get("share_target", "")[:18], 18),
        ])
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect worker-session vardiff and submission-rate state from VSD.")
    parser.add_argument("--json", action="store_true", help="Emit the probe output as JSON.")
    args = parser.parse_args()

    rows = _collect_rows()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if not rows:
        print("No worker-session runtime records found.")
        return 0
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())