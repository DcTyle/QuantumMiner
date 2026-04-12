#!/usr/bin/env python3
# ASCII-ONLY STATUS SCRIPT
# Prints GPU initiator status from VSD
from __future__ import annotations
import argparse
import time
from typing import Any

try:
    from VHW.vsd_manager import VSDManager
except Exception:
    class VSDManager:
        def __init__(self):
            self._kv = {}
        def get(self, k, d=None):
            return self._kv.get(k, d)

KEY_INIT = "vhw/gpu/init_ok"
KEY_AVAIL = "vhw/gpu/available"
KEY_PCT = "vhw/gpu/sustain_pct"
KEY_HB = "vhw/gpu/heartbeat"


def read_once(vsd: Any) -> dict:
    init_ok = bool(vsd.get(KEY_INIT, False))
    avail = bool(vsd.get(KEY_AVAIL, False))
    pct = float(vsd.get(KEY_PCT, 0.0) or 0.0)
    hb = vsd.get(KEY_HB, {}) or {}
    ts = float(hb.get("ts", 0.0) or 0.0)
    age = time.time() - ts if ts > 0.0 else -1.0
    return {
        "init_ok": init_ok,
        "available": avail,
        "sustain_pct": pct,
        "heartbeat_ts": ts,
        "heartbeat_age_s": age,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU Initiator VSD status")
    ap.add_argument("--watch", action="store_true", help="continuously print status")
    ap.add_argument("--interval", type=float, default=1.0, help="watch interval seconds")
    args = ap.parse_args()

    vsd = VSDManager()

    def print_status():
        s = read_once(vsd)
        age = s["heartbeat_age_s"]
        age_str = ("%.3f" % age) if age >= 0.0 else "n/a"
        print(
            "init_ok=%s available=%s sustain_pct=%.3f heartbeat_age_s=%s" % (
                str(s["init_ok"]).lower(),
                str(s["available"]).lower(),
                s["sustain_pct"],
                age_str,
            )
        )

    if args.watch:
        try:
            while True:
                print_status()
                time.sleep(max(0.1, float(args.interval)))
        except KeyboardInterrupt:
            return 0
    else:
        print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
