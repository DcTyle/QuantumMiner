#!/usr/bin/env python3

import subprocess
import sys
from typing import List


def _run_check(args: List[str]) -> int:
    cmd = [sys.executable, "scripts/check_import_cycles.py"] + args
    proc = subprocess.Popen(cmd)
    return proc.wait()


def main() -> None:
    codes = []
    print("[neuralis-boundary] focus bios,VHW,miner,core")
    codes.append(_run_check(["--strict", "--focus", "bios,VHW,miner,core"]))

    print("[neuralis-boundary] forbid miner -> Neuralis_AI")
    codes.append(_run_check(["--forbid-import", "Neuralis_AI", "--strict"]))

    print("[neuralis-boundary] forbid prediction_engine -> miner")
    codes.append(_run_check(["--forbid-import", "miner", "--strict", "--focus", "prediction_engine"]))

    if any(code != 0 for code in codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
