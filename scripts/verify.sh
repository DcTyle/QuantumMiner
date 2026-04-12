#!/usr/bin/env bash
set -euo pipefail

# QuantumMiner verification script
# - ASCII-only project guard assumed by docs
# - Runs strict dependency/cycle checks and unit tests

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# ASCII-only check (opt-in with STRICT_ASCII=1)
if [[ "${STRICT_ASCII:-0}" == "1" ]]; then
python3 - <<'PY'
import os, sys
skip_dirs = {'.git', '.github', '__pycache__', 'env', '.venv', 'venv', '.mypy_cache', '.pytest_cache'}
text_exts = {'.py', '.json', '.yml', '.yaml', '.toml', '.sh', '.bat', '.ps1', '.ini', '.cfg', '.conf', '.txt'}
bad = []
for root, dirs, files in os.walk('.'):
  # prune skip dirs
  dirs[:] = [d for d in dirs if d not in skip_dirs]
  for f in files:
    # Only check text-like files; skip backups with .bak_ascii
    if f.endswith('.bak_ascii'):
      continue
    _, ext = os.path.splitext(f)
    if ext not in text_exts:
      continue
    p = os.path.join(root, f)
    try:
      with open(p, 'rb') as fh:
        b = fh.read()
    except Exception:
      continue
    if any(c >= 128 for c in b):
      bad.append(p)
if bad:
  print('[verify] Non-ASCII bytes found in text files:')
  for p in bad:
    print('  -', p)
  sys.exit(1)
print('[verify] ASCII-only check passed (STRICT_ASCII).')
PY
else
  echo "[verify] ASCII-only check skipped (set STRICT_ASCII=1 to enforce)"
fi

if [[ "${VERIFY_SKIP_GUARDS:-0}" == "1" ]]; then
  echo "[verify] Guard checks skipped (VERIFY_SKIP_GUARDS=1)"
else
  # Core strict import/cycle guard focused on core subsystems
  python3 scripts/check_import_cycles.py --strict --focus bios,VHW,miner,core

  # Boundary checks
  # Enforce strictly for prediction_engine -> miner and miner -> Neuralis_AI.

  # prediction_engine must not import miner directly (path-scoped)
  USAGE_MINER="$(python3 scripts/check_import_cycles.py --check-import miner || true)"
  PE_VIOLATIONS=$(printf "%s\n" "$USAGE_MINER" | sed -n "s/^  - \(prediction_engine\/.*\)$/\1/p" || true)
  if [[ -n "$PE_VIOLATIONS" ]]; then
    echo "[verify] Boundary violation: prediction_engine -> miner" >&2
    printf "%s\n" "$PE_VIOLATIONS" >&2
    exit 1
  fi

  # miner must not import Neuralis_AI (no allowlist)
  USAGE_OUT="$(python3 scripts/check_import_cycles.py --check-import Neuralis_AI || true)"
  VIOLATIONS=$(printf "%s\n" "$USAGE_OUT" | sed -n "s/^  - \(.*\)$/\1/p" | grep -E "^miner/" || true)
  if [[ -n "$VIOLATIONS" ]]; then
    echo "[verify] Boundary violation(s): miner -> Neuralis_AI" >&2
    printf "%s\n" "$VIOLATIONS" >&2
    exit 1
  fi
fi

if [[ "${VERIFY_SKIP_TESTS:-0}" == "1" ]]; then
  echo "[verify] Unit tests skipped (VERIFY_SKIP_TESTS=1)"
elif [[ -n "${VERIFY_TEST_MODULE:-}" ]]; then
  python -m unittest -q "$VERIFY_TEST_MODULE"
else
  python -m unittest -q
fi

echo "[verify] All checks passed."