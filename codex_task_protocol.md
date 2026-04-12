=== START FILE: CODEX_TASK_PROTOCOL.md ===
# Codex Task Protocol
# Defines how Neuralis issues tasks and how Jarvis validates them.

============================================================
1. NEURALIS → TASK FORMAT
============================================================
Neuralis must emit tasks as:

{
  "task_id": "<string>",
  "targets": ["path/file1.py", "path/file2.py"],
  "operation": "replace_block" | "rewrite_file" | "insert" | "create_file",
  "code_blocks": "<FULL ASCII CODE>",
  "justification": "<why>",
  "dependencies": ["module.name", "..."],
  "architecture_tags": ["VHW", "Neuralis", "..."]
}

============================================================
2. JARVIS → VALIDATION
============================================================
Jarvis verifies:
    - ASCII-only
    - subsystem boundaries respected
    - imports valid
    - no stubs
    - no placeholders
    - correct architecture tags
    - correct file paths

Jarvis returns one of:

APPROVED: <task_id>
DENIED: <task_id> REASON: <details>
REVISE: <task_id> INSTRUCTIONS: <requirements>

============================================================
3. CODEX → EXECUTION
============================================================
Codex MUST:
    • Patch only approved tasks
    • Write EXACT code from code_blocks
    • Not alter content
    • Commit changes atomically
    • Maintain ASCII encoding

============================================================
4. VERIFICATION GATE (MANDATORY)
============================================================
Codex MUST run dependency and boundary guards BEFORE applying any patch and AGAIN immediately AFTER patches are applied. Jarvis validates these runs occurred and were clean. Any strict failure MUST block execution and trigger a RESOLVE_ISSUE or DENIED.

Pre-Patch checks (minimum):
  python3 scripts/check_import_cycles.py --strict --focus bios,VHW,miner,core

Boundary checks (examples):
  # Report modules importing a top-level package
  python3 scripts/check_import_cycles.py --check-import Neuralis_AI

  # Forbid miner -> Neuralis_AI (boundary rule)
  python3 scripts/check_import_cycles.py --forbid-import Neuralis_AI --strict

  # Forbid prediction_engine -> miner (direct)
  python3 scripts/check_import_cycles.py --forbid-import miner --strict --focus prediction_engine

Post-Patch checks:
  # Re-run core guard after patch application
  python3 scripts/check_import_cycles.py --strict --focus bios,VHW,miner,core

Exit criteria:
  - Any non-zero exit in strict mode: Codex MUST stop; Jarvis DENIES or issues RESOLVE_ISSUE.
  - Non-strict checks inform analysis only; they cannot override strict failures.
  - ASCII-only generation remains mandatory for all source outputs.
============================================================
=== END FILE ===
