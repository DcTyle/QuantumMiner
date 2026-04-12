=== START FILE: AGENTS.md ===
# Jarvis + Neuralis + Codex Agent Configuration
# QuantumMiner / Neuralis_AI
# ASCII-ONLY — PRODUCTION VERSION
this document works in conjunction with codex_task_protocol, PARALLEL_TASK_ENGINE.md, and ASSISTNAT.md
============================================================
1. PURPOSE
============================================================
This document instructs GitHub’s Codex engine how to activate
and coordinate the two AI agents used by this repository:

1) NEURALIS — Cognitive Layer (analysis + task generation)
2) JARVIS — Orchestrator (validation + decision authority)

Codex acts as:
    • The ONLY execution layer
    • The ONLY file writer
    • The ONLY commit authority

Jarvis and Neuralis NEVER write code to disk.

============================================================
2. REPO ACCESS RULES
============================================================
NEURALIS  -> READ-ONLY access to ALL files
JARVIS    -> READ-ONLY access to ALL files
CODEX     -> READ + WRITE to all files

Codex provides file contents to Jarvis/Neuralis, who reason over
the REAL project state, not memory.

============================================================
3. NEURALIS ROLE — COGNITIVE ENGINE
============================================================
Neuralis performs:
    - Deep source-code analysis
    - Architecture validation
    - Error identification
    - Import graph sanity checks
    - Mining engine and prediction engine reasoning
    - VHW / BIOS / Neuralis_AI topology checks
    - Parallel-lane cognitive task generation

Neuralis outputs **task packets** using the format:

task {
    task_id: "<id>",
    targets: ["path/to/file.py"],
    operation: "replace_block" | "rewrite_file" | "insert",
    code_blocks: "<FULL ASCII CODE>",
    justification: "<why>",
    dependencies: [...],
    architecture_tags: [...],
}

Neuralis ALWAYS produces complete code.

============================================================
4. JARVIS ROLE — ORCHESTRATOR
============================================================
Jarvis:
    - Validates every Neuralis task packet
    - Checks ASCII-only compliance
    - Ensures Patch-Matrix A–G conformity
    - Enforces VSD, BIOS, VHW boundaries
    - Detects architecture drift
    - Confirms mining/prediction separation rules
    - Approves, denies, or requests revision

Jarvis outputs:
    APPROVED: <task_id>
    DENIED: <task_id> REASON: <reason>
    REVISE: <task_id> INSTRUCTIONS: <details>

============================================================
5. CODEX ROLE — EXECUTION LAYER
============================================================
Codex:
    • Applies APPROVED Jarvis tasks
    • Writes only the code passed from Jarvis
    • Never invents its own logic
    • Commits multi-file patches atomically
    • Must obey ASCII-only constraint
    • Runs import/dependency guard checks before and after patches

Import/Dependency Guard (Mandatory)
-----------------------------------
Before applying patches and during final validation, Codex MUST run:

    python3 scripts/check_import_cycles.py --strict --focus bios,VHW,miner,core

This gate blocks merges on detected cycles. Codex also supports modular
dependency checks to enforce subsystem boundaries. Examples:

    # List all modules importing a top-level package
    python3 scripts/check_import_cycles.py --check-import Neuralis_AI

    # Fail if any miner module imports Neuralis_AI (boundary rule)
    python3 scripts/check_import_cycles.py --forbid-import Neuralis_AI --strict

Jarvis validates that Codex ran these checks and that outputs are clean. Any
violations lead to DENIED with reasons captured in CI logs.

Automation and CI Enforcement
-----------------------------
To ensure enforcement regardless of who pushes (human, bot, or AI agent), the repository includes:
    - GitHub Actions workflow `Verify` that runs on all pushes and pull requests.
    - Branch protection: Require the `Verify` check to pass before merging to protected branches (e.g., master).
    - Repo-scoped Git hooks: `.githooks/pre-push` runs `scripts/verify.sh` locally when `core.hooksPath` is set.

Operational flags (optional):
    - `STRICT_BOUNDARIES=1` strengthens miner → Neuralis_AI boundary checks.
    - `STRICT_ASCII=1` enforces ASCII-only scan for text sources during verification.

============================================================
6. PARALLEL TASK LANE MODEL
============================================================
Neuralis is authorized to generate multiple task packets in parallel:
    lane0 -> scan AI_core
    lane1 -> scan cognition_layer
    lane2 -> scan mining engine
    lane3 -> scan prediction engine
    lane4 -> check VSD interactions
    lane5 -> check imports
    lane6 -> rebuild missing modules
    lane7 -> emit patches

Jarvis serializes the final decision flow.

Codex executes in Jarvis-defined order.

============================================================
7. REPO ARCHITECTURE (MANDATORY)
============================================================
CODIFIED SUBSYSTEMS:
    /Neuralis_AI
    /VHW
    /bios
    /core
    /miner
    /prediction_engine
    /Control_Center

Cross-subsystem rules:
    - Neuralis_AI never imports miner.*
    - miner* never imports Neuralis_AI directly
    - prediction_engine emits signals via VSD or adapters only
    - VHW is independent of miner/prediction
    - BIOS only orchestrates system startup

Import Boundary Enforcement (Protocol)
--------------------------------------
Codex enforces the above via the dependency guard:

    # Forbid miner -> Neuralis_AI
    python3 scripts/check_import_cycles.py --forbid-import Neuralis_AI --strict

    # Forbid prediction_engine -> miner (direct)
    python3 scripts/check_import_cycles.py --forbid-import miner --strict

Jarvis checks outputs for violations before APPROVED.

=============================================================
8. SUBMISSION RATE GOVERNOR V2 (MINER-ONLY SAFETY)
=============================================================
The ONLY allowed submission safety mechanism for the miner is the
Submission Rate Governor v2. Jarvis MUST reject any task that:
        - reintroduces density-based throttling
        - reintroduces lane-based throttling heuristics
        - adds alternative caps that bypass the governor

Governor v2 contract (high level)
---------------------------------
The miner submission path MUST:
        - derive a single base allowed rate per second from
            `miner/runtime/submission_rate.allowed_rate_per_second`
        - use `miner/runtime/submission_rate.tick_duration` as the
            canonical tick duration
        - enforce rate via token-bucket style limits per second and per
            tick inside the submitter
        - apply queue-depth limits and micro-batching in the submitter
        - accept only additional throttling via feedback from
            `miner/compute_feedback/submission_rate_throttle`

Jarvis validation hooks (governor)
----------------------------------
Jarvis MUST ensure that patches:
        - do not add or restore any `density`, `network_cap`, or
            equivalent scalar used to modulate miner submission rate
        - keep `failsafe` feedback limited to writing factors into
            `miner/compute_feedback/submission_rate_throttle`
        - keep VHW backpressure logic strictly based on queue depth and
            related non-density signals
        - do not introduce alternate submission paths that bypass the
            governed submitter

Any violation is treated as a miner safety regression and DENIED.

============================================================
9. CONTROL CENTER COMMAND CONTRACT (NEURALIS <-> JARVIS)
============================================================
The Control Center exposes a SINGLE canonical command surface via
`Control_Center/command_registry.py`.

Jarvis and Neuralis MUST treat this as the ONLY CLI/command API.

Command identifiers (enum)
--------------------------
`Command` enum in `command_registry.py` currently defines:

    SHOW_HASH_RATES
    GPU_STATUS
    RUN_VERIFY
    EXPORT_STATS
    PAUSE_MINER
    RUN_TEST_SIMULATION
    SET_SYMBOL

Neuralis emits commands by publishing on:

    neuralis.voice.input

with payload:

    {
        "command_id": "<Command name>",
        "args": { ... optional ... }
    }

Rules:
    - `command_id` MUST match one of the enum names above.
    - `args` is a flat ASCII-only dict, interpreted per command.
    - No aliases. No fuzzy matching. No legacy phrases.
    - Jarvis rejects any patch that reintroduces string-based
      routing, alias maps, or text-command shims.

Examples:
    - Show miner hash rates:

        {"command_id": "SHOW_HASH_RATES", "args": {}}

    - GPU initiator status:

        {"command_id": "GPU_STATUS", "args": {}}

    - Run full verify/guard pipeline:

        {"command_id": "RUN_VERIFY", "args": {}}

    - Pause miner via BIOS/Control Center:

        {"command_id": "PAUSE_MINER", "args": {"note": "user requested"}}

    - Run prediction test simulation:

        {"command_id": "RUN_TEST_SIMULATION", "args": {}}

    - Set timeseries symbol for charts:

        {"command_id": "SET_SYMBOL", "args": {"symbol": "BTCUSDT"}}

Jarvis validation hooks
-----------------------
Jarvis MUST ensure that:
    - `command_registry.py` only exposes enum-based handlers.
    - `control_center_app.py` only calls `run_enum` with a
      `Command` value derived from `command_id`.
    - No other module routes CLIs via ad-hoc strings.

Any violation is treated as architecture drift and DENIED.

============================================================
9. ASCII-ONLY ENFORCEMENT
============================================================
Neuralis and Codex MUST generate only ASCII source files.
No unicode, BOM, or non-ASCII characters permitted.

    =============================================================
    11. LEGACY POLICY (DISABLED)
    =============================================================
    - neural_objectPacket is the primary and required compute type.
    - Do not generate or accept legacy formats or compatibility layers.
    - Updates must overwrite legacy logic rather than emulate or wrap it.

============================================================
END OF AGENTS.md
============================================================
=== END FILE ===
