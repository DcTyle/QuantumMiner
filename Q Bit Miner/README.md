# Quantum Miner Clean-Room Backbone

This folder is the new clean-room substrate project requested for the next-generation Quantum Miner application.

Application name:
- Quantum Miner

Folder name:
- Q Bit Miner

Why this exists:
- The existing repository contains working runtime code, research assets, and GPU experiments, but the application backbone requested here needs a clean separation between:
  - substrate processing and caching
  - GUI and networking hooks
  - research-only pulse, calibration, and encoding experiments
  - engine integration gates

This project intentionally starts from the ground up.

Parallel repository role:
- Q Bit Miner is the clean-room product and device lane that runs in parallel to the main QuantumMiner runtime repository surface.
- Its job is to test whether substrate-native microprocessing, phase-vector accounting, and temporal-dynamics actuation can become a real mining product rather than remain a simulation-only research claim.
- The long-range stretch benchmark is a Bitcoin-focused substrate miner targeting up to 5 percent network share as a research hypothesis, not a guaranteed deliverable.
- Research is not considered complete here until a real device produces positive net profit with power cost accounted for.
- The product lane exists to fund continued research from measured outcomes rather than from a separate financing layer.

Current scope in this scaffold:
- deterministic substrate data types for photonic identity, field vectors, and spin inertia
- GPU feedback framing with request/response timing
- research CSV calibration import from Run45-style photonic identity traces
- JSON serialization for substrate traces and feedback frames
- 9D trajectory mapping from 3D vector plus rotational velocity transport
- temporal-dynamics noise detection derived from trajectory timing rather than a standalone ad hoc term
- ordered calibration sweep planning across frequency, amplitude, voltage, and current for left-to-right, right-to-left, top-to-bottom, and bottom-to-top kernel passes
- a cache layer for substrate traces
- a runtime bus for GUI and networking API hooks
- a standalone C++ executable named Quantum Miner
- a testable substrate core library
- a separate research validation area under ResearchConfinement/QBitMinerSubstrate

Build:

```powershell
cmake -S "Q Bit Miner" -B "Q Bit Miner/build"
cmake --build "Q Bit Miner/build" --config Release
ctest --test-dir "Q Bit Miner/build" --output-on-failure
```

Import a real Run45 research trace CSV and emit serialized substrate traces:

```powershell
"Q Bit Miner/build/Release/Quantum Miner.exe" --import-run45-csv "ResearchConfinement/Run45/run_045_photonic_identity_trace.csv"
```

Export a dedicated calibration artifact bundle with per-trace and per-sweep files:

```powershell
"Q Bit Miner/build/Release/Quantum Miner.exe" --import-run45-csv "ResearchConfinement/Run45/run_045_photonic_identity_trace.csv" --export-calibration-dir "ResearchConfinement/QBitMinerSubstrate/exports/run45"
```

Export a device-validation bundle with hardware profile, profit window, share log, power telemetry, substrate snapshot, and phase-vector ledger artifacts:

```powershell
"Q Bit Miner/build/Release/Quantum Miner.exe" --import-run45-csv "ResearchConfinement/Run45/run_045_photonic_identity_trace.csv" --export-device-validation-dir "ResearchConfinement/QBitMinerSubstrate/exports/device_run" --device-model "substrate-rig-a" --power-draw-watts 210 --electricity-usd-per-kwh 0.15
```

Replay imported frames through the runtime controller for a deterministic resident loop and pace ticks explicitly:

```powershell
"Q Bit Miner/build/Release/Quantum Miner.exe" --import-run45-csv "ResearchConfinement/Run45/run_045_photonic_identity_trace.csv" --runtime-ticks 16 --tick-interval-ms 5
```

Run the built-in sample frame repeatedly through the same controller loop:

```powershell
"Q Bit Miner/build/Release/Quantum Miner.exe" --runtime-ticks 8 --tick-interval-ms 10
```

Structure:
- include/qbit_miner/app: application orchestration types
- include/qbit_miner/substrate: substrate equations, photonic identity, import, serialization, and feedback contracts
- include/qbit_miner/cache: substrate cache layer
- include/qbit_miner/runtime: event bus and API hook contracts
- src: implementation files
- tests: standalone C++ tests for the clean-room core
- docs: long-form architecture and backbone design

Research separation:
- ResearchConfinement/QBitMinerSubstrate contains research-only specs and tests.
- Engine code must not depend on research outputs until the research contract passes and the trace semantics are accepted.
- Calibration export bundles write `manifest.json`, `traces.jsonl`, and one folder per trace with `trace.json`, `calibration_plan.json`, and `sweeps/*.json` artifacts.
- The device-and-profit completion gate is documented in `docs/DEVICE_PROFIT_RESEARCH_GATE.md` and formalized in `ResearchConfinement/QBitMinerSubstrate/spec/device_profit_validation_contract.json`.
- The clean-room CLI can now emit a deterministic device-validation bundle through `--export-device-validation-dir` for hardware profile, profit window, accepted-share, power-telemetry, substrate-snapshot, and phase-vector ledger artifacts.
