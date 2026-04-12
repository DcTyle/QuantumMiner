# Device Profit Research Gate

Purpose:
- Define the product and validation contract for the clean-room Q Bit Miner lane.
- Keep the device-backed mining objective explicit, measurable, and separate from the main runtime repository.

## Repository role

Q Bit Miner is the parallel clean-room lane for a substrate-native mining product.

- The main QuantumMiner tree remains the broader runtime and research workspace.
- Q Bit Miner is the isolated productization path for a device intended to test phase-vector and temporal-dynamics processing under real mining load.
- The target architecture is substrate-native microprocessing rather than a conventional scheduler-driven software stack.

## Success condition

Research is not considered complete just because the math is internally consistent or the simulations look promising.

- The device must run real mining work.
- Power consumption must be measured.
- Accepted shares must be recorded.
- Gross mined value and power cost must both be accounted for.
- A positive net-profit window is the completion gate.

## 5 percent Bitcoin target

The 5 percent Bitcoin goal is treated as a stretch hypothesis for research pressure testing.

- It is a benchmark target, not an assumed outcome.
- Promotion decisions should be based on observed profitable operation and validated substrate behavior, not on aspirational share claims.

## Required evidence

The confinement-side contract requires these classes of artifact for a real device run:

- hardware profile
- profit window summary
- accepted share log
- power telemetry log
- substrate state snapshot
- phase-vector ledger

These artifact names are formalized in `ResearchConfinement/QBitMinerSubstrate/spec/device_profit_validation_contract.json`.

## Export path

The clean-room CLI now exposes a deterministic artifact export path for this bundle:

```powershell
"Q Bit Miner/build/Release/Quantum Miner.exe" --import-run45-csv "ResearchConfinement/Run45/run_045_photonic_identity_trace.csv" --export-device-validation-dir "ResearchConfinement/QBitMinerSubstrate/exports/device_run"
```

This export is still a scaffold unless `--measured-device-window` is supplied with real device-backed economics. The completion gate remains false until a measured positive net-profit window exists.