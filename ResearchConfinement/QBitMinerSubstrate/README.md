# QBitMinerSubstrate Research Gate

This folder is the research-only validation area for the clean-room Quantum Miner substrate backbone.

Purpose:
- keep GPU pulse, calibration, and encoding experiments separate from the engine
- validate photonic identity, transport, timing, and interference contracts before promotion
- provide a stable place for scenario specs and contract tests

Rules:
- no direct dependency from engine code
- research must pass here before promotion into Q Bit Miner
- schema and cases must remain ASCII-only
- device-backed mining validation is part of the gate; simulation alone does not close the research loop
- research completion now requires an observed profitable device window with power cost accounted for

Contents:
- spec/photonic_identity_trace_schema.json: canonical trace shape for research approval
- spec/calibration_export_format.json: calibration artifact bundle layout and file contract
- spec/substrate_research_cases.json: accepted reference scenarios for deterministic validation
- spec/device_profit_validation_contract.json: device-backed profitability completion gate for the parallel substrate miner lane
- spec/hardware_profile_schema.json: hardware profile artifact contract for device validation exports
- spec/profit_window_schema.json: profitability window artifact contract for device validation exports
- tests/test_qbit_miner_research_contract.py: contract test runner