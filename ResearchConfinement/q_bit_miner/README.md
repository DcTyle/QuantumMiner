# Q Bit Miner Validation Harness

This folder is the confinement-side promotion gate for the standalone Q Bit Miner runtime.

Scope:

- validate the encoded substrate envelope against existing research traces,
- validate Photonic ID packet structure,
- keep research checks separate from the runtime implementation in `Q Bit Miner/`.

Current anchor artifacts:

- `../Run45/run_045_photonic_identity_trace.csv`
- `../Run45/run_045_photonic_api_packets.json`

The harness treats Run45 as the current observed envelope for:

- axis scales,
- inertial mass proxy,
- spin momentum,
- transport terms,
- closed-loop timing,
- Photonic ID packet formatting.