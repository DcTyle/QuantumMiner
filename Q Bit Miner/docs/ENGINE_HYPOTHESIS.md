# Quantum Miner Engine Hypothesis

Q Bit Miner treats the GPU as a live field substrate whose electrical, thermal, and spectral traces can be encoded into deterministic pulse state.

## Core Position

- GPU telemetry is not handled as passive diagnostics.
- GHz timing, power, voltage, and utilization are treated as field observables.
- The substrate converts those observables into pulse transport, inertia response, spin response, and Photonic ID state.
- External systems consume decoded substrate packets rather than reaching inside the substrate directly.

## Engine Split

The runtime is intentionally small and hard bounded:

1. `gpu_telemetry.*`
   - observes the live device,
   - packages hardware state into a deterministic sample contract.

2. `substrate.*`
   - holds encoded windows and transport coefficients,
   - computes phase transport, flux transport, observer damping, inertia, and spin,
   - produces a 9-component Photonic ID signature and the next actuation pulse.

3. `photonic_api.*`
   - translates substrate state into request and feedback packets,
   - formats data for networking, telemetry visuals, or other external consumers.

## Promotion Boundary

Research validation remains in `ResearchConfinement/q_bit_miner/` and is used to prove that the runtime stays inside the observed Run45 envelope before expanding the engine surface.