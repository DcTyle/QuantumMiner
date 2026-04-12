# Quantum Miner Software Backbone Design

## 1. Scope

This document defines the clean-room software backbone for Quantum Miner.

The application name is Quantum Miner.

The source folder name is Q Bit Miner.

The goal is to build a deterministic substrate-driven software core that can:
- trace GPU field variables as encodable vectors
- cache low-level substrate state in C++
- produce GUI and networking hooks from that substrate state
- isolate research validation from engine integration
- support a future GPU calibration, encoding, and decoding system grounded in the field dynamics described in GenesisSubstrate_eq.md and the ResearchConfinement assets

This is intentionally separate from the current Python engine and current mixed runtime implementation.

## 2. Problem Statement

The current repository already contains working runtime code, research outputs, test assets, and GPU experiments. The codebase is useful, but the substrate-level processing path is not cleanly separated from the higher-level engine, GUI, and networking layers. That makes it difficult to:
- reason about deterministic field transport
- attach low-level GPU metrics to an explicit cache contract
- gate research before engine integration
- preserve one canonical data model for the GUI, network hooks, and GPU feedback loop

Quantum Miner needs a new backbone where the substrate is the source of truth.

## 3. Core Design Rule

The software must treat GPU returns as field observations rather than raw opaque telemetry.

Each observation becomes a deterministic substrate trace with:
- a photonic identity
- a 9D spectra state
- a field vector
- a spin inertia component
- a timing context
- an encoded pulse vector

The substrate trace is then cached and published to API hooks for GUI, networking, and external services.

## 4. Separation of Concerns
- trajectory_9d
   Production-oriented C++ substrate types, cache, runtime bus, and application glue.

3. Existing engine
   No direct dependency on research assets until the substrate trace contract is accepted.

This is the most important structural rule in the new layout.

## 5. Naming Rules
- trajectory_9d

Canonical trace object:
- PhotonicIdentity

Canonical low-level processing object:
- SubstrateTrace

For the software model, the substrate state is not a claim about literal particle ontology. It is a deterministic processing abstraction for encoding and decoding GPU feedback.

The main state groups are:
- field_vector
- spin_inertia
- spectra_9d
- coherence
- memory
- nexus
- observed_latency_ms

This maps directly to the research vocabulary without forcing the existing engine to absorb the full research stack.

## 7. Field Vector

The field vector is the observable transport driver.

Minimum components:
- amplitude
- voltage
- current
- accounting_time_ms
- next_feedback_time_ms
- closed_loop_latency_ms
- field_noise

This is based on the user requirement that we should replace pure frequency-only handling with amplitude, voltage, and amperage aware field data and use observer-like modulation on those signals.

## 8. Inertia, Temporal Coupling, and Spin

The software backbone must include inertia derived from the energy vector of the field and treat it as momentous spin on the photonic axis.

The coupling rule added here is mandatory for the clean-room model:
- inertia is amplified by coupling strength
- coupling strength is derived from phase coherence and relative temporal coupling
- rotational velocity emerges when 1D photon traces are applied to the three spatial axes
- noise is modeled as coupling-collision noise caused by spin rotation and axis-orientation transform trajectories

Required spin and coupling fields:
- axis_spin[3]
- axis_orientation[3]
- relative_temporal_coupling
- temporal_coupling_count
- momentum_score
- inertial_mass_proxy
- relativistic_correlation

Reference software formulas:

```text
energy_axis = [amplitude + voltage, current + flux, frequency + phase]
phase_coherence = mean(coherence, phase_closure)
temporal_factor = 1 + temporal_coupling_count * 0.25
coupling_strength = phase_coherence * relative_temporal_coupling * temporal_factor

rotational_velocity[i] = (one_d_photon_axis[i] + phase + axis_orientation[i])
                       * (1 + abs(axis_spin[i]))
                       * coupling_strength

transform_trajectory = norm(abs(rotational_velocity - axis_orientation))
                     + norm(abs(axis_spin - axis_orientation))

coupling_collision_noise = transform_trajectory
                         * coupling_strength
                         * (1 + thermal_noise * thermal_gain + field_noise * field_gain)

substrate_inertia = norm(energy_axis)
                  * (1 + momentum_score + inertial_mass_proxy + relativistic_correlation)
                  * (1 + coupling_strength + norm(rotational_velocity) + norm(axis_orientation))
                  * (1 + coherence * coherence_gain)

encoded_axis = energy_axis * observer_factor
             + axis_spin * substrate_inertia
             + rotational_velocity
             - axis_orientation * coupling_collision_noise
```

This makes spin, orientation, and temporal coupling part of the encoding path rather than post-hoc telemetry.

## 9. 9D Spectra and Photonic Identity

Every low-level substrate trace must expose a 9D spectra vector because the future hook surface needs stable IDs for field tracking.

The PhotonicIdentity object is the canonical bridge between research and runtime.

Required fields:
- trace_id
- gpu_device_id
- spectra_9d[9]
- field_vector
- spin_inertia
- coherence
- memory
- nexus
- observed_latency_ms

The trace_id is a software identity token, but it should be upgraded over time into a stricter photonic identity token once the research-side encoding rules mature.

## 10. Timing and Determinism

The system must track:
- request_time_ms

The system must model the interval between:
- when the host requested data
- when the GPU returned data
- when the substrate completed accounting for it

That timing is part of the substrate trace because the encoding decision depends on it.

1. flux_transport
2. phase_transport
```text
flux_transport = (amplitude * voltage) + (current * frequency) + integrated_feedback + derivative_signal - coupling_collision_noise

phase_transport = (frequency * dt) + phase + recurrence_alignment + observer_factor * flux + coupling_strength + norm(rotational_velocity)
```


The GPU feedback system needs a deterministic observer-like factor so encoding is not blind to coupling-collision disruption.
This factor should influence:
- encoded pulse generation
- priority for GUI updates
- network push policy
- cache retention policy for marginal traces

## 13. Cache Layer

- support rapid lookup by GUI hooks and network hooks
- allow later promotion into persistent storage without changing the trace contract

The first version uses an in-memory bounded deque. Later versions should support:
- memory-mapped trace pages
- GPU-side shared buffers
A runtime event should include:
- topic
- message
- SubstrateTrace

Primary topics:
- substrate.trace.ready
- substrate.trace.failed
- gui.trace.refresh
- network.trace.publish
- cache.trace.evict
- calibration.trace.accepted
- calibration.trace.rejected

This keeps GUI and networking code on top of the substrate rather than inlining GPU logic into the UI.

## 15. GUI Hook Design

The GUI should not compute substrate state. It should subscribe to substrate state.
The backbone needs explicit interference accounting because vector-transform collisions are now the primary noise mechanism in the clean-room substrate model.

Software interpretation:
- thermal_noise and field_noise remain observed inputs
The GUI hook contract must remain narrow:
- subscribe to runtime events

Networking should treat SubstrateTrace as the canonical external transport record.

Network responsibilities:
- publish substrate trace snapshots
- expose worker-session state to remote clients
- forward calibration acceptance results
- forward encoded pulse metadata for distributed monitoring
## 17. GPU Pulse Calibration Flow

1. Capture raw GPU pulse metrics in ResearchConfinement.
2. Apply deterministic flux transport and phase transport calculations.
3. Verify thermal and field interference accounting.
4. Compare encoded pulse vectors against expected research cases.
5. Promote accepted formulas into Q Bit Miner substrate code.
6. Only then expose the promoted trace contract to the engine.
## 18. Encoding and Decoding Responsibilities

- flux_transport
- substrate_inertia
Decoding must reconstruct enough state to explain why a pulse was produced.

- field_vector
- spin_inertia
- transport outputs
- timing context
The backbone needs explicit interference accounting because the user requirement is to track thermal and field-layer disruptions as encodable nodes.

Software interpretation:
- thermal_noise and field_noise are part of the substrate input, not a side note
- transport outputs must visibly degrade when those values rise
- the cache and GUI must record those disruptions for operator diagnosis

Later extensions should split interference into:
- thermal drift
- current ripple
- voltage sag
- memory bus contention
- PCIe return delay
- shader queue turbulence

## 20. Photonic Identity Lifecycle

PhotonicIdentity should eventually move through these stages:

1. Generated from GPU feedback and timing context.
2. Populated with spectra_9d and spin inertia values.
3. Assigned a stable trace_id.
4. Cached in the substrate layer.
5. Published to GUI and networking hooks.
6. Archived if accepted by research gating.

The key rule is stability: once emitted, a PhotonicIdentity record must remain reproducible from the same input frame.

## 21. Research Gate

Nothing from the research-side substrate experiments should enter the engine without passing the research contract tests.

Required acceptance gates:
- schema completeness
- photonic identity required fields present
- encoded axis pulse non-zero for non-zero field vectors
- substrate_inertia positive for positive energy vectors
- timing fields monotonic
- interference values bounded and preserved

This is why the new research folder exists.

## 22. C++ Backbone Responsibilities

The C++ backbone should own:
- photonic identity structures
- substrate trace computation
- cache
- runtime bus
- future GPU shared-memory interfaces

The backbone should not initially own:
- experimental research formulas not yet accepted
- current Python engine orchestration
- direct UI rendering logic
- quantum_miner executable
- qbit_miner_tests executable

Later targets:
- substrate_gpu_bridge
- qt_gui_frontend
- network_gateway
- calibration_importer

## 24. Migration Strategy

Migration should happen in phases.

Phase 1:
- stabilize the clean-room substrate contract
- prove research-side tests
- add cache and runtime event publishing

Phase 2:
- mirror substrate traces into the current GUI
- introduce network serialization contracts
- compare current engine telemetry against clean-room traces

Phase 3:
- promote accepted substrate path into the runtime stack
- reduce duplicate processing logic in the old engine

Phase 4:
- retire unstable or redundant old telemetry pathways

## 25. Research Folder Deliverables

ResearchConfinement/QBitMinerSubstrate should hold:

That folder is the proving ground for substrate formulas.

## 26. Calibration Export Bundle

The clean-room runtime should emit a dedicated calibration bundle when asked, rather than forcing research tools to scrape sweep data out of the full trace JSON.

Current bundle layout:
- manifest.json
- traces.jsonl
- one directory per trace
- each trace directory contains trace.json and calibration_plan.json
- each trace directory contains sweeps/NN_variable_direction.json files

This makes each ordered sweep step a first-class research artifact.
## 26. Minimal Data Contract
## 27. Minimal Data Contract
The minimal production trace contract should be:

```text
trace_id
gpu_device_id
spectra_9d
field_vector
spin_inertia
coherence
memory
nexus
request_time_ms
response_time_ms
encode_deadline_ms
encoded_pulse
phase_transport
flux_transport
observer_factor
substrate_inertia
```

Anything less makes the encoding path harder to audit.

## 28. GUI and Network API Hook Policy

The substrate layer should expose APIs in this order:

1. C++ function contract
2. runtime bus event contract
3. serialized trace contract
4. GUI adapter contract
5. network gateway contract

Every layer above the substrate consumes, but does not redefine, the trace.

## 29. Why This Backbone Matches the Research Direction

The user requirement emphasized:
- vector paths in fields
- energy-driven inertia
- spin on axes of the photonic system
- deterministic feedback from calibrated GPU pulses
- thermal and field interference tracking
- flux transport and phase transport
- observer-like modulation on amplitude, voltage, and amperage

This backbone directly encodes those concerns as software objects and transport outputs.

## 30. Immediate Next Engineering Tasks

The next clean-room coding steps should be:

1. Add real GPU calibration importer code that reads outputs from ResearchConfinement/Calibrationkernals.
2. Expand spectra_9d generation beyond placeholder storage.
3. Add serialization for SubstrateTrace.
4. Add runtime event adapters for GUI and network backends.
5. Promote the research acceptance suite into build automation.

## 31. Closing Constraint

Quantum Miner should not evolve by layering more ad-hoc logic on the old engine. It should evolve by proving substrate rules in research, then promoting those rules into the clean-room C++ backbone in Q Bit Miner.

That is the core architectural decision captured by this document.