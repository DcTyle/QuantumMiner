# Photonic Substrate OS

Purpose:
- Capture the research contract for a substrate-native operating model built on the repo's existing 9D trajectory and pulse telemetry work.
- Anchor the discussion in code by mapping the concept to the research-only scaffold under ResearchConfinement/prototyping/python/photonic_substrate_os.py.

## Core model

The operating model is substrate-native rather than process-loop native:

- The silicon lattice is the compute medium.
- Stable carrier waves are the process definitions.
- GPU pulse telemetry is the weak external driver.
- Temporal accounting is the scheduler.
- The saved 9D texture map is the boot and resume state.

The active process vector order in the scaffold is:

- phase
- axis_x
- axis_y
- axis_z
- flux
- coherence
- spin
- inertia
- feedback

## Research-only scaffold

The scaffold intentionally reuses existing prototype outputs instead of re-deriving a second physics model.

- PhotonicTextureMap9D stores the current field map, base tone carriers, saved resume state, and learned dynamic trajectories.
- StaticBaseTone holds the initial encoded trajectories that act as the persistent substrate reference.
- PhaseTransportObserver reads live telemetry through build_live_telemetry_payload(...) and emits a deterministic resync pulse sequence that moves the live field toward the saved texture state.
- TemporalAccountingScheduler compares the live 9D process vector against encoded carriers and emits actuation events only when temporal similarity and temporal-accuracy gates are satisfied.

## Design constraints

- Research-only: no runtime promotion from this file without validation.
- Deterministic: the same live telemetry and saved texture snapshot must produce the same resync plan and scheduler result.
- Thin integration: reuse the existing trajectory, phase-ring, photonic-identity, and temporal-accounting builders already present in the prototype lane.
- ASCII-only: all scaffold files remain ASCII-safe for repo policy compliance.

## Current implementation note

This scaffold is a contract and orchestration layer around the current prototype APIs. It does not claim that the GPU has already become a substrate-native OS in production. It gives the repository a concrete place to model:

- 9D texture persistence,
- static encoded trajectories,
- phase-transport resynchronization,
- and temporal-accounting-based process actuation.