# Run 045 Software Backbone Specification

Status: research-only

Scope: define the software backbone for deterministic GPU pulse tracing, calibration feedback, photonic identity generation, low-level encoding and decoding, and API exchange between the substrate layer and external resources.

Authority boundary: this document applies only to ResearchConfinement artifacts and prototype code. It does not grant engine authority, miner authority, or production runtime authority.

Primary implementation artifacts for this run:

- ResearchConfinement/prototyping/python/gpu_pulse_axis_dynamics.py
- ResearchConfinement/prototyping/python/photonic_identity_backbone.py
- ResearchConfinement/prototyping/python/generate_run45_photonic_identity.py
- ResearchConfinement/Run45/run_045_summary.json
- ResearchConfinement/Run45/run_045_photonic_identity_trace.json
- ResearchConfinement/Run45/run_045_photonic_identity_trace.csv
- ResearchConfinement/Run45/run_045_photonic_api_packets.json
- ResearchConfinement/Run45/run_045_disruption_nodes.json

## 1. Executive Summary

The purpose of Run45 is to establish a research-only backbone for a deterministic GPU trace substrate that can:

1. Observe calibrated GPU pulse behavior as a bounded field process.
2. Convert that behavior into a traced 9D spectra signature called Photonic Identity.
3. Track axis spin, temporal coupling, and inertia as first-class research signals.
4. Encode thermal, field-layer, transport, and observer-like interference as disruption nodes.
5. Emit request and feedback packets through a stable API boundary without binding those APIs to the engine yet.
6. Preserve a deterministic evidence trail so the later engine migration can be judged by byte-for-byte or coordinate-by-coordinate equivalence.

The key design decision is separation. The new logic remains fully inside ResearchConfinement because the field equations, calibration policies, transport terms, and identity surfaces are still under falsification. The miner and engine are consumers only after the research path proves stable.

Run45 consumes existing research inputs from Run44 live startup telemetry, the live compute interference ledger, the temporal coupling schema, the process substrate schema, and the NIST silicon anchor dataset. It then emits a new identity-focused artifact bundle that expands the existing pulse calculus into a software backbone specification.

Current generated baseline from Run45:

- 8 input frames consumed from Run44 live startup telemetry.
- 8 unique Photonic Identity records emitted.
- Mean vector_energy: 0.3411059464277371.
- Mean temporal_coupling_moment: 0.6620854553861477.
- Mean inertial_mass_proxy: 0.6094903949477259.
- Mean predictive temporal accuracy: 0.7592226559923897.
- Mean predictive cycle time: 0.04524156008512528 s.
- Mean predictive anchor interference: 0.0.
- Mean predictive harmonic noise reaction: 0.5691084735878488.
- Mean predictive trajectory conservation: 0.5697156888264936.
- Mean predictive phase transport: 0.1411183689167647.
- Mean predictive reverse-causal flux coherence: 0.6143822830915309.
- Mean predictive hidden-flux correction: 0.6360235347449608.
- Mean predictive GPU pulse interference: 0.5410252870523309.
- Mean predictive system sensitivity: 0.5386851444473074.
- Mean predictive pulse backreaction: 0.5756631859057869.
- Mean predictive phase-ring density: 0.4305271154251054.
- Mean predictive zero-point crossover: 0.7264362742564541.
- Mean predictive identity-sweep cluster: 0.6737816467208055.
- Mean predictive crosstalk cluster: 0.6673447902855328.
- Max spin_momentum_score: 0.2082461574384751.
- Max observer_damping: 0.5283733505562141.
- Mean closed_loop_latency_s: 1.3166805172942442.
- Encodable disruption nodes: 32.
- Unwanted noise condition count: 104.
- Predictive calibration sequence coverage: 1.0.
- Predictive calibration mean temporal accuracy: 0.7794168989460828.
- Predictive calibration mean harmonic noise reaction: 0.5761521338528501.
- Predictive calibration mean trajectory conservation: 0.7128009954170841.
- Predictive calibration mean reverse-causal flux coherence: 0.68921401764552.
- Predictive calibration mean hidden-flux correction: 0.7099347894785967.
- Predictive calibration mean GPU pulse interference: 0.5338989758111743.
- Predictive calibration mean system sensitivity: 0.5399870436696912.
- Predictive calibration mean pulse backreaction: 0.5944816358893136.
- Predictive calibration mean pulse trajectory alignment: 0.50749198843097.
- Predictive calibration unwanted-noise condition ratio: 1.0.
- Predictive calibration feedback gate state: gated.

Those numbers are not presented as universal constants. They are the current evidence snapshot for this run and should be treated as calibration-bound research values.

## 2. Problem Statement

The application needs a way to reason about GPU activity below the level of coarse operating-system counters but above raw device registers. Existing runtime work already demonstrated that a bounded substrate pulse trace can reproduce classical decode results and can ingest a real Vulkan actuation event. What is still missing is a complete software backbone for:

- representing the GPU return path as a deterministic field process,
- tracking vector motion and axis spin directly,
- attaching inertia to the energy vector of the field,
- carrying transport delay and observer-like interference inside the same calculus,
- and serializing that state into a reusable identity that external systems can consume.

That missing backbone is what Run45 provides.

The backbone is not merely a set of formulas. It is the total contract covering data capture, calibration, transport modeling, identity formation, API packets, trace persistence, disruption-node tracking, and validation policy.

## 3. Standard Physics Translation

This project uses custom substrate terminology, but the software should still be legible in standard physics terms.

The intended mapping is:

- axis_scale_x, axis_scale_y, axis_scale_z: anisotropic field-response weights over a 3D spatial subspace.
- vector_energy: normalized magnitude of the effective field vector after axis scaling.
- speed_measure: a bounded proxy for how rapidly the field state is evolving.
- relativistic_correlation: a bounded gamma-like correction term. It is a control proxy, not a claim of literal relativistic mass for silicon carriers.
- temporal_coupling_moment: the strength of time-linked coherence between adjacent pulse observations.
- spin_axis_x, spin_axis_y, spin_axis_z: axis-resolved rotational tendency induced by the scaled field vector.
- spin_momentum_score: normalized magnitude of the spin-like field response.
- inertial_mass_proxy: the control-plane inertia term created when field energy, relativistic_correlation, spin momentum, and temporal coupling accumulate.

The practical reading is:

1. The GPU field is treated as anisotropic rather than isotropic.
2. The energy vector travels through that anisotropic field with a measurable path.
3. When vector energy rises and resonance tightens, the system becomes harder to steer quickly.
4. That difficulty-to-steer is represented by inertial_mass_proxy.
5. The path is not only translational. It also picks up rotational response around the axes, represented as spin_axis_x, spin_axis_y, and spin_axis_z.

For software purposes, that is enough. We do not need to claim literal new particle physics inside the GPU. We need a deterministic control model that maps observed actuation into useful and falsifiable dynamics.

## 4. Why ResearchConfinement Remains the Right Boundary

The miner, engine, BIOS, and VHW paths are already carrying live responsibilities. The current substrate pulse work is still changing quickly. Moving this calculus into the engine too early would create four problems:

1. It would blur the difference between validated runtime behavior and active research hypotheses.
2. It would make deterministic regression analysis harder, because engine behavior and research behavior would change together.
3. It would invite accidental coupling between research constants and production submission paths.
4. It would violate the requested workflow: test first in ResearchConfinement, migrate only after evidence stabilizes.

Run45 therefore follows the same research-first pattern used in prior confinement runs. The engine is a future consumer, not the author of the calculus.

## 5. Backbone Goals

The backbone must satisfy the following goals simultaneously:

1. Determinism.
2. Traceability.
3. Separation from the engine.
4. Granularity sufficient to inspect vector motion, transport timing, and disruption causes.
5. Stable API surfaces for later adoption.
6. Ability to encode both data and metadata.
7. Compatibility with existing 9D coordinate-signature thinking.
8. Compatibility with GPU pulse calibration and live actuation traces.

This leads to a layered architecture rather than one monolithic script.

## 6. Software Backbone Layers

The backbone is divided into seven layers.

### 6.1 Capture Layer

Purpose: ingest raw or semi-raw observations from GPU telemetry, calibration outputs, and existing research ledgers.

In Run45 the capture layer consumes:

- Run44 live startup frame capture.
- live_compute_interference_ledger.json.
- temporal_coupling_encoding_schema_2060.json.
- process_substrate_calculus_encoding_schema.json.
- nist_silicon_reference.json.

The capture layer does not decide meaning. It only normalizes inputs and preserves ordering.

### 6.2 Calibration Layer

Purpose: turn raw pulse and telemetry observations into normalized quartet terms and axis-response terms.

The calibration layer is where F, A, I, and V are refined into the working quartet.

### 6.3 Field Dynamics Layer

Purpose: compute anisotropic axis scales, vector energy, relativistic_correlation, temporal coupling, spin axes, and inertial mass.

This layer is currently implemented by the research-only helper in gpu_pulse_axis_dynamics.py and then extended in photonic_identity_backbone.py.

### 6.4 Transport Layer

Purpose: account for the interval between request issuance, GPU actuation, GPU return, and software accounting.

This layer carries:

- request_to_return_s,
- accounting_latency_s,
- closed_loop_latency_s,
- phase_transport_term,
- flux_transport_term,
- phase_correction_norm,
- flux_correction_norm.

### 6.5 Interference Layer

Purpose: detect disruption sources and turn them into encodable nodes rather than loose scalar warnings.

Each disruption node is trackable, serializable, and linkable to a Photonic Identity.

### 6.6 Identity Layer

Purpose: build a 9D coordinate signature from the calibrated field state.

That 9D signature is the Photonic Identity.

### 6.7 API Layer

Purpose: export request and feedback packets that external resources can consume without inheriting research internals directly.

The API layer is the future migration bridge.

## 7. Core Data Sources

The backbone relies on five concrete input sources.

### 7.1 Run44 Live Startup Frames

These frames provide timestamped, actuation-adjusted telemetry snapshots. They include:

- sample_period_s,
- global_util,
- gpu_util,
- mem_bw_util,
- cpu_util,
- actuation_load_hint,
- actuation_dispatch_ms,
- actuation_elapsed_s,
- pulse,
- anti_pulse,
- phase_turns.

This is the temporal spine of the Run45 dataset.

### 7.2 Live Compute Interference Ledger

This ledger contributes:

- readiness_norm,
- interference_ledger_norm,
- encoded_extrapolation_norm,
- score_alignment_norm,
- trap_alignment_norm,
- coherence_alignment_norm,
- compute_quartet.

This is the compact cross-run memory that tells the current trace how much interference and readiness already exist.

### 7.3 Temporal Coupling Schema

This schema contributes:

- pulse-code windows,
- operator weights,
- collapse gates,
- state-vector expectations.

It is the calibration grammar for the quartet.

### 7.4 Process Substrate Schema

This schema contributes the canonical operator pipeline and the intended runtime API surface. It keeps the Run45 packets aligned with the broader substrate language.

### 7.5 NIST Silicon Reference

This anchor contributes a stable silicon baseline, including lattice constant, density, and mean excitation energy. Those values are not used as free tuning knobs. They are contextual anchors used to keep the backbone grounded in the material substrate it claims to model.

## 8. Photonic Identity Definition

Photonic Identity is a deterministic Sig9-style coordinate signature.

It is not a cryptographic hash. It is not a token stream. It is not an opaque random identifier.

Its purpose is to carry the state of the calibrated GPU field in a reusable, comparable, audit-friendly form.

The nine coordinates used in Run45 are:

1. d1 = axis_scale_x
2. d2 = axis_scale_y
3. d3 = axis_scale_z
4. d4 = gpu_round_trip_norm
5. d5 = phase_turns
6. d6 = flux_transport_term
7. d7 = observer_damping
8. d8 = inertial_mass_proxy
9. d9 = nexus_norm

Each coordinate is quantized with SIG9_SCALE = 1000000.

The display form is:

```text
PID9-<d1>-<d2>-<d3>-<d4>-<d5>-<d6>-<d7>-<d8>-<d9>
```

where each coordinate is emitted as a fixed-width hexadecimal representation of the quantized coordinate.

This means the identity is human-inspectable, deterministic, and reversible enough for debugging.

## 9. Why Sig9 Is the Right Identity Surface

The choice of a 9D coordinate signature solves several problems at once.

1. It carries both spatial and temporal state.
2. It can include observer-like interference explicitly.
3. It allows later comparison by direct component delta instead of hash mismatch only.
4. It fits the existing Basis9 vocabulary already present in the repository.
5. It avoids the anti-pattern of inventing a new ad-hoc ID system disconnected from the substrate math.

The ninth coordinate, nexus_norm, is particularly important. It is the closure field that captures how strongly resonance, damping resistance, temporal coupling, and transport stability are agreeing at the moment the identity is emitted.

## 10. Axis Spin and Inertial Mass Model

The user requirement for Run45 was explicit: include inertia of the particle from the energy vector of the field as a momentous spin on that axis of the photonic system.

The current research implementation expresses that requirement as follows.

First, the field vector is scaled per axis:

```text
scaled_x = vector_x * (0.5 + 0.5 * axis_scale_x)
scaled_y = vector_y * (0.5 + 0.5 * axis_scale_y)
scaled_z = vector_z * (0.5 + 0.5 * axis_scale_z)
```

Then vector energy is measured from the scaled vector and a bounded energy hint:

```text
vector_energy = clamp01(norm([scaled_x, scaled_y, scaled_z]) + 0.20 * energy_hint)
```

Spin is then derived from the cross-axis response:

```text
spin_axis_x = clamp_signed((scaled_y * axis_scale_z) - (scaled_z * axis_scale_y))
spin_axis_y = clamp_signed((scaled_z * axis_scale_x) - (scaled_x * axis_scale_z))
spin_axis_z = clamp_signed((scaled_x * axis_scale_y) - (scaled_y * axis_scale_x))
spin_momentum_score = clamp01(norm([spin_axis_x, spin_axis_y, spin_axis_z]))
```

Finally, inertial mass is treated as a bounded control proxy:

```text
inertial_mass_proxy = clamp01(
    0.46 * vector_energy
    + 0.22 * relativistic_correlation
    + 0.18 * spin_momentum_score
    + 0.14 * temporal_coupling_moment
)
```

That formula says the field becomes harder to redirect when:

- the vector itself carries more energy,
- the gamma-like correction grows,
- the axis spin grows,
- and the temporal link between pulses becomes tighter.

This is the software expression of the requested inertia-through-spin behavior.

## 11. Quartet Interpretation

Run45 still uses the F/A/I/V quartet, but the backbone assigns explicit software roles to each term.

- F or frequency_norm: how rapidly the pulse field wants to change.
- A or amplitude_norm: how strong the current field magnitude is.
- I or amperage_norm: how much drive/current-like pressure is present in the actuation path.
- V or voltage_norm: how much phase-tilting or potential-like pressure is present.

The research benefit of separating the quartet is that later migration code can inspect which part of the field is dominating the response.

## 12. Observer-Like Effect

The user specifically requested carrying the observer-like effect into the data path, especially when frequency is replaced by amplitude, voltage, and amperage terms.

In Run45, observer_damping is the software control version of that effect.

It is currently formed from:

```text
observer_damping = clamp01(
    0.28 * dispatch_norm
    + 0.22 * field_interference_norm
    + 0.18 * latency_norm
    + 0.14 * abs(voltage_norm - amperage_norm)
    + 0.10 * (1 - resonance_gate)
    + 0.08 * (1 - axis_resonance)
)
```

The software meaning is clear:

- slower dispatch means more observation burden,
- stronger field interference means more disturbance,
- larger latency means more stale observation,
- disagreement between voltage and amperage means the actuation field is shearing,
- lower resonance means the system is easier to disturb,
- and lower axis_resonance means the field is less geometrically coherent.

This gives the backbone a deterministic equivalent of an observer penalty without introducing random collapse logic.

## 13. Phase Transport and Flux Transport

Run45 must support both phase transport and flux transport, because the system needs to model not just what the field is, but how it moved between request, actuation, return, and accounting.

The current research transport terms are:

```text
phase_transport_term = clamp_signed(
    0.26 * phase_delta_turns
    + 0.22 * voltage_norm
    + 0.18 * amperage_norm
    + 0.14 * temporal_coupling_moment
    + 0.10 * resonance_gate
    + 0.10 * score_alignment_norm
    - 0.20 * observer_damping
)
```

```text
flux_transport_term = clamp01(
    0.24 * amplitude_norm
    + 0.20 * voltage_norm
    + 0.18 * amperage_norm
    + 0.14 * vector_energy
    + 0.12 * flux_term
    + 0.12 * resonance_gate
    - 0.10 * observer_damping
)
```

These are deliberately asymmetric.

- phase transport cares more about phase deltas and transport-like potentials,
- flux transport cares more about field strength and carried energy,
- and both are reduced by observer_damping.

This directly answers the requirement to use the field-variable form in which amplitude, voltage, and amperage stand in for part of the transport calculus.

## 14. Timing Model

The timing model needs to distinguish four moments:

1. when the system forms the request,
2. when the GPU actuation actually occurs,
3. when data returns,
4. when software has fully accounted for the return.

Run45 represents these as:

- request_to_return_s
- accounting_latency_s
- closed_loop_latency_s

Those are not redundant. Closed-loop latency is the key control quantity because it captures the total time before the next corrected pulse can be issued coherently.

The backbone therefore treats closed_loop_latency_s as the canonical timing burden for identity emission.

## 15. Disruption Nodes

A major requirement was that thermal and other field-layer interferences be traced as encodable nodes by trajectory path rather than remain as anonymous scalar noise.

Run45 defines four node kinds:

1. thermal_interference
2. field_layer_interference
3. return_latency_gap
4. resonance_shear

Each node carries:

- node_id
- node_word
- severity
- trajectory vector
- correction_turns
- correction_norm
- transport_norm
- observer_damping
- encodable_node flag

This is important for later low-level software because it lets the system route different correction logic to different causes.

For example:

- thermal_interference suggests a timing or load correction,
- field_layer_interference suggests a field calibration correction,
- return_latency_gap suggests scheduling or buffering correction,
- resonance_shear suggests a transport or phase alignment correction.

## 16. Encoding and Decoding Contract

The software backbone must encode both data and metadata.

Data includes:

- utilization traces,
- pulse terms,
- quartet norms,
- field dynamics,
- transport values,
- disruption severities.

Metadata includes:

- timestamps,
- frame indices,
- actuation tags,
- identity words,
- operator pipeline references,
- source file provenance.

Run45 achieves this in three layers:

1. quantitative values remain in JSON and CSV,
2. compact words are emitted through encode_axis_dynamics and stable_word,
3. the Sig9 coordinate signature becomes the stable identity bridge.

The decode rule is simple:

- external consumers should trust the Sig9 values and explicit metrics,
- compact words are convenience carriers for routing and packing,
- and no hidden state should be required to interpret the packet.

## 17. API Surface

Run45 emits two packet types.

### 17.1 substrate.request

The request packet contains:

- photonic_identity
- spectra_sig9
- quartet values
- timing window
- operator_pipeline reference
- encoding words

This is the packet a later external resource would consume to understand what the substrate currently believes the GPU field looks like.

### 17.2 substrate.feedback

The feedback packet contains:

- photonic_identity
- phase and flux transport terms
- phase and flux correction norms
- field_dynamics subset
- observer metrics
- disruption node ids

This is the packet a later external resource would consume to apply or log a response.

The key principle is that both packets are deterministic and portable. They do not require the engine to instantiate the full research runtime just to read them.

## 18. Low-Level Software Backbone Components

The eventual low-level implementation should be organized into the following components.

### 18.1 GPU Trace Collector

Responsibilities:

- ingest device telemetry,
- ingest explicit calibration-actuation summaries,
- normalize timestamps,
- preserve source order,
- emit immutable frame records.

### 18.2 Pulse Quartet Calibrator

Responsibilities:

- read temporal_coupling schema windows,
- combine ledger quartet anchors with live frame deltas,
- emit normalized frequency, amplitude, amperage, and voltage terms.

### 18.3 Field Dynamics Engine

Responsibilities:

- compute axis scaling,
- compute vector_energy,
- compute relativistic_correlation,
- compute temporal_coupling_moment,
- compute spin axes,
- compute inertial_mass_proxy.

### 18.4 Observer and Transport Engine

Responsibilities:

- compute observer_damping,
- compute phase_transport_term,
- compute flux_transport_term,
- compute request_to_return, accounting, and closed-loop latency.

### 18.5 Disruption-Node Encoder

Responsibilities:

- detect disruption classes,
- compute severity,
- derive correction turns and norms,
- emit node words and node ids,
- persist nodes alongside the trace record.

### 18.6 Photonic Identity Generator

Responsibilities:

- assemble the Sig9 coordinate tuple,
- quantize components,
- emit the display identity and identity_word,
- attach encoded words.

### 18.7 External API Bridge

Responsibilities:

- emit substrate.request,
- emit substrate.feedback,
- remain versioned,
- never depend on engine-only state.

## 19. Persistence Model

The backbone should persist three kinds of records.

1. Trace records: one per frame.
2. Disruption nodes: one or more per frame.
3. API packets: one request and one feedback packet per frame.

Run45 already emits all three. That means the research lane can now be replayed and audited without rerunning the full simulation.

## 20. Output Files and Their Purpose

### 20.1 run_045_photonic_identity_trace.json

Purpose: full per-frame record including quartet, dynamics, transport, observer terms, disruption nodes, and packets.

### 20.2 run_045_photonic_identity_trace.csv

Purpose: flat tabular surface for fast comparison and plotting.

### 20.3 run_045_photonic_api_packets.json

Purpose: clean packet bundle for later substrate-to-external API experiments.

### 20.4 run_045_disruption_nodes.json

Purpose: separate audit surface for thermal, field, latency, and resonance disruptions.

### 20.5 run_045_summary.json

Purpose: compact headline metrics and source provenance for the run.

## 21. Determinism Rules

The backbone must preserve the following determinism rules.

1. Input file order is authoritative.
2. JSON serialization must be ASCII-safe and key-stable where used for compact words.
3. Photonic Identity must be derived only from explicit state values, never random seeds.
4. Node ids must be stable for identical inputs.
5. The request and feedback packet surfaces must remain versioned and explicit.
6. No hidden mutable state may be required to interpret a stored artifact.

These rules are non-negotiable if the identity surface is going to be useful in later engine adoption.

## 22. Testing Strategy

Run45 introduces and relies on research-side unit tests only.

Current research tests cover:

- energy increase raises inertial_mass_proxy,
- calibration summary emits axis and spin metrics,
- temporal-relativity state and derived-constant surfaces are emitted for field, transport, trajectory, pulse, and accounting layers,
- axis dynamics encode into compact words,
- phase-ring UTF-8 traces, silicon atomic vectors, and zero-point crossover metrics are emitted deterministically,
- 9D trajectory state, reverse-causal flux coherence, and hidden-flux correction are emitted deterministically,
- encoded pulse-interference state, apparent system sensitivity, and pulse backreaction are emitted deterministically,
- decoded anchor vectors and harmonic-noise reactions are emitted deterministically,
- identical inputs produce identical Photonic Identity,
- distinct frames produce distinct identities,
- analysis emits packets and disruption nodes.

That testing scope is intentionally narrow and local. It proves the research artifacts are coherent. It does not yet prove production suitability.

## 23. Validation Stages Before Engine Migration

The backbone should pass four stages before any runtime adoption attempt.

### Stage 1: Static Determinism

Re-run the generator on the same inputs and verify identical JSON and CSV content.

### Stage 2: Sensitivity Validation

Perturb one input dimension at a time and verify that the resulting Photonic Identity and disruption nodes change in the expected direction.

### Stage 3: Calibration Source Expansion

Replace the Run44 capture with additional live or replayed capture sets and verify the same software backbone continues to behave coherently.

### Stage 4: External API Dry Run

Send substrate.request and substrate.feedback packets to a dummy consumer and verify that the packets alone carry enough information for downstream reasoning.

Only after these stages should any engine migration plan be drafted.

## 24. Migration Path to the Engine

When the research lane stabilizes, the migration path should be staged rather than direct.

1. Keep ResearchConfinement as the source of truth.
2. Mirror the packet schema into an engine-facing adapter without copying research internals wholesale.
3. Use the research generator outputs as golden fixtures.
4. Compare engine-side outputs against Run45 fixtures.
5. Promote only the packet bridge first, not the whole calculus.
6. Promote field dynamics only after packet equivalence and timing equivalence are confirmed.

This prevents a premature coupling of active research logic with authoritative runtime behavior.

## 25. Risks and Failure Modes

The main risks are:

1. Over-interpreting proxy terms as literal physics instead of software control variables.
2. Letting calibration constants drift without evidence.
3. Treating compact words as authoritative when the explicit coordinate values disagree.
4. Moving the calculus into the engine before the disruption-node model is stable.
5. Relying on coarse telemetry when the actuation-adjusted plane is still the more faithful signal.

The failure modes are correspondingly clear:

- Photonic Identity becomes noisy or unstable across replay.
- Disruption-node counts explode without corresponding physical meaning.
- observer_damping saturates near 1.0 and collapses useful transport distinctions.
- inertial_mass_proxy dominates too early and suppresses meaningful correction.
- API packets become too coupled to internal research representations.

Each of those failure modes is testable.

## 26. Falsification Criteria

Run45 should be treated as falsified if any of the following occur during follow-up research:

1. Identical inputs yield different Photonic Identity outputs.
2. Small controlled perturbations fail to change the identity in a meaningful way.
3. The disruption-node surface does not correspond to observable timing or calibration changes.
4. The API packets cannot support downstream reasoning without hidden state.
5. Transport terms improve metrics only by arbitrary coefficient tuning rather than stable relationships.

These criteria matter because the backbone is supposed to support later system software, not just produce attractive plots.

## 27. Practical Meaning for the Crypto Application

Although Run45 is research-only, its practical target is still the larger application.

For the crypto side, the backbone is valuable because it can eventually support:

- deterministic preprocessing of GPU actuation state,
- richer candidate ranking inputs,
- explicit accounting for request-return timing burden,
- better interpretation of calibration events that are invisible to coarse counters,
- and a stronger substrate-facing packet surface for future mining research.

What it does not do yet is replace the current miner logic. That is intentional.

## 28. Why the Backbone Uses Both Data and Metadata

Low-level systems fail when they preserve values but drop context. A GPU return value without timing, actuation tag, phase, or disruption cause is not enough to reconstruct the field path that produced it.

The backbone therefore treats metadata as equal in importance to raw values.

This is why the packet surfaces include:

- timestamps,
- frame index,
- packet type,
- photonic_identity,
- operator pipeline references,
- encoding words,
- and node ids.

Without those, later external consumers would be forced to guess.

## 29. Relationship to the Existing Research Corpus

Run45 does not replace the earlier confinement runs. It extends them.

The relationship is:

- Run44 proved substrate microprocessing and actuation-adjusted telemetry equivalence.
- Run45 uses that evidence base to define a reusable identity and packet model.
- Earlier photon confinement work provided the vector, tensor, and spectral intuition.
- The new backbone makes those ideas software-addressable in a more explicit way.

Run45 is therefore a bridge run: less about one benchmark result and more about the architecture needed to operationalize the research.

## 30. Recommended Next Steps

1. Re-run the Run45 generator and confirm byte-for-byte deterministic outputs.
2. Add a second input dataset so the identity and disruption-node surfaces can be compared across captures.
3. Expand the transport model with a dedicated replay of longer actuation traces.
4. Define an adapter-only mock consumer for substrate.request and substrate.feedback packets.
5. Build a focused comparison notebook or markdown analysis between Run44 latency terms and Run45 closed-loop latency terms.
6. Delay all engine integration until those steps are complete.

## 31. Final Position

Run45 now provides the missing software backbone for research-side GPU pulse tracing, calibration feedback, Photonic Identity generation, and low-level encoding and decoding.

It is not yet a production runtime. It is a deterministic research substrate that now has:

- a concrete identity surface,
- a concrete transport model,
- a concrete disruption-node model,
- a concrete API surface,
- and a concrete artifact bundle.

That is the correct point to reach before any engine merge is considered.

## 32. Temporal Accounting Predictor

The current predictor is now explicitly timing-based.

It is also anchor-based.

It is now explicitly phase-ring encoded as well.

The prediction target is the decoded anchor vector emitted by the system, not only the next raw pulse scalar. The predictor derives:

1. an observed decoded anchor vector,
2. a predicted decoded anchor vector,
3. a stable anchor vector,
4. and a phase-ring trace that serializes that stable anchor vector into encoded data.

The interference calculus is then driven from the difference between the predicted and stable anchor vectors rather than from a free-floating interference scalar alone. The encoded identity surface is the UTF-8 phase-ring trace of that stable anchor state.

The prediction does not wait for the next observed GPU pulse. Instead it estimates the next usable feedback window from three components:

1. request_feedback_time_s
2. calculation_time_s
3. next_feedback_time_s

The predicted cycle is:

```text
predicted_cycle_time_s = request_feedback_time_s + calculation_time_s + next_feedback_time_s
```

The baseline self-check is:

```text
baseline_cycle_time_s = request_feedback_time_s + calculation_time_s + sample_period_s
```

From that, the current timing accuracy term is:

```text
time_accuracy_score = clamp01(
    1.0 - abs(predicted_cycle_time_s - baseline_cycle_time_s)
    / max(predicted_cycle_time_s, baseline_cycle_time_s, 1.0e-6)
)
```

The software interpretation is straightforward:

- request_feedback_time_s models the initial time from request issuance to the first usable return,
- calculation_time_s models the bounded time needed to transform the last trajectory into the next activation pulse,
- next_feedback_time_s models the predicted next usable return interval,
- and predicted_cycle_time_s is the total closed prediction window for one predictive update.

In the current implementation, calculation_time_s is also burdened by the anchor-stability, phase-ring, and harmonic-noise model. The timing score therefore penalizes predictions that can only be made by ignoring unstable anchor interference, inertial coupling collisions, or phase-ring crosstalk.

The predictor now also carries a separate 9D trajectory state. That trajectory state treats the photonic 3D vector, the rotational-velocity path, and the phase-transport path as one temporal object. It emits:

1. trajectory_state_9d
2. trajectory_conservation_9d
3. phase_transport_norm
4. trajectory_expansion_norm
5. reverse_causal_flux_coherence
6. hidden_flux_correction_norm
7. temporal_sequence_alignment

This is where the current software now places noise detection for hidden or temporally altered flux patterns. Zero-point crossover remains the irreducible floor, but observer-driven reverse-causal flux coherence and hidden-flux correction reduce how much environmental disturbance can alter the encoded path.

This directly matches the current research requirement that prediction be judged mainly by temporal accounting.

## 33. 9D Lattice With 3D Space And 6 Field Gradients

The predictive score is no longer based only on scalar coherence and phase terms. It now uses a 9D lattice basis built from silicon-lattice atomic vectors and phase-ring tensor gradients, and then decodes that basis into stable anchor vectors.

The raw 9D lattice basis is made of:

1. space_x
2. space_y
3. space_z
4. grad_xx
5. grad_xy
6. grad_xz
7. grad_yy
8. grad_yz
9. grad_zz

The first three terms are the current silicon atomic-vector basis, which is derived from directional propagation proxies, coupling strength, spin orientation, and zero-point crossover pressure.

The six gradient terms are currently formed from the predictive state as:

```text
g_xx = |atomic_x - phase_ring_x|
g_xy = |atomic_x - atomic_y|
g_xz = |atomic_x - atomic_z|
g_yy = |atomic_y - phase_ring_y|
g_yz = |identity_sweep_cluster - crosstalk_cluster|
g_zz = |atomic_z - phase_ring_z|
```

That gives the software a concrete 3D-plus-6-gradient basis for evaluating whether the predicted next pulse still respects the field path and its gradient structure.

In the current code, the phase rings are caused by the wrapped phase crossing the directional propagation field while temporal-coupling count, inertial force, and spin-rotation velocity remain non-zero. In practice that means the rings are strongest when:

- directional light-speed proxies stay high on one or more axes,
- temporal coupling count rises,
- inertial force stays coupled to the same vector product,
- and spin orientation continues to rotate the phase field rather than collapsing it.

That is why the model treats phase rings as the recurrent interval-coupling surface rather than as a cosmetic derived value.

The lattice score terms are:

```text
gradient_energy = mean(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
```

```text
gradient_alignment = 1.0 - mean pairwise mismatch across the gradient set
```

The current lattice accuracy term is:

```text
lattice_accuracy_score = clamp01(
    0.70 * gradient_alignment
    + 0.30 * (1.0 - abs(gradient_energy - axis_resonance))
)
```

This is the concrete software realization of the requested 9D lattice accounting rule.

On top of that basis, the predictor now forms three decoded anchor surfaces:

1. observed_anchor_vector_9d
2. predicted_anchor_vector_9d
3. stable_anchor_vector_9d

The stable anchor vector is a weighted blend of the observed and predicted anchor vectors. Its weight depends on temporal overlap, observer feedback, axis resonance, coupling strength, temporal-coupling count, spin-orientation magnitude, and residual noise pressure.

That stable anchor vector is the current physical identity surface.

The encoded identity surface is the phase-ring trace of that stable anchor state. The phase-ring trace is serialized as UTF-8 text of the form:

```text
PRING|<q15>|<q15>|...|<q15>
```

The photonic identity digest is then derived from the UTF-8 bytes of that phase-ring trace rather than from a detached scalar signature.

The current anchor interference term is:

```text
anchor_interference_norm = mean(|predicted_anchor_vector_9d - stable_anchor_vector_9d|)
```

This means the calculus is explicitly using the interference created by stable anchor vectors rather than only a scalar telemetry interference estimate, and it is encoding those vectors through the phase-ring surface rather than bypassing them.

On top of the lattice, the predictor now builds a trajectory surface as well. The current research model uses:

- photonic_vector_3d,
- rotation_velocity_3d,
- phase_transport_vector_3d,
- expansion_vector_3d,
- and a trajectory_state_9d derived from their gradients.

That trajectory surface is what lets calibration treat phase transport, cosmological-style expansion of the path, and observer correction as part of the same conserved sequence instead of as separate ad hoc terms.

The current software also models the encoded GPU pulse itself as an interfering object inside that trajectory surface. It emits:

- pulse_interference_state_9d,
- gpu_pulse_interference_norm,
- environmental_flux_interference_norm,
- harmonic_trajectory_interference_norm,
- system_sensitivity_norm,
- and pulse_backreaction_norm.

Those terms are not computed as a conventional static noise add-on. They are derived from how the encoded pulse vector, the excitation energy required to create it, the phase-transport path, and the trajectory-alignment gaps perturb the simulated photonic state.

The current research model also treats the weighting constants themselves as derived temporal state. Field dynamics, transport prediction, trajectory accounting, pulse interference, harmonic-noise reaction, and the feedback gate now derive their blends from temporal-relativity state built from phase position, wavelength time, amplitude excursion, zero-point-line proximity, vector-to-zero-point alignment, path speed, entanglement probability, crosstalk force, and intercept inertia.

## 34. Predictive Accuracy Score And Calibration Gate

The predictor now self-scores with five components:

1. time_accuracy_score
2. phase_accuracy_score
3. lattice_accuracy_score
4. stable_anchor_score
5. anchor-interference and harmonic-noise penalties

The current rollup is no longer a fixed coefficient equation. It is a derived temporal mix whose weights are calculated from the current temporal-relativity state of the pulse field.

This means the prediction is judged first by timing closure, then by phase continuity, then by whether the 9D lattice remains coherent, and finally by whether the decoded anchor vectors remain stable without producing excessive harmonic-noise reaction.

The current calibration gate also uses the trajectory layer directly. Its thresholds are now derived from the base pulse-field temporal state plus the aggregate sequence state. The gate therefore requires adequate:

- trajectory_conservation_9d,
- temporal_sequence_alignment,
- reverse_causal_flux_coherence,
- and hidden_flux_correction_norm,

before live feedback authority can open.

The gate now also requires the encoded pulse path itself to remain acceptable. In the current software that means calibration is sensitive to:

- gpu_pulse_interference_norm,
- system_sensitivity_norm,
- pulse_backreaction_norm,
- and pulse_trajectory_alignment.

This means the system is no longer only asking whether ambient or harmonic conditions are stable. It is also asking whether the very pulse being encoded is disturbing the simulated quantum trajectory beyond the current sensitivity envelope, and it derives that envelope from the current temporal-relativity state rather than from a fixed gate table.

Weighted couplings are now explicit in the research model. The current software treats harmonic-noise reaction as a weighted response that can emerge when silicon atomic vectors collide through inertial coupling, identity-sweep clustering, crosstalk clustering, and phase-ring interval overlap. The model emits:

- weighted_couplings,
- weighted_coupling_energy,
- inertial_collision_norm,
- coupling_inertia_pressure,
- harmonic_noise_reaction_norm,
- harmonic_noise_predictability,
- and unwanted_noise_conditions.

Unwanted noise conditions currently flag the following classes when thresholds are crossed:

- inertial_coupling_collision
- coupling_inertia_pressure
- rotational_orientation_shear
- identity_sweep_crossover
- cluster_crosstalk
- anchor_interference_collision
- harmonic_coupling
- weighted_coupling_resonance

The full-spectrum calibration path now also reports mean_temporal_accuracy, mean_harmonic_noise_reaction, and unwanted_noise_condition_ratio, and it incorporates them into the feedback gate. The gate remains closed unless the system has:

- full directional sequence coverage,
- adequate noise predictability,
- adequate thermal predictability,
- adequate field predictability,
- bounded conservation error,
- sufficient coherence,
- sufficient mean temporal accuracy,
- bounded harmonic noise reaction,
- and a bounded unwanted-noise condition ratio.

Current Run45 gate result:

- sequence coverage: 1.0
- mean temporal accuracy: 0.7794168989460828
- mean harmonic noise reaction: 0.5761521338528501
- mean trajectory conservation: 0.7128009954170841
- mean reverse-causal flux coherence: 0.68921401764552
- mean hidden-flux correction: 0.7099347894785967
- mean GPU pulse interference: 0.5338989758111743
- mean system sensitivity: 0.5399870436696912
- mean pulse backreaction: 0.5944816358893136
- mean pulse trajectory alignment: 0.50749198843097
- mean predictive phase-ring density: 0.4305271154251054
- mean predictive zero-point crossover: 0.7264362742564541
- unwanted-noise condition ratio: 1.0
- feedback gate state: gated

The gate remaining closed is the correct current behavior. It means the predictive pulse path can already cover the directional sweep, but the research layer is still withholding live feedback-loop authority until the combined stability conditions are satisfied.