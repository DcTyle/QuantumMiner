# Temporal Dynamics Constants

Purpose:
- Define how effective constants in `core/constants_eq.py` are derived from temporal dynamics instead of static baseline values.
- Keep the implementation deterministic, ASCII-only, and compatible with the existing core equation modules.

## Inputs

The effective-constant path now consumes the existing environment scalars plus temporal-dynamics inputs:

- `env/temperature_k`
- `env/velocity_fraction_c`
- `env/flux_factor`
- `env/strain_factor`
- `env/phase_radians`
- `env/frequency_hz`
- `env/wavelength_m`
- `env/amplitude`
- `env/zero_point_offset`
- `env/vector_field`
- `env/spin_vector`
- `env/orientation_vector`

## Derived temporal terms

`core/constants_eq.py` derives the following intermediate terms before computing `h_eff`, `k_b_eff`, `c_eff`, `e_charge_eff`, `mu_0_eff`, and `epsilon_0_eff`:

- `phase_alignment`
- `zero_point_overlap`
- `vector_alignment`
- `spin_alignment`
- `orientation_alignment`
- `wave_path_length`
- `wave_path_speed`
- `cycle_time_s`
- `resonance_inertia`
- `phase_alignment_probability`
- `entanglement_weight`
- `temporal_relativity`
- `relative_temporal_position`

## Model intent

The implemented model follows this interpretation:

- Temporal position is phase-relative, not an externally fixed scalar.
- Wavelength and frequency define the carrier period and wave path speed.
- Amplitude perturbs the path length through the field.
- Zero-point proximity affects overlap and temporal normalization.
- Vector, spin, and orientation alignment weight resonance and entanglement strength.
- Effective constants remain small perturbations around reference baselines, but the perturbation is now driven by temporal-dynamics state.

## Runtime effect

All `core/*_eq.py` modules that already call `constants_eq.get_effective(...)` inherit this temporal-dynamics model automatically.
