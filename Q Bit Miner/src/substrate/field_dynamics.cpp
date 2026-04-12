#include "qbit_miner/substrate/field_dynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <sstream>

namespace qbit_miner {

namespace {

constexpr double kTau = 6.28318530717958647692;

[[nodiscard]] double clamp_unit(double value) {
    return std::clamp(value, 0.0, 1.0);
}

[[nodiscard]] double clamp_signed(double value, double limit = 1.0) {
    return std::clamp(value, -std::abs(limit), std::abs(limit));
}

[[nodiscard]] double wrap_turns(double value) {
    const double wrapped = std::fmod(value, 1.0);
    return wrapped < 0.0 ? wrapped + 1.0 : wrapped;
}

[[nodiscard]] double phase_delta_turns(double lhs, double rhs) {
    double delta = wrap_turns(lhs) - wrap_turns(rhs);
    if (delta > 0.5) {
        delta -= 1.0;
    } else if (delta < -0.5) {
        delta += 1.0;
    }
    return delta;
}

[[nodiscard]] double safe_divide(double numerator, double denominator, double fallback = 0.0) {
    return std::abs(denominator) <= 1.0e-12 ? fallback : (numerator / denominator);
}

[[nodiscard]] double normalize_positive(double value) {
    return clamp_unit(value / (1.0 + std::abs(value)));
}

[[nodiscard]] double mean_unit(std::initializer_list<double> values) {
    if (values.size() == 0U) {
        return 0.0;
    }
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return clamp_unit(total / static_cast<double>(values.size()));
}

[[nodiscard]] double vector_norm3(const std::array<double, 3>& axis) {
    return std::sqrt(std::max(0.0, (axis[0] * axis[0]) + (axis[1] * axis[1]) + (axis[2] * axis[2])));
}

[[nodiscard]] double vector_energy3(const std::array<double, 3>& axis) {
    return clamp_unit(vector_norm3(axis) / std::sqrt(3.0));
}

[[nodiscard]] double alignment(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
    const double lhs_norm = vector_norm3(lhs);
    const double rhs_norm = vector_norm3(rhs);
    if (lhs_norm <= 1.0e-12 || rhs_norm <= 1.0e-12) {
        return 0.0;
    }
    const double dot = (lhs[0] * rhs[0]) + (lhs[1] * rhs[1]) + (lhs[2] * rhs[2]);
    return clamp_unit(0.5 * ((dot / (lhs_norm * rhs_norm)) + 1.0));
}

[[nodiscard]] std::array<double, 3> vector_abs_delta(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
    return {
        std::fabs(lhs[0] - rhs[0]),
        std::fabs(lhs[1] - rhs[1]),
        std::fabs(lhs[2] - rhs[2]),
    };
}

[[nodiscard]] bool has_nonzero_spectra(const std::array<double, 9>& spectra) {
    for (double value : spectra) {
        if (std::abs(value) > 1.0e-12) {
            return true;
        }
    }
    return false;
}

}  // namespace

FieldDynamicsEngine::FieldDynamicsEngine(FieldDynamicsConfig config)
    : config_(config) {}

const FieldDynamicsConfig& FieldDynamicsEngine::config() const noexcept {
    return config_;
}

DerivedTemporalConstants FieldDynamicsEngine::derive_temporal_constants(const GpuFeedbackFrame& frame) const {
    const auto& identity = frame.photonic_identity;
    const auto& field = identity.field_vector;
    const auto& spin = identity.spin_inertia;

    const double phase_position = wrap_turns(field.phase);
    const double frequency = clamp_unit(field.frequency);
    const double amplitude = clamp_unit(field.amplitude);
    const double current = clamp_unit(field.current);
    const double voltage = clamp_unit(field.voltage);
    const double flux = clamp_unit(field.flux);

    const std::array<double, 3> vector_axis{
        clamp_signed((amplitude + voltage) - 0.5),
        clamp_signed((current + flux) - 0.5),
        clamp_signed((frequency + phase_position) - 0.5),
    };
    const std::array<double, 3> spin_axis{
        clamp_signed(spin.axis_spin[0]),
        clamp_signed(spin.axis_spin[1]),
        clamp_signed(spin.axis_spin[2]),
    };
    const std::array<double, 3> orientation_axis{
        clamp_signed(spin.axis_orientation[0]),
        clamp_signed(spin.axis_orientation[1]),
        clamp_signed(spin.axis_orientation[2]),
    };

    const double phase_alignment = clamp_unit(0.5 * (1.0 + std::cos(kTau * std::abs(phase_delta_turns(phase_position, 0.0)))));
    const double zero_point_overlap = clamp_unit(1.0 / (1.0 + (amplitude * std::abs(std::sin(kTau * phase_position)))));
    const double vector_alignment_score = alignment(vector_axis, orientation_axis);
    const double spin_alignment_score = alignment(vector_axis, spin_axis);
    const double orientation_alignment_score = alignment(orientation_axis, spin_axis);

    const double axis_scale_x = mean_unit({amplitude, voltage, vector_alignment_score, std::abs(spin_axis[0]), clamp_unit(identity.coherence)});
    const double axis_scale_y = mean_unit({current, flux, spin_alignment_score, std::abs(spin_axis[1]), clamp_unit(frame.lattice_closure)});
    const double axis_scale_z = mean_unit({frequency, phase_alignment, orientation_alignment_score, std::abs(spin_axis[2]), clamp_unit(frame.phase_closure)});
    const double axis_resonance = clamp_unit(1.0 - mean_unit({
        std::abs(axis_scale_x - axis_scale_y),
        std::abs(axis_scale_y - axis_scale_z),
        std::abs(axis_scale_z - axis_scale_x),
    }));

    const double vector_energy = mean_unit({
        vector_energy3(vector_axis),
        amplitude,
        current,
        voltage,
        clamp_unit(identity.memory),
        clamp_unit(identity.nexus),
        axis_resonance,
    });
    const double field_wavelength = safe_divide(1.0 + amplitude + (0.5 * phase_alignment), std::max(frequency, config_.numeric_epsilon), 0.0);
    const double path_speed = frequency * field_wavelength;
    const double path_speed_norm = normalize_positive(path_speed);
    const double field_wavelength_norm = normalize_positive(field_wavelength);
    const double time_per_field = safe_divide(field_wavelength, std::max(path_speed, config_.numeric_epsilon), 0.0);
    const double zero_point_line_distance = mean_unit({std::abs(axis_scale_x - 0.5), std::abs(axis_scale_y - 0.5), std::abs(axis_scale_z - 0.5), 1.0 - zero_point_overlap});
    const double zero_point_proximity = clamp_unit(1.0 - zero_point_line_distance);
    const double relative_temporal_position = mean_unit({phase_position, phase_alignment, field_wavelength_norm, clamp_unit(frame.recurrence_alignment), clamp_unit(frame.phase_closure), clamp_unit(frame.conservation_alignment)});
    const double temporal_relativity_norm = mean_unit({relative_temporal_position, zero_point_proximity, axis_resonance, 1.0 - clamp_unit(field.thermal_noise), 1.0 - clamp_unit(field.field_noise)});
    const double temporal_relativity = 1.0 + (1.5 * temporal_relativity_norm);
    const double phase_alignment_probability = clamp_unit(phase_alignment * mean_unit({vector_alignment_score, spin_alignment_score, orientation_alignment_score, clamp_unit(identity.coherence)}));
    const double resonance_intercept_force = mean_unit({clamp_unit(spin.inertial_mass_proxy), clamp_unit(spin.momentum_score), vector_energy, temporal_relativity_norm, axis_resonance});
    const double entanglement_weight = mean_unit({phase_alignment_probability, spin_alignment_score, orientation_alignment_score, zero_point_overlap});
    const double crosstalk_weight = mean_unit({clamp_unit(field.thermal_noise), clamp_unit(field.field_noise), entanglement_weight, 1.0 - axis_resonance});
    const double normalization_drive = mean_unit({axis_resonance, vector_energy, clamp_unit(identity.memory), 1.0 - crosstalk_weight});
    const double rotational_velocity_norm = mean_unit({vector_energy3(spin_axis), axis_resonance, spin_alignment_score, orientation_alignment_score});
    const double field_interference_norm = mean_unit({clamp_unit(field.thermal_noise), clamp_unit(field.field_noise), crosstalk_weight, 1.0 - zero_point_overlap, 1.0 - phase_alignment});

    DerivedTemporalConstants derived;
    derived.phase_position_turns = phase_position;
    derived.phase_alignment = phase_alignment;
    derived.zero_point_overlap = zero_point_overlap;
    derived.vector_alignment = vector_alignment_score;
    derived.spin_alignment = spin_alignment_score;
    derived.orientation_alignment = orientation_alignment_score;
    derived.axis_scale_x = axis_scale_x;
    derived.axis_scale_y = axis_scale_y;
    derived.axis_scale_z = axis_scale_z;
    derived.axis_resonance = axis_resonance;
    derived.vector_energy = vector_energy;
    derived.path_speed = path_speed;
    derived.path_speed_norm = path_speed_norm;
    derived.field_wavelength = field_wavelength;
    derived.field_wavelength_norm = field_wavelength_norm;
    derived.time_per_field = time_per_field;
    derived.zero_point_line_distance = zero_point_line_distance;
    derived.zero_point_proximity = zero_point_proximity;
    derived.relative_temporal_position = relative_temporal_position;
    derived.temporal_relativity = temporal_relativity;
    derived.temporal_relativity_norm = temporal_relativity_norm;
    derived.phase_alignment_probability = phase_alignment_probability;
    derived.resonance_intercept_force = resonance_intercept_force;
    derived.entanglement_weight = entanglement_weight;
    derived.crosstalk_weight = crosstalk_weight;
    derived.normalization_drive = normalization_drive;
    derived.rotational_velocity_norm = rotational_velocity_norm;
    derived.field_interference_norm = field_interference_norm;
    derived.observer_gain = std::max(0.05, 0.18 + (0.24 * phase_alignment_probability) + (0.12 * temporal_relativity_norm));
    derived.coupling_gain = std::max(0.05, 0.22 + (0.28 * axis_resonance) + (0.12 * entanglement_weight));
    derived.collision_gain = std::max(0.05, 0.18 + (0.22 * crosstalk_weight));
    derived.rotation_gain = std::max(0.05, 0.16 + (0.24 * rotational_velocity_norm));
    derived.coherence_gain = std::max(0.05, 0.18 + (0.22 * clamp_unit(identity.coherence)));
    derived.inertia_gain = std::max(0.05, 0.22 + (0.20 * resonance_intercept_force));
    derived.phase_gain = std::max(0.05, 0.18 + (0.20 * phase_alignment));
    derived.flux_gain = std::max(0.05, 0.22 + (0.16 * vector_energy));
    derived.expansion_gain = std::max(0.05, 0.14 + (0.14 * temporal_relativity_norm));
    derived.reverse_causal_gain = std::max(0.05, 0.18 + (0.14 * zero_point_proximity));
    derived.zero_point_gain = std::max(0.05, 0.18 + (0.18 * zero_point_overlap));
    derived.temporal_noise_gain = std::max(0.05, 0.14 + (0.18 * field_interference_norm));
    derived.effective_wavelength_step = std::max(config_.minimal_gpu_wavelength_step, config_.minimal_gpu_wavelength_step * (1.0 + field_wavelength_norm + (0.5 * temporal_relativity_norm)));
    return derived;
}

double FieldDynamicsEngine::compute_expansion_factor(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const double closed_loop = std::max(0.0, frame.timing.closed_loop_latency_ms);
    const double next_feedback = std::max(0.0, frame.timing.next_feedback_time_ms);
    const double denominator = std::max(1.0, frame.timing.encode_deadline_ms + next_feedback);
    return 1.0 + (((closed_loop + next_feedback) / denominator) * derived.expansion_gain);
}

double FieldDynamicsEngine::compute_coupling_strength(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& identity = frame.photonic_identity;
    const auto& spin = identity.spin_inertia;
    const double phase_coherence = clamp_unit((clamp_unit(identity.coherence) + clamp_unit(frame.phase_closure)) * 0.5);
    const double temporal_factor = 1.0 + (static_cast<double>(spin.temporal_coupling_count) * 0.25);
    return std::max(0.0, phase_coherence * std::max(0.0, spin.relative_temporal_coupling) * temporal_factor * derived.coupling_gain);
}

std::array<double, 3> FieldDynamicsEngine::compute_rotational_velocity(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& field = frame.photonic_identity.field_vector;
    const auto& spin = frame.photonic_identity.spin_inertia;
    const double coupling = compute_coupling_strength(frame);
    return {
        (derived.axis_scale_x + field.phase + spin.axis_orientation[0]) * (1.0 + std::fabs(spin.axis_spin[0])) * coupling * derived.rotation_gain,
        (derived.axis_scale_y + field.phase + spin.axis_orientation[1]) * (1.0 + std::fabs(spin.axis_spin[1])) * coupling * derived.rotation_gain,
        (derived.axis_scale_z + field.phase + spin.axis_orientation[2]) * (1.0 + std::fabs(spin.axis_spin[2])) * coupling * derived.rotation_gain,
    };
}

double FieldDynamicsEngine::compute_coupling_collision_noise(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& field = frame.photonic_identity.field_vector;
    const auto& spin = frame.photonic_identity.spin_inertia;
    const double coupling = compute_coupling_strength(frame);
    const auto rotational_velocity = compute_rotational_velocity(frame);
    const auto orientation_delta = vector_abs_delta(rotational_velocity, spin.axis_orientation);
    const auto spin_delta = vector_abs_delta(spin.axis_spin, spin.axis_orientation);
    const double transform_trajectory = vector_norm3(orientation_delta) + vector_norm3(spin_delta);
    const double interference_factor = 1.0 + (field.thermal_noise * derived.collision_gain) + (field.field_noise * derived.collision_gain);
    return std::max(0.0, transform_trajectory * coupling * interference_factor * derived.collision_gain);
}

double FieldDynamicsEngine::compute_observer_factor(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const double closure_term = mean_unit({clamp_unit(frame.phase_closure), clamp_unit(frame.lattice_closure), clamp_unit(frame.conservation_alignment)});
    return clamp_unit((closure_term * derived.observer_gain) + (1.0 - compute_coupling_collision_noise(frame)));
}

double FieldDynamicsEngine::compute_flux_transport(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& field = frame.photonic_identity.field_vector;
    const double energy_vector = (field.amplitude * field.voltage) + (field.current * field.frequency);
    const double feedback = frame.integrated_feedback + frame.derivative_signal + field.flux;
    return (energy_vector + feedback - compute_coupling_collision_noise(frame)) * derived.flux_gain;
}

double FieldDynamicsEngine::compute_phase_transport(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& field = frame.photonic_identity.field_vector;
    const double expansion = compute_expansion_factor(frame);
    const double coupling = compute_coupling_strength(frame);
    const double rotation_norm = vector_norm3(compute_rotational_velocity(frame));
    return ((field.frequency * derived.effective_wavelength_step) + field.phase + frame.recurrence_alignment + (field.flux * expansion) + coupling + rotation_norm) * derived.phase_gain;
}

double FieldDynamicsEngine::compute_reverse_causal_flux_coherence(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& identity = frame.photonic_identity;
    const auto& field = identity.field_vector;
    const double timing_bias = std::max(0.0, frame.timing.closed_loop_latency_ms - frame.timing.response_time_ms);
    const double timing_norm = timing_bias / std::max(1.0, frame.timing.closed_loop_latency_ms + frame.timing.next_feedback_time_ms);
    const double phase_anchor = mean_unit({clamp_unit(identity.coherence), clamp_unit(frame.phase_closure), clamp_unit(identity.nexus)});
    const double flux_bias = clamp_unit(field.flux + frame.recurrence_alignment);
    return clamp_unit(((phase_anchor + flux_bias) * 0.5 * derived.reverse_causal_gain) + (1.0 - timing_norm));
}

double FieldDynamicsEngine::compute_temporal_dynamics_noise(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const double closed_loop = std::max(0.0, frame.timing.closed_loop_latency_ms);
    const double response = std::max(0.0, frame.timing.response_time_ms);
    const double accounting = std::max(0.0, frame.timing.accounting_time_ms);
    const double phase_transport = compute_phase_transport(frame);
    const double phase_slip = std::fabs(phase_transport - frame.photonic_identity.field_vector.phase);
    const double temporal_slip = std::fabs(closed_loop - (response + accounting)) / std::max(1.0, closed_loop + frame.timing.next_feedback_time_ms);
    return std::max(0.0, (temporal_slip + safe_divide(phase_slip, std::max(1.0, std::abs(phase_transport)), 0.0)) * (1.0 - compute_reverse_causal_flux_coherence(frame)) * derived.temporal_noise_gain);
}

double FieldDynamicsEngine::compute_constant_phase_alignment(const GpuFeedbackFrame& frame) const {
    const auto& field = frame.photonic_identity.field_vector;
    const double phase_transport = compute_phase_transport(frame);
    const double phase_delta = std::fabs(phase_transport - (field.phase + frame.derivative_signal));
    return clamp_unit((1.0 - safe_divide(phase_delta, std::max(1.0, std::abs(phase_transport)), 0.0)) * compute_reverse_causal_flux_coherence(frame));
}

double FieldDynamicsEngine::compute_zero_point_overlap_score(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const double expansion = compute_expansion_factor(frame) - 1.0;
    return clamp_unit(((compute_reverse_causal_flux_coherence(frame) + compute_constant_phase_alignment(frame) + compute_coupling_strength(frame)) / 3.0) * derived.zero_point_gain + expansion - compute_temporal_dynamics_noise(frame));
}

double FieldDynamicsEngine::compute_trajectory_conservation_score(const GpuFeedbackFrame& frame) const {
    return clamp_unit((compute_reverse_causal_flux_coherence(frame) + compute_constant_phase_alignment(frame) + (1.0 - std::min(1.0, compute_temporal_dynamics_noise(frame)))) / 3.0);
}

std::array<double, 9> FieldDynamicsEngine::compute_trajectory_9d(const GpuFeedbackFrame& frame) const {
    const auto& field = frame.photonic_identity.field_vector;
    const auto rotational_velocity = compute_rotational_velocity(frame);
    const double expansion = compute_expansion_factor(frame);
    const double temporal_noise = compute_temporal_dynamics_noise(frame);
    return {
        (field.amplitude + rotational_velocity[0]) * expansion,
        (field.current + rotational_velocity[1]) * expansion,
        (field.frequency + rotational_velocity[2]) * expansion,
        compute_phase_transport(frame),
        compute_flux_transport(frame),
        compute_coupling_strength(frame),
        compute_reverse_causal_flux_coherence(frame),
        compute_zero_point_overlap_score(frame),
        compute_trajectory_conservation_score(frame) - temporal_noise,
    };
}

double FieldDynamicsEngine::compute_substrate_inertia(const GpuFeedbackFrame& frame) const {
    const auto derived = derive_temporal_constants(frame);
    const auto& identity = frame.photonic_identity;
    const auto& field = identity.field_vector;
    const auto& spin = identity.spin_inertia;
    const std::array<double, 3> energy_axis{field.amplitude + field.voltage, field.current + field.flux, field.frequency + field.phase};
    const double energy_norm = vector_norm3(energy_axis);
    const double momentum = std::max(spin.momentum_score, vector_norm3(spin.axis_spin));
    const double coherence_force = 1.0 + (clamp_unit(identity.coherence) * derived.coherence_gain);
    const double coupling_force = 1.0 + compute_coupling_strength(frame) + vector_norm3(compute_rotational_velocity(frame)) + vector_norm3(spin.axis_orientation);
    return energy_norm * (1.0 + momentum + spin.inertial_mass_proxy + spin.relativistic_correlation) * coupling_force * coherence_force * derived.inertia_gain;
}

std::array<double, 3> FieldDynamicsEngine::encode_pulse_vector(const GpuFeedbackFrame& frame) const {
    const auto& field = frame.photonic_identity.field_vector;
    const auto& spin = frame.photonic_identity.spin_inertia;
    const auto rotational_velocity = compute_rotational_velocity(frame);
    const double observer = compute_observer_factor(frame);
    const double inertia = compute_substrate_inertia(frame);
    const double collision_noise = compute_coupling_collision_noise(frame);
    return {
        ((field.amplitude + field.voltage) * observer) + (spin.axis_spin[0] * inertia) + rotational_velocity[0] - (spin.axis_orientation[0] * collision_noise),
        ((field.current + field.flux) * observer) + (spin.axis_spin[1] * inertia) + rotational_velocity[1] - (spin.axis_orientation[1] * collision_noise),
        ((field.frequency + field.phase) * observer) + (spin.axis_spin[2] * inertia) + rotational_velocity[2] - (spin.axis_orientation[2] * collision_noise),
    };
}

CalibrationPlan FieldDynamicsEngine::build_calibration_plan(const GpuFeedbackFrame& frame) const {
    CalibrationPlan plan;
    plan.minimal_wavelength_step = config_.minimal_gpu_wavelength_step;
    plan.trajectory_conservation_score = compute_trajectory_conservation_score(frame);
    plan.zero_point_overlap_score = compute_zero_point_overlap_score(frame);
    plan.reverse_causal_flux_coherence = compute_reverse_causal_flux_coherence(frame);

    const std::array<std::string, 4> variables{"frequency", "amplitude", "voltage", "current"};
    const std::array<std::string, 4> directions{"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"};
    const double alignment_score = compute_constant_phase_alignment(frame);
    for (const auto& variable : variables) {
        for (std::size_t index = 0; index < directions.size(); ++index) {
            const double directional_bias = 1.0 - (static_cast<double>(index) * 0.05);
            plan.sweeps.push_back(CalibrationSweepStep{
                variable,
                directions[index],
                plan.minimal_wavelength_step,
                clamp_unit(plan.zero_point_overlap_score * directional_bias),
                clamp_unit(alignment_score * directional_bias),
            });
        }
    }
    return plan;
}

std::string FieldDynamicsEngine::make_trace_id(const GpuFeedbackFrame& frame) const {
    if (!frame.photonic_identity.trace_id.empty()) {
        return frame.photonic_identity.trace_id;
    }
    std::ostringstream out;
    out << "pid-" << frame.photonic_identity.gpu_device_id << '-'
        << frame.timing.tick_index << '-'
        << std::fixed << std::setprecision(3)
        << frame.photonic_identity.field_vector.frequency << '-'
        << frame.photonic_identity.field_vector.amplitude;
    return out.str();
}

SubstrateTrace FieldDynamicsEngine::trace_feedback(const GpuFeedbackFrame& frame) const {
    SubstrateTrace trace;
    trace.photonic_identity = frame.photonic_identity;
    trace.timing = frame.timing;
    trace.encodable_node_count = frame.encodable_node_count;
    trace.derived_constants = derive_temporal_constants(frame);
    trace.photonic_identity.trace_id = make_trace_id(frame);
    trace.photonic_identity.observed_latency_ms = std::max(0.0, frame.timing.response_time_ms - frame.timing.request_time_ms);
    if (trace.photonic_identity.source_identity.empty()) {
        trace.photonic_identity.source_identity = trace.photonic_identity.trace_id;
    }
    if (!has_nonzero_spectra(trace.photonic_identity.spectra_9d)) {
        trace.photonic_identity.spectra_9d = {
            trace.derived_constants.axis_scale_x,
            trace.derived_constants.axis_scale_y,
            trace.derived_constants.axis_scale_z,
            trace.derived_constants.vector_energy,
            trace.derived_constants.temporal_relativity_norm,
            trace.derived_constants.resonance_intercept_force,
            trace.derived_constants.crosstalk_weight,
            compute_phase_transport(frame),
            compute_flux_transport(frame),
        };
    }
    trace.trajectory_9d = compute_trajectory_9d(frame);
    trace.rotational_velocity = compute_rotational_velocity(frame);
    trace.encoded_pulse = encode_pulse_vector(frame);
    trace.phase_transport = compute_phase_transport(frame);
    trace.flux_transport = compute_flux_transport(frame);
    trace.observer_factor = compute_observer_factor(frame);
    trace.coupling_strength = compute_coupling_strength(frame);
    trace.coupling_collision_noise = compute_coupling_collision_noise(frame);
    trace.temporal_dynamics_noise = compute_temporal_dynamics_noise(frame);
    trace.reverse_causal_flux_coherence = compute_reverse_causal_flux_coherence(frame);
    trace.zero_point_overlap_score = compute_zero_point_overlap_score(frame);
    trace.constant_phase_alignment = compute_constant_phase_alignment(frame);
    trace.trajectory_conservation_score = compute_trajectory_conservation_score(frame);
    trace.expansion_factor = compute_expansion_factor(frame);
    trace.substrate_inertia = compute_substrate_inertia(frame);
    trace.transit_budget_ms = std::max(0.0, frame.timing.encode_deadline_ms - trace.photonic_identity.observed_latency_ms);
    trace.calibration_plan = build_calibration_plan(frame);
    trace.status = trace.transit_budget_ms >= 0.0 ? "ready" : "late";
    return trace;
}

}  // namespace qbit_miner
