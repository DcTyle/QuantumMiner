#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace qbit_miner {

struct FieldVector {
    double amplitude = 0.0;
    double voltage = 0.0;
    double current = 0.0;
    double frequency = 0.0;
    double phase = 0.0;
    double flux = 0.0;
    double thermal_noise = 0.0;
    double field_noise = 0.0;
};

struct SpinInertia {
    std::array<double, 3> axis_spin {0.0, 0.0, 0.0};
    std::array<double, 3> axis_orientation {0.0, 0.0, 0.0};
    double momentum_score = 0.0;
    double inertial_mass_proxy = 0.0;
    double relativistic_correlation = 0.0;
    double relative_temporal_coupling = 0.0;
    std::uint32_t temporal_coupling_count = 0;
};

struct PhotonicIdentity {
    std::string trace_id;
    std::string source_identity;
    std::string gpu_device_id;
    std::array<double, 9> spectra_9d {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    FieldVector field_vector;
    SpinInertia spin_inertia;
    double coherence = 0.0;
    double memory = 0.0;
    double nexus = 0.0;
    double observed_latency_ms = 0.0;
};

struct SubstrateRequestContext {
    std::uint64_t tick_index = 0;
    double request_time_ms = 0.0;
    double response_time_ms = 0.0;
    double accounting_time_ms = 0.0;
    double next_feedback_time_ms = 0.0;
    double closed_loop_latency_ms = 0.0;
    double encode_deadline_ms = 0.0;
};

struct GpuFeedbackFrame {
    PhotonicIdentity photonic_identity;
    SubstrateRequestContext timing;
    std::uint32_t encodable_node_count = 0;
    double sent_signal = 0.0;
    double measured_signal = 0.0;
    double integrated_feedback = 0.0;
    double derivative_signal = 0.0;
    double lattice_closure = 0.0;
    double phase_closure = 0.0;
    double recurrence_alignment = 0.0;
    double conservation_alignment = 0.0;
};

}  // namespace qbit_miner