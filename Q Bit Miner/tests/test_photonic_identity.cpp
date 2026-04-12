#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "qbit_miner/app/application.hpp"
#include "qbit_miner/runtime/substrate_controller.hpp"
#include "qbit_miner/substrate/calibration_export.hpp"
#include "qbit_miner/substrate/device_validation_export.hpp"
#include "qbit_miner/substrate/research_calibration_importer.hpp"
#include "qbit_miner/substrate/trace_serialization.hpp"

namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

int main() {
    qbit_miner::QuantumMinerApplication app;

    qbit_miner::GpuFeedbackFrame frame;
    frame.photonic_identity.gpu_device_id = "gpu-test";
    frame.photonic_identity.coherence = 0.93;
    frame.photonic_identity.memory = 0.88;
    frame.photonic_identity.nexus = 0.51;
    frame.photonic_identity.field_vector.amplitude = 0.20;
    frame.photonic_identity.field_vector.voltage = 0.40;
    frame.photonic_identity.field_vector.current = 0.31;
    frame.photonic_identity.field_vector.frequency = 0.27;
    frame.photonic_identity.field_vector.phase = 0.18;
    frame.photonic_identity.field_vector.flux = 0.24;
    frame.photonic_identity.field_vector.thermal_noise = 0.05;
    frame.photonic_identity.field_vector.field_noise = 0.03;
    frame.photonic_identity.spin_inertia.axis_spin = {0.12, 0.09, -0.08};
    frame.photonic_identity.spin_inertia.axis_orientation = {0.10, 0.07, -0.05};
    frame.photonic_identity.spin_inertia.momentum_score = 0.22;
    frame.photonic_identity.spin_inertia.inertial_mass_proxy = 0.18;
    frame.photonic_identity.spin_inertia.relativistic_correlation = 0.07;
    frame.photonic_identity.spin_inertia.relative_temporal_coupling = 0.66;
    frame.photonic_identity.spin_inertia.temporal_coupling_count = 5;
    frame.timing.tick_index = 42;
    frame.timing.request_time_ms = 1.0;
    frame.timing.response_time_ms = 2.7;
    frame.timing.encode_deadline_ms = 4.5;
    frame.integrated_feedback = 0.44;
    frame.derivative_signal = 0.03;
    frame.lattice_closure = 0.91;
    frame.phase_closure = 0.89;
    frame.recurrence_alignment = 0.83;
    frame.conservation_alignment = 0.998;

    const qbit_miner::SubstrateTrace trace = app.process_feedback(frame);
    assert(!trace.photonic_identity.trace_id.empty());
    assert(trace.derived_constants.temporal_relativity >= 1.0);
    assert(trace.derived_constants.phase_alignment_probability > 0.0);
    assert(trace.derived_constants.zero_point_proximity >= 0.0);
    assert(trace.derived_constants.effective_wavelength_step > 0.0);
    assert(trace.substrate_inertia > 0.0);
    assert(trace.coupling_strength > 0.0);
    assert(trace.coupling_collision_noise > 0.0);
    assert(trace.temporal_dynamics_noise >= 0.0);
    assert(trace.reverse_causal_flux_coherence >= 0.0);
    assert(trace.zero_point_overlap_score >= 0.0);
    assert(trace.constant_phase_alignment >= 0.0);
    assert(trace.trajectory_conservation_score >= 0.0);
    assert(trace.expansion_factor >= 1.0);
    assert(trace.observer_factor > 0.0);
    assert(trace.trajectory_9d.size() == 9U);
    assert(std::fabs(trace.rotational_velocity[0]) > 0.0);
    assert(std::fabs(trace.encoded_pulse[0]) > 0.0);
    assert(trace.calibration_plan.sweeps.size() == 16U);
    assert(trace.calibration_plan.sweeps.front().direction == "left_to_right");
    assert(trace.calibration_plan.sweeps[1].direction == "right_to_left");
    assert(trace.calibration_plan.sweeps[2].direction == "top_to_bottom");
    assert(trace.calibration_plan.sweeps[3].direction == "bottom_to_top");
    assert(app.cache().size() == 1U);
    assert(app.cache().find_by_trace_id(trace.photonic_identity.trace_id).has_value());

    const std::string trace_json = qbit_miner::serialize_trace_json(trace);
    assert(trace_json.find("\"trace_id\"") != std::string::npos);
    assert(trace_json.find("\"coupling_strength\"") != std::string::npos);
    assert(trace_json.find("\"derived_constants\"") != std::string::npos);
    assert(trace_json.find("\"phase_alignment_probability\"") != std::string::npos);
    assert(trace_json.find("\"rotational_velocity\"") != std::string::npos);
    assert(trace_json.find("\"trajectory_9d\"") != std::string::npos);
    assert(trace_json.find("\"calibration_plan\"") != std::string::npos);
    assert(trace_json.find("\"left_to_right\"") != std::string::npos);

    const std::filesystem::path sample_csv = std::filesystem::path(__FILE__).parent_path() / "data" / "run45_import_sample.csv";
    qbit_miner::ResearchCalibrationImporter importer;
    const auto imported_frames = importer.import_run45_csv(sample_csv, "gpu-run45");
    assert(imported_frames.size() == 2U);
    assert(imported_frames.front().photonic_identity.trace_id.find("PID9-") == 0U);
    assert(imported_frames.front().timing.closed_loop_latency_ms > imported_frames.front().timing.response_time_ms);
    assert(imported_frames.front().encodable_node_count == 4U);

    const qbit_miner::SubstrateTrace imported_trace = app.process_feedback(imported_frames.front());
    const std::string imported_json = qbit_miner::serialize_trace_json(imported_trace);
    assert(imported_json.find("PID9-") != std::string::npos);
    assert(imported_json.find("\"encodable_node_count\": 4") != std::string::npos);
    assert(imported_json.find("\"source_identity\"") != std::string::npos);
    assert(imported_json.find("\"trajectory_conservation_score\"") != std::string::npos);

    const qbit_miner::SubstrateTrace imported_trace_2 = app.process_feedback(imported_frames.back());
    const std::filesystem::path export_dir = std::filesystem::current_path() / "qbit_miner_test_export";
    std::filesystem::remove_all(export_dir);

    const qbit_miner::CalibrationExportResult export_result = qbit_miner::export_calibration_bundle(
        std::vector<qbit_miner::SubstrateTrace>{imported_trace, imported_trace_2},
        export_dir
    );

    assert(export_result.trace_count == 2U);
    assert(export_result.sweep_file_count == 32U);
    assert(std::filesystem::exists(export_result.manifest_path));
    assert(std::filesystem::exists(export_result.traces_jsonl_path));

    const std::string manifest_text = read_text(export_result.manifest_path);
    assert(manifest_text.find("quantum_miner.calibration_bundle.v1") != std::string::npos);
    assert(manifest_text.find(imported_trace.photonic_identity.trace_id) != std::string::npos);

    const std::filesystem::path trace_dir = export_dir / imported_trace.photonic_identity.trace_id;
    const std::filesystem::path sweep_file = trace_dir / "sweeps" / "00_frequency_left_to_right.json";
    assert(std::filesystem::exists(trace_dir / "trace.json"));
    assert(std::filesystem::exists(trace_dir / "calibration_plan.json"));
    assert(std::filesystem::exists(sweep_file));

    const std::string plan_text = read_text(trace_dir / "calibration_plan.json");
    assert(plan_text.find("\"calibration_plan\"") != std::string::npos);
    assert(plan_text.find("\"left_to_right\"") != std::string::npos);

    const std::string sweep_text = read_text(sweep_file);
    assert(sweep_text.find("\"sweep_index\": 0") != std::string::npos);
    assert(sweep_text.find("\"variable\": \"frequency\"") != std::string::npos);
    assert(sweep_text.find("\"direction\": \"left_to_right\"") != std::string::npos);

    const std::filesystem::path device_export_dir = std::filesystem::current_path() / "qbit_miner_device_validation_export";
    std::filesystem::remove_all(device_export_dir);

    qbit_miner::DeviceValidationExportOptions validation_options;
    validation_options.hardware_profile.operator_id = "test-operator";
    validation_options.hardware_profile.device_model = "test-substrate-rig";
    validation_options.hardware_profile.driver_version = "driver-test";
    validation_options.hardware_profile.power_draw_watts = 210.0;
    validation_options.hardware_profile.electricity_cost_usd_per_kwh = 0.15;
    validation_options.estimated_reward_per_share_usd = 0.05;

    const qbit_miner::DeviceValidationExportResult device_export_result = qbit_miner::export_device_validation_bundle(
        std::vector<qbit_miner::SubstrateTrace>{imported_trace, imported_trace_2},
        device_export_dir,
        validation_options
    );

    assert(device_export_result.trace_count == 2U);
    assert(std::filesystem::exists(device_export_result.hardware_profile_path));
    assert(std::filesystem::exists(device_export_result.profit_window_path));
    assert(std::filesystem::exists(device_export_result.accepted_share_log_path));
    assert(std::filesystem::exists(device_export_result.power_telemetry_path));
    assert(std::filesystem::exists(device_export_result.substrate_state_snapshot_path));
    assert(std::filesystem::exists(device_export_result.phase_vector_ledger_path));

    const std::string hardware_profile_text = read_text(device_export_result.hardware_profile_path);
    assert(hardware_profile_text.find("\"hardware_profile_id\"") != std::string::npos);
    assert(hardware_profile_text.find("\"device_model\": \"test-substrate-rig\"") != std::string::npos);

    const std::string profit_window_text = read_text(device_export_result.profit_window_path);
    assert(profit_window_text.find("\"net_profit_usd\"") != std::string::npos);
    assert(profit_window_text.find("\"accepted_share_count\"") != std::string::npos);
    assert(profit_window_text.find("\"completion_gate_passed\": false") != std::string::npos);

    const std::string accepted_share_text = read_text(device_export_result.accepted_share_log_path);
    assert(accepted_share_text.find("\"accepted_share_score\"") != std::string::npos);
    assert(accepted_share_text.find(imported_trace.photonic_identity.trace_id) != std::string::npos);

    const std::string power_telemetry_text = read_text(device_export_result.power_telemetry_path);
    assert(power_telemetry_text.find("\"power_draw_watts\"") != std::string::npos);
    assert(power_telemetry_text.find("\"power_cost_usd\"") != std::string::npos);

    const std::string substrate_state_text = read_text(device_export_result.substrate_state_snapshot_path);
    assert(substrate_state_text.find("\"mean_trajectory_9d\"") != std::string::npos);
    assert(substrate_state_text.find("\"last_trace_id\"") != std::string::npos);

    const std::string phase_vector_ledger_text = read_text(device_export_result.phase_vector_ledger_path);
    assert(phase_vector_ledger_text.find("\"trajectory_9d\"") != std::string::npos);
    assert(phase_vector_ledger_text.find("\"encoded_pulse\"") != std::string::npos);

    qbit_miner::SubstrateControllerConfig controller_config;
    controller_config.runtime_ticks = 3U;
    qbit_miner::SubstrateController controller({}, controller_config);
    std::vector<std::string> topics;
    controller.bus().subscribe("*", [&topics](const qbit_miner::RuntimeEvent& event) {
        topics.push_back(event.topic);
    });

    const qbit_miner::SubstrateRunSummary replay_summary = controller.run_replay(
        std::vector<qbit_miner::GpuFeedbackFrame>{frame, imported_frames.front()},
        3U
    );

    assert(replay_summary.requested_ticks == 3U);
    assert(replay_summary.processed_ticks == 3U);
    assert(replay_summary.failed_ticks == 0U);
    assert(replay_summary.traces.size() == 3U);
    assert(controller.cache().size() == 3U);
    assert(std::count(topics.begin(), topics.end(), "substrate.trace.ready") == 3);
    assert(std::count(topics.begin(), topics.end(), "gui.trace.refresh") == 3);
    assert(std::count(topics.begin(), topics.end(), "network.trace.publish") == 3);
    assert(!replay_summary.traces.front().photonic_identity.trace_id.empty());
    assert(replay_summary.traces[1].photonic_identity.trace_id.find("PID9-") == 0U);

    std::filesystem::remove_all(device_export_dir);
    std::filesystem::remove_all(export_dir);
    return 0;
}