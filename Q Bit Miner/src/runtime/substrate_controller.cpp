#include "qbit_miner/runtime/substrate_controller.hpp"

#include <chrono>
#include <exception>
#include <thread>
#include <utility>

namespace qbit_miner {

namespace {

SubstrateTrace make_failure_trace(const GpuFeedbackFrame& frame) {
    SubstrateTrace trace;
    trace.photonic_identity = frame.photonic_identity;
    trace.timing = frame.timing;
    trace.encodable_node_count = frame.encodable_node_count;
    trace.status = "failed";
    return trace;
}

}  // namespace

SubstrateController::SubstrateController(FieldDynamicsConfig field_config, SubstrateControllerConfig controller_config)
    : application_(std::move(field_config)),
      config_(controller_config) {}

const SubstrateControllerConfig& SubstrateController::config() const noexcept {
    return config_;
}

QuantumMinerApplication& SubstrateController::application() noexcept {
    return application_;
}

const QuantumMinerApplication& SubstrateController::application() const noexcept {
    return application_;
}

RuntimeBus& SubstrateController::bus() noexcept {
    return application_.bus();
}

const SubstrateCache& SubstrateController::cache() const noexcept {
    return application_.cache();
}

SubstrateTrace SubstrateController::process_feedback(const GpuFeedbackFrame& frame) {
    SubstrateTrace trace = application_.process_feedback(frame);
    publish_trace_topics(trace);
    return trace;
}

SubstrateRunSummary SubstrateController::run_replay(const std::vector<GpuFeedbackFrame>& frames, std::size_t runtime_ticks) {
    SubstrateRunSummary summary;
    if (frames.empty()) {
        return summary;
    }

    const std::size_t requested_ticks = runtime_ticks == 0 ? config_.runtime_ticks : runtime_ticks;
    const std::size_t total_ticks = requested_ticks == 0 ? frames.size() : requested_ticks;
    summary.requested_ticks = total_ticks;
    summary.traces.reserve(total_ticks);

    for (std::size_t tick_index = 0; tick_index < total_ticks; ++tick_index) {
        const GpuFeedbackFrame& frame = frames[tick_index % frames.size()];
        try {
            summary.traces.push_back(process_feedback(frame));
            ++summary.processed_ticks;
        } catch (const std::exception& error) {
            publish_failure(frame, error.what());
            ++summary.failed_ticks;
        } catch (...) {
            publish_failure(frame, "Unknown controller failure");
            ++summary.failed_ticks;
        }
        sleep_if_needed(tick_index, total_ticks);
    }

    return summary;
}

void SubstrateController::publish_trace_topics(const SubstrateTrace& trace) {
    if (config_.emit_gui_trace_refresh) {
        bus().publish(RuntimeEvent{
            "gui.trace.refresh",
            "Substrate trace ready for GUI consumers",
            trace,
        });
    }

    if (config_.emit_network_trace_publish) {
        bus().publish(RuntimeEvent{
            "network.trace.publish",
            "Substrate trace ready for network publication",
            trace,
        });
    }
}

void SubstrateController::publish_failure(const GpuFeedbackFrame& frame, const std::string& message) {
    bus().publish(RuntimeEvent{
        "substrate.trace.failed",
        message,
        make_failure_trace(frame),
    });
}

void SubstrateController::sleep_if_needed(std::size_t tick_index, std::size_t total_ticks) const {
    if (config_.tick_interval_ms == 0 || (tick_index + 1) >= total_ticks) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(config_.tick_interval_ms));
}

}  // namespace qbit_miner