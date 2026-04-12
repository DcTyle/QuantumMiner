#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qbit_miner/app/application.hpp"

namespace qbit_miner {

struct SubstrateControllerConfig {
    std::size_t runtime_ticks = 0;
    std::uint32_t tick_interval_ms = 0;
    bool emit_gui_trace_refresh = true;
    bool emit_network_trace_publish = true;
};

struct SubstrateRunSummary {
    std::size_t requested_ticks = 0;
    std::size_t processed_ticks = 0;
    std::size_t failed_ticks = 0;
    std::vector<SubstrateTrace> traces;
};

class SubstrateController {
public:
    explicit SubstrateController(FieldDynamicsConfig field_config = {}, SubstrateControllerConfig controller_config = {});

    [[nodiscard]] const SubstrateControllerConfig& config() const noexcept;
    [[nodiscard]] QuantumMinerApplication& application() noexcept;
    [[nodiscard]] const QuantumMinerApplication& application() const noexcept;
    [[nodiscard]] RuntimeBus& bus() noexcept;
    [[nodiscard]] const SubstrateCache& cache() const noexcept;
    [[nodiscard]] SubstrateTrace process_feedback(const GpuFeedbackFrame& frame);
    [[nodiscard]] SubstrateRunSummary run_replay(const std::vector<GpuFeedbackFrame>& frames, std::size_t runtime_ticks = 0);

private:
    void publish_trace_topics(const SubstrateTrace& trace);
    void publish_failure(const GpuFeedbackFrame& frame, const std::string& message);
    void sleep_if_needed(std::size_t tick_index, std::size_t total_ticks) const;

    QuantumMinerApplication application_;
    SubstrateControllerConfig config_;
};

}  // namespace qbit_miner