#pragma once

#include <string>

#include "qbit_miner/cache/substrate_cache.hpp"
#include "qbit_miner/runtime/runtime_bus.hpp"

namespace qbit_miner {

class QuantumMinerApplication {
public:
    explicit QuantumMinerApplication(FieldDynamicsConfig config = {});

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] SubstrateTrace process_feedback(const GpuFeedbackFrame& frame);
    [[nodiscard]] const SubstrateCache& cache() const noexcept;
    RuntimeBus& bus() noexcept;

private:
    std::string name_;
    FieldDynamicsEngine dynamics_;
    SubstrateCache cache_;
    RuntimeBus bus_;
};

}  // namespace qbit_miner