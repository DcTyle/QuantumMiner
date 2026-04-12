#include "qbit_miner/app/application.hpp"

namespace qbit_miner {

QuantumMinerApplication::QuantumMinerApplication(FieldDynamicsConfig config)
    : name_("Quantum Miner"),
      dynamics_(config),
      cache_(512) {}

const std::string& QuantumMinerApplication::name() const noexcept {
    return name_;
}

SubstrateTrace QuantumMinerApplication::process_feedback(const GpuFeedbackFrame& frame) {
    SubstrateTrace trace = dynamics_.trace_feedback(frame);
    cache_.store(trace);
    bus_.publish(RuntimeEvent{
        "substrate.trace.ready",
        "Substrate trace cached and ready for GUI/network hooks",
        trace,
    });
    return trace;
}

const SubstrateCache& QuantumMinerApplication::cache() const noexcept {
    return cache_;
}

RuntimeBus& QuantumMinerApplication::bus() noexcept {
    return bus_;
}

}  // namespace qbit_miner