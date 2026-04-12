#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "qbit_miner/substrate/field_dynamics.hpp"

namespace qbit_miner {

struct RuntimeEvent {
    std::string topic;
    std::string message;
    SubstrateTrace trace;
};

class RuntimeBus {
public:
    using Subscriber = std::function<void(const RuntimeEvent&)>;

    void subscribe(const std::string& topic, Subscriber subscriber);
    void publish(const RuntimeEvent& event) const;

private:
    std::map<std::string, std::vector<Subscriber>> subscribers_;
};

}  // namespace qbit_miner