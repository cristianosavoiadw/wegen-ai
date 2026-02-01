#pragma once

#include <string>

namespace core {
    struct ExecutionPlan;
}

namespace engine {

class Engine {
public:
    void run(const std::string& model_path, const core::ExecutionPlan& plan);
};

} // namespace engine