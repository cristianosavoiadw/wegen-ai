#pragma once

#include <memory>

namespace core {
    struct ExecutionPlan;
}

namespace engine {

class Backend;

class BackendFactory {
public:
    static std::unique_ptr<Backend> create(const core::ExecutionPlan& plan);
};

} // namespace engine