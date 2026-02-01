#include "./backend_factory.h"
#include "./backend/backend.h"

#include "../core/execution_plan.h"

#include <stdexcept>

#include "cpu/cpu_backend.h"

namespace engine {

std::unique_ptr<Backend> BackendFactory::create(const core::ExecutionPlan& plan) {
    if (plan.backend == "cpu") {
        return std::make_unique<CpuBackend>();
    }

    throw std::runtime_error("Unknown backend: " + plan.backend);
}

} // namespace engine