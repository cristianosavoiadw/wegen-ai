#pragma once

#include <cstdint>
#include <string>

namespace engine {
    enum class QuantizationType;
}

namespace core {

enum class QuantizationPolicy {
    USE_MODEL_NATIVE,
    REQUIRE_EXACT
};

struct PowerLimits {
    double max_watts = 0.0;
};

struct ExecutionPlan {
    std::string backend;
    QuantizationPolicy quant_policy;
    engine::QuantizationType quantization;

    uint32_t max_tokens;
    PowerLimits power;

    std::string scheduler_policy;
    bool streaming = true;
};

} // namespace core