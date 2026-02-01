#pragma once

#include <string>
#include <stdexcept>

namespace engine {

enum class QuantizationType {
    UNKNOWN = 0,
    Q8_0,
    Q6_K,
    Q4_K_M
};

std::string quant_to_string(QuantizationType q);
QuantizationType quant_from_string(const std::string& s);

} // namespace engine