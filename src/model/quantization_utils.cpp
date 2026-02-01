#include "quantization_utils.h"

namespace engine {

std::string quant_to_string(QuantizationType q) {
    switch (q) {
        case QuantizationType::Q8_0:   return "Q8_0";
        case QuantizationType::Q6_K:   return "Q6_K";
        case QuantizationType::Q4_K_M: return "Q4_K_M";
        default:
            return "UNKNOWN";
    }
}

QuantizationType quant_from_string(const std::string& s) {
    if (s == "Q8_0")   return QuantizationType::Q8_0;
    if (s == "Q6_K")   return QuantizationType::Q6_K;
    if (s == "Q4_K_M") return QuantizationType::Q4_K_M;

    throw std::invalid_argument(
        "quant_from_string: unsupported quantization: " + s
    );
}

} // namespace engine
