#pragma once

#include <cstdint>
#include <string>

namespace model {

enum class QuantizationType {
    UNKNOWN = 0,
    Q8_0,
    Q6_K,
    Q4_K_M,
};

struct ModelDescriptor {
    // formato / identidade
    std::string format;          // "gguf"
    std::string model_name;      // opcional (derivado do filename)

    // capacidades estruturais
    uint32_t context_length;
    uint32_t embedding_size;
    uint32_t num_layers;

    // artefato
    QuantizationType quantization;
    size_t file_size_bytes;

    // invariantes
    bool supports_streaming = true;
};

} // namespace model
