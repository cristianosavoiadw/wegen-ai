#pragma once

#include <string>
#include <cstdint>
#include "core/context.h"
#include "tensor.h"

namespace engine {

struct ModelInfo {
    uint32_t context_length = 0;
    uint32_t embedding_dim = 0;
    uint32_t vocab_size = 0;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual void init() = 0;
    virtual ModelInfo load_model(const std::string& model_path) = 0;
    virtual void forward(const TensorView&, TensorView&) = 0;
    virtual BackendStats stats() const = 0;
};

} // namespace engine