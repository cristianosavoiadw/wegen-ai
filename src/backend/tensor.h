#pragma once

#include <cstddef>
#include <vector>

namespace engine {

struct TensorView {
    void* data = nullptr;
    std::vector<size_t> shape{};
};

} // namespace engine
