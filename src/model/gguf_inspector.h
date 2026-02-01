#pragma once

#include <string>
#include <optional>
#include <cstdint>

namespace engine {

enum class GgufMismatchPolicy {
    Error,
    Warning,
    Fallback
};

struct GgufInfo {
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;

    std::optional<int64_t> general_file_type;
    std::optional<std::string> detected_quant;
    std::optional<std::string> general_arch;
    std::optional<uint32_t> context_length;
};

struct GgufCapabilities {
    std::string quant;
    std::string arch;
    uint32_t context;
};

class GgufInspector {
public:
    static GgufInfo inspect_metadata(const std::string& path);

    static GgufCapabilities inspect_capabilities(const std::string& gguf_path);

    static std::string validate_or_resolve_quant(
        const std::string& gguf_path,
        const std::string& expected_quant,
        GgufMismatchPolicy policy
    );
};

} // namespace engine