#include "gguf_inspector.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace engine {

namespace {

// GGUF types
enum class gguf_type : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12
};

class BinReader {
public:
    explicit BinReader(const std::string& path)
        : in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("GGUF: cannot open file: " + path);
    }

    template <typename T>
    T read_le() {
        T v{};
        in_.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!in_) throw std::runtime_error("GGUF: unexpected EOF");
        return v;
    }

    std::string read_string() {
        uint64_t n = read_le<uint64_t>();
        std::string s(n, '\0');
        in_.read(s.data(), n);
        if (!in_) throw std::runtime_error("GGUF: unexpected EOF");
        return s;
    }

    void skip(uint64_t n) {
        in_.seekg(n, std::ios::cur);
    }

private:
    std::ifstream in_;
};

static std::optional<std::string> map_file_type(int64_t ft) {
    switch (ft) {
        case 4:  return "Q8_0";
        case 12: return "Q4_K_M";
        case 15: return "Q6_K";
        default: return std::nullopt;
    }
}

static int64_t read_int(BinReader& r, gguf_type t) {
    switch (t) {
        case gguf_type::INT8:   return r.read_le<int8_t>();
        case gguf_type::INT16:  return r.read_le<int16_t>();
        case gguf_type::INT32:  return r.read_le<int32_t>();
        case gguf_type::INT64:  return r.read_le<int64_t>();
        case gguf_type::UINT8:  return r.read_le<uint8_t>();
        case gguf_type::UINT16: return r.read_le<uint16_t>();
        case gguf_type::UINT32: return r.read_le<uint32_t>();
        case gguf_type::UINT64: return r.read_le<uint64_t>();
        default:
            throw std::runtime_error("GGUF: expected integer type");
    }
}

static void skip_value(BinReader& r, gguf_type t) {
    switch (t) {
        case gguf_type::STRING: r.read_string(); break;
        case gguf_type::ARRAY: {
            auto subtype = static_cast<gguf_type>(r.read_le<uint32_t>());
            uint64_t n = r.read_le<uint64_t>();
            for (uint64_t i = 0; i < n; ++i) skip_value(r, subtype);
            break;
        }
        case gguf_type::UINT8:
        case gguf_type::INT8:   r.skip(1); break;
        case gguf_type::UINT16:
        case gguf_type::INT16:  r.skip(2); break;
        case gguf_type::UINT32:
        case gguf_type::INT32:
        case gguf_type::FLOAT32: r.skip(4); break;
        case gguf_type::UINT64:
        case gguf_type::INT64:
        case gguf_type::FLOAT64: r.skip(8); break;
        case gguf_type::BOOL: r.skip(1); break;
    }
}

} // namespace

GgufInfo GgufInspector::inspect_metadata(const std::string& path) {
    BinReader r(path);

    const uint32_t magic = r.read_le<uint32_t>();
    if (magic != 0x46554747) { // 'GGUF' little-endian
        throw std::runtime_error("GGUF: invalid magic");
}

    GgufInfo info;

    info.version = r.read_le<uint32_t>();
    info.tensor_count = r.read_le<uint64_t>();
    info.kv_count = r.read_le<uint64_t>();

    for (uint64_t i = 0; i < info.kv_count; ++i) {
        std::string key = r.read_string();
        auto type = static_cast<gguf_type>(r.read_le<uint32_t>());

        if (key == "general.file_type") {
            auto v = read_int(r, type);
            info.general_file_type = v;
            info.detected_quant = map_file_type(v);
            continue;
        }

        if (key == "general.architecture" && type == gguf_type::STRING) {
            info.general_arch = r.read_string();
            continue;
        }

        if ((key == "context_length" || key == "n_ctx_train")
            && (type == gguf_type::UINT32 || type == gguf_type::INT32)) {
            info.context_length = static_cast<uint32_t>(read_int(r, type));
            continue;
        }

        skip_value(r, type);
    }

    return info;
}

GgufCapabilities GgufInspector::inspect_capabilities(
    const std::string& gguf_path
) {
    auto info = inspect_metadata(gguf_path);

    if (!info.detected_quant.has_value()) {
        throw std::runtime_error("GGUF: quantization not detected");
    }

    if (!info.general_arch.has_value()) {
        throw std::runtime_error("GGUF: architecture not detected");
    }

    if (!info.context_length.has_value()) {
        throw std::runtime_error("GGUF: context length not detected");
    }

    return {
        .quant   = *info.detected_quant,
        .arch    = *info.general_arch,
        .context = *info.context_length
    };
}

std::string GgufInspector::validate_or_resolve_quant(
    const std::string& gguf_path,
    const std::string& expected_quant,
    GgufMismatchPolicy policy
) {
    auto caps = inspect_capabilities(gguf_path);

    if (caps.quant == expected_quant) {
        return expected_quant;
    }

    std::string msg =
        "[gguf] quant mismatch: expected=" + expected_quant +
        " detected=" + caps.quant;

    if (policy == GgufMismatchPolicy::Warning) {
        std::cerr << msg << "\n";
        return expected_quant;
    }

    if (policy == GgufMismatchPolicy::Fallback) {
        std::cerr << msg << " (fallback)\n";
        return caps.quant;
    }

    throw std::runtime_error(msg);
}

} // namespace engine
