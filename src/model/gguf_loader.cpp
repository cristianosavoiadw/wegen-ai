#include "gguf_loader.h"

#include <cstring>
#include <sstream>
#include <iostream>
#include <stdexcept>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace engine {

static constexpr uint32_t GGUF_VERSION_MIN = 2;
static constexpr uint32_t GGUF_VERSION_MAX = 3;

// GGUF alinha o início do data section tipicamente em 32 bytes
static constexpr uint64_t GGUF_ALIGNMENT = 32;

/* ------------------------------------------------ */
static uint64_t align_up(uint64_t x, uint64_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

static uint32_t read_u32(const uint8_t*& p, const uint8_t* end) {
    if (p + 4 > end) throw std::runtime_error("GGUF: truncated u32");
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}

static uint64_t read_u64(const uint8_t*& p, const uint8_t* end) {
    if (p + 8 > end) throw std::runtime_error("GGUF: truncated u64");
    uint64_t v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}

static std::string read_string(const uint8_t*& p, const uint8_t* end) {
    uint64_t len = read_u64(p, end);
    if (p + len > end) throw std::runtime_error("GGUF: truncated string");
    std::string s(reinterpret_cast<const char*>(p), len);
    p += len;
    return s;
}

/* ------------------------------------------------ */
enum class GgufValueType : uint32_t {
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
    FLOAT64 = 12,
};

static uint32_t read_value_as_u32(const uint8_t*& p, const uint8_t* end, GgufValueType t) {
    switch (t) {
        case GgufValueType::UINT32:
            return read_u32(p, end);

        case GgufValueType::INT32: {
            uint32_t raw = read_u32(p, end);
            int32_t v;
            std::memcpy(&v, &raw, 4);
            if (v < 0) throw std::runtime_error("GGUF: negative int32");
            return static_cast<uint32_t>(v);
        }

        case GgufValueType::UINT64: {
            uint64_t v = read_u64(p, end);
            if (v > 0xffffffffull) throw std::runtime_error("GGUF: uint64 overflow");
            return static_cast<uint32_t>(v);
        }

        case GgufValueType::INT64: {
            uint64_t raw = read_u64(p, end);
            int64_t v;
            std::memcpy(&v, &raw, 8);
            if (v < 0 || v > 0xffffffffll)
                throw std::runtime_error("GGUF: int64 out of range");
            return static_cast<uint32_t>(v);
        }

        default:
            throw std::runtime_error("GGUF: value not convertible to u32");
    }
}

/* ------------------------------------------------ */
// pula um valor GGUF qualquer (para KV que não interessa)
static void skip_value(const uint8_t*& p, const uint8_t* end, GgufValueType t);

static void skip_array(const uint8_t*& p, const uint8_t* end) {
    auto elem_type = static_cast<GgufValueType>(read_u32(p, end));
    uint64_t n = read_u64(p, end);
    for (uint64_t i = 0; i < n; ++i) {
        skip_value(p, end, elem_type);
    }
}

static void skip_value(const uint8_t*& p, const uint8_t* end, GgufValueType t) {
    switch (t) {
        case GgufValueType::UINT8:
        case GgufValueType::INT8:
        case GgufValueType::BOOL:
            if (p + 1 > end) throw std::runtime_error("GGUF: truncated skip 1 byte");
            p += 1;
            return;

        case GgufValueType::UINT16:
        case GgufValueType::INT16:
            if (p + 2 > end) throw std::runtime_error("GGUF: truncated skip 2 bytes");
            p += 2;
            return;

        case GgufValueType::UINT32:
        case GgufValueType::INT32:
        case GgufValueType::FLOAT32:
            if (p + 4 > end) throw std::runtime_error("GGUF: truncated skip 4 bytes");
            p += 4;
            return;

        case GgufValueType::UINT64:
        case GgufValueType::INT64:
        case GgufValueType::FLOAT64:
            if (p + 8 > end) throw std::runtime_error("GGUF: truncated skip 8 bytes");
            p += 8;
            return;

        case GgufValueType::STRING: {
            uint64_t len = read_u64(p, end);
            if (p + len > end) throw std::runtime_error("GGUF: truncated skip string");
            p += len;
            return;
        }

        case GgufValueType::ARRAY:
            skip_array(p, end);
            return;

        default:
            throw std::runtime_error("GGUF: unknown KV type");
    }
}

/* ------------------------------------------------ */
uint64_t GgufTensorInfo::numel() const {
    uint64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

/* ------------------------------------------------ */
const void* GgufModel::tensor_ptr(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    if (!data_base_) return nullptr;
    return reinterpret_cast<const void*>(data_base_ + it->second.offset);
}

std::string GgufModel::summary() const {
    std::ostringstream oss;
    oss << "GGUF model: ctx=" << context_length_
        << " emb=" << embedding_dim_
        << " layers=" << n_layers_
        << " kv=" << kv_.size()
        << " tensors=" << tensors_.size()
        << " file_size=" << file_size_;
    return oss.str();
}

    // Adicionar ao gguf_loader.cpp, após o método tensor_ptr():

    GgmlType GgufModel::tensor_type(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return GgmlType::F32;  // Default
    }
    return it->second.type;
}

    const GgufTensorInfo* GgufModel::tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

/* ------------------------------------------------ */
void GgufLoader::validate_magic(const char magic[4]) {
    if (!(magic[0] == 'G' && magic[1] == 'G' &&
          magic[2] == 'U' && magic[3] == 'F')) {
        throw std::runtime_error("GGUF: invalid magic");
    }
}

/* ------------------------------------------------ */
GgufModel GgufLoader::load(const std::string& path) {
#ifndef __linux__
    throw std::runtime_error("GGUF loader: mmap only on linux");
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("GGUF: cannot open file");

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("GGUF: fstat failed");
    }

    void* base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("GGUF: mmap failed");

    GgufModel model;
    model.file_base_ = base;
    model.file_size_ = static_cast<size_t>(st.st_size);

    const uint8_t* p   = reinterpret_cast<const uint8_t*>(base);
    const uint8_t* end = p + model.file_size_;

    char magic[4];
    std::memcpy(magic, p, 4);
    p += 4;
    validate_magic(magic);

    uint32_t version = read_u32(p, end);
    if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
        ::munmap(base, model.file_size_);
        throw std::runtime_error("GGUF: unsupported version");
    }

    uint64_t n_tensors = read_u64(p, end);
    uint64_t n_kv      = read_u64(p, end);

    /* ---- KV ---- */
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = read_string(p, end);
        auto vtype = static_cast<GgufValueType>(read_u32(p, end));

        if (key == "llama.context_length" || key == "n_ctx" || key == "context_length") {
            uint32_t v = read_value_as_u32(p, end, vtype);
            model.context_length_ = v;
            model.kv_.emplace(key, std::to_string(v));
            continue;
        }

        if (key == "llama.embedding_length" || key == "n_embd") {
            uint32_t v = read_value_as_u32(p, end, vtype);
            model.embedding_dim_ = v;
            model.kv_.emplace(key, std::to_string(v));
            continue;
        }

        if (key == "llama.block_count" || key == "n_layer") {
            uint32_t v = read_value_as_u32(p, end, vtype);
            model.n_layers_ = v;
            model.kv_.emplace(key, std::to_string(v));
            continue;
        }

        // Fase 2: qualquer outro KV -> pula com segurança
        skip_value(p, end, vtype);
        // opcional: guardar o fato de existir
        // model.kv_.emplace(std::move(key), "<skipped>");
    }

    /* ---- Tensors ---- */
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensorInfo info;
        info.name   = read_string(p, end);
        info.n_dims = read_u32(p, end);

        info.dims.resize(info.n_dims);
        for (uint32_t d = 0; d < info.n_dims; ++d) {
            info.dims[d] = read_u64(p, end);
        }

        info.type   = static_cast<GgmlType>(read_u32(p, end));
        info.offset = read_u64(p, end);

        model.tensors_.emplace(info.name, std::move(info));
    }

    uint64_t cur_off = static_cast<uint64_t>(p - reinterpret_cast<const uint8_t*>(base));
    model.data_offset_ = align_up(cur_off, GGUF_ALIGNMENT);
    model.data_base_   = reinterpret_cast<const uint8_t*>(base) + model.data_offset_;

    if (model.context_length_ == 0) {
        ::munmap(base, model.file_size_);
        throw std::runtime_error("GGUF: context length not detected");
    }

    return model;
#endif
}

} // namespace engine
