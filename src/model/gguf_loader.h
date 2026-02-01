#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

/* -----------------------------
 * Tipos GGML
 * ----------------------------- */
    enum class GgmlType : uint32_t {
        F32     = 0,
        F16     = 1,
        Q4_0    = 2,
        Q4_1    = 3,
        // 4 e 5 reservados
        Q5_0    = 6,
        Q5_1    = 7,
        Q8_0    = 8,
        Q8_1    = 9,
        Q2_K    = 10,
        Q3_K    = 11,
        Q4_K    = 12,   // ← ADICIONAR
        Q5_K    = 13,   // ← ADICIONAR
        Q6_K    = 14,   // ← ADICIONAR (esse é o tipo 14!)
        Q8_K    = 15,   // ← ADICIONAR
        IQ2_XXS = 16,
        IQ2_XS  = 17,
        // ... outros
    };

/* -----------------------------
 * Tensor metadata
 * ----------------------------- */
struct GgufTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    std::vector<uint64_t> dims;
    GgmlType type{};
    uint64_t offset = 0;

    uint64_t numel() const;
};

/* -----------------------------
 * Modelo GGUF carregado
 * ----------------------------- */
class GgufModel {
public:
    uint32_t context_length() const { return context_length_; }
    uint32_t embedding_dim()  const { return embedding_dim_; }
    uint32_t n_layers()       const { return n_layers_; }

    // Acesso a tensores
    const void* tensor_ptr(const std::string& name) const;

    // NOVO: Obter tipo do tensor
    GgmlType tensor_type(const std::string& name) const;

    // NOVO: Obter informação completa do tensor
    const GgufTensorInfo* tensor_info(const std::string& name) const;

    std::string summary() const;

private:
    friend class GgufLoader;

    uint32_t context_length_ = 0;
    uint32_t embedding_dim_  = 0;
    uint32_t n_layers_       = 0;

    std::unordered_map<std::string, std::string> kv_;
    std::unordered_map<std::string, GgufTensorInfo> tensors_;

    void* file_base_ = nullptr;
    size_t file_size_ = 0;

    const uint8_t* data_base_ = nullptr;
    uint64_t data_offset_ = 0;
};

/* -----------------------------
 * Loader
 * ----------------------------- */
class GgufLoader {
public:
    static GgufModel load(const std::string& path);

private:
    static void validate_magic(const char magic[4]);
};

} // namespace engine
