#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

/* -----------------------------
 * Tipos GGML (compatível GGUF)
 * ----------------------------- */
enum class GgmlType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
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
    /* --- metadata principal --- */
    uint32_t context_length() const { return context_length_; }
    uint32_t embedding_dim()  const { return embedding_dim_; }
    uint32_t n_layers()       const { return n_layers_; }
    uint32_t vocab_size()     const { return vocab_size_; }
    uint32_t n_heads()        const { return n_heads_; }
    uint32_t n_kv_heads()     const { return n_kv_heads_; }

    /* --- acesso a tensores --- */
    const void* tensor_ptr(const std::string& name) const;
    GgmlType tensor_type(const std::string& name) const;
    const GgufTensorInfo* tensor_info(const std::string& name) const;

    std::string summary() const;

    /* --- tokenizer (do GGUF) --- */
    const std::vector<std::string>& tokenizer_tokens() const { return tokenizer_tokens_; }
    const std::vector<float>& tokenizer_scores() const { return tokenizer_scores_; }
    const std::vector<int32_t>& tokenizer_types() const { return tokenizer_types_; }

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t unk_id() const { return unk_id_; }

private:
    friend class GgufLoader;

    /* --- metadata --- */
    uint32_t context_length_ = 0;
    uint32_t embedding_dim_  = 0;
    uint32_t n_layers_       = 0;
    uint32_t vocab_size_     = 0;
    uint32_t n_heads_        = 0;
    uint32_t n_kv_heads_     = 0;

    /* --- tokenizer --- */
    std::vector<std::string> tokenizer_tokens_;
    std::vector<float> tokenizer_scores_;
    std::vector<int32_t> tokenizer_types_;

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;

    /* --- tensors --- */
    std::unordered_map<std::string, GgufTensorInfo> tensors_;

    /* --- file mapping --- */
    void* file_base_ = nullptr;
    size_t file_size_ = 0;
    uint64_t data_offset_ = 0;
    const uint8_t* data_base_ = nullptr;
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