#pragma once

#include "backend/backend.h"
#include "metrics/power_linux.h"
#include "model/gguf_loader.h"
#include "model/tokenizer.h"
#include "model/sampler.h"

#include <vector>
#include <memory>

namespace engine {

// ============================================================================
// Transformer Layer
// ============================================================================

struct TransformerLayer {
    // Attention
    const float* attn_norm_weight = nullptr;
    const float* wq = nullptr;
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* wo = nullptr;

    // FFN
    const float* ffn_norm_weight = nullptr;
    const float* w1 = nullptr;
    const float* w2 = nullptr;
    const float* w3 = nullptr;

    // RoPE
    std::vector<float> rope_freqs;

    // Dequant buffers for norm weights (F16 -> F32)
    std::vector<float> attn_norm_dequant;
    std::vector<float> ffn_norm_dequant;

    // Dequant buffers for linear weights
    std::vector<float> wq_dequant;
    std::vector<float> wk_dequant;
    std::vector<float> wv_dequant;
    std::vector<float> wo_dequant;
    std::vector<float> w1_dequant;
    std::vector<float> w2_dequant;
    std::vector<float> w3_dequant;
};

// ============================================================================
// Model Config
// ============================================================================

struct ModelConfig {
    uint32_t n_vocab = 0;
    uint32_t n_ctx = 0;
    uint32_t n_embd = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    float rope_freq_base = 10000.0f;
    float rms_norm_eps = 1e-5f;
};

// ============================================================================
// CPU Backend
// ============================================================================

class CpuBackend final : public Backend {
public:
    CpuBackend();
    ~CpuBackend();

    void init() override;
    ModelInfo load_model(const std::string& model_path) override;
    void forward(const TensorView& in, TensorView& out) override;
    BackendStats stats() const override;

    std::string generate(
        const std::string& prompt,
        int max_tokens = 50,
        const SamplingConfig& sampling = {}
    );

private:
    // Model
    GgufLoader loader_;
    GgufModel model_;
    ModelConfig config_;

    // Weights (original or dequantized)
    const float* token_embd_weight_ = nullptr;
    const float* output_norm_weight_ = nullptr;
    const float* output_weight_ = nullptr;

    // Dequant buffers
    std::vector<float> token_embd_dequant_;
    std::vector<float> output_norm_dequant_;
    std::vector<float> output_dequant_;

    // Layers
    std::vector<TransformerLayer> layers_;

    // Tokenizer & Sampler
    std::unique_ptr<SimpleTokenizer> tokenizer_;
    std::unique_ptr<Sampler> sampler_;

    // Working buffers
    std::vector<float> embed_buf_;
    std::vector<float> hidden_buf_;
    std::vector<float> logits_buf_;

    // KV Cache (future optimization)
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;
    int kv_pos_ = 0;

    // Metrics
    BackendStats last_stats_{};
    PowerLinux power_{};
    bool power_ok_ = false;
    double energy_start_ = 0.0;

    // Internal helpers
    void extract_weights();
    void dequantize_weights();
    void init_rope_freqs();

    void forward_layer(int layer_idx, float* hidden, int seq_len);
    void forward_attention(const TransformerLayer& layer, float* hidden, int seq_len);
    void forward_ffn(const TransformerLayer& layer, float* hidden, int seq_len);
};

} // namespace engine
