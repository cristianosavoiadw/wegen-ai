#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ops.h"
#include "backend/cpu/quants.h"

#include <iostream>
#include <chrono>
#include <cmath>

namespace engine {

// Declaração forward
namespace ops {
    void dequantize_auto(float* dst, const void* src, int n, GgmlType type);
}

CpuBackend::CpuBackend() = default;
CpuBackend::~CpuBackend() = default;

// ============================================================================
// INICIALIZAÇÃO
// ============================================================================

void CpuBackend::init() {
    std::cout << "[cpu] init()\n";
    last_stats_ = BackendStats{};

    power_ok_ = power_.init();
    if (!power_ok_) {
        std::cout << "[cpu] powercap not available\n";
    } else {
        std::cout << "[cpu] energy path: " << power_.energy_path() << "\n";
        auto joules = power_.read_joules();
        if (joules) {
            energy_start_ = *joules;
        }
    }
}

// ============================================================================
// CARREGAMENTO DO MODELO
// ============================================================================

    ModelInfo CpuBackend::load_model(const std::string& path) {
    std::cout << "[cpu] loading model: " << path << "\n";

    model_ = loader_.load(path);
    std::cout << "[cpu] " << model_.summary() << "\n";

    // Configuração
    config_.n_ctx = model_.context_length();
    config_.n_embd = model_.embedding_dim();
    config_.n_layers = model_.n_layers();
    config_.n_vocab = 32000;
    config_.n_heads = 32;
    config_.n_kv_heads = 8;

    std::cout << "[cpu] config: "
              << "vocab=" << config_.n_vocab
              << " ctx=" << config_.n_ctx
              << " embd=" << config_.n_embd
              << " layers=" << config_.n_layers
              << " heads=" << config_.n_heads << "\n";

    // Extrai pesos (ponteiros brutos)
    extract_weights();

    // NOVO: Dequantiza pesos se necessário
    dequantize_weights();

    // Inicializa RoPE
    init_rope_freqs();

    // Aloca buffers
    embed_buf_.resize(config_.n_embd);
    hidden_buf_.resize(config_.n_embd);
    logits_buf_.resize(config_.n_vocab);

    // KV Cache
    size_t kv_cache_size = config_.n_layers * config_.n_ctx * config_.n_embd;
    k_cache_.resize(kv_cache_size, 0.0f);
    v_cache_.resize(kv_cache_size, 0.0f);
    kv_pos_ = 0;

    // Tokenizer e sampler
    tokenizer_ = std::make_unique<SimpleTokenizer>();
    tokenizer_->load_from_gguf(path);

    sampler_ = std::make_unique<Sampler>();

    std::cout << "[cpu] model loaded successfully\n";

    return ModelInfo{
        .context_length = config_.n_ctx,
        .embedding_dim  = config_.n_embd,
        .vocab_size     = config_.n_vocab
    };
}
void CpuBackend::dequantize_weights() {
    std::cout << "[cpu] dequantizing weights...\n";

    // === TOKEN EMBEDDING ===
    if (token_embd_weight_) {
        auto type = model_.tensor_type("token_embd.weight");

        if (type != GgmlType::F32) {
            std::cout << "[cpu] dequantizing token_embd (type=" << static_cast<int>(type) << ")\n";

            size_t size = config_.n_vocab * config_.n_embd;
            token_embd_dequant_.resize(size);

            ops::dequantize_auto(
                token_embd_dequant_.data(),
                token_embd_weight_,
                size,
                type
            );

            token_embd_weight_ = token_embd_dequant_.data();
            std::cout << "[cpu] token_embd dequantized: " << size << " elements\n";
        } else {
            std::cout << "[cpu] token_embd already F32\n";
        }
    }

    // === OUTPUT WEIGHT ===
    if (output_weight_) {
        GgmlType type = GgmlType::F32;
        const char* tensor_name = nullptr;

        if (model_.tensor_info("output.weight")) {
            tensor_name = "output.weight";
            type = model_.tensor_type("output.weight");
        } else if (model_.tensor_info("lm_head.weight")) {
            tensor_name = "lm_head.weight";
            type = model_.tensor_type("lm_head.weight");
        }

        if (tensor_name && type != GgmlType::F32) {
            std::cout << "[cpu] dequantizing " << tensor_name
                      << " (type=" << static_cast<int>(type) << ")\n";

            size_t size = config_.n_embd * config_.n_vocab;
            output_dequant_.resize(size);

            ops::dequantize_auto(
                output_dequant_.data(),
                output_weight_,
                size,
                type
            );

            output_weight_ = output_dequant_.data();
            std::cout << "[cpu] output_weight dequantized: " << size << " elements\n";
        } else {
            std::cout << "[cpu] output_weight already F32 or tied\n";
        }
    }

    // === LAYERS ===
    std::cout << "[cpu] layer dequantization skipped (Phase 3.5)\n";

    std::cout << "[cpu] dequantization complete\n";
}

// ============================================================================
// EXTRAÇÃO DE PESOS
// ============================================================================

void CpuBackend::extract_weights() {
    // Embedding
    token_embd_weight_ = static_cast<const float*>(
        model_.tensor_ptr("token_embd.weight")
    );

    // Output
    output_norm_weight_ = static_cast<const float*>(
        model_.tensor_ptr("output_norm.weight")
    );

    output_weight_ = static_cast<const float*>(
        model_.tensor_ptr("output.weight")
    );

    // Se não encontrar output.weight, tenta lm_head
    if (!output_weight_) {
        output_weight_ = static_cast<const float*>(
            model_.tensor_ptr("lm_head.weight")
        );
    }

    // Layers
    layers_.resize(config_.n_layers);

    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "blk." + std::to_string(i) + ".";

        // Attention norm
        layer.attn_norm_weight = static_cast<const float*>(
            model_.tensor_ptr(prefix + "attn_norm.weight")
        );

        // Projections Q, K, V, O
        layer.wq = static_cast<const float*>(
            model_.tensor_ptr(prefix + "attn_q.weight")
        );

        layer.wk = static_cast<const float*>(
            model_.tensor_ptr(prefix + "attn_k.weight")
        );

        layer.wv = static_cast<const float*>(
            model_.tensor_ptr(prefix + "attn_v.weight")
        );

        layer.wo = static_cast<const float*>(
            model_.tensor_ptr(prefix + "attn_output.weight")
        );

        // FFN norm
        layer.ffn_norm_weight = static_cast<const float*>(
            model_.tensor_ptr(prefix + "ffn_norm.weight")
        );

        // FFN projections
        layer.w1 = static_cast<const float*>(
            model_.tensor_ptr(prefix + "ffn_gate.weight")
        );

        layer.w2 = static_cast<const float*>(
            model_.tensor_ptr(prefix + "ffn_down.weight")
        );

        layer.w3 = static_cast<const float*>(
            model_.tensor_ptr(prefix + "ffn_up.weight")
        );
    }

    std::cout << "[cpu] weights extracted\n";
}

// ============================================================================
// INICIALIZAÇÃO ROPE
// ============================================================================

void CpuBackend::init_rope_freqs() {
    const int head_dim = config_.n_embd / config_.n_heads;
    const int half_dim = head_dim / 2;

    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        auto& layer = layers_[i];
        layer.rope_freqs.resize(half_dim);

        for (int d = 0; d < half_dim; ++d) {
            float freq = 1.0f / std::pow(
                config_.rope_freq_base,
                static_cast<float>(2 * d) / head_dim
            );
            layer.rope_freqs[d] = freq;
        }
    }
}

// ============================================================================
// FORWARD PASS (1 token)
// ============================================================================

void CpuBackend::forward(const TensorView& in, TensorView& out) {
    auto t0 = std::chrono::steady_clock::now();

    if (!in.data || !out.data) {
        std::cerr << "[cpu] forward: NULL tensors\n";
        return;
    }

    int32_t token_id = *static_cast<const int32_t*>(in.data);
    float* logits = static_cast<float*>(out.data);

    // Verifica bounds
    if (token_id < 0 || static_cast<uint32_t>(token_id) >= config_.n_vocab) {
        std::cerr << "[cpu] token_id out of range: " << token_id << "\n";
        ops::fill_f32(logits, 0.0f, config_.n_vocab);
        return;
    }

    // 1. Embedding
    if (token_embd_weight_) {
        const float* emb = token_embd_weight_ + (token_id * config_.n_embd);
        ops::copy_f32(hidden_buf_.data(), emb, config_.n_embd);
    } else {
        ops::fill_f32(hidden_buf_.data(), 0.0f, config_.n_embd);
    }

    // 2. SKIP LAYERS POR ENQUANTO (teste)
    std::cout << "[cpu] WARNING: skipping layers (testing)\n";
    // for (uint32_t i = 0; i < config_.n_layers; ++i) {
    //     forward_layer(i, hidden_buf_.data(), 1);
    // }

    // 3. Norm
    if (output_norm_weight_) {
        ops::rms_norm_f32(
            hidden_buf_.data(), hidden_buf_.data(),
            output_norm_weight_, config_.n_embd
        );
    }

    // 4. Output projection
    if (output_weight_) {
        std::cout << "[cpu] calling matmul: M=1, N=" << config_.n_vocab
                  << ", K=" << config_.n_embd << "\n";

        ops::matmul_f32(
            hidden_buf_.data(),
            output_weight_,
            logits,
            1, config_.n_vocab, config_.n_embd
        );
    } else {
        ops::fill_f32(logits, 0.0f, config_.n_vocab);
    }

    kv_pos_++;

    auto t1 = std::chrono::steady_clock::now();

    last_stats_.exec_time_ms +=
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    last_stats_.tokens_total += 1;
}

// ============================================================================
// FORWARD LAYER
// ============================================================================

void CpuBackend::forward_layer(int layer_idx, float* hidden, int seq_len) {
    const auto& layer = layers_[layer_idx];

    // Residual connection
    std::vector<float> residual(config_.n_embd);
    ops::copy_f32(residual.data(), hidden, config_.n_embd);

    // 1. Attention
    forward_attention(layer, hidden, seq_len);

    // Add residual
    ops::add_f32(hidden, residual.data(), config_.n_embd);

    // Save new residual
    ops::copy_f32(residual.data(), hidden, config_.n_embd);

    // 2. FFN
    forward_ffn(layer, hidden, seq_len);

    // Add residual
    ops::add_f32(hidden, residual.data(), config_.n_embd);
}

// continua na próxima parte...
// Continuação de cpu_backend.cpp

// ============================================================================
// ATTENTION (simplificado - sem GQA por enquanto)
// ============================================================================

void CpuBackend::forward_attention(
    const TransformerLayer& layer,
    float* hidden,
    int seq_len
) {
    const int head_dim = config_.n_embd / config_.n_heads;

    // RMS Norm
    if (layer.attn_norm_weight) {
        ops::rms_norm_f32(
            hidden, hidden,
            layer.attn_norm_weight,
            config_.n_embd,
            config_.rms_norm_eps
        );
    }

    // Buffers para Q, K, V
    std::vector<float> Q(config_.n_embd);
    std::vector<float> K(config_.n_embd);
    std::vector<float> V(config_.n_embd);

    // Projeções Q, K, V
    if (layer.wq) {
        ops::matmul_f32(hidden, layer.wq, Q.data(),
                       seq_len, config_.n_embd, config_.n_embd);
    }

    if (layer.wk) {
        ops::matmul_f32(hidden, layer.wk, K.data(),
                       seq_len, config_.n_embd, config_.n_embd);
    }

    if (layer.wv) {
        ops::matmul_f32(hidden, layer.wv, V.data(),
                       seq_len, config_.n_embd, config_.n_embd);
    }

    // RoPE (Rotary Position Embedding)
    if (!layer.rope_freqs.empty()) {
        ops::rope_f32(
            Q.data(), layer.rope_freqs.data(),
            seq_len, config_.n_heads, head_dim,
            kv_pos_
        );

        ops::rope_f32(
            K.data(), layer.rope_freqs.data(),
            seq_len, config_.n_heads, head_dim,
            kv_pos_
        );
    }

    // FASE 3 Simplificada: Attention sem KV cache
    // FASE 4: Implementar KV cache completo

    std::vector<float> attn_out(config_.n_embd);
    ops::attention_f32(
        attn_out.data(),
        Q.data(), K.data(), V.data(),
        seq_len, config_.n_embd
    );

    // Output projection
    if (layer.wo) {
        ops::matmul_f32(
            attn_out.data(), layer.wo, hidden,
            seq_len, config_.n_embd, config_.n_embd
        );
    } else {
        ops::copy_f32(hidden, attn_out.data(), config_.n_embd);
    }
}

// ============================================================================
// FFN (Feed-Forward Network)
// ============================================================================

void CpuBackend::forward_ffn(
    const TransformerLayer& layer,
    float* hidden,
    int seq_len
) {
    // RMS Norm
    if (layer.ffn_norm_weight) {
        ops::rms_norm_f32(
            hidden, hidden,
            layer.ffn_norm_weight,
            config_.n_embd,
            config_.rms_norm_eps
        );
    }

    // FFN dimension (tipicamente 4 * n_embd ou similar)
    const int ffn_dim = config_.n_embd * 4;  // TODO: extrair do GGUF

    std::vector<float> gate(ffn_dim);
    std::vector<float> up(ffn_dim);

    // Gate projection + SiLU
    if (layer.w1) {
        ops::matmul_f32(hidden, layer.w1, gate.data(),
                       seq_len, ffn_dim, config_.n_embd);
        ops::silu_f32(gate.data(), ffn_dim);
    }

    // Up projection
    if (layer.w3) {
        ops::matmul_f32(hidden, layer.w3, up.data(),
                       seq_len, ffn_dim, config_.n_embd);
    }

    // Element-wise multiply
    ops::mul_f32(gate.data(), gate.data(), up.data(), ffn_dim);

    // Down projection
    if (layer.w2) {
        ops::matmul_f32(gate.data(), layer.w2, hidden,
                       seq_len, config_.n_embd, ffn_dim);
    }
}

// ============================================================================
// GERAÇÃO DE TEXTO
// ============================================================================

std::string CpuBackend::generate(
    const std::string& prompt,
    int max_tokens,
    const SamplingConfig& sampling
) {
    std::cout << "[cpu] generating from prompt: \"" << prompt << "\"\n";

    // 1. Tokeniza prompt
    auto tokens = tokenizer_->encode(prompt);

    std::cout << "[cpu] prompt tokens: " << tokens.size() << "\n";

    // 2. Processa prompt (prefill)
    // FASE 3: Processa token por token (ineficiente)
    // FASE 4: Batch prefill

    for (size_t i = 0; i < tokens.size(); ++i) {
        TensorView in_view;
        in_view.data = &tokens[i];

        TensorView out_view;
        out_view.data = logits_buf_.data();

        forward(in_view, out_view);
    }

    // 3. Geração autoregressiva
    std::vector<int32_t> generated_tokens;

    sampler_ = std::make_unique<Sampler>(sampling);

    for (int i = 0; i < max_tokens; ++i) {
        // Sample próximo token
        int32_t next_token = sampler_->sample(
            logits_buf_.data(),
            config_.n_vocab
        );

        // Verifica EOS
        if (next_token == tokenizer_->eos_token()) {
            std::cout << "[cpu] EOS token generated\n";
            break;
        }

        generated_tokens.push_back(next_token);

        // Forward pass com novo token
        TensorView in_view;
        in_view.data = &next_token;

        TensorView out_view;
        out_view.data = logits_buf_.data();

        forward(in_view, out_view);
    }

    std::cout << "[cpu] generated " << generated_tokens.size() << " tokens\n";

    // 4. Detokeniza
    std::string result = tokenizer_->decode(generated_tokens);

    return result;
}

// ============================================================================
// ESTATÍSTICAS
// ============================================================================

BackendStats CpuBackend::stats() const {
    BackendStats s = last_stats_;

    // Calcula tokens/sec
    if (s.exec_time_ms > 0) {
        s.tokens_per_sec = (s.tokens_total * 1000.0) / s.exec_time_ms;
    }

    // Métricas de energia
    if (power_ok_) {
        auto energy_end = power_.read_joules();
        if (energy_end) {
            double total_joules = *energy_end - energy_start_;
            double time_seconds = s.exec_time_ms / 1000.0;

            if (time_seconds > 0) {
                s.watts_avg = total_joules / time_seconds;

                if (s.watts_avg > 0) {
                    s.tokens_per_watt = s.tokens_per_sec / s.watts_avg;
                }
            }

            s.energy_total_joules = total_joules;
        }
    }

    return s;
}

} // namespace engine
