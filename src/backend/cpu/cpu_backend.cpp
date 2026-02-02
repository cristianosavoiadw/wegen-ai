#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ops.h"
#include "backend/cpu/quants.h"
#include "model/gguf_loader.h"

#include <iostream>
#include <chrono>
#include <cmath>

namespace engine {

// forward
namespace ops {
    void dequantize_auto(float* dst, const void* src, int n, GgmlType type);
}

/* ================================================= */

CpuBackend::CpuBackend() = default;


/* ================================================= */

void CpuBackend::init() {
    std::cout << "[cpu] init()\n";
    last_stats_ = BackendStats{};
}

/* ================================================= */
/* LOAD MODEL */
/* ================================================= */

ModelInfo CpuBackend::load_model(const std::string& path) {

    std::cout << "[cpu] loading model: " << path << "\n";

    model_ = GgufLoader::load(path);

    std::cout << "[cpu] " << model_.summary() << "\n";

    /* ---- Config ---- */

    config_.n_ctx     = model_.context_length();
    config_.n_embd    = model_.embedding_dim();
    config_.n_layers  = model_.n_layers();

    config_.n_vocab     = 32000;
    config_.n_heads     = 32;
    config_.n_kv_heads  = 8;

    std::cout << "[cpu] config: "
              << "vocab=" << config_.n_vocab
              << " ctx=" << config_.n_ctx
              << " emb=" << config_.n_embd
              << " layers=" << config_.n_layers
              << " heads=" << config_.n_heads << "\n";

    /* ---- Extract raw weights ---- */

    extract_weights();

    /* ---- Dequant ---- */

    dequantize_weights();

    /* ---- Buffers ---- */

    embed_buf_.resize(config_.n_embd);
    hidden_buf_.resize(config_.n_embd);
    logits_buf_.resize(config_.n_vocab);

    /* ---- KV cache ---- */

    size_t kv_size = config_.n_layers * config_.n_ctx * config_.n_embd;

    k_cache_.resize(kv_size, 0.0f);
    v_cache_.resize(kv_size, 0.0f);
    kv_pos_ = 0;

    std::cout << "[cpu] model loaded successfully\n";

    return ModelInfo{
        .context_length = config_.n_ctx,
        .embedding_dim  = config_.n_embd,
        .vocab_size     = config_.n_vocab
    };
}

/* ================================================= */
/* DEQUANT */
/* ================================================= */

void CpuBackend::dequantize_weights() {

    std::cout << "[cpu] dequantizing weights...\n";

    /* ---- token embedding ---- */

    if (token_embd_weight_) {

        auto type = model_.tensor_type("token_embd.weight");

        if (type != GgmlType::F32) {

            size_t size = (size_t)config_.n_vocab * config_.n_embd;

            token_embd_dequant_.resize(size);

            ops::dequantize_auto(
                token_embd_dequant_.data(),
                token_embd_weight_,
                size,
                type
            );

            token_embd_weight_ = token_embd_dequant_.data();
        }
    }

    /* ---- output ---- */

    if (output_weight_) {

        auto type = model_.tensor_type("output.weight");

        if (type != GgmlType::F32) {

            size_t size = (size_t)config_.n_vocab * config_.n_embd;

            output_dequant_.resize(size);

            ops::dequantize_auto(
                output_dequant_.data(),
                output_weight_,
                size,
                type
            );

            output_weight_ = output_dequant_.data();
        }
    }

    /* ---- output norm ---- */

    if (output_norm_weight_) {

        auto type = model_.tensor_type("output_norm.weight");

        if (type != GgmlType::F32) {

            output_norm_dequant_.resize(config_.n_embd);

            ops::dequantize_auto(
                output_norm_dequant_.data(),
                output_norm_weight_,
                config_.n_embd,
                type
            );

            output_norm_weight_ = output_norm_dequant_.data();
        }
    }

    /* ---- layers ---- */

    for (uint32_t i = 0; i < config_.n_layers; ++i) {

        auto& L = layers_[i];

        std::string p = "blk." + std::to_string(i) + ".";

        auto dq = [&](const float*& w,
                      std::vector<float>& buf,
                      const std::string& name,
                      size_t n) {

            if (!w) return;

            auto t = model_.tensor_type(name);

            if (t != GgmlType::F32) {

                buf.resize(n);

                ops::dequantize_auto(
                    buf.data(),
                    w,
                    n,
                    t
                );

                w = buf.data();
            }
        };

        dq(L.attn_norm_weight, L.attn_norm_dequant,
           p + "attn_norm.weight", config_.n_embd);

        dq(L.wq, L.wq_dequant, p + "attn_q.weight",
           (size_t)config_.n_embd * config_.n_embd);

        dq(L.wk, L.wk_dequant, p + "attn_k.weight",
           (size_t)config_.n_embd * config_.n_embd);

        dq(L.wv, L.wv_dequant, p + "attn_v.weight",
           (size_t)config_.n_embd * config_.n_embd);

        dq(L.wo, L.wo_dequant, p + "attn_output.weight",
           (size_t)config_.n_embd * config_.n_embd);

        dq(L.ffn_norm_weight, L.ffn_norm_dequant,
           p + "ffn_norm.weight", config_.n_embd);

        size_t ffn_dim = (size_t)config_.n_embd * 4;

        dq(L.w1, L.w1_dequant, p + "ffn_gate.weight",
           ffn_dim * config_.n_embd);

        dq(L.w2, L.w2_dequant, p + "ffn_down.weight",
           config_.n_embd * ffn_dim);

        dq(L.w3, L.w3_dequant, p + "ffn_up.weight",
           ffn_dim * config_.n_embd);
    }

    std::cout << "[cpu] dequant done\n";
}

/* ================================================= */
/* EXTRACT */
/* ================================================= */

void CpuBackend::extract_weights() {

    token_embd_weight_ = (const float*)
        model_.tensor_ptr("token_embd.weight");

    output_norm_weight_ = (const float*)
        model_.tensor_ptr("output_norm.weight");

    output_weight_ = (const float*)
        model_.tensor_ptr("output.weight");

    if (!output_weight_) {
        output_weight_ = (const float*)
            model_.tensor_ptr("lm_head.weight");
    }

    layers_.resize(config_.n_layers);

    for (uint32_t i = 0; i < config_.n_layers; ++i) {

        auto& L = layers_[i];

        std::string p = "blk." + std::to_string(i) + ".";

        L.attn_norm_weight = (const float*)
            model_.tensor_ptr(p + "attn_norm.weight");

        L.wq = (const float*)model_.tensor_ptr(p + "attn_q.weight");
        L.wk = (const float*)model_.tensor_ptr(p + "attn_k.weight");
        L.wv = (const float*)model_.tensor_ptr(p + "attn_v.weight");
        L.wo = (const float*)model_.tensor_ptr(p + "attn_output.weight");

        L.ffn_norm_weight = (const float*)
            model_.tensor_ptr(p + "ffn_norm.weight");

        L.w1 = (const float*)model_.tensor_ptr(p + "ffn_gate.weight");
        L.w2 = (const float*)model_.tensor_ptr(p + "ffn_down.weight");
        L.w3 = (const float*)model_.tensor_ptr(p + "ffn_up.weight");
    }

    std::cout << "[cpu] weights extracted\n";
}

/* ================================================= */
/* FORWARD TOKEN */
/* ================================================= */

void CpuBackend::forward(const TensorView& in, TensorView& out) {

    auto t0 = std::chrono::steady_clock::now();

    int32_t token_id = *(int32_t*)in.data;
    float* logits = (float*)out.data;

    if (token_id < 0 || token_id >= (int32_t)config_.n_vocab) {
        ops::fill_f32(logits, 0.0f, config_.n_vocab);
        return;
    }

    /* ---- embedding ---- */

    const float* emb = token_embd_weight_ +
                       token_id * config_.n_embd;

    ops::copy_f32(hidden_buf_.data(), emb, config_.n_embd);

    /* ---- transformer ---- */

    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        forward_layer(i, hidden_buf_.data(), 1);
    }

    /* ---- norm ---- */

    if (output_norm_weight_) {
        ops::rms_norm_f32(
            hidden_buf_.data(),
            hidden_buf_.data(),
            output_norm_weight_,
            config_.n_embd
        );
    }

    /* ---- output ---- */

    ops::matmul_f32(
        hidden_buf_.data(),
        output_weight_,
        logits,
        1,
        config_.n_vocab,
        config_.n_embd
    );

    kv_pos_++;

    auto t1 = std::chrono::steady_clock::now();

    last_stats_.exec_time_ms +=
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    last_stats_.tokens_total++;
}

/* ================================================= */
/* LAYER */
/* ================================================= */

void CpuBackend::forward_layer(int idx, float* hidden, int seq_len) {

    auto& L = layers_[idx];

    std::vector<float> residual(config_.n_embd);
    ops::copy_f32(residual.data(), hidden, config_.n_embd);

    forward_attention(L, hidden, seq_len);

    ops::add_f32(hidden, residual.data(), config_.n_embd);

    ops::copy_f32(residual.data(), hidden, config_.n_embd);

    forward_ffn(L, hidden, seq_len);

    ops::add_f32(hidden, residual.data(), config_.n_embd);
}

/* ================================================= */
/* ATTENTION */
/* ================================================= */

void CpuBackend::forward_attention(
    const TransformerLayer& L,
    float* hidden,
    int seq_len
) {

    if (L.attn_norm_weight) {
        ops::rms_norm_f32(
            hidden, hidden,
            L.attn_norm_weight,
            config_.n_embd
        );
    }

    std::vector<float> Q(config_.n_embd);
    std::vector<float> K(config_.n_embd);
    std::vector<float> V(config_.n_embd);

    ops::matmul_f32(hidden, L.wq, Q.data(),
                   seq_len, config_.n_embd, config_.n_embd);

    ops::matmul_f32(hidden, L.wk, K.data(),
                   seq_len, config_.n_embd, config_.n_embd);

    ops::matmul_f32(hidden, L.wv, V.data(),
                   seq_len, config_.n_embd, config_.n_embd);

    std::vector<float> out(config_.n_embd);

    ops::attention_f32(
        out.data(),
        Q.data(), K.data(), V.data(),
        seq_len, config_.n_embd
    );

    ops::matmul_f32(
        out.data(), L.wo, hidden,
        seq_len, config_.n_embd, config_.n_embd
    );
}

/* ================================================= */
/* FFN */
/* ================================================= */

void CpuBackend::forward_ffn(
    const TransformerLayer& L,
    float* hidden,
    int seq_len
) {

    if (L.ffn_norm_weight) {
        ops::rms_norm_f32(
            hidden, hidden,
            L.ffn_norm_weight,
            config_.n_embd
        );
    }

    int ffn_dim = config_.n_embd * 4;

    std::vector<float> gate(ffn_dim);
    std::vector<float> up(ffn_dim);

    ops::matmul_f32(hidden, L.w1, gate.data(),
                   seq_len, ffn_dim, config_.n_embd);

    ops::silu_f32(gate.data(), ffn_dim);

    ops::matmul_f32(hidden, L.w3, up.data(),
                   seq_len, ffn_dim, config_.n_embd);

    ops::mul_f32(gate.data(), gate.data(), up.data(), ffn_dim);

    ops::matmul_f32(
        gate.data(), L.w2, hidden,
        seq_len, config_.n_embd, ffn_dim
    );
}

/* ================================================= */
/* STATS */
/* ================================================= */

BackendStats CpuBackend::stats() const {

    BackendStats s = last_stats_;

    if (s.exec_time_ms > 0) {
        s.tokens_per_sec =
            (s.tokens_total * 1000.0) / s.exec_time_ms;
    }

    return s;
}

std::string CpuBackend::generate(
    const std::string& prompt,
    int max_tokens,
    const SamplingConfig& sampling
) {
    std::cout << "[cpu] generating from prompt: \"" << prompt << "\"\n";

    // Se ainda não tem tokenizer real, placeholder simples:
    // (isso vai ser substituído pelo GGUF tokenizer depois)

    std::vector<int32_t> tokens;

    // Hack temporário: usa token 1 como start
    tokens.push_back(1);

    // Prefill
    for (int32_t t : tokens) {
        TensorView in;
        in.data = &t;

        TensorView out;
        out.data = logits_buf_.data();

        forward(in, out);
    }

    std::vector<int32_t> generated;

    for (int i = 0; i < max_tokens; ++i) {

        int32_t next = 0;

        // argmax simples (temporário, depois entra sampler real)
        float best = logits_buf_[0];

        for (uint32_t j = 1; j < config_.n_vocab; ++j) {
            if (logits_buf_[j] > best) {
                best = logits_buf_[j];
                next = j;
            }
        }

        generated.push_back(next);

        TensorView in;
        in.data = &next;

        TensorView out;
        out.data = logits_buf_.data();

        forward(in, out);
    }

    std::cout << "[cpu] generated " << generated.size() << " tokens\n";

    // Sem tokenizer real ainda → retorna ids como string

    std::string result;
    for (auto t : generated) {
        result += "[" + std::to_string(t) + "]";
    }

    return result;
}


} // namespace engine
