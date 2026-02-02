#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/ops.h"
#include "backend/cpu/quants.h"
#include "model/gguf_loader.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>

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

    config_.n_vocab     = model_.vocab_size();
    config_.n_heads     = model_.n_heads();
    config_.n_kv_heads  = model_.n_kv_heads();

    std::cout << "[cpu] config: "
          << "vocab=" << config_.n_vocab
          << " ctx=" << config_.n_ctx
          << " emb=" << config_.n_embd
          << " layers=" << config_.n_layers
          << " heads=" << config_.n_heads
          << " kv_heads=" << config_.n_kv_heads << "\n";

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

    /* ---- Tokenizer e Sampler ---- */

    std::cout << "[cpu] initializing tokenizer...\n";
    tokenizer_ = std::make_unique<SimpleTokenizer>();
    if (!tokenizer_->load_from_gguf(path)) {
        std::cerr << "[cpu] WARNING: tokenizer failed to load\n";
    }

    std::cout << "[cpu] initializing sampler...\n";
    sampler_ = std::make_unique<Sampler>();

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

    auto dq = [&](const std::string& name,
                  const float*& w,
                  std::vector<float>& buf) {
        if (!w) return;

        const auto* info = model_.tensor_info(name);
        if (!info) {
            std::cerr << "[cpu] WARNING: tensor not found: " << name << "\n";
            return;
        }

        const auto t = info->type;
        if (t == GgmlType::F32) return;

        const uint64_t n64 = info->numel();
        if (n64 == 0) return;

        if (n64 > uint64_t(std::numeric_limits<int>::max())) {
            throw std::runtime_error("[cpu] tensor too large for dequantize_auto int n");
        }

        const int n = static_cast<int>(n64);
        buf.resize((size_t)n64);

        ops::dequantize_auto(buf.data(), w, n, t);
        w = buf.data();
    };

    // token embd
    dq("token_embd.weight", token_embd_weight_, token_embd_dequant_);
    if (token_embd_weight_) {
        std::cout << "[cpu] token_embd ok\n";
    }

    // output norm
    dq("output_norm.weight", output_norm_weight_, output_norm_dequant_);
    if (output_norm_weight_) {
        std::cout << "[cpu] output_norm ok\n";
    }

    // output weight (output.weight ou lm_head.weight)
    const std::string out_name =
        model_.tensor_ptr("output.weight") ? "output.weight" : "lm_head.weight";

    dq(out_name, output_weight_, output_dequant_);
    std::cout << "[cpu] output weight ok (" << out_name << ")\n";

    // layers
    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        auto& L = layers_[i];
        const std::string p = "blk." + std::to_string(i) + ".";

        dq(p + "attn_norm.weight",   L.attn_norm_weight, L.attn_norm_dequant);
        dq(p + "attn_q.weight",      L.wq,              L.wq_dequant);
        dq(p + "attn_k.weight",      L.wk,              L.wk_dequant);
        dq(p + "attn_v.weight",      L.wv,              L.wv_dequant);
        dq(p + "attn_output.weight", L.wo,              L.wo_dequant);

        dq(p + "ffn_norm.weight",    L.ffn_norm_weight, L.ffn_norm_dequant);
        dq(p + "ffn_gate.weight",    L.w1,              L.w1_dequant);
        dq(p + "ffn_down.weight",    L.w2,              L.w2_dequant);
        dq(p + "ffn_up.weight",      L.w3,              L.w3_dequant);
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
// Adicione no forward() do cpu_backend.cpp:

void CpuBackend::forward(const TensorView& in, TensorView& out) {
    int32_t token_id = *static_cast<const int32_t*>(in.data);
    float* logits = static_cast<float*>(out.data);

    std::cout << "[forward] START token_id=" << token_id << "\n";

    if (token_id < 0 || static_cast<uint32_t>(token_id) >= config_.n_vocab) {
        std::cerr << "[forward] ERROR: token_id out of range!\n";
        return;
    }

    // 1. Embedding
    if (token_embd_weight_) {
        const float* emb = token_embd_weight_ + (token_id * config_.n_embd);
        ops::copy_f32(hidden_buf_.data(), emb, config_.n_embd);

        std::cout << "[forward] after embedding: hidden[0]=" << hidden_buf_[0]
                  << " hidden[100]=" << hidden_buf_[100] << "\n";

        if (std::isnan(hidden_buf_[0]) || std::isinf(hidden_buf_[0])) {
            std::cerr << "[ERROR] NaN/Inf AFTER EMBEDDING!\n";
            std::cerr << "[DEBUG] token_embd_weight_[0]=" << token_embd_weight_[0] << "\n";
            std::cerr << "[DEBUG] emb[0]=" << emb[0] << "\n";
            return;
        }
    } else {
        std::cerr << "[ERROR] token_embd_weight_ is NULL!\n";
        return;
    }

    // 2. Layers
    std::cout << "[forward] processing " << config_.n_layers << " layers...\n";

    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        forward_layer(i, hidden_buf_.data(), 1);

        // Check após cada layer
        if (std::isnan(hidden_buf_[0]) || std::isinf(hidden_buf_[0])) {
            std::cerr << "[ERROR] NaN/Inf AFTER LAYER " << i << "!\n";
            std::cerr << "[DEBUG] hidden[0]=" << hidden_buf_[0] << "\n";
            std::cerr << "[DEBUG] hidden[10]=" << hidden_buf_[10] << "\n";
            return;
        }

        if (i == 0 || i == config_.n_layers - 1) {
            std::cout << "[forward] after layer " << i << ": hidden[0]="
                      << hidden_buf_[0] << "\n";
        }
    }

    std::cout << "[forward] layers done\n";

    // 3. Output norm
    if (output_norm_weight_) {
        std::cout << "[forward] applying output_norm...\n";

        ops::rms_norm_f32(
            hidden_buf_.data(),
            hidden_buf_.data(),
            output_norm_weight_,
            config_.n_embd,
            1e-5f
        );

        std::cout << "[forward] after output_norm: hidden[0]=" << hidden_buf_[0] << "\n";

        if (std::isnan(hidden_buf_[0]) || std::isinf(hidden_buf_[0])) {
            std::cerr << "[ERROR] NaN/Inf AFTER OUTPUT_NORM!\n";
            std::cerr << "[DEBUG] output_norm_weight_[0]=" << output_norm_weight_[0] << "\n";
            return;
        }
    } else {
        std::cout << "[forward] WARNING: output_norm_weight_ is NULL, skipping\n";
    }

    // 4. Output projection
    if (output_weight_) {
        std::cout << "[forward] applying output projection...\n";

        ops::matmul_f32(
            hidden_buf_.data(),
            output_weight_,
            logits,
            1, config_.n_vocab, config_.n_embd
        );

        std::cout << "[forward] after matmul: logits[0]=" << logits[0]
                  << " logits[100]=" << logits[100] << "\n";

        if (std::isnan(logits[0]) || std::isinf(logits[0])) {
            std::cerr << "[ERROR] NaN/Inf AFTER OUTPUT PROJECTION!\n";
            std::cerr << "[DEBUG] hidden[0]=" << hidden_buf_[0] << "\n";
            std::cerr << "[DEBUG] output_weight_[0]=" << output_weight_[0] << "\n";
            return;
        }
    } else {
        std::cerr << "[ERROR] output_weight_ is NULL!\n";
        return;
    }

    std::cout << "[forward] DONE\n";
}
/* ================================================= */
/* LAYER */
/* ================================================= */

void CpuBackend::forward_layer(int layer_idx, float* hidden, int seq_len) {
    const int n = static_cast<int>(config_.n_embd);
    const auto& L = layers_[layer_idx];

    // Residual 1
    std::vector<float> residual(n);
    ops::copy_f32(residual.data(), hidden, n);

    forward_attention(L, hidden, seq_len);
    ops::add_f32(hidden, residual.data(), n);

    // Residual 2
    ops::copy_f32(residual.data(), hidden, n);

    forward_ffn(L, hidden, seq_len);
    ops::add_f32(hidden, residual.data(), n);
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
    std::cout << std::flush;  // ← FORÇAR OUTPUT

    // 1. Tokeniza
    std::cout << "[debug] tokenizing..." << std::endl;
    auto tokens = tokenizer_->encode(prompt);
    std::cout << "[debug] got " << tokens.size() << " tokens: ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;

    if (tokens.empty()) {
        std::cerr << "[error] tokenization returned empty!\n";
        return "";
    }

    // 2. Prefill
    std::cout << "[debug] prefill starting..." << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "[prefill] " << (i+1) << "/" << tokens.size() << std::endl;

        TensorView in_view;
        in_view.data = &tokens[i];

        TensorView out_view;
        out_view.data = logits_buf_.data();

        forward(in_view, out_view);
    }
    std::cout << "[debug] prefill done" << std::endl;

    // No generate(), após o prefill:

std::cout << "[debug] prefill done\n";

// ========== DEBUG DOS LOGITS ==========
std::cout << "[debug] checking logits after prefill...\n";
float max_logit = -INFINITY;
float min_logit = INFINITY;
int max_idx = 0;
bool has_nan = false;

for (size_t i = 0; i < config_.n_vocab; ++i) {
    float val = logits_buf_[i];

    if (std::isnan(val) || std::isinf(val)) {
        std::cerr << "[ERROR] NaN/Inf at logit[" << i << "]\n";
        has_nan = true;
        break;
    }

    if (val > max_logit) {
        max_logit = val;
        max_idx = i;
    }
    if (val < min_logit) {
        min_logit = val;
    }
}

if (has_nan) {
    std::cerr << "[ERROR] Logits contain NaN/Inf! Cannot continue.\n";
    return "";
}

std::cout << "[debug] logits range: min=" << min_logit
          << " max=" << max_logit
          << " (at token=" << max_idx << ")\n";

std::cout << "[debug] first 10 logits: ";
for (int i = 0; i < 10; ++i) {
    std::cout << logits_buf_[i] << " ";
}
std::cout << "\n";

// Teste com greedy sampler
SamplingConfig greedy_config;
greedy_config.strategy = SamplingStrategy::GREEDY;
Sampler greedy_sampler(greedy_config);

int32_t test_token = greedy_sampler.sample(logits_buf_.data(), config_.n_vocab);
std::cout << "[debug] greedy sampler chose: " << test_token << "\n";

// Verificar se todos os logits são iguais
bool all_equal = true;
float first_val = logits_buf_[0];
for (size_t i = 1; i < std::min(config_.n_vocab, 100u); ++i) {
    if (std::abs(logits_buf_[i] - first_val) > 1e-6) {
        all_equal = false;
        break;
    }
}

if (all_equal) {
    std::cerr << "[WARNING] All logits are equal! Model may not be working.\n";
}
// ========== FIM DEBUG ==========

std::cout << "[debug] decode starting...\n";



    // 3. Generate
    std::cout << "[debug] decode starting..." << std::endl;
    std::vector<int32_t> generated_tokens;

    sampler_ = std::make_unique<Sampler>(sampling);

    for (int i = 0; i < max_tokens; ++i) {
        std::cout << "[decode] " << (i+1) << "/" << max_tokens << std::endl;

        int32_t next_token = sampler_->sample(
            logits_buf_.data(),
            config_.n_vocab
        );

        std::cout << "[decode] sampled: " << next_token << std::endl;

        if (next_token == tokenizer_->eos_token()) {
            std::cout << "[cpu] EOS" << std::endl;
            break;
        }

        generated_tokens.push_back(next_token);

        TensorView in_view;
        in_view.data = &next_token;

        TensorView out_view;
        out_view.data = logits_buf_.data();

        forward(in_view, out_view);
    }

    std::cout << "[debug] decoding text..." << std::endl;
    std::string result = tokenizer_->decode(generated_tokens);
    std::cout << "[debug] done!" << std::endl;

    return result;
}

} // namespace engine
