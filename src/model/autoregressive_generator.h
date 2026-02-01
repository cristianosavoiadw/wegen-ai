#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <chrono>
#include <functional>

namespace engine {

class SimpleTokenizer;
class Sampler;
class Backend;

// ============================================================================
// Configuração de Geração
// ============================================================================

struct GenerationConfig {
    // Limites
    int max_tokens = 512;
    int max_context_length = 2048;

    // Stopping criteria
    std::vector<int32_t> stop_tokens;  // EOS, etc
    float min_probability = 0.0f;      // Stop se prob < threshold

    // Streaming
    bool stream = true;
    std::function<void(int32_t)> token_callback;  // Called for each token

    // Performance
    bool use_kv_cache = true;
    int prefill_batch_size = 32;  // Batch prompt tokens

    // Logging
    bool verbose = false;
};

// ============================================================================
// Estatísticas de Geração
// ============================================================================

struct GenerationStats {
    // Tokens
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int total_tokens = 0;

    // Tempo
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;

    // Throughput
    double tokens_per_sec = 0.0;
    double prefill_tokens_per_sec = 0.0;
    double decode_tokens_per_sec = 0.0;

    // Stopping reason
    enum StopReason {
        MAX_TOKENS,
        EOS_TOKEN,
        STOP_TOKEN,
        MIN_PROBABILITY,
        ERROR
    } stop_reason;

    void print() const;
};

// ============================================================================
// Autoregressive Generator
// ============================================================================

class AutoregressiveGenerator {
public:
    AutoregressiveGenerator(
        Backend* backend,
        SimpleTokenizer* tokenizer,
        Sampler* sampler
    );

    // Gera texto dado um prompt
    std::string generate(
        const std::string& prompt,
        const GenerationConfig& config = {}
    );

    // Gera continuação de tokens
    std::vector<int32_t> generate_tokens(
        const std::vector<int32_t>& prompt_tokens,
        const GenerationConfig& config = {}
    );

    // Estatísticas da última geração
    const GenerationStats& stats() const { return stats_; }

private:
    Backend* backend_;
    SimpleTokenizer* tokenizer_;
    Sampler* sampler_;

    GenerationStats stats_;

    // Internal phases
    void prefill_phase(
        const std::vector<int32_t>& prompt_tokens,
        const GenerationConfig& config
    );

    void decode_phase(
        std::vector<int32_t>& output_tokens,
        const GenerationConfig& config
    );

    bool should_stop(
        int32_t token,
        int generated_count,
        float probability,
        const GenerationConfig& config
    ) const;

    // Buffers
    std::vector<float> logits_buffer_;
};

// ============================================================================
// Batch Prefill (para múltiplos prompts)
// ============================================================================

struct BatchGenerationRequest {
    std::string prompt;
    GenerationConfig config;
    int request_id = 0;
};

struct BatchGenerationResult {
    std::string generated_text;
    GenerationStats stats;
    int request_id = 0;
};

class BatchGenerator {
public:
    BatchGenerator(
        Backend* backend,
        SimpleTokenizer* tokenizer
    );

    // Processa múltiplas requests em batch
    std::vector<BatchGenerationResult> generate_batch(
        const std::vector<BatchGenerationRequest>& requests
    );

private:
    Backend* backend_;
    SimpleTokenizer* tokenizer_;
};

} // namespace engine