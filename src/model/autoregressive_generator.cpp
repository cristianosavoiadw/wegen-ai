#include "autoregressive_generator.h"
#include "tokenizer.h"
#include "sampler.h"
#include "../backend/backend.h"
#include "../backend/tensor.h"

#include <iostream>
#include <algorithm>
#include <cmath>

namespace engine {

// ============================================================================
// GenerationStats
// ============================================================================

void GenerationStats::print() const {
    std::cout << "\n=== Generation Statistics ===\n";
    std::cout << "Tokens:\n";
    std::cout << "  Prompt: " << prompt_tokens << "\n";
    std::cout << "  Generated: " << generated_tokens << "\n";
    std::cout << "  Total: " << total_tokens << "\n";

    std::cout << "\nTiming:\n";
    std::cout << "  Prefill: " << prefill_ms << " ms\n";
    std::cout << "  Decode: " << decode_ms << " ms\n";
    std::cout << "  Total: " << total_ms << " ms\n";

    std::cout << "\nThroughput:\n";
    std::cout << "  Overall: " << tokens_per_sec << " tokens/sec\n";
    std::cout << "  Prefill: " << prefill_tokens_per_sec << " tokens/sec\n";
    std::cout << "  Decode: " << decode_tokens_per_sec << " tokens/sec\n";

    std::cout << "\nStop Reason: ";
    switch (stop_reason) {
        case MAX_TOKENS: std::cout << "MAX_TOKENS\n"; break;
        case EOS_TOKEN: std::cout << "EOS_TOKEN\n"; break;
        case STOP_TOKEN: std::cout << "STOP_TOKEN\n"; break;
        case MIN_PROBABILITY: std::cout << "MIN_PROBABILITY\n"; break;
        case ERROR: std::cout << "ERROR\n"; break;
    }
    std::cout << "============================\n\n";
}

// ============================================================================
// AutoregressiveGenerator
// ============================================================================

AutoregressiveGenerator::AutoregressiveGenerator(
    Backend* backend,
    SimpleTokenizer* tokenizer,
    Sampler* sampler
) : backend_(backend), tokenizer_(tokenizer), sampler_(sampler) {
}

std::string AutoregressiveGenerator::generate(
    const std::string& prompt,
    const GenerationConfig& config
) {
    auto start_time = std::chrono::steady_clock::now();

    // 1. Tokenize prompt
    auto prompt_tokens = tokenizer_->encode(prompt);

    if (config.verbose) {
        std::cout << "[gen] prompt tokens: " << prompt_tokens.size() << "\n";
    }

    // 2. Generate tokens
    auto output_tokens = generate_tokens(prompt_tokens, config);

    // 3. Detokenize
    std::string result = tokenizer_->decode(output_tokens);

    auto end_time = std::chrono::steady_clock::now();
    stats_.total_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

    // Update stats
    stats_.total_tokens = stats_.prompt_tokens + stats_.generated_tokens;
    if (stats_.total_ms > 0) {
        stats_.tokens_per_sec = (stats_.total_tokens * 1000.0) / stats_.total_ms;
    }

    if (config.verbose) {
        stats_.print();
    }

    return result;
}

std::vector<int32_t> AutoregressiveGenerator::generate_tokens(
    const std::vector<int32_t>& prompt_tokens,
    const GenerationConfig& config
) {
    stats_ = GenerationStats{};  // Reset
    stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());

    std::vector<int32_t> output_tokens;
    output_tokens.reserve(config.max_tokens);

    // FASE 1: Prefill (processa prompt)
    auto prefill_start = std::chrono::steady_clock::now();
    prefill_phase(prompt_tokens, config);
    auto prefill_end = std::chrono::steady_clock::now();

    stats_.prefill_ms = std::chrono::duration<double, std::milli>(
        prefill_end - prefill_start
    ).count();

    if (stats_.prefill_ms > 0 && stats_.prompt_tokens > 0) {
        stats_.prefill_tokens_per_sec =
            (stats_.prompt_tokens * 1000.0) / stats_.prefill_ms;
    }

    // FASE 2: Decode (gera tokens autoregressivamente)
    auto decode_start = std::chrono::steady_clock::now();
    decode_phase(output_tokens, config);
    auto decode_end = std::chrono::steady_clock::now();

    stats_.decode_ms = std::chrono::duration<double, std::milli>(
        decode_end - decode_start
    ).count();

    stats_.generated_tokens = static_cast<int>(output_tokens.size());

    if (stats_.decode_ms > 0 && stats_.generated_tokens > 0) {
        stats_.decode_tokens_per_sec =
            (stats_.generated_tokens * 1000.0) / stats_.decode_ms;
    }

    return output_tokens;
}

// ============================================================================
// PREFILL PHASE
// ============================================================================

void AutoregressiveGenerator::prefill_phase(
    const std::vector<int32_t>& prompt_tokens,
    const GenerationConfig& config
) {
    if (config.verbose) {
        std::cout << "[gen] prefill phase: " << prompt_tokens.size() << " tokens\n";
    }

    // TODO FASE 4: Batch prefill (processa múltiplos tokens por vez)
    // Por enquanto: token-by-token

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        int32_t token = prompt_tokens[i];

        // Preparar input
        TensorView in_view;
        in_view.data = const_cast<int32_t*>(&token);
        in_view.shape = {1};

        // Alocar buffer de logits se necessário
        if (logits_buffer_.empty()) {
            logits_buffer_.resize(tokenizer_->vocab_size());
        }

        TensorView out_view;
        out_view.data = logits_buffer_.data();
        out_view.shape = {tokenizer_->vocab_size()};

        // Forward pass
        backend_->forward(in_view, out_view);

        if (config.verbose && i % 10 == 0) {
            std::cout << "[gen] prefill progress: " << i << "/"
                      << prompt_tokens.size() << "\n";
        }
    }

    if (config.verbose) {
        std::cout << "[gen] prefill complete\n";
    }
}

// ============================================================================
// DECODE PHASE (autoregressivo)
// ============================================================================

void AutoregressiveGenerator::decode_phase(
    std::vector<int32_t>& output_tokens,
    const GenerationConfig& config
) {
    if (config.verbose) {
        std::cout << "[gen] decode phase: max " << config.max_tokens << " tokens\n";
    }

    int32_t current_token = 0;  // Será setado pelo primeiro sample

    for (int i = 0; i < config.max_tokens; ++i) {
        // 1. Forward pass (usa último token ou logits do prefill)
        if (i > 0) {
            TensorView in_view;
            in_view.data = &current_token;
            in_view.shape = {1};

            TensorView out_view;
            out_view.data = logits_buffer_.data();
            out_view.shape = {tokenizer_->vocab_size()};

            backend_->forward(in_view, out_view);
        }

        // 2. Sample próximo token
        current_token = sampler_->sample(
            logits_buffer_.data(),
            static_cast<int>(tokenizer_->vocab_size())
        );

        // 3. Calcula probabilidade (para stopping criterion)
        float probability = 0.0f;
        if (config.min_probability > 0.0f) {
            // Softmax para obter probabilidade real
            float max_logit = *std::max_element(
                logits_buffer_.begin(),
                logits_buffer_.end()
            );

            float sum = 0.0f;
            for (float logit : logits_buffer_) {
                sum += std::exp(logit - max_logit);
            }

            float logit = logits_buffer_[current_token];
            probability = std::exp(logit - max_logit) / sum;
        }

        // 4. Check stopping criteria
        if (should_stop(current_token, i, probability, config)) {
            break;
        }

        // 5. Add to output
        output_tokens.push_back(current_token);

        // 6. Callback (streaming)
        if (config.stream && config.token_callback) {
            config.token_callback(current_token);
        }

        // 7. Log progress
        if (config.verbose && (i + 1) % 10 == 0) {
            std::cout << "[gen] generated " << (i + 1) << " tokens\n";
        }
    }

    if (config.verbose) {
        std::cout << "[gen] decode complete: " << output_tokens.size()
                  << " tokens generated\n";
    }
}

// ============================================================================
// STOPPING CRITERIA
// ============================================================================

bool AutoregressiveGenerator::should_stop(
    int32_t token,
    int generated_count,
    float probability,
    const GenerationConfig& config
) const {
    // 1. EOS token
    if (token == tokenizer_->eos_token()) {
        const_cast<AutoregressiveGenerator*>(this)->stats_.stop_reason =
            GenerationStats::EOS_TOKEN;
        return true;
    }

    // 2. Stop tokens
    if (!config.stop_tokens.empty()) {
        auto it = std::find(
            config.stop_tokens.begin(),
            config.stop_tokens.end(),
            token
        );
        if (it != config.stop_tokens.end()) {
            const_cast<AutoregressiveGenerator*>(this)->stats_.stop_reason =
                GenerationStats::STOP_TOKEN;
            return true;
        }
    }

    // 3. Min probability
    if (config.min_probability > 0.0f && probability < config.min_probability) {
        const_cast<AutoregressiveGenerator*>(this)->stats_.stop_reason =
            GenerationStats::MIN_PROBABILITY;
        return true;
    }

    // 4. Max tokens
    if (generated_count >= config.max_tokens - 1) {
        const_cast<AutoregressiveGenerator*>(this)->stats_.stop_reason =
            GenerationStats::MAX_TOKENS;
        return true;
    }

    return false;
}

// ============================================================================
// BatchGenerator (STUB - Fase 4)
// ============================================================================

BatchGenerator::BatchGenerator(
    Backend* backend,
    SimpleTokenizer* tokenizer
) : backend_(backend), tokenizer_(tokenizer) {
}

std::vector<BatchGenerationResult> BatchGenerator::generate_batch(
    const std::vector<BatchGenerationRequest>& requests
) {
    // TODO FASE 4: Implementar batch real
    // Por enquanto: sequencial

    std::vector<BatchGenerationResult> results;
    results.reserve(requests.size());

    for (const auto& req : requests) {
        AutoregressiveGenerator gen(backend_, tokenizer_, nullptr);

        std::string output = gen.generate(req.prompt, req.config);

        BatchGenerationResult result;
        result.generated_text = output;
        result.stats = gen.stats();
        result.request_id = req.request_id;

        results.push_back(result);
    }

    return results;
}

} // namespace engine