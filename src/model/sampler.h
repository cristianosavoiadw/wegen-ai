#pragma once

#include <vector>
#include <cstdint>
#include <random>

namespace engine {

    // Estratégias de sampling
    enum class SamplingStrategy {
        GREEDY,       // Escolhe token com maior probabilidade
        TEMPERATURE,  // Sampling com temperatura
        TOP_K,        // Sampling top-k
        TOP_P         // Nucleus sampling (top-p)
    };

    struct SamplingConfig {
        SamplingStrategy strategy = SamplingStrategy::GREEDY;
        float temperature = 1.0f;
        int top_k = 40;
        float top_p = 0.95f;
        uint32_t seed = 42;
    };

    class Sampler {
    public:
        explicit Sampler(const SamplingConfig& config = {});

        // Escolhe próximo token baseado em logits
        // logits: [vocab_size]
        int32_t sample(const float* logits, int vocab_size);

        // Variante com vector
        int32_t sample(const std::vector<float>& logits);

    private:
        SamplingConfig config_;
        std::mt19937 rng_;

        // Implementações de sampling
        int32_t sample_greedy(const float* logits, int vocab_size);
        int32_t sample_temperature(const float* logits, int vocab_size);
        int32_t sample_top_k(const float* logits, int vocab_size);
        int32_t sample_top_p(const float* logits, int vocab_size);

        // Converte logits → probabilidades com temperatura
        void softmax_with_temperature(
            std::vector<float>& probs,
            const float* logits,
            int vocab_size
        );
    };

} // namespace engine