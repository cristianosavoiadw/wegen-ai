#include "model/sampler.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace engine {

Sampler::Sampler(const SamplingConfig& config)
    : config_(config), rng_(config.seed) {}

int32_t Sampler::sample(const float* logits, int vocab_size) {
    switch (config_.strategy) {
        case SamplingStrategy::GREEDY:
            return sample_greedy(logits, vocab_size);
        
        case SamplingStrategy::TEMPERATURE:
            return sample_temperature(logits, vocab_size);
        
        case SamplingStrategy::TOP_K:
            return sample_top_k(logits, vocab_size);
        
        case SamplingStrategy::TOP_P:
            return sample_top_p(logits, vocab_size);
        
        default:
            return sample_greedy(logits, vocab_size);
    }
}

int32_t Sampler::sample(const std::vector<float>& logits) {
    return sample(logits.data(), static_cast<int>(logits.size()));
}

// ============================================================================
// GREEDY SAMPLING
// ============================================================================

int32_t Sampler::sample_greedy(const float* logits, int vocab_size) {
    // Retorna índice do maior logit
    int32_t max_idx = 0;
    float max_val = logits[0];
    
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// ============================================================================
// TEMPERATURE SAMPLING
// ============================================================================

int32_t Sampler::sample_temperature(const float* logits, int vocab_size) {
    std::vector<float> probs(vocab_size);
    softmax_with_temperature(probs, logits, vocab_size);
    
    // Sampling categórico
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}

// ============================================================================
// TOP-K SAMPLING
// ============================================================================

int32_t Sampler::sample_top_k(const float* logits, int vocab_size) {
    // Cria pares (logit, índice)
    std::vector<std::pair<float, int32_t>> logit_pairs;
    logit_pairs.reserve(vocab_size);
    
    for (int i = 0; i < vocab_size; ++i) {
        logit_pairs.emplace_back(logits[i], i);
    }
    
    // Ordena por logit (decrescente)
    std::partial_sort(
        logit_pairs.begin(),
        logit_pairs.begin() + std::min(config_.top_k, vocab_size),
        logit_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Mantém apenas top-k
    int k = std::min(config_.top_k, vocab_size);
    logit_pairs.resize(k);
    
    // Softmax nos top-k
    std::vector<float> top_k_logits(k);
    for (int i = 0; i < k; ++i) {
        top_k_logits[i] = logit_pairs[i].first;
    }
    
    std::vector<float> probs(k);
    softmax_with_temperature(probs, top_k_logits.data(), k);
    
    // Sampling
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    int sampled_idx = dist(rng_);
    
    return logit_pairs[sampled_idx].second;
}

// ============================================================================
// TOP-P (NUCLEUS) SAMPLING
// ============================================================================

int32_t Sampler::sample_top_p(const float* logits, int vocab_size) {
    // Cria pares (probabilidade, índice)
    std::vector<float> probs(vocab_size);
    softmax_with_temperature(probs, logits, vocab_size);
    
    std::vector<std::pair<float, int32_t>> prob_pairs;
    prob_pairs.reserve(vocab_size);
    
    for (int i = 0; i < vocab_size; ++i) {
        prob_pairs.emplace_back(probs[i], i);
    }
    
    // Ordena por probabilidade (decrescente)
    std::sort(
        prob_pairs.begin(),
        prob_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Acumula probabilidades até atingir top_p
    float cumsum = 0.0f;
    int nucleus_size = 0;
    
    for (const auto& [prob, idx] : prob_pairs) {
        cumsum += prob;
        nucleus_size++;
        
        if (cumsum >= config_.top_p) {
            break;
        }
    }
    
    // Renormaliza núcleo
    prob_pairs.resize(nucleus_size);
    
    std::vector<float> nucleus_probs;
    nucleus_probs.reserve(nucleus_size);
    
    for (const auto& [prob, idx] : prob_pairs) {
        nucleus_probs.push_back(prob);
    }
    
    // Sampling
    std::discrete_distribution<int32_t> dist(nucleus_probs.begin(), nucleus_probs.end());
    int sampled_idx = dist(rng_);
    
    return prob_pairs[sampled_idx].second;
}

// ============================================================================
// SOFTMAX COM TEMPERATURA
// ============================================================================

void Sampler::softmax_with_temperature(
    std::vector<float>& probs,
    const float* logits,
    int vocab_size
) {
    probs.resize(vocab_size);
    
    // Divide por temperatura
    float inv_temp = 1.0f / config_.temperature;
    
    // Encontra max (estabilidade numérica)
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    
    // exp((logit - max) / T)
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) * inv_temp);
        sum += probs[i];
    }
    
    // Normaliza
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] *= inv_sum;
    }
}

} // namespace engine
