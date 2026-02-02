#include "backend/cpu/ops.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace engine {
namespace ops {

// ============================================================================
// MATMUL
// ============================================================================

void matmul_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // Implementação básica (não otimizada)
    // Versão futura: usar BLAS ou vetorização AVX2
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// OPERAÇÕES BÁSICAS
// ============================================================================

void add_f32(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void mul_f32(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

void copy_f32(float* dst, const float* src, int n) {
    std::memcpy(dst, src, n * sizeof(float));
}

void fill_f32(float* dst, float value, size_t n) {
    std::fill(dst, dst + n, value);
}

// ============================================================================
// RMS NORM
// ============================================================================

void rms_norm_f32(
    float* out,
    const float* in,
    const float* weight,
    int n,
    float eps
) {
    // Calcula RMS
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += in[i] * in[i];
    }
    
    float rms = std::sqrt(sum_sq / n + eps);
    float scale = 1.0f / rms;
    
    // Normaliza e aplica weight
    for (int i = 0; i < n; ++i) {
        out[i] = in[i] * scale * weight[i];
    }
}

// ============================================================================
// SOFTMAX
// ============================================================================

void softmax_f32(float* out, const float* in, int n) {
    // Encontra o máximo (estabilidade numérica)
    float max_val = in[0];
    for (int i = 1; i < n; ++i) {
        max_val = std::max(max_val, in[i]);
    }
    
    // exp(x - max) e soma
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }
    
    // Normaliza
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        out[i] *= inv_sum;
    }
}

void softmax_inplace_f32(float* x, int n) {
    softmax_f32(x, x, n);
}

// ============================================================================
// ATTENTION (simplificado)
// ============================================================================

void attention_f32(
    float* out,
    const float* Q,
    const float* K, 
    const float* V,
    int seq_len,
    int dim
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    
    // Buffer para scores [seq_len, seq_len]
    auto scores = new float[seq_len * seq_len];
    
    // Q @ K^T / sqrt(d)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < dim; ++k) {
                dot += Q[i * dim + k] * K[j * dim + k];
            }
            scores[i * seq_len + j] = dot * scale;
        }
        
        // Softmax na linha i
        softmax_inplace_f32(&scores[i * seq_len], seq_len);
    }
    
    // scores @ V
    for (int i = 0; i < seq_len; ++i) {
        for (int k = 0; k < dim; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                sum += scores[i * seq_len + j] * V[j * dim + k];
            }
            out[i * dim + k] = sum;
        }
    }
    
    delete[] scores;
}

// ============================================================================
// ROPE
// ============================================================================

void rope_f32(
    float* x,
    const float* freq,
    int seq_len,
    int n_heads,
    int head_dim,
    int pos_offset
) {
    const int half_dim = head_dim / 2;
    
    for (int pos = 0; pos < seq_len; ++pos) {
        const int actual_pos = pos + pos_offset;
        
        for (int h = 0; h < n_heads; ++h) {
            float* head = x + (pos * n_heads + h) * head_dim;
            
            for (int d = 0; d < half_dim; ++d) {
                float theta = actual_pos * freq[d];
                float cos_theta = std::cos(theta);
                float sin_theta = std::sin(theta);
                
                float x0 = head[d];
                float x1 = head[d + half_dim];
                
                head[d]            = x0 * cos_theta - x1 * sin_theta;
                head[d + half_dim] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

// ============================================================================
// ATIVAÇÕES
// ============================================================================

void silu_f32(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void gelu_f32(float* x, int n) {
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        float cube = v * v * v;
        x[i] = 0.5f * v * (1.0f + std::tanh(sqrt_2_pi * (v + 0.044715f * cube)));
    }
}

} // namespace ops
} // namespace engine
