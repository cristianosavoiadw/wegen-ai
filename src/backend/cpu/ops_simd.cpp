#include "ops_simd.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace engine {
namespace ops {
namespace simd {

// ============================================================================
// CONSTANTES
// ============================================================================

constexpr int TILE_SIZE = 64;  // Tamanho do tile para matmul
constexpr int SIMD_WIDTH = 8;   // AVX2: 8 floats por vez

// ============================================================================
// MATMUL OTIMIZADO
// ============================================================================

void matmul_f32_optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
#ifdef __AVX2__
    // Tiling para melhor uso de cache
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;

    // Zero output
    std::memset(C, 0, M * N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += BM) {
        for (int j0 = 0; j0 < N; j0 += BN) {
            for (int k0 = 0; k0 < K; k0 += BK) {
                // Process tile
                int i_max = std::min(i0 + BM, M);
                int j_max = std::min(j0 + BN, N);
                int k_max = std::min(k0 + BK, K);

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; j += 8) {
                        // Acumula 8 elementos por vez
                        __m256 sum = _mm256_setzero_ps();

                        for (int k = k0; k < k_max; ++k) {
                            __m256 a_val = _mm256_broadcast_ss(&A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            sum = _mm256_fmadd_ps(a_val, b_vec, sum);
                        }

                        // Acumula resultado
                        __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                        c_vec = _mm256_add_ps(c_vec, sum);
                        _mm256_storeu_ps(&C[i * N + j], c_vec);
                    }
                }
            }
        }
    }
#else
    // Fallback scalar
    ops::matmul_f32(A, B, C, M, N, K);
#endif
}

// ============================================================================
// DOT PRODUCT
// ============================================================================

float dot_product_f32(const float* a, const float* b, int n) {
#ifdef __AVX2__
    __m256 sum_vec = _mm256_setzero_ps();

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // Horizontal sum
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    // Tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

// ============================================================================
// OPERAÇÕES VETORIAIS
// ============================================================================

void add_f32_simd(float* dst, const float* src, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 dst_vec = _mm256_loadu_ps(&dst[i]);
        __m256 src_vec = _mm256_loadu_ps(&src[i]);
        __m256 result = _mm256_add_ps(dst_vec, src_vec);
        _mm256_storeu_ps(&dst[i], result);
    }

    // Tail
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
#else
    ops::add_f32(dst, src, n);
#endif
}

void mul_f32_simd(float* dst, const float* a, const float* b, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&dst[i], result);
    }

    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
#else
    ops::mul_f32(dst, a, b, n);
#endif
}

void scale_f32_simd(float* dst, const float* src, float scale, int n) {
#ifdef __AVX2__
    __m256 scale_vec = _mm256_set1_ps(scale);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 src_vec = _mm256_loadu_ps(&src[i]);
        __m256 result = _mm256_mul_ps(src_vec, scale_vec);
        _mm256_storeu_ps(&dst[i], result);
    }

    for (; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
#else
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
#endif
}

// ============================================================================
// RMS NORM
// ============================================================================

void rms_norm_f32_simd(
    float* out,
    const float* in,
    const float* weight,
    int n,
    float eps
) {
#ifdef __AVX2__
    // Calcula sum of squares
    __m256 sum_sq_vec = _mm256_setzero_ps();

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&in[i]);
        sum_sq_vec = _mm256_fmadd_ps(in_vec, in_vec, sum_sq_vec);
    }

    // Horizontal sum
    float sum_sq_array[8];
    _mm256_storeu_ps(sum_sq_array, sum_sq_vec);
    float sum_sq = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3] +
                   sum_sq_array[4] + sum_sq_array[5] + sum_sq_array[6] + sum_sq_array[7];

    // Tail
    for (; i < n; ++i) {
        sum_sq += in[i] * in[i];
    }

    // RMS
    float rms = std::sqrt(sum_sq / n + eps);
    float scale = 1.0f / rms;
    __m256 scale_vec = _mm256_set1_ps(scale);

    // Normalize and apply weight
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&in[i]);
        __m256 weight_vec = _mm256_loadu_ps(&weight[i]);
        __m256 norm = _mm256_mul_ps(in_vec, scale_vec);
        __m256 result = _mm256_mul_ps(norm, weight_vec);
        _mm256_storeu_ps(&out[i], result);
    }

    for (; i < n; ++i) {
        out[i] = in[i] * scale * weight[i];
    }
#else
    ops::rms_norm_f32(out, in, weight, n, eps);
#endif
}

// ============================================================================
// SOFTMAX
// ============================================================================

void softmax_f32_simd(float* out, const float* in, int n) {
#ifdef __AVX2__
    // Find max
    __m256 max_vec = _mm256_set1_ps(-INFINITY);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&in[i]);
        max_vec = _mm256_max_ps(max_vec, in_vec);
    }

    float max_array[8];
    _mm256_storeu_ps(max_array, max_vec);
    float max_val = std::max({
        max_array[0], max_array[1], max_array[2], max_array[3],
        max_array[4], max_array[5], max_array[6], max_array[7]
    });

    for (; i < n; ++i) {
        max_val = std::max(max_val, in[i]);
    }

    // exp(x - max) and sum
    __m256 max_vec_broadcast = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();

    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&in[i]);
        __m256 diff = _mm256_sub_ps(in_vec, max_vec_broadcast);

        // exp approximation (or use precise exp)
        float exp_vals[8];
        _mm256_storeu_ps(exp_vals, diff);
        for (int j = 0; j < 8; ++j) {
            exp_vals[j] = std::exp(exp_vals[j]);
        }
        __m256 exp_vec = _mm256_loadu_ps(exp_vals);

        _mm256_storeu_ps(&out[i], exp_vec);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }

    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    for (; i < n; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&out[i]);
        __m256 result = _mm256_mul_ps(out_vec, inv_sum_vec);
        _mm256_storeu_ps(&out[i], result);
    }

    for (; i < n; ++i) {
        out[i] *= inv_sum;
    }
#else
    ops::softmax_f32(out, in, n);
#endif
}

// ============================================================================
// SILU
// ============================================================================

void silu_f32_simd(float* x, int n) {
    // SiLU = x / (1 + exp(-x))
    // Para AVX2, precisamos de exp aproximado ou usar scalar

    for (int i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ============================================================================
// ROPE (complexo, versão scalar por enquanto)
// ============================================================================

void rope_f32_simd(
    float* x,
    const float* freq,
    int seq_len,
    int n_heads,
    int head_dim,
    int pos_offset
) {
    // TODO: Implementar versão SIMD
    ops::rope_f32(x, freq, seq_len, n_heads, head_dim, pos_offset);
}

// ============================================================================
// UTILITIES
// ============================================================================

bool is_avx2_available() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

void benchmark_ops() {
    std::cout << "=== SIMD Benchmark ===\n";
    std::cout << "AVX2 available: " << (is_avx2_available() ? "YES" : "NO") << "\n";

    // TODO: Implementar benchmarks
}

} // namespace simd
} // namespace ops
} // namespace engine//
// Created by wegen on 2/1/26.
//