#pragma once

#include "backend/cpu/ops.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace engine {
namespace ops {
namespace simd {

// ============================================================================
// MATMUL OTIMIZADO
// ============================================================================

// Matmul com tiling e AVX2
void matmul_f32_optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// Matmul com transposição de B (melhora cache)
void matmul_f32_transposed(
    const float* A, const float* B_T, float* C,
    int M, int N, int K
);

// ============================================================================
// DOT PRODUCT OTIMIZADO
// ============================================================================

float dot_product_f32(const float* a, const float* b, int n);

// ============================================================================
// OPERAÇÕES VETORIAIS
// ============================================================================

void add_f32_simd(float* dst, const float* src, int n);
void mul_f32_simd(float* dst, const float* a, const float* b, int n);
void scale_f32_simd(float* dst, const float* src, float scale, int n);

// ============================================================================
// NORMALIZAÇÃO
// ============================================================================

void rms_norm_f32_simd(
    float* out,
    const float* in,
    const float* weight,
    int n,
    float eps = 1e-5f
);

// ============================================================================
// SOFTMAX OTIMIZADO
// ============================================================================

void softmax_f32_simd(float* out, const float* in, int n);

// ============================================================================
// ATIVAÇÕES
// ============================================================================

void silu_f32_simd(float* x, int n);
void gelu_f32_simd(float* x, int n);

// ============================================================================
// ROPE OTIMIZADO
// ============================================================================

void rope_f32_simd(
    float* x,
    const float* freq,
    int seq_len,
    int n_heads,
    int head_dim,
    int pos_offset = 0
);

// ============================================================================
// UTILITIES
// ============================================================================

// Verifica se AVX2 está disponível
bool is_avx2_available();

// Benchmark: compara versões scalar vs SIMD
void benchmark_ops();

} // namespace simd
} // namespace ops
} // namespace engine