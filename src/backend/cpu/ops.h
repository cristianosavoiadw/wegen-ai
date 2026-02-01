#pragma once

#include <cstdint>
#include <cmath>
#include "model/gguf_loader.h"  // Para GgmlType

namespace engine {
namespace ops {

// ============================================================================
// OPERAÇÕES BÁSICAS
// ============================================================================

void matmul_f32(const float* A, const float* B, float* C, int M, int N, int K);
void add_f32(float* dst, const float* src, int n);
void mul_f32(float* dst, const float* a, const float* b, int n);
void copy_f32(float* dst, const float* src, int n);
void fill_f32(float* dst, float value, int n);

// ============================================================================
// NORMALIZAÇÃO
// ============================================================================

void rms_norm_f32(
    float* out,
    const float* in,
    const float* weight,
    int n,
    float eps = 1e-5f
);

// ============================================================================
// ATENÇÃO
// ============================================================================

void softmax_f32(float* out, const float* in, int n);
void softmax_inplace_f32(float* x, int n);

void attention_f32(
    float* out,
    const float* Q,
    const float* K,
    const float* V,
    int seq_len,
    int dim
);

// ============================================================================
// ROTARY POSITION EMBEDDING (RoPE)
// ============================================================================

void rope_f32(
    float* x,
    const float* freq,
    int seq_len,
    int n_heads,
    int head_dim,
    int pos_offset = 0
);

// ============================================================================
// ATIVAÇÕES
// ============================================================================

void silu_f32(float* x, int n);
void gelu_f32(float* x, int n);

// ============================================================================
// DEQUANTIZAÇÃO
// ============================================================================

// Dequantiza Q4_K_M → F32
void dequantize_q4_k_m(float* dst, const void* src, int n);

// Dequantiza Q8_0 → F32
void dequantize_q8_0(float* dst, const void* src, int n);

// Dequantiza Q6_K → F32
void dequantize_q6_k(float* dst, const void* src, int n);

// Dispatcher automático baseado no tipo
void dequantize_auto(
    float* dst,
    const void* src,
    int n,
    GgmlType type
);

} // namespace ops
} // namespace engine
