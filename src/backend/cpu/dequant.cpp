#include "backend/cpu/ops.h"
#include "backend/cpu/quants.h"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace engine {
namespace ops {

using namespace quants;

// ============================================================================
// DEQUANTIZAÇÃO Q4_K_M
// ============================================================================

void dequantize_q4_k_m(float* dst, const void* src, int n) {
    const int nb = n / QK_K;  // Número de blocos
    
    if (n % QK_K != 0) {
        std::cerr << "[dequant] WARNING: n=" << n << " not multiple of " << QK_K << "\n";
    }
    
    const block_q4_K* blocks = static_cast<const block_q4_K*>(src);
    
    for (int b = 0; b < nb; ++b) {
        const block_q4_K& block = blocks[b];
        float* block_dst = dst + b * QK_K;
        
        // Decodifica delta e dmin
        float d = read_fp16(block.d);
        float dmin = read_fp16(block.dmin);
        
        // Decodifica scales
        // Q4_K usa esquema complexo de scales com 6 bits
        uint8_t scales[8];
        uint8_t mins[8];
        
        // Extrai scales e mins dos 12 bytes
        // Layout dos scales: 6 bits por scale, empacotado
        const uint8_t* sc = block.scales;
        
        // Primeira metade (scales 0-3)
        scales[0] = sc[0] & 0x3F;
        scales[1] = sc[1] & 0x3F;
        scales[2] = sc[2] & 0x3F;
        scales[3] = sc[3] & 0x3F;
        
        // Segunda metade (scales 4-7)
        scales[4] = ((sc[0] >> 6) & 0x03) | ((sc[4] & 0x0F) << 2);
        scales[5] = ((sc[1] >> 6) & 0x03) | ((sc[5] & 0x0F) << 2);
        scales[6] = ((sc[2] >> 6) & 0x03) | ((sc[6] & 0x0F) << 2);
        scales[7] = ((sc[3] >> 6) & 0x03) | ((sc[7] & 0x0F) << 2);
        
        // Mins
        mins[0] = sc[8] & 0x3F;
        mins[1] = sc[9] & 0x3F;
        mins[2] = sc[10] & 0x3F;
        mins[3] = sc[11] & 0x3F;
        
        mins[4] = ((sc[8] >> 6) & 0x03) | ((sc[4] >> 4) << 2);
        mins[5] = ((sc[9] >> 6) & 0x03) | ((sc[5] >> 4) << 2);
        mins[6] = ((sc[10] >> 6) & 0x03) | ((sc[6] >> 4) << 2);
        mins[7] = ((sc[11] >> 6) & 0x03) | ((sc[7] >> 4) << 2);
        
        // Dequantiza os 256 valores
        // Divididos em 8 grupos de 32
        for (int group = 0; group < 8; ++group) {
            const float scale = d * scales[group];
            const float min = dmin * mins[group];
            
            // Cada grupo tem 32 valores (16 bytes de qs)
            const uint8_t* qs_group = block.qs + group * 16;
            
            for (int i = 0; i < 16; ++i) {
                uint8_t packed = qs_group[i];
                
                // 2 valores de 4 bits por byte
                uint8_t q0 = packed & 0x0F;
                uint8_t q1 = (packed >> 4) & 0x0F;
                
                int idx = group * 32 + i * 2;
                
                // Converte 4-bit → float
                block_dst[idx + 0] = scale * q0 - min;
                block_dst[idx + 1] = scale * q1 - min;
            }
        }
    }
}

// ============================================================================
// DEQUANTIZAÇÃO Q8_0 (mais simples, para comparação)
// ============================================================================

void dequantize_q8_0(float* dst, const void* src, int n) {
    const int nb = n / QK8_0;
    const block_q8_0* blocks = static_cast<const block_q8_0*>(src);
    
    for (int b = 0; b < nb; ++b) {
        const block_q8_0& block = blocks[b];
        const float d = block.d;
        
        for (int i = 0; i < QK8_0; ++i) {
            dst[b * QK8_0 + i] = block.qs[i] * d;
        }
    }
}

// ============================================================================
// DEQUANTIZAÇÃO Q6_K
// ============================================================================
void dequantize_q6_k(float* dst, const void* src, int n) {
    const int nb = n / QK_K;

    if (n % QK_K != 0) {
        std::cerr << "[dequant] WARNING: Q6_K n=" << n
                  << " not multiple of " << QK_K << "\n";
    }

    const block_q6_K* blocks = static_cast<const block_q6_K*>(src);

    for (int b = 0; b < nb; ++b) {
        const block_q6_K& block = blocks[b];

        // ✅ d é FP16 no ggml
        const float d = read_fp16(block.d);

        // 16 grupos * 16 valores = 256
        for (int group = 0; group < 16; ++group) {
            const float sc = static_cast<float>(block.scales[group]) * d;

            for (int i = 0; i < 16; ++i) {
                const int idx = group * 16 + i;

                const int ql_idx = idx / 2;
                const int qh_idx = idx / 4;

                const uint8_t ql = block.ql[ql_idx];
                const uint8_t qh = block.qh[qh_idx];

                const int shift_l = (idx % 2) * 4;
                const int shift_h = (idx % 4) * 2;

                const uint8_t q_low  = (ql >> shift_l) & 0x0F;
                const uint8_t q_high = (qh >> shift_h) & 0x03;

                const int q = int(q_low | (q_high << 4)); // 0..63
                dst[b * QK_K + idx] = (float(q) - 32.0f) * sc;
            }
        }
    }
}

// ============================================================================
// DISPATCHER - Detecta tipo e chama dequantização correta
// ============================================================================

    void dequantize_auto(
        float* dst,
        const void* src,
        int n,
        GgmlType type
    ) {
    switch (type) {

        case GgmlType::F32:
            std::memcpy(dst, src, (size_t)n * sizeof(float));
            return;

        case GgmlType::F16: {
            const uint8_t* src_u8 = static_cast<const uint8_t*>(src);
            for (int i = 0; i < n; ++i) {
                dst[i] = read_fp16(src_u8 + (size_t)i * 2);
            }
            return;
        }

        case GgmlType::Q4_K:
            dequantize_q4_k_m(dst, src, n);
            return;

        case GgmlType::Q6_K:
            dequantize_q6_k(dst, src, n);
            return;

        case GgmlType::Q8_0:
            dequantize_q8_0(dst, src, n);
            return;

            // ===== NÃO SUPORTADOS (NÃO MINTA MAPEANDO ERRADO) =====

        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
            std::cerr << "[dequant] Q4_0/Q4_1 not supported (would corrupt)\n";
            std::memset(dst, 0, (size_t)n * sizeof(float));
            return;

        case GgmlType::Q8_1:
            std::cerr << "[dequant] Q8_1 not supported (different layout)\n";
            std::memset(dst, 0, (size_t)n * sizeof(float));
            return;

        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q5_K:
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ2_XS:
            std::cerr << "[dequant] type " << static_cast<int>(type)
                      << " not implemented, returning zeros\n";
            std::memset(dst, 0, (size_t)n * sizeof(float));
            return;

        default:
            std::cerr << "[dequant] unknown type " << static_cast<int>(type)
                      << ", returning zeros\n";
            std::memset(dst, 0, (size_t)n * sizeof(float));
            return;
    }
}

} // namespace ops
} // namespace engine
