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
    const block_q6_K* blocks = static_cast<const block_q6_K*>(src);
    
    for (int b = 0; b < nb; ++b) {
        const block_q6_K& block = blocks[b];
        const float d = block.d;
        
        // Q6_K divide em 16 grupos de 16 valores
        for (int group = 0; group < 16; ++group) {
            const int8_t scale = block.scales[group];
            
            for (int i = 0; i < 16; ++i) {
                int idx = group * 16 + i;
                
                // Combina lower 4 bits (ql) com upper 2 bits (qh)
                int ql_idx = idx / 2;
                int qh_idx = idx / 4;
                
                uint8_t ql = block.ql[ql_idx];
                uint8_t qh = block.qh[qh_idx];
                
                // Extrai os bits corretos
                int shift_l = (idx % 2) * 4;
                int shift_h = (idx % 4) * 2;
                
                uint8_t q_low = (ql >> shift_l) & 0x0F;
                uint8_t q_high = (qh >> shift_h) & 0x03;
                
                // Combina em 6 bits
                int q = q_low | (q_high << 4);
                
                // Converte para float
                dst[b * QK_K + idx] = (q - 32) * scale * d;
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
            std::memcpy(dst, src, n * sizeof(float));
            break;

        case GgmlType::F16: {
            const uint8_t* src_u8 = static_cast<const uint8_t*>(src);
            for (int i = 0; i < n; ++i) {
                dst[i] = read_fp16(src_u8 + i * 2);
            }
            break;
        }

        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q4_K:    // ← TIPO 12
            dequantize_q4_k_m(dst, src, n);
            break;

        case GgmlType::Q6_K:    // ← TIPO 14
            dequantize_q6_k(dst, src, n);
            break;

        case GgmlType::Q8_0:
        case GgmlType::Q8_1:
            dequantize_q8_0(dst, src, n);
            break;

        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q5_K:
            std::cerr << "[dequant] Q5 not implemented yet, using zeros\n";
            std::memset(dst, 0, n * sizeof(float));
            break;

        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q8_K:
            std::cerr << "[dequant] Type " << static_cast<int>(type)
                      << " not implemented yet, using zeros\n";
            std::memset(dst, 0, n * sizeof(float));
            break;

        default:
            std::cerr << "[dequant] Unknown type: " << static_cast<int>(type) << "\n";
            std::memset(dst, 0, n * sizeof(float));
            break;
    }
}

} // namespace ops
} // namespace engine
