#pragma once

#include <cstdint>
#include <cstring>

namespace engine {
namespace quants {

// Constantes
constexpr int QK_K = 256;

// Estruturas
struct block_q4_K {
    uint8_t d[2];
    uint8_t dmin[2];
    uint8_t scales[12];
    uint8_t qs[QK_K/2];
};

static_assert(sizeof(block_q4_K) == 2 + 2 + 12 + 128, "Wrong Q4_K block size");

constexpr int QK8_0 = 32;

struct block_q8_0 {
    float d;
    int8_t qs[QK8_0];
};

static_assert(sizeof(block_q8_0) == 4 + 32, "Wrong Q8_0 block size");

struct block_q6_K {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    float d;
};

static_assert(sizeof(block_q6_K) == 128 + 64 + 16 + 4, "Wrong Q6_K block size");

// FP16 → FP32 conversão (CORRIGIDA)
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp_mant = (h & 0x7fff) << 13;
    uint32_t exp = exp_mant & 0x7f800000;

    uint32_t result;

    if (exp == 0x7f800000) {
        // Inf/NaN
        result = sign | 0x7f800000 | (exp_mant & 0x007fffff);
    } else if (exp == 0) {
        // Zero ou denormal
        if (exp_mant == 0) {
            result = sign;  // Zero
        } else {
            // Denormal - normaliza
            const uint32_t magic = 0x3f000000;
            float tmp;
            std::memcpy(&tmp, &magic, 4);

            uint32_t tmp_bits;
            std::memcpy(&tmp_bits, &tmp, 4);
            tmp_bits = magic | (exp_mant >> 13);
            std::memcpy(&tmp, &tmp_bits, 4);

            tmp -= 0.5f;
            std::memcpy(&result, &tmp, 4);
            result |= sign;
        }
    } else {
        // Normal
        result = sign | (exp_mant + 0x38000000);
    }

    float out;
    std::memcpy(&out, &result, 4);
    return out;
}

inline float read_fp16(const uint8_t* data) {
    uint16_t h = data[0] | (data[1] << 8);
    return fp16_to_fp32(h);
}

} // namespace quants
} // namespace engine