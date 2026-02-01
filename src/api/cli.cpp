#include <iostream>
#include <string>
#include <new>

#include "../core/context.h"
#include "../core/engine.h"

using namespace engine;

static const char* ENGINE_VERSION = "0.1.0";

/* ----------------------------
 * Parsers simples
 * ---------------------------- */

BackendType parse_backend(const std::string& v) {
    if (v == "cpu_avx2") return BackendType::CPU_AVX2;
    if (v == "cuda") return BackendType::CUDA;
    throw std::runtime_error("Invalid backend: " + v);
}

QuantizationType parse_quant(const std::string& v) {
    if (v == "q8_0") return QuantizationType::Q8_0;
    if (v == "q6_k") return QuantizationType::Q6_K;
    if (v == "q4_k_m") return QuantizationType::Q4_K_M;
    throw std::runtime_error("Invalid quantization: " + v);
}

/* ----------------------------
 * Main
 * ---------------------------- */

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string cmd = argv[1];

        if (cmd == "--version" || cmd == "version") {
            std::cout << "engine " << ENGINE_VERSION << std::endl;
            return 0;
        }

        if (cmd == "run") {
            ExecutionPlan plan;
            plan.backend = BackendType::CPU_AVX2;
            plan.quantization = QuantizationType::Q8_0;

            /* ----------------------------
             * Parse args
             * ---------------------------- */
            for (int i = 2; i < argc; ++i) {
                std::string arg = argv[i];

                if (arg == "--model" && i + 1 < argc) {
                    plan.model_path = argv[++i];

                } else if (arg == "--backend" && i + 1 < argc) {
                    plan.backend = parse_backend(argv[++i]);

                } else if (arg == "--quant" && i + 1 < argc) {
                    plan.quantization = parse_quant(argv[++i]);

                } else if (arg == "--max-watts" && i + 1 < argc) {
                    plan.limits.max_watts = std::stoul(argv[++i]);

                } else if (arg == "--max-tokens" && i + 1 < argc) {
                    plan.limits.max_tokens = std::stoul(argv[++i]);
                }
            }

            /* ----------------------------
             * Run engine
             * ---------------------------- */
            Engine engine;
            engine.run(plan);
            return 0;
        }
    }

    /* ----------------------------
     * Help
     * ---------------------------- */
    std::cout
        << "Engine_LLMs\n"
        << "Usage:\n"
        << "  engine --version\n"
        << "  engine run --model <path> "
           "[--backend cpu_avx2|cuda] "
           "[--quant q8_0|q6_k|q4_k_m] "
           "[--max-tokens N] "
           "[--max-watts N]\n";

    return 0;
}
