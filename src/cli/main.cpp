#include <iostream>
#include <string>

#include "core/engine.h"
#include "core/execution_plan.h"
#include "core/version.h"
#include "model/quantization_utils.h"
#include "model/sampler.h"
#include "scheduler/scheduler.h"
#include "backend/cpu/cpu_backend.h"

static void print_usage() {
    std::cerr <<
        "Usage:\n"
        "  engine run --model <path> [options]\n"
        "  engine generate --model <path> --prompt <text> [options]\n"
        "  engine scheduler --model <path> [options]\n"
        "  engine --version\n\n"
        "Options:\n"
        "  --model <path>        Path to GGUF model\n"
        "  --prompt <text>       Prompt for generation\n"
        "  --max-tokens <n>      Max tokens (default: 16)\n"
        "  --backend <type>      Backend type (default: cpu)\n"
        "  --temperature <f>     Sampling temperature (default: 1.0)\n"
        "  --top-k <n>           Top-k sampling (default: 40)\n"
        "  --top-p <f>           Top-p sampling (default: 0.95)\n";
}

static bool parse_common_args(
    int argc,
    char** argv,
    std::string& model_path,
    core::ExecutionPlan& plan
) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--max-tokens" && i + 1 < argc) {
            plan.max_tokens = std::stoul(argv[++i]);
        }
        else if (arg == "--backend" && i + 1 < argc) {
            plan.backend = argv[++i];
        }
    }

    return !model_path.empty();
}

static engine::SamplingConfig parse_sampling_args(int argc, char** argv) {
    engine::SamplingConfig config;
    config.strategy = engine::SamplingStrategy::TEMPERATURE;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        }
        else if (arg == "--top-k" && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
            config.strategy = engine::SamplingStrategy::TOP_K;
        }
        else if (arg == "--top-p" && i + 1 < argc) {
            config.top_p = std::stof(argv[++i]);
            config.strategy = engine::SamplingStrategy::TOP_P;
        }
    }

    return config;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];
    std::string model_path;

    core::ExecutionPlan plan;
    plan.backend = "cpu";
    plan.max_tokens = 16;
    plan.scheduler_policy = "default";
    plan.quant_policy = core::QuantizationPolicy::USE_MODEL_NATIVE;
    plan.quantization = engine::QuantizationType::Q4_K_M;
    plan.streaming = true;

    /* ───────────────────────────────────────────── */
    if (command == "--version") {
        std::cout << "Engine_LLMs " << ENGINE_VERSION << "\n";
        return 0;
    }

    /* ───────────────────────────────────────────── */
    if (command == "run") {
        if (!parse_common_args(argc, argv, model_path, plan)) {
            print_usage();
            return 2;
        }

        engine::Engine engine;
        engine.run(model_path, plan);
        return 0;
    }

    /* ───────────────────────────────────────────── */
    if (command == "generate") {
        if (!parse_common_args(argc, argv, model_path, plan)) {
            print_usage();
            return 2;
        }

        // Extrai prompt
        std::string prompt;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
                prompt = argv[++i];
                break;
            }
        }

        if (prompt.empty()) {
            std::cerr << "Error: --prompt is required for generate command\n";
            return 2;
        }

        // Parsing de sampling
        auto sampling_config = parse_sampling_args(argc, argv);

        // Cria backend diretamente para usar generate()
        engine::CpuBackend backend;
        backend.init();
        backend.load_model(model_path);

        // Gera texto
        std::string result = backend.generate(prompt, plan.max_tokens, sampling_config);

        // Output
        std::cout << "\n=== Generated Text ===\n";
        std::cout << result << "\n";
        std::cout << "======================\n\n";

        // Estatísticas
        auto stats = backend.stats();
        std::cout << "Statistics:\n";
        std::cout << "  Tokens: " << stats.tokens_total << "\n";
        std::cout << "  Time: " << stats.exec_time_ms << " ms\n";
        std::cout << "  Tokens/sec: " << stats.tokens_per_sec << "\n";

        if (stats.watts_avg > 0) {
            std::cout << "  Power: " << stats.watts_avg << " W\n";
            std::cout << "  Tokens/Watt: " << stats.tokens_per_watt << "\n";
        }

        return 0;
    }

    /* ───────────────────────────────────────────── */
    if (command == "scheduler") {
        if (!parse_common_args(argc, argv, model_path, plan)) {
            print_usage();
            return 2;
        }

        engine::Scheduler scheduler;

        core::ExecutionPlan p1 = plan;
        core::ExecutionPlan p2 = plan;
        p2.max_tokens = plan.max_tokens * 2;

        scheduler.submit(p1, 1);
        scheduler.submit(p2, 10);

        while (!scheduler.empty()) {
            scheduler.run_batch();
        }

        return 0;
    }

    print_usage();
    return 1;
}
