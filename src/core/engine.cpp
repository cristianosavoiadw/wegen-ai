#include "./engine.h"
#include "../core/execution_plan.h"
#include "../backend/backend_factory.h"
#include "../backend/backend.h"
#include "../backend/tensor.h"

#include <iostream>

namespace engine {

void Engine::run(const std::string& model_path, const core::ExecutionPlan& plan) {
    // ─────────────────────────────────────────────
    // Banner do produto (antes de qualquer backend)
    // ─────────────────────────────────────────────
    std::cout << "Iniciando WeOS...\n";

    // logs técnicos (opcional manter)
    std::cout << "[weos] backend: " << plan.backend << "\n";
    std::cout << "[weos] max_tokens: " << plan.max_tokens << "\n";

    auto backend = BackendFactory::create(plan);

    backend->init();
    auto model_info = backend->load_model(model_path);


    std::cout << "[engine] model context: " << model_info.context_length << "\n";
    std::cout << "[engine] model embedding: " << model_info.embedding_dim << "\n";

    TensorView in{};
    TensorView out{};

    for (uint32_t i = 0; i < plan.max_tokens; ++i) {
        backend->forward(in, out);
    }

    auto stats = backend->stats();
    std::cout << "[engine] execution complete\n";
    std::cout << "{ \"tokens\": " << stats.tokens_total
              << ", \"exec_time_ms\": " << stats.exec_time_ms << " }\n";
}

} // namespace engine
