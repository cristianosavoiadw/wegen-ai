cmake_minimum_required(VERSION 3.20)

project(Engine_LLMs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(engine
        # CLI
        src/cli/main.cpp

        # Core
        src/core/engine.cpp

        # Backend
        src/backend/backend_factory.cpp
        src/backend/cpu/cpu_backend.cpp
        src/backend/cpu/ops.cpp
        src/backend/cpu/dequant.cpp

        # Model
        src/model/gguf_inspector.cpp
        src/model/gguf_loader.cpp
        src/model/quantization_utils.cpp
        src/model/tokenizer.cpp
        src/model/sampler.cpp

        # Scheduler
        src/scheduler/scheduler.cpp

        # Metrics
        src/metrics/power_linux.cpp
)

target_include_directories(engine PRIVATE
        src
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(engine PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -O3
            -march=native
    )
endif()

if (UNIX AND NOT APPLE)
    target_link_libraries(engine PRIVATE pthread)
endif()

# =========================
# TESTES (CI / AGENTS)
# =========================

enable_testing()

add_test(
        NAME engine_runs
        COMMAND engine --help
)
