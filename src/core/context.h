#pragma once

#include <cstdint>

namespace engine {

    struct BackendStats {
        uint64_t tokens_total = 0;
        double exec_time_ms = 0.0;

        // Métricas derivadas
        double tokens_per_sec = 0.0;

        // Energia (quando disponível)
        double watts_avg = 0.0;
        double tokens_per_watt = 0.0;
        double energy_total_joules = 0.0;
    };

} // namespace engine