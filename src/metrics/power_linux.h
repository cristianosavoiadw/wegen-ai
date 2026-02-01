#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace engine {

/**
 * Medidor de energia via Linux powercap (RAPL-like).
 *
 * Fonte típica (exemplos):
 *   /sys/class/powercap/intel-rapl:0/energy_uj
 *   /sys/class/powercap/intel-rapl:0:0/energy_uj
 *
 * Unidades:
 *   energy_uj => microjoules
 */
class PowerLinux {
public:
    PowerLinux() = default;

    // Inicializa o medidor (descobre um energy_uj válido).
    bool init();

    // Lê energia acumulada (em Joules).
    std::optional<double> read_joules() const;

    // Caminho selecionado (para debug).
    const std::string& energy_path() const { return energy_uj_path_; }

private:
    std::string energy_uj_path_;

    static std::optional<uint64_t> read_u64_file(const std::string& path);
};

} // namespace engine
