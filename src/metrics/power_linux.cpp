#include "./power_linux.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace engine {

std::optional<uint64_t> PowerLinux::read_u64_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return std::nullopt;

    uint64_t v = 0;
    f >> v;
    if (f.fail()) return std::nullopt;

    return v;
}

bool PowerLinux::init() {
    // Busca por qualquer energy_uj em /sys/class/powercap
    // Exemplos comuns:
    //   /sys/class/powercap/intel-rapl:0/energy_uj
    //   /sys/class/powercap/intel-rapl:0:0/energy_uj
    //   /sys/class/powercap/amd-rapl:0/energy_uj (depende do kernel/driver)
    const fs::path base{"/sys/class/powercap"};

    if (!fs::exists(base) || !fs::is_directory(base)) {
        return false;
    }

    std::vector<std::string> candidates;

    // Coleta todos os arquivos energy_uj dentro da árvore
    for (const auto& entry : fs::recursive_directory_iterator(base)) {
        if (!entry.is_regular_file()) continue;
        const auto p = entry.path();
        if (p.filename() == "energy_uj") {
            candidates.push_back(p.string());
        }
    }

    // Heurística simples: pegar o primeiro candidato que consegue ler.
    // (No futuro: priorizar package-level vs core-level, etc.)
    for (const auto& c : candidates) {
        auto v = read_u64_file(c);
        if (v.has_value()) {
            energy_uj_path_ = c;
            return true;
        }
    }

    return false;
}

std::optional<double> PowerLinux::read_joules() const {
    if (energy_uj_path_.empty()) return std::nullopt;

    auto uj = read_u64_file(energy_uj_path_);
    if (!uj.has_value()) return std::nullopt;

    // microjoules -> joules
    return static_cast<double>(*uj) / 1'000'000.0;
}

} // namespace engine
