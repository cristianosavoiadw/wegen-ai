# Engine de Execução de Modelos — Stack Tecnológico v0.1

## 1. Objetivo do Documento

Este documento define **o stack tecnológico oficial do projeto**, incluindo:

- linguagens
- toolchain
- dependências permitidas
- ambientes suportados

O objetivo é:
- evitar decisões ad-hoc
- garantir reprodutibilidade
- manter foco em performance e energia

Tudo fora deste stack é considerado **fora de padrão**.

---

## 2. Princípios do Stack

O stack segue os seguintes princípios:

- **Infra-first**: foco em runtime, não em UX
- **Baixo nível explícito**: controle total de custo
- **Poucas dependências**: cada lib precisa justificar sua existência
- **Portabilidade Linux**: ambiente alvo é produção real

---

## 3. Linguagens Oficiais

### 3.1 Linguagem Principal (Core)

**C++20**

Motivos:
- controle de memória
- SIMD e vetorização
- acesso direto a hardware
- maturidade para runtimes

Todo código do core do motor DEVE ser escrito em C++.

---

### 3.2 Linguagens Auxiliares

#### Python 3.10+
Uso permitido apenas para:
- ferramentas
- scripts de benchmark
- SDKs externos
- integração com WeOS

Python **NÃO** pode:
- fazer parte do run loop
- executar ops
- controlar backend

---

#### Rust (opcional, futuro)

Pode ser usado para:
- scheduler avançado
- control plane
- tooling

Não obrigatório no v0.1.

---

## 4. Sistema Operacional

### Plataforma alvo

- Linux x86_64
- Ubuntu 22.04 LTS (referência)

### Ambiente de desenvolvimento

- WSL2 (Ubuntu)
- VPS Linux para benchmarks reais

---

## 5. Toolchain de Build

### 5.1 Compilador

- `clang` (recomendado)
- `gcc` (alternativo)

Requisitos:
- suporte completo a C++20
- suporte a AVX2

---

### 5.2 Build System

- **CMake**
- **Ninja** (recomendado)

Motivos:
- portabilidade
- builds rápidos
- integração com CI

---

### 5.3 Linker

- `lld` (preferencial)
- `ld` (fallback)

---

## 6. Dependências Permitidas

### 6.1 Logging

- `spdlog`

Requisitos:
- logging síncrono e assíncrono
- baixo overhead

---

### 6.2 Serialização

Uma das opções:
- JSON manual (v0.1)
- Protobuf ou FlatBuffers (futuro)

Nenhuma dependência pesada é obrigatória no início.

---

### 6.3 Benchmark e Profiling

- `google-benchmark` (benchmarks)
- `perf` (Linux)
- `htop` / `top`

---

## 7. Backend e Hardware

### 7.1 CPU

- AVX2 obrigatório
- AVX512 opcional
- arquitetura x86_64

---

### 7.2 GPU (fora do v0.1)

Planejado:
- CUDA (NVIDIA)
- NVML para energia
- ROCm (futuro)

---

## 8. Medição de Energia

### CPU
- RAPL (`/sys/class/powercap`)
- leitura direta do sistema

### GPU (futuro)
- NVML
- ROCm SMI

Energia é **métrica de primeira classe**.

---

## 9. IDEs e Ferramentas de Desenvolvimento

### IDEs recomendadas
- CLion
- VS Code + clangd
- PyCharm (para tooling e Python)

Nenhuma IDE é obrigatória.

---

## 10. Ferramentas Proibidas no Core

Não podem ser usadas no core do motor:

- PyTorch
- TensorFlow
- JAX
- Triton
- ONNX Runtime
- Ray
- Kubernetes SDKs

Essas ferramentas não atendem aos requisitos de determinismo e custo.

---

## 11. Versionamento

- Git como VCS
- versionamento semântico (`MAJOR.MINOR.PATCH`)
- versão do engine exposta em runtime

---

## 12. Stack de Integração com WeOS

- API local (HTTP ou gRPC)
- JSON como formato inicial
- autenticação e governança feitas no WeOS

O motor não implementa lógica de identidade.

---

## 13. Resumo do Stack

```text
Core:        C++20
Build:       CMake + Ninja
OS:          Linux (Ubuntu 22.04)
CPU:         x86_64 AVX2
Energy:      RAPL
CLI/API:     C++ (HTTP local)
Tooling:     Python 3.10+
