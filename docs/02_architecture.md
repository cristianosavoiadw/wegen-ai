# Engine de Execução de Modelos — Arquitetura v0.1

## 1. Objetivo da Arquitetura

Esta arquitetura define a estrutura interna do motor de execução de modelos, garantindo:

- separação clara de responsabilidades
- determinismo operacional
- extensibilidade por contrato
- controle explícito de decisões de execução

A arquitetura **não otimiza prematuramente**, mas **não bloqueia otimizações futuras**.

---

## 2. Visão Geral em Camadas

┌────────────────────────────┐
│ CLI / API                  │
├────────────────────────────┤
│ Control Plane              │
├────────────────────────────┤
│ Scheduler                  │
├────────────────────────────┤
│ Execution Engine           │
├────────────────────────────┤
│ Backend Drivers            │
├────────────────────────────┤
│ Hardware                   │
└────────────────────────────┘



Cada camada possui **contratos explícitos** e não acessa camadas inferiores fora desses contratos.


engine/
├── CMakeLists.txt
├── README.md
│
├── docs/
│   ├── 00_overview.md
│   ├── 01_requirements.md
│   ├── 02_architecture.md
│   ├── 03_execution_flow.md
│   ├── 04_metrics.md
│   └── 05_weos_integration.md
│
├── src/
│   ├── core/
│   │   ├── engine.h
│   │   ├── engine.cpp
│   │   ├── run_loop.cpp
│   │   └── context.h
│   │
│   ├── scheduler/
│   │   ├── scheduler.h
│   │   ├── simple_scheduler.cpp
│   │   └── policies/
│   │
│   ├── model/
│   │   ├── model_loader.cpp    
(src/model/gguf_inspector.h
src/model/gguf_inspector.cpp
src/model/quantization_utils.h
src/model/quantization_utils.cpp
)
│   │   ├── gguf_parser.cpp
│   │   └── quantization.h
│   │
│   ├── memory/
│   │   ├── kv_cache.cpp
│   │   ├── arena.cpp
│   │   └── memory_stats.cpp
│   │
│   ├── backend/
│   │   ├── backend.h (novos arquvivos -> backend_factory.h + backend_factory.cpp (add)
│   │   ├── cpu/
│   │   │   ├── cpu_backend.cpp
│   │   │   └── ops_avx2.cpp
│   │   └── cuda/
│   │       └── cuda_backend.cpp
│   │
│   ├── metrics/
│   │   ├── metrics.h
│   │   ├── power_linux.cpp
│   │   └── profiler.cpp
│   │
│   └── api/
│       ├── cli.cpp
│       └── server.cpp
│
├── tests/
│   ├── unit/
│   └── benchmarks/
│
└── tools/
    ├── benchmark_runner/
    └── model_inspector/


---

## 3. Componentes Principais

### 3.1 CLI / API

Responsabilidades:
- Receber comandos de execução
- Validar entrada do usuário
- Traduzir entrada em um **Execution Plan**

Não faz:
- decisões de performance
- lógica de inferência
- acesso direto a hardware

---

### 3.2 Control Plane

Responsabilidades:
- Gerenciar o ciclo de vida da execução
- Aplicar o Execution Plan
- Coletar e expor métricas
- Receber sugestões de agentes externos (WeOS)

O Control Plane **não executa ops**.

---

### 3.3 Scheduler

Responsabilidades:
- Decidir ordem de execução
- Definir batching
- Priorizar execuções com base em políticas

Características:
- Não conhece detalhes do backend
- Opera apenas sobre metadados e custos estimados
- Pode ser substituído por políticas futuras

---

### 3.4 Execution Engine

Responsabilidades:
- Executar o loop principal de inferência
- Gerenciar:
  - fluxo de tokens
  - estado do modelo
  - KV cache
- Coordenar chamadas ao backend

O Execution Engine **não decide**:
- quantização
- backend
- política de custo

Essas decisões vêm do Execution Plan.

---

### 3.5 Backend Drivers

Responsabilidades:
- Implementar operações numéricas reais
- Gerenciar memória de baixo nível
- Executar ops como:
  - matmul
  - attention
  - norm
  - activation

Características:
- Cada backend implementa a mesma interface
- Nenhuma lógica de negócio
- Substituível em runtime

---

### 3.6 Hardware

Inclui:
- CPU
- GPU (futuro)
- Aceleradores

O motor nunca acessa hardware diretamente sem passar por um backend.

---

## 4. Execution Plan (Objeto Central)

O **Execution Plan** é o contrato que conecta todas as camadas.

Ele define:
- qual modelo será executado
- como será executado
- sob quais restrições

Exemplo conceitual:

`json
{
  "model": "llama3-8b",
  "backend": "cpu_avx2",
  "quantization": "q4_k_m",
  "scheduler_policy": "latency_energy_balanced",
  "limits": {
    "max_watts": 25
  }
}


## 5. Separação Modelo × Execução

A arquitetura separa explicitamente
Model Definition
≠
Execution Plan
O modelo é um artefato lógico

A execução é uma decisão operacional

Isso permite:

trocar quantização sem trocar modelo

otimização automática por agentes

reprodutibilidade

6. Fluxo de Execução (alto nível)

CLI/API recebe o comando

Control Plane valida e cria Execution Plan

Scheduler decide ordem/batch

Execution Engine inicia run loop

Backend executa ops

Métricas são coletadas

Resultados e métricas são retornados

O fluxo detalhado será descrito em 03_execution_flow.md.

7. Observabilidade na Arquitetura

Métricas são coletadas em três níveis:

Execution Engine (latência, tokens)

Backend (tempo de ops)

Sistema (energia)

Todas convergem para o Control Plane.

8. Extensibilidade Planejada

A arquitetura permite, sem refatoração estrutural:

novos backends (CUDA, Metal, ROCm)

novos schedulers

quantização adicional

agentes de otimização externos

9. Restrições Arquiteturais

Nenhuma dependência obrigatória de frameworks de ML

Nenhuma lógica de decisão escondida em backend

Nenhuma mutação de código em runtime

Toda otimização deve ser expressável via Execution Plan

10. Princípio-guia da Arquitetura

“O engine não decide como rodar.
Ele executa decisões explicitamente declaradas.”