# Engine de Execução de Modelos — Documentação Completa v0.1

> **Motor de execução de modelos de linguagem focado em performance por watt, determinismo e observabilidade**

---

## Índice

1. [Requisitos](#1-requisitos)
2. [Arquitetura](#2-arquitetura)
3. [Fluxo de Execução](#3-fluxo-de-execução)
4. [Métricas](#4-métricas)
5. [Integração WeOS](#5-integração-weos)

---

# 1. Requisitos

## 1.1 Visão Geral

Este projeto define um **motor de execução de modelos de linguagem (LLMs)** focado em:

- **Desempenho por watt**
- **Determinismo**
- **Observabilidade de baixo nível**
- **Integração nativa com o WeOS** (controle por agentes)

### O que o engine NÃO é

- ❌ Um framework de ML
- ❌ Um wrapper de PyTorch
- ❌ Um SDK de aplicação

### O que o engine É

- ✅ Um runtime de execução
- ✅ Um sistema de scheduling e otimização
- ✅ Uma camada base de infraestrutura de IA

---

## 1.2 Objetivos do Projeto

### OBJ-01 — Performance consciente de energia

Maximizar **tokens por watt** como métrica primária, mantendo qualidade aceitável.

### OBJ-02 — Determinismo operacional

```
Mesma entrada + Mesma configuração ⇒ Mesmo resultado e métricas dentro de tolerância previsível
```

### OBJ-03 — Modularidade por contrato

Separar explicitamente:
- Modelo
- Plano de execução
- Backend de hardware

### OBJ-04 — Governança e automação

Permitir que agentes do WeOS:
- Observem
- Testem
- Proponham otimizações

Sem alterar código core em produção.

---

## 1.3 Escopo (v0.1)

### ✅ Incluído

- Execução de inferência LLM (texto → tokens)
- Backend CPU (AVX2)
- Quantização: Q8_0, Q6_K, Q4_K_M
- Métricas de performance e energia (Linux)
- CLI e API local

### ⏳ Fora de escopo (v0.1)

- Treinamento de modelos
- Fine-tuning
- Distributed inference
- Kubernetes ou orquestração externa
- Backend GPU (planejado para fases futuras)

---

## 1.4 Requisitos Funcionais

### RF-01 — Execução de modelos

O motor **DEVE**:
- Executar modelos compatíveis com GGUF
- Suportar inferência incremental (streaming de tokens)
- Aceitar prompts em texto

### RF-02 — Plano de Execução (Execution Plan)

O motor **DEVE**:
- Separar definição do modelo de decisões de execução
- Representar decisões de execução em um **Execution Plan explícito**

O Execution Plan **DEVE** conter no mínimo:
- Backend selecionado
- Estratégia de quantização
- Política de scheduler
- Limites operacionais (ex: watts máximos)

### RF-03 — Quantização

O motor **DEVE**:
- Tratar quantização como **parâmetro de execução**, não como atributo fixo do modelo
- Suportar no mínimo:
  - Q8_0
  - Q6_K
  - Q4_K_M
- Permitir trocar quantização sem trocar o artefato lógico do modelo

### RF-04 — Backends plugáveis

O motor **DEVE**:
- Definir uma interface única de backend
- Permitir múltiplas implementações
- Selecionar backend em runtime via Execution Plan

### RF-05 — Scheduler

O motor **DEVE**:
- Possuir scheduler desacoplado do engine
- Suportar execução simples e batching
- Permitir políticas baseadas em latência, custo e energia

### RF-06 — Observabilidade

O motor **DEVE** coletar por execução:
- Tokens por segundo
- Latência (p50, p95)
- Consumo médio de energia (watts)
- Tokens por watt

As métricas **DEVEM** ser:
- Acessíveis via CLI
- Acessíveis via API
- Serializáveis em JSON

### RF-07 — Interface de controle

O motor **DEVE** fornecer:
- CLI para execução local
- API local (HTTP ou gRPC)
- Endpoints claros para integração com o WeOS

---

## 1.5 Requisitos Não Funcionais

### RNF-01 — Performance

- Overhead do runtime < 5%
- Nenhuma dependência obrigatória de frameworks de ML de alto nível

### RNF-02 — Determinismo

- Nenhuma heurística implícita não documentada
- Decisões de execução sempre explícitas no Execution Plan

### RNF-03 — Portabilidade

- Plataforma alvo: Linux x86_64
- Build via CMake
- Compilador compatível com C++20

### RNF-04 — Segurança

- O motor **NÃO** executa código arbitrário
- Configurações **DEVEM** ser validadas por schema
- Nenhuma auto-modificação de código em runtime

---

## 1.6 Métricas de Sucesso (KPIs)

O projeto será considerado bem-sucedido quando:

- ✅ Execuções puderem ser comparadas por **tokens/watt**
- ✅ A troca de quantização não exigir troca de modelo
- ✅ Agentes externos puderem otimizar execução sem alterar código
- ✅ O engine for observável e reproduzível

---

## 1.7 Princípios Arquiteturais

- **Simplicidade > abstração excessiva**
- **Contratos explícitos > heurísticas ocultas**
- **Infra-first, não app-first**
- **Energia é métrica de primeira classe**

---

## 1.8 Direção Futura (não vinculante)

- Backend CUDA
- KV cache quantizado
- Scheduler energy-aware com aprendizado
- Execução concorrente multi-modelo

---

## 1.9 Frase-guia do Projeto

> **"Este motor não executa modelos.**  
> **Ele executa decisões de execução sobre modelos."**

---

# 2. Arquitetura

## 2.1 Objetivo da Arquitetura

Esta arquitetura define a estrutura interna do motor de execução de modelos, garantindo:

- Separação clara de responsabilidades
- Determinismo operacional
- Extensibilidade por contrato
- Controle explícito de decisões de execução

A arquitetura **não otimiza prematuramente**, mas **não bloqueia otimizações futuras**.

---

## 2.2 Visão Geral em Camadas

```
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
```

Cada camada possui **contratos explícitos** e não acessa camadas inferiores fora desses contratos.

---

## 2.3 Componentes Principais

### 2.3.1 CLI / API

**Responsabilidades:**
- Receber comandos de execução
- Validar entrada do usuário
- Traduzir entrada em um **Execution Plan**

**Não faz:**
- Decisões de performance
- Lógica de inferência
- Acesso direto a hardware

---

### 2.3.2 Control Plane

**Responsabilidades:**
- Gerenciar o ciclo de vida da execução
- Aplicar o Execution Plan
- Coletar e expor métricas
- Receber sugestões de agentes externos (WeOS)

O Control Plane **não executa ops**.

---

### 2.3.3 Scheduler

**Responsabilidades:**
- Decidir ordem de execução
- Definir batching
- Priorizar execuções com base em políticas

**Características:**
- Não conhece detalhes do backend
- Opera apenas sobre metadados e custos estimados
- Pode ser substituído por políticas futuras

---

### 2.3.4 Execution Engine

**Responsabilidades:**
- Executar o loop principal de inferência
- Gerenciar:
  - Fluxo de tokens
  - Estado do modelo
  - KV cache
- Coordenar chamadas ao backend

O Execution Engine **não decide**:
- Quantização
- Backend
- Política de custo

Essas decisões vêm do Execution Plan.

---

### 2.3.5 Backend Drivers

**Responsabilidades:**
- Implementar operações numéricas reais
- Gerenciar memória de baixo nível
- Executar ops como:
  - matmul
  - attention
  - norm
  - activation

**Características:**
- Cada backend implementa a mesma interface
- Nenhuma lógica de negócio
- Substituível em runtime

---

### 2.3.6 Hardware

Inclui:
- CPU
- GPU (futuro)
- Aceleradores

O motor nunca acessa hardware diretamente sem passar por um backend.

---

## 2.4 Execution Plan (Objeto Central)

O **Execution Plan** é o contrato que conecta todas as camadas.

Ele define:
- Qual modelo será executado
- Como será executado
- Sob quais restrições

### Exemplo conceitual:

```json
{
  "model": "llama3-8b",
  "backend": "cpu_avx2",
  "quantization": "q4_k_m",
  "scheduler_policy": "latency_energy_balanced",
  "limits": {
    "max_watts": 25
  }
}
```

---

## 2.5 Separação Modelo × Execução

A arquitetura separa explicitamente:

```
Model Definition ≠ Execution Plan
```

- O **modelo** é um artefato lógico
- A **execução** é uma decisão operacional

Isso permite:
- Trocar quantização sem trocar modelo
- Otimização automática por agentes
- Reprodutibilidade

---

## 2.6 Fluxo de Execução (alto nível)

1. CLI/API recebe o comando
2. Control Plane valida e cria Execution Plan
3. Scheduler decide ordem/batch
4. Execution Engine inicia run loop
5. Backend executa ops
6. Métricas são coletadas
7. Resultados e métricas são retornados

> O fluxo detalhado é descrito na seção 3.

---

## 2.7 Observabilidade na Arquitetura

Métricas são coletadas em três níveis:

1. **Execution Engine** (latência, tokens)
2. **Backend** (tempo de ops)
3. **Sistema** (energia)

Todas convergem para o Control Plane.

---

## 2.8 Extensibilidade Planejada

A arquitetura permite, sem refatoração estrutural:

- Novos backends (CUDA, Metal, ROCm)
- Novos schedulers
- Quantização adicional
- Agentes de otimização externos

---

## 2.9 Restrições Arquiteturais

- ❌ Nenhuma dependência obrigatória de frameworks de ML
- ❌ Nenhuma lógica de decisão escondida em backend
- ❌ Nenhuma mutação de código em runtime
- ✅ Toda otimização deve ser expressável via Execution Plan

---

## 2.10 Princípio-guia da Arquitetura

> **"O engine não decide como rodar.**  
> **Ele executa decisões explicitamente declaradas."**

---

# 3. Fluxo de Execução

## 3.1 Objetivo do Documento

Este documento descreve **o fluxo detalhado de execução** do motor, desde o recebimento de um prompt até a geração final de tokens, incluindo:

- Criação do Execution Plan
- Interação entre camadas
- Responsabilidades de cada componente
- Pontos de coleta de métricas

Este fluxo é **determinístico por definição**.

---

## 3.2 Visão Geral do Fluxo

```
Prompt
  → CLI / API
  → Control Plane
  → Execution Plan
  → Scheduler
  → Execution Engine (run loop)
  → Backend (ops)
  → Tokens (stream)
```

- Nenhuma etapa é implícita
- Nenhuma decisão ocorre fora do Execution Plan

---

## 3.3 Entrada da Execução (CLI / API)

### 3.3.1 Entrada do Usuário

**Exemplo via CLI:**

```bash
engine run \
  --model llama3-8b.gguf \
  --quant q4_k_m \
  --backend cpu_avx2 \
  --max-watts 25 \
  --prompt "Olá mundo"
```

**Ou via API:**

```json
{
  "model": "llama3-8b",
  "prompt": "Olá mundo",
  "execution": {
    "backend": "cpu_avx2",
    "quantization": "q4_k_m",
    "limits": {
      "max_watts": 25
    }
  }
}
```

---

## 3.4 Criação do Execution Plan (Control Plane)

O Control Plane:

1. Valida a entrada
2. Resolve o modelo
3. Seleciona backend disponível
4. Cria o Execution Plan imutável

**Exemplo conceitual:**

```json
{
  "model_ref": "llama3-8b",
  "backend": "cpu_avx2",
  "quantization": "q4_k_m",
  "scheduler_policy": "default",
  "limits": {
    "max_watts": 25,
    "max_tokens": 512
  }
}
```

---

## 3.5 Scheduler

O Scheduler recebe o Execution Plan e:

- Decide se a execução entra imediatamente
- Ou se será agrupada (batch)
- Define prioridade baseada em política

**No v0.1:**
- Scheduler simples
- Sem preempção
- Sem reordenação complexa

**Saída do Scheduler:**
- Autorização de execução
- Parâmetros de batch (se aplicável)

---

## 3.6 Inicialização da Execução

O Execution Engine executa:

1. Inicialização de contexto
2. Alocação de memória (arena)
3. Inicialização do KV cache
4. Inicialização do backend

Nenhum token é gerado nesta fase.

---

## 3.7 Run Loop (Coração do Motor)

O run loop executa token por token, até atingir condição de parada.

### 3.7.1 Pseudocódigo do Run Loop

```python
while not finished:
  prepare_input()
  logits = backend.forward()
  token = sample(logits)
  update_kv_cache(token)
  emit_token(token)
  collect_metrics()
```

Cada iteração do loop é observável.

---

## 3.8 Execução de Ops (Backend)

O backend executa, por token:

1. Embedding lookup
2. Projeções (Q, K, V)
3. Attention
4. Feed-forward
5. Normalização
6. Logits finais

O backend:
- Respeita quantização definida no Execution Plan
- Não decide política
- Não coleta métricas globais

---

## 3.9 KV Cache

O KV cache:
- Armazena estados intermediários
- É gerenciado pelo Execution Engine
- Pode usar diferentes layouts internos

**No v0.1:**
- KV cache em memória principal
- Precisão fixa por execução

---

## 3.10 Streaming de Tokens

Cada token gerado é:

- Imediatamente emitido (CLI / API)
- Registrado para métricas
- Contabilizado para limites de execução

Streaming é parte do contrato, não otimização.

---

## 3.11 Coleta de Métricas

As métricas são coletadas em três níveis:

### 3.11.1 Engine
- Tokens gerados
- Tempo por token
- Latência acumulada

### 3.11.2 Backend
- Tempo de execução de ops
- Uso de memória

### 3.11.3 Sistema
- Consumo médio de energia (Linux)
- Uso de CPU

Todas as métricas convergem para um único objeto de execução.

---

## 3.12 Condições de Parada

A execução termina quando qualquer condição é atingida:

- Token de fim de sequência
- Limite de tokens
- Limite de energia
- Interrupção externa

A condição de parada é registrada.

---

## 3.13 Finalização da Execução

Ao finalizar:

1. Backend é encerrado
2. Memória é liberada
3. Métricas finais são consolidadas
4. Resultado é retornado

**Exemplo de saída:**

```json
{
  "result": "texto gerado",
  "metrics": {
    "tokens": 128,
    "tokens_per_sec": 75.4,
    "watts_avg": 18.2,
    "tokens_per_watt": 4.14
  }
}
```

---

## 3.14 Garantias do Fluxo

O fluxo garante:

- Nenhuma decisão implícita
- Nenhuma mutação de código
- Nenhuma dependência externa oculta
- Reprodutibilidade

---

## 3.15 Princípio-guia do Fluxo

> **"O run loop é simples.**  
> **A inteligência está nas decisões declaradas antes dele."**

---

# 4. Métricas

## 4.1 Objetivo do Documento

Este documento define **as métricas oficiais do motor**, suas fórmulas, fontes e critérios de validade.

As métricas aqui descritas são:
- Parte do contrato do engine
- Base para comparação entre execuções
- Insumo direto para agentes de otimização do WeOS

Nenhuma métrica fora deste documento é considerada oficial.

---

## 4.2 Princípios de Medição

Todas as métricas **DEVEM** obedecer aos seguintes princípios:

- **Determinismo:** mesmas condições ⇒ métricas comparáveis
- **Baixo overhead:** coleta não pode distorcer o resultado
- **Observabilidade explícita:** nada implícito ou inferido
- **Reprodutibilidade:** métricas devem ser auditáveis

---

## 4.3 Unidade de Execução

Todas as métricas são associadas a uma **execução única**, definida por:

- Modelo
- Execution Plan
- Backend
- Hardware
- Versão do engine

Nenhuma métrica é global ou agregada implicitamente.

---

## 4.4 Métricas Primárias (Obrigatórias)

### 4.4.1 Tokens Gerados (`tokens_total`)

- **Descrição:** Número total de tokens gerados
- **Unidade:** tokens
- **Fonte:** Execution Engine
- **Obrigatória:** Sim

---

### 4.4.2 Tokens por Segundo (`tokens_per_sec`)

- **Descrição:** Vazão média de geração de tokens
- **Unidade:** tokens/segundo
- **Fórmula:**

```
tokens_per_sec = tokens_total / tempo_total_execucao
```

- **Fonte:** Execution Engine
- **Obrigatória:** Sim

---

### 4.4.3 Latência Total (`latency_total_ms`)

- **Descrição:** Tempo total da execução
- **Unidade:** milissegundos (ms)
- **Fonte:** Execution Engine
- **Obrigatória:** Sim

---

### 4.4.4 Latência Percentil (`latency_p50_ms`, `latency_p95_ms`)

- **Descrição:** Latência por token em percentis
- **Unidade:** milissegundos (ms)
- **Fonte:** Execution Engine
- **Obrigatória:** p50 e p95

---

### 4.4.5 Consumo Médio de Energia (`watts_avg`)

- **Descrição:** Consumo médio de energia durante a execução
- **Unidade:** watts (W)
- **Fonte:** Sistema operacional (Linux)
- **Obrigatória:** Sim

**Notas:**
- Medido apenas durante o período ativo da execução
- Exclui idle anterior e posterior

---

### 4.4.6 Tokens por Watt (`tokens_per_watt`)

- **Descrição:** Eficiência energética da execução
- **Unidade:** tokens/watt
- **Fórmula:**

```
tokens_per_watt = tokens_per_sec / watts_avg
```

- **Fonte:** Cálculo derivado
- **Obrigatória:** Sim

> **Esta é a métrica primária de sucesso do motor.**

---

## 4.5 Métricas Secundárias (v0.1)

### 4.5.1 Uso de Memória (`memory_peak_mb`)

- **Descrição:** Pico de memória durante a execução
- **Unidade:** megabytes (MB)
- **Fonte:** Runtime / sistema
- **Obrigatória:** Não (v0.1)

---

### 4.5.2 Tempo por Operação (`op_time_ms`)

- **Descrição:** Tempo gasto por tipo de operação (matmul, attention, etc.)
- **Unidade:** milissegundos (ms)
- **Fonte:** Backend
- **Obrigatória:** Opcional (debug / benchmark)

---

## 4.6 Fontes de Medição (Linux)

### 4.6.1 Tempo

- `std::chrono` (C++)
- Clock monotônico
- Resolução mínima: microssegundos

### 4.6.2 Energia (CPU)

**Fontes possíveis:**
- RAPL (`/sys/class/powercap`)
- Interfaces equivalentes AMD/Intel

**Requisitos:**
- Leitura antes e depois da execução
- Cálculo de média no intervalo ativo

### 4.6.3 Energia (GPU – futuro)

- NVML (NVIDIA)
- ROCm SMI

(Não obrigatório no v0.1)

---

## 4.7 Tolerâncias e Comparabilidade

Duas execuções são consideradas **comparáveis** se:

- Mesmo modelo
- Mesmo Execution Plan
- Mesmo backend
- Mesmo hardware
- Mesma versão do engine

**Tolerâncias aceitáveis:**
- Variação ≤ 3% em `tokens_per_sec`
- Variação ≤ 5% em `watts_avg`

---

## 4.8 Serialização das Métricas

Todas as métricas **DEVEM** ser serializáveis em JSON.

**Exemplo:**

```json
{
  "engine_version": "0.1.0",
  "model": "llama3-8b",
  "backend": "cpu_avx2",
  "quantization": "q4_k_m",
  "metrics": {
    "tokens_total": 256,
    "tokens_per_sec": 74.3,
    "latency_p50_ms": 12.4,
    "latency_p95_ms": 18.9,
    "watts_avg": 18.1,
    "tokens_per_watt": 4.10
  }
}
```

---

## 4.9 Uso das Métricas por Agentes (WeOS)

Agentes externos **PODEM:**
- Observar métricas
- Comparar execuções
- Sugerir novos Execution Plans

Agentes **NÃO PODEM:**
- Alterar métricas
- Redefinir fórmulas
- Ocultar medições

---

## 4.10 Antipadrões Proibidos

- ❌ Métricas implícitas
- ❌ Métricas calculadas fora do engine
- ❌ "Benchmarks mágicos"
- ❌ Comparação entre execuções não equivalentes

---

## 4.11 Princípio-guia das Métricas

> **"Se não pode ser medido, não pode ser otimizado —**  
> **e se não é reprodutível, não é infraestrutura."**

---

# 5. Integração WeOS

## 5.1 Objetivo do Documento

Este documento define **como o motor de execução se integra ao WeOS**, permitindo que agentes de IA:

- Observem execuções reais
- Coletem métricas
- Proponham otimizações
- Validem melhorias

Tudo isso **sem alterar código core** do motor.

A integração é feita exclusivamente por **contratos explícitos**.

---

## 5.2 Papel do Motor dentro do WeOS

No ecossistema WeOS, o motor atua como:

- **Execution Runtime**
- **Fonte de métricas de baixo nível**
- **Alvo de otimização por agentes**

O motor **não é**:
- Responsável por orquestração global
- Responsável por decisão de negócio
- Responsável por versionamento de agentes

---

## 5.3 Princípio Fundamental da Integração

> **O WeOS decide.**  
> **O motor executa.**  
> **O motor mede.**  
> **O WeOS aprende e otimiza.**

Nenhuma inteligência de otimização reside dentro do core do motor.

---

## 5.4 Modelo de Integração

A integração ocorre via **Control Plane do motor**, exposto por API local.

```
WeOS Agent
    ↓
API do Motor (Control Plane)
    ↓
Execution Engine
```

O WeOS nunca acessa:
- Backend diretamente
- Memória
- Hardware

---

## 5.5 Contrato de Execução (Run Request)

O WeOS envia ao motor um **Run Request**, que é traduzido internamente em um Execution Plan.

**Exemplo:**

```json
{
  "model": "llama3-8b",
  "prompt": "Explique inferência eficiente",
  "execution": {
    "backend": "cpu_avx2",
    "quantization": "q4_k_m",
    "scheduler_policy": "default",
    "limits": {
      "max_tokens": 256,
      "max_watts": 25
    }
  }
}
```

O motor **DEVE** rejeitar requests inválidos.

---

## 5.6 Contrato de Métricas

Após cada execução, o motor retorna métricas estruturadas.

**Exemplo:**

```json
{
  "execution_id": "exec_9f32",
  "engine_version": "0.1.0",
  "metrics": {
    "tokens_total": 256,
    "tokens_per_sec": 73.8,
    "latency_p95_ms": 19.1,
    "watts_avg": 18.4,
    "tokens_per_watt": 4.01
  }
}
```

Essas métricas são:
- Imutáveis
- Auditáveis
- Comparáveis

---

## 5.7 Loop de Otimização por Agentes

O WeOS pode executar o seguinte ciclo:

1. Executar workload real
2. Coletar métricas
3. Comparar execuções
4. Propor novo Execution Plan
5. Executar novamente
6. Validar melhoria ou descartar

Esse loop não exige alteração de código do motor.

---

## 5.8 Tipos de Agentes no WeOS

### 5.8.1 Agente Observador

- Coleta métricas
- Detecta padrões
- Não altera execução

### 5.8.2 Agente Experimental

- Executa variações controladas
- Compara métricas
- Registra resultados

### 5.8.3 Agente Otimizador

- Propõe novos Execution Plans
- Aplica somente se KPI melhorar
- Respeita limites definidos

---

## 5.9 Restrições de Segurança

O motor **DEVE** garantir:

- ❌ Nenhuma execução de código arbitrário
- ❌ Nenhuma auto-modificação de binários
- ❌ Nenhuma alteração de backend em runtime sem plano explícito
- ❌ Nenhuma persistência de estado oculto entre execuções

---

## 5.10 Versionamento e Governança

- Cada versão do motor possui um identificador único
- Métricas incluem a versão do motor
- Agentes **DEVEM** registrar qual versão foi usada

Mudanças estruturais no motor exigem:
- Nova versão
- Nova validação
- Nova comparação de métricas

---

## 5.11 Benefícios da Integração

Essa integração permite que o WeOS:

- ✅ Transforme uso real em inteligência técnica
- ✅ Otimize custo e energia continuamente
- ✅ Mantenha governança e auditabilidade
- ✅ Trate o motor como infraestrutura viva

---

## 5.12 Limitações Conhecidas (v0.1)

- ⏳ Sem auto-tuning interno
- ⏳ Sem aprendizado persistente no motor
- ⏳ Sem alteração dinâmica de código
- ⏳ Sem controle distribuído

Essas capacidades residem exclusivamente no WeOS.

---

## 5.13 Princípio-guia da Integração

> **"O motor é determinístico.**  
> **A inteligência evolutiva vive no WeOS."**

---

## 5.14 Encerramento do Ciclo de Documentação

Com este documento, o projeto possui:

- ✅ Requisitos formais
- ✅ Arquitetura definida
- ✅ Fluxo de execução explícito
- ✅ Métricas padronizadas
- ✅ Contrato de integração com agentes

---

## Observações Finais

Esta documentação representa a versão **0.1** do motor de execução de modelos. Todos os requisitos, arquitetura e especificações estão sujeitos a evolução conforme o projeto amadurece.

**Princípio fundamental:**

> **"Infraestrutura > Aplicação"**  
> **"Contratos explícitos > Heurísticas ocultas"**  
> **"Energia é métrica de primeira classe"**

---

**Última atualização:** v0.1  
**Licença:** [Definir]  
**Contato:** WeGen-AI Brasil