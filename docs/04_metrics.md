# Engine de Execução de Modelos — Métricas v0.1

## 1. Objetivo do Documento

Este documento define **as métricas oficiais do motor**, suas fórmulas, fontes e critérios de validade.

As métricas aqui descritas são:
- parte do contrato do engine
- base para comparação entre execuções
- insumo direto para agentes de otimização do WeOS

Nenhuma métrica fora deste documento é considerada oficial.

---

## 2. Princípios de Medição

Todas as métricas DEVEM obedecer aos seguintes princípios:

- **Determinismo**: mesmas condições ⇒ métricas comparáveis
- **Baixo overhead**: coleta não pode distorcer o resultado
- **Observabilidade explícita**: nada implícito ou inferido
- **Reprodutibilidade**: métricas devem ser auditáveis

---

## 3. Unidade de Execução

Todas as métricas são associadas a uma **execução única**, definida por:

- modelo
- Execution Plan
- backend
- hardware
- versão do engine

Nenhuma métrica é global ou agregada implicitamente.

---

## 4. Métricas Primárias (Obrigatórias)

### 4.1 Tokens Gerados (`tokens_total`)

- **Descrição**: número total de tokens gerados
- **Unidade**: tokens
- **Fonte**: Execution Engine
- **Obrigatória**: sim

---

### 4.2 Tokens por Segundo (`tokens_per_sec`)

- **Descrição**: vazão média de geração de tokens
- **Unidade**: tokens/segundo
- **Fórmula**:

`text
tokens_per_sec = tokens_total / tempo_total_execucao

Fonte: Execution Engine

Obrigatória: sim

4.3 Latência Total (latency_total_ms)

Descrição: tempo total da execução

Unidade: milissegundos (ms)

Fonte: Execution Engine

Obrigatória: sim

4.4 Latência Percentil (latency_p50_ms, latency_p95_ms)

Descrição: latência por token em percentis

Unidade: milissegundos (ms)

Fonte: Execution Engine

Obrigatória: p50 e p95

4.5 Consumo Médio de Energia (watts_avg)

Descrição: consumo médio de energia durante a execução

Unidade: watts (W)

Fonte: sistema operacional (Linux)

Obrigatória: sim

Notas:

medido apenas durante o período ativo da execução

exclui idle anterior e posterior

4.6 Tokens por Watt (tokens_per_watt)

Descrição: eficiência energética da execução

Unidade: tokens/watt

Fórmula:

tokens_per_watt = tokens_per_sec / watts_avg


Fonte: cálculo derivado

Obrigatória: sim

Esta é a métrica primária de sucesso do motor.

5. Métricas Secundárias (v0.1)
5.1 Uso de Memória (memory_peak_mb)

Descrição: pico de memória durante a execução

Unidade: megabytes (MB)

Fonte: runtime / sistema

Obrigatória: não (v0.1)

5.2 Tempo por Operação (op_time_ms)

Descrição: tempo gasto por tipo de operação (matmul, attention, etc.)

Unidade: milissegundos (ms)

Fonte: backend

Obrigatória: opcional (debug / benchmark)

6. Fontes de Medição (Linux)
6.1 Tempo

std::chrono (C++)

clock monotônico

resolução mínima: microssegundos

6.2 Energia (CPU)

Fontes possíveis:

RAPL (/sys/class/powercap)

interfaces equivalentes AMD/Intel

Requisitos:

leitura antes e depois da execução

cálculo de média no intervalo ativo

6.3 Energia (GPU – futuro)

NVML (NVIDIA)

ROCm SMI

(Não obrigatório no v0.1)

7. Tolerâncias e Comparabilidade

Duas execuções são consideradas comparáveis se:

mesmo modelo

mesmo Execution Plan

mesmo backend

mesmo hardware

mesma versão do engine

Tolerâncias aceitáveis:

variação ≤ 3% em tokens_per_sec

variação ≤ 5% em watts_avg

8. Serialização das Métricas

Todas as métricas DEVEM ser serializáveis em JSON.

Exemplo:

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

9. Uso das Métricas por Agentes (WeOS)

Agentes externos PODEM:

observar métricas

comparar execuções

sugerir novos Execution Plans

Agentes NÃO PODEM:

alterar métricas

redefinir fórmulas

ocultar medições

10. Antipadrões Proibidos

métricas implícitas

métricas calculadas fora do engine

“benchmarks mágicos”

comparação entre execuções não equivalentes

11. Princípio-guia das Métricas

“Se não pode ser medido,
não pode ser otimizado —
e se não é reprodutível,
não é infraestrutura.”
