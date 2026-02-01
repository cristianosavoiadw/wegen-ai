# Engine de Execução de Modelos — Fluxo de Execução v0.1

## 1. Objetivo do Documento

Este documento descreve **o fluxo detalhado de execução** do motor, desde o recebimento de um prompt até a geração final de tokens, incluindo:

- criação do Execution Plan
- interação entre camadas
- responsabilidades de cada componente
- pontos de coleta de métricas

Este fluxo é **determinístico por definição**.

---

## 2. Visão Geral do Fluxo

Fluxo resumido:

Prompt
→ CLI / API
→ Control Plane
→ Execution Plan
→ Scheduler
→ Execution Engine (run loop)
→ Backend (ops)
→ Tokens (stream)


Nenhuma etapa é implícita.  
Nenhuma decisão ocorre fora do Execution Plan.

---

## 3. Entrada da Execução (CLI / API)

### 3.1 Entrada do Usuário

Exemplo via CLI:

`bash
engine run \
  --model llama3-8b.gguf \
  --quant q4_k_m \
  --backend cpu_avx2 \
  --max-watts 25 \
  --prompt "Olá mundo"

Ou via API:
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

4. Criação do Execution Plan (Control Plane)

O Control Plane:

- 1. Valida a entrada

- 2. Resolve o modelo

- 3. Seleciona backend disponível

- 4. Cria o Execution Plan imutável

Exemplo conceitual:

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



5. Scheduler

O Scheduler recebe o Execution Plan e:

decide se a execução entra imediatamente

ou se será agrupada (batch)

define prioridade baseada em política

No v0.1:

scheduler simples

sem preempção

sem reordenação complexa

Saída do Scheduler:

autorização de execução

parâmetros de batch (se aplicável)

6. Inicialização da Execução

O Execution Engine executa:

Inicialização de contexto

Alocação de memória (arena)

Inicialização do KV cache

Inicialização do backend

Nenhum token é gerado nesta fase.

7. Run Loop (Coração do Motor)

O run loop executa token por token, até atingir condição de parada.

7.1 Pseudocódigo do Run Loop
while not finished:
  prepare_input()
  logits = backend.forward()
  token = sample(logits)
  update_kv_cache(token)
  emit_token(token)
  collect_metrics()

Cada iteração do loop é observável.

8. Execução de Ops (Backend)

O backend executa, por token:

Embedding lookup

Projeções (Q, K, V)

Attention

Feed-forward

Normalização

Logits finais

O backend:

respeita quantização definida no Execution Plan

não decide política

não coleta métricas globais

9. KV Cache

O KV cache:

armazena estados intermediários

é gerenciado pelo Execution Engine

pode usar diferentes layouts internos

No v0.1:

KV cache em memória principal

precisão fixa por execução

10. Streaming de Tokens

Cada token gerado é:

imediatamente emitido (CLI / API)

registrado para métricas

contabilizado para limites de execução

Streaming é parte do contrato, não otimização.

11. Coleta de Métricas

As métricas são coletadas em três níveis:

11.1 Engine

tokens gerados

tempo por token

latência acumulada

11.2 Backend

tempo de execução de ops

uso de memória

11.3 Sistema

consumo médio de energia (Linux)

uso de CPU

Todas as métricas convergem para um único objeto de execução.

12. Condições de Parada

A execução termina quando qualquer condição é atingida:

token de fim de sequência

limite de tokens

limite de energia

interrupção externa

A condição de parada é registrada.

13. Finalização da Execução

Ao finalizar:

backend é encerrado

memória é liberada

métricas finais são consolidadas

resultado é retornado

Exemplo de saída:

{
  "result": "texto gerado",
  "metrics": {
    "tokens": 128,
    "tokens_per_sec": 75.4,
    "watts_avg": 18.2,
    "tokens_per_watt": 4.14
  }
}

14. Garantias do Fluxo

O fluxo garante:

nenhuma decisão implícita

nenhuma mutação de código

nenhuma dependência externa oculta

reprodutibilidade

15. Princípio-guia do Fluxo

“O run loop é simples.
A inteligência está nas decisões declaradas antes dele.”
