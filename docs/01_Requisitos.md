# Engine de Execução de Modelos — Requisitos v0.1

## 1. Visão Geral

Este projeto define um **motor de execução de modelos de linguagem (LLMs)** focado em:

- desempenho por watt  
- determinismo  
- observabilidade de baixo nível  
- integração nativa com o WeOS (controle por agentes)

O engine **não é**:
- um framework de ML
- um wrapper de PyTorch
- um SDK de aplicação

O engine **é**:
- um runtime de execução
- um sistema de scheduling e otimização
- uma camada base de infraestrutura de IA

---

## 2. Objetivos do Projeto

### OBJ-01 — Performance consciente de energia
Maximizar **tokens por watt** como métrica primária, mantendo qualidade aceitável.

### OBJ-02 — Determinismo operacional
Mesma entrada + mesma configuração ⇒ mesmo resultado e métricas dentro de tolerância previsível.

### OBJ-03 — Modularidade por contrato
Separar explicitamente:
- modelo
- plano de execução
- backend de hardware

### OBJ-04 — Governança e automação
Permitir que agentes do WeOS:
- observem
- testem
- proponham otimizações  
sem alterar código core em produção.

---

## 3. Escopo (v0.1)

### Incluído
- Execução de inferência LLM (texto → tokens)
- Backend CPU (AVX2)
- Quantização: Q8_0, Q6_K, Q4_K_M
- Métricas de performance e energia (Linux)
- CLI e API local

### Fora de escopo (v0.1)
- Treinamento de modelos
- Fine-tuning
- Distributed inference
- Kubernetes ou orquestração externa
- Backend GPU (planejado para fases futuras)

---

## 4. Requisitos Funcionais

### RF-01 — Execução de modelos
O motor DEVE:
- executar modelos compatíveis com GGUF
- suportar inferência incremental (streaming de tokens)
- aceitar prompts em texto

---

### RF-02 — Plano de Execução (Execution Plan)
O motor DEVE:
- separar definição do modelo de decisões de execução
- representar decisões de execução em um **Execution Plan explícito**

O Execution Plan DEVE conter no mínimo:
- backend selecionado
- estratégia de quantização
- política de scheduler
- limites operacionais (ex: watts máximos)

---

### RF-03 — Quantização
O motor DEVE:
- tratar quantização como **parâmetro de execução**, não como atributo fixo do modelo
- suportar no mínimo:
  - Q8_0
  - Q6_K
  - Q4_K_M
- permitir trocar quantização sem trocar o artefato lógico do modelo

---

### RF-04 — Backends plugáveis
O motor DEVE:
- definir uma interface única de backend
- permitir múltiplas implementações
- selecionar backend em runtime via Execution Plan

---

### RF-05 — Scheduler
O motor DEVE:
- possuir scheduler desacoplado do engine
- suportar execução simples e batching
- permitir políticas baseadas em latência, custo e energia

---

### RF-06 — Observabilidade
O motor DEVE coletar por execução:
- tokens por segundo
- latência (p50, p95)
- consumo médio de energia (watts)
- tokens por watt

As métricas DEVEM ser:
- acessíveis via CLI
- acessíveis via API
- serializáveis em JSON

---

### RF-07 — Interface de controle
O motor DEVE fornecer:
- CLI para execução local
- API local (HTTP ou gRPC)
- endpoints claros para integração com o WeOS

---

## 5. Requisitos Não Funcionais

### RNF-01 — Performance
- Overhead do runtime < 5%
- Nenhuma dependência obrigatória de frameworks de ML de alto nível

---

### RNF-02 — Determinismo
- Nenhuma heurística implícita não documentada
- Decisões de execução sempre explícitas no Execution Plan

---

### RNF-03 — Portabilidade
- Plataforma alvo: Linux x86_64
- Build via CMake
- Compilador compatível com C++20

---

### RNF-04 — Segurança
- O motor NÃO executa código arbitrário
- Configurações DEVEM ser validadas por schema
- Nenhuma auto-modificação de código em runtime

---

## 6. Métricas de Sucesso (KPIs)

O projeto será considerado bem-sucedido quando:

- execuções puderem ser comparadas por **tokens/watt**
- a troca de quantização não exigir troca de modelo
- agentes externos puderem otimizar execução sem alterar código
- o engine for observável e reproduzível

---

## 7. Princípios Arquiteturais

- Simplicidade > abstração excessiva  
- Contratos explícitos > heurísticas ocultas  
- Infra-first, não app-first  
- Energia é métrica de primeira classe  

---

## 8. Direção futura (não vinculante)

- Backend CUDA
- KV cache quantizado
- Scheduler energy-aware com aprendizado
- Execução concorrente multi-modelo

---

## 9. Frase-guia do projeto

> “Este motor não executa modelos.  
> Ele executa **decisões de execução** sobre modelos.”
