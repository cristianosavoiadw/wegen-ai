# Phase 1 — Canonical Test

## Goal
Validate correct execution, GGUF validation and energy metrics.

## Command

engine run \
  --model llama3-8b.Q4_K_M.gguf \
  --max-tokens 128 \
  --output json

## Expected Behavior

- Exit code: 0
- Stdout: valid JSON
- Stderr: empty or logs only

## Required JSON Fields

engine.version  
engine.backend  

model.arch  
model.context  
model.quant_detected  

metrics.tokens_total  
metrics.tokens_per_sec  
metrics.watts_avg  
metrics.tokens_per_watt  

## Error Cases

### Invalid model path
- Exit code: 2
- Error message on stderr

### Runtime failure
- Exit code: 1
