# Industry LM unit report — SmolLM2-135M-Instruct

**Date:** 2026-07-17  
**Lab:** FSOT Formal-GPU  
**Not used:** FSOT-2.1-Instruct (separate project)

## Why this model

| Property | Value |
|----------|--------|
| Industry source | Hugging Face `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Format | **SafeTensors** (`model.safetensors`) |
| Architecture | LlamaForCausalLM (modern stack: RMSNorm, GQA, SiLU MLP, RoPE) |
| Size | **~134.5M params**, **~265 MiB** VRAM (bf16) on RTX 5070 |
| Role | Code-agnostic portable unit for multi-language GPU hosting |

## Code-agnostic path

```
SafeTensors weights
    → portable_schema.json  (tensor bank + graph IR)
    → backend implements IR ops
         industry: transformers SDPA + RMSNorm
         FSOT:     replaceable attn / norm nodes
         future:   Rust / Zig / pure CUDA host reading same bank
```

Schema: `industry_lm/portable/portable_schema.json`

## Baseline (industry)

- Prefill ~**49 ms** / forward (short prompt, 20× avg)  
- Generation smoke: *“The capital of France is Paris…”* — coherent  
- Weights load cleanly from SafeTensors  

## FSOT bridge (same weights)

- Model prefill logits finite  
- On activation-scaled QKV from the model’s residual scale:  
  - torch FSOT sparse attn still slower than fused SDPA (host path)  
  - **norm:** RMS currently faster than coherence_norm on dense residual (expected — different algorithm; quality tradeoff next)  
- **Device path** (standalone CUDA sparse, archive math) remains the throughput winner on microbench shapes  

## Longer seq

| S | SDPA (torch) | FSOT torch sparse | FSOT **CUDA** sparse | CUDA vs SDPA |
|---|--------------|-------------------|----------------------|--------------|
| 256 | ~0.22 ms | ~2.6 ms | **~0.032 ms** | **~7× faster** |
| 512 | ~0.21 ms | ~2.6 ms | **~0.046 ms** | **~4.5× faster** |
| 1024 | ~0.52 ms | ~4.3 ms | **~0.088 ms** | **~6× faster** |

Host torch sparse is not the competitive arm; **CUDA** is (archive collapse sparsity).

## Next for true LLM port

1. Swap **one layer’s attention** in SmolLM forward to FSOT CUDA consensus (same QKV projections from weights).  
2. Quality gate: next-token KL / short generation match rate vs baseline.  
3. Export graph IR runners in **Rust** and **Zig** reading the same SafeTensors bank.  
