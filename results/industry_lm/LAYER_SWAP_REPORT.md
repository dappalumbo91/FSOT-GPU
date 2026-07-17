# SmolLM2 layer swap — FSOT attention in a real industry model

**Date:** 2026-07-17  
**Model:** HuggingFaceTB/SmolLM2-135M-Instruct (SafeTensors)  
**GPU:** RTX 5070  
**Change:** Replace **layer 0 only** `LlamaAttention` with **FSOT consensus** (archive collapse θ + coh gate, CUDA DLL).  
**Unchanged:** All Q/K/V/O weights, RoPE, other 29 layers, industry SDPA.

## What we wired

```
hidden → q/k/v_proj (industry weights)
      → RoPE (industry)
      → GQA expand
      → FSOT CUDA consensus   ← replaces SDPA
      → o_proj (industry)
```

- Module: `industry_lm/fsot_layer_swap.py`  
- CUDA lib: `phase2_native_gpu/cuda/fsot_attn_lib.dll`  
- Eval: `python industry_lm/run_layer_swap_eval.py`  
- Ledger: `layer_swap_eval.json`

## Quality (next-token, greedy)

| Metric | Value |
|--------|--------|
| Argmax agreement (5 prompts) | **60%** |
| Mean KL(base ‖ fsot) | **1.23** |
| Mean top-5 overlap | **0.44** |

| Prompt | Base next | FSOT next | Match |
|--------|-----------|-----------|-------|
| The capital of France is | Paris | the | no |
| def fibonacci(n): | (newline indent) | same | yes |
| derivative of x^2 is | (space) | a | no |
| Once upon a time | , | , | yes |
| 2 + 2 = | (space) | (space) | yes |

**Honest read:** One layer of a 30-layer model, **without retraining**, will not match SDPA logits. 60% next-token agreement and partial top-5 overlap show the model still runs and often stays in-distribution; full parity needs multi-layer FSOT + adaptation or distillation.

## Throughput

| | Baseline | FSOT layer-0 |
|--|----------|--------------|
| Prefill (short) | ~36.1 ms | ~36.5 ms (~same) |
| Gen tokens/s | ~29.4 | **~30.2** (~**1.02×**) |

Expected: only **1/30** of attention is FSOT; MLP/other layers still dominate. Microbench wins are on the **attention op**; end-to-end speedup grows as more layers swap **and** the kernel stays on-device (device-pointer path enabled).

## Generation smoke (baseline still coherent)

Baseline continues to produce fluent text (Paris, fibonacci, derivative). FSOT-l0 generates at similar speed; content diverges where layer-0 attention differs — by design of the experiment.

## FSOT math still in force

- Collapse θ = C_eff · P_var  
- Active keys only (coh > 0.5)  
- No exp  

## Next to push quality *and* speed

1. Swap **all** layers (or every other) + short LoRA/FSOT-LR adaptation on a tiny corpus.  
2. Keep device-pointer CUDA (no H2D) — already preferred in `fsot_cuda_ops.py`.  
3. Quality gate: raise argmax agreement / lower KL before claiming “drop-in.”  
4. Measure tokens/s again after multi-layer swap (attention fraction ↑).
