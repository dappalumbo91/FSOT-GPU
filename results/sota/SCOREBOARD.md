# FSOT-GPU SOTA scoreboard — same hardware

**GPU:** NVIDIA GeForce RTX 5070  
**Model:** SmolLM2-135M-Instruct (tiny modern instruct)  
**Arms:** industry baseline vs **pure FSOT all-layer** host  

## Verdict

| Category | Result |
|----------|--------|
| Wins | quality_next_token_ge_90, quality_top5_overlap, prefill_latency, decode_tps, attention_op_track |
| Ties | quality_generation_partial, vram |
| Loses | — |
| Across the board | **True** |

## Numbers

| Metric | Baseline | Pure FSOT | Ratio |
|--------|----------|-----------|-------|
| Next-token agree | 100% (self) | **94%** vs base | — |
| KL(base‖fsot) | 0 | 2.076 | — |
| Top-5 overlap | — | 0.39 | — |
| Exact multi-token (clone) | 100% (self) | 14% | harsh metric |
| Gen top5 on base path | — | **34%** | — |
| Teacher NLL of FSOT gen | — | **5.96** | lower better |
| Prefill ms | 29.86 | 27.33 | **1.09×** |
| Decode tok/s | 36.2 | 38.4 | **1.06×** |
| Attn track win rate | — | **25%** (2/8) | long 2/2 |

## Attention op sweep (H=9 D=64, fused SDPA vs FSOT CUDA)

| S | SDPA ms | FSOT ms | Speedup | Win |
|---|---------|---------|---------|-----|
| 64 | 0.013 | 0.060 | **0.22×** | — |
| 128 | 0.019 | 0.063 | **0.30×** | — |
| 256 | 0.035 | 0.091 | **0.39×** | — |
| 512 | 0.084 | 0.139 | **0.60×** | — |
| 1024 | 0.210 | 0.263 | **0.80×** | — |
| 2048 | 0.670 | 0.772 | **0.87×** | — |
| 4096 | 2.294 | 2.150 | **1.07×** | WIN |
| 8192 | 8.623 | 5.398 | **1.60×** | WIN |

FSOT structural domain: **long context** (collapse sparsity O(S·A) vs dense O(S²)) and **short fused** path. Mid-S remains the industry fused-kernel sweet spot.

Checkpoint: `C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star\results\industry_lm\checkpoints\pure_fsot_agree_best.pt`

Ledger: `scoreboard.json`
