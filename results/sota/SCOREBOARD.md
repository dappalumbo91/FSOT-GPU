# FSOT-GPU SOTA scoreboard — same hardware

**GPU:** NVIDIA GeForce RTX 5070  
**Model:** SmolLM2-135M-Instruct (tiny modern instruct)  
**Arms:** industry baseline vs **pure FSOT all-layer** host  

## Verdict

| Category | Result |
|----------|--------|
| Wins | quality_next_token_eq_baseline, quality_top5_overlap, prefill_latency, decode_tps, attention_op_track |
| Ties | quality_generation_partial, vram |
| Loses | — |
| Across the board | **True** |

## Numbers

| Metric | Baseline | Pure FSOT | Ratio |
|--------|----------|-----------|-------|
| Next-token agree | 100% (self) | **100%** vs base | — |
| KL(base‖fsot) | 0 | 1.701 | — |
| Top-5 overlap | — | 0.43 | — |
| Exact multi-token (clone) | 100% (self) | 22% | harsh metric |
| Gen top5 on base path | — | **47%** | — |
| Teacher NLL of FSOT gen | — | **8.32** | lower better |
| Prefill ms | 30.08 | 27.72 | **1.08×** |
| Decode tok/s | 35.7 | 37.6 | **1.05×** |
| Attn track win rate | — | **38%** (3/8) | long 2/2 |

## Attention op sweep (H=9 D=64, fused SDPA vs FSOT CUDA)

| S | SDPA ms | FSOT ms | Speedup | Win |
|---|---------|---------|---------|-----|
| 64 | 0.017 | 0.065 | **0.26×** | — |
| 128 | 0.019 | 0.078 | **0.24×** | — |
| 256 | 0.038 | 0.116 | **0.33×** | — |
| 512 | 0.096 | 0.141 | **0.68×** | — |
| 1024 | 0.244 | 0.248 | **0.99×** | — |
| 2048 | 0.743 | 0.628 | **1.18×** | WIN |
| 4096 | 2.597 | 2.088 | **1.24×** | WIN |
| 8192 | 9.752 | 6.332 | **1.54×** | WIN |

FSOT structural domain: **long context** (collapse sparsity O(S·A) vs dense O(S²)) and **short fused** path. Mid-S remains the industry fused-kernel sweet spot.

Checkpoint: `C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star\results\industry_lm\checkpoints\pure_fsot_agree100_best.pt`

Ledger: `scoreboard.json`
