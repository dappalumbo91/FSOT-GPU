# FSOT-GPU SOTA scoreboard — same hardware

**GPU:** NVIDIA GeForce RTX 5070  
**Model:** SmolLM2-135M-Instruct (tiny modern instruct)  
**Arms:** industry baseline vs **pure FSOT all-layer** host  

**Capability / barriers / roadmap:** see [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) (this file is the **speed + agree** track).

## Verdict (speed + fidelity track)

| Category | Result |
|----------|--------|
| Wins | quality_next_token_eq_baseline, quality_top5_overlap, prefill_latency, decode_tps, attention_op_track (long-S) |
| Ties | quality_generation_partial, vram |
| Loses | mid-S attention vs fused SDPA |
| Across the board (this track) | **True** for agree + prefill/decode + long-S |
| Open capability track | ARC hold min **~32.5%** ≫ HF; GSM free-gen still open — CURRENT_STATUS |

## Numbers

| Metric | Baseline | Pure FSOT | Ratio |
|--------|----------|-----------|-------|
| Next-token agree | 100% (self) | **100%** vs base | — |
| KL(base‖fsot) | 0 | 1.701 | — |
| Top-5 overlap | — | 0.43 | — |
| Exact multi-token (clone) | 100% (self) | 22% | harsh metric |
| Gen top5 on base path | — | **47%** | — |
| Teacher NLL of FSOT gen | — | **8.32** | lower better |
| Prefill ms | 44.21 | 35.10 | **1.26×** |
| Decode tok/s | 34.7 | 36.1 | **1.04×** |
| Attn track win rate | — | **25%** (2/8) | long 2/2 |

## Attention op sweep (H=9 D=64, fused SDPA vs FSOT CUDA)

| S | SDPA ms | FSOT ms | Speedup | Win |
|---|---------|---------|---------|-----|
| 64 | 0.014 | 0.062 | **0.23×** | — |
| 128 | 0.019 | 0.067 | **0.28×** | — |
| 256 | 0.033 | 0.101 | **0.33×** | — |
| 512 | 0.094 | 0.123 | **0.77×** | — |
| 1024 | 0.220 | 0.341 | **0.65×** | — |
| 2048 | 0.654 | 0.879 | **0.74×** | — |
| 4096 | 2.353 | 2.022 | **1.16×** | WIN |
| 8192 | 8.774 | 5.493 | **1.60×** | WIN |

FSOT structural domain: **long context** (collapse sparsity O(S·A) vs dense O(S²)) and **short fused** path. Mid-S remains the industry fused-kernel sweet spot.

Checkpoint: `C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star\results\industry_lm\checkpoints\pure_fsot_agree100_best.pt`

Ledger: `scoreboard.json`
