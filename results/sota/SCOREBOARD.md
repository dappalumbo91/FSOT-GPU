# FSOT-GPU SOTA scoreboard — same hardware

**GPU:** NVIDIA GeForce RTX 5070  
**Model:** SmolLM2-135M-Instruct (tiny modern instruct)  
**Arms:** industry baseline vs **pure FSOT all-layer** host  

## Verdict

| Category | Result |
|----------|--------|
| Wins | quality_next_token_ge_90, quality_top5_overlap, prefill_latency, decode_tps |
| Ties | vram |
| Loses | attention_op |
| Across the board | **False** |

## Numbers

| Metric | Baseline | Pure FSOT | Ratio |
|--------|----------|-----------|-------|
| Next-token agree | 100% (self) | **94%** vs base | — |
| KL(base‖fsot) | 0 | 2.076 | — |
| Top-5 overlap | — | 0.39 | — |
| Prefill ms | 30.68 | 28.05 | **1.09×** |
| Decode tok/s | 35.5 | 37.6 | **1.06×** |
| Attn op ms (H9 S256) | 0.040 | 0.161 | **0.25×** |

Checkpoint: `C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star\results\industry_lm\checkpoints\pure_fsot_agree_best.pt`

Ledger: `scoreboard.json`
