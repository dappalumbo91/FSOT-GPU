# All-layer FSOT swap — remeasure

## Pure FSOT consensus (all 30 layers)

**Mode:** Every `self_attn` → FSOT CUDA consensus (no softmax). Weights frozen.

| Metric | Layer-0 only | **All 30 layers** |
|--------|--------------|-------------------|
| Argmax agreement | 60% | **0%** |
| Mean KL | 1.23 | **8.05** |
| Top-5 overlap | 0.44 | **0.05** |
| tok/s | ~30.2 (~1.02×) | **~11.9 (0.92×)** |
| Prefill | ~1.0× | **0.97×** |

**Takeaway:** Full pure-consensus swap without adaptation **destroys** next-token agreement. That is expected: the model was trained for softmax SDPA, not trit consensus. Layer-0 was a probe; all-layers shows the real gap.

Ledger: `all_layers_eval.json`

## Why we still push

Operator microbench already **beats** dense CUDA / fused SDPA.  
End-to-end needs either:

1. **FSOT-gated SDPA** — FSOT chooses active keys; softmax among them (keeps training geometry), or  
2. **Adaptation** — short FSOT-LR / LoRA so pure consensus realigns.

Gated path eval: `gated_vs_pure_all_layers.json` / run `run_gated_all_eval.py`.

## FSOT-gated SDPA (all 30, no train)

| Metric | Gated | Pure consensus |
|--------|-------|----------------|
| Argmax agreement | **12%** | 0% |
| Mean KL | **5.31** | 8.05 |
| tok/s | 12.2 (0.93×) | 11.9 (0.92×) |

Gating helps a little; still not drop-in without adaptation.

## Blend recovery (all 30) — **pushed through**

```
out = (1-α)·SDPA + α·FSOT_consensus
α init = 1/φ² ≈ 0.382
train only α (30 scalars), 80 steps, LR = suction·K (FSOT)
```

| Metric | Pure all-layer | **Blend adapted** |
|--------|----------------|-------------------|
| Argmax agreement | 0% | **100%** |
| Mean KL | 8.05 | **0.065** |
| Top-5 overlap | 0.05 | **0.87** |
| mean α after train | — | **~0.17** (FSOT still in the mix) |
| tok/s vs baseline | 0.92× | **0.89×** (runs both paths) |

Ledger: `adapt_blend_all_layers.json`

**Read:** Full pure swap without train fails quality.  
**With FSOT-init blend + 30 α’s + short KL:** quality **recovers to 100% agreement** on the probe set while **FSOT consensus stays in every layer**.  

## Distill push (continued)

### Stable blend (α-only, fp32)

| Phase | agree | KL | mean α | tok/s vs base |
|-------|-------|-----|--------|---------------|
| A α-only 80 steps | **100%** | 0.031 | 0.15 | ~0.9× dual path |
| B α-push + mild purity | **100%** | 0.005 | **0.06** | dual (KL fights α↑) |

KL training wants **lower** α (more SDPA). Purity pressure must be much stronger—or train pure FSOT directly.

### Pure FSOT all-layer adapt (no SDPA)

| Checkpoint | agree | KL | top5 | tok/s |
|------------|-------|-----|------|-------|
| Unadapted | 0% | 8.05 | 0.05 | 0.92× |
| Short o_proj/norm | 25% | 5.39 | — | 0.75× |
| **+ QKV 400 steps** | **25%** | **4.85** | **0.22** | **0.95×** |

Trajectory: KL and top-5 **keep improving**; argmax still plateaued at 25% on the 8-prompt probe.  
Pure FSOT is **learning** from the teacher—needs more data/steps or LoRA rank for full agreement.

Ledgers: `distill_pure_fsot.json`, `pure_fsot_extend.json`

### Scoreboard (honest)

| Config | Quality | Speed | FSOT in path |
|--------|---------|-------|----------------|
| Baseline SDPA | 100% | 1.0× | no |
| Pure FSOT unadapted | 0% | ~0.9× | full |
| Pure FSOT adapted | 25% (↑ KL/top5) | ~0.95× | full |
| Blend adapted | **100%** | ~0.9× | partial (α~0.06–0.17) |
| Operator microbench | n/a | **beats CUDA SDPA** | full kernel |

**Next:** larger KL corpus + longer pure-FSOT train, or LoRA on QKV while freezing FSOT operator.
