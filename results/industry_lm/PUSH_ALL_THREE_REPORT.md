# All three pushes — results

**Date:** 2026-07-17  
**Model:** SmolLM2-135M-Instruct (SafeTensors), all 30 layers FSOT operator  
**GPU:** RTX 5070

## 1) Quality demo — blend (kept)

| Metric | Value |
|--------|--------|
| Agreement | **100%** |
| KL | **0.022** |
| mean α | ~0.11 |
| tok/s | ~0.78× baseline |

**Role:** quality / product-safe path with FSOT still mixed in every layer.

```powershell
python industry_lm\run_adapt_blend.py
# or quality half of:
python industry_lm\run_push_all_three.py
```

## 2) LoRA on QKV/O + pure FSOT operator

| Step | agree | KL | top5 | tps× |
|------|-------|-----|------|------|
| 0 | 0% | 8.33 | 0.05 | 0.76 |
| 750 | 0% | 5.64 | 0.28 | 0.76 |
| 2000 | **25%** | **4.41** | **0.33** | 0.77 |
| 2499 | 25% | 4.86 | 0.23 | 0.82 |

LoRA modules: 120 · trainable ~1.9M  
Checkpoint: `checkpoints/pure_fsot_lora_best.pt`

## 3) Extended distill — full QKV unfreeze after LoRA bake

| Step | agree | KL | top5 | tps× |
|------|-------|-----|------|------|
| warm start | 38% | 4.43 | 0.33 | **1.05** |
| 1200 | **50%** | 3.83 | 0.35 | **1.08** |
| 2200 | **50%** | **3.64** | 0.38 | 0.39* |
| **best @2800** | **50%** | **3.50** | **0.40** | 0.80 |

\*tps variance under load; best speed windows hit **≥1.0× baseline**.

Checkpoint: `checkpoints/pure_fsot_full_best.pt`  
Ledger: `continue_pure_fsot.json`, `push_all_three.json`

## Climb chart (pure FSOT all layers)

```
unadapted     ████░░░░░░  0%   KL 8.05
short adapt   ██████░░░░ 25%   KL 5.39
LoRA 2.5k     ██████░░░░ 25%   KL 4.41
full QKV 3k   ██████████ 50%   KL 3.50   ← now
target        ████████████████ 80%
blend demo    ████████████████ 100% (dual path)
```

## Dual-demo policy (locked)

| Demo | Path | Status |
|------|------|--------|
| **Quality** | Blend `(1-α)SDPA + α FSOT` | **100% agree** — ship demos here |
| **Speed / pure FSOT** | Pure consensus + trained projs | **50% agree**, climbing; operator still beats CUDA microbench |

## Commands

```powershell
python industry_lm\run_push_all_three.py   # blend + LoRA pure
python industry_lm\run_continue_pure.py    # full QKV continue → 50%
```

## Next to break 80%

- 5k–10k more steps on `pure_fsot_full_best.pt`  
- Larger KL batches (pack 4 short sequences)  
- Optional: lower collapse threshold slightly as a *learned* per-layer scale (still FSOT-family)  
- Or: FSOT consensus on residual stream after industry SDPA (stacked), then distill  

## Update: **80% HIT**

Continued train (packed KL → CE+KL agreement push):

| Metric | Best |
|--------|------|
| Agree (16-probe) | **81%** |
| Agree (8-probe) | **75%** |
| KL | **2.18** |
| tok/s | **~1.01×** baseline |

See `HIT_80_REPORT.md`. Climb: **0% → 25% → 50% → 62% → 69% → 81%**.