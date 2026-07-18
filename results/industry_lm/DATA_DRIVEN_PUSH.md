# Data-driven push v2

**Status: IMPROVED — pushed (confirmed 3× re-eval)**

Policy: train freely; **GitHub only when gates beat the prior best after multi-rep verify.**

## Gates (frozen holds)

| Axis | Prior best | Promoted | Δ | Confirmed |
|------|------------|----------|---|-----------|
| **ARC min** = min(Easy, Challenge) | **25.0%** | **27.5%** | **+2.5 pts** | yes (3/3 reps) |
| ARC-Easy hold | 31.7% | 30.0% | −1.7 pts | held above base HF |
| ARC-Challenge hold | 25.0% | **27.5%** | **+2.5 pts** | yes |
| GSM first-digit | 30% | 30% | 0 | flat |
| GSM TF | 53.8% | 53.8% | 0 | flat |
| GSM free exact | 0% | 0% | 0 | still collapsed |
| Balanced | 2.115 | **2.171** | **+0.056** | yes |
| Agree16 | 100% | 100% | 0 | floor held |

Start host: `pure_fsot_answer_locked_best.pt`  
Promote step: Phase A step 1 (Challenge-overweight letter CE + high retention)  
Verify: `industry_lm/verify_gates.py` (3 deterministic reps each)

## What we did

1. **Primary gate = `min(Easy hold, Challenge hold)`** — stop Easy-only climb that bleeds Challenge  
2. **GSM objectives = first-digit + TF** (not free exact alone)  
3. **Phase A** ARC bottleneck only (overweight weaker hold)  
4. **Phase B** LM-head first-digit (body frozen) — no further promote this run  
5. **Early stop** on arc_min regression streak  
6. **No push** on unconfirmed / regressed runs (v1 full-mix destroyed ARC → discarded)

## Host comparison (pre-train)

| Host | arc_min | Easy | Challenge |
|------|---------|------|-----------|
| answer_locked | **25%** | 32% | 25% |
| granular | 22% | 33% | 22% |
| 12x3 | 18% | 18% | **42%** |

## Ckpts (local)

- `pure_fsot_data_driven_best.pt` — promoted  
- `pure_fsot_granular_best.pt` — updated to same  

## Re-run

```bash
python -u industry_lm/run_data_driven_push.py
python -u industry_lm/verify_gates.py   # require CONFIRMED_IMPROVE before git push
```
