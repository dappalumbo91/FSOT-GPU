# Barrier diagnosis at SOTA peak

**Primary barrier (refined):** `DIGIT_ARGMAX_COLLAPSE_AFTER_SPACE`  
**Secondary:** `ARC_LETTER_MODE_COLLAPSE` (~80% free-gen **D**)  
**Also:** hold noise ~±9% on n=40–60

## Smoking-gun measurements (peak host)

| Probe | Result | Meaning |
|-------|--------|---------|
| Agree16 | 100% | Fidelity OK |
| ARC min FSOT vs HF | **32.5% vs 8%** | Capability real vs baseline |
| After `####` next token | **space 100%** | Format OK |
| TF “first” of gold `" 18"` | **100%** | Was measuring **space**, not digit |
| After `####`+space argmax | **always `1` (40/40)** | **Root GSM barrier** |
| Free first-digit | 30% | ≈ P(gold starts with 1) under collapse |
| ARC free pred | ~80% **D** | Letter mode collapse |
| Boot arc_min 90% band | 20%–38% | Noisy holds — need clear deltas |

## Ranked barriers

1. **DIGIT_ARGMAX_COLLAPSE_AFTER_SPACE** (sev 1.0)  
   After correct space, model always picks digit token `1` → free-gen soup `1200000…`.  
   → Lever: digit-only CE after forced space, row-mask 0–9, balanced first-digit curriculum.

2. **ARC_LETTER_MODE_COLLAPSE** (sev 0.8)  
   Free-gen ~80% **D** on Easy hold.  
   → Lever: letter-only softmax + smoothing + balanced letters (fragile on tied embed).

3. **EVAL_NOISE** (sev 0.5)  
   Bootstrap half-width ~9 pts on arc_min.  
   → Larger holds + multi-rep promote.

4. **NARROW_HOLD_RIDGE**  
   Tiny letter CE drops hold 33%→25%.  
   → Don’t CE-touch shared embed for ARC without LoRA.

## Breakthrough attempt: digit de-collapse

| Metric | Before | After (promoted) |
|--------|--------|------------------|
| Digit after space | 30% | **35%** |
| Argmax `1` fraction | **100%** | **80%** (also 2,4 appear) |
| Free first-digit | 30% | 30% (still soup mode) |
| ARC min | 32.5% | **32.5% held** |
| Agree | 100% | 100% |

Collapse **started to break** without sacrificing ARC. Next: continue digit curriculum until argmax diversifies further and free-gen first-digit moves.

## Commands

```bash
python -u industry_lm/run_barrier_diagnosis.py
python -u industry_lm/diagnose_digit_tf.py
python -u industry_lm/run_sota_digit_decollapse.py
```
