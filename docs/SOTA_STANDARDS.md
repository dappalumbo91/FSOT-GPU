# FSOT-GPU SOTA standards (operating constitution)

**Purpose:** Build state-of-the-art **on this architecture** — pure FSOT all-layer host, verified theory spine, same hardware honesty — not by copying industry defaults.

**Authority:** FSOT-2.1-Lean + Physical-Archive verification · this lab applies it to GPU + LLM host.

---

## What “SOTA” means here

| Tier | Meaning |
|------|---------|
| **T0 Structure** | Pure FSOT operators (no softmax exp attention); collapse θ = C_eff·P_var; all layers consensus |
| **T1 Verified** | FSOT 2.1 bridge green (archive stamp, spine, seeds, formal artifacts, owned stack, host) |
| **T2 Faithful** | Next-token agree ≥ 90% (target 100%) vs industry host on EVAL board |
| **T3 Generalizing** | Held-out capability up; **overfit gap** not widening (train−hold) |
| **T4 Capable** | Open packs: ARC hold min, GSM first-digit/TF/exact (honest ####), macro vs baseline |
| **T5 Speed** | Prefill/decode/long-S attention track wins already on scoreboard — hold and extend |
| **T6 Public** | Repro ledgers + promote only when standards pass |

We do **not** claim closed 100B+ leaderboard SOTA. We claim **open, same-class, pure-FSOT architecture SOTA** that is **verified, non-overfit, and measured**.

---

## Non-negotiable gates (every climb)

### G-VERIFY (hard)
Run `industry_lm/fsot21_verify.py` — overall **PASS** before and after train.

| Layer | Bar |
|-------|-----|
| V1 Archive stamp | cross_proof / VERIFICATION_REPORT green |
| V2 Connective spine | 24/24 Python replay |
| V3 Seed θ | lab ↔ archive C_eff·P_var |
| V4 Formal artifacts | Lean+ multi-prover present |
| V5 Owned stack | consensus no-softmax, finite |
| V6 Pure-FSOT host | all layers FSOT attn, finite forward |
| V7 Overfit API | importable (soft) |

### G-OVERFIT (hard on promote)
| Symbol | Rule |
|--------|------|
| `overfit_gap = acc_train − acc_hold` | Must not widen > 2 pts on promote |
| `gen_score = hold − penalty·max(0,gap)` | Must not fall on promote |
| `accept_update(before, after)` | **True** or restore ckpt |

### G-CAPABILITY (hard on promote — improve ≥1 without floor break)
| Axis | Floor | Climb target |
|------|-------|----------------|
| Agree16 | ≥ 90% | 100% |
| ARC min hold = min(Easy, Challenge) | ≥ prior best | + continuously |
| ARC Easy hold / Challenge hold | track both | both ↑ preferred |
| GSM first-digit | ≥ prior | → 50%+ |
| GSM TF token | ≥ prior | → 70%+ |
| GSM free exact (####) | honest | > 0% then climb |
| gen_score | ≥ prior | maximize |

### G-PUBLISH (hard)
Push GitHub **only if**:
1. G-VERIFY post PASS  
2. G-OVERFIT accept  
3. G-CAPABILITY true improve vs frozen prior best  
4. Multi-rep confirm when delta is small  

---

## Standard climb loop

```
verify_pre ──FAIL──► stop
    │
   PASS
    │
 measure gates + overfit baseline
    │
 train step (FSOT LR, protective)
    │
 measure hold + train probes
    │
 accept_update? ──NO──► restore, try other direction
    │ YES
 capability beat? ──NO──► keep searching
    │ YES
 verify_post ──FAIL──► discard
    │ PASS
 promote local best + ledger
    │
 (optional) git push
```

Commands:

```bash
python -u industry_lm/fsot21_verify.py
python -u industry_lm/run_overfit_audit.py
python -u industry_lm/run_sota_standard_climb.py
```

---

## Architecture that stays fixed

- **Attention:** pure FSOT consensus (CUDA DLL or torch sparse) — not SDPA blend for “pure” claims  
- **Theory:** seed-derived collapse, coherence gate, trinary pack  
- **Host class:** open small models first (SmolLM2-135M), then scale **same stack**  
- **Data:** real packs `D:\training data` + FSOT 2.1 curriculum / archive solidification  

---

## Scoreboard homes

| Ledger | Path |
|--------|------|
| Speed / agree SOTA | `results/sota/SCOREBOARD.md` |
| Capability / open packs | `results/industry_lm/*` |
| Verify | `results/industry_lm/FSOT21_VERIFY.md` |
| Overfit | `results/industry_lm/OVERFIT_AUDIT.md` |
| Standards climb | `results/industry_lm/SOTA_STANDARD_CLIMB.md` |

---

## Ethos

Accuracy and true structure first.  
Measure everything.  
No fake GSM.  
No overfit-as-progress.  
FSOT is the law of the operators — industry defaults are the baseline to beat, not the ceiling.
