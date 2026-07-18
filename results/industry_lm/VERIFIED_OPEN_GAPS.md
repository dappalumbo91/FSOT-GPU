# Verified open-gap push

**Capability: NO_PUSH** (no gate beat prior best)  
**Verification: PASS** pre + post (FSOT 2.1 bridge)

## Why verification first

Open gaps (GSM first-digit, free exact, dual ARC holds) are refined only while the **FSOT 2.1 verification ledger stays green**. Accuracy without theory contract is not promotion.

## Verification layers (`industry_lm/fsot21_verify.py`)

| Layer | What | This run |
|-------|------|----------|
| V1 Archive stamp | Physical-Archive cross_proof + VERIFICATION_REPORT | **OK** (`overall_ok`, GREEN) |
| V2 Connective spine | Re-prove 24 exported obligations in Python | **OK** 24/24 |
| V3 Seed alignment | Lab `COLLAPSE_THRESHOLD` ↔ archive `C_EFF·P_VAR` | **OK** |
| V4 Formal artifacts | Lean / Coq / Isabelle / F\* present in phase1 | **OK** |
| V5 Owned stack | Scalar, trinary pack, coherence, consensus **no softmax** | **OK** |
| V6 Pure-FSOT host | All layers `FsotLlamaAttention`, finite forward | **OK** |

```bash
python -u industry_lm/fsot21_verify.py
python -u industry_lm/run_verified_open_gaps.py   # aborts if verify fails
```

## Capability (start = data_driven best, arc_min 27.5%)

| Axis | Start | After A/B | Promote? |
|------|-------|-----------|----------|
| ARC min | 27.5% | 27.5% (restored; train regressed) | no |
| GSM first-digit | 30% | 30% | no |
| GSM TF | 54% | 54% | no |
| Agree | 100% | 100% | held |
| Verify pre/post | PASS | PASS | held |

Phase A dual-ARC micro-LR still **dropped** arc_min within 50–100 steps → early stop + restore.  
Phase B head first-digit did not move first-digit; stopped on arc_min noise/drop.

## Policy

| Condition | Action |
|-----------|--------|
| Verify FAIL | Do not train / do not push |
| Verify PASS, capability flat or worse | **No GitHub capability promote** (this run) |
| Verify PASS + capability IMPROVED | Push |

## Next (still open, under green verify)

1. **ARC:** even softer updates (LoRA / last block only) or preference on wrong letter only  
2. **GSM first-digit:** constrained logit CE over digits 0–9 only at head  
3. Keep re-running `fsot21_verify.py` as the merge gate for every climb
