# Capability recovery — stop thrash, restore the real bar

## What went wrong

The lab **already had** pure-FSOT 135M capability on record:

| Axis | Documented (2026-07-18) | After thrash (audit) |
|------|-------------------------|----------------------|
| ARC min | **~32.5%** | 18–25% on remaining hosts |
| Agree16 | **100%** | often held |
| gen_score | **~0.319** | ~0.26–0.30 |
| vs HF ARC | **32.5% vs ~8%** | claim diluted |

Digit-only and barrier-only scripts **overwrote** `pure_fsot_sota_standard_best.pt` with weak labs. That is not FSOT. That is weight destruction.

## Frozen target (non-negotiable)

Until recovered, **success is distance to this package**, not a 22% “package score” story:

```
ARC min  ≥ 0.325
Agree16  ≥ 0.90  (hold 1.00)
gen_score ≥ 0.319
G-VERIFY PASS
pure FSOT all-layer attention
```

## Recovery protocol

1. **Do not load** polluted digit-phase standards as spine if live ARC &lt; 0.28.  
2. Start from **fidelity-pure** hosts: agree100 / fulldof / 12x3 (pure FSOT swap).  
3. Train with **proven** `run_sota_standard_climb` task mix + **FSOT suction–poof LR** + scalar loss scale.  
4. Promote to `pure_fsot_sota_standard_best.pt` **only** when ARC min ≥ 0.30 and agree ≥ 0.90.  
5. Digit uncollapse is **secondary** until ARC package is restored.

## Command

```powershell
python -u industry_lm/run_recover_capability.py
# or
python -u industry_lm/run_train.py --mode recover_capability
```

## Honesty

If the 32.5% weight file is gone from disk, recovery is **re-climb**, not nostalgia. The point of the tool is to **get those numbers back under pure FSOT**, using the theory spine, not invent new metrics that make 22% look like progress.
