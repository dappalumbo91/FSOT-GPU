# Overfit audit

Brother's tidbit, made operational: **error that shows over to the system**
so we can prefer non-overfitting directions.

## Metric

| Symbol | Meaning |
|--------|---------|
| `err = 1 − acc` | Error on a split |
| `overfit_gap = acc_train − acc_hold` | Positive ⇒ practice ≫ fresh |
| `gen_score = mean_hold − penalty·max(0, gap)` | What we want to **maximize** |
| `overfit_flag` | Gap above threshold |

## Host ranking (by gen_score)

| Host | Hold acc | Overfit gap | gen_score | Flag |
|------|----------|-------------|-----------|------|
| data_driven_best | 29% | -7% | **0.292** | False |
| granular_best | 29% | -7% | **0.292** | False |
| 12x3_best | 30% | +3% | **0.272** | True |
| baseline_hf | 16% | +5% | **0.114** | True |

## Use in training

```python
from overfit_metrics import accept_update, direction_label
ok, reasons = accept_update(before=rep0, after=rep1)
if not ok: restore_checkpoint()  # curb overfit direction
```

Ledger: `overfit_audit.json`
