# Digit de-collapse

**Barrier:** after `####`+space, argmax was **always `1`** (40/40).  
Free first-digit ~30% ≈ share of golds starting with 1.  
TF “first” was **space** (100%), not the digit.

| Metric | Start | Final |
|--------|-------|-------|
| Digit after space | 30% | 35% |
| Argmax top | 1@100% | 1@80% |
| Free first-digit | 30% | 30% |
| ARC min | 32% | 32% |
| gen_score | 0.319 | 0.319 |

**Promote:** True
