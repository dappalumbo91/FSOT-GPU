# SOTA standard climb

**Status: IMPROVED — eligible to push**

Constitution: [`docs/SOTA_STANDARDS.md`](../../docs/SOTA_STANDARDS.md)

## Gates

| Gate | Pre | Post / Final |
|------|-----|--------------|
| G-VERIFY | PASS | PASS |
| G-OVERFIT gen_score | 0.292 | 0.319 (Δ +0.028) |
| G-OVERFIT gap | -7% | -10% |
| G-CAP arc_min | 28% | 32% |
| G-CAP gsm_first | 30% | 30% |
| G-CAP agree | 100% | 100% |

## Capability table

| Axis | Start | Final | Δ |
|------|-------|-------|---|
| ARC min | 28% | 32% | +5% |
| ARC-Easy hold | 30% | 33% | +3% |
| ARC-Challenge hold | 28% | 32% | +5% |
| GSM first-digit | 30% | 30% | +0% |
| GSM TF | 54% | 54% | +0% |
| GSM free exact | 0% | 0% | +0% |
| Balanced | 2.171 | 2.334 | +0.163 |
| gen_score | 0.292 | 0.319 | +0.028 |

Promote reasons: ['arc_min 27.5%→32.5%', 'both_arc_holds_up', 'balanced→2.334', 'gen_score_up']  
Elapsed: 1166s  
Architecture: pure FSOT all-layer · SmolLM2-135M · RTX-class host
