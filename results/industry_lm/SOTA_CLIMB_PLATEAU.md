# SOTA climb plateau note (post e69208c best)

**Current best (held):** `pure_fsot_sota_standard_best.pt`  
- ARC min **32.5%** (Easy 33.3% / Challenge 32.5%)  
- gen_score **0.319**, overfit gap **−10%**, agree **100%**  
- GSM first-digit **30%**, free exact **0%**  
- Verify **PASS**

## Attempts this session (all NO_PUSH — standards held)

| Run | Idea | Result |
|-----|------|--------|
| digit-vocab CE (full head) | Rank among 0–9 | ARC min collapsed → overfit reject restore |
| digit-row mask + head | Mask non-digit lm_head rows | Still ARC bleed; first-digit flat 30% |
| tied-embed digit+letter mask | Surgical embed rows only | min dips 32→28; restored; first flat |
| ARC-only letter-row | No GSM CE | min cannot break 32.5% without hold regress |

**Conclusion:** 32.5% ARC min is a **local peak** under pure embed-row CE. Further letter/digit CE on the tied embedding walks **overfit / hold-regress** directions that standards correctly reject.

## What still works

- G-VERIFY green every run  
- G-OVERFIT rejects memorize steps  
- G-PUBLISH blocked without true improve  

## Next levers (not yet run)

1. **Do not CE-touch tied embed** for ARC — train a small side adapter (LoRA on last block) with overfit accept  
2. **GSM:** constrained decode scoring + separate digit probe head (not shared embed)  
3. **Data:** more diverse held-out; larger pure-FSOT host same stack  
4. **FSOT 2.1 curriculum** literacy path in parallel (doesn't risk ARC min)  

Policy: keep best ckpt; no false promote.
