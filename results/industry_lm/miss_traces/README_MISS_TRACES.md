# Miss traces (wrong-answer audit)

Generated: 2026-07-18T03:57:25.258535+00:00

## How to use

Open the **`.md`** file for a readable trail of every miss:

1. **QUESTION / PROMPT** — what was asked  
2. **GOLD ANSWER** — correct target  
3. **MODEL PRED** — parsed answer  
4. **MODEL THOUGHT / GENERATION** — full free-gen path (how it tried)  
5. **DIAGNOSIS** — tags (empty gen, wrong number, regurgitated question, …)

## Files

| Arm | Human log | JSONL | Summary |
|-----|-----------|-------|---------|
| Pure FSOT | [`miss_trace_fsot.md`](miss_trace_fsot.md) | `miss_trace_fsot.jsonl` | `miss_trace_fsot_summary.json` |
| Baseline | [`miss_trace_baseline.md`](miss_trace_baseline.md) | `miss_trace_baseline.jsonl` | `miss_trace_baseline_summary.json` |

## Snapshot scores

**FSOT:** {'gsm8k': {'n': 40, 'acc': 0.0, 'misses': 40}, 'arc': {'n': 40, 'acc': 0.275, 'misses': 29}, 'math': {'n': 30, 'acc': 0.06666666666666667, 'misses': 28}}  

**Baseline:** {'gsm8k': {'n': 40, 'acc': 0.025, 'misses': 39}, 'arc': {'n': 40, 'acc': 0.125, 'misses': 35}, 'math': {'n': 30, 'acc': 0.1, 'misses': 27}}

Ckpt: `C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star\results\industry_lm\checkpoints\pure_fsot_realdata_best.pt`
