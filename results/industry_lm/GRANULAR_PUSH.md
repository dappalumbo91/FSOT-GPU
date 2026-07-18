# Granular metrics push

Multi-axis accuracy — not only headline GSM% / ARC%.

## Why more metrics?

Headline **exact match** hides signal:

| What 0% free-gen hid | Granular axis that still moves |
|----------------------|--------------------------------|
| “GSM is totally broken” | **TF token acc ~55%** (better than HF baseline 45%) |
| Same | **First-digit match ~30%** (not random ~10%) |
| Same | **Format-ok ~100%** (emits a number; wrong one) |
| “ARC 35%” on first-40 | **Held-out Easy 20→33%**; **Challenge hold** separate |
| Mode soup | **Mode collapse flag** + mode string + fraction |

Training on only free-gen exact is blind when free-gen is collapsed but TF/first-digit still carry gradient signal.

## Metric dictionary

| Axis | Meaning |
|------|---------|
| **GSM free exact** | First number in free-gen after `####` == gold |
| **GSM first-digit** | Leading digit of pred matches gold |
| **GSM format-ok** | Any parseable number produced |
| **GSM len-match** | Digit length of pred == gold |
| **GSM constrained exact** | Greedy decode restricted to space/0-9/newline == gold |
| **GSM TF token / first** | Teacher-forced gold digit accuracy (format learning) |
| **Mode collapse** | ≥40% of free-gen preds are the same string |
| **ARC free exact** | Letter free-gen == gold |
| **ARC first-token letter** | Argmax next token is the correct letter form |
| **ARC TF first** | Teacher-forced first letter token correct |
| **Held-out ARC** | Shuffled hold of Easy/Challenge (not first-40 only) |
| **Legacy40** | First 40 Easy rows (old scoreboard, easy to overfit) |
| **Composite** | Weighted multi-axis (ARC free + TF + GSM constrained/TF/first + agree) |

Code: `industry_lm/granular_metrics.py`  
Train: `industry_lm/run_granular_push.py` (+ `run_granular_balance.py`)

## Scoreboard (n≈40 GSM / 60 Easy hold / 40 Challenge hold)

| Axis | Baseline HF | Start FSOT | Best FSOT |
|------|-------------|------------|-----------|
| Agree16 | 100% | 100% | **100%** |
| GSM free exact | 2% | 0% | **0%** |
| GSM first-digit | 28% | **30%** | **30%** |
| GSM format-ok | ~98% | ~100% | ~100% |
| GSM constrained | 2% | 0% | **0%** |
| GSM TF token | 45% | **55%** | **54%** |
| ARC-Easy **hold** | 8% | 20% | **32–33%** |
| ARC-Easy legacy40 | 12%* | 35% | ~20–35% (oscillates) |
| ARC-Challenge hold | 12% | **40%** | **22–25%** (protect next) |
| Composite | — | 2.29 | **2.63** |

\*Baseline legacy ARC on Easy first-40 was ~12% in prior answer-locked runs; hold slice here is harder (8%).

### Wins this push
- **ARC-Easy held-out: 20% → 32%** (+12 pts), still **≫ base 8%**
- **Composite 2.29 → 2.63**
- **GSM TF > HF baseline** (55% vs 45%) — pure FSOT host *does* learn digit TF better
- First-digit **30%** is a real partial-credit signal (exact stays 0%)

### Still broken
- Free-gen GSM: **mode collapse** (`1200000` / `1000000` soup) → exact 0%, constrained 0%
- Heavy ARC CE trades **Challenge hold** down from a lucky/high start (40% → ~22%)
- Legacy40 ≠ held-out (overfit risk on first-40)

## Deltas (granular push best @ step 700)

| Delta | Value |
|-------|-------|
| ARC-Easy hold | **+12 pts** (20→32%) |
| ARC-Challenge hold | −15 pts (40→25%) — follow-up balance kept best E=33% C=22% |
| GSM free exact | 0 |
| GSM constrained | 0 |
| GSM TF | ~flat ~54% |
| Agree | held 100% |

## How to re-run

```bash
python -u industry_lm/run_granular_push.py
python -u industry_lm/run_granular_balance.py
```

Ckpt: `results/industry_lm/checkpoints/pure_fsot_granular_best.pt` (local, gitignored)

## Next levers (data-driven)

1. **Score first-digit + TF as train objectives / gates** — free exact alone is mute under collapse  
2. **Constrained decode at inference for GSM** once first-digit >50% (still 0% exact under constraint today → need stronger digit ranking)  
3. **ARC multi-set gate**: maximize `min(Easy_hold, Challenge_hold)` not Easy alone  
4. **Larger host or RAG-free chain** only if #### free exact is hard gate at 135M  

## JSON ledger

`results/industry_lm/granular_push.json` — full baseline/start/final + metric definitions + history.
