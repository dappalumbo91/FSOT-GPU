# Answer-locked train (plateau break)

Protocol: **GSM #### gold tokens only → ARC letter only → merge**, pure FSOT all-layer host (SmolLM2-135M), FSOT LR (`lr0≈2.05e-5`).

## Scoreboard (honest #### / letter)

| | Baseline (HF) | Start (`12x3`) | GSM-best | ARC-best | Final (best ckpt) |
|--|---------------|----------------|----------|----------|-------------------|
| **GSM** | 2% | 0% | **0%** | — | **0%** |
| **ARC** | 12% | 35% | — | **32%** | **35%** |
| **Agree16** | — | 100% | 100% | 100% | 100% |

- Beats base ARC: **yes** (35% vs 12%)
- Beats base GSM: **no** (0% vs 2% under strict ####)
- ARC held >35% for 3 consecutive train evals: **no** (oscillates 12–35% under heavy CE; protect mode holds start)
- Checkpoint: `checkpoints/pure_fsot_answer_locked_best.pt` (merge step 1, arc=35%)

## What ran

1. **Phase A (GSM)** — multi-token TF on `####` + first-token + post-space digit CE; easy short-gold curriculum; anti-collapse LR bump.
   - Loss fell; free-gen stayed mode-collapsed (`1500000` / `1000000` / `1200000`…).
   - Stopped at step 600 after 6 zero evals; **did not promote collapsed weights**.
   - Reloaded clean `pure_fsot_12x3_best.pt` for ARC.

2. **Phase B (ARC)** — letter-only CE. Start was already **35%**, so **protect mode** (LR×0.25, high retention, max 600 steps).
   - First aggressive run (prior) collapsed ARC to **12%** if allowed to run full 1500+merge.
   - Protect run: drop floor hit → reloaded best (32%) and stopped.

3. **Phase C (merge)** — ARC-heavy (3:1), tiny LR, early stop on ARC drop.
   - Best at step 1: **ARC 35%**, GSM 0%, agree 100%.
   - Final metrics evaluated from **best checkpoint**, not last step.

## Miss-trace diagnosis

- GSM: `wrong_numeric_final` + mode collapse after `####` (same mega-number across almost all items).
- ARC: `wrong_choice_letter` only; no format collapse.
- Trails: `miss_traces/miss_gsm_step*.md`, `miss_arc_step*.md`, `miss_merge_step*.md`, `miss_answer_locked_final.md`.

## Conclusions (plateau)

| Finding | Detail |
|---------|--------|
| Loss ≠ metric | Answer-token CE drops while GSM free-gen stays 0% |
| GSM @ 135M pure FSOT | #### path collapses to a single numeric soup; capacity/format ceiling |
| ARC | Already **~3× baseline** from `12x3`; heavy answer-lock **hurts** hold; protect preserves 35% |
| Plateau break | **Not broken upward** on GSM; ARC **held at start ceiling** without regression |

## Next levers (not done here)

1. GSM format change (e.g. `Answer:` single-digit packs, constrained digit decoding) — not more CE on collapsed `####`.
2. Larger open host in same pure-FSOT stack if GSM #### is required.
3. ARC: diversity eval (held-out Easy/Challenge) + light preference/letter mix at LR floor only.
4. Scoreboard gate: publish 35% ARC / 100% agree / honest GSM 0% as current open-source same-class line.

```bash
python -u industry_lm/run_answer_locked.py
```
