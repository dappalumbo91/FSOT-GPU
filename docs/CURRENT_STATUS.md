# FSOT-GPU — current status & roadmap

**Last updated:** 2026-07-18  
**Repo:** [dappalumbo91/FSOT-GPU](https://github.com/dappalumbo91/FSOT-GPU)  
**Theory authority:** [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)  
**Hardware (this lab):** NVIDIA GeForce RTX 5070  

This document is the **public position**: where we are, what is measured, what is blocked, and what we do next. Numbers are honest (held-out, multi-axis, no leaky GSM scoring).

---

## One-line position

**Pure FSOT all-layer attention is production-ready for fidelity and long-context speed on a tiny open host.** Capability climb is **standards-gated** (verify + overfit + holds). ARC held-out **min ≈ 32.5%** (≫ HF ~8%); GSM free-gen is still **digit-collapsed** after `####` — root cause identified and partially broken.

---

## Architecture (fixed)

| Piece | Status |
|-------|--------|
| Attention | **Pure FSOT consensus** (no softmax exp) — all layers swapped on industry host |
| Collapse θ | `C_eff · P_var ≈ 0.917` (seed-derived, aligned with archive `fsot_compute`) |
| Host class | SmolLM2-135M-Instruct, full DoF, pure FSOT |
| Theory bind | FSOT 2.1 verification bridge (archive stamp + spine + host) |
| Product split | This repo = **GPU + host**; **FSOT-2.1-Instruct** is separate unless linked |

---

## Scoreboard — where we are now

### A. Structure & speed (won / holding)

| Track | Result | Ledger |
|-------|--------|--------|
| Next-token agree (EVAL16) | **100%** vs industry host | `results/sota/SCOREBOARD.md` |
| Prefill / decode | Prefill **~1.09–1.26×**, decode **≥1×** baseline | same |
| Long-context attention (S≥4096) | **WIN** (up to ~1.6× at S=8192) | same |
| Mid-S fused SDPA | Still industry sweet spot | **OPEN** |
| Sparse vs dense softmax CUDA | Up to **~89×** | competitive suite |

### B. Verification & process (operating system)

| Gate | Status | Tool |
|------|--------|------|
| G-VERIFY (FSOT 2.1 bridge V1–V7) | **PASS** | `industry_lm/fsot21_verify.py` |
| G-OVERFIT (`gen_score`, train−hold gap) | **PASS (API + audits)** | `industry_lm/overfit_metrics.py` |
| G-CAPABILITY (held-out ARC/GSM) | **Climbing** | `run_sota_standard_climb.py` |
| G-PUBLISH | Push only on real improve | `docs/SOTA_STANDARDS.md` |

### C. Open capability packs (honest)

**Host checkpoint (local, gitignored):** `pure_fsot_sota_standard_best.pt`  
(Also promoted via digit de-collapse for space-digit metric.)

| Axis | Pure FSOT (best) | HF baseline (same holds) | Note |
|------|------------------|---------------------------|------|
| Agree16 | **100%** | 100% (self) | Fidelity floor |
| ARC-Easy hold | **~33%** | **~8%** | Shuffled hold, not first-40 only |
| ARC-Challenge hold | **~32.5%** | **~12%** | |
| **ARC min** = min(E,C) | **~32.5%** | **~8%** | Primary ARC gate |
| gen_score (overfit) | **~0.32** | lower | Hold quality − gap penalty |
| GSM free first-digit | **~30%** | ~28% | Misleading if collapsed |
| GSM digit after `####`+space | **35%** (was 30%) | — | **True** first-digit signal |
| Argmax digit after space | was **100% → `1`**; now **~80% `1`** | — | Collapse cracking |
| GSM free exact (`####`) | **0%** | ~2% | Still collapsed soup |
| GSM TF first token | **100% space** | — | Format OK; not digit skill |

**Do not read “TF first 100%” as digit mastery** — that token is usually **leading space**. Digit skill is measured **after** forced space.

### D. Diagnosed barriers (current blockers)

| Priority | Barrier | Evidence | Status |
|----------|---------|----------|--------|
| **1** | Digit argmax collapse after space | After `####`+space, argmax was **always `1`** (40/40) → free soup `1200000…` | **Partially broken** (1@100%→~80%; space-digit 30%→35%) |
| **2** | ARC free-gen letter collapse | ~**80%** predictions **D** on Easy hold | Open |
| **3** | Hold noise | Bootstrap arc_min ~20–38% on n≈40–60 | Mitigate with clear deltas + multi-rep |
| **4** | Mid-S attention | Fused SDPA still wins mid lengths | Open (speed track) |

Full write-up: [`results/industry_lm/BARRIER_DIAGNOSIS.md`](../results/industry_lm/BARRIER_DIAGNOSIS.md)

---

## How we climb (standards)

Constitution: [`docs/SOTA_STANDARDS.md`](SOTA_STANDARDS.md)

```
verify_pre → measure (capability + overfit) → train → accept_update?
  → capability beat? → verify_post → promote → (optional) git push
```

```powershell
python -u industry_lm/fsot21_verify.py
python -u industry_lm/run_barrier_diagnosis.py
python -u industry_lm/run_sota_standard_climb.py
python -u industry_lm/run_sota_digit_decollapse.py
```

**Policy:** no GitHub capability claim unless G-VERIFY + G-OVERFIT + G-CAPABILITY all improve (or an explicit barrier metric like space-digit under ARC floor).

---

## Where we are going next

### Near-term (same 135M pure FSOT host)

1. **Finish digit de-collapse**  
   - Target: digit-after-space **≥45–50%**, argmax-`1` fraction **&lt;50%**  
   - Hold ARC min ≥ 32% and agree ≥ 90%  
   - Then free-gen first-digit / free exact should start to move  

2. **ARC letter D collapse**  
   - Letter-only softmax + balanced sampling **or** LoRA last-block (avoid fragile full embed CE)  
   - Goal: free-gen letter entropy up; ARC min **&gt;35%** with 3-eval hold  

3. **Larger holds / multi-rep**  
   - Reduce promote false positives from ±9% noise  

### Mid-term (same architecture, scale stack)

4. **FSOT 2.1 curriculum path** (literacy / solidification data) — parallel, low risk to ARC peak  
5. **Larger open pure-FSOT host** (same layer-swap + consensus) once 135M plateaus on holds  
6. **Mid-S attention** kernel path (close G6)  

### North star

**State-of-the-art open, same-class pure-FSOT systems** — verified structure, non-overfit direction, held-out capability, and speed where FSOT sparsity wins — not closed 100B leaderboard theater.

---

## Key docs & ledgers

| Doc | Purpose |
|-----|---------|
| [GOALS.md](GOALS.md) | Mission & non-negotiables |
| [SOTA_STANDARDS.md](SOTA_STANDARDS.md) | Climb constitution |
| [OPEN_SOURCE_SOTA_GATES.md](OPEN_SOURCE_SOTA_GATES.md) | Gate board |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | **This page** — live position |
| `results/sota/SCOREBOARD.md` | Speed / agree SOTA |
| `results/industry_lm/BARRIER_DIAGNOSIS.md` | Plateau root cause |
| `results/industry_lm/SOTA_DIGIT_DECOLLAPSE.md` | Digit collapse breakthrough |
| `results/industry_lm/FSOT21_VERIFY.md` | Last verify ledger |

---

## Checkpoint note

Best weights live under `results/industry_lm/checkpoints/` (**gitignored** — too large for git).  
Reproduce via scripts + data roots (`D:\training data`, FSOT 2.1 archive paths). Recipe: pure FSOT swap-all + `pure_fsot_sota_standard_best.pt` / digit de-collapse promote.
