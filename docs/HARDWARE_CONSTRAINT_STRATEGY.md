# Hardware-constrained low-parameter SOTA via FSOT

**Mission:** State-of-the-art *capability for the parameter class* on **this** box  
(RTX 5070 12 GB · ~32 GB RAM) — **not** “beat 70B by adding parameters.”

**Authority:** FSOT substrate math (operators, pin, gates) · host weights free under FSOT guide  
See [`CLAIMS_SPLIT.md`](CLAIMS_SPLIT.md).

---

## The constraint

| Resource | Reality | Industry default |
|----------|---------|------------------|
| VRAM | ~12 GB | Scale model until OOM |
| Params | 135M open host fits cleanly | Bigger is better |
| Tokens | Finite budget | Shuffle more web text |
| Time | One lab machine | Cluster train |

**Wrong answer:** buy a bigger GPU / bigger model first.  
**FSOT answer:** extract more **structure per parameter** and more **work per watt** on silicon.

---

## How FSOT gets around the constraint

### 1. Operators instead of width (substrate)

- **Consensus attention (no softmax exp)** — pure FSOT default; fidelity floor 100% agree on EVAL16.
- **Collapse θ = C_eff · P_var** — sparsity where structure allows (long-S already **wins** fused SDPA).
- **Trinary packing / coherence** — denser useful state on the same die (memory headroom → longer context / larger batch for the same VRAM).
- **Pin bind (G-VERIFY)** — refuse train that drifts off seed/archive authority (no wasted runs on broken theory bind).

Host stays **135M free weights**. FSOT does **not** claim those weights are seed-derived.

### 2. Learning efficiency instead of scale (plasticity)

- **Standards climb only** — promote when verify + overfit + capability improve.
- **Anti-collapse objectives** — digit / letter / format collapse burn capacity without skill; kill them first.
- **Curriculum from atlas folds** — train *what residual pressure says matters next*, not only shuffled packs.
- **Suction–poof LR** — FSOT-scheduled updates, not blind Adam.

A 135M model that **does not collapse** and is **curriculum-steered** can beat a same-class HF baseline (already: ARC min ~32% vs ~8%) without growing params.

### 3. Same-class SOTA definition (honest)

We claim **open same-class pure-FSOT SOTA** on this hardware:

| Axis | Bar |
|------|-----|
| Fidelity | Agree16 ≥ 90% (hold 100%) |
| Capability | ARC hold min ≫ same HF host; GSM space-digit / free exact climbing |
| Speed | Prefill/decode win; long-S win; mid-S closed later |
| Verify | G-VERIFY green every promote |
| Overfit | gen_score / gap accept |

We do **not** claim closed 100B leaderboard SOTA. We claim **more intelligence per parameter on constrained silicon**.

### 4. What we will **not** do under VRAM pressure

| Temptation | Why not (yet) |
|------------|----------------|
| Jump to 7B host | OOM / thrash; two frontiers at once |
| Softmax blend labeled pure FSOT | Breaks structure claim |
| Promote low-ARC digit labs onto production host | Destroys capability density |
| Genetics + quantum curriculum before train entrypoint stable | Token burn |

---

## Operating rules (hard)

1. **Production host path** = pure FSOT all-layer + **high ARC floor** (≥ ~30% hold min) + digit skill.  
2. **Lab digit runs** write `pure_fsot_digit_lab_*.pt` only — **never** overwrite `pure_fsot_sota_standard_best.pt` unless `arc_min ≥ 0.30`.  
3. **Transfer** = re-train digit anti-mode **on** the high-ARC host (or merge digit-row embeds under ARC retention) — not replace ARC host with a low-ARC lab.  
4. **One GPU process** for heavy train; verify pre/post mandatory.  
5. Next size tier (360M / 0.5B) only after 135M pure stack holds all gates.

---

## Current levers (priority)

| Priority | Lever | Why it multiplies 135M |
|----------|-------|-------------------------|
| P0 | Digit uncollapse on **high-ARC** host | Unlock GSM without growing params |
| P0 | Protect ARC min / gen_score | Capability density |
| P1 | ARC letter-D uncollapse | Free-gen diversity |
| P1 | FSOT curriculum folds | More signal per token |
| P2 | Mid-S attention kernel | Speed without bigger model |
| P2 | Trinary pack in forward | VRAM → context/batch |
| P3 | 360M same operators | Only after 135M stable |

---

## One-line north star

**Small open host · pure FSOT substrate · pin-green · non-overfit · held-out capability above frozen floor — punch above parameter class on this board.**
