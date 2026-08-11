# FSOT LLM construction board

**Definition of done (v1):**  
Open host + pure FSOT operators (or documented hybrid) + pin verify green + curriculum/plasticity under FSOT + held-out capability above a frozen floor + overfit gate + reproducible train/verify entrypoint — **without** claiming zero free parameters *inside the weight tensors*.

**Claims:** [`CLAIMS_SPLIT.md`](CLAIMS_SPLIT.md)  
**Climb constitution:** [`SOTA_STANDARDS.md`](SOTA_STANDARDS.md)  
**Live position:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md)  
**Last board refresh:** 2026-08-11  

**North star (hardware):** low-parameter same-class SOTA on RTX 5070 via FSOT substrate — see [`HARDWARE_CONSTRAINT_STRATEGY.md`](HARDWARE_CONSTRAINT_STRATEGY.md). Entry: `run_train.py --mode hardware_sota`.

---

## Workspace map (this machine)

| Location | GitHub | Role |
|----------|--------|------|
| `Desktop\gpu exparment for lean coq isabell andf star` | [FSOT-GPU](https://github.com/dappalumbo91/FSOT-GPU) | **Construction home** — CUDA ops, pure FSOT host, verify, climb |
| `Desktop\FSOT-2.1-Lean` | [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) | Theory authority / pin / atlas |
| `Desktop\fsot 2.1 llm` | [fsot-2.1-llm](https://github.com/dappalumbo91/fsot-2.1-llm) | Parallel LLM research product (Qwen path, folds) — **do not merge ontologies**; transfer operators only |
| `I:\FSOT-Physical-Archive\…` | (archive master) | Cross-proof stamp, solidification data |
| `D:\training data` | local | Capability packs + corpus |

**Sibling embodiments (curriculum feed later, not v1 blockers):** FSOT-Genetics, fsot quantum, neuron-zig (Track A).

---

## Non-negotiables (1–3)

| # | Item | Status | Where |
|---|------|--------|--------|
| 1 | **Pin bind** — every LLM run verifies vs FSOT-2.1-Lean / archive | **LIVE** — keep mandatory | `industry_lm/fsot21_verify.py` (V1–V7), auto-refine pre/post |
| 2 | **Split of claims** — residual = zero free; host weights = free under FSOT guide | **LOCKED** | [`CLAIMS_SPLIT.md`](CLAIMS_SPLIT.md) |
| 3 | **Standards climb only** | **LIVE** | [`SOTA_STANDARDS.md`](SOTA_STANDARDS.md) |

---

## A. Architecture (4–9)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 4 | **Host choice** | **FROZEN v1** | SmolLM2-135M-Instruct. **Next tier (explicit):** SmolLM2-360M or Qwen2.5-0.5B-class *same operators* after 135M pure stack stable. |
| 5 | **Attention operator** | **PURE default** | Consensus, no softmax. Fused SDPA allowed only under **hybrid** claim (mid-S speed track); forbidden for pure-FSOT fidelity claims. |
| 6 | **Collapse / coherence gates** | **LIVE** | `C_eff · P_var` seed-derived; gated attn + `fsot_lib` |
| 7 | **Trinary / packing** | **PARTIAL** | In GPU stack + CUDA; make mandatory in LLM **forward contract** (not optional switch) |
| 8 | **Layer policy** | **PARTIAL** | All-layer pure FSOT is production path; hybrid layer-swap exists. Need **one config language**. |
| 9 | **Tokenizer** | **v1 DECIDED** | Industry tokenizer unchanged. No new vocab block. |

---

## B. Learning process (10–14)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 10 | **FSOT curriculum engine** | **PARTIAL** | phase1 + v2 corpus; atlas *fold priority* not yet the sole scheduler |
| 11 | **Plasticity / loss modulation** | **PARTIAL** | suction–poof LR, digit anti-collapse loss; unify under one regime signal API |
| 12 | **Coherence-aware accept/reject** | **PARTIAL** | `overfit_metrics.accept_update` + climb scripts; single API still needed |
| 13 | **Anti-collapse objectives** | **IN PROGRESS** | Digit-1, ARC letter-D, format; digit script has anti-one hinge (local WIP) |
| 14 | **Held-out routing** | **PARTIAL** | Disjoint ARC/GSM splits; domain-label-respecting holds still thin |

---

## C. Data surface (15–17)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 15 | **Curriculum corpus vN** | **PARTIAL** | v1 + v2 chunks; formalize schema + regenerable hash |
| 16 | **Domain pack map** | **OPEN** | Tag packs → atlas domains for genetics/quantum later |
| 17 | **Leak policy** | **PARTIAL** | Hold splits exist; write hard denylist for residual benchmarks |

---

## D. Training & runtime infrastructure (18–21)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 18 | **Single train entrypoint** | **SCAFFOLD** | `industry_lm/run_train.py` → config-driven host + ops + curriculum + gates |
| 19 | **Checkpoint contract** | **PARTIAL** | Weights gitignored; need mandatory metadata (pin, mode, curriculum hash, scores) |
| 20 | **Auto-refine loop** | **LIVE** | Harden: never promote without G-VERIFY; barrier-named levers only |
| 21 | **Parity path** | **PARTIAL** | Microbench Py/Rust/Zig/CUDA green; **LLM-forward op parity** still open |

---

## E. Evaluation (22–26)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 22 | **Fidelity floor** | **HOLDING** | EVAL16 agree **100%** |
| 23 | **Speed ledger** | **HOLDING / OPEN mid-S** | Prefill/decode/long-S win; mid-S SDPA still ahead |
| 24 | **Capability holds** | **CLIMBING** | ARC min ~32.5%; GSM digit collapse partially diagnosed |
| 25 | **FSOT-native probes** | **OPEN** | Atlas reasoning / residual honesty / domain routing suite |
| 26 | **Overfit metric** | **LIVE** | `gen_score` / gap hard on promote |

---

## F. Scale path (27–30) — after 135M pure stable

| # | Item | Status |
|---|------|--------|
| 27 | Next open host size | **QUEUED** (360M / 0.5B class) |
| 28 | Mid-S attention kernel | **OPEN** |
| 29 | Multi-domain curriculum expansion | **QUEUED** — genetics/quantum only after train entrypoint + gates stable |
| 30 | Product surface (one cmd verify + sample) | **PARTIAL** — scripts exist; polish for outsiders |

---

## G. Explicitly separate (31–33)

| # | Item | Policy |
|---|------|--------|
| 31 | Neuron-zig pure bio mind | **Track A** — different ontology; no next-token SOTA requirement |
| 32 | Front-end / art / packaging | Pitch later; not construction |
| 33 | Full ToE social acceptance | Not a model build item |

---

## Suggested phase order (token budget)

| Phase | Items | Why |
|-------|-------|-----|
| **1** | 1–3, 18–20, 22–26 | Stop paying for runs that cannot promote |
| **2** | 10–14, 15–17 | FSOT steers learning |
| **3** | 5–8, 21, 28 | Operators + parity + mid-S |
| **4** | 27, 29–30 | Scale + domain packs |
| **Parallel low duty** | Track A | Don’t starve LLM path |

---

## Repo sync snapshot (2026-08-11)

| Repo | Local vs `origin/main` | Notes |
|------|------------------------|-------|
| **FSOT-GPU** | **Even** (0 ahead / 0 behind) | Dirty: digit de-collapse experiment + result ledgers (promote **False** on last run — do not claim as win) |
| **FSOT-2.1-Lean** | **Fast-forwarded** to `aeb0679` | Local `data/benchmark_margin_audit.json` still dirty |
| **FSOT-2.1-Lean-clean-repro** | Behind tip (snapshot clone) | Keep as clean-repro artifact unless you want full re-clone |
| **fsot-2.1-llm** | Dirty worktree, many untracked modules | Parallel track; transfer ops only ([`FSOT_GPU_TRANSFER.md`](file:///C:/Users/damia/Desktop/fsot%202.1%20llm/docs/FSOT_GPU_TRANSFER.md)) |

---

## Immediate next engineering (Phase 1)

1. Keep G-VERIFY mandatory on every train path (already wired; fail closed in `run_train.py`).  
2. Single config + entrypoint (`run_train.py` + `config/train_fsot_llm.yaml`).  
3. Checkpoint metadata schema written on every promote.  
4. Auto-refine: reject promote if verify red or lever does not name a barrier.  
5. Freeze eval floors in one ledger file for promote comparison.  

Capability barrier still open (do not burn tokens on new host size until entrypoint + gates are one path):

- Digit after space ≥ 45–50%, argmax-`1` < 50%, ARC min floor held  
- Then ARC letter-D collapse  
- Then free GSM exact > 0%
