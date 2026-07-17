# FSOT Formal-GPU — Research Blueprint

**Project working title:** FSOT Formal-GPU / `fsot_lib`  
**Theory:** Fluid Spacetime Omni-Theory (FSOT) — Damian Arthur Palumbo  
**Lab:** Desktop `gpu exparment for lean coq isabell andf star`  
**Archive authority:** `I:\FSOT-Physical-Archive` + public [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)  
**Status:** Active R&D — solidify → measure → exceed → then open-source  

**Out of scope for this repo:** FSOT-2.1-Instruct (separate LLM lab — do not merge claims or code paths without explicit decision).

---

## 1. Research thesis

> If GPU computation is specified by **FSOT seed geometry** and **cross-checked** in Lean, Coq, Isabelle, F\*, Rust, Zig, and Python, then the same contracts can be ported to **any** systems language with low effort — and the resulting kernels can be made **more correct and more efficient** for FSOT-native learning than industry float32/softmax defaults alone.

Two claims, separated deliberately:

| Claim type | Statement | Gate before public “SOTA” language |
|------------|-----------|-------------------------------------|
| **Portability** | Same constants + trinary + Φ + collapse across languages | Multi-language parity ledger `overall_ok` |
| **Capability** | FSOT-native GPU path meets or beats a named baseline on a preregistered metric | Competitive eval ledger with kill criteria |

We publish **portability** when the ledger is green. We claim **capability leadership** only after beat/exceed gates pass.

---

## 2. System architecture (blueprint)

```
                    ┌──────────────────────────────────┐
                    │  FSOT AUTHORITY (immutable seeds) │
                    │  π, e, φ, γ, Catalan → Φ engine   │
                    │  Archive fsot_compute + Lean hub  │
                    └────────────────┬─────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
   Formal twins              Runtime twins                 Device twins
   Lean · Coq                Python fsot_lib               CUDA sm_120
   Isabelle · F*             Rust fsot_math / kernel       Zig VRAM crystal
                             Ada/SPARK (reference)         Torch adapter
          │                          │                          │
          └──────────── cross-parity harness ───────────────────┘
                                     │
                                     ▼
                    results/*/parity_ledger.json
                    (fail-closed: no hand-waved “ok”)
```

### Owned modules (product)

| Module | Role | Industry thing it replaces |
|--------|------|----------------------------|
| Seeds / Φ scalar | Numeric truth | Free hyperparameters |
| Trinary + pack | State + memory density | Binary-only tensors |
| Collapse threshold | Measurement | Soft thresholds without theory |
| Consensus attention | Coupling | Softmax attention |
| Coherence norm | Stabilization | Learned LayerNorm affine |
| Suction–poof LR | Training dynamics | Free Adam schedule |
| Formal specs | Contracts | “Trust the framework” |

### Optional backends (not the product)

- PyTorch CUDA tensors  
- Native `nvcc` kernels (`-arch=sm_120`)  
- CPU pure Python  

---

## 3. Verification strategy

Same pattern as FSOT 2.1 cross-proof:

1. **Golden values** exported from Python `fsot_lib` + archive `fsot_compute`  
2. **Rust** re-reads constants (and optionally scalar)  
3. **Zig** re-implements pack + collapse counts  
4. **Lean / Coq / F\*** discharge packing / boot / memory lemmas  
5. **CUDA** round-trip pack on hardware  
6. Single JSON ledger: all must agree within ε or fail  

Portability theorem (informal):  
*If language L implements seeds, collapse, pack, and Φ within ε of the golden ledger, L is an FSOT GPU host.*

That is why “any coding language” is not marketing — it is **contract implementation**.

---

## 4. Competitive strategy (beat, then publish)

See `docs/COMPETITIVE_POSITION.md`.

Minimum path to **capability** claims:

1. Name baseline (e.g. PyTorch softmax attention TFLOPS-normalized utility, or pack density, or energy proxy).  
2. Preregister metric + dataset/synthetic task.  
3. Run A/B on same RTX 5070.  
4. Record win/lose/tie in ledger.  
5. Only then use language like “exceeds standard attention on …”  

Until then, public language is: **verified FSOT GPU contracts + working kernels**, not “best LLM training stack.”

---

## 5. Open source decision (deferred)

| Option | Pros | Cons |
|--------|------|------|
| **MIT** | Simple, maximum reuse | Weak patent language |
| **Apache-2.0** | Patent grant, enterprise-friendly | Slightly longer |

Recommendation when ready: **Apache-2.0** for a systems/GPU library (patent clarity), unless you prefer MIT for max simplicity.

**Do not** push a public repo until:

- [ ] License chosen  
- [ ] `docs/HOW_IT_WORKS.md` accurate  
- [ ] Parity ledger green on this machine  
- [ ] Competitive claims limited to measured facts  
- [ ] No accidental inclusion of FSOT-2.1-Instruct weights/adapters  
- [ ] README maps to archive + public FSOT-2.1-Lean without forking authority  

---

## 6. Roadmap (execution, not years)

| Milestone | Deliverable |
|-----------|-------------|
| **M0** | Lab scaffold, CUDA repair, `fsot_lib`, formal seeds — **done** |
| **M1** | Multi-lang parity harness green (Py/Rust/Zig/formal/CUDA) |
| **M2** | Φ tile + consensus hot path in native CUDA |
| **M3** | Preregistered competitive microbench vs softmax baseline — **Round 01 done** (stability win, throughput open) |
| **M4** | Public GitHub release (license + docs + CI parity) |
| **M5** | Scale FSOT cortical train; optional link to other FSOT labs |

---

## 7. Directory map (this lab)

```
fsot_lib/                 owned runtime library
phase1_formal_gpu/        Lean · Coq · Isabelle · F*
phase2_native_gpu/        CUDA + host engines
phase0_baseline/          hardware probes
docs/                     blueprint, how/why, competitive, owned stack
parity/                   cross-language checkers
results/                  machine-readable ledgers
scripts/                  env, build, smoke
```
