# Experiment Plan — FSOT Formal GPU → LLM

## Success criteria (honest)

| Horizon | Pass condition |
|---------|----------------|
| Phase 0 | GPU probe OK; Φ GPU matches CPU within 1e-5 relative (f32) / 1e-12 (f64 where available); formal toolchains smoke green |
| Phase 1 | At least one packing lemma proved in Lean; F\* boot scalar matches archive canonical `0.09928895626861721` |
| Phase 2 | Custom CUDA trinary pack+unpack round-trips; Φ kernel within formal tolerance; bandwidth/util report vs PyTorch baseline |
| Phase 3 | Toy FSOT-native model trains on small corpus with obligation gate; not yet “frontier LLM” |

Building a frontier LLM is multi-year. This plan ships **verifiable intermediate science** at each phase.

---

## Phase 0 — Baseline (now)

1. Inventory: CUDA, PyTorch, nvcc, Lean, Coq, Isabelle, F\*, archive paths  
2. `probe_gpu.py` — device props, matmul TFLOPS-ish, memory alloc  
3. `fsot_scalar_gpu.py` — FSOT boot scalar on CPU (reference) and GPU  
4. Smoke formal binaries from `07_Portable-Toolchain`  
5. Write `results/phase0/baseline_report.json`

## Phase 1 — Formal GPU model

1. Lean: `Trinary.lean` — type `Trinary := Zmod 3` or inductive ±1/0; packing to `UInt64` bitfields  
2. Lean: `GpuMemory.lean` — abstract device heap, sector ownership  
3. Coq: extract or re-prove packing injectivity (Flocq optional later)  
4. Isabelle: warp size divisibility lemmas (32 / 64)  
5. F\*: import/link `FSOTScalarKernel` boot value as oracle for GPU tests  
6. Export JSON obligations → Python checker (same pattern as archive cross-proof)

## Phase 2 — Native GPU access

1. CUDA: `trinary_pack.cu` / `trinary_unpack.cu`  
2. CUDA: `fsot_scalar_tile.cu` — vectorized Φ or reduced form  
3. Python: ctypes / pybind / cupy bridge  
4. Compare: PyTorch matmul baseline vs FSOT kernels on same data  
5. Optional: re-host Zig VRAM allocator concepts as CUDA memory pool

## Phase 3 — Path to FSOT LLMs

1. Minimal transformer block with FSOT-modulated layer (start tiny: 1–4 layers, small d_model)  
2. Obligation checks every N steps against FSOT seed predictions  
3. Co-train continuous LTM stream (SR-ITE style) with discrete token path  
4. Scale only after gates stay green  
5. Document where formal methods *actually* improve utilization (memory packing, reduced precision waste, correctness-first kernels)

---

## Method notes

- **Always compare** against a PyTorch baseline on the same GPU — “better” must mean measurable (throughput, energy proxy, error, or proof strength).  
- **Never claim** formal verification of all of CUDA; claim verification of *our contracts* and *numeric obligations*.  
- **Reuse** archive cross-proof runner patterns; do not reinvent the seven-way gauntlet.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Blackwell (sm_120) tooling lag | Use PyTorch wheels that already see the device; prefer PTX/JIT where driver allows |
| Formal ≠ fast | Spec first, then profile; keep float path for bulk ops |
| Scope explosion | Phase gates; toy models before “LLM” branding |
| Path portability | `config/paths.json` + `set_env.ps1` |

---

## Immediate next actions after Phase 0

1. Prove Lean trinary pack injectivity  
2. Wire F\* boot scalar into `fsot_scalar_gpu.py` golden check  
3. Write first CUDA pack kernel skeleton  
