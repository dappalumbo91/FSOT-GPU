# Architecture — FSOT Formal GPU

## 1. Problem framing

Conventional LLM training:

```
Text → token ids → float tensors → matmul / attention → softmax → loss → Adam → GPU kernels
```

Every step is optimized for binary floating-point throughput. Correctness is empirical (loss goes down). Memory layout is opaque (allocator + cuBLAS heuristics).

FSOT Formal-GPU inverts the priority:

```
FSOT axioms (seeds)
  → formal obligations (Lean/Coq/Isabelle/F*)
  → GPU memory & kernel contracts
  → executable kernels (CUDA / Zig / Rust)
  → learning dynamics that preserve contracts
```

PyTorch remains the *integration bus* (data loading, autograd for float paths, profiling). Authority for FSOT scalars and trinary packing is the formal stack + mirrored runtime.

---

## 2. Layers

### L0 — Theory spine (existing)

From `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full`:

- Seed constants: π, e, φ, γ, Catalan
- Scalar Φ engine (`FSOT.Scalar`, F\* `FSOTScalarKernel`)
- Domain obligations (cross-proof runner)
- Trinary OS / hardware motifs

### L1 — Formal GPU model (this experiment, Phase 1)

Specify, do not yet accelerate:

| Spec | Language | Intent |
|------|----------|--------|
| Device memory as finite map of voxels | Lean | Type-safe VRAM model |
| Trinary packing (2 bits / state, warp-aligned) | Lean + Coq | Density + invariants |
| Tile / warp size constraints | Isabelle | Scheduling predicates |
| Scalar kernel boot value | F\* | Matches Rust/Python oracle |
| No out-of-bounds, no race on exclusive sectors | all | Safety contracts |

### L2 — Runtime bridge (Phase 0–2)

| Component | Role |
|-----------|------|
| `probe_gpu.py` | Discover device, SM, bandwidth basline |
| `fsot_scalar_gpu.py` | Φ on CPU vs GPU float; match formal boot scalar |
| CUDA kernels | Phase 2: trinary pack/unpack, Φ tile, resonance update |
| Zig VRAM allocator (archive) | Prior art: payload JSON → simulated crystal |

### L3 — Learning (Phase 3)

Candidate FSOT-native primitives (research, not claimed shipped):

1. **Resonance attention** — attention weights modulated by FSOT phase/resonance fields rather than pure QKᵀ/√d.
2. **Trinary residual channels** — discrete state lanes for “spin” of tokens/features.
3. **LTM wave substrate** — continuous flow memory (SR-ITE style) co-resident in VRAM with verified layout.
4. **Obligation-checked training step** — after N steps, numeric panel must stay within FSOT error gates.

---

## 3. Data flow (Phase 0–2)

```
                    ┌─────────────────────────────┐
  Archive seeds ───►│ config/fsot_seeds.json      │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   Lean/F* specs            Python CPU Φ              CUDA/PyTorch Φ
   (proof)                  (mpmath/f64)              (float32/64)
         │                         │                         │
         └────────────┬────────────┴────────────┬────────────┘
                      ▼                         ▼
              cross_check report          results/phase0/*.json
```

---

## 4. Mapping formal languages to GPU work

| Language | Strength for this project | GPU use |
|----------|---------------------------|---------|
| **Lean 4** | Primary math authority, mathlib, executable specs | Spec of tensors, proofs of packing injectivity |
| **Coq/Rocq** | Extraction to OCaml; mature floating-point libraries (Flocq) | Certified float semantics of Φ reductions |
| **Isabelle/HOL** | Heavy automation (sledgehammer) for scheduling lemmas | Warp/tile predicates, refinement proofs |
| **F\*** | Effectful specs, C extraction via KaRaMeL | Kernel interface contracts → C/CUDA host glue |

None of Lean/Coq/Isabelle/F\* *run on the GPU* as the primary runtime. They **constrain** what the GPU is allowed to do; CUDA/Zig/Rust **execute** the constrained model.

---

## 5. Hardware contract (this machine)

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM | ~12227 MiB |
| CUDA | 13.2 |
| Driver | 595.95+ |
| Compute capability | 12.0 (Blackwell) |
| Target crystal boundary | 12800 MB (from existing Zig allocator) |

VRAM layout sketch (Phase 1 formalize → Phase 2 implement):

```
[0 ........ header ........]
[ FSOT boot constants (page-aligned) ]
[ Trinary packed state banks ]
[ Continuous Φ / float workspace ]
[ LTM / stream windows ]
[ Scratch / PyTorch interop buffers ]
```

---

## 6. Relationship to SR-ITE soul stack

The continuous cognition substrate already has:

- Zig wave + LTM on virtual substrate  
- Python decoder / mulling  
- Coq + Lean soul formulas  
- VRAM payload tensors  

This experiment **does not replace** that stack. It **grounds** its GPU path in the same formal spine so that future “LLM through FSOT languages” work shares one verified numeric and memory story.
