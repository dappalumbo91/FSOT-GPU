# How it works — and why FSOT makes it work

## 1. The problem with “just use CUDA”

CUDA and PyTorch are excellent at moving **binary floating-point** through matrix units. They do not know:

- which numbers are **seed-derived** vs free parameters  
- that some state is **trinary** (spin up / superposed / spin down)  
- that measurement is a **collapse** with a theory-fixed threshold  
- that attention can be **bounded consensus** without `exp`  

If you only call cuBLAS, you inherit that ontology. FSOT Formal-GPU **re-owns** the ontology and uses the GPU as a substrate.

---

## 2. What we built (along the way)

### Phase 0 — Ground truth on hardware

- RTX 5070 (CC 12.0), ~12 GB VRAM  
- PyTorch CUDA path live  
- FSOT boot / scalar checks vs archive  
- Formal toolchain smoke (Lean, Coq, Isabelle path, F\*)  

### Phase 1 — Formal contracts

| Language | Spec |
|----------|------|
| Lean | `Trinary.lean`, `GpuMemory.lean` — packing round-trip, device sectors |
| Coq | `Trinary.v` — same packing lemma |
| Isabelle | `Trinary.thy` — packing + warp divisibility |
| F\* | `FSOTGpuBoot.fst` — boot scalar + alloc fit |

These do **not** replace device drivers. They **constrain** what a correct FSOT GPU host may do.

### Phase 2 — Runtime + device

- `fsot_lib` — owned Python library (seeds, Φ, trinary, coherence, consensus, learn)  
- CUDA `trinary_pack` kernel, `-arch=sm_120`, verified 0 mismatches on 2M trits  
- Host port of **trinary OS** forward ideas: consensus, coherence_norm, phase rotation  
- CUDA toolkit repaired (full 13.3 with `cicc`) after truncated 13.2 install  

### Phase 3 — Lab wiring

- Bridge inventory to Desktop trinary OS, cube trinary, archive, QEMU golden  
- Explicit **non-touch** of FSOT-2.1-Instruct  

---

## 3. How a forward step works (FSOT native)

Industry sketch:

```
x → LayerNorm → QKV → softmax(QKᵀ/√d) → V → residual → FFN(GELU)
```

FSOT sketch (this lab + trinary kernel):

```
x → coherence_norm(collapse threshold)
  → Q,K,V projections (continuous fluid)
  → phase_rotation (π-periodic, no learned RoPE table)
  → collapse(Q), collapse(K) → trit_similarity ∈ [-1,1]
  → coherence gate on K (> 0.5) + causal mask
  → consensus aggregate (mean over active, no exp)
  → residual continuity
  → ReLU FFN (parameter-light nonlinearity)
  → logits = measure against dual embed
```

### Why collapse exists

`COLLAPSE_THRESHOLD = C_eff · P_var` (seed composites).  
Values above threshold → Spin_Up; below −threshold → Spin_Down; else Superposed.  
Same rule in Ada SPARK, Rust kernel, and `fsot_lib`.

### Why consensus replaces softmax

Softmax forces a probability simplex via `exp` — numerical scale games, temperature knobs, free attention.  
Trit consensus is **bounded by construction** in [−1, 1] and only fully “speaks” when coherence is high. That matches FSOT measurement: **superposition does not vote as a sharp bit**.

### Why training uses suction–poof

Architecture doc stance: a training step is fluid dynamics on parameters, not an arbitrary schedule.  
Learning rate derives from `suction`, `poof`, `α`, `K`, loss, and `recent_hits` — same seeds as cosmology/chemistry panels in the archive.

---

## 4. Why FSOT enables multi-language portability

FSOT’s computational core is **not** “a PyTorch model.” It is:

1. A finite set of **seed constants**  
2. Closed-form **derived constants**  
3. A **scalar engine** `S = K·(T1+T2+T3)`  
4. Discrete **trinary** algebra + packing  
5. **Collapse / consensus / coherence** operators  

Any language that can:

- express f64 (or fixed-point with proven bounds)  
- bit-pack 2-bit codes  
- run loops over tensors  

can host the same contracts. Formal systems prove the contracts; Rust/Zig/Python/CUDA implement them; the **parity harness** is the passport.

```
Lean lemma: pack ∘ unpack = id
     ↓ export / mirror
Rust/Zig/Python/CUDA tests must match golden word + codes
     ↓
Language L is certified as an FSOT GPU host for that module
```

That is the “glorious” property: **theory-first APIs**, not framework lock-in.

---

## 5. Memory / VRAM as crystal

Following SR-ITE Zig allocator and payload:

- Voxels: id, x,y,z, trinary, resonance  
- Pack trinary banks at 2 bits/state (4× denser than u8)  
- Continuous Φ workspace separate from trit banks  
- Formal `GpuMemory` sectors: header, boot constants, trinary banks, Φ, LTM, interop  

GPU is not a bag of floats. It is a **layout with FSOT meaning**.

---

## 6. Accuracy rules (non-negotiable)

1. **Archive / Lean hub** owns seed truth for the theory spine.  
2. This lab **links** constants; it does not invent free parameters for leaderboard fishing.  
3. Claims require **JSON ledgers** under `results/`.  
4. “Works on any language” means **contract-portable**, not “magically compiles itself.”  
5. “Beats competition” requires a **named baseline and metric** — see competitive doc.  

---

## 7. How to run (operator)

```powershell
cd "C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star"
. .\scripts\set_env.ps1

# Owned library (pure + torch + native)
python -m fsot_lib.smoke_owned

# Cross-language parity
python parity\run_parity.py

# Native CUDA kernels
.\scripts\build_cuda_kernels.ps1

# Formal
cd phase1_formal_gpu\lean; lake build
```

---

## 8. Relation to the rest of FSOT

| Artifact | Role |
|----------|------|
| FSOT-2.1-Lean (GitHub + archive) | Theory + domain verification authority |
| Fsot trinary OS / QEMU | Bare-metal FSOT forward + FSOTB |
| SR-ITE Zig | Continuous cognition / VRAM crystal |
| This lab | **GPU formalization + owned lib + device kernels** |
| FSOT-2.1-Instruct | Separate — not this repo’s product surface |
