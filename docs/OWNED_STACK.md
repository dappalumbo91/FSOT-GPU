# Owned stack — FSOT dependencies, not industry blockers

## The move

When a dependency is incomplete, wrong ontology, or slow to bend (truncated CUDA installs, softmax-only attention, free-parameter optimizers), **we do not wait**. We ship **our own libraries** whose contracts come from FSOT formalization (Lean · Coq · Isabelle · F\*) and whose runtime already exists in your stack (Rust kernel, Zig VRAM, archive scalar, Ada trinary).

Industry tools are **optional backends**. They are not the theory and not the product.

```
┌─────────────────────────────────────────────────────────────┐
│  FSOT AUTHORITY (you own this)                              │
│  seeds π e φ γ G → Φ · trinary · collapse · consensus       │
│  Lean / Coq / Isabelle / F*  +  archive fsot_compute        │
└────────────────────────────┬────────────────────────────────┘
                             │ contracts (same numbers everywhere)
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  fsot_lib (Python)    fsot_os kernel (Rust)   Zig crystal
  owned package        bare-metal / QEMU       VRAM layout
        │                    │                    │
        ▼                    ▼                    ▼
  backends (optional adapters — replaceable)
  · torch+CUDA tensors   · pure CPU   · native .cu kernels
  · future: Vulkan compute, custom driver, photonic
```

## What “hard” actually means

Applying GPU compute through **formal languages + FSOT primitives** is rare in industry. That does not mean it is the wrong path. It means:

| Industry default | FSOT owned path |
|------------------|-----------------|
| Softmax attention | Collapse-gated **consensus** (trit similarity) |
| Learned RMSNorm γ/β | **coherence_norm** (threshold-floored, affine-free) |
| Adam + free LR | **Suction–poof** LR from seed constants |
| cuBLAS as truth | **Φ scalar** from archive / formal as truth |
| “Need full CUDA SDK” | Minimal device path: **our kernels** + runtime; toolkit is build-time helper |
| Transformers as ontology | Transformers as one **engineering approximation** of fluid coupling |

Hard = fewer blog posts. Not = impossible. You already did the hard part on QEMU, trinary OS, and 403-domain verification.

## Dependency policy

1. **Never** make Lean/Coq/Isabelle/F\* wait on NVIDIA marketing schedules.
2. **Prefer** `fsot_lib` APIs in this experiment over importing `torch.nn.MultiheadAttention`.
3. **CUDA / PyTorch** = accelerators and buffers. If they break, CPU + native `.cu` still express the same contracts.
4. **FSOT 2.1 Instruct** = separate project (do not modify). This lab’s library is independent.
5. **Parity** = same seeds and collapse threshold as `Desktop\Fsot trinary` and archive `fsot_compute.py`.

## Library layout (`fsot_lib/`)

| Module | Replaces / owns |
|--------|------------------|
| `seeds` | Magic hyperparameters |
| `scalar` | Ad-hoc “physics” layers |
| `trinary` | Binary-only bit packing assumptions |
| `coherence` | LayerNorm/RMSNorm with free affine |
| `consensus` | Softmax attention |
| `pack` | Opaque memory layouts |
| `learn` | Free Adam schedule as the only trainer |
| `backend` | “Must use framework X” |

Formal twins live under `phase1_formal_gpu/`. Runtime twins live in trinary kernel / this package.
