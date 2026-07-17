# Naming & GitHub recommendation

## Is this notable enough for GitHub?

**Yes — as a research release**, with careful claims.

| Claim type | OK to publish? |
|------------|----------------|
| FSOT-native GPU attention kernel + formal/parity stack | **Yes** |
| Microbench: beats dense softmax CUDA / often beats fused SDPA on sparsity regime | **Yes** (with numbers + hardware) |
| Real HF SafeTensors model runs with all-layer FSOT attention after adaptation | **Yes** |
| 80%+ next-token agreement on a fixed probe set | **Yes** (state the probe set size) |
| “SOTA LLM” / “beats FlashAttention in production” / “AGI” | **No** |

**Why publish now (even before 90%):**

- Reproducible code + ledgers  
- Clear theory → kernel → model pipeline  
- Separates **FSOT Formal-GPU** from the separate **FSOT-2.1-Instruct** product line  

**What to wait for before loud marketing:** 90% pure-FSOT probe agree, a few generation examples, one clean README figure.

---

## Suggested names

Pick one **product name** + keep FSOT in the subtitle.

### Top recommendation

**`FSOT-GPU`**  
*Fluid Spacetime Omni-Theory GPU stack — formal specs, sparse trinary attention, portable LLM host*

Short, searchable, matches your brand.

### Strong alternatives

| Name | Vibe |
|------|------|
| **`fsot-formal-gpu`** | Emphasizes Lean/Coq/Isabelle/F\* + GPU |
| **`FSOT-Crystal`** | VRAM-as-crystal / SR-ITE metaphor |
| **`FSOT-Consensus`** | Highlights consensus attention (no softmax) |
| **`Trinary-GPU`** | Too generic; weaker FSOT link |
| **`FSOT-Blackwell`** | Too hardware-specific |

### Repo name suggestion

```
github.com/dappalumbo91/FSOT-GPU
```

or

```
github.com/dappalumbo91/fsot-formal-gpu
```

### Tagline options

1. *Verified-first GPU compute for Fluid Spacetime Omni-Theory*  
2. *Sparse trinary attention from FSOT — formalized, CUDA-native, LLM-portable*  
3. *Theory-shaped kernels for language models (Lean · Coq · CUDA · SafeTensors)*

---

## License (when you open-source)

- **Apache-2.0** recommended (patent grant, industry-friendly)  
- Draft already in lab: `LICENSE.Apache-2.0.txt`  
- MIT also fine for maximum simplicity  

---

## What goes in the public repo

**Include**

- `fsot_lib/`, formal seeds, CUDA kernels (source), parity harness  
- `industry_lm/` scripts (not necessarily full model weights)  
- docs + JSON ledgers (results)  
- README with layman section + measured tables  

**Exclude / don’t commit**

- Large model weights (link Hugging Face model id instead)  
- FSOT-2.1-Instruct private adapters unless intentional  
- Secrets, absolute machine paths in docs (use placeholders)  

**Point to**

- [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) as theory authority  

---

## Suggested README opening (public)

> **FSOT-GPU** implements Fluid Spacetime Omni-Theory operators on NVIDIA GPUs: collapse-threshold trinary packing, coherence-gated sparse consensus attention (no softmax), cross-checked with Lean/Coq/Isabelle/F\*/Rust/Zig.  
> We host a small industry SafeTensors model (SmolLM2-135M) with all attention layers replaced by FSOT consensus after lightweight adaptation, reaching **≥80% next-token agreement** with the baseline on a fixed probe set, while attention microbenchmarks show large speedups vs dense CUDA softmax.

---

## Decision checklist before `git push`

- [ ] Final name: **FSOT-GPU** (or your pick)  
- [ ] License: Apache-2.0 or MIT  
- [ ] 90% pure-FSOT optional but nice  
- [ ] Sanitize paths / secrets  
- [ ] Explicit “not FSOT-2.1-Instruct”  
- [ ] You say **go** to create/push the public repo  

This lab folder is already the draft repository content.
