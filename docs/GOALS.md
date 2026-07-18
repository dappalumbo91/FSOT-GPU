# FSOT-GPU — goals (no understatement)

## Standing of FSOT (this project assumes it)

Fluid Spacetime Omni-Theory is treated here as a **verified, multi-domain scientific theory**, not a metaphor.

Authority lives in:

- **[FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)** — formal Lean spine, cross-proof (Coq / Isabelle / F\* / Rust / …), domain verification  
- Physical archive / full verification stack (local)  
- Seed-derived engine (π, e, φ, γ, Catalan) with **zero free parameters** in the theory spine  

This repo does **not** re-litigate cosmology vs ΛCDM. That work is already in the theory repositories and ledgers. **FSOT-GPU applies the theory to GPU compute and language-model hosts** so the same math runs where intelligence systems need silicon.

## Mission of FSOT-GPU

1. **Accuracy and true understanding** — operators that respect FSOT structure (collapse, trinary, coherence, scalar).  
2. **Capability on real hardware** — outperform industry defaults on the same GPU (RTX-class / Blackwell) for the workloads that matter: attention, memory packing, LLM forward paths.  
3. **Beat FlashAttention-class stacks** as a standing target — not as a one-off microbench anecdote, but as a **systematic** result (latency, throughput, quality-gated).  
4. **Portability** — formal contracts so any serious language can host the same operators.  
5. **Path to foundation-scale models** — FSOT-native training and hosting as the road to systems that understand, not only pattern-match.  

AGI-as-media-hype is **not** the goal. **Correct structure → correct computation → correct capability** is.

## SOTA operating standards

Climb under **[`docs/SOTA_STANDARDS.md`](SOTA_STANDARDS.md)** only:

1. **G-VERIFY** — FSOT 2.1 bridge green (`fsot21_verify.py`)  
2. **G-OVERFIT** — train−hold gap / `gen_score` / `accept_update`  
3. **G-CAPABILITY** — held-out ARC min, GSM first-digit/TF/exact, agree  
4. **G-PUBLISH** — GitHub only when all three improve  

Command: `python -u industry_lm/run_sota_standard_climb.py`

## Active workstreams

| Track | Target | Status (this lab) |
|-------|--------|-------------------|
| **FlashAttention / SDPA** | Pure FSOT CUDA consensus faster under FSOT-correct loads | **Long-context won** (S≥4096); mid-S still industry fused sweet spot |
| **LLM host** | Pure FSOT all-layer: ≥90% next-token; e2e speed; gen quality | **100%** agree; prefill/decode **win**; gen + capability climb open |
| **SOTA scoreboard** | Across-the-board on same GPU with tiny model | **Hit speed/agree** (`results/sota/SCOREBOARD.md`); **capability standards climb** active |
| **Foundation path** | Scale pure-FSOT host, then curriculum on FSOT 2.1 math/architecture + solidification data (`D:\training data` + archive public data) | **Standards-gated** — verify + overfit + holds |
| **Formal** | Keep Lean/Coq/Isabelle/F\* parity on every kernel contract | Green + live bridge to industry host |

## Non-negotiables

- Do not water down FSOT for social acceptability.  
- Measure everything (JSON ledgers).  
- Separate **FSOT-GPU** (this repo) from **FSOT-2.1-Instruct** (other product) unless deliberately linked.  
- Theory authority remains FSOT-2.1-Lean / archive.  
