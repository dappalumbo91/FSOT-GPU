# Claims split — residual law vs host weights

**Status:** locked (non-negotiable #2)  
**Authority:** FSOT-2.1-Lean residual spine · this repo applies it to GPU + LLM host  
**Date:** 2026-08-11

---

## One rule, two claims

| Layer | Free parameters? | What is claimed |
|-------|------------------|-----------------|
| **Theory residual law / substrate math** | **Zero** | Seed-derived scalar / collapse / domain residual law (`π, e, φ, γ, Catalan`; pin bind; archive green gate). No per-row least-squares knobs in the residual formula. |
| **Host weight tensors (LLM)** | **Allowed** | Industry open-weight transformers are **not** a three-parameter residual engine. Weights are ordinary free parameters. FSOT owns *operators, learning direction, routing, gates, and silicon substrate* — not “every matrix is seed-derived.” |

### Substrate vs model architecture (plain language)

- **Substrate (hardware + FSOT operators):** follows FSOT mathematics — seeds, collapse θ (`C_eff · P_var`), consensus attention (no softmax for pure claims), trinary packing, coherence, pin verify. This is the silicon / operator contract.
- **Large language model host:** a normal multi-million-parameter network (SmolLM2-class). It is **architecturally not** the residual “few-parameter” theory spine. Training updates weights under FSOT guidance; it does **not** pretend the LLM *is* the residual law.
- **Product claim for v1:** pure (or documented hybrid) FSOT *ops* on an open host — not “LLM has zero free parameters.”

**Do not fight yourself:**

- Saying “zero free parameters” about **FSOT residual / formula authority** is correct and already closed in FSOT-2.1-Lean.
- Saying “zero free parameters **inside the weight tensors** of an LLM host” is **false** and is **not** a v1 claim.
- Pure FSOT attention / collapse θ / trinary packing are **operators** (structure). Weights are **host capacity**.

---

## What FSOT owns in the LLM path

1. **Operators** — consensus attention (no softmax exp for pure claims), collapse θ (`C_eff · P_var`), coherence gates, trinary pack.  
2. **Learning direction** — curriculum priority, plasticity / loss modulation, accept–reject under overfit + residual-style gates.  
3. **Routing** — domain folds from the atlas (what trains next; hold vs train labels).  
4. **Gates** — G-VERIFY pin bind, G-OVERFIT, G-CAPABILITY, G-PUBLISH.

## What the host is allowed to own

1. **Weight tensors** — full FT, LoRA, or any trainable DoF on the chosen open family.  
2. **Industry tokenizer** (v1) — vocab unchanged; FSOT-influenced tokenization is later expansion.  
3. **Checkpoint storage** — large `.pt` / SafeTensors gitignored; **ledgers + metadata tracked**.

---

## Promotion language (public)

**Allowed:**

> Open host + pure FSOT operators (or documented hybrid) + pin verify green + curriculum/plasticity under FSOT + held-out capability above a frozen floor + overfit gate — without claiming zero free parameters inside the weight tensors.

**Forbidden on promote:**

- “Zero free parameters” referring to the full LLM stack.  
- Capability claims without G-VERIFY + G-OVERFIT + held-out improve.  
- Softmax/SDPA-blend results labeled “pure FSOT” without documenting hybrid mode.

---

## Cross-repo consistency

| Repo | Role in the split |
|------|-------------------|
| [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) | Residual law, pin, multiprover, domain atlas — **zero free** |
| [FSOT-GPU](https://github.com/dappalumbo91/FSOT-GPU) | Operators + pure-FSOT host path + climb gates — **weights free under FSOT guide** |
| [fsot-2.1-llm](https://github.com/dappalumbo91/fsot-2.1-llm) | Parallel product/research LLM stack; must obey the same claims split when quoting residual honesty |
| Genetics / quantum / bare-metal | Domain embodiments of residual law; feed **curriculum packs**, not a second residual engine |

See also: [`FSOT_LLM_CONSTRUCTION.md`](FSOT_LLM_CONSTRUCTION.md) · [`SOTA_STANDARDS.md`](SOTA_STANDARDS.md) · [`GOALS.md`](GOALS.md)
