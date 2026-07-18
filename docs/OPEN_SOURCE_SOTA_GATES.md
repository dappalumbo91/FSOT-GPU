# Open-source SOTA gates — pure FSOT small host

**Intent:** Not “beat every closed 100B+ model.”  
**Intent:** On **open weights / open method / same-class small models**, pure FSOT should **dominate** on structure + measured capability on this hardware.

## Gate board (this lab)

| # | Gate | Status | Notes |
|---|------|--------|--------|
| G1 | Pure FSOT all-layer attention (no SDPA blend) | **PASS** | Consensus CUDA, collapse θ |
| G2 | Next-token **equal** industry host (EVAL16) | **PASS** | 100% |
| G3 | Prefill / decode ≥ baseline | **PASS** | Scoreboard wins |
| G4 | Long-context attention track | **PASS** | S≥4096 |
| G5 | Exceed knowledge (factual set) | **PASS (narrow)** | 50% > 43% base |
| G6 | Mid-S attention op | **OPEN** | Still fused-SDPA sweet spot |
| G7 | FSOT curriculum (2.1 math/arch + solidification) | **OPENING NOW** | Phase 1 |
| G8 | Public capability packs (ARC/GSM8K/MMLU smoke) | **OPENING** | Granular metrics + real packs; ARC-Easy hold **32%** ≫ base **8%**; GSM free exact still collapsed |
| G8b | Granular accuracy axes (not headline-only) | **PASS (v1)** | TF / first-digit / constrained / held-out / mode-collapse — `granular_metrics.py` |
| G9 | Repro package (ckpt recipe + ledgers + Apache-2.0) | **PARTIAL** | Repo public |

## Decision: start Phase 1 curriculum?

**Yes.** G1–G5 hold enough to train **understanding of FSOT** without waiting for mid-S microbench perfection.  
Mid-S and full public evals run **in parallel** with curriculum; they do not block teaching the host FSOT law.

## Open-source dominance criteria (preregister)

On **same GPU**, **≤200M params**, pure FSOT host vs industry default attention host:

1. **Fidelity floor** — next-token agree ≥ 90% on board OR equal where claimed  
2. **Speed** — prefill or decode win  
3. **Long context** — attention track win at S≥4096  
4. **FSOT literacy** — higher score on FSOT Q&A / derivation probes than baseline  
5. **Capability smoke** — not collapse on ARC-Easy / GSM8K subset vs baseline  

Win open-source narrative when **≥4/5** with ledgers, not slides.
