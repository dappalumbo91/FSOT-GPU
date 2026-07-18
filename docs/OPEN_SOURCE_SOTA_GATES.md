# Open-source SOTA gates — pure FSOT small host

**Intent:** Not “beat every closed 100B+ model.”  
**Intent:** On **open weights / open method / same-class small models**, pure FSOT should **dominate** on structure + measured capability on this hardware.

**Live narrative:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md) · Climb rules: [`SOTA_STANDARDS.md`](SOTA_STANDARDS.md)

## Gate board (this lab)

| # | Gate | Status | Notes |
|---|------|--------|--------|
| G1 | Pure FSOT all-layer attention (no SDPA blend) | **PASS** | Consensus CUDA / torch sparse, collapse θ |
| G2 | Next-token **equal** industry host (EVAL16) | **PASS** | **100%** |
| G3 | Prefill / decode ≥ baseline | **PASS** | Scoreboard wins |
| G4 | Long-context attention track | **PASS** | S≥4096 |
| G5 | Exceed knowledge (factual set) | **PASS (narrow)** | 50% > 43% base |
| G6 | Mid-S attention op | **OPEN** | Still fused-SDPA sweet spot |
| G7 | FSOT curriculum (2.1 math/arch + solidification) | **OPEN** | Parallel to capability climb |
| G8 | Public capability packs (ARC/GSM held-out) | **OPENING** | ARC min **~32.5%** ≫ HF **~8%**; GSM free exact still collapsed |
| G8b | Granular accuracy axes | **PASS (v1)** | TF / first-digit / constrained / hold / mode — `granular_metrics.py` |
| G8c | FSOT 2.1 verification bridge | **PASS** | V1–V7 — `fsot21_verify.py` |
| G8d | Overfit gap metric | **PASS (v1)** | `gen_score`, `accept_update` — `overfit_metrics.py` |
| G8e | Barrier diagnosis + digit de-collapse | **OPENING** | Root cause: digit argmax always `1` after space; **30%→35%** space-digit, **1@100%→~80%** |
| G9 | Repro package (ckpt recipe + ledgers + Apache-2.0) | **PARTIAL** | Repo public; large ckpts local/gitignored |

## Open-source dominance criteria (preregister)

On **same GPU**, **≤200M params**, pure FSOT host vs industry default attention host:

1. **Fidelity floor** — next-token agree ≥ 90% (we have **100%**)  
2. **Speed** — prefill or decode win (**PASS**)  
3. **Long context** — attention track win at S≥4096 (**PASS**)  
4. **FSOT literacy** — curriculum probes (**OPEN**)  
5. **Capability smoke** — ARC/GSM held-out not collapsed vs baseline (**ARC PASS**; **GSM OPEN**)  

**Current:** **4/5** structure/speed/ARC; GSM free-gen is the remaining hard open under honest #### scoring.

## Publish rule

Push capability claims only when **G-VERIFY + G-OVERFIT + G-CAPABILITY** improve (see SOTA standards). Barrier metrics (e.g. space-digit, argmax-`1` fraction) may promote when ARC floor holds.
