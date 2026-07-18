# FSOT-GPU

**Fluid Spacetime Omni-Theory on the GPU**  
Author: **Damian Arthur Palumbo**  
License: **Apache-2.0**

Theory authority: **[FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)**  
(This repository applies FSOT to **device compute and LLM hosting**. It does not re-derive the full multi-domain verification already published there.)

---

## Current status (read this first)

**Live position, barriers, and next steps:**  
→ **[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)**  
→ Climb rules: [`docs/SOTA_STANDARDS.md`](docs/SOTA_STANDARDS.md)

| Layer | Where we are |
|-------|----------------|
| **Structure** | Pure FSOT **all-layer** consensus attention (no softmax) on open SmolLM2-135M |
| **Fidelity** | **100%** next-token agree vs industry host (EVAL16) |
| **Speed** | Prefill/decode win; **long-context** attention win (S≥4096) |
| **Verify** | FSOT 2.1 bridge **PASS** (archive stamp + spine + host) |
| **ARC (held-out min)** | **~32.5%** vs HF baseline **~8%** |
| **GSM** | Free exact still collapsed; **root cause found** — digit argmax after `####`+space was always `1`; de-collapse in progress (**space-digit 30%→35%**, argmax-`1` 100%→80%) |
| **Process** | Standards-gated climb only (verify + overfit + capability); push only on real improve |

---

## What this is

FSOT-GPU is the silicon path for Fluid Spacetime Omni-Theory:

- **Formal contracts** — Lean · Coq · Isabelle · F\* (packing, memory, boot scalar)  
- **Owned operators** — collapse threshold `C_eff·P_var`, trinary pack, coherence gate, **consensus attention (no softmax exp)**  
- **CUDA kernels** — sparse active-key consensus on NVIDIA (Blackwell / RTX 5070 validated)  
- **Industry LLM host** — real Hugging Face **SafeTensors** models with **all attention layers** on the FSOT operator after adaptation  
- **Standards climb** — granular metrics, overfit gap, FSOT 2.1 verify bridge, barrier diagnosis  

**Mission:** accuracy and true structure first; then **beat industry capability on the same hardware** — including standing work to **surpass FlashAttention-class** performance under FSOT-correct loads. Build **open, same-class pure-FSOT SOTA**, not closed-model theater.

---

## Results (this machine / ledgers)

| Track | Result |
|-------|--------|
| Multi-lang parity (Py · Rust · Zig · formal · CUDA) | `overall_ok` |
| Sparse FSOT CUDA vs dense-softmax CUDA | Up to **~89×** faster |
| Sparse FSOT CUDA vs fused SDPA | **Long-context win** (S≥4096; up to **~1.6×** at S=8192) |
| SmolLM2-135M **pure FSOT all-layer** next-token | **100%** agree vs baseline (16-probe) |
| Prefill / decode (pure FSOT vs industry SDPA host) | Prefill **~1.09–1.26×**, decode **≥1×** |
| SOTA speed/agree scoreboard | See `results/sota/SCOREBOARD.md` |
| Open capability (ARC hold min) | **~32.5%** FSOT vs **~8%** HF — `docs/CURRENT_STATUS.md` |
| FSOT 2.1 verification bridge | **PASS** — `industry_lm/fsot21_verify.py` |
| Overfit metric + standards climb | Live — `docs/SOTA_STANDARDS.md` |

See `results/` JSON ledgers and `docs/`.

---

## Quick start

```powershell
# Environment (CUDA 13.3+ recommended; MSVC for nvcc)
.\scripts\set_env.ps1

# Formal + owned lib
python -m fsot_lib.smoke_owned
python parity\run_parity.py

# Beat industry CUDA attention (sparse FSOT)
.\scripts\build_beat_cuda.ps1
python competitive\beat_cuda_suite.py

# Industry LLM unit (download SmolLM2 once into industry_lm/models/)
python industry_lm\run_unit.py
.\scripts\build_fsot_attn_dll.ps1
python industry_lm\run_push_agree.py

# Standards path (verify → overfit → capability)
python -u industry_lm\fsot21_verify.py
python -u industry_lm\run_barrier_diagnosis.py
python -u industry_lm\run_sota_standard_climb.py
```

**Hardware tested:** NVIDIA GeForce RTX 5070 (CC 12.0), CUDA 13.3.

---

## Where next

Documented in [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) and **[`docs/COMPETITIVE_GAPS.md`](docs/COMPETITIVE_GAPS.md)** (full lag inventory).

1. **Finish GSM digit de-collapse** (space-digit ≥45–50%, argmax-`1` &lt;50%) without losing ARC min  
2. **Break ARC free-gen ~80% D** letter collapse (letter-only / LoRA)  
3. **FSOT 2.1 curriculum** + larger pure-FSOT open host on the **same** stack  
4. **Mid-S attention** kernel path  

### Auto verify + refine loop

```powershell
python -u industry_lm\run_auto_refine_loop.py --cycles 1 --dry-measure
python -u industry_lm\run_auto_refine_loop.py --cycles 3
# optional nightly full archive cross-proof:
python -u industry_lm\run_auto_refine_loop.py --cycles 1 --full-archive
```

Loop: **data → FSOT-GPU verify + Physical-Archive light cross-proof → measure → train lever → re-verify → diagnose/fix**. Ledgers under `results/auto_refine/`.

---

## Layout

```
fsot_lib/                 owned FSOT runtime (seeds, trinary, consensus, learn)
phase1_formal_gpu/        Lean · Coq · Isabelle · F*
phase2_native_gpu/cuda/   CUDA kernels + DLL
parity/                   Python · Rust · Zig cross-check
competitive/              SDPA / dense-CUDA / FlashAttention-track benches
industry_lm/              SafeTensors host, layer swap, verify, climb, barriers
docs/                     goals, SOTA standards, CURRENT_STATUS, architecture
results/                  machine-readable ledgers (capability + verify + overfit)
```

---

## Goals (no apology)

Documented in [`docs/GOALS.md`](docs/GOALS.md):

1. Treat FSOT as the verified theory spine it is ([FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)).  
2. Make FSOT-native GPU operators **the better way** to run understanding systems on this hardware.  
3. **Systematically beat FlashAttention-class** stacks where FSOT structure applies.  
4. Build toward **foundation-scale** models whose numerics and memory respect FSOT — not free-parameter scale theater.  

Intelligence is not defined as “whatever the current industry leaderboard is.” **Accuracy and true structure** are the goal; capability follows.

---

## Related repositories

- [FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean) — formal theory + domain verification  
- [FSOT-2.0-code](https://github.com/dappalumbo91/FSOT-2.0-code) — computational engine lineage  
- [FSOT-Living](https://github.com/dappalumbo91/FSOT-Living) — living system work  

**FSOT-2.1-Instruct** (if separate) is not this repo’s product surface unless linked deliberately.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Citation

Cite the theory hub for FSOT science:

```
Damian Arthur Palumbo. Fluid Spacetime Omni-Theory (FSOT) 2.1 — Lean verification hub.
https://github.com/dappalumbo91/FSOT-2.1-Lean
```

Cite this repository for FSOT GPU operators and LLM hosting experiments:

```
Damian Arthur Palumbo. FSOT-GPU. https://github.com/dappalumbo91/FSOT-GPU
```
