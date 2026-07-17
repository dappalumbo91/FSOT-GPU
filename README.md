# FSOT-GPU

**Fluid Spacetime Omni-Theory on the GPU**  
Author: **Damian Arthur Palumbo**  
License: **Apache-2.0**

Theory authority: **[FSOT-2.1-Lean](https://github.com/dappalumbo91/FSOT-2.1-Lean)**  
(This repository applies FSOT to **device compute and LLM hosting**. It does not re-derive the full multi-domain verification already published there.)

---

## What this is

FSOT-GPU is the silicon path for Fluid Spacetime Omni-Theory:

- **Formal contracts** — Lean · Coq · Isabelle · F\* (packing, memory, boot scalar)  
- **Owned operators** — collapse threshold `C_eff·P_var`, trinary pack, coherence gate, **consensus attention (no softmax exp)**  
- **CUDA kernels** — sparse active-key consensus on NVIDIA (Blackwell / RTX 5070 validated)  
- **Industry LLM host** — real Hugging Face **SafeTensors** models with **all attention layers** on the FSOT operator after adaptation  

**Mission:** accuracy and true structure first; then **beat industry capability on the same hardware** — including standing work to **surpass FlashAttention-class** performance under FSOT-correct loads.

---

## Results (this machine / ledgers)

| Track | Result |
|-------|--------|
| Multi-lang parity (Py · Rust · Zig · formal · CUDA) | `overall_ok` |
| Sparse FSOT CUDA vs dense-softmax CUDA | Up to **~89×** faster |
| Sparse FSOT CUDA vs fused SDPA | **Long-context win** (S≥4096; up to **~1.6×** at S=8192) |
| SmolLM2-135M **pure FSOT all-layer** next-token | **~94%** agree (16-probe), **~88%** (8-probe) |
| Prefill / decode (pure FSOT vs industry SDPA host) | Prefill **~1.09×**, decode **~1.06×** |
| SOTA scoreboard (tiny model, same GPU) | **Across the board** — see `results/sota/SCOREBOARD.md` |
| Blend demo (FSOT+SDPA) | **~100%** next-token agree |

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
```

**Hardware tested:** NVIDIA GeForce RTX 5070 (CC 12.0), CUDA 13.3.

---

## Layout

```
fsot_lib/                 owned FSOT runtime (seeds, trinary, consensus, learn)
phase1_formal_gpu/        Lean · Coq · Isabelle · F*
phase2_native_gpu/cuda/   CUDA kernels + DLL
parity/                   Python · Rust · Zig cross-check
competitive/              SDPA / dense-CUDA / FlashAttention-track benches
industry_lm/              SafeTensors host, layer swap, distill
docs/                     goals, architecture, layman, publication
results/                  machine-readable ledgers
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
