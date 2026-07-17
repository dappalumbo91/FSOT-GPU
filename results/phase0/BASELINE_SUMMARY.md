# Phase 0 Baseline Summary

**Date:** 2026-07-17  
**Experiment:** FSOT Formal-GPU (Lean · Coq · Isabelle · F\*)

## Hardware

| Item | Result |
|------|--------|
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM | 12226.56 MiB |
| Compute capability | 12.0 |
| Driver CUDA | 13.2 (`nvidia-smi`) |
| PyTorch | 2.11.0+cu128 (reports CUDA 12.8 runtime) |
| Matmul FP32 ~ | 18.7 / 23.2 / 22.5 TFLOPS (1k / 2k / 4k) |
| H2D copy | ~10.5 GiB/s |

## FSOT boot scalar

| Check | Result |
|-------|--------|
| CPU Φ_boot | 0.099288956268616835 |
| Canonical (F\*/archive) | 0.09928895626861721 |
| CPU vs canonical | rel_err ~ 3.8e-15 — **match** |
| GPU f64 vs CPU | max abs err **0** |
| GPU f32 vs CPU | max abs err ~ 4.7e-7 — **within 1e-5 rel** |

## Trinary packing (PyTorch reference)

| Check | Result |
|-------|--------|
| Roundtrip | **OK** (8192×32 states on CUDA) |
| Compression | 4× (uint8 → 2-bit pack in u64) |

## Formal toolchains

| Tool | Status |
|------|--------|
| Lake / Lean 4.32.0 | **OK** — `Trinary.lean` + `GpuMemory.lean` build |
| coqc (Rocq portable) | **OK** — `Trinary.v` |
| F\* portable | **OK** — `FSOTGpuBoot.fst` verified |
| Isabelle portable | **present** (smoke path) |

## Conclusion

Phase 0 **pass**. Environment is ready for Phase 1 formal depth and Phase 2 CUDA kernel wiring. No need to reinstall Python/PyTorch/CUDA — already operational on this machine.
