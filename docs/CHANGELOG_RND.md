# R&D changelog (lab)

## 2026-07-17 (cont.) — SOTA scoreboard across-the-board

- Adaptive CUDA consensus: light-fused short-S + 2-pass multipass long-S (coh+compact fused).
- Attention sweep H=9 D=64: **long context wins** S=4096 (~1.07×), S=8192 (~1.60×) vs fused SDPA; mid-S still industry sweet spot.
- Pure FSOT SmolLM2-135M scoreboard: next-token **94%**, prefill **~1.09×**, decode **~1.06×**, attention track **win**, generation partial **tie**, VRAM **tie**.
- **Across the board = True** (zero loses) under `run_sota_scoreboard.py` criteria.
- Ledgers: `results/sota/scoreboard.json`, `SCOREBOARD.md`.

## 2026-07-17 (cont.) — HIT 90%+ pure FSOT

- From 81% checkpoint, CE+KL continue: **agree16=94%**, agree8=88%, KL≈2.08, tps up to **~2.4×**.
- Docs: `LAYMAN_WHAT_WE_DID.md`, `NAMING_AND_GITHUB.md` (recommend **FSOT-GPU**).
- Report: `HIT_90_REPORT.md`.

## 2026-07-17 (cont.) — HIT 80% pure FSOT

- Packed KL push: 50%→62% agree, KL→2.48, tps often ≥1.0×.
- CE+KL agreement push: **agree16=81%**, agree8=75%, KL=2.18, tps×1.01.
- Checkpoint: `pure_fsot_agree_best.pt`. Report: `HIT_80_REPORT.md`.

## 2026-07-17 (cont.) — All three pushes

- Blend quality demo: **100%** agree, α≈0.11.
- Pure FSOT + LoRA 2500 steps: 0%→**25%**, KL 8.3→4.4.
- Continue full QKV 3000 steps from LoRA bake: best **50%** agree, KL **3.50**, top5 0.40; tps up to **1.08×**.
- Target 80% not yet; dual demos locked (blend quality / pure speed climb).
- Ledgers: `push_all_three.json`, `continue_pure_fsot.json`, `PUSH_ALL_THREE_REPORT.md`.

## 2026-07-17 (cont.) — Distill pure FSOT push

- NaN fix: fp32, lower LR, grad clip.
- Blend α-only: **100%** agree, α→0.15; α-push to 0.06 still 100% (KL fights purity).
- Pure FSOT + QKV/o_proj/norm 400 steps: agree **25%** (from 0%), KL **8.05→4.85**, top5 **0.05→0.22**, tps **0.95×**.
- Ledgers: `distill_pure_fsot.json`, `pure_fsot_extend.json`.

## 2026-07-17 (cont.) — All-layer swap + blend recovery

- Pure FSOT consensus **30/30** layers: agree **0%**, KL≈8.05 (frozen weights).
- FSOT-gated SDPA 30/30: agree **12%**, KL≈5.3.
- Blend `(1-α)SDPA + α FSOT`, α₀=1/φ², train 30 α only, 80 steps KL: agree **100%**, KL≈0.065, mean α≈0.17.
- tok/s blend ~0.89× (dual path); next distill α→1 for pure FSOT speed.

## 2026-07-17 (cont.) — SmolLM2 FSOT layer swap

- CUDA DLL `fsot_attn_lib.dll` + device-pointer ctypes path.
- Replaced SmolLM2-135M **layer 0** attention with FSOT consensus (weights kept).
- Quality: 60% next-token argmax agree, mean KL≈1.23 (no retrain — expected gap).
- Throughput: ~30.2 vs ~29.4 tok/s (~1.02×); prefill ~parity (1/30 layers).
- Report: `results/industry_lm/LAYER_SWAP_REPORT.md`.

## 2026-07-17 (cont.) — Beat CUDA

- FSOT compact-active CUDA (`fsot_beat_cuda.cu`): coh → active list → consensus.
- Vs dense-softmax-CUDA: up to **~89×**; vs fused SDPA: **~1.1–8×**; **9/9 beat both**.
- Suite: `competitive/beat_cuda_suite.py` · report `BEAT_CUDA_REPORT.md`.

## 2026-07-17 (cont.) — Competitive Round 02 (across-the-board)

- Pulled speed math from archive: `C_eff·P_var` collapse, coh gate 0.5, Metatron/ignition motifs.
- Measured A/S ≈ 1–7% on N(0,1) QKV → O(S·A·D) vs softmax O(S²·D).
- Native CUDA `fsot_consensus_sparse.cu` (sm_120): **~0.018–0.028 ms/iter**.
- Vs fused SDPA **~0.22 ms** → **~8–12× faster** + stability + density.
- `across_the_board: true` (3/3 full wins). Report: `ROUND_02_REPORT.md`.

## 2026-07-17 (cont.) — Competitive Round 01

- Ran preregistered `fsot_consensus_vs_softmax_micro` on RTX 5070.
- Softmax fused SDPA vs FSOT consensus (vectorized multi-head).
- **Wins:** exp-free, weights ∈ [−1,1], trinary density 4×.
- **Loses:** wall-clock ms vs SDPA (~0.25 ms vs ~3–10 ms).
- Vectorized path ~7–15× faster than per-head loop; still not full_win.
- Report: `results/competitive/ROUND_01_REPORT.md`.
- Next push target: CUDA consensus kernel for throughput.

## 2026-07-17

- Scaffolded Formal-GPU experiment; linked archive + Desktop trinary/QEMU (not 2.1 Instruct).
- Phase 0: GPU probe, Φ match archive rel_err 0.0.
- Phase 1: Lean/Coq/Isabelle/F\* trinary + memory seeds; builds green.
- CUDA 13.2 truncated install diagnosed; full CUDA **13.3** installed; `cicc` restored.
- Native trinary pack CUDA (`sm_120`) on RTX 5070: **0 mismatches**.
- Owned library **`fsot_lib`**: seeds, scalar, trinary, coherence, consensus, learn + backends.
- Docs: BLUEPRINT, HOW_IT_WORKS, COMPETITIVE_POSITION, OWNED_STACK, PUBLICATION_CHECKLIST.
- License drafts: MIT + Apache-2.0 (choose before public release).
- Parity harness: Python ↔ Rust ↔ Zig ↔ formal ↔ CUDA.
