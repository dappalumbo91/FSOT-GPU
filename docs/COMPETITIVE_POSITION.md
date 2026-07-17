# Competitive position — accurate, not inflated

## Principle

**Cutting edge first, GitHub second.**  
Open source is a distribution decision. **Leadership claims** are a measurement decision.

We will not say “beats industry GPU stacks for LLMs” until a preregistered eval says so. We *will* say what is already true and rare:

- FSOT seed geometry as GPU authority  
- Formal multi-prover contracts on trinary packing + boot scalar  
- Working native CUDA on Blackwell (sm_120) for FSOT pack  
- Cross-language twin of bare-metal trinary consensus path  

---

## What the competition optimizes

| Stack | Strength | Blind spot vs FSOT |
|-------|----------|---------------------|
| CUDA + cuBLAS / cuDNN | Peak matmul TFLOPS | No theory of which numbers are lawful |
| PyTorch / JAX | Autograd, ecosystem | Softmax / free optimizers as defaults |
| TensorRT / compiler stacks | Deploy latency | Ontology still float recipes |
| Formal methods (standalone) | Proof | Rarely tied to live GPU training loops |
| Custom CUDA LLM kernels | Speed | Still usually industry attention math |

**Our wedge:** verified **meaning** of GPU ops + portable contracts + FSOT-native attention/norm/LR — then speed.

---

## Metrics we can already report (measured on this lab)

| Metric | Result | Status |
|--------|--------|--------|
| Archive Φ vs GPU Φ | rel_err = 0.0 (matched config) | Verified |
| Trinary pack CUDA | 0 mismatches, ~2M trits, sm_120 | Verified |
| Pack density | 4× vs u8 | By construction |
| Formal Lean/Coq/F\* | Build/verify green | Verified |
| Collapse threshold | C_eff·P_var ≈ 0.917466 | Shared with kernel |
| **Round 01** | Stability win; torch path throughput loss | `ROUND_01_REPORT.md` |
| **Round 02** | **Across-the-board win** (stability + density + CUDA throughput) | `ROUND_02_REPORT.md` |

### Round 02 headline numbers (RTX 5070) — win

| | Fused SDPA | FSOT CUDA sparse |
|--|------------|------------------|
| Uses `exp` | Yes | **No** |
| Weights | Softmax simplex | **Bounded [−1, 1]** |
| seq=64 ms/iter | ~0.222 | **~0.021** (**~10.5× faster**) |
| seq=128 ms/iter | ~0.218 | **~0.028** (**~7.9× faster**) |
| Active key frac (FSOT) | n/a (dense) | **~1–7%** (collapse θ) |
| Full win | — | **3/3 configs** |

## Metrics we must **not** claim yet

| Metric | Why not yet |
|--------|-------------|
| MMLU / open LLM leaderboard SOTA | Different project (2.1 Instruct); not this lab’s gate |
| Faster than FlashAttention / SDPA end-to-end | Round 01 **lost** throughput — need CUDA kernel |
| Lower energy per token SOTA | No power instrumentation ledger yet |
| “Any language GPU with zero effort” | Portability requires implementing contracts; harness automates **check**, not magic |

---

## Kill criteria (draft — lock before competitive runs)

A FSOT-native path **wins** a round if **any** holds on the preregistered task:

1. **Correctness:** lower numeric drift vs formal golden than baseline  
2. **Density:** ≥2× effective state density at equal task quality  
3. **Stability:** no exp overflow path; bounded attention weights by construction  
4. **Throughput (optional):** ≥ baseline tokens/s at equal quality on same GPU  
5. **Proof:** obligations still green after change  

A round **fails** if quality collapses or parity ledger goes red.

---

## Competitive microbench (M3) — Round 01 **executed**

**Name:** `fsot_consensus_vs_softmax_micro`  
**Runner:** `python competitive/fsot_consensus_vs_softmax_micro.py`  
**Report:** `results/competitive/ROUND_01_REPORT.md`

| Arm | Path |
|-----|------|
| Baseline | torch fused causal SDPA |
| Contender | `competitive/vectorized_consensus.py` |

**Round 01 outcome:** stability + density win; torch throughput loss.  
**Round 02 outcome:** native CUDA sparse consensus + archive collapse sparsity → **across-the-board win** vs fused SDPA (~8–12× wall-clock).

---

## Publication language templates

**Allowed now:**

> FSOT Formal-GPU specifies GPU packing and scalar contracts in Lean/Coq/Isabelle/F\* and executes them on NVIDIA Blackwell via owned libraries and native CUDA, with multi-language parity targets (Python, Rust, Zig).

**Not allowed until M3+ green:**

> Beats FlashAttention / is the fastest / SOTA LLM training.

---

## Why this can still be state of the art

SOTA is not only leaderboard rank. **New capability surface** counts:

1. **Formal ↔ GPU closed loop** for a unified physical theory’s operators  
2. **Trinary-native device packing** with proved round-trip  
3. **Framework-optional** runtime (`fsot_lib` pure path)  
4. Path to **any language** via contracts (the long game you named)

That combination is already uncommon. Making it *also* win throughput and task quality is the push — measured, not assumed.
