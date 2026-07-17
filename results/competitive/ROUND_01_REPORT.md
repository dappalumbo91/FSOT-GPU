# Competitive Round 01 — `fsot_consensus_vs_softmax_micro`

**Date:** 2026-07-17  
**Device:** NVIDIA GeForce RTX 5070  
**Seed:** 20260717  
**Machine ledger:** `fsot_consensus_vs_softmax_micro.json`

## Arms

| Arm | Implementation |
|-----|----------------|
| Baseline | `torch.nn.functional.scaled_dot_product_attention` (causal, fused) |
| Contender | FSOT vectorized multi-head consensus (`competitive/vectorized_consensus.py`) |
| Reference | Per-head loop (`fsot_lib.consensus`) — contract twin, slower |

## Measured wall time (ms / iter)

| seq | heads | d | Softmax SDPA | FSOT vectorized | Speedup (FSOT/soft) |
|-----|-------|---|--------------|-----------------|---------------------|
| 32 | 8 | 16 | ~0.24 | ~3.5 | ~0.07× |
| 64 | 8 | 32 | ~0.25 | ~5.6 | ~0.05× |
| 128 | 8 | 64 | ~0.25 | ~10.5 | ~0.02× |

Vectorized path is **~7–15× faster** than the naive per-head Python loop, but still behind fused SDPA.

## Kill criteria verdict

| Criterion | Winner | Notes |
|-----------|--------|-------|
| Stability (no exp, weights ∈ [−1,1]) | **FSOT** | All 3 configs; softmax uses exp |
| Density (trinary pack 4×) | **FSOT** | By construction (state banks) |
| Throughput (ms) | **Softmax SDPA** | Fused industry kernel |
| Finite outputs | Tie | Both finite on this seed |
| Full win (stability + ≤2× slower) | **None** | FSOT ~14–40× slower on host path |

**Round status:** `ok: true` for **stability-unique differentiation**.  
**Not** a throughput SOTA claim.

## Allowed public language (locked to this round)

> On the preregistered microbench `fsot_consensus_vs_softmax_micro` (RTX 5070), FSOT consensus provides **exp-free** attention weights **bounded in [−1, 1]** with collapse threshold **C_eff·P_var ≈ 0.917466**; the softmax baseline uses exp. Throughput remains behind fused SDPA on the current host/torch path — **stability and density are the present competitive edge**.

## Next engineering (close the gap)

1. Native CUDA consensus kernel (`sm_120`) — same contract as vectorized torch  
2. Fuse collapse + sim + gate in one launch  
3. Re-run this microbench; target full_win on seq≤64  
4. Only then expand task-quality competitive rounds  

## How to reproduce

```powershell
cd "C:\Users\damia\Desktop\gpu exparment for lean coq isabell andf star"
python competitive\fsot_consensus_vs_softmax_micro.py
```
