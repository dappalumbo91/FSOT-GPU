"""
Faster FSOT multi-head consensus for competitive throughput (still exp-free).

Batched trit similarity + coherence gates — same contract as consensus_aggregate.
"""
from __future__ import annotations

import torch

from fsot_lib.seeds import COLLAPSE_THRESHOLD


def collapse_codes(x: torch.Tensor) -> torch.Tensor:
    """x [...] → int8 codes {0,1,2}."""
    up = x > COLLAPSE_THRESHOLD
    down = x < -COLLAPSE_THRESHOLD
    codes = torch.ones(x.shape, device=x.device, dtype=torch.int8)
    codes = torch.where(up, torch.tensor(2, device=x.device, dtype=torch.int8), codes)
    codes = torch.where(down, torch.tensor(0, device=x.device, dtype=torch.int8), codes)
    return codes


def apply_phase_rotation_batch(h: torch.Tensor) -> torch.Tensor:
    """h: [heads, seq, dim]"""
    heads, seq, dim = h.shape
    out = h.clone()
    pos = torch.arange(seq, device=h.device, dtype=h.dtype)
    theta = 2.0 * pos  # [seq]
    cs = torch.cos(theta).view(1, seq, 1)
    sn = torch.sin(theta).view(1, seq, 1)
    pairs = dim // 2
    for p in range(pairs):
        a = out[:, :, 2 * p]
        b = out[:, :, 2 * p + 1]
        out[:, :, 2 * p] = cs.squeeze(-1) * a - sn.squeeze(-1) * b
        out[:, :, 2 * p + 1] = sn.squeeze(-1) * a + cs.squeeze(-1) * b
    return out


def consensus_multihead_fast(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q,k,v: [H, S, D] float
    Returns [H, S, D] — no exp, weights in [-1,1] after trit consensus.
    """
    dtype = v.dtype
    q = apply_phase_rotation_batch(q.to(torch.float64))
    k = apply_phase_rotation_batch(k.to(torch.float64))
    v64 = v.to(torch.float64)
    H, S, D = q.shape

    tq = collapse_codes(q)  # [H,S,D]
    tk = collapse_codes(k)
    # trit sim: [H, Sq, Sk]
    tq_e = tq.unsqueeze(2)  # H,Sq,1,D
    tk_e = tk.unsqueeze(1)  # H,1,Sk,D
    super_mask = (tq_e == 1) | (tk_e == 1)
    same = (tq_e == tk_e) & ~super_mask
    opp = (tq_e != tk_e) & ~super_mask
    sim = (same.to(torch.float64) - opp.to(torch.float64)).mean(dim=-1)  # H,Sq,Sk

    # coherence of K rows
    k_coh = (k.abs() > COLLAPSE_THRESHOLD).to(torch.float64).mean(dim=-1)  # H,S
    idx = torch.arange(S, device=q.device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # Sq,Sk
    gate = (k_coh > 0.5).unsqueeze(1) & causal.unsqueeze(0)  # H,Sq,Sk
    w = torch.where(gate, sim, torch.zeros_like(sim))
    active = (w != 0).to(torch.float64).sum(dim=-1, keepdim=True).clamp_min(1.0)
    out = torch.matmul(w, v64) / active
    return out.to(dtype)
