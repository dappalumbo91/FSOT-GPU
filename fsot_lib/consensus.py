"""
Consensus attention — owns the coupling path (no softmax).

Twin: Desktop Fsot trinary kernel lattice.rs consensus_aggregate.
"""

from __future__ import annotations

from fsot_lib.coherence import position_coherence
from fsot_lib.trinary import trit_similarity


def apply_phase_rotation(h, positions=None):
    """
    pi-periodic phase: theta = 2 * position (kernel lattice.rs).
    h: torch [seq, dim] or ignored pure path.
    """
    import torch

    out = h.clone()
    seq, dim = h.shape
    if positions is None:
        positions = torch.arange(seq, device=h.device, dtype=h.dtype)
    else:
        positions = positions.to(h.dtype)
    theta = 2.0 * positions
    cs, sn = torch.cos(theta), torch.sin(theta)
    pairs = dim // 2
    for k in range(pairs):
        a, b = out[:, 2 * k], out[:, 2 * k + 1]
        out[:, 2 * k] = cs * a - sn * b
        out[:, 2 * k + 1] = sn * a + cs * b
    return out


def consensus_aggregate(q, k, v):
    """
    Collapse-gated attention. q,k,v: [seq, head_dim] torch tensors.
    No exp, no softmax denominator.
    """
    import torch

    seq = q.shape[0]
    sim = trit_similarity(q, k)  # [Sq, Sk]
    k_coh = position_coherence(k)
    idx = torch.arange(seq, device=q.device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
    gate = (k_coh > 0.5).unsqueeze(0) & causal
    w = torch.where(gate, sim, torch.zeros_like(sim))
    active = (w != 0).to(torch.float64).sum(dim=-1, keepdim=True).clamp_min(1.0)
    out = (w @ v.to(torch.float64)) / active
    return out.to(v.dtype)
