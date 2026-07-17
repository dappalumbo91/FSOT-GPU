"""
Fully batched FSOT sparse consensus — no Python per-head loop.
Uses collapse θ and coh>0.5 gate from archive/kernel.
"""
from __future__ import annotations

import torch

from fsot_lib.seeds import COLLAPSE_THRESHOLD


def _phase_batch(x: torch.Tensor) -> torch.Tensor:
    """x [H,S,D] float32"""
    H, S, D = x.shape
    pairs = D // 2
    if pairs == 0:
        return x
    pos = torch.arange(S, device=x.device, dtype=x.dtype)
    cs = torch.cos(2.0 * pos).view(1, S, 1)
    sn = torch.sin(2.0 * pos).view(1, S, 1)
    out = x.clone()
    view = out[:, :, : pairs * 2].reshape(H, S, pairs, 2)
    a, b = view[..., 0], view[..., 1]
    view[..., 0] = cs * a - sn * b
    view[..., 1] = sn * a + cs * b
    return out


def consensus_batched(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q,k,v: [H,S,D] → [H,S,D]
    Dense masked S×S but float32 + int8 collapse; when active frac is low
    the weight matmul is still S×S — for true sparse use padded path below.
    """
    H, S, D = q.shape
    q = _phase_batch(q.float())
    k = _phase_batch(k.float())
    v = v.float()

    k_coh = (k.abs() > COLLAPSE_THRESHOLD).float().mean(-1)  # H,S
    tq = torch.ones(q.shape, device=q.device, dtype=torch.int8)
    tq = torch.where(q > COLLAPSE_THRESHOLD, torch.tensor(2, dtype=torch.int8, device=q.device), tq)
    tq = torch.where(q < -COLLAPSE_THRESHOLD, torch.tensor(0, dtype=torch.int8, device=q.device), tq)
    tk = torch.ones(k.shape, device=k.device, dtype=torch.int8)
    tk = torch.where(k > COLLAPSE_THRESHOLD, torch.tensor(2, dtype=torch.int8, device=k.device), tk)
    tk = torch.where(k < -COLLAPSE_THRESHOLD, torch.tensor(0, dtype=torch.int8, device=k.device), tk)

    # trit sim H,S,S
    tq_e = tq.unsqueeze(2)
    tk_e = tk.unsqueeze(1)
    super_m = (tq_e == 1) | (tk_e == 1)
    same = (tq_e == tk_e) & ~super_m
    opp = (tq_e != tk_e) & ~super_m
    sim = (same.float() - opp.float()).mean(-1)

    idx = torch.arange(S, device=q.device)
    causal = idx.view(1, S, 1) >= idx.view(1, 1, S)
    gate = (k_coh > 0.5).unsqueeze(1) & causal
    w = torch.where(gate, sim, torch.zeros_like(sim))
    # Zero superposed-only rows: already in sim
    active = (w != 0).float().sum(-1, keepdim=True).clamp_min(1.0)
    return (torch.matmul(w, v) / active).to(v.dtype)


def consensus_true_sparse_padded(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    True O(S·A) style: pad active keys per head to max_A, then [H,S,A] sim.
    Best when A ≪ S (FSOT collapse regime).
    """
    H, S, D = q.shape
    device = q.device
    q = _phase_batch(q.float())
    k = _phase_batch(k.float())
    v = v.float()

    k_coh = (k.abs() > COLLAPSE_THRESHOLD).float().mean(-1)  # H,S
    active_mask = k_coh > 0.5  # H,S
    # max active count
    counts = active_mask.sum(-1)  # H
    max_a = int(counts.max().item())
    if max_a == 0:
        return torch.zeros_like(v)

    # Build padded indices [H, max_a] with 0 fill; valid mask
    idx_pad = torch.zeros(H, max_a, device=device, dtype=torch.long)
    valid = torch.zeros(H, max_a, device=device, dtype=torch.bool)
    for h in range(H):
        ix = active_mask[h].nonzero(as_tuple=False).squeeze(-1)
        a = ix.numel()
        if a > 0:
            idx_pad[h, :a] = ix
            valid[h, :a] = True

    # Gather k,v active: [H, max_a, D]
    # advanced indexing
    h_ix = torch.arange(H, device=device).unsqueeze(1).expand(H, max_a)
    k_act = k[h_ix, idx_pad]  # H,A,D
    v_act = v[h_ix, idx_pad]
    pos_act = idx_pad  # H,A positions

    tq = torch.ones_like(q, dtype=torch.int8)
    tq = torch.where(q > COLLAPSE_THRESHOLD, torch.tensor(2, dtype=torch.int8, device=device), tq)
    tq = torch.where(q < -COLLAPSE_THRESHOLD, torch.tensor(0, dtype=torch.int8, device=device), tq)
    tk = torch.ones_like(k_act, dtype=torch.int8)
    tk = torch.where(k_act > COLLAPSE_THRESHOLD, torch.tensor(2, dtype=torch.int8, device=device), tk)
    tk = torch.where(k_act < -COLLAPSE_THRESHOLD, torch.tensor(0, dtype=torch.int8, device=device), tk)

    # sim [H,S,A]
    tq_e = tq.unsqueeze(2)  # H,S,1,D
    tk_e = tk.unsqueeze(1)  # H,1,A,D
    super_m = (tq_e == 1) | (tk_e == 1)
    same = (tq_e == tk_e) & ~super_m
    opp = (tq_e != tk_e) & ~super_m
    sim = (same.float() - opp.float()).mean(-1)  # H,S,A

    qpos = torch.arange(S, device=device).view(1, S, 1)
    causal = pos_act.unsqueeze(1) <= qpos  # H,S,A
    gate = valid.unsqueeze(1) & causal
    w = torch.where(gate, sim, torch.zeros_like(sim))
    active = (w != 0).float().sum(-1, keepdim=True).clamp_min(1.0)
    out = torch.matmul(w, v_act) / active  # H,S,D
    return out
