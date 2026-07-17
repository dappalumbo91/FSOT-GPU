"""
FSOT sparse consensus — wall-clock path from archive math.

Uses:
  collapse threshold θ = C_eff · P_var  (Scalar.lean / fsot_compute)
  coherence gate 0.5                      (trinary kernel lattice.rs)
  trit similarity with superposed = 0     (same)

Complexity: O(H·S·A·D) with A = #{keys: coh>0.5} ≪ S under FSOT collapse,
vs softmax O(H·S²·D) always.

Contract-identical to consensus_aggregate when the same Q,K,V are used.
"""
from __future__ import annotations

import torch

from fsot_lib.seeds import COLLAPSE_THRESHOLD, SEEDS

# Archive: Metatron pathways 27 — preferred tile multiple when packing
METATRON = 27
# Archive: ignition coherence Gate/Eq ≈ 0.3922 (optional soft stats only)
IGNITION = float(SEEDS.phi / (1.0 + SEEDS.phi) / 1.5759)  # ~0.392 matches gate/eq


def _collapse_codes(x: torch.Tensor) -> torch.Tensor:
    """float → {0,1,2} int16 for fast compare."""
    up = x > COLLAPSE_THRESHOLD
    down = x < -COLLAPSE_THRESHOLD
    codes = torch.ones(x.shape, device=x.device, dtype=torch.int16)
    codes = torch.where(up, torch.tensor(2, device=x.device, dtype=torch.int16), codes)
    codes = torch.where(down, torch.tensor(0, device=x.device, dtype=torch.int16), codes)
    return codes


def _phase_rotate_inplace(h: torch.Tensor) -> torch.Tensor:
    """h [S,D] float32/64 — pi-periodic phase (kernel)."""
    s, d = h.shape
    out = h
    pos = torch.arange(s, device=h.device, dtype=h.dtype)
    theta = 2.0 * pos
    cs = torch.cos(theta)
    sn = torch.sin(theta)
    pairs = d // 2
    # vectorized pairs via reshape
    if pairs == 0:
        return out
    view = out[:, : pairs * 2].reshape(s, pairs, 2)
    a = view[:, :, 0]
    b = view[:, :, 1]
    # broadcast cs,sn over pairs
    cs2 = cs.unsqueeze(1)
    sn2 = sn.unsqueeze(1)
    na = cs2 * a - sn2 * b
    nb = sn2 * a + cs2 * b
    view[:, :, 0] = na
    view[:, :, 1] = nb
    return out


def consensus_head_sparse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    One head. q,k,v: [S, D]  (prefer float32 on GPU).
    """
    s, d = q.shape
    device = q.device
    dtype = v.dtype

    q = _phase_rotate_inplace(q.contiguous())
    k = _phase_rotate_inplace(k.contiguous())

    # Coherence of keys — archive θ_coll
    k_coh = (k.abs() > COLLAPSE_THRESHOLD).to(torch.float32).mean(dim=-1)  # [S]
    active_mask = k_coh > 0.5  # kernel gate
    active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)  # [A]

    out = torch.zeros(s, d, device=device, dtype=torch.float32)
    if active_idx.numel() == 0:
        return out.to(dtype)

    tq = _collapse_codes(q)  # [S,D]
    tk_all = _collapse_codes(k)
    tk = tk_all.index_select(0, active_idx)  # [A,D]
    vk = v.index_select(0, active_idx).to(torch.float32)  # [A,D]
    a_pos = active_idx.to(torch.int64)  # key positions

    # For each query position, only keys with a_pos <= q (causal)
    # Batched: sim [S, A]
    tq_e = tq.unsqueeze(1)  # S,1,D
    tk_e = tk.unsqueeze(0)  # 1,A,D
    super_m = (tq_e == 1) | (tk_e == 1)
    same = (tq_e == tk_e) & ~super_m
    opp = (tq_e != tk_e) & ~super_m
    sim = (same.to(torch.float32) - opp.to(torch.float32)).mean(dim=-1)  # S,A

    q_idx = torch.arange(s, device=device).unsqueeze(1)  # S,1
    causal = a_pos.unsqueeze(0) <= q_idx  # S,A
    w = torch.where(causal, sim, torch.zeros_like(sim))
    # drop exact zeros from active count
    nz = w != 0
    active_count = nz.to(torch.float32).sum(dim=-1, keepdim=True).clamp_min(1.0)
    # zero weights already 0
    out = (w @ vk) / active_count
    return out.to(dtype)


def consensus_multihead_sparse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q,k,v: [H,S,D] → [H,S,D] float32 preferred."""
    h = q.shape[0]
    # float32 hot path (threshold still exact f64 constant compared in float32 — OK within 1e-6)
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
    outs = []
    for i in range(h):
        outs.append(consensus_head_sparse(q32[i], k32[i], v32[i]))
    return torch.stack(outs, dim=0).to(v.dtype)


def consensus_multihead_sparse_fast(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Fully batched sparse-ish path: still computes S×S style only on a
    *masked* weight matrix but skips phase on inactive via early coh.
    Better GPU occupancy when A is moderate.

    Actually uses dense sim only among all keys then multiplies by
    (coh>0.5) mask — cheaper than f64 vectorized path, float32, fused.
    When A is tiny, per-head sparse above is better; we pick at runtime.
    """
    H, S, D = q.shape
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # Phase rotate batch
    pos = torch.arange(S, device=q.device, dtype=q.dtype)
    theta = 2.0 * pos
    cs = torch.cos(theta).view(1, S, 1)
    sn = torch.sin(theta).view(1, S, 1)
    pairs = D // 2
    if pairs > 0:
        qv = q[:, :, : pairs * 2].reshape(H, S, pairs, 2).clone()
        kv = k[:, :, : pairs * 2].reshape(H, S, pairs, 2).clone()
        qa, qb = qv[..., 0], qv[..., 1]
        ka, kb = kv[..., 0], kv[..., 1]
        qv[..., 0] = cs * qa - sn * qb
        qv[..., 1] = sn * qa + cs * qb
        kv[..., 0] = cs * ka - sn * kb
        kv[..., 1] = sn * ka + cs * kb
        q = q.clone()
        k = k.clone()
        q[:, :, : pairs * 2] = qv.reshape(H, S, pairs * 2)
        k[:, :, : pairs * 2] = kv.reshape(H, S, pairs * 2)

    k_coh = (k.abs() > COLLAPSE_THRESHOLD).to(torch.float32).mean(dim=-1)  # H,S
    # If extremely sparse, use per-head gather path
    frac = (k_coh > 0.5).to(torch.float32).mean().item()
    if frac < 0.15:
        return consensus_multihead_sparse(q, k, v)

    tq = _collapse_codes(q)
    tk = _collapse_codes(k)
    tq_e = tq.unsqueeze(2)
    tk_e = tk.unsqueeze(1)
    super_m = (tq_e == 1) | (tk_e == 1)
    same = (tq_e == tk_e) & ~super_m
    opp = (tq_e != tk_e) & ~super_m
    sim = (same.to(torch.float32) - opp.to(torch.float32)).mean(dim=-1)  # H,S,S

    idx = torch.arange(S, device=q.device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
    gate = (k_coh > 0.5).unsqueeze(1) & causal.unsqueeze(0)
    w = torch.where(gate, sim, torch.zeros_like(sim))
    active = (w != 0).to(torch.float32).sum(dim=-1, keepdim=True).clamp_min(1.0)
    out = torch.matmul(w, v) / active
    return out


def auto_consensus(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Runtime pick sparse gather vs masked dense based on measured A/S."""
    with torch.no_grad():
        k32 = k.to(torch.float32)
        coh = (k32.abs() > COLLAPSE_THRESHOLD).to(torch.float32).mean(dim=-1)
        frac = float((coh > 0.5).to(torch.float32).mean().item())
    if frac < 0.20:
        return consensus_multihead_sparse(q, k, v)
    return consensus_multihead_sparse_fast(q, k, v)
