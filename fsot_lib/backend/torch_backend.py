"""Torch/CUDA adapter — buffers and speed only."""

from __future__ import annotations

import math

import torch

from fsot_lib.seeds import SEEDS


def scalar_torch(
    *,
    N: float = 1.0,
    P: float = 1.0,
    D_eff: float = 25.0,
    delta_psi: float = 1.0,
    recent_hits: float = 0.0,
    rho: float = 1.0,
    observed: bool = False,
    delta_theta: float = 1.0,
    scale: float = 1.0,
    amplitude: float = 1.0,
    trend_bias: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    s = SEEDS
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    dt = torch.float64
    N_t = torch.tensor(N, dtype=dt, device=d)
    P_t = torch.tensor(P, dtype=dt, device=d)
    D = torch.tensor(D_eff, dtype=dt, device=d)
    dp = torch.tensor(delta_psi, dtype=dt, device=d)
    hits = torch.tensor(recent_hits, dtype=dt, device=d)
    dth = torch.tensor(delta_theta, dtype=dt, device=d)
    rho_t = torch.tensor(rho, dtype=dt, device=d)

    def t(v: float) -> torch.Tensor:
        return torch.tensor(v, dtype=dt, device=d)

    growth = torch.exp(t(s.alpha) * (1 - hits / N_t) * t(s.gamma) / t(s.phi))
    base = (
        (N_t * P_t / torch.sqrt(D))
        * torch.cos((t(s.psi_con) + dp) / t(s.eta_eff))
        * torch.exp(-t(s.alpha) * hits / N_t + rho_t + t(s.b_in) * dp)
        * (1 + growth * t(s.c_eff))
    )
    t1 = base * (1 + t(s.p_new) * torch.log(D / 25.0))
    if observed:
        t1 = t1 * torch.exp(t(s.c_factor) * t(s.p_var)) * torch.cos(dp + t(s.p_var))
    t2 = t(scale * amplitude + trend_bias)
    valve = (
        t(s.beta)
        * torch.cos(dp)
        * (N_t * P_t / torch.sqrt(D))
        * (1 + t(s.chaos) * (D - 25.0) / 25.0)
        * (1 + t(s.poof) * torch.cos(t(s.theta_s) + t(s.pi)) + t(s.suction) * torch.sin(t(s.theta_s)))
    )
    acoustic = (
        1.0
        + (t(s.a_bleed) * torch.sin(dth) ** 2) / t(s.phi)
        + (t(s.a_in) * torch.cos(dth) ** 2) / t(s.phi)
    )
    phase = 1.0 + t(s.b_in) * t(s.p_var)
    return t(s.k) * (t1 + t2 + valve * acoustic * phase)


def prefer_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
