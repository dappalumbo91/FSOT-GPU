"""FSOT scalar S = K·(T1+T2+T3). Pure Python + optional torch device."""

from __future__ import annotations

import math
from typing import Union

from fsot_lib.seeds import SEEDS

Number = Union[float, "torch.Tensor"]


def compute_scalar(
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
    device: str | None = None,
) -> float:
    """
    Pure-float path (always available — no CUDA required).
    Matches archive compute_scalar structure.
    """
    s = SEEDS
    growth = math.exp(s.alpha * (1.0 - recent_hits / N) * s.gamma / s.phi)
    base = (
        (N * P / math.sqrt(D_eff))
        * math.cos((s.psi_con + delta_psi) / s.eta_eff)
        * math.exp(-s.alpha * recent_hits / N + rho + s.b_in * delta_psi)
        * (1.0 + growth * s.c_eff)
    )
    t1 = base * (1.0 + s.p_new * math.log(D_eff / 25.0))
    if observed:
        t1 = t1 * math.exp(s.c_factor * s.p_var) * math.cos(delta_psi + s.p_var)
    t2 = scale * amplitude + trend_bias
    valve = (
        s.beta
        * math.cos(delta_psi)
        * (N * P / math.sqrt(D_eff))
        * (1.0 + s.chaos * (D_eff - 25.0) / 25.0)
        * (1.0 + s.poof * math.cos(s.theta_s + s.pi) + s.suction * math.sin(s.theta_s))
    )
    acoustic = (
        1.0
        + (s.a_bleed * math.sin(delta_theta) ** 2) / s.phi
        + (s.a_in * math.cos(delta_theta) ** 2) / s.phi
    )
    phase = 1.0 + s.b_in * s.p_var
    t3 = valve * acoustic * phase
    return s.k * (t1 + t2 + t3)


def compute_scalar_torch(**kwargs):
    """Optional torch path — backend only. Import torch lazily."""
    import torch
    from fsot_lib.backend.torch_backend import scalar_torch

    return scalar_torch(**kwargs)
